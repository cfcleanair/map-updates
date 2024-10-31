from flask import Flask, jsonify, Response
from flask_cors import CORS
import schedule
import threading
import time
from google.oauth2 import service_account
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping
import json
from datetime import datetime, time, timedelta
import numpy as np
import gspread
from gspread_dataframe import get_as_dataframe
from typing import Dict, Any, List
import os

app = Flask(__name__)
CORS(app)

# Global variables to store the latest data
latest_odor_geojson = None
latest_waste_geojson = None
last_update_time = None

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime, time, and timedelta objects."""
    def default(self, obj: Any) -> str:
        if isinstance(obj, (datetime, time)):
            return obj.isoformat()
        elif isinstance(obj, timedelta):
            return str(obj)
        return super().default(obj)

def get_google_credentials():
    """Get Google credentials from environment variable"""
    credentials_json = os.getenv('GOOGLE_CREDENTIALS')
    if not credentials_json:
        raise ValueError("GOOGLE_CREDENTIALS environment variable not set")
    
    credentials_info = json.loads(credentials_json)
    credentials = service_account.Credentials.from_service_account_info(
        credentials_info,
        scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    )
    return credentials

def calculate_decay_rate(initial_intensity: float) -> float:
    """Calculate decay rate based on initial intensity."""
    if initial_intensity == 0:
        return 0
    return initial_intensity / 100

def calculate_intensity(row: pd.Series) -> float:
    """Calculate current intensity based on initial intensity and time elapsed."""
    initial_intensity = row['עוצמת הריח']
    time_elapsed = row['time_elapsed_minutes']
    decay_rate = calculate_decay_rate(initial_intensity)
    current_intensity = max(0, initial_intensity - (decay_rate * time_elapsed))
    return round(current_intensity, 2)

def split_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Split coordinates string into latitude and longitude."""
    coords_split = df['קואורדינטות'].str.split(',', expand=True)
    if coords_split.shape[1] >= 2:
        df['lat'] = coords_split[0].astype(float)
        df['lon'] = coords_split[1].astype(float)
    else:
        df['lat'] = np.nan
        df['lon'] = np.nan
    return df

def randomize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Randomize locations within 0 to 300 meters."""
    earth_radius = 6378137
    max_distance = 300

    random_distances = np.random.uniform(0, max_distance, size=len(df))
    random_bearings = np.random.uniform(0, 360, size=len(df))
    random_bearings_rad = np.deg2rad(random_bearings)

    lat_rad = np.deg2rad(df['lat'].values)
    lon_rad = np.deg2rad(df['lon'].values)

    new_lat_rad = np.arcsin(
        np.sin(lat_rad) * np.cos(random_distances / earth_radius) +
        np.cos(lat_rad) * np.sin(random_distances / earth_radius) * np.cos(random_bearings_rad)
    )

    new_lon_rad = lon_rad + np.arctan2(
        np.sin(random_bearings_rad) * np.sin(random_distances / earth_radius) * np.cos(lat_rad),
        np.cos(random_distances / earth_radius) - np.sin(lat_rad) * np.sin(new_lat_rad)
    )

    df['lat'] = np.rad2deg(new_lat_rad)
    df['lon'] = np.rad2deg(new_lon_rad)

    return df

def create_geojson_no_buffer(gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    """Create GeoJSON from GeoDataFrame without buffer."""
    gdf = gdf.replace({np.nan: None})
    features: List[Dict[str, Any]] = []
    for _, row in gdf.iterrows():
        geom = row['geometry']
        props = row.drop('geometry').to_dict()
        geom_json = mapping(geom)
        feature = {
            "type": "Feature",
            "geometry": geom_json,
            "properties": props
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features
    }

def update_data():
    """Update the GeoJSON data"""
    global latest_odor_geojson, latest_waste_geojson, last_update_time
    
    try:
        print(f"\nUpdating data at {datetime.now()}")
        
        # Get credentials and connect to Google Sheets
        creds = get_google_credentials()
        gc = gspread.authorize(creds)
        
        # Open the Google Sheet
        sheet_url = "https://docs.google.com/spreadsheets/d/1i2Z4V07JzhkZSGub1acEJy6IZueKO8Imviih9_MO3tc/edit?gid=0#gid=0"
        spreadsheet = gc.open_by_url(sheet_url)
        worksheet = spreadsheet.worksheet('Sheet1')
        
        # Load data into DataFrame
        df_original = get_as_dataframe(worksheet)
        
        # Verify required columns
        required_columns = [
            'תאריך שליחת הדיווח',
            'שעת שליחת הדיווח',
            'סוג דיווח',
            'קואורדינטות',
            'עוצמת הריח',
            'צבע העשן'
        ]
        
        missing_columns = [col for col in required_columns if col not in df_original.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")
        
        # Process datetime
        df_original['datetime'] = pd.to_datetime(
            df_original['תאריך שליחת הדיווח'].astype(str) + ' ' + df_original['שעת שליחת הדיווח'].astype(str),
            format='%d/%m/%Y %H:%M:%S',
            errors='coerce'
        )
        
        # Calculate time elapsed
        current_time = pd.Timestamp.now()
        df_original['time_elapsed_minutes'] = (current_time - df_original['datetime']).dt.total_seconds() / 60
        df_original['time_elapsed_minutes'] = df_original['time_elapsed_minutes'].clip(lower=0, upper=10000)
        
        # Split data by report type
        odor_nuisance_df = df_original[df_original['סוג דיווח'] == 'מפגע ריח']
        waste_nuisance_df = df_original[df_original['סוג דיווח'] == 'מפגע פסולת']
        
        # Filter valid coordinates
        odor_nuisance_df = odor_nuisance_df[odor_nuisance_df['קואורדינטות'].notna()]
        waste_nuisance_df = waste_nuisance_df[waste_nuisance_df['קואורדינטות'].notna()]
        
        odor_nuisance_df = odor_nuisance_df[~odor_nuisance_df['קואורדינטות'].astype(str).str.contains('המיקום לא נמצא', na=False)]
        waste_nuisance_df = waste_nuisance_df[~waste_nuisance_df['קואורדינטות'].astype(str).str.contains('המיקום לא נמצא', na=False)]
        
        # Process waste nuisance data
        valid_smoke_colors = ['לבן', 'אפור', 'שחור']
        waste_nuisance_df = waste_nuisance_df[
            waste_nuisance_df['צבע העשן'].isin(valid_smoke_colors) &
            waste_nuisance_df['צבע העשן'].notna() &
            (waste_nuisance_df['צבע העשן'] != 'אין עשן')
        ]
        
        # Process coordinates
        odor_nuisance_df = split_coordinates(odor_nuisance_df)
        waste_nuisance_df = split_coordinates(waste_nuisance_df)
        
        # Process intensity
        odor_nuisance_df['עוצמת הריח'] = pd.to_numeric(odor_nuisance_df['עוצמת הריח'], errors='coerce').fillna(0)
        odor_nuisance_df['intensity'] = odor_nuisance_df.apply(calculate_intensity, axis=1)
        
        # Randomize coordinates
        odor_nuisance_df = randomize_coordinates(odor_nuisance_df)
        
        # Create GeoDataFrames
        odor_gdf = gpd.GeoDataFrame(
            odor_nuisance_df,
            geometry=gpd.points_from_xy(odor_nuisance_df['lon'], odor_nuisance_df['lat']),
            crs='EPSG:4326'
        )
        waste_gdf = gpd.GeoDataFrame(
            waste_nuisance_df,
            geometry=gpd.points_from_xy(waste_nuisance_df['lon'], waste_nuisance_df['lat']),
            crs='EPSG:4326'
        )
        
        # Update global variables
        latest_odor_geojson = create_geojson_no_buffer(odor_gdf)
        latest_waste_geojson = create_geojson_no_buffer(waste_gdf)
        last_update_time = datetime.now()
        
        print("Data update completed successfully")
        
    except Exception as e:
        print(f"Error updating data: {str(e)}")

@app.route('/')
def home():
    """Home page with basic status"""
    return jsonify({
        "status": "running",
        "last_update": last_update_time.isoformat() if last_update_time else None,
        "endpoints": {
            "odor": "/odor",
            "waste": "/waste",
            "status": "/status"
        }
    })

@app.route('/odor')
def serve_odor_geojson():
    """Serve the latest odor nuisance GeoJSON data"""
    if latest_odor_geojson is None:
        return jsonify({"error": "No data available"}), 503
    
    return Response(
        json.dumps(latest_odor_geojson, cls=CustomJSONEncoder),
        mimetype='application/json',
        headers={
            'Access-Control-Allow-Origin': '*',
            'Last-Modified': last_update_time.strftime("%a, %d %b %Y %H:%M:%S GMT")
        }
    )

@app.route('/waste')
def serve_waste_geojson():
    """Serve the latest waste nuisance GeoJSON data"""
    if latest_waste_geojson is None:
        return jsonify({"error": "No data available"}), 503
    
    return Response(
        json.dumps(latest_waste_geojson, cls=CustomJSONEncoder),
        mimetype='application/json',
        headers={
            'Access-Control-Allow-Origin': '*',
            'Last-Modified': last_update_time.strftime("%a, %d %b %Y %H:%M:%S GMT")
        }
    )

@app.route('/status')
def server_status():
    """Serve server status information"""
    return jsonify({
        "status": "running",
        "last_update": last_update_time.isoformat() if last_update_time else None,
        "has_odor_data": latest_odor_geojson is not None,
        "has_waste_data": latest_waste_geojson is not None
    })

def run_schedule():
    """Run the scheduler in a separate thread"""
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    # Initial data update
    update_data()
    
    # Schedule updates every 2 minutes
    schedule.every(2).minutes.do(update_data)
    
    # Start the scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_schedule)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))