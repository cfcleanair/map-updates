from flask import Flask, jsonify, Response
from flask_cors import CORS
from flask_caching import Cache
import gzip
import schedule
import threading
import time as time_module
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

# Initialize Flask app with caching
app = Flask(__name__)
CORS(app)
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 60  # Cache for 1 minute
})

# Global variables
latest_odor_geojson = None
last_update_time = None

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime, time, and timedelta objects."""
    def default(self, obj: Any) -> str:
        if isinstance(obj, (datetime, time)):
            return obj.isoformat()
        elif isinstance(obj, timedelta):
            return str(obj)
        return super().default(obj)

def gzip_response(data):
    """Compress response data using gzip"""
    gzip_buffer = gzip.compress(json.dumps(data, cls=CustomJSONEncoder).encode('utf-8'))
    return Response(
        gzip_buffer,
        mimetype='application/json',
        headers={
            'Access-Control-Allow-Origin': '*',
            'Content-Encoding': 'gzip',
            'Last-Modified': last_update_time.strftime("%a, %d %b %Y %H:%M:%S GMT"),
            'Cache-Control': 'public, max-age=60'  # Allow client caching for 1 minute
        }
    )

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
    global latest_odor_geojson, last_update_time
    
    try:
        print(f"\n[{datetime.now()}] Starting data update...")
        
        # Get credentials and connect to Google Sheets
        print("Getting Google credentials...")
        creds = get_google_credentials()
        print("Authorizing with Google Sheets...")
        gc = gspread.authorize(creds)
        
        # Open the Google Sheet
        print("Opening Google Sheet...")
        sheet_url = "https://docs.google.com/spreadsheets/d/1i2Z4V07JzhkZSGub1acEJy6IZueKO8Imviih9_MO3tc/edit?gid=0#gid=0"
        spreadsheet = gc.open_by_url(sheet_url)
        worksheet = spreadsheet.worksheet('Sheet1')
        
        # Load data into DataFrame
        print("Loading data into DataFrame...")
        df_original = get_as_dataframe(worksheet)
        print(f"Loaded {len(df_original)} rows from Google Sheet")
        
        # Verify required columns
        required_columns = [
            'תאריך שליחת הדיווח',
            'שעת שליחת הדיווח',
            'סוג דיווח',
            'קואורדינטות',
            'עוצמת הריח',
            'צבע העשן',
            'בדיקה',
            'ספאם'
        ]
        
        print("Checking required columns...")
        missing_columns = [col for col in required_columns if col not in df_original.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")
        print("All required columns present")
        
        # Filter out test and spam entries
        print("Filtering test and spam entries...")
        df_original = df_original[
            (df_original['בדיקה'].fillna(0) != 1) & 
            (df_original['ספאם'].fillna(0) != 1)
        ]
        
        # Filter out invalid coordinates
        print("Filtering invalid coordinates...")
        df_original = df_original[
            df_original['קואורדינטות'].notna() & 
            ~df_original['קואורדינטות'].astype(str).str.contains('המיקום לא נמצא', na=False)
        ]
        
        # Process datetime
        print("Processing datetime...")
        df_original['datetime'] = pd.to_datetime(
            df_original['תאריך שליחת הדיווח'].astype(str) + ' ' + df_original['שעת שליחת הדיווח'].astype(str),
            format='%d/%m/%Y %H:%M:%S',
            errors='coerce'
        )

        # Get current time
        current_time = datetime.now()

        # Randomly select 50 indices
        random_indices = np.random.choice(df_original.index, size=50, replace=False)

        # Update only those 50 rows with current time
        df_original.loc[random_indices, 'datetime'] = current_time
        
        # Calculate time elapsed
        df_original['time_elapsed_minutes'] = (current_time - df_original['datetime']).dt.total_seconds() / 60
        df_original['time_elapsed_minutes'] = df_original['time_elapsed_minutes'].clip(lower=0, upper=10000)

        # Split into odor and waste reports
        print("Processing reports...")
        odor_df = df_original[df_original['סוג דיווח'] == 'מפגע ריח'].copy()
        waste_df = df_original[df_original['סוג דיווח'] == 'מפגע פסולת'].copy()
        
        # Process waste reports
        valid_smoke_colors = ['לבן', 'אפור', 'שחור']
        valid_waste_df = waste_df[
            waste_df['צבע העשן'].isin(valid_smoke_colors) &
            waste_df['צבע העשן'].notna() &
            (waste_df['צבע העשן'] != 'אין עשן')
        ].copy()
        
        # Set intensity to 6 for all valid waste reports
        valid_waste_df['עוצמת הריח'] = 6
        
        # Process coordinates for both datasets
        print("Processing coordinates...")
        odor_df = split_coordinates(odor_df)
        valid_waste_df = split_coordinates(valid_waste_df)
        
        # Process intensity for odor reports
        print("Processing intensity values...")
        odor_df['עוצמת הריח'] = pd.to_numeric(odor_df['עוצמת הריח'], errors='coerce').fillna(0)
        odor_df['intensity'] = odor_df.apply(calculate_intensity, axis=1)
        odor_df = odor_df[odor_df['intensity'] > 0]
        
        # Calculate intensity for waste reports (using intensity of 6)
        valid_waste_df['intensity'] = valid_waste_df.apply(calculate_intensity, axis=1)
        
        # Randomize coordinates only for odor reports
        print("Randomizing coordinates for odor reports...")
        odor_df = randomize_coordinates(odor_df)
        
        # Combine odor and waste reports
        combined_df = pd.concat([odor_df, valid_waste_df])
        
        # Create GeoDataFrame
        print("Creating GeoDataFrame...")
        if len(combined_df) > 0:
            combined_gdf = gpd.GeoDataFrame(
                combined_df,
                geometry=gpd.points_from_xy(combined_df['lon'], combined_df['lat']),
                crs='EPSG:4326'
            )
        else:
            combined_gdf = gpd.GeoDataFrame(
                columns=combined_df.columns.tolist() + ['geometry'],
                crs='EPSG:4326'
            )
        
        # Update global variables
        print("Converting to GeoJSON...")
        latest_odor_geojson = create_geojson_no_buffer(combined_gdf)
        last_update_time = datetime.now()
        
        # Clear the cache when new data arrives
        cache.delete('odor_data')
        
        print(f"[{datetime.now()}] Data update completed successfully")
        print(f"Final counts: {len(odor_df)} odor points, {len(valid_waste_df)} waste points, {len(combined_gdf)} total points")
        
    except Exception as e:
        print(f"[{datetime.now()}] ERROR updating data: {str(e)}")
        print("Error details:", e.__class__.__name__)
        import traceback
        print(traceback.format_exc())

@app.route('/')
def home():
    """Home page with basic status"""
    return jsonify({
        "status": "running",
        "last_update": last_update_time.isoformat() if last_update_time else None,
        "endpoints": {
            "odor": "/odor",
            "status": "/status"
        }
    })

@app.route('/odor')
@cache.cached(timeout=60, key_prefix='odor_data')
def serve_odor_geojson():
    """Serve the latest odor nuisance GeoJSON data"""
    if latest_odor_geojson is None:
        return jsonify({"error": "No data available"}), 503
    
    return gzip_response(latest_odor_geojson)

@app.route('/status')
def server_status():
    """Serve server status information"""
    return jsonify({
        "status": "running",
        "last_update": last_update_time.isoformat() if last_update_time else None,
        "has_odor_data": latest_odor_geojson is not None
    })

def run_schedule():
    """Run the scheduler in a separate thread"""
    while True:
        schedule.run_pending()
        time_module.sleep(1)

def initialize_app():
    """Initialize the application with data and start scheduler"""
    print("Initializing application...")
    try:
        # Initial data update
        print("Performing initial data fetch...")
        update_data()
        print(f"Initial data fetch completed at {datetime.now()}")
        
        # Schedule updates every 1 minute
        schedule.every(1).minutes.do(update_data)
        print("Scheduled updates every 1 minute")
        
        # Start scheduler in background
        scheduler_thread = threading.Thread(target=run_schedule)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        print("Scheduler started successfully")
        
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        import traceback
        print(traceback.format_exc())

# Initialize the app when module loads
initialize_app()

if __name__ == "__main__":
    # Run Flask app
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)