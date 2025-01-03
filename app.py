from flask import Flask, jsonify, Response, request
from flask_cors import CORS
from flask_caching import Cache
import gzip
import schedule
import threading
import time as time_module
from google.oauth2 import service_account
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, Point
from typing import Dict, Any, List, Union, Optional, Tuple
import json
from datetime import datetime, time, timedelta, timezone
from dateutil import tz
import numpy as np
import gspread
from gspread_dataframe import get_as_dataframe
import os

app = Flask(__name__)
CORS(app)
cache = Cache(app, config={'CACHE_TYPE': 'simple', 'CACHE_DEFAULT_TIMEOUT': 60})

latest_odor_geojson: Optional[Dict[str, Any]] = None
last_update_time: Optional[datetime] = None

REQUIRED_COLUMNS = [
    'תאריך שליחת הדיווח',
    'שעת שליחת הדיווח',
    'סוג דיווח',
    'קואורדינטות',
    'עוצמת הריח',
    'צבע העשן',
    'בדיקה',
    'ספאם'
]
FIRST_REPORT_DATE = datetime(2024, 4, 4, tzinfo=tz.gettz('Asia/Jerusalem'))

class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for datetime and timedelta objects.
    
    Parameters:
        obj (Any): Object to encode
        
    Returns:
        str: ISO format string representation of the object
    """
    def default(self, obj: Any) -> str:
        if isinstance(obj, (datetime, time)):
            return obj.isoformat()
        elif isinstance(obj, timedelta):
            return str(obj)
        return super().default(obj)

def gzip_response(data: Dict[str, Any]) -> Response:
    """
    Create a gzipped HTTP response with proper headers.
    
    Parameters:
        data (Dict[str, Any]): Dictionary to be JSON-encoded and compressed
        
    Returns:
        Response: Flask Response object with gzipped content and appropriate headers
    """
    gzip_buffer = gzip.compress(json.dumps(data, cls=CustomJSONEncoder).encode('utf-8'))
    return Response(
        gzip_buffer,
        mimetype='application/json',
        headers={
            'Access-Control-Allow-Origin': '*',
            'Content-Encoding': 'gzip',
            'Last-Modified': last_update_time.strftime("%a, %d %b %Y %H:%M:%S GMT"),
            'Cache-Control': 'public, max-age=60'
        }
    )

def get_google_credentials() -> service_account.Credentials:
    """
    Retrieve Google service account credentials from environment variables.
    
    Returns:
        service_account.Credentials: Google service account credentials object
        
    Raises:
        ValueError: If GOOGLE_CREDENTIALS environment variable is not set
    """
    credentials_json = os.getenv('GOOGLE_CREDENTIALS')
    if not credentials_json:
        raise ValueError("GOOGLE_CREDENTIALS environment variable not set")
    
    credentials_info = json.loads(credentials_json)
    return service_account.Credentials.from_service_account_info(
        credentials_info,
        scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    )

def process_raw_dataframe(df: pd.DataFrame, reference_time: Optional[datetime] = None) -> pd.DataFrame:
    """
    Process and clean the raw dataframe from Google Sheets.
    
    Parameters:
        df (pd.DataFrame): Raw DataFrame containing odor reports
        reference_time (Optional[datetime]): Timestamp to use for calculations, defaults to current time
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with processed datetime and elapsed time calculations
        
    Raises:
        ValueError: If required columns are missing from the DataFrame
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    
    israel_tz = tz.gettz('Asia/Jerusalem')
    df = df[REQUIRED_COLUMNS].copy()
    df = df[(df['בדיקה'].fillna(0) != 1) & (df['ספאם'].fillna(0) != 1)]
    df = df[df['קואורדינטות'].notna() & ~df['קואורדינטות'].astype(str).str.contains('המיקום לא נמצא', na=False)]
    
    df['datetime'] = pd.to_datetime(
        df['תאריך שליחת הדיווח'].astype(str) + ' ' + df['שעת שליחת הדיווח'].astype(str),
        format='%d/%m/%Y %H:%M:%S',
        errors='coerce'
    ).dt.tz_localize(None)
    
    try:
        df['datetime'] = df['datetime'].dt.tz_localize('Asia/Jerusalem', ambiguous=True, nonexistent='shift_forward')
    except Exception as e:
        print(f"Warning: Time localization error: {str(e)}")
    
    df = df[df['datetime'].notna()]
    
    if reference_time is None:
        reference_time = datetime.now(timezone.utc).astimezone(israel_tz)
    elif reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=israel_tz)
    
    df['time_elapsed_minutes'] = (reference_time - df['datetime']).dt.total_seconds() / 60
    df['time_elapsed_minutes'] = df['time_elapsed_minutes'].clip(lower=0)
    
    return df

def calculate_decay_rate(initial_intensity: float) -> float:
    """
    Calculate decay rate based on initial intensity.
    
    Parameters:
        initial_intensity (float): Initial odor intensity value
        
    Returns:
        float: Calculated decay rate per minute
    """
    return initial_intensity / 100 if initial_intensity != 0 else 0

def calculate_intensity(row: pd.Series) -> float:
    """
    Calculate current intensity based on elapsed time and initial intensity.
    
    Parameters:
        row (pd.Series): DataFrame row containing time_elapsed_minutes and initial intensity
        
    Returns:
        float: Current intensity value after decay calculation
    """
    if row['time_elapsed_minutes'] > 100:
        return 0.0
    
    initial_intensity = float(row['עוצמת הריח'])
    time_elapsed = float(row['time_elapsed_minutes'])
    decay_rate = calculate_decay_rate(initial_intensity)
    current_intensity = max(0, initial_intensity - (decay_rate * time_elapsed))
    return round(current_intensity, 2)

def split_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split coordinate string into separate latitude and longitude columns.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing coordinate strings
        
    Returns:
        pd.DataFrame: DataFrame with separated lat/lon columns
    """
    df = df[df['קואורדינטות'].str.match(r'^-?\d+\.?\d*,-?\d+\.?\d*$', na=False)]
    
    coords_split = df['קואורדינטות'].str.split(',', expand=True)
    if coords_split.shape[1] >= 2:
        df['lat'] = pd.to_numeric(coords_split[0], errors='coerce')
        df['lon'] = pd.to_numeric(coords_split[1], errors='coerce')
    else:
        df['lat'] = np.nan
        df['lon'] = np.nan
    
    return df[df['lat'].notna() & df['lon'].notna()]

def randomize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add random offset to coordinates within a 300m radius.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing lat/lon coordinates
        
    Returns:
        pd.DataFrame: DataFrame with randomized coordinates
    """
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
    """
    Convert GeoDataFrame to GeoJSON format.
    
    Parameters:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing point geometries and properties
        
    Returns:
        Dict[str, Any]: GeoJSON compatible dictionary
    """
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

def generate_heatmap_for_timestamp(timestamp: datetime) -> Dict[str, Any]:
    """
    Generate heatmap data for a specific timestamp.
    
    Parameters:
        timestamp (datetime): Reference time for calculating odor intensities
        
    Returns:
        Dict[str, Any]: GeoJSON compatible dictionary containing heatmap data
        
    Raises:
        Exception: If there's an error accessing or processing the data
    """
    creds = get_google_credentials()
    gc = gspread.authorize(creds)
    
    sheet_url = "https://docs.google.com/spreadsheets/d/1PMm_4Xkrv4Bmy7p9pI8Smnqzl12xgBVotYBEb2O45cg"
    spreadsheet = gc.open_by_url(sheet_url)
    worksheet = spreadsheet.worksheet('Sheet1')
    
    df_original = process_raw_dataframe(get_as_dataframe(worksheet), timestamp)
    
    odor_df = df_original[df_original['סוג דיווח'] == 'מפגע ריח'].copy()
    waste_df = df_original[df_original['סוג דיווח'] == 'מפגע פסולת'].copy()
    
    valid_smoke_colors = ['לבן', 'אפור', 'שחור']
    valid_waste_df = waste_df[
        waste_df['צבע העשן'].isin(valid_smoke_colors) &
        waste_df['צבע העשן'].notna() &
        (waste_df['צבע העשן'] != 'אין עשן')
    ].copy()
    
    valid_waste_df['עוצמת הריח'] = 6
    
    odor_df = split_coordinates(odor_df)
    valid_waste_df = split_coordinates(valid_waste_df)
    
    odor_df['עוצמת הריח'] = pd.to_numeric(odor_df['עוצמת הריח'], errors='coerce').fillna(0)
    odor_df['intensity'] = odor_df.apply(calculate_intensity, axis=1)
    valid_waste_df['intensity'] = valid_waste_df.apply(calculate_intensity, axis=1)
    
    odor_df = randomize_coordinates(odor_df)
    
    combined_df = pd.concat([odor_df, valid_waste_df])
    combined_df = combined_df[combined_df['intensity'] > 0]
    
    if len(combined_df) > 0:
        combined_df = combined_df.sort_values(by='datetime', ascending=True)
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
    
    return create_geojson_no_buffer(combined_gdf)

def update_data() -> None:
    """
    Fetch and process latest odor reports, updating global state.
    
    Updates:
        latest_odor_geojson (Dict[str, Any]): Latest processed GeoJSON data
        last_update_time (datetime): Timestamp of last successful update
    """
    global latest_odor_geojson, last_update_time
    try:
        latest_odor_geojson = generate_heatmap_for_timestamp(datetime.now(tz.gettz('Asia/Jerusalem')))
        last_update_time = datetime.now()
        cache.delete('odor_data')
    except Exception as e:
        print(f"[{datetime.now()}] ERROR updating data: {str(e)}")
        print("Error details:", e.__class__.__name__)
        import traceback
        print(traceback.format_exc())
        
def calculate_intensity_no_decay(intensity: float) -> float:
    """
    Calculate intensity without applying decay over time.
    
    Parameters:
        intensity (float): Initial intensity value
        
    Returns:
        float: Original intensity value, rounded to 2 decimal places
    """
    return round(float(intensity), 2)

def generate_heatmap_for_timerange(start_time: datetime, end_time: datetime) -> Dict[str, Any]:
    """
    Generate heatmap data for a specific time range without decay.
    
    Parameters:
        start_time (datetime): Start of the time range
        end_time (datetime): End of the time range
        
    Returns:
        Dict[str, Any]: GeoJSON compatible dictionary containing heatmap data
        
    Raises:
        Exception: If there's an error accessing or processing the data
    """
    israel_tz = tz.gettz('Asia/Jerusalem')
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=israel_tz)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=israel_tz)
        
    creds = get_google_credentials()
    gc = gspread.authorize(creds)
    
    sheet_url = "https://docs.google.com/spreadsheets/d/1PMm_4Xkrv4Bmy7p9pI8Smnqzl12xgBVotYBEb2O45cg"
    spreadsheet = gc.open_by_url(sheet_url)
    worksheet = spreadsheet.worksheet('Sheet1')
    
    df = process_raw_dataframe(get_as_dataframe(worksheet))
    df = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)]
    
    odor_df = df[df['סוג דיווח'] == 'מפגע ריח'].copy()
    waste_df = df[df['סוג דיווח'] == 'מפגע פסולת'].copy()
    
    valid_smoke_colors = ['לבן', 'אפור', 'שחור']
    valid_waste_df = waste_df[
        waste_df['צבע העשן'].isin(valid_smoke_colors) &
        waste_df['צבע העשן'].notna() &
        (waste_df['צבע העשן'] != 'אין עשן')
    ].copy()
    
    valid_waste_df['עוצמת הריח'] = 6
    
    odor_df = split_coordinates(odor_df)
    valid_waste_df = split_coordinates(valid_waste_df)
    
    odor_df['עוצמת הריח'] = pd.to_numeric(odor_df['עוצמת הריח'], errors='coerce').fillna(0)
    odor_df['intensity'] = odor_df['עוצמת הריח'].apply(calculate_intensity_no_decay)
    valid_waste_df['intensity'] = valid_waste_df['עוצמת הריח'].apply(calculate_intensity_no_decay)
    
    odor_df = randomize_coordinates(odor_df)
    
    combined_df = pd.concat([odor_df, valid_waste_df])
    combined_df = combined_df[combined_df['intensity'] > 0]
    
    if len(combined_df) > 0:
        combined_df = combined_df.sort_values(by='datetime', ascending=True)
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
    
    return create_geojson_no_buffer(combined_gdf)

@app.route('/')
def home() -> Dict[str, Any]:
    """
    Serve home endpoint with API status.
    
    Returns:
        Dict[str, Any]: Dictionary containing API status and available endpoints
    """
    return jsonify({
        "status": "running",
        "last_update": last_update_time.isoformat() if last_update_time else None,
        "endpoints": {
            "odor": "/odor",
            "odor_historical": "/odor/historical",
            "odor_timerange": "/odor/timerange",
            "odor_archive": "/odor/archive",
            "status": "/status"
        }
    })

@app.route('/odor/archive')
@cache.cached(timeout=300)  # Cache for 5 minutes since historical data changes less frequently
def serve_archive_odor_geojson() -> Union[Response, Tuple[Dict[str, str], int]]:
    """
    Serve complete archive of all odor data as GeoJSON.
    
    Returns:
        Union[Response, Tuple[Dict[str, str], int]]: Gzipped GeoJSON Response or error tuple
    """
    try:
        archive_geojson = generate_heatmap_for_timerange(FIRST_REPORT_DATE, 
            datetime.now(tz.gettz('Asia/Jerusalem')) + timedelta(days=1))
        return gzip_response(archive_geojson)
    except Exception as e:
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

@app.route('/odor')
@cache.cached(timeout=60, key_prefix='odor_data')
def serve_odor_geojson() -> Union[Response, Tuple[Dict[str, str], int]]:
    """
    Serve latest odor data as GeoJSON.
    
    Returns:
        Union[Response, Tuple[Dict[str, str], int]]: Gzipped GeoJSON Response or error tuple
    """
    if latest_odor_geojson is None:
        return jsonify({"error": "No data available"}), 503
    return gzip_response(latest_odor_geojson)

@app.route('/odor/historical')
def serve_historical_odor_geojson() -> Union[Response, Tuple[Dict[str, str], int]]:
    """
    Serve odor data as GeoJSON for a specific timestamp.
    
    Parameters:
        timestamp (str): ISO format timestamp from query parameter
        
    Returns:
        Union[Response, Tuple[Dict[str, str], int]]: Gzipped GeoJSON Response or error tuple
        
    Raises:
        ValueError: If timestamp format is invalid
        Exception: If there's an error processing the request
    """
    timestamp_str = request.args.get('timestamp')
    if not timestamp_str:
        return jsonify({"error": "Missing timestamp parameter"}), 400
    
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
        historical_geojson = generate_heatmap_for_timestamp(timestamp)
        return gzip_response(historical_geojson)
    except ValueError as e:
        return jsonify({"error": f"Invalid timestamp format: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

@app.route('/odor/timerange')
def serve_timerange_odor_geojson() -> Union[Response, Tuple[Dict[str, str], int]]:
    """
    Serve odor data as GeoJSON for a specific time range without decay.
    
    Parameters:
        start_time (str): ISO format timestamp for range start
        end_time (str): ISO format timestamp for range end
        
    Returns:
        Union[Response, Tuple[Dict[str, str], int]]: Gzipped GeoJSON Response or error tuple
        
    Raises:
        ValueError: If timestamp format is invalid
        Exception: If there's an error processing the request
    """
    start_time_str = request.args.get('start_time')
    end_time_str = request.args.get('end_time')
    
    if not start_time_str or not end_time_str:
        return jsonify({"error": "Missing start_time or end_time parameter"}), 400
    
    try:
        start_time = datetime.fromisoformat(start_time_str)
        end_time = datetime.fromisoformat(end_time_str)
        
        if start_time > end_time:
            return jsonify({"error": "start_time must be before end_time"}), 400
            
        timerange_geojson = generate_heatmap_for_timerange(start_time, end_time)
        return gzip_response(timerange_geojson)
        
    except ValueError as e:
        return jsonify({"error": f"Invalid timestamp format: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

@app.route('/status')
def server_status() -> Dict[str, Any]:
    """
    Serve server status information.
    
    Returns:
        Dict[str, Any]: Dictionary containing current server status details
    """
    return jsonify({
        "status": "running",
        "last_update": last_update_time.isoformat() if last_update_time else None,
        "has_odor_data": latest_odor_geojson is not None
    })

def run_schedule() -> None:
    """
    Run scheduler loop to update data periodically.
    
    Note:
        This function runs indefinitely in a separate thread
        and updates the data every minute
    """
    while True:
        schedule.run_pending()
        time_module.sleep(1)

def initialize_app() -> None:
    """
    Initialize the application and start scheduler.
    
    Note:
        - Performs initial data update
        - Sets up minute-by-minute scheduler
        - Starts scheduler in daemon thread
        
    Raises:
        Exception: If initialization fails
    """
    try:
        update_data()
        schedule.every(1).minutes.do(update_data)
        scheduler_thread = threading.Thread(target=run_schedule)
        scheduler_thread.daemon = True
        scheduler_thread.start()
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        import traceback
        print(traceback.format_exc())

initialize_app()

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)