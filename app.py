"""Flask service that serves odor and waste report heatmaps as GeoJSON.

Fetches crowdsourced reports from a Google Sheets backend, processes them into
intensity-weighted point features (with optional time-based decay and random
coordinate jitter), and exposes them through cached HTTP endpoints. A background
scheduler refreshes the latest snapshot once per minute so the public `/odor`
endpoint stays responsive without blocking on a remote sheet read.
"""

from flask import Flask, jsonify, Response, request
from flask_cors import CORS
from flask_caching import Cache
import gzip
import schedule
import threading
import time as time_module
import traceback
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
    'ספאם',
    'אופי הריח',
    'חומר נשרף משוער',
    'תסמינים רפואיים'
]
FIRST_REPORT_DATE = datetime(2024, 4, 4, tzinfo=tz.gettz('Asia/Jerusalem'))
SHEET_URL = "https://docs.google.com/spreadsheets/d/1PMm_4Xkrv4Bmy7p9pI8Smnqzl12xgBVotYBEb2O45cg"
WORKSHEET_NAME = 'Sheet1'

class CustomJSONEncoder(json.JSONEncoder):
    """JSON encoder that serializes datetime, time, and timedelta values.

    Extends the standard library encoder so that response payloads containing
    datetime-like objects can be dumped to JSON without raising TypeError.
    """

    def default(self, obj: Any) -> str:
        """Convert datetime, time, and timedelta objects into JSON-safe strings.

        Args:
            obj (Any): Object that the base encoder did not know how to handle.

        Returns:
            str: ISO 8601 string for ``datetime``/``time`` values, ``str(obj)``
            for ``timedelta`` values, or the base encoder's fallback for any
            other type.

        Raises:
            TypeError: Propagated from ``super().default`` when ``obj`` is not a
                supported type.
        """
        if isinstance(obj, (datetime, time)):
            return obj.isoformat()
        elif isinstance(obj, timedelta):
            return str(obj)
        return super().default(obj)

def gzip_response(data: Dict[str, Any]) -> Response:
    """Encode a payload as JSON and return it gzipped with caching headers.

    Args:
        data (Dict[str, Any]): Mapping that will be serialized with
            ``CustomJSONEncoder`` before being gzipped.

    Returns:
        Response: Flask response carrying the gzipped JSON body together with
        CORS, ``Content-Encoding: gzip``, ``Cache-Control``, and (when
        ``last_update_time`` is populated) ``Last-Modified`` and
        ``X-Data-Last-Update`` headers.
    """
    gzip_buffer = gzip.compress(json.dumps(data, cls=CustomJSONEncoder).encode('utf-8'))
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Content-Encoding': 'gzip',
        'Cache-Control': 'public, max-age=60'
    }
    if last_update_time:
        headers['Last-Modified'] = last_update_time.astimezone(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
        headers['X-Data-Last-Update'] = last_update_time.isoformat()
    return Response(gzip_buffer, mimetype='application/json', headers=headers)

def get_google_credentials() -> service_account.Credentials:
    """Build Google service-account credentials from the ``GOOGLE_CREDENTIALS`` env var.

    Args:
        None.

    Returns:
        service_account.Credentials: Credentials scoped for Google Sheets and
        Drive access, constructed from the JSON blob stored in the
        ``GOOGLE_CREDENTIALS`` environment variable.

    Raises:
        ValueError: If ``GOOGLE_CREDENTIALS`` is unset or empty.
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
    """Clean the raw Google Sheets dataframe and compute report ages.

    Validates that every column in ``REQUIRED_COLUMNS`` is present, drops rows
    flagged as test or spam submissions, removes entries with unusable
    coordinates, parses the date and time columns into a Jerusalem-localized
    ``datetime``, and finally computes ``time_elapsed_minutes`` relative to
    ``reference_time``.

    Args:
        df (pd.DataFrame): Raw frame as returned by
            ``gspread_dataframe.get_as_dataframe``.
        reference_time (Optional[datetime]): Reference point for the elapsed
            time calculation. Naive datetimes are treated as Asia/Jerusalem;
            ``None`` (the default) uses the current time in that zone.

    Returns:
        pd.DataFrame: Filtered frame retaining the original required columns
        (minus ``בדיקה`` and ``ספאם``) plus ``datetime`` and
        ``time_elapsed_minutes`` (clipped to be non-negative).

    Raises:
        ValueError: If any column listed in ``REQUIRED_COLUMNS`` is missing
            from ``df``.
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    
    israel_tz = tz.gettz('Asia/Jerusalem')
    df = df[REQUIRED_COLUMNS].copy()
    df = df[(df['בדיקה'].fillna(0) != 1) & (df['ספאם'].fillna(0) != 1)]
    df = df.drop(columns=['בדיקה', 'ספאם'])
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
    """Return the per-minute linear decay rate for a given starting intensity.

    The model assumes a report decays linearly to zero over 100 minutes, so the
    rate is simply ``initial_intensity / 100``. A zero starting intensity
    short-circuits to ``0`` to avoid division-style ambiguity downstream.

    Args:
        initial_intensity (float): Reported odor intensity at the time of
            submission.

    Returns:
        float: Decay rate expressed in intensity units per minute.
    """
    return initial_intensity / 100 if initial_intensity != 0 else 0

def calculate_intensity(row: pd.Series) -> float:
    """Compute the current intensity for a report after linear decay.

    Reports older than 100 minutes are treated as fully decayed and return
    zero; otherwise the intensity is decreased by ``decay_rate * time_elapsed``
    and clamped to be non-negative.

    Args:
        row (pd.Series): Row that must expose ``time_elapsed_minutes`` and the
            Hebrew ``עוצמת הריח`` (initial intensity) field.

    Returns:
        float: Decayed intensity rounded to two decimal places.
    """
    if row['time_elapsed_minutes'] > 100:
        return 0.0
    
    initial_intensity = float(row['עוצמת הריח'])
    time_elapsed = float(row['time_elapsed_minutes'])
    decay_rate = calculate_decay_rate(initial_intensity)
    current_intensity = max(0, initial_intensity - (decay_rate * time_elapsed))
    return round(current_intensity, 2)

def split_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the ``קואורדינטות`` ``"lat,lon"`` strings into numeric columns.

    Filters rows whose coordinate string does not match a basic ``lat,lon``
    pattern, then splits the remaining values into numeric ``lat`` and ``lon``
    columns. Rows that fail numeric conversion are dropped.

    Args:
        df (pd.DataFrame): Frame containing a ``קואורדינטות`` column with
            comma-separated coordinate strings.

    Returns:
        pd.DataFrame: Copy of ``df`` with valid rows only, augmented with
        ``lat`` and ``lon`` float columns.
    """
    df = df[df['קואורדינטות'].str.match(r'^-?\d+\.?\d*,-?\d+\.?\d*$', na=False)].copy()
    coords_split = df['קואורדינטות'].str.split(',', expand=True)
    df.loc[:, 'lat'] = pd.to_numeric(coords_split[0], errors='coerce')
    df.loc[:, 'lon'] = pd.to_numeric(coords_split[1], errors='coerce')
    return df[df['lat'].notna() & df['lon'].notna()].copy()

def randomize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Jitter ``lat``/``lon`` values by up to 300 metres on the WGS84 ellipsoid.

    For each row a random bearing and distance (0–300 m) are sampled and the
    coordinate is displaced accordingly, providing privacy for reporters while
    preserving the rough geographic distribution of the incidents.

    Args:
        df (pd.DataFrame): Frame with numeric ``lat`` and ``lon`` columns in
            degrees.

    Returns:
        pd.DataFrame: The same frame mutated in place with the displaced
        coordinates assigned back to ``lat`` and ``lon``.
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
    """Convert a ``GeoDataFrame`` into a GeoJSON ``FeatureCollection`` dictionary.

    NaN values are normalised to ``None`` so the resulting JSON is valid, and
    each non-geometry column is emitted as a feature property.

    Args:
        gdf (gpd.GeoDataFrame): Frame whose ``geometry`` column holds the
            feature geometries; all other columns become ``properties``.

    Returns:
        Dict[str, Any]: GeoJSON-compatible dictionary with ``type`` set to
        ``"FeatureCollection"`` and one ``Feature`` per input row.
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
    """Build a decayed heatmap GeoJSON snapshot relative to ``timestamp``.

    Pulls the full Google Sheet, splits odor reports from waste reports
    (keeping only waste rows with visible smoke), normalises waste intensities
    to ``6``, applies linear decay to both groups, jitters odor coordinates,
    and packages the surviving points into a GeoJSON ``FeatureCollection``
    sorted by submission time.

    Args:
        timestamp (datetime): Reference time used by ``process_raw_dataframe``
            when computing how stale each report is.

    Returns:
        Dict[str, Any]: GeoJSON ``FeatureCollection`` containing every report
        whose decayed intensity is still positive.

    Raises:
        ValueError: Propagated from ``get_google_credentials`` or
            ``process_raw_dataframe`` when configuration or data is invalid.
        Exception: Propagated network or gspread errors when the sheet cannot
            be read.
    """
    creds = get_google_credentials()
    gc = gspread.authorize(creds)
    spreadsheet = gc.open_by_url(SHEET_URL)
    worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
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
    """Refresh the cached odor snapshot and invalidate downstream caches.

    Regenerates ``latest_odor_geojson`` for the current Jerusalem time, records
    ``last_update_time``, and evicts the ``odor_data`` cache entry so the next
    request reflects the fresh data. Any exception is caught and logged; the
    previous snapshot is left untouched so the service keeps serving stale but
    valid data.

    Args:
        None.

    Returns:
        None.
    """
    global latest_odor_geojson, last_update_time
    try:
        latest_odor_geojson = generate_heatmap_for_timestamp(datetime.now(tz.gettz('Asia/Jerusalem')))
        last_update_time = datetime.now(tz.gettz('Asia/Jerusalem'))
        cache.delete('odor_data')
    except Exception as e:
        print(f"[{datetime.now()}] ERROR updating data: {str(e)}")
        print("Error details:", e.__class__.__name__)
        print(traceback.format_exc())

def calculate_intensity_no_decay(intensity: float) -> float:
    """Return the input intensity unchanged, rounded to two decimal places.

    Used by the time-range endpoint where historical reports should not fade
    with elapsed time.

    Args:
        intensity (float): Reported odor intensity.

    Returns:
        float: ``intensity`` cast to ``float`` and rounded to two decimals.
    """
    return round(float(intensity), 2)

def generate_heatmap_for_timerange(start_time: datetime, end_time: datetime) -> Dict[str, Any]:
    """Build a non-decayed heatmap GeoJSON snapshot for ``[start_time, end_time]``.

    Identical pipeline to :func:`generate_heatmap_for_timestamp` except that
    intensities are preserved at their original value (no time decay) and the
    raw dataframe is filtered to the requested window before processing. Naive
    inputs are interpreted as Asia/Jerusalem.

    Args:
        start_time (datetime): Inclusive lower bound of the report window.
        end_time (datetime): Inclusive upper bound of the report window.

    Returns:
        Dict[str, Any]: GeoJSON ``FeatureCollection`` covering every valid
        report whose timestamp falls within the requested window.

    Raises:
        ValueError: Propagated from ``get_google_credentials`` or
            ``process_raw_dataframe`` when configuration or data is invalid.
        Exception: Propagated network or gspread errors when the sheet cannot
            be read.
    """
    israel_tz = tz.gettz('Asia/Jerusalem')
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=israel_tz)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=israel_tz)
    creds = get_google_credentials()
    gc = gspread.authorize(creds)
    spreadsheet = gc.open_by_url(SHEET_URL)
    worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
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
    """Serve a JSON description of the running service and its endpoints.

    Args:
        None.

    Returns:
        Dict[str, Any]: Flask JSON response with the keys ``status``,
        ``last_update`` (ISO string or ``None``), and ``endpoints`` (a map of
        the public routes exposed by this service).
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
@cache.cached(timeout=300)
def serve_archive_odor_geojson() -> Union[Response, Tuple[Dict[str, str], int]]:
    """Serve the full archive of reports from ``FIRST_REPORT_DATE`` to tomorrow.

    Cached for five minutes by Flask-Caching to amortise the cost of pulling
    the entire sheet.

    Args:
        None.

    Returns:
        Union[Response, Tuple[Dict[str, str], int]]: Gzipped GeoJSON
        ``Response`` on success, or a ``(json_error, 500)`` tuple if the
        underlying heatmap generation fails.
    """
    try:
        archive_geojson = generate_heatmap_for_timerange(FIRST_REPORT_DATE, datetime.now(tz.gettz('Asia/Jerusalem')) + timedelta(days=1))
        return gzip_response(archive_geojson)
    except Exception as e:
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

@app.route('/odor')
def serve_odor_geojson() -> Union[Response, Tuple[Dict[str, str], int]]:
    """Serve the latest cached odor snapshot, refreshing it if stale.

    Triggers :func:`update_data` synchronously when the cached snapshot is
    older than 75 seconds (or absent) so the response always reflects fresh
    data when the background scheduler has fallen behind.

    Args:
        None.

    Returns:
        Union[Response, Tuple[Dict[str, str], int]]: Gzipped GeoJSON
        ``Response`` on success, or a ``({"error": ...}, 503)`` tuple when no
        snapshot is available yet.
    """
    israel_tz = tz.gettz('Asia/Jerusalem')
    now_il = datetime.now(israel_tz)
    stale = (last_update_time is None) or ((now_il - last_update_time) > timedelta(seconds=75))
    if stale:
        update_data()
    if latest_odor_geojson is None:
        return jsonify({"error": "No data available"}), 503
    return gzip_response(latest_odor_geojson)

@app.route('/odor/historical')
def serve_historical_odor_geojson() -> Union[Response, Tuple[Dict[str, str], int]]:
    """Serve a decayed heatmap snapshot for a single user-supplied timestamp.

    Reads the ISO 8601 ``timestamp`` query string parameter and delegates to
    :func:`generate_heatmap_for_timestamp`. Validation and processing errors
    are caught and reported as JSON.

    Args:
        None. The ``timestamp`` query parameter is read from
        ``flask.request.args`` rather than the function signature.

    Returns:
        Union[Response, Tuple[Dict[str, str], int]]: Gzipped GeoJSON
        ``Response`` on success, or a ``({"error": ...}, status)`` tuple with
        ``400`` for missing/invalid timestamps and ``500`` for processing
        failures.

    Raises:
        ValueError: Caught internally and surfaced as a ``400`` JSON error
            when ``timestamp`` cannot be parsed.
        Exception: Caught internally and surfaced as a ``500`` JSON error for
            unexpected downstream failures.
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
    """Serve a non-decayed heatmap snapshot for a user-supplied time window.

    Reads the ISO 8601 ``start_time`` and ``end_time`` query parameters,
    strips any spurious ``?...`` suffix appended by some clients, and
    delegates to :func:`generate_heatmap_for_timerange`.

    Args:
        None. ``start_time`` and ``end_time`` are read from
        ``flask.request.args`` rather than the function signature.

    Returns:
        Union[Response, Tuple[Dict[str, str], int]]: Gzipped GeoJSON
        ``Response`` on success, or a ``({"error": ...}, status)`` tuple with
        ``400`` for missing/invalid input (including reversed ranges) and
        ``500`` for processing failures.

    Raises:
        ValueError: Caught internally and surfaced as a ``400`` JSON error
            when either timestamp cannot be parsed.
        Exception: Caught internally and surfaced as a ``500`` JSON error for
            unexpected downstream failures.
    """
    start_time_str = request.args.get('start_time')
    end_time_str = request.args.get('end_time')
    if not start_time_str or not end_time_str:
        return jsonify({"error": "Missing start_time or end_time parameter"}), 400
    start_time_str = start_time_str.split('?')[0]
    end_time_str = end_time_str.split('?')[0]
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
    """Serve a lightweight health/status payload for monitoring.

    Args:
        None.

    Returns:
        Dict[str, Any]: Flask JSON response containing ``status``,
        ``last_update`` (ISO 8601 or ``None``), and ``has_odor_data`` (whether
        a cached snapshot is currently available).
    """
    return jsonify({
        "status": "running",
        "last_update": last_update_time.isoformat() if last_update_time else None,
        "has_odor_data": latest_odor_geojson is not None
    })

def run_schedule() -> None:
    """Drive the ``schedule`` library's run loop until the process exits.

    Intended to be run inside a daemon thread spawned by
    :func:`initialize_app`; ticks once per second so any due jobs (currently
    only the per-minute :func:`update_data` call) fire close to their target
    time.

    Args:
        None.

    Returns:
        None. The function loops forever.
    """
    while True:
        schedule.run_pending()
        time_module.sleep(1)

def initialize_app() -> None:
    """Warm the cache and start the background scheduler thread.

    Calls :func:`update_data` once at import time so the first request is
    served from a populated cache, registers ``update_data`` to run every
    minute via the ``schedule`` library, and starts :func:`run_schedule` in a
    daemon thread so it does not block process shutdown. Any exception is
    caught and logged with a traceback so a transient sheet failure on boot
    does not crash the application.

    Args:
        None.

    Returns:
        None.
    """
    try:
        update_data()
        schedule.every(1).minutes.do(update_data)
        scheduler_thread = threading.Thread(target=run_schedule)
        scheduler_thread.daemon = True
        scheduler_thread.start()
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        print(traceback.format_exc())

initialize_app()

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
