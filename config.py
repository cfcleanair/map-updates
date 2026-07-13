"""Static configuration and environment access for the map-updates service.

Holds the import-time constants (Google Sheet location, required Hebrew column
headers, archive start date) together with lazy accessors for the environment
variables the Flask app reads at runtime. Centralising these keeps ``app.py``
free of scattered ``os.environ`` reads while preserving the original read
timing: the constants below are evaluated once at import, whereas the secret
and port are fetched on every call so behaviour is byte-identical to the
in-line reads they replaced.
"""

from datetime import datetime
from typing import Optional
from dateutil import tz
import os

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


def get_google_credentials_json() -> Optional[str]:
    """Read the raw ``GOOGLE_CREDENTIALS`` JSON blob from the environment.

    Performed lazily on every call (never cached) so the value picked up
    matches whatever is set at request time.

    Returns:
        Optional[str]: The JSON string if ``GOOGLE_CREDENTIALS`` is set,
        otherwise ``None``.
    """
    return os.getenv('GOOGLE_CREDENTIALS')


def get_port() -> int:
    """Resolve the port the development server binds to.

    Returns:
        int: The value of the ``PORT`` environment variable, or ``8080`` when
        it is unset.
    """
    return int(os.getenv('PORT', 8080))
