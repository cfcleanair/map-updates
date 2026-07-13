"""Characterization tests pinning the exact GeoJSON output of the heatmap generators.

Locks down the current behaviour of ``generate_heatmap_for_timestamp`` and
``generate_heatmap_for_timerange`` before the two near-duplicate functions are
unified. Synthetic report frames are pushed through a patched sheet fetch with a
frozen clock and a fixed RNG seed, and the full serialized output is compared
against golden snapshots committed alongside these tests. The golden files are
written automatically on the first run (when absent) and asserted against on
every run thereafter, so a refactor that changes any byte of output fails here.

Run with:
    uv run --with-requirements requirements.txt --with pytest --with freezegun pytest
"""

import ast
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dateutil import tz
from freezegun import freeze_time

import config
import app

FIXTURES = Path(__file__).parent / "fixtures"
SEED = 12345
FROZEN_NOW = "2026-07-10T10:00:00Z"  # 13:00 Asia/Jerusalem
IL = tz.gettz("Asia/Jerusalem")


def _literals():
    """Lift the Hebrew routing literals from app.py so tests never retype them.

    Returns:
        dict: ``odor`` / ``waste`` report-type keys, the ``smoke_colors`` list,
        and the ``loc_not_found`` coordinate sentinel, all read verbatim from
        the module source.
    """
    src = Path(app.__file__).read_text(encoding="utf-8")
    report_types = re.findall(r"== '([^']+)'\]\.copy\(\)", src)
    smoke_colors = ast.literal_eval(
        re.search(r"valid_smoke_colors = (\[[^\]]+\])", src).group(1)
    )
    loc_not_found = re.search(r"str\.contains\('([^']+)'", src).group(1)
    return {
        "odor": report_types[0],
        "waste": report_types[1],
        "smoke_colors": smoke_colors,
        "loc_not_found": loc_not_found,
    }


def build_reports_df():
    """Build the synthetic raw sheet frame exercising every processing branch.

    Returns:
        pd.DataFrame: Frame keyed by ``config.REQUIRED_COLUMNS`` holding valid
        odor rows, valid and invalid waste rows, and test/spam/bad-coordinate
        rows that must be filtered out.
    """
    cols = config.REQUIRED_COLUMNS
    lit = _literals()
    odor, waste = lit["odor"], lit["waste"]
    colors = lit["smoke_colors"]
    loc_nf = lit["loc_not_found"]
    nan = np.nan

    # (date, time, type, coords, intensity, smoke, test, spam, nature, burned, symptoms)
    rows = [
        ("10/07/2026", "12:00:00", odor, "32.05,34.75", 8, nan, 0, 0, "a", "b", "c"),
        ("10/07/2026", "12:05:00", odor, "32.06,34.76", 5, nan, 0, 0, "a", "b", "c"),
        ("10/07/2026", "12:10:00", odor, "32.07,34.77", 10, nan, 0, 0, "a", "b", "c"),
        (
            "10/07/2026",
            "12:15:00",
            waste,
            "32.08,34.78",
            3,
            colors[0],
            0,
            0,
            "a",
            "b",
            "c",
        ),
        (
            "10/07/2026",
            "12:20:00",
            waste,
            "32.09,34.79",
            3,
            colors[2],
            0,
            0,
            "a",
            "b",
            "c",
        ),
        ("10/07/2026", "12:25:00", waste, "32.10,34.80", 3, nan, 0, 0, "a", "b", "c"),
        ("10/07/2026", "12:30:00", odor, "32.11,34.81", 7, nan, 1, 0, "a", "b", "c"),
        ("10/07/2026", "12:35:00", odor, "32.12,34.82", 7, nan, 0, 1, "a", "b", "c"),
        ("10/07/2026", "12:40:00", odor, loc_nf, 7, nan, 0, 0, "a", "b", "c"),
        ("10/07/2026", "12:45:00", odor, "abc", 7, nan, 0, 0, "a", "b", "c"),
    ]
    return pd.DataFrame(rows, columns=cols)


class _FakeWorksheet:
    """Opaque stand-in for a gspread worksheet handle."""


class _FakeSpreadsheet:
    """Spreadsheet stub whose ``worksheet`` returns a worksheet stand-in."""

    def worksheet(self, name):
        """Return a fake worksheet regardless of the requested name.

        Args:
            name (str): Ignored worksheet name.

        Returns:
            _FakeWorksheet: A placeholder worksheet handle.
        """
        return _FakeWorksheet()


class _FakeGC:
    """gspread client stub whose ``open_by_url`` returns a spreadsheet stub."""

    def open_by_url(self, url):
        """Return a fake spreadsheet regardless of the requested URL.

        Args:
            url (str): Ignored spreadsheet URL.

        Returns:
            _FakeSpreadsheet: A placeholder spreadsheet handle.
        """
        return _FakeSpreadsheet()


@pytest.fixture(autouse=True)
def _stub_sheet(monkeypatch):
    """Serve the synthetic frame in place of the live Google Sheet.

    Also clears the background ``schedule`` job so the daemon refresh thread
    cannot consume the RNG mid-test and disturb the seeded coordinate jitter.
    """
    app.schedule.clear()
    df = build_reports_df()
    monkeypatch.setattr(app, "get_google_credentials", lambda: None)
    monkeypatch.setattr(app.gspread, "authorize", lambda creds: _FakeGC())
    monkeypatch.setattr(app, "get_as_dataframe", lambda ws: df.copy(deep=True))


def _canonical(output):
    """Round-trip an output dict through the app's JSON encoder.

    Args:
        output (dict): GeoJSON ``FeatureCollection`` returned by a generator.

    Returns:
        dict: JSON-normalised copy (datetimes stringified, tuples listified) so
        it compares cleanly against a loaded golden file.
    """
    return json.loads(json.dumps(output, cls=app.CustomJSONEncoder))


def _run(func, *args):
    """Invoke a generator deterministically under a frozen clock and fixed seed.

    Args:
        func (callable): The heatmap generator to call.
        *args: Positional arguments forwarded to ``func``.

    Returns:
        dict: The canonicalised GeoJSON output.
    """
    with freeze_time(FROZEN_NOW):
        np.random.seed(SEED)
        return _canonical(func(*args))


def _check(output, name):
    """Compare output against a golden snapshot, writing it if absent.

    On the first run the golden file is created and the test is skipped; every
    subsequent run asserts byte-for-byte equality.

    Args:
        output (dict): Canonicalised generator output.
        name (str): Golden fixture base name.
    """
    path = FIXTURES / f"{name}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pytest.skip(f"wrote golden snapshot {name}")
    golden = json.loads(path.read_text(encoding="utf-8"))
    assert output == golden


def test_timestamp_nonempty():
    """Decayed snapshot with surviving odor and smoke-waste points is unchanged."""
    ref = datetime(2026, 7, 10, 12, 50, tzinfo=IL)
    _check(_run(app.generate_heatmap_for_timestamp, ref), "timestamp_nonempty")


def test_timestamp_empty():
    """Fully-decayed reference produces the empty FeatureCollection path unchanged."""
    ref = datetime(2026, 7, 10, 20, 0, tzinfo=IL)
    _check(_run(app.generate_heatmap_for_timestamp, ref), "timestamp_empty")


def test_timerange_nonempty():
    """Non-decayed window snapshot over all reports is unchanged."""
    start = datetime(2026, 7, 10, 0, 0, tzinfo=IL)
    end = datetime(2026, 7, 11, 0, 0, tzinfo=IL)
    _check(_run(app.generate_heatmap_for_timerange, start, end), "timerange_nonempty")


def test_timerange_empty_window_raises():
    """A window matching no reports raises KeyError (pinned current behavior).

    The window empties the frame before ``split_coordinates`` runs, where
    ``str.split(expand=True)`` yields a column-less frame and ``coords_split[0]``
    raises ``KeyError``; the route handler turns this into a 500. Locked so the
    refactor preserves this exact (pre-existing) behavior rather than silently
    "fixing" it.
    """
    start = datetime(2025, 1, 1, 0, 0, tzinfo=IL)
    end = datetime(2025, 1, 2, 0, 0, tzinfo=IL)
    with freeze_time(FROZEN_NOW):
        np.random.seed(SEED)
        with pytest.raises(KeyError):
            app.generate_heatmap_for_timerange(start, end)
