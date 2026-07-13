"""Pytest bootstrap that puts the repo root on ``sys.path``.

``app.py`` and ``config.py`` live at the repo root (deploy-load-bearing under
``gunicorn app:app``), so the suite under ``tests/`` imports them as top-level
modules. This makes the root importable regardless of pytest's rootdir
detection. It is test-only tooling and is never loaded by the Railway buildpack.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
