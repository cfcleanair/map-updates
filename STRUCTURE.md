# Structure

Annotated map of the repo, the deploy couplings that must not drift, and the
frozen zones.

## File tree

```
map-updates/
├── app.py                     # Flask app + heatmap pipeline; module-level `app`, runs initialize_app() at import
├── config.py                  # ALL constants + env access (SHEET_URL, WORKSHEET_NAME, FIRST_REPORT_DATE, REQUIRED_COLUMNS, get_google_credentials_json, get_port)
├── Procfile                   # Railway backend: gunicorn app:app --preload
├── runtime.txt                # pins python-3.9 for the Railway buildpack
├── requirements.txt           # backend deps installed by Railway
├── .env.example               # documents GOOGLE_CREDENTIALS + PORT
├── README.md                  # what it does, URLs, env vars, run/test/deploy
├── STRUCTURE.md               # this file
├── tests/                     # pytest characterization suite (pytest)
│   └── test_heatmap.py        # exact-output tests for the heatmap generators
├── public/                    # SEPARATE deploy target — GitHub Pages
│   └── index.html             # MapLibre frontend; DATA_URL points at the Railway backend
└── .github/
    └── workflows/
        └── deploy.yml         # publishes public/ to GitHub Pages on push to main
```

## Deploy couplings (must not drift)

- **`Procfile` → `gunicorn app:app`**: `app.py` must stay at repo root and export
  a module-level `app`. Renaming the file or the variable breaks the Railway boot.
- **`--preload` + module-level `initialize_app()`**: the blocking first fetch and
  daemon scheduler thread run once, before workers fork. Must stay at module scope
  (not under `if __name__ == '__main__'` and not inside an app factory) or workers
  boot with a cold/empty snapshot.
- **`runtime.txt` = `python-3.9`**: all backend code must be 3.9-compatible.
- **`public/index.html` `DATA_URL`** = `https://web-production-7226.up.railway.app/odor`:
  the frontend's only link to the backend. Changing the backend host means editing
  this constant.
- **`.github/workflows/deploy.yml` `path: './public'`**: the GitHub Pages artifact
  root. The frontend must live under `public/`.
- **`config.py` constants ↔ Google Sheet**: `SHEET_URL`, `WORKSHEET_NAME`, and every
  header in `REQUIRED_COLUMNS` must byte-match the live sheet.

## Frozen zones (never refactor, reformat, or find-replace)

- **Hebrew string literals** in `config.py` (`REQUIRED_COLUMNS`) and `app.py`
  (report-type keys `מפגע ריח` / `מפגע פסולת`, smoke colors `לבן`/`אפור`/`שחור`,
  the `אין עשן` / `המיקום לא נמצא` sentinels). These are exact-match sheet headers
  and routing keys — byte-exact or broken.
- **`initialize_app()` + `run_schedule()`** module-level wiring — required for
  `gunicorn --preload`. Do not relocate.
- **`public/`** — owned by the GitHub Pages deploy, not the backend layout.
