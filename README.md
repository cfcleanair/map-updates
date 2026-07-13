# map-updates

Flask service that turns crowdsourced odor and waste reports (stored in a Google
Sheet) into intensity-weighted GeoJSON heatmaps, plus a static MapLibre frontend
that renders them. Odor reports decay linearly to zero over 100 minutes and are
jittered up to 300 m for privacy; waste reports with visible smoke are pinned to
a fixed intensity. A background scheduler refreshes the live snapshot once per
minute so `/odor` stays responsive without blocking on a remote sheet read.

## Deploy targets

Two independent deploys from this one repo:

- **Backend (API)** — Railway, via `Procfile` (`gunicorn app:app`). Runtime pinned
  to Python 3.9 by `runtime.txt`. Live at
  `https://web-production-7226.up.railway.app` (see `public/index.html`'s `DATA_URL`).
- **Frontend (map)** — GitHub Pages, via `.github/workflows/deploy.yml`, which
  publishes `public/` on every push to `main`.

## HTTP endpoints

| Route | Description |
|-------|-------------|
| `/` | Service status + endpoint index (JSON) |
| `/odor` | Latest decayed snapshot, gzipped GeoJSON (cached, self-refreshing) |
| `/odor/historical?timestamp=<iso>` | Decayed snapshot at a given timestamp |
| `/odor/timerange?start_time=<iso>&end_time=<iso>` | Non-decayed window |
| `/odor/archive` | Full archive from `FIRST_REPORT_DATE` to tomorrow (cached 5 min) |
| `/status` | Health payload (last update, whether data is loaded) |

## Environment variables

| Variable | Required | Default | Read by | Purpose |
|----------|----------|---------|---------|---------|
| `GOOGLE_CREDENTIALS` | yes | — | `config.get_google_credentials_json()` | Service-account JSON blob for Sheets/Drive read access |
| `PORT` | no | `8080` | `config.get_port()` | Dev-server bind port (`Procfile` uses `${PORT:-8000}` in prod) |

See `.env.example`. All env access lives in `config.py`.

## Run locally

```bash
export GOOGLE_CREDENTIALS="$(cat service-account.json)"
uv run --with-requirements requirements.txt python app.py
# serves on http://0.0.0.0:8080
```

Open `public/index.html` directly (or serve it statically) to view the map; it
fetches from the `DATA_URL` hard-coded near the top of the file.

## Test

```bash
uv run --with-requirements requirements.txt --with pytest pytest
```

## Deploy notes

- `app.py` must stay at the repo root exporting a module-level `app`; the
  `Procfile` target is `gunicorn app:app --preload`. `--preload` runs
  `initialize_app()` (a blocking first fetch plus a daemon scheduler thread) once
  before workers fork.
- `runtime.txt` pins `python-3.9`; keep all backend code 3.9-compatible.
- Do not restructure `public/` — it is a separate GitHub Pages deploy target.
