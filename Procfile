web: gunicorn app:app \
      --bind 0.0.0.0:${PORT:-8000} \      # same port Koyeb routes + safe local fallback
      --workers 2 \                       # 2 worker processes (0.5 vCPU × 2 ≈ 1 logical core)
      --threads 4 \                       # each worker can serve 4 concurrent requests
      --preload \                         # import heavy libs once at master-start → faster forks
      --keep-alive 75 \                   # keep connections warm, good for browsers / proxies
      --timeout 90 \                      # long enough for Google-Sheets round-trips
      --log-level info                    # useful diagnostics without flooding logs
