#!/usr/bin/env sh
set -e

PORT="${PORT:-8000}"

echo "Starting server on port ${PORT}..."
exec gunicorn -k uvicorn.workers.UvicornWorker app.main:app \
  --bind "0.0.0.0:${PORT}" \
  --workers 1 \
  --timeout 120