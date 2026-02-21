#!/usr/bin/env sh
set -e

PORT="${PORT:-8000}"

MODEL_DIR="/app/models"
MODEL_FILE="${MODEL_DIR}/xgb_model.json"
MODEL_URL="${MODEL_URL:-}"

mkdir -p "${MODEL_DIR}"

if [ ! -f "${MODEL_FILE}" ]; then
  if [ -z "${MODEL_URL}" ]; then
    echo "ERROR: MODEL_URL is not set and model file not found: ${MODEL_FILE}"
    exit 1
  fi
  echo "Downloading model from: ${MODEL_URL}"
  curl -L "${MODEL_URL}" -o "${MODEL_FILE}"
  echo "Downloaded model:"
  ls -lh "${MODEL_FILE}"
fi

echo "Starting server on port ${PORT}..."
exec gunicorn -k uvicorn.workers.UvicornWorker app.main:app \
  --bind "0.0.0.0:${PORT}" \
  --workers 1 \
  --timeout 120
