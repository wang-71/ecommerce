# ---------- Frontend build ----------
FROM node:20-alpine AS frontend-build
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


# ---------- Backend runtime ----------
FROM python:3.12-slim AS backend
WORKDIR /app

# System deps (optional but often useful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY app/ ./app/
COPY src/ ./src/

# Copy model artifacts (keep these in git)
COPY models/ ./models/

# Do NOT copy training data into image
# COPY data/ ./data/raw/

# Create runtime dirs (safe even if not used)
RUN mkdir -p /app/data/raw /app/results

# Railway provides PORT env var; keep your command
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh
CMD ["/app/start.sh"]