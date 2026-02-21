# ---------- 1) Build frontend ----------
FROM node:20-alpine AS frontend-build
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---------- 2) Build backend ----------
FROM python:3.12-slim AS backend
WORKDIR /app

# 安装依赖（如果你有 xgboost/scipy，slim 一般也够用）
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 复制后端代码与模型/数据
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# 复制前端 build 产物到 /app/frontend/dist
COPY --from=frontend-build /frontend/dist ./frontend/dist

# Railway 会给 PORT 环境变量
ENV PORT=8000
EXPOSE 8000

# 生产启动：绑定 0.0.0.0 + 使用 $PORT
CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:${PORT} --workers 1 --timeout 120"]