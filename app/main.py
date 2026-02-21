from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
from pathlib import Path

from app.schemas import PredictRequest, PredictResponse
from src.inference import load_bundle, predict_sales

BASE_DIR = Path(__file__).resolve().parents[1]   # project root
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"

app = FastAPI(title="Rossmann Sales Forecast API", version="0.1")

# ✅ 生产环境：可以先放宽，之后再收紧到你的域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BUNDLE = None

@app.on_event("startup")
def startup():
    global BUNDLE
    BUNDLE = load_bundle(
        model_dir=str(BASE_DIR / "models"),
        data_dir=str(BASE_DIR / "data" / "raw"),
    )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = pd.DataFrame([r.model_dump() for r in req.rows])
    preds = predict_sales(df, BUNDLE, use_store_weights=True)  # ✅ weighted
    return {"predictions": [float(x) for x in preds.tolist()]}

# ✅ 前端打包产物挂载到根路径
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")