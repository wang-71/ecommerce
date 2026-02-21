from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path

from app.schemas import PredictRequest, PredictResponse
from src.inference import load_bundle, predict_sales

# ✅ 只创建一次 app
app = FastAPI(title="Rossmann Sales Forecast API", version="0.1")

# ✅ CORS：让 React (localhost:5173) 能访问后端
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BUNDLE = None
BASE_DIR = Path(__file__).resolve().parents[1]  # project root


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

    # ✅ 默认不要开 store weights（除非你确定 CLI 也开了）
    preds = predict_sales(df, BUNDLE, use_store_weights=True)

    return {"predictions": [float(x) for x in preds.tolist()]}