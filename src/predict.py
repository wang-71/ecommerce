# src/predict.py
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import xgboost as xgb


# -------------------------
# Logging
# -------------------------
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("predict")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
        logger.addHandler(h)
    return logger


# -------------------------
# Feature engineering (same intent as train.py)
# -------------------------
def add_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("Found invalid dates in column 'Date'.")
    df["year"] = df["Date"].dt.year.astype("int32")
    df["month"] = df["Date"].dt.month.astype("int8")
    df["day"] = df["Date"].dt.day.astype("int8")
    return df


def add_is_promo_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    month2str = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    df["monthstr"] = df["month"].map(month2str)

    promo = df.get("PromoInterval")
    if promo is None:
        df["IsPromoMonth"] = 0
        return df

    promo = promo.fillna("0")

    def _is_promo(row) -> int:
        pi = row["PromoInterval"]
        if pi == 0 or pi == "0":
            return 0
        return 1 if row["monthstr"] in str(pi) else 0

    df["IsPromoMonth"] = df.apply(_is_promo, axis=1).astype("int8")
    return df


def map_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mappings = {
        "0": 0, 0: 0, b"0": 0,
        "a": 1, b"a": 1,
        "b": 2, b"b": 2,
        "c": 3, b"c": 3,
        "d": 4, b"d": 4,
    }
    for col in ["StoreType", "Assortment", "StateHoliday"]:
        if col in df.columns:
            df[col] = df[col].replace(mappings).astype("Int64")
    return df


def build_test_features(test: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    # Merge store
    test = test.merge(store, on="Store", how="left")

    # Feature engineering
    test = add_date_parts(test)
    test = add_is_promo_month(test)
    test = map_categoricals(test)

    # Drop columns (match training intent)
    df_test = test.drop(["Date", "monthstr", "PromoInterval", "Open"], axis=1, errors="ignore")

    return df_test


# -------------------------
# Prediction
# -------------------------
def load_store_weights(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    w = pd.read_csv(path)
    # Expect columns: Store, weight
    if "Store" not in w.columns or "weight" not in w.columns:
        raise ValueError("store_weights.csv must have columns: Store, weight")
    return w[["Store", "weight"]]


def main():
    logger = setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw", help="Folder with test.csv and store.csv")
    parser.add_argument("--model-dir", default="models", help="Folder with xgb_model.json, feature_names.json, (optional) store_weights.csv")
    parser.add_argument("--output", default="results.csv", help="Output submission csv path")
    parser.add_argument("--use-store-weights", action="store_true", help="Apply per-store calibration weights if store_weights.csv exists")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    out_path = Path(args.output)

    # Load raw
    test_path = data_dir / "test.csv"
    store_path = data_dir / "store.csv"
    if not test_path.exists() or not store_path.exists():
        raise FileNotFoundError("Missing test.csv or store.csv in --data-dir")

    logger.info("Loading raw test/store...")
    test = pd.read_csv(test_path, dtype={"StateHoliday": "string"})
    store = pd.read_csv(store_path)

    # Build features
    logger.info("Building test features...")
    df_test = build_test_features(test, store)

    # Load feature list from training, align columns
    feat_path = model_dir / "feature_names.json"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing {feat_path}. Run training first to generate feature_names.json")

    feature_names: List[str] = json.loads(feat_path.read_text(encoding="utf-8"))

    # Ensure all required features exist; fill missing with 0
    missing = [c for c in feature_names if c not in df_test.columns]
    if missing:
        logger.warning(f"Missing {len(missing)} features in test. Filling with 0: {missing[:10]}{'...' if len(missing)>10 else ''}")
        for c in missing:
            df_test[c] = 0

    # Drop extra columns not used in training (but keep Store for weight merge)
    # We'll build model matrix X_test using feature_names only
    X_test = df_test[feature_names].copy()

    # Load model
    model_path = model_dir / "xgb_model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}. Run training first to generate xgb_model.json")

    logger.info("Loading model...")
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    # Predict log space
    logger.info("Predicting...")
    dtest = xgb.DMatrix(X_test)
    # best_iteration may not be available after loading; safe default: all trees
    y_pred_log = booster.predict(dtest)

    # Optional: apply store weights in log space
    if args.use_store_weights:
        sw_path = model_dir / "store_weights.csv"
        sw = load_store_weights(sw_path)
        if sw is None:
            logger.warning("--use-store-weights was set but store_weights.csv not found. Skipping calibration.")
        else:
            if "Store" not in df_test.columns:
                raise KeyError("df_test has no 'Store' column required for per-store weights.")
            logger.info("Applying per-store weights (log-space scaling)...")
            tmp = pd.DataFrame({"Store": df_test["Store"].values})
            tmp = tmp.merge(sw, on="Store", how="left")
            # Stores not found -> weight=1.0
            w = tmp["weight"].fillna(1.0).values.astype(float)
            y_pred_log = y_pred_log * w

    # Convert back to Sales
    sales_pred = np.expm1(y_pred_log)
    # Kaggle expects non-negative
    sales_pred = np.clip(sales_pred, 0, None)

    # Submission Id column in test is "Id"
    if "Id" not in test.columns:
        raise KeyError("test.csv must contain 'Id' column for Kaggle submission.")
    submission = pd.DataFrame({"Id": test["Id"].values, "Sales": sales_pred})

    logger.info(f"Saving submission to: {out_path.resolve()}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)

    logger.info("Done.")


if __name__ == "__main__":
    main()
