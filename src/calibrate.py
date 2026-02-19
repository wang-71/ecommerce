# src/calibrate.py
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("calibrate")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
        logger.addHandler(h)
    return logger


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)))


# --------- Feature engineering (复制你 train/predict 的逻辑，必须一致) ---------
def add_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
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


def build_train_features(train: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    train = train.loc[train["Sales"] > 0].copy()
    train = train.merge(store, on="Store", how="left")
    train = add_date_parts(train)
    train = add_is_promo_month(train)
    train = map_categoricals(train)
    return train


def make_model_matrix(train_fe: pd.DataFrame) -> pd.DataFrame:
    # 保留 Store/year/month/day 给 split & calibration 用；训练时 drop 的列这里也 drop
    df_train = train_fe.drop(["Date", "monthstr", "PromoInterval", "Customers", "Open"], axis=1, errors="ignore")
    return df_train


def time_split_by_weeks(df_train: pd.DataFrame, val_weeks: int):
    df = df_train.copy()
    date = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]), errors="coerce")
    df["_date"] = date
    cutoff = df["_date"].max() - pd.Timedelta(days=val_weeks * 7)
    val = df.loc[df["_date"] > cutoff].drop(columns=["_date"])
    trn = df.loc[df["_date"] <= cutoff].drop(columns=["_date"])
    return trn, val


# --------- calibration ---------
def calibrate_per_store_log_space(
    val_df: pd.DataFrame,
    y_val_log: np.ndarray,
    yhat_val_log: np.ndarray,
    w_start: float,
    w_end: float,
    step: float,
) -> pd.DataFrame:
    if "Store" not in val_df.columns:
        raise KeyError("Validation dataframe must contain 'Store' column.")

    weights = np.arange(w_start, w_end + 1e-12, step)
    store_ids = val_df["Store"].values

    y_true = np.expm1(y_val_log)  # Sales space

    rows = []
    for store in np.unique(store_ids):
        idx = store_ids == store
        yt = y_true[idx]
        yp_log = yhat_val_log[idx]

        best_w = 1.0
        best_err = float("inf")
        for w in weights:
            err = rmspe(yt, np.expm1(yp_log * w))
            if err < best_err:
                best_err = err
                best_w = float(w)

        rows.append((int(store), best_w, best_err))

    return pd.DataFrame(rows, columns=["Store", "weight", "val_rmspe"])


def main():
    logger = setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--val-weeks", type=int, default=6)

    parser.add_argument("--w-start", type=float, default=0.98)
    parser.add_argument("--w-end", type=float, default=1.02)
    parser.add_argument("--step", type=float, default=0.001)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    # load model + features
    model_path = model_dir / "xgb_model.json"
    feat_path = model_dir / "feature_names.json"
    if not model_path.exists() or not feat_path.exists():
        raise FileNotFoundError("Need xgb_model.json and feature_names.json in --model-dir")

    feature_names: List[str] = json.loads(feat_path.read_text(encoding="utf-8"))
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    # load data
    train = pd.read_csv(data_dir / "train.csv", dtype={"StateHoliday": "string"})
    store = pd.read_csv(data_dir / "store.csv")

    # build same features & split
    train_fe = build_train_features(train, store)
    df_train = make_model_matrix(train_fe)

    trn, val = time_split_by_weeks(df_train, args.val_weeks)

    y_val_log = np.log1p(val["Sales"].values)
    X_val = val.drop(columns=["Sales"])

    # align features exactly
    missing = [c for c in feature_names if c not in X_val.columns]
    for c in missing:
        X_val[c] = 0
    X_val = X_val[feature_names]

    # predict log space
    yhat_val_log = booster.predict(xgb.DMatrix(X_val))

    # calibrate
    logger.info("Calibrating per-store weights...")
    store_weights = calibrate_per_store_log_space(
        val_df=val,  # val still has Store column
        y_val_log=y_val_log,
        yhat_val_log=yhat_val_log,
        w_start=args.w_start,
        w_end=args.w_end,
        step=args.step,
    )

    out_path = model_dir / "store_weights.csv"
    store_weights.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path.resolve()}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
