from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    data_dir: str = "data/raw"
    output_dir: str = "models"
    val_weeks: int = 6
    seed: int = 10

    # XGBoost params
    eta: float = 0.03
    max_depth: int = 10
    subsample: float = 0.9
    colsample_bytree: float = 0.7
    num_boost_round: int = 6000
    early_stopping_rounds: int = 100
    verbosity: int = 1


# -------------------------
# Logging
# -------------------------
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
        logger.addHandler(h)
    return logger


# -------------------------
# Metrics
# -------------------------
def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSPE in original Sales space."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)))


def make_rmspe_custom_metric():
    """
    XGBoost custom metric (XGBoost >= 2.x uses custom_metric=...).
    Here labels/preds are in log1p space, we convert back to Sales.
    """
    def _metric(preds: np.ndarray, dmat: xgb.DMatrix):
        y_log = dmat.get_label()
        y_true = np.expm1(y_log)
        y_pred = np.expm1(preds)
        return "rmspe", rmspe(y_true, y_pred)
    return _metric


# -------------------------
# Data loading
# -------------------------
def load_raw(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    store_path = data_dir / "store.csv"

    if not train_path.exists() or not test_path.exists() or not store_path.exists():
        raise FileNotFoundError(
            f"Missing files in {data_dir}. Expect train.csv, test.csv, store.csv"
        )

    # StateHoliday: keep as string to avoid weird dtype issues (bytes/object)
    train = pd.read_csv(train_path, dtype={"StateHoliday": "string"})
    test = pd.read_csv(test_path, dtype={"StateHoliday": "string"})
    store = pd.read_csv(store_path)
    return train, test, store


# -------------------------
# Feature engineering
# -------------------------
def add_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure datetime
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

    # PromoInterval: sometimes missing; treat missing/0 as no promo month
    promo = df.get("PromoInterval")
    if promo is None:
        df["IsPromoMonth"] = 0
        return df

    promo = promo.fillna("0")

    # Vectorized-ish approach (still ok)
    def _is_promo(row) -> int:
        pi = row["PromoInterval"]
        if pi == 0 or pi == "0":
            return 0
        return 1 if row["monthstr"] in str(pi) else 0

    df["IsPromoMonth"] = df.apply(_is_promo, axis=1).astype("int8")
    return df


def map_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Map StoreType/Assortment/StateHoliday: a/b/c/d + 0 -> int.
    Return df and mapping metadata for saving.
    """
    df = df.copy()
    mappings = {
        "0": 0, 0: 0, b"0": 0,
        "a": 1, b"a": 1,
        "b": 2, b"b": 2,
        "c": 3, b"c": 3,
        "d": 4, b"d": 4,
    }

    meta = {}
    for col in ["StoreType", "Assortment", "StateHoliday"]:
        if col not in df.columns:
            continue
        # Replace then cast to nullable Int64 (handles NAs safely)
        df[col] = df[col].replace(mappings).astype("Int64")
        meta[col] = {str(k): int(v) for k, v in mappings.items() if isinstance(v, int)}

    return df, meta


def build_dataset(train: pd.DataFrame, test: pd.DataFrame, store: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    # Drop Sales<=0 (your notebook did this)
    train = train.loc[train["Sales"] > 0].copy()

    # Merge store
    train = train.merge(store, on="Store", how="left")
    test = test.merge(store, on="Store", how="left")

    # Feature engineering
    train = add_date_parts(train)
    test = add_date_parts(test)

    train = add_is_promo_month(train)
    test = add_is_promo_month(test)

    train, map_meta_train = map_categoricals(train)
    test, map_meta_test = map_categoricals(test)

    meta = {
        "categorical_mappings": map_meta_train,  # should be same for test
    }
    return train, test, meta


def make_model_matrices(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Drop columns like your notebook
    df_train = train.drop(["Date", "monthstr", "PromoInterval", "Customers", "Open"], axis=1, errors="ignore")
    df_test = test.drop(["Date", "monthstr", "PromoInterval", "Id", "Open"], axis=1, errors="ignore")
    return df_train, df_test


def time_split_by_weeks(df_train: pd.DataFrame, val_weeks: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by last N weeks (time-aware). We assume df_train has Date removed already,
    so we rely on original ordering: but that's fragile.

    Better: split by actual Date before dropping Date.
    If you still have 'year/month/day', we can reconstruct a pseudo-date.
    """
    df = df_train.copy()

    # Reconstruct date for robust split
    date = pd.to_datetime(
        df[["year", "month", "day"]],
        errors="coerce"
    )
    df["_date"] = date

    max_date = df["_date"].max()
    cutoff = max_date - pd.Timedelta(days=val_weeks * 7)

    val = df.loc[df["_date"] > cutoff].drop(columns=["_date"])
    trn = df.loc[df["_date"] <= cutoff].drop(columns=["_date"])
    return trn, val


# -------------------------
# Train / Eval / Save
# -------------------------
def train_xgb(
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    X_val: pd.DataFrame,
    y_val_log: np.ndarray,
    cfg: TrainConfig,
):
    params = {
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "eta": cfg.eta,
        "max_depth": cfg.max_depth,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "seed": cfg.seed,
        "verbosity": cfg.verbosity,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train_log)
    dval = xgb.DMatrix(X_val, label=y_val_log)

    evals = [(dtrain, "train"), (dval, "validation")]
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=cfg.num_boost_round,
        evals=evals,
        early_stopping_rounds=cfg.early_stopping_rounds,
        custom_metric=make_rmspe_custom_metric(),
        verbose_eval=True,
    )
    return model


def save_artifacts(
    model: xgb.Booster,
    out_dir: Path,
    cfg: TrainConfig,
    feature_names: List[str],
    meta: Dict,
    metrics: Dict,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "xgb_model.json"
    model.save_model(str(model_path))

    (out_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    (out_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main():
    logger = setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw", help="Path to raw data folder containing train.csv/test.csv/store.csv")
    parser.add_argument("--output-dir", default="models", help="Output folder for model artifacts")
    parser.add_argument("--val-weeks", type=int, default=6, help="How many weeks for validation split (last N weeks)")
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        val_weeks=args.val_weeks,
        seed=args.seed,
    )

    np.random.seed(cfg.seed)

    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.output_dir)

    logger.info(f"Loading data from: {data_dir.resolve()}")
    train, test, store = load_raw(data_dir)

    logger.info("Building features...")
    train_fe, test_fe, meta = build_dataset(train, test, store)
    df_train, df_test = make_model_matrices(train_fe, test_fe)

    # Split
    logger.info(f"Time split: last {cfg.val_weeks} weeks as validation")
    train_part, val_part = time_split_by_weeks(df_train, cfg.val_weeks)

    # Targets in log space
    y_train_log = np.log1p(train_part["Sales"].values)
    y_val_log = np.log1p(val_part["Sales"].values)

    X_train = train_part.drop(columns=["Sales"])
    X_val = val_part.drop(columns=["Sales"])

    feature_names = list(X_train.columns)

    logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # Train
    logger.info("Training XGBoost...")
    model = train_xgb(X_train, y_train_log, X_val, y_val_log, cfg)

    # Eval in original Sales space
    yhat_val_log = model.predict(xgb.DMatrix(X_val), iteration_range=(0, model.best_iteration + 1))
    val_rmspe = rmspe(np.expm1(y_val_log), np.expm1(yhat_val_log))

    metrics = {
        "val_rmspe": val_rmspe,
        "best_iteration": int(model.best_iteration),
        "best_score": float(model.best_score) if model.best_score is not None else None,
    }
    logger.info(f"Validation RMSPE: {val_rmspe:.6f}")
    logger.info(f"Saving artifacts to: {out_dir.resolve()}")

    save_artifacts(
        model=model,
        out_dir=out_dir,
        cfg=cfg,
        feature_names=feature_names,
        meta=meta,
        metrics=metrics,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
