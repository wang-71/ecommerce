from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb


# -------------------------
# Feature engineering (MATCH src.predict.py)
# -------------------------
def add_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("Found invalid dates in column 'Date'.")

    df["year"] = df["Date"].dt.year.astype("int32")
    df["month"] = df["Date"].dt.month.astype("int8")
    df["day"] = df["Date"].dt.day.astype("int8")

    # ✅ 补齐 DayOfWeek（Rossmann: 1=Mon ... 7=Sun）
    if "DayOfWeek" not in df.columns:
        df["DayOfWeek"] = (df["Date"].dt.dayofweek + 1).astype("int8")

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


def build_test_features_like_predict(test_like: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """
    Exactly match src.predict.py:build_test_features behavior.
    test_like should have at least: Store, Date, Promo, StateHoliday, SchoolHoliday
    (Open may be present; will be dropped.)
    """
    df = test_like.copy()

    # Match predict.py reading dtype={"StateHoliday": "string"} as much as possible
    if "StateHoliday" in df.columns:
        df["StateHoliday"] = df["StateHoliday"].astype("string")

    # Merge store
    df = df.merge(store, on="Store", how="left")

    # Feature engineering
    df = add_date_parts(df)
    df = add_is_promo_month(df)
    df = map_categoricals(df)

    # Drop columns (match src.predict.py)
    df_test = df.drop(["Date", "monthstr", "PromoInterval", "Open"], axis=1, errors="ignore")
    return df_test


# -------------------------
# Bundle
# -------------------------
@dataclass
class InferenceBundle:
    booster: xgb.Booster
    feature_names: List[str]
    store_df: pd.DataFrame
    store_weights: Optional[pd.DataFrame] = None  # columns: Store, weight


def _load_store_weights(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    w = pd.read_csv(path)
    if "Store" not in w.columns or "weight" not in w.columns:
        raise ValueError("store_weights.csv must have columns: Store, weight")
    return w[["Store", "weight"]]


def load_bundle(model_dir: str = "models", data_dir: str = "data/raw") -> InferenceBundle:
    model_dir_p = Path(model_dir)
    data_dir_p = Path(data_dir)

    booster = xgb.Booster()
    booster.load_model(str(model_dir_p / "xgb_model.json"))

    feature_names = json.loads((model_dir_p / "feature_names.json").read_text(encoding="utf-8"))

    store_df = pd.read_csv(data_dir_p / "store.csv")

    # Optional, same as src.predict.py
    store_weights = _load_store_weights(model_dir_p / "store_weights.csv")

    return InferenceBundle(
        booster=booster,
        feature_names=feature_names,
        store_df=store_df,
        store_weights=store_weights,
    )


# -------------------------
# Predict (MATCH src.predict.py)
# -------------------------
def predict_sales(
    input_rows: pd.DataFrame,
    bundle: InferenceBundle,
    use_store_weights: bool = False,
) -> pd.Series:
    """
    Return predictions in original Sales space (NOT log space).
    Matches src.predict.py:
      - build features like build_test_features
      - align to feature_names (missing -> 0)
      - booster.predict(dtest) using ALL trees (no best_iteration truncation)
      - optional store weight scaling in log space
      - expm1 + clip >= 0
    """
    df_test = build_test_features_like_predict(input_rows, bundle.store_df)

    # Fill missing features with 0, then select only training features
    for c in bundle.feature_names:
        if c not in df_test.columns:
            df_test[c] = 0

    X = df_test[bundle.feature_names].copy()

    dtest = xgb.DMatrix(X)
    y_pred_log = bundle.booster.predict(dtest)  # IMPORTANT: all trees, same as src.predict.py

    # Optional store weights (same as src.predict.py)
    if use_store_weights:
        if bundle.store_weights is not None:
            if "Store" not in df_test.columns:
                raise KeyError("df_test has no 'Store' column required for per-store weights.")
            tmp = pd.DataFrame({"Store": df_test["Store"].values})
            tmp = tmp.merge(bundle.store_weights, on="Store", how="left")
            w = tmp["weight"].fillna(1.0).values.astype(float)
            y_pred_log = y_pred_log * w

    sales_pred = np.expm1(y_pred_log)
    sales_pred = np.clip(sales_pred, 0, None)

    return pd.Series(sales_pred)