import numpy as np
import pandas as pd
from joblib import dump as joblib_dump, load as joblib_load  # noqa: F401  (kept for compatibility)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from .ml_utils import flaml_regression_fit  # relative import if packaged; else use from ml_utils import ...
# --- Add to MLAgent/price_anomaly.py ---
from .config import PRICE_PATH, MODEL_DIR
import os

def build_price_preprocessor():
    numeric_features = [
        "Quantity","Weight_kg","Volume_m3","Distance_km",
        "Lead_Time_Days","Contract_Length_Months","Fuel_Price_USD"
    ]
    categorical_features = ["SKU_ID","Vendor_ID","Region","Season","Carrier_Type","Currency"]
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )
    return pre, numeric_features, categorical_features

def fit_price_model(df: pd.DataFrame, time_budget_s: int):
    pre, num_cols, cat_cols = build_price_preprocessor()
    feats = num_cols + cat_cols
    X = df[feats]
    y = df["Unit_Price_USD"].astype(float).values
    X_proc = pre.fit_transform(X)
    automl = flaml_regression_fit(
        X_proc, y, time_budget_s=time_budget_s, metric="mae", log_file="flaml_price.log"
    )
    return {"automl": automl, "pre": pre, "num_cols": num_cols, "cat_cols": cat_cols}

def score_price_model(bundle, df: pd.DataFrame):
    feats = bundle["num_cols"] + bundle["cat_cols"]
    X_all = bundle["pre"].transform(df[feats])
    y_all = df["Unit_Price_USD"].astype(float).values
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    y_pred_te = bundle["automl"].predict(X_te)
    return (
        mean_absolute_error(y_te, y_pred_te),
        np.sqrt(mean_squared_error(y_te, y_pred_te)),
        r2_score(y_te, y_pred_te),
    )

def detect_anomalies(df: pd.DataFrame, bundle, percentile=95):
    feats = bundle["num_cols"] + bundle["cat_cols"]
    X_all = bundle["pre"].transform(df[feats])
    preds = bundle["automl"].predict(X_all)
    residuals = np.abs(df["Unit_Price_USD"].astype(float).values - preds)
    thr = np.percentile(residuals, percentile)
    out = df.copy()
    out["Predicted_Unit_Price"] = preds
    out["Residual_Abs"] = residuals
    out["Anomaly_Flag"] = (out["Residual_Abs"] > thr).astype(int)
    return out, thr

def save_price_model(bundle, path: str = PRICE_PATH):
    """Persist the trained price anomaly bundle (pre + automl + metadata)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib_dump(bundle, path)

def load_price_model(path: str = PRICE_PATH):
    """Load a previously saved price anomaly bundle; return None if missing."""
    try:
        return joblib_load(path)
    except Exception:
        return None

def get_price_model(use_pretrained: bool, df=None, time_budget_s: int = 60, save_new: bool = False):
    """
    - If use_pretrained=True: try to load; if not found and df is provided, train and (optionally) save.
    - If use_pretrained=False: train on df and (optionally) save.
    Returns the bundle or None if nothing available.
    """
    if use_pretrained:
        bundle = load_price_model()
        if bundle is not None:
            return bundle
        # Fallback: train if CSV is available
        if df is None:
            return None
        bundle = fit_price_model(df, time_budget_s=time_budget_s)
        if save_new:
            save_price_model(bundle)
        return bundle
    else:
        if df is None:
            return None
        bundle = fit_price_model(df, time_budget_s=time_budget_s)
        if save_new:
            save_price_model(bundle)
        return bundle
