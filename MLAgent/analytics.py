import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from .ml_utils import flaml_regression_fit  # relative import if packaged; else use from ml_utils import ...

# ---------------- Vendor/SKU feature tables & clustering ----------------
def vendor_features(df: pd.DataFrame):
    agg = {
        "Unit_Price_USD": ["mean","std"],
        "Quantity": "sum",
        "SKU_ID": pd.Series.nunique,
        "Region": pd.Series.nunique
    }
    g = df.groupby("Vendor_ID").agg(agg)
    g.columns = ["Avg_Price","Price_Variability","Total_Volume","Num_SKUs","Regions_Served"]
    return g.reset_index()

def sku_features(df: pd.DataFrame):
    agg = {
        "Unit_Price_USD": ["mean","std"],
        "Quantity": "sum",
        "Vendor_ID": pd.Series.nunique
    }
    g = df.groupby("SKU_ID").agg(agg)
    g.columns = ["Avg_Price","Price_Variability","Total_Quantity","Num_Vendors"]
    return g.reset_index()

def cluster_table_train(table: pd.DataFrame, n_clusters=3, feature_cols=None, random_state=42):
    feature_cols = feature_cols or [c for c in table.columns if c not in ("Vendor_ID","SKU_ID")]
    n = len(table)
    k = max(1, min(n_clusters, n))
    scaler = StandardScaler()
    Z = scaler.fit_transform(table[feature_cols].fillna(0.0))
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(Z)
    out = table.copy()
    out["Cluster"] = labels
    return out

# ---------------- Discount curves (quantity-only & enriched) -----------
def discount_curve_training(df_ctx: pd.DataFrame, time_budget_s=45):
    feats = ["Quantity"]
    target = "Unit_Price_USD"
    pre = ColumnTransformer(transformers=[("num", StandardScaler(), ["Quantity"])])
    X_proc = pre.fit_transform(df_ctx[feats])
    y = df_ctx[target].astype(float).values
    automl = flaml_regression_fit(
        X_proc, y, time_budget_s=time_budget_s, metric="mae", log_file="flaml_discount.log"
    )
    return automl, pre, feats

def discount_curve_training_enriched(df_ctx: pd.DataFrame, time_budget_s=45):
    """Train using Quantity + (Region, Season, Carrier_Type, Currency).
       Excludes SKU_ID and Vendor_ID as predictors; those are grouping keys only.
    """
    num_feats = ["Quantity"]
    cat_feats = ["Region", "Season", "Carrier_Type", "Currency"]

    df_local = df_ctx.copy()
    for c in cat_feats:
        if c not in df_local.columns:
            df_local[c] = "UNK"
    df_local[cat_feats] = df_local[cat_feats].fillna("UNK")

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_feats),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
        ]
    )
    feats = num_feats + cat_feats
    X_proc = pre.fit_transform(df_local[feats])
    y = df_local["Unit_Price_USD"].astype(float).values

    automl = flaml_regression_fit(
        X_proc, y, time_budget_s=time_budget_s, metric="mae",
        log_file="flaml_discount_enriched.log"
    )
    return automl, pre, feats, cat_feats

def majority_context(df_ctx: pd.DataFrame):
    """Pick most common values within a SKU+Vendor slice for scoring without exposing UI filters."""
    ctx = {}
    for c in ["Region", "Season", "Carrier_Type", "Currency"]:
        if c in df_ctx.columns and df_ctx[c].notna().any():
            m = df_ctx[c].mode(dropna=True)
            ctx[c] = str(m.iloc[0]) if len(m) else "UNK"
        else:
            ctx[c] = "UNK"
    return ctx

def discount_curve_grid(automl, pre, feats, qty_grid, ctx: dict | None = None, cat_feats: list | None = None):
    """Build grid predictions; if ctx + cat_feats provided, include categorical context for enriched model."""
    if ctx and cat_feats:
        grid_df = pd.DataFrame({"Quantity": qty_grid})
        for c in cat_feats:
            grid_df[c] = ctx.get(c, "UNK")
    else:
        grid_df = pd.DataFrame({"Quantity": qty_grid})

    Xg = pre.transform(grid_df[feats])
    preds = automl.predict(Xg)
    grid_df["Predicted_Unit_Price"] = preds
    return grid_df

def predict_for_vendor(model_entry, qty):
    df_one = pd.DataFrame({"Quantity": [qty]})
    Xg = model_entry["pre"].transform(df_one[model_entry["feats"]])
    return float(model_entry["automl"].predict(Xg)[0])

def predict_for_vendor_enriched(model_entry, qty: int, ctx: dict):
    df_one = pd.DataFrame({"Quantity": [qty]})
    for c in model_entry["cat_feats"]:
        df_one[c] = [ctx.get(c, "UNK")]
    Xg = model_entry["pre"].transform(df_one[model_entry["feats"]])
    return float(model_entry["automl"].predict(Xg)[0])

# ---------------------- Plots (unchanged) ----------------------
def fig_residuals_scatter(df_anom: pd.DataFrame, threshold: float):
    plt.figure(figsize=(8,5))
    plt.scatter(df_anom["Predicted_Unit_Price"], df_anom["Residual_Abs"], s=12, alpha=0.5)
    plt.axhline(threshold, linestyle="--")
    plt.xlabel("Predicted Unit Price (USD)")
    plt.ylabel("|Residual|")
    plt.title("Residuals vs Predicted (dashed = anomaly threshold)")
    st.pyplot(plt.gcf()); plt.close()

def fig_vendor_clusters(vendors_df: pd.DataFrame):
    plt.figure(figsize=(7,5))
    for c in sorted(vendors_df["Cluster"].unique()):
        sub = vendors_df[vendors_df["Cluster"] == c]
        plt.scatter(sub["Avg_Price"], sub["Price_Variability"], s=70, alpha=0.8, label=f"Cluster {c}")
    plt.xlabel("Avg Price"); plt.ylabel("Price Variability (std)")
    plt.title("Vendor Segmentation"); plt.legend()
    st.pyplot(plt.gcf()); plt.close()

def fig_sku_clusters(skus_df: pd.DataFrame):
    plt.figure(figsize=(7,5))
    for c in sorted(skus_df["Cluster"].unique()):
        sub = skus_df[skus_df["Cluster"] == c]
        plt.scatter(sub["Avg_Price"], sub["Price_Variability"], s=70, alpha=0.8, label=f"Cluster {c}")
    plt.xlabel("Avg Price"); plt.ylabel("Price Variability (std)")
    plt.title("SKU Segmentation"); plt.legend()
    st.pyplot(plt.gcf()); plt.close()

def fig_discount_multi(curves_by_vendor: dict, title="Vendor Discount Curves"):
    plt.figure(figsize=(9,6))
    for vendor, entry in curves_by_vendor.items():
        curve_df = entry["curve"]
        plt.plot(curve_df["Quantity"], curve_df["Predicted_Unit_Price"], marker="o", label=str(vendor))
    plt.xlabel("Quantity"); plt.ylabel("Predicted Unit Price (USD)")
    plt.title(title); plt.grid(True); plt.legend(title="Vendor", fontsize=9)
    st.pyplot(plt.gcf()); plt.close()
