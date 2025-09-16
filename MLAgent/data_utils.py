from __future__ import annotations
from typing import List, Sequence, Dict, Optional
import re
import numpy as np
import pandas as pd
import streamlit as st

# ---------- basic IO ----------
def check_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c not in df.columns]

@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

# ---------- quantity helpers ----------
def make_qty_grid_from_range(qmin: int, qmax: int, qstep: int) -> List[int]:
    return list(range(int(qmin), int(qmax) + 1, int(qstep)))

def parse_qty_inputs(qmin_str: str, qmax_str: str, qstep_str: str, target_str: str):
    missing = [name for name, val in [
        ("min qty", qmin_str), ("max qty", qmax_str),
        ("step", qstep_str), ("compare at quantity", target_str)
    ] if not str(val).strip()]
    if missing:
        raise ValueError(f"Please provide: {', '.join(missing)}")
    try:
        qmin = int(qmin_str); qmax = int(qmax_str); qstep = int(qstep_str); target = int(target_str)
    except ValueError:
        raise ValueError("Quantity fields must be integers.")
    if qmin < 1:           raise ValueError("Min Qty must be ≥ 1.")
    if qstep <= 0:         raise ValueError("Step must be ≥ 1.")
    if qmax < qmin:        raise ValueError("Max Qty must be ≥ Min Qty.")
    qty_grid = make_qty_grid_from_range(qmin, qmax, qstep)
    return qmin, qmax, qstep, target, qty_grid

# -------- Quantity extraction from question --------
_QTY_PATTERNS = [
    r'\bq\s*=\s*(\d+)\b',
    r'\bqty\s*[:=]?\s*(\d+)\b',
    r'\bquantity\s*[:=]?\s*(\d+)\b',
    r'\bat\s*(?:a\s*)?(?:qty|quantity)?\s*(\d+)\b',
    r'\bfor\s*(\d+)\s*(?:units|pcs|pieces|piece)\b',
    r'\b(\d+)\s*(?:units|pcs|pieces|piece)\b',
]

def extract_target_qty_from_question(q: str):
    """Best-effort integer Q extractor from free text."""
    if not q:
        return None
    t = str(q).lower()
    for pat in _QTY_PATTERNS:
        m = re.search(pat, t)
        if m:
            try:
                val = int(m.group(1))
                if val > 0:
                    return val
            except Exception:
                pass
    return None

# -------- Derive a nice (qmin, qmax, step) from data & target --------
def _nice_step_for_range(qmin: int, qmax: int, approx_steps: int = 10) -> int:
    rng = max(1, int(qmax) - int(qmin))
    raw = max(1, rng // approx_steps)
    # round to 1/2/5/10 * 10^k
    base = 1
    while base * 10 < raw:
        base *= 10
    for mult in (1, 2, 5, 10):
        s = base * mult
        if s >= raw:
            return s
    return max(1, raw)

def infer_qty_defaults(base_df: pd.DataFrame, target_qty: int | None, approx_steps: int = 10):
    """
    Returns (qmin, qmax, step, target_qty_final, qty_grid) inferred from the slice.
    Ensures target is inside [qmin, qmax] and uses a 'nice' step.
    """
    if target_qty is not None and target_qty <= 0:
        target_qty = None

    qcol = base_df.get("Quantity", None)
    if qcol is None or qcol.dropna().empty:
        # Fallback: synthesize around target or a neutral default
        target = int(target_qty or 50)
        step = max(1, target // approx_steps or 1)
        qmin = max(1, target - 5 * step)
        qmax = target + 5 * step
        grid = list(range(qmin, qmax + 1, step))
        return qmin, qmax, step, target, grid

    q = qcol.dropna().astype(int).clip(lower=1)
    p10, p90 = int(np.percentile(q, 10)), int(np.percentile(q, 90))
    # start with robust range
    qmin, qmax = p10, p90
    target = int(target_qty or int(np.median(q)))
    # expand to include target comfortably
    if target < qmin:
        qmin = target
    if target > qmax:
        qmax = target
    # pad a bit if range too tight
    if qmax - qmin < 5:
        pad = max(1, (qmax - qmin) // 2 or 2)
        qmin = max(1, qmin - pad)
        qmax = qmax + pad

    step = _nice_step_for_range(qmin, qmax, approx_steps=approx_steps)
    # guarantee target is on grid: if not, that's fine—we predict for target directly anyway
    grid = list(range(int(qmin), int(qmax) + 1, int(step)))
    return int(qmin), int(qmax), int(step), int(target), grid


# ---------- filtering ----------
def _to_str_col(series: pd.Series) -> pd.Series:
    return series.astype(str)

def apply_filters(
    df: pd.DataFrame,
    sku: Optional[str] = None,
    vendors: Optional[Sequence[str]] = None,
    regions: Optional[Sequence[str]] = None,
    seasons: Optional[Sequence[str]] = None,
    carriers: Optional[Sequence[str]] = None,
    currencies: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Legacy helper: apply a *single* SKU filter plus multi-selects.
    sku=None -> no SKU filter. Empty iterables -> no filter for that dim.
    """
    out = df.copy()
    if sku is not None:
        out = out[_to_str_col(out["SKU_ID"]) == str(sku)]
    if vendors:
        out = out[_to_str_col(out["Vendor_ID"]).isin([str(v) for v in vendors])]
    if regions:
        out = out[_to_str_col(out["Region"]).isin([str(x) for x in regions])]
    if seasons:
        out = out[_to_str_col(out["Season"]).isin([str(x) for x in seasons])]
    if carriers:
        out = out[_to_str_col(out["Carrier_Type"]).isin([str(x) for x in carriers])]
    if currencies:
        out = out[_to_str_col(out["Currency"]).isin([str(x).upper() for x in currencies])]
    return out

def apply_filters_v2(
    df: pd.DataFrame,
    skus: Optional[Sequence[str]] = None,
    vendors: Optional[Sequence[str]] = None,
    regions: Optional[Sequence[str]] = None,
    seasons: Optional[Sequence[str]] = None,
    carriers: Optional[Sequence[str]] = None,
    currencies: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    New helper: **multi-SKU** filtering + other dims.
    Any arg None/empty => no filter for that dimension.
    """
    out = df.copy()
    if skus:
        out = out[_to_str_col(out["SKU_ID"]).isin([str(s) for s in skus])]
    if vendors:
        out = out[_to_str_col(out["Vendor_ID"]).isin([str(v) for v in vendors])]
    if regions:
        out = out[_to_str_col(out["Region"]).isin([str(x) for x in regions])]
    if seasons:
        out = out[_to_str_col(out["Season"]).isin([str(x) for x in seasons])]
    if carriers:
        out = out[_to_str_col(out["Carrier_Type"]).isin([str(x) for x in carriers])]
    if currencies:
        out = out[_to_str_col(out["Currency"]).isin([str(x).upper() for x in currencies])]
    return out

# ---------- canonicalization ----------
def _canon_one(token: str, prefix: str) -> str:
    """Turn '1' / 'vendor 1' / 'Vendor_1' into 'Vendor_1' (or SKU_123)."""
    if token is None:
        return None
    t = str(token).strip()
    m = re.search(r'(\d+)', t)
    if m:
        return f"{prefix}_{m.group(1)}"
    if re.match(rf"^{prefix}_[0-9]+$", t, flags=re.IGNORECASE):
        num = t.split("_", 1)[1]
        return f"{prefix}_{num}"
    return t

def canonicalize_filters(df: pd.DataFrame, filters: Dict) -> Dict:
    """
    Align LLM-extracted filters with **actual** values present in df.
      - Vendors: Vendor_<id>
      - SKUs:    SKU_<id>
      - Region/Season/Carrier/Currency: case-preserving intersection
    """
    filters = filters or {}
    vendors_all   = set(_to_str_col(df["Vendor_ID"]).unique())
    skus_all      = set(_to_str_col(df["SKU_ID"]).unique())
    regions_all   = set(_to_str_col(df["Region"]).unique())
    seasons_all   = set(_to_str_col(df["Season"]).unique())
    carriers_all  = set(_to_str_col(df["Carrier_Type"]).unique())
    currs_all     = set(_to_str_col(df["Currency"]).unique())

    def norm_and_intersect(raw, canon_fn, universe):
        if not raw: return []
        out = []
        for x in raw:
            cx = canon_fn(x)
            if cx in universe:
                out.append(cx)
        return sorted(list(set(out)))

    vendors = norm_and_intersect(filters.get("vendors"), lambda x: _canon_one(x, "Vendor"), vendors_all)
    skus    = norm_and_intersect(filters.get("skus"),    lambda x: _canon_one(x, "SKU"),    skus_all)
    regions   = norm_and_intersect(filters.get("regions"),   lambda x: str(x), regions_all)
    seasons   = norm_and_intersect(filters.get("seasons"),   lambda x: str(x), seasons_all)
    carriers  = norm_and_intersect(filters.get("carriers"),  lambda x: str(x), carriers_all)
    currencies= norm_and_intersect(filters.get("currencies"),lambda x: str(x).upper(), currs_all)

    return {
        "vendors": vendors,
        "skus": skus,
        "regions": regions,
        "seasons": seasons,
        "carriers": carriers,
        "currencies": currencies,
    }
