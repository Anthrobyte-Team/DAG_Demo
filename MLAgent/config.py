import os

# -------- Excel engine fallback (xlsxwriter -> openpyxl) --------
try:
    import xlsxwriter  # noqa: F401
    EXCEL_ENGINE = "xlsxwriter"
except Exception:
    try:
        import openpyxl  # noqa: F401
        EXCEL_ENGINE = "openpyxl"
    except Exception:
        EXCEL_ENGINE = None

# ---------------------------
# Config & paths
# ---------------------------
REQUIRED_COLUMNS = [
    "SKU_ID","Vendor_ID","Quantity","Region","Season","Carrier_Type",
    "Weight_kg","Volume_m3","Distance_km","Lead_Time_Days",
    "Contract_Length_Months","Fuel_Price_USD","Currency","Unit_Price_USD"
]

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
PRICE_PATH = os.path.join(MODEL_DIR, "price_regression.pkl")
