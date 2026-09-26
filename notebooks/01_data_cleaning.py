"""
Phase 1 - Data Cleaning  (v2 - fixed for pandas 2.x / Windows cp1252)
Supply Chain Delay / Risk Analysis
"""
import sys, io
# Force UTF-8 output so arrow characters don't crash on Windows cp1252 terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import os, json, warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW    = os.path.join(BASE, "data", "DataCoSupplyChainDataset.csv")
OUTPUT = os.path.join(BASE, "outputs", "cleaned_data.csv")
os.makedirs(os.path.join(BASE, "outputs"), exist_ok=True)

# ── 1. Load ────────────────────────────────────────────────────────────────────
print("Loading dataset ...")
try:
    df = pd.read_csv(RAW, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(RAW, encoding="latin-1")

print(f"Raw shape : {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
missing_init = df.isnull().sum()
missing_init = missing_init[missing_init > 0]
print(f"\nMissing values:\n{missing_init}")

# ── 2. Standardise column names ────────────────────────────────────────────────
df.columns = (df.columns
              .str.strip()
              .str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
              .str.replace(r"_+", "_", regex=True)
              .str.strip("_"))

print(f"\nNormalised columns:\n{df.columns.tolist()}")

# ── 3. Datetime conversion (pandas 2.x compatible) ────────────────────────────
date_candidates = [c for c in df.columns if "date" in c.lower()]
print(f"\nDate-like columns found: {date_candidates}")

for col in date_candidates:
    try:
        # pandas 2.x: infer_datetime_format removed; just use errors='coerce'
        df[col] = pd.to_datetime(df[col], errors="coerce")
        nat_count = df[col].isna().sum()
        print(f"  Converted '{col}' -> datetime  (NaT count: {nat_count})")
    except Exception as e:
        print(f"  Could not convert '{col}': {e}")

# ── 4. Identify key columns ────────────────────────────────────────────────────
def find_col(df, *keywords):
    for kw in keywords:
        for c in df.columns:
            if kw.lower() in c.lower():
                return c
    return None

col_late_risk      = find_col(df, "Late_delivery_risk")
col_delivery_stat  = find_col(df, "Delivery_Status")
col_ship_mode      = find_col(df, "Shipping_Mode")
col_order_date     = find_col(df, "order_date")
col_ship_date_str  = find_col(df, "shipping_date")   # the date string col (not days)
col_days_sched     = find_col(df, "Days_for_shipment_scheduled")
col_days_real      = find_col(df, "Days_for_shipping_real")
col_region         = find_col(df, "Order_Region")
col_market         = find_col(df, "Market")
col_category       = find_col(df, "Category_Name")
col_qty            = find_col(df, "Order_Item_Quantity")
col_discount       = find_col(df, "Order_Item_Discount")

print("\nKey column mapping:")
key_map_display = {
    "late_risk":        col_late_risk,
    "delivery_status":  col_delivery_stat,
    "shipping_mode":    col_ship_mode,
    "order_date":       col_order_date,
    "shipping_date":    col_ship_date_str,
    "days_scheduled":   col_days_sched,
    "days_real":        col_days_real,
    "region":           col_region,
    "market":           col_market,
    "category":         col_category,
    "quantity":         col_qty,
    "discount":         col_discount,
}
for k, v in key_map_display.items():
    print(f"  {k:22s} -> {v}")

# ── 5. Check consistency: Delivery Status vs Late_delivery_risk ────────────────
if col_delivery_stat and col_late_risk:
    print(f"\nDelivery Status value counts:\n{df[col_delivery_stat].value_counts()}")
    print(f"\nLate delivery risk value counts:\n{df[col_late_risk].value_counts()}")
    cross = pd.crosstab(df[col_delivery_stat], df[col_late_risk])
    print(f"\nCross-tab (Delivery Status vs Late_delivery_risk):\n{cross}")
    print("\nNote: Delivery_Status='Late delivery' perfectly maps to Late_delivery_risk=1.")
    print("The target column is internally consistent - no leakage risk in EDA/stats,")
    print("but Delivery_Status MUST be excluded from the predictive model features.")

# ── 6. Missing value handling ──────────────────────────────────────────────────
miss = df.isnull().sum()
miss = miss[miss > 0].sort_values(ascending=False)
print(f"\nMissing value summary (before cleaning):\n{miss}")

# Drop rows where target is missing
if col_late_risk and df[col_late_risk].isnull().sum() > 0:
    before = len(df)
    df = df.dropna(subset=[col_late_risk])
    print(f"Dropped {before - len(df)} rows with missing Late_delivery_risk.")

# For numeric cols: median impute if <5% missing, else drop column
cols_to_drop = []
impute_map = {}
for col in df.select_dtypes(include=[np.number]).columns:
    pct_miss = df[col].isnull().mean()
    if 0 < pct_miss < 0.05:
        median_val = df[col].median()
        impute_map[col] = median_val
        print(f"Imputed '{col}' ({pct_miss:.1%} missing) with median={median_val:.2f}")
    elif pct_miss >= 0.05:
        cols_to_drop.append(col)
        print(f"Dropping column '{col}' ({pct_miss:.1%} missing > 5% threshold).")

df = df.assign(**{col: df[col].fillna(val) for col, val in impute_map.items()})
df = df.drop(columns=cols_to_drop)

# For string/object cols: mode impute if <5%, else drop
cols_to_drop_str = []
impute_map_str = {}
for col in df.select_dtypes(include=["object", "str"]).columns:
    pct_miss = df[col].isnull().mean()
    if 0 < pct_miss < 0.05:
        mode_val = df[col].mode()[0]
        impute_map_str[col] = mode_val
        print(f"Imputed '{col}' ({pct_miss:.1%} missing) with mode='{mode_val}'")
    elif pct_miss >= 0.05:
        cols_to_drop_str.append(col)
        print(f"Dropping column '{col}' ({pct_miss:.1%} missing > 5% threshold).")

df = df.assign(**{col: df[col].fillna(val) for col, val in impute_map_str.items()})
df = df.drop(columns=cols_to_drop_str)

# ── 7. Standardise categorical text fields ────────────────────────────────────
cat_cols = [col_ship_mode, col_region, col_market, col_category, col_delivery_stat]
cat_cols = [c for c in cat_cols if c and c in df.columns]
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip().str.title()
    print(f"  Standardised '{col}' -> unique: {sorted(df[col].unique())}")

# ── 8. Engineer: delay_days = real - scheduled ────────────────────────────────
if (col_days_sched and col_days_real and
        col_days_sched in df.columns and col_days_real in df.columns):
    df["delay_days"] = df[col_days_real] - df[col_days_sched]
    print(f"\nCreated 'delay_days' = real - scheduled shipping days.")
    print(df["delay_days"].describe())
else:
    print("Could not create delay_days - scheduled/real day columns not found.")

# ── 9. Final inspection & save ─────────────────────────────────────────────────
print(f"\nCleaned shape : {df.shape}")
remaining_miss = df.isnull().sum()
remaining_miss = remaining_miss[remaining_miss > 0]
if len(remaining_miss):
    print(f"Remaining missing:\n{remaining_miss}")
else:
    print("No missing values remain.")

df.to_csv(OUTPUT, index=False, encoding="utf-8")
print(f"\nSaved cleaned data -> {OUTPUT}")

# ── 10. Save column map for downstream scripts ─────────────────────────────────
col_map_path = os.path.join(BASE, "outputs", "col_map.json")
col_map = {
    "late_risk":        col_late_risk,
    "delivery_status":  col_delivery_stat,
    "shipping_mode":    col_ship_mode,
    "order_date":       col_order_date,
    "shipping_date":    col_ship_date_str,
    "days_scheduled":   col_days_sched,
    "days_real":        col_days_real,
    "delay_days":       "delay_days" if "delay_days" in df.columns else None,
    "region":           col_region,
    "market":           col_market,
    "category":         col_category,
    "quantity":         col_qty,
    "discount":         col_discount,
}
# Remove None values for cleanliness
col_map = {k: v for k, v in col_map.items() if v is not None}

with open(col_map_path, "w") as f:
    json.dump(col_map, f, indent=2)
print(f"Column map saved -> {col_map_path}")
print("\nPHASE 1 COMPLETE.")
