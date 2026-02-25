"""
InSDN Dataset Preprocessing Script
=====================================
Consolidates the three InSDN CSV files (Normal_data.csv, OVS.csv,
metasploitable-2.csv) into a single cleaned CSV ready for TCN training.

Steps performed:
  1. Load and concatenate all three CSVs
  2. Strip whitespace from column names and string values
  3. Drop identifier / meta columns (Flow ID, IPs, ports, Timestamp)
  4. Encode binary label: Normal=0, Attack=1
  5. Replace Inf / -Inf with NaN, then drop those rows
  6. Drop remaining NaN rows
  7. Remove zero-variance (constant) columns
  8. Remove near-constant columns (>99.9 % same value)
  9. Remove highly correlated feature pairs (Pearson |r| > 0.98)
 10. Save to insdn_consolidated.csv

Usage:
    pip install pandas numpy scikit-learn
    python preprocess.py
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
DATASET_DIR = Path("InSDN_DatasetCSV")
CSV_FILES   = ["Normal_data.csv", "OVS.csv", "metasploitable-2.csv"]
OUTPUT_FILE = Path("insdn_consolidated.csv")

# Columns that carry no predictive value (identifiers / metadata)
DROP_COLS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp"]

# ── 1. Load & concatenate ──────────────────────────────────────────────────
print("=" * 60)
print("Step 1 – Loading CSV files …")
frames = []
for fname in CSV_FILES:
    path = DATASET_DIR / fname
    df   = pd.read_csv(path, low_memory=False)
    # strip whitespace from all column names
    df.columns = df.columns.str.strip()
    print(f"  {fname}: {len(df):,} rows  |  columns: {df.shape[1]}")
    frames.append(df)

df = pd.concat(frames, ignore_index=True)
print(f"\n  Combined shape: {df.shape}")

# ── 2. Strip string values ─────────────────────────────────────────────────
print("\nStep 2 – Stripping whitespace from string values …")
df["Label"] = df["Label"].astype(str).str.strip()

# ── 3. Drop identifier columns ─────────────────────────────────────────────
print("\nStep 3 – Dropping identifier/meta columns …")
existing_drop = [c for c in DROP_COLS if c in df.columns]
df.drop(columns=existing_drop, inplace=True)
print(f"  Dropped: {existing_drop}")
print(f"  Remaining columns: {df.shape[1]}")

# ── 4. Binary label encoding ───────────────────────────────────────────────
print("\nStep 4 – Encoding labels (Normal=0, Attack=1) …")
print("  Raw label distribution:")
print(df["Label"].value_counts().to_string())

df["Label"] = df["Label"].apply(lambda x: 0 if x == "Normal" else 1)
print("\n  Binary label counts:")
print(df["Label"].value_counts().to_string())

# ── 5. Replace Inf values with NaN ────────────────────────────────────────
print("\nStep 5 – Replacing Inf/-Inf with NaN …")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
inf_count = np.isinf(df[numeric_cols].values).sum()
df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
print(f"  Inf values replaced: {inf_count:,}")

# ── 6. Drop NaN rows ──────────────────────────────────────────────────────
print("\nStep 6 – Dropping rows with NaN values …")
before = len(df)
df.dropna(inplace=True)
after  = len(df)
print(f"  Dropped {before - after:,} rows  →  {after:,} rows remain")

# ── 7. Drop zero-variance columns ─────────────────────────────────────────
print("\nStep 7 – Removing zero-variance columns …")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "Label"]
std_vals     = df[numeric_cols].std()
zero_var     = std_vals[std_vals == 0].index.tolist()
df.drop(columns=zero_var, inplace=True)
print(f"  Removed {len(zero_var)} zero-variance column(s): {zero_var}")

# ── 8. Drop near-constant columns (>99.9 % same value) ────────────────────
print("\nStep 8 – Removing near-constant columns (>99.9 % same value) …")
feature_cols   = [c for c in df.columns if c != "Label"]
near_const     = []
for col in feature_cols:
    top_freq = df[col].value_counts(normalize=True).iloc[0]
    if top_freq > 0.999:
        near_const.append(col)
df.drop(columns=near_const, inplace=True)
print(f"  Removed {len(near_const)} near-constant column(s): {near_const}")

# ── 9. Remove highly correlated features (|r| > 0.98) ─────────────────────
print("\nStep 9 – Removing highly correlated features (|r| > 0.98) …")
feature_cols = [c for c in df.columns if c != "Label"]
corr_matrix  = df[feature_cols].corr().abs()
upper        = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_corr = [col for col in upper.columns if any(upper[col] > 0.98)]
df.drop(columns=to_drop_corr, inplace=True)
print(f"  Removed {len(to_drop_corr)} highly correlated column(s)")
print(f"  Remaining feature columns: {df.shape[1] - 1}")

# ── 10. Save output ────────────────────────────────────────────────────────
print(f"\nStep 10 – Saving consolidated dataset …")
df.to_csv(OUTPUT_FILE, index=False)
print(f"  Saved: {OUTPUT_FILE}")
print(f"  Final shape: {df.shape}")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
feature_cols = [c for c in df.columns if c != "Label"]
print(f"  Total samples  : {len(df):,}")
print(f"  Feature count  : {len(feature_cols)}")
print(f"  Benign (0)     : {(df['Label']==0).sum():,}")
print(f"  Attack (1)     : {(df['Label']==1).sum():,}")
print(f"  Imbalance ratio: 1 : {(df['Label']==1).sum() / (df['Label']==0).sum():.2f}")
print(f"\n  Feature list:")
for i, c in enumerate(feature_cols, 1):
    print(f"    {i:2d}. {c}")
print("=" * 60)
print(f"\n  Upload  '{OUTPUT_FILE}'  to Google Colab and run the TCN training script.")
