"""
check_datasets.py — checks every CKD-named CSV in data/ and flags
whether it looks like real clinical data or likely synthetic/fake data,
based on whether numeric columns look like real biological measurements
(uneven, skewed) vs computer-generated ones (suspiciously uniform).
"""
import pandas as pd
import numpy as np
import os

FILES_TO_CHECK = [
    "bangladesh_ckd.csv",
    "ckd-dataset-v2.csv",
    "kaggle_ckd.csv",
    "kidney_disease_dataset.csv",
    "kidney_disease.csv",
]

for fname in FILES_TO_CHECK:
    path = os.path.join("data", fname)
    if not os.path.exists(path):
        print(f"\n{fname}: FILE NOT FOUND, skipping.")
        continue

    print(f"\n{'='*60}\n{fname}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"   Could not read as CSV: {e}")
        continue

    print(f"   Rows: {len(df)}, Columns: {df.shape[1]}")
    print(f"   Column names: {list(df.columns)[:10]}{'...' if df.shape[1] > 10 else ''}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]
    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) < 10:
            continue
        mean, std, mn, mx = vals.mean(), vals.std(), vals.min(), vals.max()
        expected_uniform_std = (mx - mn) / np.sqrt(12)
        ratio = std / expected_uniform_std if expected_uniform_std > 0 else 0
        flag = "⚠️ LOOKS SYNTHETIC (too uniform)" if 0.85 < ratio < 1.15 else "OK — looks like real varied data"
        print(f"   {col}: mean={mean:.2f} std={std:.2f} min={mn:.2f} max={mx:.2f}  → {flag}")

print(f"\n{'='*60}\nDone. Files marked OK are safer to consider using.")