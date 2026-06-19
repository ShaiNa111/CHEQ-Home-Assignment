#!/usr/bin/env python3
"""
setup_data.py
=============
Downloads the Customer Churn (Telco) dataset and saves it to data/churn.csv.
Run this once before starting the MCP server.

Usage
-----
    python setup_data.py

Source: Hugging Face `datasets` library — aai510-group1/telco-customer-churn
Requires the `datasets` package: pip install datasets
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
DATA_PATH = os.path.join(DATA_DIR, "churn.csv")

HF_DATASET_ID = "aai510-group1/telco-customer-churn"


def try_huggingface() -> bool:
    try:
        from datasets import load_dataset  # type: ignore
        print(f"⬇  Downloading from Hugging Face ({HF_DATASET_ID}) via `datasets` lib…")
        ds = load_dataset(HF_DATASET_ID, split="train")
        df = ds.to_pandas()
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"✓  Saved {len(df):,} rows → {DATA_PATH}")
        print(f"   Columns: {', '.join(df.columns.tolist())}")
        return True
    except Exception as exc:
        print(f"   HuggingFace `datasets` lib attempt failed: {exc}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        import pandas as pd
        df = pd.read_csv(DATA_PATH)
        print(f"Dataset already exists: {DATA_PATH}  ({len(df):,} rows)")
        print("Delete data/churn.csv and re-run to refresh.")
        sys.exit(0)

    print("=== Customer Churn Dataset Setup ===\n")

    if try_huggingface():
        sys.exit(0)

    print("\n❌  Setup failed. Make sure the `datasets` package is installed:")
    print("    pip install datasets")
    print(f"\n    Or download the dataset manually from:")
    print(f"    https://huggingface.co/datasets/{HF_DATASET_ID}")
    print(f"    Save it as:  {DATA_PATH}")
    sys.exit(1)
