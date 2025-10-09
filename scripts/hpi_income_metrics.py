#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import re
import pandas as pd
import numpy as np
from typing import Optional, List

# Optional geopandas support
try:
    import geopandas as gpd
except ImportError:
    gpd = None

# ===============================
# CONFIG
# ===============================

def _safe_root() -> str:
    try:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    except NameError:
        return os.getcwd()

ROOT = _safe_root()
PATHS = {
    "input_parquet": os.path.join(ROOT, "data", "processed", "income_hpi_home_rent_at_county.parquet"),
    "processed_dir": os.path.join(ROOT, "data", "processed"),
    "quality_dir": os.path.join(ROOT, "data", "quality"),
}
TARGET_CRS = "EPSG:4326"


# ===============================
# UTILITIES
# ===============================

def print_step(msg: str):
    """Consistent console logging."""
    print(f"▶ {msg}")

def ensure_dirs():
    for k in ["processed_dir", "quality_dir"]:
        os.makedirs(PATHS[k], exist_ok=True)

def check_columns(df: pd.DataFrame, required: List[str], name: str):
    """Raise a clear error if required columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {name}: {missing}")


# ===============================
# DATA CHECKS & CLEANING
# ===============================

def data_quality_report(df: pd.DataFrame) -> dict:
    """Return and print a simple data quality summary."""
    report = {
        "rows": len(df),
        "cols": len(df.columns),
        "duplicates": int(df.duplicated().sum()),
        "missing_by_col": df.isna().sum().to_dict(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }
    print_step(f"✅ Data Quality Report: {report['rows']} rows, {report['cols']} columns")
    return report


# ===============================
# METRIC CALCULATIONS
# ===============================

def add_monthly_rent_income(df: pd.DataFrame) -> pd.DataFrame:
    """Convert annual renter income to monthly income."""
    check_columns(df, ["median_renters_income"], "monthly_rent_income")
    df = df.copy()
    df["median_monthly_rent_income"] = df["median_household_income"] / 12
    return df

def calculate_hai(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Housing Affordability Index."""
    check_columns(df, ["median_home_value", "median_household_income"], "HAI")
    df = df.copy()
    df["HAI"] = df["median_home_value"] / df["median_household_income"]
    df["HAI"].replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def calculate_rai(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Rent Affordability Index."""
    check_columns(df, ["median_gross_rent", "median_household_income"], "RAI")
    df = df.copy()
    df["RAI"] = df["median_monthly_rent_income"] / df["median_gross_rent"]
    df["RAI"].replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def index_variable(df: pd.DataFrame, variable: str, group_col: Optional[str] = "county_fips_full") -> pd.DataFrame:
    """Index a numeric variable by its first available year within each group."""
    if variable not in df.columns:
        raise KeyError(f"Missing column {variable} for indexing.")
    if df[variable].dtype not in [np.float64, np.int64, np.float32, np.int32]:
        raise ValueError(f"{variable} must be numeric to index.")
    df = df.copy()
    df["base_value"] = df.sort_values("year").groupby(group_col)[variable].transform("first")
    df[f"{variable}_indexed"] = df[variable] / df["base_value"]
    return df


# ===============================
# PIPELINE LOGIC
# ===============================

def process_affordability_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Run all key transformations."""
    df = add_monthly_rent_income(df)
    df = calculate_rai(df)
    df = calculate_hai(df)
    return df


# ===============================
# SAVE FUNCTIONS
# ===============================

def save_outputs(df: pd.DataFrame, report: dict):
    """Save processed data and quality report."""
    ensure_dirs()
    out_csv = os.path.join(PATHS["processed_dir"], "hpi_income_metrics_processed.csv")
    out_parquet = os.path.join(PATHS["processed_dir"], "hpi_income_metrics.parquet")
    report_path = os.path.join(PATHS["quality_dir"], "data_quality_report.json")

    df.to_csv(out_csv, index=False)
    df.to_parquet(out_parquet, index=False)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print_step(f"📁 Saved CSV to {out_csv}")
    print_step(f"📁 Saved Parquet to {out_parquet}")
    print_step(f"📋 Saved Quality Report to {report_path}")


# ===============================
# MAIN
# ===============================

def main():
    print_step("Loading dataset...")
    df = pd.read_parquet(PATHS["input_parquet"])
    print_step(f"Loaded dataframe with shape {df.shape}")

    report = data_quality_report(df)
    df = process_affordability_metrics(df)

    # Select key columns
    keep_cols = [
        "county_fips_full", "year", "county_name",
        "median_household_income", "median_renters_income",
        "median_home_value", "median_gross_rent",
        "income_change", "source_home", "source_rent",
        "median_monthly_rent_income", "RAI", "HAI"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    save_outputs(df, report)
    print_step("✅ Processing complete.")


if __name__ == "__main__":
    main()
