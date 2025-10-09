#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import pandas as pd
import geopandas as gpd
from typing import Optional, List, Dict


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
    "geo_dir": os.path.join(ROOT, "data", "geo"),
    "processed_dir": os.path.join(ROOT, "data", "processed"),
    "quality_dir": os.path.join(ROOT, "data", "quality"),
    "fig_maps_dir": os.path.join(ROOT, "figures", "maps"),
    "home_value_data": os.path.join(ROOT, "data", "processed", "home_value_at_county.parquet"),
    "rent_data": os.path.join(ROOT, "data", "processed", "rent_at_county.parquet"),
    "median_income_csv": os.path.join(ROOT, "data", "processed", "income_at_county.csv"),
    "median_renters_income_csv": os.path.join(ROOT, "data", "processed", "renters_income.csv"),
    "output_merged_parquet": os.path.join(ROOT, "data", "processed", "income_hpi_home_rent_at_county.parquet"),
    "profile_json": os.path.join(ROOT, "data", "quality", "home_and_rent_and_income_merge.json"),
}

TARGET_CRS = "EPSG:4326"


# ===============================
# UTILITY FUNCTIONS
# ===============================

def ensure_dirs():
    for key in ["processed_dir", "geo_dir", "quality_dir", "fig_maps_dir"]:
        os.makedirs(PATHS[key], exist_ok=True)

def print_step(msg: str):
    print(f"▶ {msg}")

def snake_case(s: str) -> str:
    s = re.sub(r"[\s\-/]+", "_", s.strip())
    s = re.sub(r"[^0-9a-zA-Z_]+", "", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [snake_case(c) for c in df.columns]
    return df


# ===============================
# DATA LOADING
# ===============================

def load_parquet(path: str, expected_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing parquet file: {path}")
    df = pd.read_parquet(path)
    df = standardize_columns(df)
    if expected_cols:
        df = df[[c for c in expected_cols if c in df.columns]]
    return df

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing CSV file: {path}")
    df = pd.read_csv(path)
    df = standardize_columns(df)
    return df

def load_geojson(path: str, first_rows: Optional[int] = None) -> gpd.GeoDataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing GeoJSON: {path}")
    gdf = gpd.read_file(path, rows=first_rows) if first_rows else gpd.read_file(path)
    if gdf.crs is None or gdf.crs.to_string() != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)
    return gdf


# ===============================
# CLEANING HELPERS
# ===============================

def coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure expected numeric/string types exist."""
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    if "county_fips_full" in df.columns:
        df["county_fips_full"] = df["county_fips_full"].astype(str).str.zfill(5)
    return df

def merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame, on: List[str]) -> pd.DataFrame:
    """Safe merge with dtype alignment."""
    for c in on:
        if df1[c].dtype != df2[c].dtype:
            df2[c] = df2[c].astype(df1[c].dtype)
    merged = df1.merge(df2, on=on, how="left", suffixes=("_left", "_right"))
    return merged


# ===============================
# DOMAIN-SPECIFIC LOADERS
# ===============================

def get_home_value_data() -> pd.DataFrame:
    df = load_parquet(PATHS["home_value_data"])
    df = coerce_columns(df)
    df = df.drop_duplicates(subset=["county_fips_full", "year"])

    df["median_home_value"] = pd.to_numeric(df.get("median_home_value"), errors="coerce")
    return df

def get_rent_data() -> pd.DataFrame:
    df = load_parquet(PATHS["rent_data"])
    df = coerce_columns(df)
    # Drop duplicate rows if 'county_fips_full' and 'year' are the same
    df = df.drop_duplicates(subset=["county_fips_full", "year"])
    df["median_gross_rent"] = pd.to_numeric(df.get("median_gross_rent"), errors="coerce")
    return df

def get_income_data() -> pd.DataFrame:
    income = load_csv(PATHS["median_income_csv"])
    income = income.drop_duplicates(subset=["county_fips_full", "year"])
    renters = load_csv(PATHS["median_renters_income_csv"])

    renters = renters.drop_duplicates(subset=["county_fips_full"])
    renters = renters.rename(columns={"value": "median_renters_income"})
    # Where the year is 2000 multiply by 12 to get annual income
    renters.loc[renters["year"] == 2000, "median_renters_income"] *= 12
    merged = income.merge(
        renters[["county_fips_full", "median_renters_income"]],
        on="county_fips_full",
        how="left"
    )
    return coerce_columns(merged)


# ===============================
# MERGING LOGIC
# ===============================

def merge_home_rent_income() -> pd.DataFrame:
    print_step("Loading income, home value, and rent data...")
    income_df = get_income_data()
    home_df = get_home_value_data()
    rent_df = get_rent_data()

    print_step("Merging home and rent data...")
    home_rent = merge_datasets(home_df, rent_df, ["county_fips_full", "year"])
    print_step("Merging income with home/rent data...")
    merged = merge_datasets(income_df, home_rent, ["county_fips_full", "year"])
    print_step(f"✅ Final merged shape: {merged.shape}")
    return merged


# ===============================
# QUALITY / PROFILING
# ===============================

def profile_dataframe(df: pd.DataFrame) -> Dict:
    profile = {
        "merged_counties_number": df["county_fips_full"].nunique() if "county_fips_full" in df.columns else None,
        "years_covered": df["year"].nunique() if "year" in df.columns else None,
        "total_records": len(df),
        "columns": df.columns.tolist(),
        "number_of_counties_by_year": df.groupby("year")["county_fips_full"].nunique().to_dict() if {"year", "county_fips_full"}.issubset(df.columns) else None,
        "missing_values_summary": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.apply(lambda x: x.name).to_dict(),
    }
    return profile


# ===============================
# MAIN
# ===============================

def main():
    ensure_dirs()
    t0 = time.time()

    merged = merge_home_rent_income()

    print_step("Saving merged dataset...")
    merged.to_parquet(PATHS["output_merged_parquet"], index=False)

    print_step("Building profile JSON...")
    profile = profile_dataframe(merged)
    with open(PATHS["profile_json"], "w") as f:
        json.dump(profile, f, indent=4)

    print_step(f"✅ Done in {time.time() - t0:.2f} seconds")
    print_step(f"Output written to: {PATHS['output_merged_parquet']}")
    print_step(f"Profile saved to: {PATHS['profile_json']}")


if __name__ == "__main__":
    main()
