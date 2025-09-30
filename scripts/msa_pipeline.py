#!/usr/bin/env python3
import os
import sys
import re
import json
from typing import Optional, List

import pandas as pd
import numpy as np
from pandas.tseries.offsets import YearEnd

# Try geopandas; handle environments without it.
try:
    import geopandas as gpd
except Exception:  # noqa: F841
    gpd = None


# -----------------------------
# Config
# -----------------------------
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
PATHS = {
    "input_csv": os.path.join(ROOT, "hpi_at_county.csv"),
    "shapefiles_dir": os.path.join(ROOT, "shapefiles"),
    "processed_dir": os.path.join(ROOT, "data", "processed"),
    "geo_dir": os.path.join(ROOT, "data", "geo"),
    "quality_dir": os.path.join(ROOT, "data", "quality"),
    "fig_maps_dir": os.path.join(ROOT, "figures", "maps"),
}

TARGET_CRS = "EPSG:4326"
INDEX_BASE_DATE: Optional[str] = None  # e.g., "2015-12-31"; None => first available per county

# Prefer the plain HPI column first, then other variants if present
PRIMARY_METRIC_PREFERENCE = [
    re.compile(r"^hpi$", re.I),
    re.compile(r"^hpi_with_2000_base$", re.I),
    re.compile(r"^hpi_with_1990_base$", re.I),
    re.compile(r"hpi", re.I),
]

RESULTS_COLUMS = ['hpi', 'chg1', 'yoy']
YEARS = list(range(1989, 2025))

YOY_PERIODS = {"A": 1}


# -----------------------------
# Utilities
# -----------------------------

def ensure_dirs():
    for p in [PATHS["processed_dir"], PATHS["geo_dir"], PATHS["quality_dir"], PATHS["fig_maps_dir"]]:
        os.makedirs(p, exist_ok=True)


def snake_case(s: str) -> str:
    s = re.sub(r"[\s\-/]+", "_", s.strip())
    s = re.sub(r"[^0-9a-zA-Z_]+", "", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [snake_case(c) for c in df.columns]
    return df


def normalize_fips(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.extract(r"(\d+)", expand=False)
    s = s.str.zfill(5)
    s = s.where(s.str.match(r"^\d{5}$"), np.nan)
    return s


def clean_county_name(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    return s


def choose_primary_metric(df: pd.DataFrame, exclude_cols: List[str]) -> Optional[str]:
    numeric_cols = [c for c in df.columns if c not in exclude_cols]
    # try to coerce potential numeric columns
    for c in numeric_cols:
        try:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        except Exception:
            pass
    numeric_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return None
    # Prefer expected metric names
    for patt in PRIMARY_METRIC_PREFERENCE:
        for c in numeric_cols:
            if patt.search(c):
                return c
    return numeric_cols[0]


def compute_derived_metrics(df: pd.DataFrame, key_col: str, period_col: str, value_col: str) -> pd.DataFrame:
    df = df[[key_col, period_col, value_col]].dropna().copy()
    df = df.sort_values([key_col, period_col])
    # Index base
    if INDEX_BASE_DATE is not None:
        base_date = pd.Timestamp(INDEX_BASE_DATE)
        base_vals = df[df[period_col] == base_date].set_index(key_col)[value_col]
        df["index_base100"] = df.apply(
            lambda r: (r[value_col] / base_vals.get(r[key_col], np.nan)) * 100.0,
            axis=1,
        )
    else:
        # base = first available per key
        df["index_base100"] = df.groupby(key_col)[value_col].transform(lambda s: (s / s.iloc[0]) * 100.0)

    # Period-over-period change and YoY (annual => 1-year lag)
    df = df.set_index([key_col, period_col]).sort_index()
    df["chg1"] = df.groupby(level=0)[value_col].pct_change(1)
    df["yoy"] = df.groupby(level=0)[value_col].pct_change(YOY_PERIODS["A"])
    df = df.reset_index()
    return df


def save_df(df: pd.DataFrame, path_parquet: str, fallback_csv: Optional[str] = None):
    try:
        df.to_parquet(path_parquet, index=False)
    except Exception:
        if fallback_csv:
            df.to_csv(fallback_csv, index=False)


def load_county_geometries() -> Optional["gpd.GeoDataFrame"]:
    """Load county geometries from an existing counties.geojson in data/geo.
    Falls back to shapefile search if needed (not expected in current repo).
    """
    if gpd is None:
        print("geopandas not available; skipping shapefile processing.")
        return None

    # Prefer counties.geojson already in repo
    gj = os.path.join(PATHS["geo_dir"], "counties.geojson")
    if os.path.isfile(gj):
        try:
            gdf = gpd.read_file(gj)
            # Standardize CRS
            if gdf.crs is None:
                gdf.set_crs(TARGET_CRS, inplace=True)
            else:
                gdf = gdf.to_crs(TARGET_CRS)
            # Expect GEOID and NAMELSAD (sometimes NAME or NAMELSAD)
            lower = {c.lower(): c for c in gdf.columns}
            geoid_col = next((lower[c] for c in ["geoid", "fips", "county_fips_full"] if c in lower), None)
            if geoid_col is None:
                print("GEOID-like column not found in counties geojson.")
                return None
            name_col = next((lower[c] for c in ["namelsad", "name", "county_name"] if c in lower), None)
            if name_col is None:
                name_col = geoid_col  # placeholder
            out = gdf[[geoid_col, name_col, "geometry"]].rename(columns={geoid_col: "GEOID", name_col: "NAMELSAD"})
            out["GEOID"] = normalize_fips(out["GEOID"])  # ensure padded
            out = out.dropna(subset=["GEOID"])  # keep valid
            return out
        except Exception as e:
            print(f"Failed to read counties geojson: {e}")
            return None

    # Fallback: look for shapefile in shapefiles_dir (unlikely to be counties in this repo)
    try:
        entries = os.listdir(PATHS["shapefiles_dir"]) if os.path.isdir(PATHS["shapefiles_dir"]) else []
        shp = [e for e in entries if e.lower().endswith(".shp") and "county" in e.lower()]
        if shp:
            gdf = gpd.read_file(os.path.join(PATHS["shapefiles_dir"], shp[0]))
            if gdf.crs is None:
                gdf.set_crs(TARGET_CRS, inplace=True)
            else:
                gdf = gdf.to_crs(TARGET_CRS)
            lower = {c.lower(): c for c in gdf.columns}
            geoid_col = next((lower[c] for c in ["geoid", "fips", "countyfp", "geoid10"] if c in lower), None)
            name_col = next((lower[c] for c in ["namelsad", "name", "countyname"] if c in lower), None)
            if geoid_col is None:
                print("GEOID-like column not found in counties shapefile.")
                return None
            if name_col is None:
                name_col = geoid_col
            out = gdf[[geoid_col, name_col, "geometry"]].rename(columns={geoid_col: "GEOID", name_col: "NAMELSAD"})
            out["GEOID"] = normalize_fips(out["GEOID"])  # ensure padded
            out = out.dropna(subset=["GEOID"])  # keep valid
            return out
    except Exception as e:
        print(f"Failed to read counties shapefile: {e}")
    return None


# -----------------------------
# Pipeline (County-level HPI)
# -----------------------------

def main():
    ensure_dirs()

    # Phase 1: Ingest & Profile
    if not os.path.isfile(PATHS["input_csv"]):
        print(f"Input CSV not found: {PATHS['input_csv']}")
        sys.exit(1)

    # Read with dtype guard to preserve leading zeros in FIPS
    df = pd.read_csv(PATHS["input_csv"], dtype=str)
    df = standardize_columns(df)
    print("Columns in input data:", df.columns.tolist())
    
    profile = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
    }

    # Required columns for county-level series
    required = {"fips_code", "year"}
    if not required.issubset(df.columns):
        print("Error: Required columns for county-level processing are missing (need fips_code and year).")
        with open(os.path.join(PATHS["quality_dir"], "qa_error.json"), "w") as f:
            json.dump({"error": "missing_required_columns", "profile": profile}, f, indent=2)
        sys.exit(2)

    # Phase 2: Identifier Standardization (county)
    df["county_fips_full"] = normalize_fips(df["fips_code"])  # 5-digit county FIPS
    if "county" in df.columns:
        df["county_name"] = clean_county_name(df["county"])  # preserve label
    else:
        df["county_name"] = np.nan
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.strip()
    else:
        df["state"] = np.nan

    # Keep rows with a valid county FIPS
    df = df[~df["county_fips_full"].isna()].copy()
    
    print("Going to do time series processing for", df["county_fips_full"].nunique(), "counties.")
    # Phase 3: Time Normalization (annual by Year)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    # Use Dec 31 of each year for period
    df["period"] = pd.to_datetime(df["year"].astype(str) + "-12-31", errors="coerce")
    if df["period"].isna().all():
        print("Error: Date construction failed for all rows.")
        with open(os.path.join(PATHS["quality_dir"], "qa_error.json"), "w") as f:
            json.dump({"error": "date_construct_failed", "profile": profile}, f, indent=2)
        sys.exit(3)
    df = df.loc[df.year.isin(YEARS)].copy() # Only use years in the defined range
    df = df.drop(columns=['hpi_with_1990_base', 'hpi_with_2000_base'], errors='ignore')
    # Determine primary metric column (prefer 'hpi')
    # Coerce potential metric columns to numeric
    for c in ["hpi"]:#, "hpi_with_1990_base", "hpi_with_2000_base", "annual_change"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    exclude = {"county_fips_full", "county_name", "state", "fips_code", "year", "period"}
    primary_metric = choose_primary_metric(df, exclude_cols=list(exclude))
    if primary_metric is None:
        print("Error: Could not identify a numeric metric column to process.")
        with open(os.path.join(PATHS["quality_dir"], "qa_error.json"), "w") as f:
            json.dump({"error": "no_metric_column", "profile": profile}, f, indent=2)
        sys.exit(4)

    # Derived metrics (annual YoY = 1)
    derived = compute_derived_metrics(df, key_col="county_fips_full", period_col="period", value_col=primary_metric)
    derived = derived.merge(
        df[["county_fips_full", "county_name", "state", "year", "period"]].drop_duplicates(),
        on=["county_fips_full", "period"],
        how="left",
    )

    # Assemble long and wide outputs
    long_df = derived[[
        "county_fips_full", "county_name", "state", "year", "period", primary_metric, "chg1", "yoy", "index_base100"
    ]].rename(columns={primary_metric: "value"})
    long_df["freq"] = "A"
    long_df["metric"] = primary_metric
    long_df["source"] = "hpi_at_county.csv"

    wide_df = derived.pivot_table(
        index=["county_fips_full", "period", "year"],
        values=[primary_metric, "chg1", "yoy", "index_base100"],
    ).reset_index().rename(columns={primary_metric: primary_metric})

    print("Processing complete. Saving outputs...")
    # Save tabular outputs
    save_df(long_df, os.path.join(PATHS["processed_dir"], "county_timeseries_long.parquet"), os.path.join(PATHS["processed_dir"], "county_timeseries_long.csv"))
    save_df(wide_df, os.path.join(PATHS["processed_dir"], "county_timeseries_wide.parquet"), os.path.join(PATHS["processed_dir"], "county_timeseries_wide.csv"))

    # Latest snapshot by year
    latest_year = long_df["year"].max()
    latest_df = long_df[long_df["year"] == latest_year].copy()
    latest_df.to_csv(os.path.join(PATHS["processed_dir"], "county_latest.csv"), index=False)

    print("Saving profile and geometry join...")
    # Basic profiling output
    prof = {
        **profile,
        "metric_col": primary_metric,
        "period_min": str(pd.to_datetime(long_df["period"]).min()),
        "period_max": str(pd.to_datetime(long_df["period"]).max()),
        "n_counties": int(long_df["county_fips_full"].nunique()),
        "years": sorted(pd.unique(long_df["year"]).tolist()),
    }
    with open(os.path.join(PATHS["quality_dir"], "hpi_county_geo_profile.json"), "w") as f:
        json.dump(prof, f, indent=2)

    # Phase 4: Geometry join (using data/geo/counties.geojson)
    gdf = load_county_geometries()
    if gdf is None or gdf.empty:
        print("No county geometries available; saving tabular data only.")
        with open(os.path.join(PATHS["quality_dir"], "unmatched_geometry.json"), "w") as f:
            json.dump({"missing_county_geometry": True}, f, indent=2)
        print("Tabular outputs saved to data/processed. Geo join skipped.")
        return

    # Merge to all-year long_df to create a county-year GeoJSON
    if gpd is None:
        print("geopandas is not installed; cannot create GeoJSON. Install geopandas to export geometry.")
        with open(os.path.join(PATHS["quality_dir"], "unmatched_geometry.json"), "w") as f:
            json.dump({"geopandas_missing": True}, f, indent=2)
        return

    merged = gdf.merge(
        long_df,
        left_on="GEOID",
        right_on="county_fips_full",
        how="right",
    )
    print("Merged geometries for", merged["county_fips_full"].nunique(), "county-years.")
    # Unmatched diagnostics (rows without geometry)
    unmatched = long_df[~long_df["county_fips_full"].isin(gdf["GEOID"])]["county_fips_full"].unique().tolist()
    with open(os.path.join(PATHS["quality_dir"], "unmatched_join.json"), "w") as f:
        json.dump({
            "unmatched_count": int(len(unmatched)),
            "unmatched_samples": unmatched[:20],
            "latest_year": int(latest_year),
        }, f, indent=2)

    # Save GeoJSON of county-year data (HPI and derived metrics)
    try:
        merged_gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=TARGET_CRS)
        merged_gdf.to_file(os.path.join(PATHS["geo_dir"], "hpi_at_county.geojson"), driver="GeoJSON")
    except Exception as e:
        print(f"Failed to save hpi_at_county.geojson: {e}")

    print("County-level HPI processing complete. Outputs saved to data/processed and data/geo.")


if __name__ == "__main__":
    main()
