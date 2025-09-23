#!/usr/bin/env python3
import os
import sys
import re
import json
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd

# Try geopandas; handle environments without it.
try:
    import geopandas as gpd
except Exception as e:  # noqa: F841
    gpd = None


# -----------------------------
# Config
# -----------------------------
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
PATHS = {
    "input_csv": os.path.join(ROOT, "hpi_at_county.csv"),
    "processed_dir": os.path.join(ROOT, "data", "processed"),
    "geo_dir": os.path.join(ROOT, "data", "geo"),
    "quality_dir": os.path.join(ROOT, "data", "quality"),
    "fig_maps_dir": os.path.join(ROOT, "figures", "maps"),
    "counties_geojson": os.path.join(ROOT, "data", "geo", "counties.geojson"),
}

TARGET_CRS = "EPSG:4326"
INDEX_BASE_DATE = None  # e.g., "2015-01-01"; None => first available per county
PRIMARY_METRIC_PREFERENCE = [
    re.compile(r"^index_sa$", re.I),
    re.compile(r"^index_nsa$", re.I),
    re.compile(r"^hpi(_sa)?$", re.I),
    re.compile(r"hpi", re.I),
]

GRANULARITY_ORDER = {"monthly": 3, "quarterly": 2, "annual": 1}
FREQ_ALIAS = {"monthly": "M", "quarterly": "Q", "annual": "A"}
YOY_PERIODS = {"M": 12, "Q": 4, "A": 1}


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


def detect_county_identifier_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (county_fips_col, county_name_col)
    """
    cols = set(df.columns)
    fips_candidates = [
        "fips", "county_fips", "geoid", "county_geoid", "place_id", "countyfp", "cnty_fips",
    ]
    name_candidates = [
        "county_name", "name", "place_name", "area_name", "county", "county_namelsad",
    ]
    fips_col = next((c for c in fips_candidates if c in cols), None)
    name_col = next((c for c in name_candidates if c in cols), None)
    return fips_col, name_col


def normalize_county_fips(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.extract(r"(\d+)", expand=False)
    s = s.str.zfill(5)
    s = s.where(s.str.match(r"^\d{5}$"), np.nan)
    return s


def clean_county_name(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.title()
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


def construct_period_from_components(
    df: pd.DataFrame,
    freq_col: str = "frequency",
    yr_col: str = "yr",
    per_col: str = "period",
) -> pd.Series:
    f = df[freq_col].astype(str).str.strip().str.lower()
    y = pd.to_numeric(df[yr_col], errors="coerce")
    p = pd.to_numeric(df[per_col], errors="coerce")

    # Initialize with NaT
    period_dt = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    # Monthly
    m_mask = f == "monthly"
    if m_mask.any():
        y_m = y[m_mask]
        p_m = p[m_mask].clip(lower=1, upper=12)
        period_dt.loc[m_mask] = pd.to_datetime(
            (y_m.astype(int)).astype(str) + "-" + (p_m.astype(int)).astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )

    # Quarterly (period 1-4 => months 3,6,9,12; use quarter end)
    q_mask = f == "quarterly"
    if q_mask.any():
        y_q = y[q_mask]
        qnum = p[q_mask].clip(lower=1, upper=4)
        months = (qnum.astype(int) * 3).astype(int)
        dt = pd.to_datetime(
            (y_q.astype(int)).astype(str) + "-" + months.astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )
        period_dt.loc[q_mask] = dt + MonthEnd(0)

    # Annual (use year end)
    a_mask = f == "annual"
    if a_mask.any():
        y_a = y[a_mask]
        dt = pd.to_datetime(y_a.astype(int).astype(str) + "-12-31", errors="coerce")
        period_dt.loc[a_mask] = dt

    return period_dt


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

    # Period-over-period change and YoY placeholder
    df = df.set_index([key_col, period_col]).sort_index()
    df["chg1"] = df.groupby(level=0)[value_col].pct_change(1)
    df["yoy"] = np.nan
    df = df.reset_index()
    return df


def add_yoy_by_freq(df: pd.DataFrame, key_col: str, period_col: str, value_col: str, freq_alias: str) -> pd.DataFrame:
    yoy_periods = YOY_PERIODS.get(freq_alias, 12)
    df = df.sort_values([key_col, period_col])
    df["yoy"] = df.groupby(key_col)[value_col].pct_change(yoy_periods)
    return df


def save_df(df: pd.DataFrame, path_parquet: str, fallback_csv: Optional[str] = None):
    try:
        df.to_parquet(path_parquet, index=False)
    except Exception:
        if fallback_csv:
            df.to_csv(fallback_csv, index=False)


# -----------------------------
# Geometry utilities (County)
# -----------------------------

def load_and_clean_county_geometries(path: str):
    if gpd is None:
        print("geopandas not available; skipping shapefile processing.")
        return None
    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        print(f"Failed to read counties geojson: {e}")
        return None

    # Standardize CRS
    try:
        if gdf.crs is None:
            gdf.set_crs(TARGET_CRS, inplace=True)
        else:
            gdf = gdf.to_crs(TARGET_CRS)
    except Exception as e:
        print(f"CRS handling error: {e}")

    # Identify county FIPS/GEOID
    cols = {c.lower(): c for c in gdf.columns}
    fips_field = None
    if "geoid" in cols:
        fips_field = cols["geoid"]
    elif "countyfp" in cols and "statefp" in cols:
        # Combine if needed
        gdf["county_fips"] = gdf[cols["statefp"]].astype(str).str.zfill(2) + gdf[cols["countyfp"]].astype(str).str.zfill(3)
    elif "fips" in cols:
        fips_field = cols["fips"]

    if fips_field is not None:
        gdf["county_fips"] = gdf[fips_field].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(5)

    if "county_fips" not in gdf.columns:
        print("Could not find county FIPS field in geometries; available columns:", list(gdf.columns))
        return None

    # Identify and clean county name from common fields
    name_field = None
    for c in ["namelsad", "name", "countyname", "county", "namelsadlong", "fullname"]:
        if c in cols:
            name_field = cols[c]
            break

    if name_field is not None:
        try:
            gdf["county_name"] = clean_county_name(gdf[name_field])
        except Exception:
            gdf["county_name"] = np.nan
    else:
        gdf["county_name"] = np.nan

    gdf = gdf[~gdf["county_fips"].isna()].copy()

    # One geometry per county_fips; keep first
    gdf = gdf.sort_values("county_fips").drop_duplicates(subset=["county_fips"], keep="first")

    return gdf[["county_fips", "county_name", "geometry"]]


# -----------------------------
# Pipeline
# -----------------------------

def main():
    ensure_dirs()

    # Phase 1: Ingest & Profile
    if not os.path.isfile(PATHS["input_csv"]):
        print(f"Input CSV not found: {PATHS['input_csv']}")
        sys.exit(1)

    # Read with dtype guard
    df = pd.read_csv(PATHS["input_csv"], dtype=str)  # read as str to avoid leading zero loss
    df = standardize_columns(df)

    profile = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "columns": list(df.columns),
    }

    # Phase 2: Identifier Standardization
    fips_col, name_col = detect_county_identifier_columns(df)

    if fips_col is not None:
        df["county_fips"] = normalize_county_fips(df[fips_col])
    else:
        df["county_fips"] = np.nan

    if name_col is not None:
        df["county_name"] = clean_county_name(df[name_col])
    else:
        df["county_name"] = np.nan

    # Keep rows with a valid county FIPS only (filter out national/state aggregates)
    df = df[~df["county_fips"].isna()].copy()

    # Phase 3: Time Normalization (from frequency + yr + period)
    required_time_cols = {"frequency", "yr", "period"}
    if not required_time_cols.issubset(df.columns):
        print("Error: Required time columns (frequency, yr, period) are missing.")
        with open(os.path.join(PATHS["quality_dir"], "county_qa_error.json"), "w") as f:
            json.dump({"error": "missing_time_columns", "profile": profile}, f, indent=2)
        sys.exit(2)

    df["frequency"] = df["frequency"].astype(str).str.strip().str.lower()
    df = df[df["frequency"].isin(GRANULARITY_ORDER.keys())].copy()

    # Choose most granular frequency per county
    freq_rank = df.assign(_rank=df["frequency"].map(GRANULARITY_ORDER))
    best_rank = freq_rank.groupby("county_fips")["_rank"].transform("max")
    df = df[freq_rank["_rank"] == best_rank].copy()
    df["freq_alias"] = df["frequency"].map(FREQ_ALIAS)

    # Construct datetime period
    df["period"] = construct_period_from_components(df, "frequency", "yr", "period")
    if df["period"].isna().all():
        print("Error: Date construction failed for all rows.")
        with open(os.path.join(PATHS["quality_dir"], "county_qa_error.json"), "w") as f:
            json.dump({"error": "date_construct_failed", "profile": profile}, f, indent=2)
        sys.exit(3)

    # Determine primary metric column
    exclude = {"county_fips", "county_name", "frequency", "yr", "period", "freq_alias"}
    primary_metric = choose_primary_metric(df, exclude_cols=list(exclude))
    if primary_metric is None:
        for c in ["index_sa", "index_nsa"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        primary_metric = choose_primary_metric(df, exclude_cols=list(exclude))
    if primary_metric is None:
        print("Error: Could not identify a numeric metric column to process.")
        with open(os.path.join(PATHS["quality_dir"], "county_qa_error.json"), "w") as f:
            json.dump({"error": "no_metric_column", "profile": profile}, f, indent=2)
        sys.exit(4)

    # Derived metrics computed per frequency alias
    parts = []
    for fa in sorted(df["freq_alias"].dropna().unique()):
        sub = df[df["freq_alias"] == fa].copy()
        # coerce metric numeric
        sub[primary_metric] = pd.to_numeric(sub[primary_metric], errors="coerce")
        derived = compute_derived_metrics(sub, "county_fips", "period", primary_metric)
        # Add yoy by freq
        derived = add_yoy_by_freq(derived, "county_fips", "period", primary_metric, fa)
        derived["freq_alias"] = fa
        parts.append(derived)

    if not parts:
        print("Error: No data after frequency filtering and metric computation.")
        with open(os.path.join(PATHS["quality_dir"], "county_qa_error.json"), "w") as f:
            json.dump({"error": "no_data_after_filter"}, f, indent=2)
        sys.exit(5)

    derived_all = pd.concat(parts, ignore_index=True)

    # Create long and wide tables
    long_df = derived_all.merge(
        df[["county_fips", "county_name", "period"]].drop_duplicates(),
        on=["county_fips", "period"],
        how="left",
    )
    long_df = long_df[["county_fips", "county_name", "period", primary_metric, "chg1", "yoy", "index_base100", "freq_alias"]]
    long_df = long_df.rename(columns={primary_metric: "value", "freq_alias": "freq"})
    long_df["metric"] = primary_metric
    long_df["source"] = "hpi_at_county.csv"

    wide_df = derived_all.pivot_table(
        index=["county_fips", "period"],
        values=[primary_metric, "chg1", "yoy", "index_base100"],
    ).reset_index()

    # Save tabular outputs
    save_df(
        long_df,
        os.path.join(PATHS["processed_dir"], "county_timeseries_long.parquet"),
        os.path.join(PATHS["processed_dir"], "county_timeseries_long.csv"),
    )
    save_df(
        wide_df,
        os.path.join(PATHS["processed_dir"], "county_timeseries_wide.parquet"),
        os.path.join(PATHS["processed_dir"], "county_timeseries_wide.csv"),
    )

    # Latest snapshot
    latest_period = wide_df["period"].max()
    latest_df = wide_df[wide_df["period"] == latest_period].copy()
    latest_df.to_csv(os.path.join(PATHS["processed_dir"], "county_latest.csv"), index=False)

    # Basic profiling output
    prof = {
        **profile,
        "metric_col": primary_metric,
        "period_min": str(pd.to_datetime(long_df["period"]).min()),
        "period_max": str(pd.to_datetime(long_df["period"]).max()),
        "freq_aliases": sorted(long_df["freq"].dropna().unique().tolist()),
        "n_counties": int(long_df["county_fips"].nunique()),
    }
    with open(os.path.join(PATHS["quality_dir"], "county_profile.json"), "w") as f:
        json.dump(prof, f, indent=2)

    # Phase 4: County Geometries
    if not os.path.isfile(PATHS["counties_geojson"]):
        print("County GeoJSON not found at data/geo/counties.geojson. Skipping geometry join.")
        with open(os.path.join(PATHS["quality_dir"], "county_unmatched_geometry.json"), "w") as f:
            json.dump({"missing_geojson": True}, f, indent=2)
        print("Phases 1–3 complete. Phase 4–5 pending county geometries.")
        return

    if gpd is None:
        print("geopandas is not installed; cannot process county geometries. Install geopandas to complete Phase 4–5.")
        with open(os.path.join(PATHS["quality_dir"], "county_unmatched_geometry.json"), "w") as f:
            json.dump({"geopandas_missing": True}, f, indent=2)
        return

    gdf = load_and_clean_county_geometries(PATHS["counties_geojson"])
    if gdf is None or gdf.empty:
        print("County geometry load returned empty; skipping joins.")
        with open(os.path.join(PATHS["quality_dir"], "county_unmatched_geometry.json"), "w") as f:
            json.dump({"geometry_load_failed": True}, f, indent=2)
        print("Phases 1–3 completed. Phase 4–5 pending valid geometry.")
        return

    # Persist cleaned geometries
    try:
        gdf.to_file(os.path.join(PATHS["geo_dir"], "county_geometries_clean.geojson"), driver="GeoJSON")
    except Exception as e:
        print(f"Failed to save cleaned county geometries: {e}")

    # Phase 5: Join & Outputs (latest period)
    latest_df["county_fips"] = latest_df["county_fips"].astype(str).str.zfill(5)  # ensure zero-padded

    # Attach county names from input data and clean
    name_lookup = df[["county_fips", "county_name"]].dropna().drop_duplicates(subset=["county_fips"])  # type: ignore
    latest_df = latest_df.merge(name_lookup, on="county_fips", how="left")
    latest_df["county_name"] = clean_county_name(latest_df["county_name"])  # type: ignore

    # Join geometries by exact FIPS
    geo_latest = gdf.merge(latest_df, on="county_fips", how="left")

    # Unmatched report
    missing = latest_df.loc[~latest_df["county_fips"].isin(set(gdf["county_fips"])), "county_fips"].dropna().unique().tolist()

    # Save geo outputs
    try:
        geo_latest.to_file(os.path.join(PATHS["geo_dir"], "county_timeseries_latest.geojson"), driver="GeoJSON")
    except Exception as e:
        print(f"Failed to save latest county geojson: {e}")

    with open(os.path.join(PATHS["quality_dir"], "county_unmatched_join.json"), "w") as f:
        json.dump({
            "unmatched_fips_count": int(len(missing)),
            "unmatched_fips_samples": sorted(missing)[:20],
            "latest_period": str(latest_period),
        }, f, indent=2)

    print("County pipeline complete. Outputs saved to data/processed and data/geo.")


if __name__ == "__main__":
    main()
