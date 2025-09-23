#!/usr/bin/env python3
import os
import sys
import re
import json
import difflib
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd, YearEnd

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
    "shapefiles_dir": os.path.join(ROOT, "shapefiles"),
    "processed_dir": os.path.join(ROOT, "data", "processed"),
    "geo_dir": os.path.join(ROOT, "data", "geo"),
    "quality_dir": os.path.join(ROOT, "data", "quality"),
    "fig_maps_dir": os.path.join(ROOT, "figures", "maps"),
}

TARGET_CRS = "EPSG:4326"
INDEX_BASE_DATE = None  # e.g., "2015-01-01"; None => first available per CBSA
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


def detect_msa_identifier_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (cbsa_code_col, msa_name_col)
    """
    cols = set(df.columns)
    cbsa_candidates = [
        "cbsa", "cbsa_code", "cbsa_id", "cbsacode", "cbsa_code_str",
        "cbsafp", "cbsa_fp", "cbsa_fips", "geoid_cbsa", "geoid",
    ]
    name_candidates = [
        "msa_name", "cbsa_title", "name", "area_name", "msa", "market", "metro_area",
    ]
    cbsa_col = next((c for c in cbsa_candidates if c in cols), None)
    name_col = next((c for c in name_candidates if c in cols), None)
    return cbsa_col, name_col


def normalize_cbsa_code(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.extract(r"(\d+)", expand=False)
    s = s.str.zfill(5)
    s = s.where(s.str.match(r"^\d{5}$"), np.nan)
    return s


def clean_msa_name(series: pd.Series) -> pd.Series:
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


def construct_period_from_components(df: pd.DataFrame, freq_col: str = "frequency", yr_col: str = "yr", per_col: str = "period") -> pd.Series:
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


def compute_derived_metrics(df: pd.DataFrame, cbsa_col: str, period_col: str, value_col: str) -> pd.DataFrame:
    df = df[[cbsa_col, period_col, value_col]].dropna().copy()
    df = df.sort_values([cbsa_col, period_col])
    # Index base
    if INDEX_BASE_DATE is not None:
        base_date = pd.Timestamp(INDEX_BASE_DATE)
        base_vals = df[df[period_col] == base_date].set_index(cbsa_col)[value_col]
        df["index_base100"] = df.apply(
            lambda r: (r[value_col] / base_vals.get(r[cbsa_col], np.nan)) * 100.0,
            axis=1,
        )
    else:
        # base = first available per cbsa
        df["index_base100"] = df.groupby(cbsa_col)[value_col].transform(lambda s: (s / s.iloc[0]) * 100.0)

    # Period-over-period change and YoY
    df = df.set_index([cbsa_col, period_col]).sort_index()
    df["chg1"] = df.groupby(level=0)[value_col].pct_change(1)
    # yoy to be computed outside when we know yoy_periods per freq; temporarily keep placeholder
    df["yoy"] = np.nan
    df = df.reset_index()
    return df


def add_yoy_by_freq(df: pd.DataFrame, cbsa_col: str, period_col: str, value_col: str, freq_alias: str) -> pd.DataFrame:
    yoy_periods = YOY_PERIODS.get(freq_alias, 12)
    df = df.sort_values([cbsa_col, period_col])
    df["yoy"] = df.groupby(cbsa_col)[value_col].pct_change(yoy_periods)
    return df


def save_df(df: pd.DataFrame, path_parquet: str, fallback_csv: Optional[str] = None):
    try:
        df.to_parquet(path_parquet, index=False)
    except Exception:
        if fallback_csv:
            df.to_csv(fallback_csv, index=False)


def find_shapefile_or_geojson(shp_dir: str) -> Optional[str]:
    if not os.path.isdir(shp_dir):
        return None
    entries = os.listdir(shp_dir)
    # Prefer .shp, otherwise .geojson or .json
    shp = [e for e in entries if e.lower().endswith(".shp")]
    if shp:
        return os.path.join(shp_dir, shp[0])
    gj = [e for e in entries if e.lower().endswith((".geojson", ".json"))]
    if gj:
        return os.path.join(shp_dir, gj[0])
    return None


def load_and_clean_geometries(path: str):
    if gpd is None:
        print("geopandas not available; skipping shapefile processing.")
        return None
    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        print(f"Failed to read shapefile/geojson: {e}")
        return None
    # Standardize CRS
    try:
        if gdf.crs is None:
            gdf.set_crs(TARGET_CRS, inplace=True)
        else:
            gdf = gdf.to_crs(TARGET_CRS)
    except Exception as e:
        print(f"CRS handling error: {e}")

    # Identify CBSA field
    cols = {c.lower(): c for c in gdf.columns}
    cbsa_field = None
    for c in ["cbsafp", "cbsa", "cbsa_code", "cbsacode", "geoid"]:
        if c in cols:
            cbsa_field = cols[c]
            break
    if cbsa_field is None:
        print("Could not find CBSA code field in geometries; available columns:", list(gdf.columns))
        return None

    gdf["cbsa_code"] = normalize_cbsa_code(gdf[cbsa_field])
    gdf = gdf[~gdf["cbsa_code"].isna()].copy()

    # Identify and clean MSA name from common fields
    name_candidates = ["cbsa_title", "name", "cbsa_name", "metdivname", "namelsad", "title"]
    name_field = next((cols[c] for c in name_candidates if c in cols), None)
    if name_field is not None:
        try:
            gdf["msa_name"] = clean_msa_name(gdf[name_field])
        except Exception:
            gdf["msa_name"] = np.nan
    else:
        gdf["msa_name"] = np.nan

    gdf = gdf.dropna(subset=["msa_name"]).copy()

    # One geometry per cbsa_code; keep first to preserve msa_name
    gdf = gdf.sort_values("cbsa_code").drop_duplicates(subset=["cbsa_code"], keep="first")

    return gdf[["cbsa_code", "msa_name", "geometry"]]


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
    cbsa_col, name_col = detect_msa_identifier_columns(df)

    # Fallbacks for FHFA schema
    if cbsa_col is None and "place_id" in df.columns:
        cbsa_col = "place_id"
    if name_col is None and "place_name" in df.columns:
        name_col = "place_name"

    if cbsa_col is not None:
        df["cbsa_code"] = normalize_cbsa_code(df[cbsa_col])
    else:
        df["cbsa_code"] = np.nan

    if name_col is not None:
        df["msa_name"] = clean_msa_name(df[name_col])
    else:
        df["msa_name"] = np.nan

    # Keep rows with a valid CBSA code only (filter out national/division/state)
    df = df[~df["cbsa_code"].isna()].copy()

    # Phase 3: Time Normalization (from frequency + yr + period)
    if not set(["frequency", "yr", "period"]).issubset(df.columns):
        print("Error: Required time columns (frequency, yr, period) are missing.")
        with open(os.path.join(PATHS["quality_dir"], "qa_error.json"), "w") as f:
            json.dump({"error": "missing_time_columns", "profile": profile}, f, indent=2)
        sys.exit(2)

    df["frequency"] = df["frequency"].astype(str).str.strip().str.lower()
    df = df[df["frequency"].isin(GRANULARITY_ORDER.keys())].copy()

    # Choose most granular frequency per CBSA
    freq_rank = df.assign(_rank=df["frequency"].map(GRANULARITY_ORDER))
    best_rank = freq_rank.groupby("cbsa_code")["_rank"].transform("max")
    df = df[freq_rank["_rank"] == best_rank].copy()
    df["freq_alias"] = df["frequency"].map(FREQ_ALIAS)

    # Construct datetime period
    df["period"] = construct_period_from_components(df, "frequency", "yr", "period")
    if df["period"].isna().all():
        print("Error: Date construction failed for all rows.")
        with open(os.path.join(PATHS["quality_dir"], "qa_error.json"), "w") as f:
            json.dump({"error": "date_construct_failed", "profile": profile}, f, indent=2)
        sys.exit(3)

    # Determine primary metric column
    exclude = {"cbsa_code", "msa_name", "frequency", "yr", "period", "freq_alias"}
    primary_metric = choose_primary_metric(df, exclude_cols=list(exclude))
    if primary_metric is None:
        for c in ["index_sa", "index_nsa"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        primary_metric = choose_primary_metric(df, exclude_cols=list(exclude))
    if primary_metric is None:
        print("Error: Could not identify a numeric metric column to process.")
        with open(os.path.join(PATHS["quality_dir"], "qa_error.json"), "w") as f:
            json.dump({"error": "no_metric_column", "profile": profile}, f, indent=2)
        sys.exit(4)

    # Derived metrics computed per frequency alias
    parts = []
    for fa in sorted(df["freq_alias"].dropna().unique()):
        sub = df[df["freq_alias"] == fa].copy()
        derived = compute_derived_metrics(sub, "cbsa_code", "period", primary_metric)
        # Add yoy by freq
        derived = add_yoy_by_freq(derived, "cbsa_code", "period", primary_metric, fa)
        derived["freq_alias"] = fa
        parts.append(derived)

    if not parts:
        print("Error: No data after frequency filtering and metric computation.")
        with open(os.path.join(PATHS["quality_dir"], "qa_error.json"), "w") as f:
            json.dump({"error": "no_data_after_filter"}, f, indent=2)
        sys.exit(5)

    derived_all = pd.concat(parts, ignore_index=True)

    # Create long and wide tables
    long_df = derived_all.merge(df[["cbsa_code", "msa_name", "period"]].drop_duplicates(), on=["cbsa_code", "period"], how="left")
    long_df = long_df[["cbsa_code", "msa_name", "period", primary_metric, "chg1", "yoy", "index_base100", "freq_alias"]]
    long_df = long_df.rename(columns={primary_metric: "value", "freq_alias": "freq"})
    long_df["metric"] = primary_metric
    long_df["source"] = "hpi_master.csv"

    wide_df = derived_all.pivot_table(index=["cbsa_code", "period"], values=[primary_metric, "chg1", "yoy", "index_base100"]).reset_index()

    # Save tabular outputs
    save_df(long_df, os.path.join(PATHS["processed_dir"], "msa_timeseries_long.parquet"), os.path.join(PATHS["processed_dir"], "msa_timeseries_long.csv"))
    save_df(wide_df, os.path.join(PATHS["processed_dir"], "msa_timeseries_wide.parquet"), os.path.join(PATHS["processed_dir"], "msa_timeseries_wide.csv"))

    # Latest snapshot
    latest_period = wide_df["period"].max()
    latest_df = wide_df[wide_df["period"] == latest_period].copy()
    latest_df.to_csv(os.path.join(PATHS["processed_dir"], "msa_latest.csv"), index=False)

    # Basic profiling output
    prof = {
        **profile,
        "metric_col": primary_metric,
        "period_min": str(pd.to_datetime(long_df["period"]).min()),
        "period_max": str(pd.to_datetime(long_df["period"]).max()),
        "freq_aliases": sorted(long_df["freq"].dropna().unique().tolist()),
        "n_msas": int(long_df["cbsa_code"].nunique()),
    }
    with open(os.path.join(PATHS["quality_dir"], "profile.json"), "w") as f:
        json.dump(prof, f, indent=2)

    # Phase 4: Shapefile Processing
    shp_path = find_shapefile_or_geojson(PATHS["shapefiles_dir"])
    if shp_path is None:
        print("No shapefile or geojson found in shapefiles/. Skipping geometry steps (Phase 4 & partial Phase 5).")
        with open(os.path.join(PATHS["quality_dir"], "unmatched_geometry.json"), "w") as f:
            json.dump({"missing_shapefile": True}, f, indent=2)
        print("Phases 1–3 completed. Phase 4–5 pending shapefile availability.")
        return

    if gpd is None:
        print("geopandas is not installed; cannot process shapefile. Install geopandas to complete Phase 4–5.")
        with open(os.path.join(PATHS["quality_dir"], "unmatched_geometry.json"), "w") as f:
            json.dump({"geopandas_missing": True}, f, indent=2)
        return

    gdf = load_and_clean_geometries(shp_path)
    if gdf is None or gdf.empty:
        print("Geometry load returned empty; skipping joins.")
        with open(os.path.join(PATHS["quality_dir"], "unmatched_geometry.json"), "w") as f:
            json.dump({"geometry_load_failed": True}, f, indent=2)
        print("Phases 1–3 completed. Phase 4–5 pending valid geometry.")
        return

    # Persist cleaned geometries
    try:
        gdf.to_file(os.path.join(PATHS["geo_dir"], "msa_geometries_clean.geojson"), driver="GeoJSON")
    except Exception as e:
        print(f"Failed to save cleaned geometries: {e}")

    # Phase 5: Join & Outputs (latest period)
    latest_df["cbsa_code"] = normalize_cbsa_code(latest_df["cbsa_code"])  # type: ignore

    # Attach MSA names from input data and clean
    name_lookup = df[["cbsa_code", "msa_name"]].dropna().drop_duplicates(subset=["cbsa_code"])  # type: ignore
    latest_df = latest_df.merge(name_lookup, on="cbsa_code", how="left")
    latest_df["msa_name"] = clean_msa_name(latest_df["msa_name"])  # type: ignore
    latest_df['msa_name'] = latest_df['msa_name'].str.replace(" (Msad)", "", regex=False)  # type: ignore

    # Fuzzy match MSA names to geometries
    # Change the match algorithm, first check if the states are the same after the , then do fuzzy match
    def _best_match_map(src_names: pd.Series, tgt_names: pd.Series, cutoff: float = 0.8):
        src_list = sorted(set(src_names.dropna()))
        tgt_list = sorted(set(tgt_names.dropna()))
        tgt_set = set(tgt_list)
        mapping = {}
        scores = {}
        for s in src_list:
            if s in tgt_set:
                mapping[s] = s
                scores[s] = 1.0
                continue
            best_name = None
            best_score = 0.0
            for t in tgt_list:
                r = difflib.SequenceMatcher(None, s, t).ratio()
                if r > best_score:
                    best_score = r
                    best_name = t
            if best_score >= cutoff and best_name is not None:
                mapping[s] = best_name
                scores[s] = float(best_score)
        return mapping, scores
    
    def _closest_match(src_names: pd.Series, tgt_names: pd.Series):
        src_list = sorted(set(src_names.dropna()))
        tgt_list = sorted(set(tgt_names.dropna()))
        mapping = {}
        scores = {}
        for s in src_list:
            best_name = None
            best_score = 0.0
            for t in tgt_list:
                r = difflib.SequenceMatcher(None, s, t).ratio()
                if r > best_score:
                    best_score = r
                    best_name = t
            mapping[s] = best_name
            scores[s] = best_score
        return mapping, scores
    
    name_mapping, name_scores = _best_match_map(latest_df["msa_name"], gdf["msa_name"], cutoff=0.8)  # type: ignore
    latest_df["_matched_name"] = latest_df["msa_name"].map(name_mapping)
    latest_df["_match_score"] = latest_df["msa_name"].map(name_scores)

    # Build and save match profile
    name_to_cbsa = dict(zip(gdf["msa_name"], gdf["cbsa_code"]))  # type: ignore
    profile_df = latest_df[["cbsa_code", "msa_name", "_matched_name", "_match_score"]].copy()
    profile_df["matched_cbsa_code"] = profile_df["_matched_name"].map(name_to_cbsa)
    profile_df["exact_match"] = profile_df["_match_score"].apply(lambda x: bool(x == 1.0))

    # Build unmatched profile
    if latest_df["_matched_name"].isna().any():
        unmatched = latest_df.loc[latest_df["_matched_name"].isna(), ["cbsa_code", "msa_name"]].copy()
        # Find closest even if below cutoff
        unmatched_map, unmatched_scores = _closest_match(unmatched["msa_name"], gdf["msa_name"])
        unmatched["_closest_name"] = unmatched["msa_name"].map(unmatched_map)
        unmatched["_closest_score"] = unmatched["msa_name"].map(unmatched_scores)
        unmatched["closest_cbsa_code"] = unmatched["_closest_name"].map(name_to_cbsa)
        # Append into profile_df for one combined export
        profile_df = pd.concat([profile_df, unmatched], ignore_index=True, sort=False)

    # Save profile as before
    profile_csv = os.path.join(PATHS["quality_dir"], "msa_name_match_profile.csv")
    try:
        profile_df.to_csv(profile_csv, index=False)
    except Exception as e:
        print(f"Failed to save match profile CSV: {e}")


    # Join geometries using matched names
    geo_latest = gdf.merge(latest_df, left_on="msa_name", right_on="_matched_name", how="left")
    unmatched_names = sorted(set(latest_df.loc[latest_df["_matched_name"].isna(), "msa_name"].dropna().tolist()))

    # Save geo outputs
    try:
        geo_latest.to_file(os.path.join(PATHS["geo_dir"], "msa_timeseries_latest.geojson"), driver="GeoJSON")
    except Exception as e:
        print(f"Failed to save latest geojson: {e}")

    with open(os.path.join(PATHS["quality_dir"], "unmatched_join.json"), "w") as f:
        json.dump({
            "unmatched_name_count": int(len(unmatched_names)),
            "unmatched_name_samples": unmatched_names[:20],
            "latest_period": str(latest_period),
        }, f, indent=2)

    print("Phases 1–5 complete. Outputs saved to data/processed and data/geo.")


if __name__ == "__main__":
    main()
