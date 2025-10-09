#!/usr/bin/env python3
import os
import json
from typing import List, Optional, Tuple

import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk
import pyarrow.parquet as pq
import numpy as np

# -----------------------------
# Config and paths
# -----------------------------
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_PARQUET = os.path.join(ROOT, "data", "processed", "hpi_income_metrics.parquet")
PATHS = {
    "geo_dir": os.path.join(ROOT, "data", "geo"),
    "counties_geojson": os.path.join(ROOT, "data", "geo", "counties.parquet"),
}
NUMERIC_COLS = [
    "median_household_income",
    "income_change",
    "median_gross_rent",
    "RAI",
    "HAI"
]
ID_COLS = ["county_fips_full", "county_name", "year"]

# -----------------------------
# Utilities
# -----------------------------
@st.cache_data(show_spinner=False)
def load_year_data(years: List[int]) -> pd.DataFrame:
    """Load only selected years from metrics parquet (massive RAM saver)."""
    filters = [("year", "in", years)]
    table = pq.read_table(DEFAULT_PARQUET, filters=filters)
    df = table.to_pandas()
    if "period" in df.columns:
        df = df.drop(columns=["period"])
    return df.round(2)

@st.cache_data(show_spinner=False)
def _load_geodataframe(path: Optional[str] = None) -> gpd.GeoDataFrame:
    """Load counties geometry efficiently, prefer a simplified parquet."""
    if os.path.isfile(PATHS["counties_geojson"]):
        gdf = gpd.read_parquet(PATHS["counties_geojson"], columns=["GEOID", "geometry"])
        if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        return gdf

    # fallback to full file and simplify once
    path = path or PATHS["counties_geojson"]
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing counties geometry file: {path}")

    if path.endswith(".parquet"):
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path)
    gdf = gdf.to_crs("EPSG:4326")

    keep = [c for c in ["GEOID", "geometry"] if c in gdf.columns]
    gdf = gdf[keep].copy()
    gdf["geometry"] = gdf["geometry"].simplify(0.12, preserve_topology=True)

    gdf.to_parquet(PATHS["counties_geojson"], compression="snappy")
    return gdf

def _detect_value_columns(gdf: pd.DataFrame) -> List[str]:
    cols = [c for c in NUMERIC_COLS if c in gdf.columns]
    if not cols:
        for c in gdf.columns:
            if c in ID_COLS or c == "geometry":
                continue
            try:
                pd.to_numeric(gdf[c].dropna().head(10))
                cols.append(c)
            except Exception:
                continue
    return cols

def calculate_change(df: pd.DataFrame, year1: int, year2: int, value_col: str, change_col: str) -> pd.DataFrame:
    df1 = df[df['year'] == year1][['county_fips_full', value_col]].rename(columns={value_col: f'{value_col}_{year1}'})
    df2 = df[df['year'] == year2][['county_fips_full', value_col]].rename(columns={value_col: f'{value_col}_{year2}'})
    df_merged = pd.merge(df1, df2, on='county_fips_full', how='inner')
    df_merged[change_col] = ((df_merged[f'{value_col}_{year2}'] - df_merged[f'{value_col}_{year1}']) / df_merged[f'{value_col}_{year1}']) * 100
    df_merged[change_col] = df_merged[change_col].round(2)
    df = pd.merge(df, df_merged[['county_fips_full', change_col]], on='county_fips_full', how='left')
    return df

def _compute_color_scale(series: pd.Series, diverging: bool = False, reverse: bool = False) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.DataFrame({"r": 200, "g": 200, "b": 200, "a": 180}, index=s.index)
    vmin = valid.quantile(0.02); vmax = valid.quantile(0.98)
    if pd.isna(vmin) or pd.isna(vmax) or vmin == vmax:
        vmin, vmax = valid.min(), valid.max()
    t = (s - vmin) / (vmax - vmin)
    t = t.clip(0, 1)
    if reverse: t = 1.0 - t

    def ramp_blue(x):
        if pd.isna(x): return (200, 200, 200)
        c0, c1 = (247, 251, 255), (8, 48, 107)
        r = int(c0[0] + (c1[0] - c0[0]) * x)
        g = int(c0[1] + (c1[1] - c0[1]) * x)
        b = int(c0[2] + (c1[2] - c0[2]) * x)
        return r, g, b

    def ramp_diverging(x):
        if pd.isna(x): return (200, 200, 200)
        if x < 0.5:
            xr, c0, c1 = x / 0.5, (49, 130, 189), (255, 255, 255)
        else:
            xr, c0, c1 = (x - 0.5) / 0.5, (255, 255, 255), (202, 0, 32)
        r = int(c0[0] + (c1[0] - c0[0]) * xr)
        g = int(c0[1] + (c1[1] - c0[1]) * xr)
        b = int(c0[2] + (c1[2] - c0[2]) * xr)
        return r, g, b

    rgb = [ramp_diverging(x) if diverging else ramp_blue(x) for x in t]
    rgba = pd.DataFrame(rgb, index=s.index, columns=["r", "g", "b"])
    rgba["a"] = 180
    return rgba

def _to_geojson_dict(gdf: gpd.GeoDataFrame) -> dict:
    return json.loads(gdf.to_json())

# -----------------------------
# Streamlit app
# -----------------------------
def main():
    st.set_page_config(page_title="County Income & Housing Affordability", layout="wide")
    st.title("Interactive County Map: Median Income & Housing Affordability")

    years_all = sorted(pd.read_parquet(DEFAULT_PARQUET, columns=["year"])["year"].unique())
    st.sidebar.header("Filters")
    mode = st.sidebar.radio("View mode:", ["Single year", "Compare two years"], index=0)
    compare_mode = (mode == "Compare two years")

    if mode == "Single year":
        year = st.sidebar.selectbox("Select a year", years_all, index=len(years_all)-1)
        df_year = load_year_data([year])
        sel_metric = st.sidebar.selectbox("Color by", _detect_value_columns(df_year))
        st.write(f"📊 Showing data for **{year}**")
    else:
        year1 = st.sidebar.selectbox("Select Year 1", years_all, index=0, key="year1")
        year2 = st.sidebar.selectbox("Select Year 2", years_all, index=len(years_all)-1, key="year2")
        if year2 <= year1:
            st.error("⚠️ Please ensure Year 2 is greater than Year 1.")
            return
        df_year = load_year_data([year1, year2])
        sel_metric = st.sidebar.selectbox("Metric to compare", _detect_value_columns(df_year))
        df_year = calculate_change(df_year, year1, year2, sel_metric, f"{sel_metric}_change")
        st.success(f"Comparing **{sel_metric}** from {year1} → {year2}")
        
    gdf = _load_geodataframe()[["GEOID", "geometry"]].rename(
        columns={"GEOID": "county_fips_full"}
    )
    gdf["county_fips_full"] = gdf["county_fips_full"].astype(str)
    df_year["county_fips_full"] = df_year["county_fips_full"].astype(str)

    metric_col = f"{sel_metric}_change" if compare_mode and sel_metric else sel_metric
    if metric_col in df_year.columns:
        s = pd.to_numeric(df_year[metric_col], errors="coerce")
        diverging = (s.min(skipna=True) < 0) and (s.max(skipna=True) > 0)
        reverse = sel_metric in ["RAI", "RAI_change"]
        rgba = _compute_color_scale(s, diverging=diverging, reverse=reverse)
        for ch in ["r", "g", "b", "a"]:
            df_year[f"_c_{ch}"] = rgba[ch].values
        df_year["_fill_color"] = df_year.apply(
            lambda r: [int(r["_c_r"]), int(r["_c_g"]), int(r["_c_b"]), int(r["_c_a"])], axis=1
        )
    else:
        df_year["_fill_color"] = [[100,149,237,180]] * len(df_year)

    gdf_year = gdf.merge(df_year, on="county_fips_full", how="left")
    gdf_year = gdf_year[~gdf_year.geometry.isna()].copy()
    if gdf_year.empty:
        st.warning("No geometries found after merging with county shapes.")
        return

    props_keep = [
        "county_fips_full", "county_name", "year",
        metric_col, "_fill_color",
        "median_household_income", "median_gross_rent", "median_home_value", "HAI", "RAI"
    ]
    props_keep = [c for c in props_keep if c in gdf_year.columns]
    gdf_year = gdf_year[["geometry"] + props_keep]
    
    # Drop duplicated columns before JSON export
    gdf_year = gdf_year.loc[:, ~gdf_year.columns.duplicated()]
    gdf_year = gdf_year.rename(columns=lambda x: x.strip())
    keep_cols = ["geometry", "county_fips_full", "county_name", "_fill_color", metric_col]
    gdf_year = gdf_year[[c for c in keep_cols if c in gdf_year.columns]]
    import sys
    
    gj = _to_geojson_dict(gdf_year)
    st.caption(f"GeoJSON size: {sys.getsizeof(json.dumps(gj)) / 1024**2:.1f} MB")
    layer = pdk.Layer(
        "GeoJsonLayer",
        data=gj,
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color="properties._fill_color",
        get_line_color=[80, 80, 80],
        line_width_min_pixels=0.5,
        opacity=0.75,
    )

    tooltip = {"html": "<b>{county_name}</b>", "style": {"backgroundColor": "steelblue", "color": "white"}}

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=38.5, longitude=-98.0, zoom=3),
        map_style=None,
        tooltip=tooltip,
    )
    st.pydeck_chart(r, use_container_width=True)

if __name__ == "__main__":
    main()
 