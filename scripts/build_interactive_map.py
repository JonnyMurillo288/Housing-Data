#!/usr/bin/env python3
import os
import json
from typing import List, Optional, Tuple

import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk
from shapely import wkb
import numpy as np

# -----------------------------
# Config and paths
# -----------------------------
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_PARQUET = os.path.join(ROOT, "data", "processed", "hpi_income_metrics.parquet")
PATHS = {
    "shapefiles_dir": os.path.join(ROOT, "shapefiles"),
    "geo_dir": os.path.join(ROOT, "data", "geo"),
    "counties_geojson": os.path.join(ROOT, "data", "geo", "counties.geojson"),
    "merged_parquet": os.path.join(ROOT, "data", "processed", "income_rent_at_county.parquet"),
}


# Expected columns from combine_income_house_price.py
NUMERIC_COLS = [
    "median_household_income",
    "income_change",
    "RAI",
    "HAI"
]
ID_COLS = [
    "county_fips_full",
    "county_name",
    "year",
]


# -----------------------------
# Utilities
# -----------------------------

def _load_dataframe() -> "pd.DataFrame":
    """Load DataFrame from Parquet or csv   """
    last_err: Optional[Exception] = None
    
    path = DEFAULT_PARQUET
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    try:
        if path.endswith(".parquet"):
            # Requires pyarrow
            df = pd.read_parquet(path)  # type: ignore[arg-type]
        elif path.endswith(".csv"):
            df = pd.read_csv(path, dtype={"county_fips_full": str})
        else:
            raise ValueError(f"Unsupported file format: {path}")
        return df
    except Exception as e:
        last_err = e

    if last_err is not None:
        st.error(f"Failed to load any dataset. Last error: {last_err}")
    else:
        st.error(
            "Could not find dataset. Expected one of: \n"
            f"- {DEFAULT_PARQUET}"
        )
    st.stop()
    raise RuntimeError("Unreachable")

def _load_geodataframe(path: Optional [os.path.dirname] = None) -> "gpd.GeoDataFrame":
    """Load GeoDataFrame from one of the given paths (Parquet or GeoJSON).

    Tries each path in order until one works. If `paths` is None, uses a default list.
    """
    last_err: Optional[Exception] = None
    paths = [DEFAULT_PARQUET, PATHS["counties_geojson"]]
    
    if not path: 
        path = PATHS['counties_geojson']
    if not os.path.isfile(path):
        last_err = "Error with file path"
    try:
        if path.endswith(".parquet"):
            # Requires pyarrow
            gdf = gpd.read_parquet(path)  # type: ignore[arg-type]
        elif path.endswith(".geojson") or path.endswith(".json"):
            gdf = gpd.read_file(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        elif gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        return gdf
    except Exception as e:
        last_err = e
    if last_err is not None:
        raise FileNotFoundError("No valid data file found.") from last_err
    else:
        raise FileNotFoundError(
            "Could not find dataset. Expected one of: \n"
            f"- {PATHS['counties_geojson']}"
        )
    

def _detect_value_columns(gdf: "gpd.GeoDataFrame") -> List[str]:
    cols = []
    for c in NUMERIC_COLS:
        if c in gdf.columns:
            cols.append(c)
    # Fallback: all numeric cols except ids
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
    ''' This function will take in two years from the user and then calculate the difference in HAI or RAI between the two years
        INPUTS:
        df - dataframe with year, county_fips_full, and value_col
        year1 - first year to compare
        year2 - second year to compare
        value_col - column name of value to compare
        change_col - column name of new column to store the change
    ''' 
    df1 = df[df['year'] == year1][['county_fips_full', value_col]].rename(columns={value_col: f'{value_col}_{year1}'})
    df2 = df[df['year'] == year2][['county_fips_full', value_col]].rename(columns={value_col: f'{value_col}_{year2}'})
    df_merged = pd.merge(df1, df2, on='county_fips_full', how='inner')
    df_merged[change_col] = ((df_merged[f'{value_col}_{year2}'] - df_merged[f'{value_col}_{year1}']) / df_merged[f'{value_col}_{year1}']) * 100
    df_merged[change_col] = df_merged[change_col].round(2)
    df = pd.merge(
        df,
        df_merged[['county_fips_full', change_col, f'{value_col}_{year2}', f'{value_col}_{year1}']],
        on='county_fips_full',
        how='left',
        suffixes=("", "_dup")
    )
    df = df.drop([c for c in df.columns if c.endswith("_dup")], axis=1)
    # df['metric_year1'] = df[df[df['year'] == year1]][f'{value_col}_{year1}']
    # df["metric_year2"] = df[f'{value_col}_{year2}']
    return df


def _compute_color_scale(series: pd.Series, n_bins: int = 9, diverging: bool = False, reverse: bool = False) -> pd.DataFrame:
    """Compute RGBA colors for values in a series using a simple color ramp.

    Returns a DataFrame with columns [r, g, b, a] aligned to the input index.
    """
    s = pd.to_numeric(series, errors="coerce")

    # Robust bounds
    vmin = s.quantile(0.02)
    vmax = s.quantile(0.98)
    if pd.isna(vmin) or pd.isna(vmax) or vmin == vmax:
        vmin = s.min()
        vmax = s.max()
    if pd.isna(vmin) or pd.isna(vmax) or vmin == vmax:
        # Degenerate; single color
        rgba = pd.DataFrame(
            {"r": 100, "g": 149, "b": 237, "a": 180},
            index=s.index,
        )
        return rgba

    # Normalize 0..1
    t = (s - vmin) / (vmax - vmin)
    t = t.clip(0, 1).fillna(0.0)

    # 🔄 Reverse for RAI (or if explicitly requested)
    if reverse:
        t = 1.0 - t

    # Simple sequential ramp (light -> dark blue)
    def ramp_blue(x: float) -> Tuple[int, int, int]:
        c0 = (247, 251, 255)  # light
        c1 = (8, 48, 107)     # dark
        r = int(c0[0] + (c1[0] - c0[0]) * x)
        g = int(c0[1] + (c1[1] - c0[1]) * x)
        b = int(c0[2] + (c1[2] - c0[2]) * x)
        return r, g, b

    # Diverging ramp (blue-white-red)
    def ramp_diverging(x: float) -> Tuple[int, int, int]:
        if x < 0.5:
            xr = x / 0.5
            c0 = (49, 130, 189)
            c1 = (255, 255, 255)
        else:
            xr = (x - 0.5) / 0.5
            c0 = (255, 255, 255)
            c1 = (202, 0, 32)
        r = int(c0[0] + (c1[0] - c0[0]) * xr)
        g = int(c0[1] + (c1[1] - c0[1]) * xr)
        b = int(c0[2] + (c1[2] - c0[2]) * xr)
        return r, g, b

    rgb = [ramp_diverging(x) if diverging else ramp_blue(x) for x in t]
    rgba = pd.DataFrame(rgb, index=s.index, columns=["r", "g", "b"])
    rgba["a"] = 180
    return rgba


def _to_geojson_dict(gdf: "gpd.GeoDataFrame") -> dict:
    # gdf.to_json() returns a JSON string; convert to dict for pydeck
    gj_str = gdf.to_json()
    return json.loads(gj_str)


# -----------------------------
# Streamlit UI
# -----------------------------


# Reworked version of the interactive map using Streamlit and PyDeck.
# Going to get the dataframe from the processed data
# Then going to allow the user to select a year and a metric to color by
# Then join the dataframe with counties geojson
# Then render the map with pydeck

def main():    
    st.set_page_config(page_title="County Income & Housing Afforbility", layout="wide")
    st.title("Interactive County Map: Median Income & Housing Affordability")

    # -----------------------------
    # Load Data
    # -----------------------------
    with st.spinner("Loading datasets..."):
        df = _load_dataframe()
        if 'period' in df.columns:    
            df = df.drop(columns=['period'])
        df = df.round(2)

        gdf = gpd.read_file(PATHS["counties_geojson"])[["GEOID", "geometry"]].rename(columns={"GEOID": "county_fips_full"})
        gdf['county_fips_full'] = gdf['county_fips_full'].astype(df['county_fips_full'].dtype)

        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        elif gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

    years = sorted(df["year"].dropna().unique()) if "year" in df.columns else []
    value_cols = _detect_value_columns(df)

    # -----------------------------
    # Sidebar Controls
    # -----------------------------
    st.sidebar.header("Filters")
    mode = st.sidebar.radio("View mode:", ["Single year", "Compare two years"], index=0)

    compare_mode = (mode == "Compare two years")

    if mode == "Single year":
        year = st.sidebar.selectbox("Select a year", years, index=len(years)-1)
        df_year = df[df["year"] == year].copy()
        sel_metric = st.sidebar.selectbox("Color by", value_cols) if value_cols else None
        st.write(f"📊 Showing data for **{year}**")

    else:  # Compare two years
        year1 = st.sidebar.selectbox("Select Year 1", years, index=0, key="year1")
        year2 = st.sidebar.selectbox("Select Year 2", years, index=len(years)-1, key="year2")

        if year2 <= year1:
            st.error("⚠️ Please ensure Year 2 is greater than Year 1.")
            return

        sel_metric = st.sidebar.selectbox("Metric to compare", value_cols)
        df_year = df[(df["year"] == year1) | (df["year"] == year2)].copy()
        df_year = calculate_change(df_year, year1, year2, sel_metric, f"{sel_metric}_change")
        df_year = calculate_change(df_year, year1, year2, "median_household_income", f"median_household_income_change")
        df_year = calculate_change(df_year, year1, year2, "median_home_value", f"median_home_value_change")
        df_year = calculate_change(df_year, year1, year2, "median_gross_rent", f"median_gross_rent_price_change")
        df_year = calculate_change(df_year, year1, year2, "HAI", f"HAI_change")
        df_year = calculate_change(df_year, year1, year2, "RAI", f"RAI_change")
        
        
        desc = ""
        if sel_metric == "HAI":
            desc = " Housing Affordability Index"
        elif sel_metric == "RAI":
            desc = " Rent Affordability Index"
    
        st.success(f"Comparing **{desc}** from {year1} → {year2}")

    # -----------------------------
    # Prepare Data for Map
    # -----------------------------
    if df_year.empty:
        st.warning("No data for the selected filter.")
        return

    metric_col = f"{sel_metric}_change" if compare_mode and sel_metric else sel_metric
    if metric_col and metric_col in df_year.columns:
        s = pd.to_numeric(df_year[metric_col], errors="coerce")
        diverging = (s.min(skipna=True) < 0) and (s.max(skipna=True) > 0)
        if sel_metric in ["RAI", "RAI_change"]:
            reverse = True
        else:
            reverse = False
        rgba = _compute_color_scale(s, diverging=diverging,reverse=reverse)
        for ch in ["r", "g", "b", "a"]:
            df_year[f"_c_{ch}"] = rgba[ch].values
        df_year["_fill_color"] = df_year.apply(lambda r: [int(r["_c_r"]), int(r["_c_g"]), int(r["_c_b"]), int(r["_c_a"])], axis=1)
    else:
        df_year["_fill_color"] = [[100, 149, 237, 180]] * len(df_year)

    gdf_year = gdf.merge(df_year, on="county_fips_full", how="left")
    gdf_year = gdf_year[~gdf_year.geometry.isna()].copy()
    if gdf_year.empty:
        st.warning("No geometries found after merging with county shapes.")
        return
    # st.write("Valid geometries:", gdf_year.geometry.notnull().sum())
    # st.write("Geometry types:", gdf_year.geometry.geom_type.value_counts())
    
    # Simplify geometries just a bit (optional)
    gdf_year["geometry"] = gdf_year["geometry"].simplify(0.005, preserve_topology=True) 
    
    
    def fmt_currency(x):
        return f"${x:,.0f}" if pd.notnull(x) else "N/A"

    def fmt_ratio(x):
        return f"{x:,.2f}" if pd.notnull(x) else "N/A"
    
    # ---- FORMAT ALL NUMERIC COLUMNS ----
    gdf_year["median_household_income_fmt"] = gdf_year["median_household_income"].apply(fmt_currency)
    gdf_year["median_gross_rent_fmt"]       = gdf_year["median_gross_rent"].apply(fmt_currency)
    gdf_year["median_home_value_fmt"]       = gdf_year["median_home_value"].apply(fmt_currency)
    gdf_year['HAI_change_fmt']              = gdf_year['HAI_change'].apply(fmt_ratio)
    gdf_year['RAI_change_fmt']              = gdf_year['RAI_change'].apply(fmt_ratio)

    # Format compare-mode columns
    if compare_mode:
        for y in [year1, year2]:
            if f"median_household_income_{y}" in gdf_year:
                gdf_year[f"median_household_income_{y}_fmt"] = gdf_year[f"median_household_income_{y}"].apply(fmt_currency)
            if f"median_home_value_{y}" in gdf_year:
                gdf_year[f"median_home_value_{y}_fmt"] = gdf_year[f"median_home_value_{y}"].apply(fmt_currency)
            if f"median_gross_rent_{y}" in gdf_year:
                gdf_year[f"median_gross_rent_{y}_fmt"] = gdf_year[f"median_gross_rent_{y}"].apply(fmt_currency)
            if f"HAI_{y}" in gdf_year:
                gdf_year[f"HAI_{y}_fmt"] = gdf_year[f"HAI_{y}"].apply(fmt_ratio)
            if f"RAI_{y}" in gdf_year:
                gdf_year[f"RAI_{y}_fmt"] = gdf_year[f"RAI_{y}"].apply(fmt_ratio)

    gj = _to_geojson_dict(gdf_year)    # st.json(gj["features"][0]["properties"])
    
    # -----------------------------
    # Map View
    # -----------------------------
    minx, miny, maxx, maxy = gdf_year.total_bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    zoom = 4.5

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
    
    
    # -----------------------------
    # Tooltip setup
    # -----------------------------
    # Default tooltip
    tooltip = {
        "html": (
            "<b>{county_name} County</b><br/>"
            "Median Household Income: {median_household_income_fmt}<br/>"
            "Median Rent: {median_gross_rent_fmt}<br/>"
            "Median Home Value: {median_home_value_fmt}"
        ),
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }

    # Non-compare mode
    if not compare_mode: 
        if sel_metric == "HAI":
            tooltip = {
                "html": (
                    "<b>{county_name} County</b><br/>"
                    f"HAI: {{metric_value_fmt}}<br/>"
                    "Median Income: {median_household_income_fmt}<br/>"
                    "Median Home Value: {median_home_value_fmt}"
                ),
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }

        elif sel_metric == "RAI":
            tooltip = {
                "html": (
                    "<b>{county_name} County</b><br/>"
                    f"RAI: {{metric_value_fmt}}<br/>"
                    "Median Income: {median_household_income_fmt}<br/>"
                    "Median Rent: {median_gross_rent_fmt}"
                ),
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }

        else:  # Generic metric
            tooltip = {
                "html": (
                    "<b>{county_name} County</b><br/>"
                    f"{sel_metric}: {{metric_value_fmt}}<br/>"
                    "Median Income: {median_household_income_fmt}"
                ),
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }

    # Compare mode
    if compare_mode:
        if sel_metric == "HAI":
            tooltip = {
                "html": (
                    f"<b>{{county_name}} County</b><br/>"
                    f"HAI Change: {{HAI_change_fmt}}%<br/>"
                    f"Year 1 ({year1}) HAI: {{HAI_{year1}_fmt}}<br/>"
                    f"Year 2 ({year2}) HAI: {{HAI_{year2}_fmt}}<br/>"
                    f"Median Income {year1}: {{median_household_income_{year1}_fmt}}<br/>"
                    f"Median Income {year2}: {{median_household_income_{year2}_fmt}}<br/>"
                    f"Median Home Value {year1}: {{median_home_value_{year1}_fmt}}<br/>"
                    f"Median Home Value {year2}: {{median_home_value_{year2}_fmt}}"
                ),
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }

        elif sel_metric == "RAI":
            tooltip = {
                "html": (
                    f"<b>{{county_name}} County</b><br/>"
                    f"RAI Change: {{RAI_change_fmt}}%<br/>"
                    f"Year 1 ({year1}) RAI: {{RAI_{year1}_fmt}}<br/>"
                    f"Year 2 ({year2}) RAI: {{RAI_{year2}_fmt}}<br/>"
                    f"Median Income {year1}: {{median_household_income_{year1}_fmt}}<br/>"
                    f"Median Income {year2}: {{median_household_income_{year2}_fmt}}<br/>"
                    f"Median Rent {year1}: {{median_gross_rent_{year1}_fmt}}<br/>"
                    f"Median Rent {year2}: {{median_gross_rent_{year2}_fmt}}"
                ),
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }

        else:  # Comparing just income or another metric
            tooltip = {
                "html": (
                    f"<b>{{county_name}} County</b><br/>"
                    f"{sel_metric} {year1}: {{median_household_income_{year1}_fmt}}<br/>"
                    f"{sel_metric} {year2}: {{median_household_income_{year2}_fmt}}"
                ),
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(
            latitude=38.5,
            longitude=-98.0,
            zoom=3,   # adjust zoom level to taste
            pitch=0,
            bearing=0
        ),
        map_style=None,  # use default background
        tooltip=tooltip,
    )

    st.subheader("County Map")
    st.write("Hover over a county for details.")
    if sel_metric is "HAI":
        "HAI (Housing Affordability Index) is the ratio of the median home price divided by the median household income. A HAI of 3 means that the price of the median house in a county is 3x the median household income."
    elif sel_metric is "RAI":
        "RAI (Rent Affordability Index) is the ratio of Monthly median household income to the monthly median rent. A RAI of 3 means that a family with median income makes three times as much as the median monthly rent price."
    st.pydeck_chart(r, use_container_width=True)

    # -----------------------------
    # Data Table
    # -----------------------------
    with st.expander("Show data for selected year(s)"):
        cols_to_show = [c for c in ID_COLS if c in gdf_year.columns]
        other_cols = [c for c in gdf_year.columns if c not in cols_to_show + ["geometry"] and not c.startswith("_c_") and c != "_fill_color"]
        st.dataframe(gdf_year[cols_to_show + other_cols].reset_index(drop=True))

    # Create a button for users to download the data as either csv or geojson
    with st.expander("Download data"):
        csv = gdf_year[cols_to_show + other_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='county_income_hpi_data.csv',
            mime='text/csv',
        )
        geojson_str = json.dumps(gj)
        st.download_button(
            label="Download data as GeoJSON",
            data=geojson_str,
            file_name='county_income_hpi_data.geojson',
            mime='application/geo+json',
        )

if __name__ == "__main__":
    # Allow running via `streamlit run scripts/build_interactive_map.py`
    main()
