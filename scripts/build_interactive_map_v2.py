#!/usr/bin/env python3
import os
import json
from typing import List, Optional, Tuple
import streamlit.components.v1 as components


import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk
from shapely import wkb
import numpy as np
import sys
import matplotlib.colors as mcolors

# -----------------------------
# Config and paths
# -----------------------------
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_PARQUET = os.path.join(ROOT, "data", "processed", "hpi_income_metrics.parquet")
PATHS = {
    "shapefiles_dir": os.path.join(ROOT, "shapefiles"),
    "geo_dir": os.path.join(ROOT, "data", "geo"),
    "counties_geojson": os.path.join(ROOT, "data", "geo", "counties.parquet"),
    "merged_parquet": os.path.join(ROOT, "data", "processed", "income_rent_at_county.parquet"),
}
 

# Expected columns from combine_income_house_price.py
NUMERIC_COLS = [
    "median_household_income",
    "income_change",
    "median_gross_rent",
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
# This function will adjust the columns and the dtypes to reduce memory usage
def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """ Reduce memory usage by adjusting column dtypes. """
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["object"]).columns:
        num_unique_values = df[col].nunique()
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5:
            df[col] = df[col].astype("category")
    return df

def drop_unused_columns(df: pd.DataFrame, cols_to_drop: List[str]) -> pd.DataFrame:
    """ Drop columns not in required_cols to save memory. """
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    return df.drop(columns=cols_to_drop)


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
        unused_cols = ['time','state_fips','county_fips','geoid','name','namelsad',
                       'geometry','state_fips_left','county_fips_left','state_fips_right',
                       'county_fips_right','value']
        
        df = drop_unused_columns(df, unused_cols)
        df = optimize_dataframe(df)
        return df
    except Exception as e:
        last_err = e

    if last_err:
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
        if gdf.crs == None:
            gdf.set_crs("EPSG:4326", inplace=True)
        elif gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        return gdf
    except Exception as e:
        last_err = e
    if last_err:
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
#     st.write(
#     df[df["year"].isin([year1, year2])][
#         ["year", "county_fips_full", "median_monthly_rent_income"]
#     ].groupby("year")["median_monthly_rent_income"].describe()
# )
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


def _compute_color_scale(series: pd.Series,
                         diverging: bool = False,
                         reverse: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Compute RGBA colors for percent-change style metrics.
    Bins emphasize near-zero changes:
      [-inf, 0), [0–10), [10–20), [20–40), [40–60), [60–80), [80–100), [100–150), [150, inf)
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0)

    # ------------------------------------------------------------
    # 1. Define semi-fixed asymmetric breaks
    # ------------------------------------------------------------
    bin_edges = np.array([-np.inf, 0, 10, 20, 40, 60, 80, 100, 150, np.inf])

    # Normalize values by their bin position (for continuous mapping)
    bin_ids = np.digitize(s, bin_edges) - 1
    t = bin_ids / (len(bin_edges) - 2)
    t = np.clip(t, 0, 1)
    if reverse:
        t = 1.0 - t

    # ------------------------------------------------------------
    # 2. Custom diverging ramp (dark blue → white → progressive reds)
    # ------------------------------------------------------------
    color_ramp = [
        (33, 102, 172, 255),    # dark blue  (<0)
        (146, 197, 222, 255),   # light blue (near 0)
        (255, 255, 255, 255),   # white (0–10)
        (254, 229, 217, 255),   # pale red
        (252, 187, 161, 255),
        (252, 146, 114, 255),
        (251, 106, 74, 255),
        (239, 59, 44, 255),
        (202, 0, 32, 255)       # deep red (≥150)
    ]

    def ramp_custom(v):
        idx = min(int(v * (len(color_ramp) - 1)), len(color_ramp) - 1)
        return color_ramp[idx]

    rgb = [ramp_custom(v) for v in t]
    rgba = pd.DataFrame(rgb, index=s.index, columns=["r", "g", "b", "a"])

    return rgba, {"breaks": bin_edges, "colors": color_ramp}

def render_dynamic_legend(series,
                          diverging=False,
                          reverse=False,
                          label=None,
                          colors_rgba=None,
                          labels=None):
    if reverse:
        colors_rgba = colors_rgba[::-1]
        labels = labels[::-1]

    hex_colors = [mcolors.to_hex([r/255, g/255, b/255]) for r, g, b, a in colors_rgba]
    legend_title = label or "Value"
    st.markdown(
        f"""
        <div style="
            position: relative;
            margin-top: -420px;  /* overlap pydeck area */
            margin-left: calc(100% - 260px);
            width: 240px;
            background: rgba(255,255,255,0.95);
            border-radius: 10px;
            padding: 10px 15px;
            box-shadow: 0 0 8px rgba(0,0,0,0.3);
            font-family: Arial, sans-serif;
            font-size: 13px;
            line-height: 1.2em;
            color = #222;
            z-index: 1000;">
            <b>{legend_title}</b><br><br>
            {"".join([
                f"<div style='display:flex;align-items:center;margin-bottom:3px;'>"
                f"<div style='width:22px;height:12px;background:{color};margin-right:8px;border:1px solid #aaa;'></div>"
                f"<span>{text}</span></div>"
                for color, text in zip(hex_colors, labels)
            ])}
        </div>
        """,
        unsafe_allow_html=True
    )
 
    # # ---------------------------------------------------------
    # # HTML legend: fixed overlay on top of PyDeck chart
    # # ---------------------------------------------------------
    # legend_html = f"""
    # <style>
    # .custom-legend {{
    #     position: absolute;
    #     right: 25px;
    #     bottom: 30px;
    #     background: rgba(255,255,255,0.95);
    #     border-radius: 10px;
    #     padding: 10px 15px;
    #     box-shadow: 0 0 8px rgba(0,0,0,0.3);
    #     font-family: Arial, sans-serif;
    #     font-size: 13px;
    #     line-height: 1.2em;
    #     z-index: 1000;
    # }}
    # .legend-row {{
    #     display: flex;
    #     align-items: center;
    #     margin-bottom: 3px;
    # }}
    # .legend-color {{
    #     width: 22px;
    #     height: 12px;
    #     margin-right: 8px;
    #     border: 1px solid #aaa;
    # }}
    # </style>

    # <div class="custom-legend">
    #     <b>{legend_title}</b><br><br>
    # """

    # for color, text in zip(hex_colors, labels):
    #     legend_html += f"""
    #     <div class="legend-row">
    #         <div class="legend-color" style="background:{color};"></div>
    #         <span>{text}</span>
    #     </div>
    #     """

    # legend_html += "</div>"

    # # Use negative top margin and transparent background so it overlays pydeck
    # components.html(legend_html, height=0, width=0)

import re

# =====================================
# Normalize and standardize county names globally
# =====================================
def normalize_county_name(name: str) -> str:
    """Clean and standardize county names for consistent grouping."""
    if pd.isna(name):
        return None
    name = str(name).strip()

    # Remove administrative suffixes
    name = re.sub(r"\s+(County|Parish|City|Borough|Municipality|Census Area)$", "", name)

    # Remove trailing state names and abbreviations
    name = re.sub(r",\s*[A-Z][a-z]+$", "", name)   # e.g., ", California"
    name = re.sub(r",\s*[A-Z]{2}$", "", name)       # e.g., ", CA"

    # Clean any leftover commas or whitespace
    name = name.strip(" ,")

    return name




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

        gdf = gpd.read_parquet(PATHS["counties_geojson"])[["GEOID", "geometry"]].rename(columns={"GEOID": "county_fips_full"})
        gdf['county_fips_full'] = gdf['county_fips_full'].astype(df['county_fips_full'].dtype)

        if gdf.crs == None:
            gdf.set_crs("EPSG:4326", inplace=True)
        elif gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326") 

    years = sorted(df["year"].dropna().unique()) if "year" in df.columns else []
    value_cols = _detect_value_columns(df)
    df['median_monthly_rent_income'] = df['median_household_income'] / 12
    # Apply cleaning
    df["county_name_base"] = df["county_name"].apply(normalize_county_name)

    # Find the longest full name for each base name group
    longest_names = (
        df.groupby("county_fips_full")["county_name"]
        .apply(lambda x: max(x, key=len))
        .to_dict()
    )

    # Replace every name with its longest version
    df["county_name_clean"] = df["county_fips_full"].map(longest_names)

    # Drop helper column
    df.drop(columns=["county_name_base"], inplace=True)

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
        df_year = calculate_change(df_year, year1, year2, "median_monthly_rent_income", f"median_monthly_rent_income_change")
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
        rgba, color_meta = _compute_color_scale(s, diverging=diverging, reverse=reverse)
        for ch in ["r", "g", "b", "a"]:
            df_year[f"_c_{ch}"] = rgba[ch].values
        df_year["_fill_color"] = df_year.apply(lambda r: [int(r["_c_r"]), int(r["_c_g"]), int(r["_c_b"]), int(r["_c_a"])], axis=1)
    else:
        df_year["_fill_color"] = [[100, 149, 237, 180]] * len(df_year)

    gdf_year = gdf.merge(df_year, on="county_fips_full", how="left")
    gdf_year = gdf_year[~gdf_year.geometry.isna() & gdf_year[sel_metric].notna()].copy()
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
    gdf_year['median_monthly_rent_income_fmt'] = gdf_year['median_monthly_rent_income'].apply(fmt_currency)
    gdf_year["median_gross_rent_fmt"]       = gdf_year["median_gross_rent"].apply(fmt_currency)
    gdf_year["median_home_value_fmt"]       = gdf_year["median_home_value"].apply(fmt_currency)
    if compare_mode:
        gdf_year['HAI_change_fmt']              = gdf_year['HAI_change'].apply(fmt_ratio)
        gdf_year['RAI_change_fmt']              = gdf_year['RAI_change'].apply(fmt_ratio)

    # Format compare-mode columns
    if compare_mode:
        for y in [year1, year2]:
            if f"median_household_income_{y}" in gdf_year:
                gdf_year[f"median_household_income_{y}_fmt"] = gdf_year[f"median_household_income_{y}"].apply(fmt_currency)
            if f"median_monthly_rent_income_{y}" in gdf_year:
                gdf_year[f"median_monthly_rent_income_{y}_fmt"] = gdf_year[f"median_monthly_rent_income_{y}"].apply(fmt_currency)
            if f"median_home_value_{y}" in gdf_year:
                gdf_year[f"median_home_value_{y}_fmt"] = gdf_year[f"median_home_value_{y}"].apply(fmt_currency)
            if f"median_gross_rent_{y}" in gdf_year:
                gdf_year[f"median_gross_rent_{y}_fmt"] = gdf_year[f"median_gross_rent_{y}"].apply(fmt_currency)
            if f"HAI_{y}" in gdf_year:
                gdf_year[f"HAI_{y}_fmt"] = gdf_year[f"HAI_{y}"].apply(fmt_ratio)
            if f"RAI_{y}" in gdf_year:
                gdf_year[f"RAI_{y}_fmt"] = gdf_year[f"RAI_{y}"].apply(fmt_ratio)

    gj = _to_geojson_dict(gdf_year)    # st.json(gj["features"][0]["properties"])
    st.caption(f"GeoJSON size: {sys.getsizeof(json.dumps(gj)) / 1024**2:.1f} MB")
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
                    "Median Income: {median_monthly_rent_income_fmt}<br/>"
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
                    f"Median Income {year1}: {{median_monthly_rent_income_{year1}_fmt}}<br/>"
                    f"Median Income {year2}: {{median_monthly_rent_income_{year2}_fmt}}<br/>"
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
    if sel_metric == "HAI":
        "HAI (Housing Affordability Index) is the ratio of the median home price divided by the median household income. A HAI of 3 means that the price of the median house in a county is 3x the median household income."
    elif sel_metric == "RAI":
        "RAI (Rent Affordability Index) is the ratio of Monthly median household income to the monthly median rent. A RAI of 3 means that a family with median income makes three times as much as the median monthly rent price."
    st.pydeck_chart(r, use_container_width=True)
    
    # ====================================================
    # --- LEGEND USING SAME COLOR RAMP AS MAP ---
    # ====================================================
    # color_meta was returned by _compute_color_scale()
    breaks = color_meta["breaks"]
    colors_rgba = color_meta["colors"]

    # Build readable labels aligned with the fixed breaks
    labels = []
    for i in range(len(breaks) - 1):
        low, high = breaks[i], breaks[i + 1]
        if i == 0:
            labels.append(f"≤ {high}")
        elif i == len(breaks) - 2:
            labels.append(f"≥ {low}")
        else:
            labels.append(f"{low} – {high}")

    # Reverse ramp for RAI or reverse scale
    if reverse:
        colors_rgba = colors_rgba[::-1]
        labels = labels[::-1]

    # Render the legend (HTML component)
    render_dynamic_legend(
        df_year[metric_col],
        diverging=diverging,
        reverse=reverse,
        label=f"{sel_metric} Change (%)" if compare_mode else sel_metric,
        colors_rgba=colors_rgba,
        labels=labels
    )
    
    # =============================
    # County time series line chart
    # =============================
    st.subheader("County time series")

    plot_metric = sel_metric

    # ---------------------------------------------
    # Create a unique, display-friendly county name
    # ---------------------------------------------
    if "state_name" in df.columns:
        df["county_name_display"] = df["county_name_clean"] + ", " + df["state_name"]
    else:
        # fallback if state_name column not available: use original county_name
        df["county_name_display"] = df["county_name_clean"]

    county_options = (
        df[["county_fips_full", "county_name_display"]]
        .drop_duplicates()
        .sort_values("county_name_display")
    )
    name_to_fips = dict(zip(county_options["county_name_display"], county_options["county_fips_full"]))

    # Default to SF if available
    default_selection = (
        ["San Francisco County, California"]
        if "San Francisco County, California" in county_options["county_name_display"].tolist()
        else []
    )

    selected_county_names = st.multiselect(
        "Select counties to compare",
        options=county_options["county_name_display"].tolist(),
        default=default_selection,
        help=f"Shows {plot_metric} by year for the selected counties."
    )

    if not selected_county_names:
        st.info("Select one or more counties above to see a time series chart.")
    else:
        selected_fips = [name_to_fips[n] for n in selected_county_names if n in name_to_fips]

        # -----------------------------
        # Build tidy dataframe for selected counties
        # -----------------------------
        plot_df = (
            df[df["county_fips_full"].isin(selected_fips)]
            .loc[:, ["year", "county_name_display", plot_metric]]
            .dropna(subset=["year", "county_name_display", plot_metric])
        )

        # Merge duplicates (same county/year) — take max
        plot_df = (
            plot_df.groupby(["year", "county_name_display"], as_index=False)[plot_metric]
            .max(numeric_only=True)
        )

        if plot_df.empty:
            st.warning(f"No data available for {plot_metric} across years for the selected counties.")
        else:
            import altair as alt
            y_title = plot_metric.replace("_", " ").title()

            chart = (
                alt.Chart(plot_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("year:O", title="Year", sort=sorted(plot_df["year"].unique().tolist())),
                    y=alt.Y(f"{plot_metric}:Q", title=y_title),
                    color=alt.Color(
                        "county_name_display:N",
                        title="County",
                        scale=alt.Scale(domain=selected_county_names),  # <-- Only selected
                        legend=alt.Legend(title="County"),
                    ),
                    tooltip=[
                        alt.Tooltip("county_name_display:N", title="County"),
                        alt.Tooltip("year:O", title="Year"),
                        alt.Tooltip(f"{plot_metric}:Q", title=y_title, format=",.2f"),
                    ],
                )
                .properties(height=360)
            )

            st.altair_chart(chart, use_container_width=True)



    # -----------------------------
    # Data Table
    # -----------------------------
    with st.expander("Show data for selected year(s)"):
        cols_to_show = [c for c in ID_COLS if c in gdf_year.columns]
        other_cols = [
            c for c in gdf_year.columns
            if c not in cols_to_show + ["geometry"]
            and not c.startswith("_c_")
            and c != "_fill_color"
        ]
        st.dataframe(gdf_year[cols_to_show + other_cols].reset_index(drop=True))

if __name__ == "__main__":
    # Allow running via `streamlit run scripts/build_interactive_map.py`
    main()
