#!/usr/bin/env python3
import os
import json
from typing import List, Optional, Tuple

import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk
from shapely import wkb


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


''' This function will take in two years from the user and then calculate the difference in HAI or RAI between the two years
    INPUTS:
    df - dataframe with year, county_fips_full, and value_col
    year1 - first year to compare
    year2 - second year to compare
    value_col - column name of value to compare
    change_col - column name of new column to store the change
'''
def calculate_change(df: pd.DataFrame, year1: int, year2: int, value_col: str, change_col: str) -> pd.DataFrame:
    df1 = df[df['year'] == year1][['county_fips_full', value_col]].rename(columns={value_col: f'{value_col}_{year1}'})
    df2 = df[df['year'] == year2][['county_fips_full', value_col]].rename(columns={value_col: f'{value_col}_{year2}'})
    df_merged = pd.merge(df1, df2, on='county_fips_full', how='inner')
    df_merged[change_col] = ((df_merged[f'{value_col}_{year2}'] - df_merged[f'{value_col}_{year1}']) / df_merged[f'{value_col}_{year1}']) * 100
    df = pd.merge(df, df_merged[['county_fips_full', change_col]], on='county_fips_full', how='left')
    return df


def _compute_color_scale(series: pd.Series, n_bins: int = 9, diverging: bool = False) -> pd.DataFrame:
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
            {
                "r": 100,
                "g": 149,
                "b": 237,
                "a": 180,
            },
            index=s.index,
        )
        return rgba

    # Normalize 0..1
    t = (s - vmin) / (vmax - vmin)
    t = t.clip(0, 1).fillna(0.0)

    # Simple sequential ramp (light -> dark blue)
    def ramp_blue(x: float) -> Tuple[int, int, int]:
        # interpolate between [247,251,255] and [8,48,107]
        c0 = (247, 251, 255)
        c1 = (8, 48, 107)
        r = int(c0[0] + (c1[0] - c0[0]) * x)
        g = int(c0[1] + (c1[1] - c0[1]) * x)
        b = int(c0[2] + (c1[2] - c0[2]) * x)
        return r, g, b

    # Diverging ramp (blue-white-red)
    def ramp_diverging(x: float) -> Tuple[int, int, int]:
        # x in [0,1]; 0 -> blue, 0.5 -> white, 1 -> red
        if x < 0.5:
            # blue -> white
            xr = x / 0.5
            c0 = (49, 130, 189)
            c1 = (255, 255, 255)
        else:
            # white -> red
            xr = (x - 0.5) / 0.5
            c0 = (255, 255, 255)
            c1 = (202, 0, 32)
        r = int(c0[0] + (c1[0] - c0[0]) * xr)
        g = int(c0[1] + (c1[1] - c0[1]) * xr)
        b = int(c0[2] + (c1[2] - c0[2]) * xr)
        return r, g, b

    rgb = [ramp_diverging(x) if diverging else ramp_blue(x) for x in t]
    rgba = pd.DataFrame(rgb, index=s.index, columns=["r", "g", "b"])  # type: ignore
    rgba["a"] = 180
    return rgba


def _to_geojson_dict(gdf: "gpd.GeoDataFrame") -> dict:
    # gdf.to_json() returns a JSON string; convert to dict for pydeck
    gj_str = gdf.to_json()
    return json.loads(gj_str)


# -----------------------------
# Streamlit UI
# -----------------------------

'''
Reworked version of the interactive map using Streamlit and PyDeck.
Going to get the dataframe from the processed data
Then going to allow the user to select a year and a metric to color by
Then join the dataframe with counties geojson
Then render the map with pydeck
'''

def main():    
    st.set_page_config(page_title="County Income & HPI Map", layout="wide")
    st.title("Interactive County Map: Income and HPI")

    with st.spinner("Loading geospatial dataset..."):
        df = _load_dataframe()
    if 'period' in df.columns:    
        df = df.drop(columns=['period'])
    
    # Round numeric columns for display
    cols_to_round = [
    "HAI",
    "RAI",
    "median_household_income",
    "income_change",
    "median_home_value",
    "median_gross_rent"
]

    df[cols_to_round] = df[cols_to_round].round(2)

    # Ensure expected columns
    missing_ids = [c for c in ["year", "county_fips_full"] if c not in df.columns]
    if missing_ids:
        st.warning(
            "Missing expected identifier columns: " + ", ".join(missing_ids) + ". "
            "The app will still try to render available data."
        )

    # Bring in the geometry from counties geojson if not present
    gdf = gpd.read_file(PATHS["counties_geojson"])[["GEOID", "geometry"]].rename(columns={"GEOID": "county_fips_full"})
    gdf['county_fips_full'] = gdf['county_fips_full'].astype(df['county_fips_full'].dtype)

    # Ensure geometry is valid
    if gdf.geometry.dtype == "object":
        try:
            gdf["geometry"] = gdf["geometry"].apply(wkb.loads)
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
        except Exception as e:
            st.error(f"Could not restore geometry: {e}")
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    elif gdf.crs.to_string() != "EPSG:4326":
        try:
            gdf = gdf.to_crs("EPSG:4326")
        except Exception as e:
            st.error(f"Could not reproject geometry to EPSG:4326: {e}")
            
    # Sidebar controls
#     st.sidebar.header("Filters")
#     if years:
#         sel_year = st.sidebar.selectbox("Year", years, index=len(years) - 1)
#         df_year = df[df["year"] == sel_year].copy()
#     else:
#         st.sidebar.info("No 'year' column found; showing all data.")
#         df_year = df.copy()
#         sel_year = None
        
#     # View mode: single year or compare two years    
#     mode = st.radio(
#     "View mode:",
#     ["Single year", "Compare two years"]
# )
    # -----------------
    # Sidebar controls
    # Have user select single year or compare two years
    # if single year, show a dropdown of years
    # if compare two years, show two dropdowns of years and a dropdown of metrics to compare
    # -----------------
    
    years = sorted([int(y) for y in df["year"].dropna().unique()]) if "year" in df.columns else []
    
    st.sidebar.header("Filters")
    mode = st.sidebar.radio(
    "View mode:",
    ["Single year", "Compare two years"], 
    index=0  # default to "Single year"
)
    
    # Get the numeric columns available for coloring and comparing
    value_cols = _detect_value_columns(df)
    
    if mode == "Single year":
        year = st.sidebar.selectbox("Select a year", years, index=len(years)-1)
        df_year = df[df["year"] == year].copy()
        st.write(f"📊 Showing data for **{year}**")
        if value_cols:
            sel_metric = st.sidebar.selectbox("Color by", value_cols, index=0)
        else:
            sel_metric = None
            st.sidebar.info("No numeric columns detected to color by.")
    
    elif mode == "Compare two years":
        col1, col2 = st.columns(2)
        with col1:
            year1 = st.sidebar.selectbox("Select Year 1", years, index=0, key="year1")
        with col2:
            year2 = st.sidebar.selectbox("Select Year 2", years, index=len(years)-1, key="year2")

        if year2 <= year1:
            st.error("⚠️ Please make sure Year 2 is greater than Year 1.")
        else:
            sel_metric = st.sidebar.selectbox("Select a metric to compare", value_cols)
            st.success(f"Comparing **{sel_metric}** from {year1} → {year2}")

            # Pair down to the two years
            df_year = df[(df["year"] == year1) | (df["year"] == year2)].copy()
            # Calculate the percentage change and add as a new column
            df_year = calculate_change(df_year,year1,year2,sel_metric,f'{sel_metric}_change')

    
    
    st.sidebar.caption("Tip: Hover polygons to see detailed attributes in the tooltip.")

    if df_year.empty:
        st.warning("No records for the selected year.")
        return

    # Compute colors
    diverging = False
    if sel_metric is not None:
        # Consider diverging palette if data crosses zero (e.g., changes)
        s = pd.to_numeric(df_year[sel_metric], errors="coerce")
        diverging = (s.min(skipna=True) < 0) and (s.max(skipna=True) > 0)
        rgba = _compute_color_scale(s, diverging=diverging)
        for ch in ["r", "g", "b", "a"]:
            df_year[f"_c_{ch}"] = rgba[ch].values
        df_year["_fill_color"] = df_year.apply(lambda r: [int(r["_c_r"]), int(r["_c_g"]), int(r["_c_b"]), int(r["_c_a"])], axis=1)
    else:
        df_year["_fill_color"] = [[100, 149, 237, 180]] * len(df_year)


    gdf_year = gdf.merge(df_year, on="county_fips_full", how="left")

    gdf_year = gdf_year[~gdf_year["geometry"].isna()].copy()
    # gdf_year = gdf_year[~(gdf_year['hpi_value'] == None)].copy()

    if gdf_year.empty:
        st.warning("No geometries found after merging with county shapes.")
        return
    
    # st.write(gdf_year.columns)

    def tooltip_html(df_year, metric: Optional[str], year1: Optional[int] = None, year2: Optional[int] = None, compare_mode: bool = False) -> str:
        parts = []
        
        # Build a formatted % change column (safe for missing values)
        if metric is not None and f"{metric}_change" in df_year.columns:
            df_year[f"{metric}_change_fmt"] = df_year[f"{metric}_change"].map(
                lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
            )

        if compare_mode:
            # ----------------------------
            # TWO-YEAR COMPARISON TOOLTIP
            # ----------------------------
            if "county_name" in df_year.columns:
                parts.append("County: {county_name}")
            if metric is not None and metric in df_year.columns:
                # Year1 + Year2 values
                if year1 is not None and f"{metric}_year1" in df_year.columns:
                    parts.append(f"{metric} ({year1}): {{{metric}_year1}}")
                if year2 is not None and f"{metric}_year2" in df_year.columns:
                    parts.append(f"{metric} ({year2}): {{{metric}_year2}}")
                # Percent change
                parts.append(f"{metric} % change: {{{metric}_change_fmt}}")

            if metric == "RAI" and "median_gross_rent" in df_year.columns:
                parts.append("Rental Price: ${median_gross_rent}")
            if metric == "HAI" and "median_home_value" in df_year.columns:
                parts.append("Median Home Price: ${median_home_value}")
            if "median_household_income" in df_year.columns:
                parts.append("Median HH Income: ${median_household_income}")

        else:
            # ----------------------------
            # SINGLE-YEAR TOOLTIP (full data)
            # ----------------------------
            if "county_name" in df_year.columns:
                parts.append("County: {county_name}")
            if "HAI" in df_year.columns:
                parts.append("Housing Affordability Index: {HAI}")
            if "RAI" in df_year.columns:
                parts.append("Rent Affordability Index: {RAI}")
            if "year" in df_year.columns:
                parts.append("Year: {year}")
            if "median_household_income" in df_year.columns:
                parts.append("Median HH Income: ${median_household_income}")
            if "income_change" in df_year.columns:
                parts.append("Income Change: {income_change}%")
            if metric is not None and metric in df_year.columns:
                parts.append(f"{metric}: {{{metric}}}")

        return "<br/>".join(parts)
    # ----------------------------
    # RENDER MAP
    # ----------------------------
    st.subheader("County Map")
    
    # Convert to GeoJSON (properties include _fill_color and all columns)
    st.write(gdf_year.shape)
    gdf_year["geometry"] = gdf_year["geometry"].simplify(0.01, preserve_topology=True)
    gj = _to_geojson_dict(gdf_year)

    # Determine initial view from bounds
    try:
        minx, miny, maxx, maxy = gdf_year.total_bounds
        center_lat = (miny + maxy) / 2
        center_lon = (minx + maxx) / 2
        zoom = 4
    except Exception:
        center_lat, center_lon, zoom = 39.5, -98.35, 4

    # Build deck.gl layer
    if pdk is None:
        st.error("pydeck is required for rendering. Install via: pip install pydeck")
        st.stop()
    # st.write("Row count",len(gdf_year))

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=gj,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        get_fill_color="properties._fill_color",
        get_line_color=[80, 80, 80],
        line_width_min_pixels=0.5,
        opacity=0.75,
    )    
    # Define the initial view state
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=8,          # adjust as needed
        pitch=0,
        bearing=0
    )
    # st.write("Geometry type in current view:", type(gdf.geometry.iloc[0]))
    # st.write("First Geometry", gdf.geometry.iloc[0])
    # st.write("Number of Valid Geometries:", gdf.geometry.notnull().sum())
    
    compare_mode = mode == "Compare two years"
    tooltip = {
        "html": tooltip_html(df_year,metric = sel_metric, compare_mode = compare_mode),
        "style": {
            "backgroundColor": "#f0f0f0",
            "color": "#111",
        },
    }
    # st.write(gdf_year.head())
    # st.map(gdf_year)

    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom)

    r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    map_style="light",  # free, no token
    tooltip=tooltip
    )
    
    
    
    st.pydeck_chart(r, use_container_width=True)

    # Legend-ish caption
    if sel_metric is not None:
        st.caption(
            f"Color scale based on '{sel_metric}' for year {sel_year if sel_year is not None else 'All'}. "
            + ("Diverging scale (blue-white-red) used." if diverging else "Sequential blue scale used.")
        )

    # Data table
    with st.expander("Show data for selected year"):
        # Show a neat subset: ids + metric(s)
        cols_to_show = [c for c in ID_COLS if c in gdf_year.columns]
        other_cols = [c for c in gdf_year.columns if c not in cols_to_show + ["geometry"] and not c.startswith("_c_") and c != "_fill_color"]
        st.dataframe(gdf_year[cols_to_show + other_cols].reset_index(drop=True))


if __name__ == "__main__":
    # Allow running via `streamlit run scripts/build_interactive_map.py`
    main()
