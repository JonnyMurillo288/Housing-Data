import requests
import pandas as pd
import os
import geopandas as gpd
from typing import Optional, Tuple, List
import json
import re


ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
PATHS = {
    "shapefiles_dir": os.path.join(ROOT, "shapefiles"),
    "hpi_geojson": os.path.join(ROOT, "data", "geo", "hpi_at_county.geojson"),
    "hpi_long_parquet": os.path.join(ROOT, "data", "processed", "county_timeseries_long.parquet"),
    "processed_dir": os.path.join(ROOT, "data", "processed"),
    "geo_dir": os.path.join(ROOT, "data", "geo"),
    "quality_dir": os.path.join(ROOT, "data", "quality"),
    "fig_maps_dir": os.path.join(ROOT, "figures", "maps"),
    "CENSUS_API_KEY": os.path.join(ROOT,"census_api.txt"),
    "merged_parquet": os.path.join(ROOT, "data", "processed", "income_hpi_at_county.parquet"),
}

TARGET_CRS = "EPSG:4326"
# ------
# Utility functions
# ------
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


# ------
# Reading Data functions
# ------

def get_income_geojson(fallback_geojson: Optional[str] = None):
    """Function for getting the income geojson file, with optional fallback."""
    income_geojson_path = os.path.join(PATHS["geo_dir"], "income_at_county.geojson")
    if os.path.isfile(income_geojson_path):
        gdf = gpd.read_file(income_geojson_path)
        gdf = gdf.to_crs(TARGET_CRS)
    elif fallback_geojson and os.path.isfile(fallback_geojson):
        gdf = gpd.read_file(fallback_geojson)
        gdf = gdf.to_crs(TARGET_CRS)
    else:
        raise FileNotFoundError("Income GeoJSON file not found.")
    
    return gdf
    
    
def get_hpi_geojson(fallback_geojson: Optional[str] = None, first_rows: Optional[int]= None):
    """Function for getting the HPI geojson file, with optional fallback."""
    hpi_geojson_path = os.path.join(PATHS["geo_dir"], "hpi_at_county.geojson")
    if os.path.isfile(hpi_geojson_path):
        if first_rows is not None:
            gdf = gpd.read_file(hpi_geojson_path, rows=first_rows)
            gdf = gdf.to_crs(TARGET_CRS)
        else: 
            gdf = gpd.read_file(hpi_geojson_path)
            gdf = gdf.to_crs(TARGET_CRS)
    elif fallback_geojson and os.path.isfile(fallback_geojson):
        if first_rows is not None:
            gdf = gpd.read_file(fallback_geojson, rows=first_rows)
            gdf = gdf.to_crs(TARGET_CRS)
        else: 
            gdf = gpd.read_file(fallback_geojson)
            gdf = gdf.to_crs(TARGET_CRS)
    else:
        raise FileNotFoundError("HPI GeoJSON file not found.")
    
    return gdf

# ------
# Data checks functions
# ------

def change_dtype(df: pd.DataFrame, col: str, dtype) -> pd.DataFrame:
    """Change the data type of a specified column in a DataFrame."""
    df = df.copy()
    if col in df.columns:
        df[col] = df[col].astype(dtype)
    else:
        raise KeyError(f"Column '{col}' not found in DataFrame.")
    return df

def check_merge_columns(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, on: List[str]) -> bool:
    """Check if two GeoDataFrames can be merged on specified columns."""
    for col in on:
        if col not in gdf1.columns or col not in gdf2.columns:
            return False
        if gdf1[col].dtype != gdf2[col].dtype:
            raise ValueError(f"Column '{col}' has different data types: {gdf1[col].dtype} vs {gdf2[col].dtype}")
    return True

def check_CRS(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> bool:
    """Check if two GeoDataFrames have the same CRS."""
    return gdf1.crs == gdf2.crs


import time

if __name__ == "__main__":
    ensure_dirs()
    t = time.time()
    
    income_gdf = get_income_geojson()
    print("Got income GeoDataFrame with columns:", income_gdf.columns.tolist())
    
    # hpi_gdf = get_hpi_geojson(first_rows=5)
    # print("got HPI GeoDataFrame with columns:", hpi_gdf.columns.tolist())
    # Updated Data Types
    # hpi_gdf = change_dtype(hpi_gdf, "county_fips_full", income_gdf.county_fips_full.dtype)
    
    # Read in the parquest file for HPI
    # This file does not have geometry so we will merge without geometry and then add geometry later
    # This is because the parquet file is much smaller and faster to read in
    # than the geojson file with geometry
    hpi_df = pd.read_parquet(PATHS["hpi_long_parquet"])
    hpi_df = standardize_columns(hpi_df)
    hpi_df = change_dtype(hpi_df, "county_fips_full", income_gdf.county_fips_full.dtype)
    print("Got HPI DataFrame with columns:", hpi_df.columns.tolist())    
    print("Shape of HPI DataFrame:", hpi_df.shape)

    # if not check_CRS(income_gdf, hpi_df):
    #     raise ValueError("CRS mismatch between income and HPI GeoDataFrames.")
    if not check_merge_columns(income_gdf, hpi_df, on=["county_fips_full", "year"]):
        raise ValueError("Merge columns missing in one of the GeoDataFrames.")

    income_df = income_gdf[["county_fips_full", "year", "median_household_income",'income_change',"county_name",'geometry']].copy()
    hpi_df = hpi_df[["county_fips_full", "year", "value","chg1",'county_name']].copy()
    hpi_df = hpi_df.rename(columns={"value": "hpi_value", "chg1": "hpi_chg1"}) # Rename columns for clarity when merged
    
    print("Shape of Income GeoDataFrame:", income_gdf.shape)
    print("Going to merge the two GeoDataFrames on 'county_fips_full' and 'year'")

    merged_df = income_df.merge(hpi_df,
        on=["county_fips_full", "year"],
        suffixes=('_income', '_hpi')
    )  
    print("Merge took", time.time() - t, "seconds")
    # breakpoint()

    # Make sure there is a valid geometry column
    if 'geometry' not in merged_df.columns:
        raise ValueError("Merged DataFrame does not have a 'geometry' column.")
    if not isinstance(merged_df.geometry.iloc[0], (gpd.geoseries.GeoSeries,)):
        merged_df = merged_df.drop(columns=['geometry'])
        # Read in raw counties shapefile to get geometry
        counties_shapefile = os.path.join(PATHS["geo_dir"], "counties.geojson")
        counties_gdf = gpd.read_file(counties_shapefile).to_crs(TARGET_CRS)
        counties_gdf = counties_gdf.rename(columns={"GEOID": "county_fips_full"})
        counties_gdf = counties_gdf[['county_fips_full','geometry']]
        counties_gdf = change_dtype(counties_gdf,"county_fips_full",merged_df.county_fips_full.dtype)
        merged_df = merged_df.merge(counties_gdf[["county_fips_full", "geometry"]], on="county_fips_full", how="left")
        # breakpoint()
    # if not isinstance(merged_df.geometry.iloc[0], (gpd.geoseries.GeoSeries,)):
    #     raise ValueError("The 'geometry' column in the merged DataFrame is not valid geometries.")
    if merged_df.crs is None:
        merged_df.set_crs(TARGET_CRS, inplace=True)
    elif merged_df.crs.to_string() != TARGET_CRS:
        merged_df = merged_df.to_crs(TARGET_CRS)
        
    print(merged_df.total_bounds)    
        
    print("Successfully merged GeoDataFrames. Merged columns:", merged_df.columns.tolist())
    output_geojson_path = os.path.join(PATHS["geo_dir"], "income_hpi_at_county.geojson")
    merged_df.to_file(output_geojson_path, driver="GeoJSON")
    merged_df.to_parquet(PATHS["merged_parquet"])
    # merged_df.to_file(PATHS["processed_dir"] + "/income_hpi_at_county.csv")
    # merged_gdf.to_file(output_geojson_path, driver="GeoJSON")
    print(f"Merged GeoDataFrame saved to {output_geojson_path}")
    
    # prof = {
    #     "merged_counties_number": merged_gdf["county_fips_full"].nunique(),
    #     "years_covered": merged_gdf["year"].nunique(),
    #     "total_records": len(merged_gdf),
    #     "merged_counties": merged_gdf["NAMESLAD"].unique().tolist(),
    #     "income_unmerged_counties": [i for i in income_gdf.NAMESLAD.unique().tolist() if i not in hpi_gdf.NAMESLAD.unique().tolist()],   # Placeholder, would need logic to determine unmerged counties
    #     "hpi_unmerged_counties": [i for i in hpi_gdf.NAMESLAD.unique().tolist() if i not in income_gdf.NAMESLAD.unique().tolist()]   # Placeholder, would need logic to determine unmerged counties
    # }
    print("Generating data profile...")
    print(income_gdf.columns)
    print(hpi_df.columns)
    prof = {
        "merged_counties_number": merged_df["county_fips_full"].nunique(),
        "years_covered": merged_df["year"].nunique(),
        "total_records": len(merged_df),
        "income_unmerged_counties": [i for i in income_gdf.county_name.unique().tolist() if i not in hpi_df.county_name.unique().tolist()],   # Placeholder, would need logic to determine unmerged counties
        "hpi_unmerged_counties": [i for i in hpi_df.county_name.unique().tolist() if i not in income_gdf.county_name.unique().tolist()]   # Placeholder, would need logic to determine unmerged counties
    }    
    
    
    prof_path = os.path.join(PATHS["quality_dir"], "income_hpi_data_profile.json")
    with open(prof_path, "w") as f:
        json.dump(prof, f, indent=4)
    print(f"Data profile saved to {prof_path}")    