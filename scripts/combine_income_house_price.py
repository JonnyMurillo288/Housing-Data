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
    "processed_dir": os.path.join(ROOT, "data", "processed"),
    "geo_dir": os.path.join(ROOT, "data", "geo"),
    "quality_dir": os.path.join(ROOT, "data", "quality"),
    "fig_maps_dir": os.path.join(ROOT, "figures", "maps"),
    "CENSUS_API_KEY": os.path.join(ROOT,"census_api.txt")
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
    
    
def get_hpi_geojson(fallback_geojson: Optional[str] = None):
    """Function for getting the HPI geojson file, with optional fallback."""
    hpi_geojson_path = os.path.join(PATHS["geo_dir"], "hpi_at_county.geojson")
    if os.path.isfile(hpi_geojson_path):
        gdf = gpd.read_file(hpi_geojson_path)
        gdf = gdf.to_crs(TARGET_CRS)
    elif fallback_geojson and os.path.isfile(fallback_geojson):
        gdf = gpd.read_file(fallback_geojson)
        gdf = gdf.to_crs(TARGET_CRS)
    else:
        raise FileNotFoundError("HPI GeoJSON file not found.")
    
    return gdf

# ------
# Data checks functions
# ------

def check_merge_columns(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, on: List[str]) -> bool:
    """Check if two GeoDataFrames can be merged on specified columns."""
    for col in on:
        if col not in gdf1.columns or col not in gdf2.columns:
            return False
    return True

def check_CRS(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> bool:
    """Check if two GeoDataFrames have the same CRS."""
    return gdf1.crs == gdf2.crs



if __name__ == "__main__":
    ensure_dirs()
    
    income_gdf = get_income_geojson()
    print("Got income GeoDataFrame with columns:", income_gdf.columns.tolist())
    
    hpi_gdf = get_hpi_geojson()
    print("got HPI GeoDataFrame with columns:", hpi_gdf.columns.tolist())
    
    if not check_CRS(income_gdf, hpi_gdf):
        raise ValueError("CRS mismatch between income and HPI GeoDataFrames.")
    if not check_merge_columns(income_gdf, hpi_gdf, on=["county_fips_full", "year"]):
        raise ValueError("Merge columns missing in one of the GeoDataFrames.")
    
    merged_gdf = income_gdf.merge(hpi_gdf, on=["county_fips_full", "year"], suffixes=('_income', '_hpi'))
    print("Successfully merged GeoDataFrames. Merged columns:", merged_gdf.columns.tolist())
    output_geojson_path = os.path.join(PATHS["geo_dir"], "income_hpi_at_county.geojson")
    merged_gdf.to_file(output_geojson_path, driver="GeoJSON")
    print(f"Merged GeoDataFrame saved to {output_geojson_path}")
    
    prof = {
        "merged_counties_number": merged_gdf["county_fips_full"].nunique(),
        "years_covered": merged_gdf["year"].nunique(),
        "total_records": len(merged_gdf),
        "merged_counties": merged_gdf["NAMESLAD"].unique().tolist(),
        "income_unmerged_counties": [i for i in income_gdf.NAMESLAD.unique().tolist() if i not in hpi_gdf.NAMESLAD.unique().tolist()],   # Placeholder, would need logic to determine unmerged counties
        "hpi_unmerged_counties": [i for i in hpi_gdf.NAMESLAD.unique().tolist() if i not in income_gdf.NAMESLAD.unique().tolist()]   # Placeholder, would need logic to determine unmerged counties
    }
    
    
    
    prof_path = os.path.join(PATHS["quality_dir"], "income_hpi_data_profile.json")
    with open(prof_path, "w") as f:
        json.dump(prof, f, indent=4)
    print(f"Data profile saved to {prof_path}")    