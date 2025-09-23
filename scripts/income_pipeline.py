import requests
import pandas as pd
import os
import geopandas as gpd
from typing import Optional, Tuple, List
import json


YEARS = [i for i in range(2009, 2024)]  # ACS 1-year data available from 2009 to 2023

# If true look through existing housing price data for the years we need to get income for
FIND_YEARS = True # whether to find years with data

API_KEY = "YOUR_CENSUS_API_KEY"  # optional but recommended
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
PATHS = {
    "income_csv": os.path.join(ROOT, "hpi_at_county.csv"),
    "output_csv": os.path.join(ROOT, "income_at_county.csv"),
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

def get_API_KEY() -> Optional[str]:
    if os.path.isfile(PATHS["CENSUS_API_KEY"]):
        with open(PATHS["CENSUS_API_KEY"], "r") as f:
            return f.read().strip()
    else:
        return ValueError("Census API key file not found.")


# ------
# Data fetching and processing functions
# ------

def get_county_income(year: int, api_key: str = None) -> pd.DataFrame:
    """
    Fetch yearly county-level median household income from ACS 1-year estimates.

    Parameters:
        year (int): Year of data (must be available in ACS 1-year API, e.g., 2009–2023).
        api_key (str, optional): Your Census API key (recommended for large queries).

    Returns:
        pd.DataFrame: County-level median household income.
    """
    # ACS 1-year endpoint
    base_url = f"https://api.census.gov/data/{year}/acs/acs1"

    # Variables: B19013_001E = Median household income in the past 12 months (in inflation-adjusted dollars)
    params = {
        "get": "NAME,B19013_001E",
        "for": "county:*"
    }

    if api_key:
        params["key"] = api_key

    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.json()

    # First row is header
    df = pd.DataFrame(data[1:], columns=data[0])

    # Convert income to numeric
    df["B19013_001E"] = pd.to_numeric(df["B19013_001E"], errors="coerce")

    # Rename columns
    df = df.rename(columns={
        "NAME": "county_name",
        "B19013_001E": "median_household_income",
        "state": "state_fips",
        "county": "county_fips"
    })

    # Create a combined FIPS code
    df["county_fips_full"] = df["state_fips"] + df["county_fips"]

    return df

def calculate_year_over_year_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate year-over-year change in median household income.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'year', 'county_fips_full', and 'median_household_income'.
    """
    
    df = df.sort_values(by=["county_fips_full", "year"])
    df["income_change"] = df.groupby("county_fips_full")["median_household_income"].pct_change() * 100
    df_change = df.dropna(subset=["income_change"])
    
    return df_change

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
        
    keep_cols = ['STATEFP','COUNTYFP','GEOID'
                 'NAME', 'NAMELSAD','geometry']
    return gdf[[keep_cols]]

def get_years_with_hpi_data(input_csv: str) -> List[int]:
    try:
        df_hpi = pd.read_csv(input_csv)
        years = df_hpi['year'].unique().tolist()
        return [y for y in years if y in YEARS]
    except Exception as e:
        print(f"Error reading HPI data: {e}")
        return []

# Example usage:
if __name__ == "__main__":
    API_KEY = get_API_KEY()
    if FIND_YEARS:
        YEARS = get_years_with_hpi_data(PATHS["income_csv"])
        
    raw_income_dfs = []
    for year in YEARS:
        df = get_county_income(year,API_KEY)  
        # df = df[["county_fips_full", "median_household_income"]]
        raw_income_dfs.append(df)
          
    income_df = pd.concat(raw_income_dfs, ignore_index=True) 
    income_df = calculate_year_over_year_change(income_df)
    income_df["year"] = income_df["year"].astype(int)
    income_df = income_df.sort_values(by=["county_fips_full", "year"])
    county_shape_path = find_shapefile_or_geojson(PATHS["shapefiles_dir"])
    if county_shape_path:
        gdf_counties = load_and_clean_geometries(county_shape_path)
        if gdf_counties is not None:
            income_df = income_df.merge(gdf_counties, left_on="county_fips_full", right_on="GEOID", how="left")
            merged_counties = income_df["county_name"].notna().unique().tolist()
            unmerged_counties = gdf_counties[~gdf_counties["GEOID"].isin(income_df["county_fips_full"])]["NAMESLAD"].unique().tolist()
            print(f"Merged geometries for {merged_counties} counties.")
            income_gdf = gpd.GeoDataFrame(income_df, geometry="geometry", crs=TARGET_CRS)
            save_df(income_df, PATHS["output_csv"].replace(".csv", ".parquet"), PATHS["output_csv"])
        else:
            print("Failed to load county geometries; saving income data without geometries.")
            save_df(income_df, PATHS["output_csv"].replace(".csv", ".parquet"), PATHS["output_csv"])
    else:
        print("No shapefile/geojson found; saving income data without geometries.")
        save_df(income_df, PATHS["output_csv"].replace(".csv", ".parquet"), PATHS["output_csv"])
        
    prof = {
        "total_years_processed": len(YEARS),
        "total_counties": income_df["county_fips_full"].nunique(),
        "years": YEARS,
        "merged_counties_number": len(merged_counties) if county_shape_path else 0,
        "unmerged_counties_number": len(unmerged_counties) if county_shape_path else 0,
        "counties_merged_with_geometry": merged_counties,
        "counties_missing_geometry": unmerged_counties
    }
    
    prof_path = os.path.join(PATHS["quality_dir"], "income_data_profile.json")
    os.makedirs(PATHS["quality_dir"], exist_ok=True)
    with open(prof_path, "w") as f:
        json.dump(prof, f, indent=4)

