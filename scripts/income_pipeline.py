import requests
import pandas as pd
import os
import geopandas as gpd
from typing import Optional, Tuple, List
import json


YEARS = [i for i in range(1991, 2024)]  # ACS 1-year data available from 2009 to 2023

# If true look through existing housing price data for the years we need to get income for
FIND_YEARS = True # whether to find years with data

API_KEY = "YOUR_CENSUS_API_KEY"  # optional but recommended
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
PATHS = {
    "hpi_csv": os.path.join(ROOT,"hpi_at_county.csv"),
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
import requests
import pandas as pd

# Map decennial census years to the API dataset and variable code
DECENNIAL_CONFIG = {
    1970: {"dataset": "1970/sf3", "var": "H043001"},       # placeholder, check NHGIS/codebooks
    1980: {"dataset": "1980/sf3", "var": "H058A001"},      # placeholder, check NHGIS/codebooks
    1990: {"dataset": "1990/sf3", "var": "P080A001"},      # Median HH income in 1989 dollars
    2000: {"dataset": "2000/dec/sf3", "var": "P053001"},   # Median HH income in 1999 dollars
    2010: {"dataset": "2010/acs/acs5", "var": "B19013_001E"},  # ACS 5-year
    2020: {"dataset": "2020/acs/acs5", "var": "B19013_001E"},  # ACS 5-year
}

import requests
import pandas as pd

# Decennial / ACS5 configs
DECENNIAL_CONFIG = {
    2000: {"dataset": "2000/dec/sf3", "var": "P053001"},      # Median HH income (1999$)
    2010: {"dataset": "2010/acs/acs5", "var": "B19013_001E"}, # ACS 5-year median HH income
    2020: {"dataset": "2020/acs/acs5", "var": "B19013_001E"}, # ACS 5-year median HH income
}


# Decennial / ACS5 configs
DECENNIAL_CONFIG = {
    2000: {"dataset": "2000/dec/sf3", "var": "P053001"},      # Median HH income (1999$)
    2010: {"dataset": "2010/acs/acs5", "var": "B19013_001E"}, # ACS 5-year median HH income
    2020: {"dataset": "2020/acs/acs5", "var": "B19013_001E"}, # ACS 5-year median HH income
}

def get_county_income(years, api_key: str = None) -> pd.DataFrame:
    """
    Fetch county-level median household income for a list of years.
      - Decennial years (2000, 2010, 2020): Census/ACS5
      - Other years (1989+): SAIPE (skips missing years like 1990, 1991, 1992, 1994, 1996)

    Parameters:
        years (list[int]): Years to fetch
        api_key (str, optional): Census API key

    Returns:
        pd.DataFrame: Combined panel of counties × years
    """
    results = []

    for year in years:
        if year in DECENNIAL_CONFIG:
            # Use Decennial Census / ACS5
            cfg = DECENNIAL_CONFIG[year]
            base_url = f"https://api.census.gov/data/{cfg['dataset']}"
            params = {"get": f"NAME,{cfg['var']}", "for": "county:*"}
            if api_key:
                params["key"] = api_key

            resp = requests.get(base_url, params=params)
            resp.raise_for_status()
            data = resp.json()

            df = pd.DataFrame(data[1:], columns=data[0])
            df[cfg["var"]] = pd.to_numeric(df[cfg["var"]], errors="coerce")
            df = df.rename(columns={
                "NAME": "county_name",
                cfg["var"]: "median_household_income",
                "state": "state_fips",
                "county": "county_fips"
            })
            df["source"] = cfg["dataset"]

        else:
            # Use SAIPE
            base_url = "https://api.census.gov/data/timeseries/poverty/saipe"
            params = {"get": "NAME,SAEMHI_PT", "for": "county:*", "time": str(year)}
            if api_key:
                params["key"] = api_key

            resp = requests.get(base_url, params=params)
            if resp.status_code == 204:
                print(f"⚠️ No SAIPE county data for {year}, skipping.")
                continue

            resp.raise_for_status()
            data = resp.json()

            df = pd.DataFrame(data[1:], columns=data[0])
            df["SAEMHI_PT"] = pd.to_numeric(df["SAEMHI_PT"], errors="coerce")
            df = df.rename(columns={
                "NAME": "county_name",
                "SAEMHI_PT": "median_household_income",
                "state": "state_fips",
                "county": "county_fips"
            })
            df["source"] = "saipe"

        # Common fields
        df["county_fips_full"] = df["state_fips"] + df["county_fips"]
        df["year"] = year
        results.append(df)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


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

def save_geojson(gdf: gpd.GeoDataFrame, dir_path: str, filename: str):
    if gpd is None:
        print("geopandas not available; skipping geojson save.")
        return
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, filename)
    try:
        gdf.to_file(path, driver="GeoJSON")
    except Exception as e:
        print(f"Failed to save GeoJSON: {e}")

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
    keep_cols = ['GEOID','NAME', 'NAMELSAD','geometry']
    
    # Convert GEOID to string and pad with leading zeros if necessary
    gdf['GEOID'] = gdf['GEOID'].astype(str).str.zfill(5)    
    print(gdf.columns)
    return gdf[keep_cols]

def get_years_with_hpi_data(input_csv: str) -> List[int]:
    try:
        df_hpi = pd.read_csv(input_csv)
        years = df_hpi['Year'].unique().tolist()
        return [y for y in years if y in YEARS]
    except Exception as e:
        print(f"Error reading HPI data: {e}")
        return []

# Example usage:
if __name__ == "__main__":
    # API_KEY = get_API_KEY()
    # if FIND_YEARS:
    #     YEARS = get_years_with_hpi_data(PATHS["hpi_csv"])
        
    # raw_income_dfs = []
    # print('Processing income data for years:', min(YEARS), 'to', max(YEARS))
    # # for year in YEARS:
    # income_df = get_county_income(YEARS,API_KEY)  
    #     # df = df[["county_fips_full", "median_household_income"]]
    # # raw_income_dfs.append(df)
          
    # # income_df = pd.concat(raw_income_dfs, ignore_index=True) 
    # income_df = calculate_year_over_year_change(income_df)
    # income_df["year"] = income_df["year"].astype(int)
    # income_df = income_df.sort_values(by=["county_fips_full", "year"])
    # income_df.to_csv(PATHS["output_csv"].replace(".csv", "_raw.csv"), index=False) #save raw data
    income_df = pd.read_csv(PATHS["output_csv"].replace(".csv", "_raw.csv")) #reload to ensure clean types
    # Merge with geometries if available
    county_shape_path = find_shapefile_or_geojson(PATHS["shapefiles_dir"])
    if county_shape_path:
        gdf_counties = load_and_clean_geometries(county_shape_path)
        if gdf_counties is not None:
            gdf_counties['GEOID'] = gdf_counties['GEOID'].astype(str).str.zfill(5)
            gdf_counties['GEOID'] = gdf_counties['GEOID'].astype(int)
            income_df = income_df.merge(gdf_counties, left_on="county_fips_full", right_on="GEOID", how="left")
            merged_counties = income_df["county_name"].unique().tolist()
            unmerged_counties = gdf_counties[~gdf_counties["GEOID"].isin(income_df["county_fips_full"])]["NAMELSAD"].unique().tolist()
            print(f"Merged geometries for {len(merged_counties)} counties.")
            income_gdf = gpd.GeoDataFrame(income_df, geometry="geometry", crs=TARGET_CRS)
            save_df(income_df, PATHS["output_csv"].replace(".csv", ".parquet"), PATHS["output_csv"])
            save_geojson = save_geojson(income_gdf, PATHS["geo_dir"], "income_at_county.geojson")
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

