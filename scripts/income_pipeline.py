import requests
import pandas as pd
import os
import geopandas as gpd
from typing import Optional, Tuple, List
import json
import sys


#YEARS = [i for i in range(1991, 2024)]  # ACS 1-year data available from 2009 to 2023

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

# ------
# Auto Configure the years and variables
# ------
import requests
import pandas as pd


TARGET_CRS = "EPSG:4326"

# ----------------------------
# Validate CSV path
# ----------------------------
def validate_output_path(path: str) -> Tuple[str, str]:
    if not path.lower().endswith(".csv"):
        raise ValueError("Output path must end with .csv")
    base, ext = os.path.splitext(path)
    return base, ext


# ----------------------------
# Auto dataset + variable selection
# ----------------------------
def auto_config(years, variable=None):
    config = {}
    for year in years:
        # Decennial
        if year == 2000:
            dataset = "2000/dec/sf3"
            var = "P060002"  # Median renter HH income (1999$)
        elif year >= 2009:
            dataset = f"{year}/acs/acs5"
            var = variable if variable else "B25119_002E"
        else:
            dataset = "saipe"
            var = "SAEMHI_PT"

        config[year] = {"dataset": dataset, "var": var}
    return config

# ----------------------------
# Parse CLI arguments
# ----------------------------
def parse_args():
    if len(sys.argv) < 2:
        print("Usage: python script.py <output.csv> [start_year] [variable]")
        print("Example: python script.py output.csv 2010 B25119_002E")
        # B25119_002E is median HH Income Owner Occupied
        # B25119_003E is median HH Income Renter Occupied
        sys.exit(1)

    # --- Output path ---
    try:
        base, ext = validate_output_path(sys.argv[1])
        PATHS["output_csv"] = sys.argv[1]
        print(f"✅ Output path set to: {PATHS['output_csv']}")
    except ValueError as ve:
        print(f"❌ Invalid output path provided: {ve}")
        sys.exit(1)

    # --- Start year ---
    if len(sys.argv) > 2:
        try:
            start_year = int(sys.argv[2])
            YEARS = [y for y in range(start_year, 2024) if y >= 1989]
        except ValueError:
            print("❌ Invalid start year provided.")
            sys.exit(1)
    else:
        YEARS = [2000, 2010, 2020]

    print(f"📅 Years selected: {YEARS[0]} → {YEARS[-1]}")

    # --- Variable ---
    variable = sys.argv[3] if len(sys.argv) > 3 else None

    # Build config automatically
    CONFIG = auto_config(YEARS, variable)
    return YEARS, CONFIG, PATHS["output_csv"]
# ------
# Utility functions
# ------

def get_API_KEY() -> Optional[str]:
    if os.path.isfile(PATHS["CENSUS_API_KEY"]):
        with open(PATHS["CENSUS_API_KEY"], "r") as f:
            return f.read().strip()
    else:
        return ValueError("Census API key file not found.")




import requests
import pandas as pd

import requests
import pandas as pd

import requests
import pandas as pd

def get_county_income(years, config, api_key: str = None) -> pd.DataFrame:
    """
    Fetch county-level data for the given Census configuration.
    Ensures only one row per (county, year) even if ACS tables return
    multiple records per county (e.g., B25119_*).
    """
    results = []

    for year in years:
        cfg = config.get(year)
        if not cfg:
            print(f"⚠️ No configuration for {year}, skipping.")
            continue

        dataset = cfg["dataset"]
        var = cfg["var"]

        try:
            # Decide between ACS/Decennial and SAIPE
            if "acs" in dataset or "dec" in dataset:
                base_url = f"https://api.census.gov/data/{dataset}"
                params = {
                    "get": f"NAME,{var}",
                    "for": "county:*",
                    "in": "state:*",          # ✅ ensures unique counties
                }
                if api_key:
                    params["key"] = api_key

                resp = requests.get(base_url, params=params)
                resp.raise_for_status()
                data = resp.json()

                df = pd.DataFrame(data[1:], columns=data[0])
                df[var] = pd.to_numeric(df[var], errors="coerce")

                # Group by county to collapse multiple ACS subcategories
                df = (
                    df.groupby(["state", "county", "NAME"], as_index=False)
                      .agg({var: "median"})  # ✅ only one per county
                )

                df = df.rename(columns={
                    "NAME": "county_name",
                    var: "value",
                    "state": "state_fips",
                    "county": "county_fips"
                })
                df["source"] = dataset

            else:
                # ---- SAIPE fallback ----
                base_url = "https://api.census.gov/data/timeseries/poverty/saipe"
                params = {
                    "get": "NAME,SAEMHI_PT",
                    "for": "county:*",
                    "in": "state:*",
                    "time": str(year),
                }
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
                    "SAEMHI_PT": "value",
                    "state": "state_fips",
                    "county": "county_fips"
                })
                df["source"] = "saipe"

            # ---- Common cleanup ----
            df = df.dropna(subset=["value"])             # drop blanks
            df["county_fips_full"] = df["state_fips"].astype(str) + df["county_fips"].astype(str)
            df["year"] = year
            results.append(df)

        except requests.exceptions.HTTPError as e:
            print(f"HTTP error for {year}: {e}")
        except Exception as e:
            print(f"Error processing {year}: {e}")

    if not results:
        print("⚠️ No data fetched.")
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)

    # ---- Final guarantee: one row per county per year ----
    out = (
        out.sort_values(["year", "county_fips_full", "source"])
           .drop_duplicates(subset=["county_fips_full", "year"], keep="first")
    )
    return out.reset_index(drop=True)

# Find the median income column name based on the variable used
def _get_median_income_col(df: pd.DataFrame) -> str:
    # Search through the columns and find the one that has median and income in it
    cols = df.columns.tolist()
    for col in cols:
        if "median" in col.lower() and "income" in col.lower():
            return col
    # Fallback to 'value' if no specific column found
    return "value"

def calculate_year_over_year_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate year-over-year change in median household income.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'year', 'county_fips_full', and '_get_median_income'.
    """
    
    df = df.sort_values(by=["county_fips_full", "year"])
    income_column  = _get_median_income_col(df)
    df["income_change"] = df.groupby("county_fips_full")[income_column].pct_change() * 100
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
    API_KEY = get_API_KEY()
    if API_KEY is None:
        print("No Census API key found; proceeding without it may lead to rate limiting.")
        
    # Parse arguments
    YEARS, CONFIG, PATHS['output_csv'] = parse_args()

    print("\n✅ Final auto-generated configuration:")
    for year, cfg in CONFIG.items():
        print(f"  {year}: dataset={cfg['dataset']}, var={cfg['var']}")
    
    print(f"Getting data for years:{YEARS[0]} -> {YEARS[-1]}")
    print("Config:", CONFIG)
    print("Output path:", PATHS["output_csv"])
    
    income_df = get_county_income(YEARS, CONFIG, API_KEY)

    # Get API key
        
    raw_income_dfs = []
    print('Processing income data for years:', min(YEARS), 'to', max(YEARS))
    # for year in YEARS:
        # df = df[["county_fips_full", "median_household_income"]]
    # raw_income_dfs.append(df)
          
    # income_df = pd.concat(raw_income_dfs, ignore_index=True) 
    income_df = calculate_year_over_year_change(income_df)
    income_df["year"] = income_df["year"].astype(int)
    income_df = income_df.sort_values(by=["county_fips_full", "year"])
    income_df.to_csv(PATHS["output_csv"].replace(".csv", "_raw.csv"), index=False) #save raw data
    income_df = pd.read_csv(PATHS["output_csv"].replace(".csv", "_raw.csv")) #reload to ensure clean types
    save_df(income_df, PATHS["output_csv"].replace(".csv", ".parquet"), PATHS["output_csv"])
    
    print(f"✅ Income data saved to: {PATHS['output_csv']}")        
    prof = {
        "total_years_processed": len(YEARS),
        "total_counties": income_df["county_fips_full"].nunique(),
        "columns": income_df.columns.tolist(),
        "years": YEARS,
    }
    for year in YEARS:
        count = income_df[income_df["year"] == year]["county_fips_full"].nunique()
        prof[f"counties_with_data_{year}"] = count
    
    prof_path = os.path.join(PATHS["quality_dir"], "income_data_profile.json")
    os.makedirs(PATHS["quality_dir"], exist_ok=True)
    with open(prof_path, "w") as f:
        json.dump(prof, f, indent=4)

