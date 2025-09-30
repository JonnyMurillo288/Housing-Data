#!/usr/bin/env python3
import requests
import pandas as pd
import os
import geopandas as gpd
from typing import Optional, List
import json

YEARS = list(range(1990, 2024))  # 1990 Census, 2000 Census, ACS 2005–2023

API_KEY = "YOUR_CENSUS_API_KEY"
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
PATHS = {
    "output_csv": os.path.join(ROOT, "home_value_at_county.csv"),
    "shapefiles_dir": os.path.join(ROOT, "shapefiles"),
    "processed_dir": os.path.join(ROOT, "data", "processed"),
    "geo_dir": os.path.join(ROOT, "data", "geo"),
    "quality_dir": os.path.join(ROOT, "data", "quality"),
    "CENSUS_API_KEY": os.path.join(ROOT,"census_api.txt")
}

TARGET_CRS = "EPSG:4326"

# --------------------
# Variable configs
# --------------------
HOME_VALUE_CONFIG = {
    2000: {"dataset": "2000/dec/sf3", "var": "H085001"},     # Median value owner-occupied (2000)
}

# ACS 1-year for larger counties, 2005–2008
for y in range(2005, 2009):
    HOME_VALUE_CONFIG[y] = {"dataset": f"{y}/acs/acs1", "var": "B25077_001E"}

# ACS 5-year for all counties, 2009–2023
for y in range(2009, 2024):
    HOME_VALUE_CONFIG[y] = {"dataset": f"{y}/acs/acs5", "var": "B25077_001E"}

# --------------------
# Functions
# --------------------
def get_API_KEY() -> Optional[str]:
    if os.path.isfile(PATHS["CENSUS_API_KEY"]):
        with open(PATHS["CENSUS_API_KEY"], "r") as f:
            return f.read().strip()
    return None

def load_decennial_1980(path: str) -> pd.DataFrame:
    """
    Load NHGIS-style CSV for 1980 decennial home values.
    Assumes columns: C8J001 (median home value), GISJOIN.
    """
    df = pd.read_csv(path, dtype=str)
    df["C8J001"] = pd.to_numeric(df["C8J001"], errors="coerce")

    # parse GISJOIN -> state_fips + county_fips
    df["state_fips"] = df["GISJOIN"].str[1:3]
    df["county_fips"] = df["GISJOIN"].str[3:6]
    df["county_fips_full"] = df["state_fips"] + df["county_fips"]

    df = df.rename(columns={"C8J001": "median_home_value"})
    df["year"] = 1980
    df["source"] = "nhgis_1980"
    return df[["county_fips_full", "state_fips", "county_fips",
               "median_home_value", "year", "source"]]

def load_decennial_1990(path: str) -> pd.DataFrame:
    """
    Load NHGIS-style CSV for 1990 decennial home values.
    Assumes columns: FCL001 (median home value), GISJOIN.
    """
    df = pd.read_csv(path, dtype=str)
    df["FCL001"] = pd.to_numeric(df["FCL001"], errors="coerce")

    df["state_fips"] = df["GISJOIN"].str[1:3]
    df["county_fips"] = df["GISJOIN"].str[3:6]
    df["county_fips_full"] = df["state_fips"] + df["county_fips"]

    df = df.rename(columns={"FCL001": "median_home_value"})
    df["year"] = 1990
    df["source"] = "nhgis_1990"
    return df[["county_fips_full", "state_fips", "county_fips",
               "median_home_value", "year", "source"]]


def get_county_home_value(years, api_key: str = None) -> pd.DataFrame:
    results = []
    for year in years:
        if year not in HOME_VALUE_CONFIG:
            print(f"⚠️ No home value data available for {year}, skipping.")
            continue

        cfg = HOME_VALUE_CONFIG[year]
        base_url = f"https://api.census.gov/data/{cfg['dataset']}"
        params = {"get": f"NAME,{cfg['var']}", "for": "county:*"}
        if api_key:
            params["key"] = api_key

        resp = requests.get(base_url, params=params)
        if resp.status_code == 204:
            print(f"⚠️ No data for {year}, skipping.")
            continue
        resp.raise_for_status()

        data = resp.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        df[cfg["var"]] = pd.to_numeric(df[cfg["var"]], errors="coerce")

        df = df.rename(columns={
            "NAME": "county_name",
            cfg["var"]: "median_home_value",
            "state": "state_fips",
            "county": "county_fips"
        })
        df["county_fips_full"] = df["state_fips"] + df["county_fips"]
        df["year"] = year
        df["source"] = cfg["dataset"]
        results.append(df)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def save_df(df: pd.DataFrame, path_parquet: str, fallback_csv: Optional[str] = None):
    try:
        df.to_parquet(path_parquet, index=False)
    except Exception:
        if fallback_csv:
            df.to_csv(fallback_csv, index=False)

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    api_key = get_API_KEY() or API_KEY
    print("Processing home value data...")


    # Offline CSVs
    home1980 = load_decennial_1980(
        os.path.join(ROOT, "data_1980.csv")
    )
    home1990 = load_decennial_1990(
        os.path.join(ROOT, "median_home_value_1990.csv")
    )

    # API years: 2000 + ACS 2005–2023
    home_df_api = get_county_home_value(YEARS, api_key)

    # Combine everything
    home_df = pd.concat([home1980, home1990, home_df_api], ignore_index=True)

    # Save raw
    raw_csv = PATHS["output_csv"].replace(".csv", "_raw.csv")
    home_df.to_csv(raw_csv, index=False)

    # Save parquet
    save_df(home_df, PATHS["output_csv"].replace(".csv", ".parquet"), PATHS["output_csv"])

    # Profile
    prof = {
        "Columns": home_df.columns.tolist(),
        "total_years_processed": home_df["year"].nunique(),
        "total_counties": home_df["county_fips_full"].nunique(),
        "years": sorted(home_df["year"].unique().tolist())
    }
    prof_path = os.path.join(PATHS["quality_dir"], "home_value_data_profile.json")
    os.makedirs(PATHS["quality_dir"], exist_ok=True)
    with open(prof_path, "w") as f:
        json.dump(prof, f, indent=4)

    print(f"✅ Processed {len(home_df)} records across {prof['total_years_processed']} years.")
