#!/usr/bin/env python3
import requests
import pandas as pd
import os
import json
from typing import Optional

YEARS = list(range(2000, 2024))  # API supports 2000 Census + ACS 2005–2023
API_KEY = "YOUR_CENSUS_API_KEY"

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
PATHS = {
    "output_csv": os.path.join(ROOT, "rent_at_county.csv"),
    "processed_dir": os.path.join(ROOT, "data", "processed"),
    "quality_dir": os.path.join(ROOT, "data", "quality"),
    "CENSUS_API_KEY": os.path.join(ROOT,"census_api.txt")
}

# --------------------
# API Variable configs
# --------------------
RENT_CONFIG = {
    2000: {"dataset": "2000/dec/sf3", "var": "H063001"},       # Median gross rent (2000)
}
# ACS 1-year (large counties only) 2005–2008
for y in range(2005, 2009):
    RENT_CONFIG[y] = {"dataset": f"{y}/acs/acs1", "var": "B25064_001E"}
# ACS 5-year (all counties) 2009–2023
for y in range(2009, 2024):
    RENT_CONFIG[y] = {"dataset": f"{y}/acs/acs5", "var": "B25064_001E"}

# --------------------
# Functions
# --------------------
def get_API_KEY() -> Optional[str]:
    if os.path.isfile(PATHS["CENSUS_API_KEY"]):
        with open(PATHS["CENSUS_API_KEY"], "r") as f:
            return f.read().strip()
    return None

def get_county_rent(years, api_key: str = None) -> pd.DataFrame:
    results = []
    for year in years:
        if year not in RENT_CONFIG:
            print(f"⚠️ No rent data via API for {year}, skipping.")
            continue
        cfg = RENT_CONFIG[year]
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
            cfg["var"]: "median_gross_rent",
            "state": "state_fips",
            "county": "county_fips"
        })
        df["county_fips_full"] = df["state_fips"] + df["county_fips"]
        df["year"] = year
        df["source"] = cfg["dataset"]
        results.append(df)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def load_offline_rent_csv(path: str, year: int) -> pd.DataFrame:
    """
    Load NHGIS-style CSV for 1980/1990 rent values.
    - Uses GISJOIN to build FIPS
    - Median rent = last column
    """
    df = pd.read_csv(path, dtype=str)
    # Last column as numeric rent
    rent_col = df.columns[-1]
    df[rent_col] = pd.to_numeric(df[rent_col], errors="coerce")

    df["state_fips"] = df["GISJOIN"].str[1:3]
    df["county_fips"] = df["GISJOIN"].str[3:6]
    df["county_fips_full"] = df["state_fips"] + df["county_fips"]

    df = df.rename(columns={rent_col: "median_gross_rent"})
    df["year"] = year
    df["source"] = f"nhgis_{year}"
    return df[["county_fips_full", "state_fips", "county_fips",
               "median_gross_rent", "year", "source"]]

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
    print("Processing median rent data...")
    
        # Offline CSVs (1980, 1990)
    rent1980 = load_offline_rent_csv(os.path.join(ROOT,"data_1980.csv"), 1980)
    rent1990 = load_offline_rent_csv(os.path.join(ROOT,"median_rent_1990.csv"), 1990)

    # API data (2000–2023)
    rent_df_api = get_county_rent(YEARS, api_key)


    # Combine
    rent_df = pd.concat([rent1980, rent1990, rent_df_api], ignore_index=True)

    # Save raw
    raw_csv = PATHS["output_csv"].replace(".csv", "_raw.csv")
    rent_df.to_csv(raw_csv, index=False)

    # Save processed
    save_df(rent_df, PATHS["output_csv"].replace(".csv", ".parquet"), PATHS["output_csv"])

    # Profile
    prof = {
        "Columns": rent_df.columns.tolist(),
        "total_years_processed": rent_df["year"].nunique(),
        "total_counties": rent_df["county_fips_full"].nunique(),
        "years": sorted(rent_df["year"].unique().tolist())
    }
    prof_path = os.path.join(PATHS["quality_dir"], "rent_data_profile.json")
    os.makedirs(PATHS["quality_dir"], exist_ok=True)
    with open(prof_path, "w") as f:
        json.dump(prof, f, indent=4)

    print(f"✅ Processed {len(rent_df)} records across {prof['total_years_processed']} years.")
