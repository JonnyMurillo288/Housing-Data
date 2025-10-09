#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
County-level median gross rent (1980–2023)
Combines:
- NHGIS CSVs for 1980 & 1990
- 2000 SF3 Decennial Census
- ACS 1-year (2005–2008)
- ACS 5-year (2009–2023)
"""

import os
import json
import requests
import pandas as pd
from typing import Optional, List

# =====================================
# CONFIG
# =====================================
YEARS = list(range(1980, 2024))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PATHS = {
    "output_csv": os.path.join(ROOT, "data", "processed", "rent_at_county.csv"),
    "quality_dir": os.path.join(ROOT, "data", "quality"),
    "CENSUS_API_KEY": os.path.join(ROOT, "census_api.txt"),
}

# -----------------------------
# API variable mapping
# -----------------------------
RENT_CONFIG = {
    2000: {"dataset": "2000/dec/sf3", "var": "H063001"},
}
for y in range(2005, 2009):
    RENT_CONFIG[y] = {"dataset": f"{y}/acs/acs1", "var": "B25064_001E"}
for y in range(2009, 2024):
    RENT_CONFIG[y] = {"dataset": f"{y}/acs/acs5", "var": "B25064_001E"}


# =====================================
# UTILITIES
# =====================================
def get_API_KEY() -> Optional[str]:
    """Read Census API key from file, if available."""
    if os.path.isfile(PATHS["CENSUS_API_KEY"]):
        with open(PATHS["CENSUS_API_KEY"]) as f:
            return f.read().strip()
    return None


def _load_nhgis_rent(path: str, year: int) -> pd.DataFrame:
    """Load NHGIS-style rent CSV (1980 or 1990)."""
    df = pd.read_csv(path, dtype=str)
    rent_col = df.columns[-1]  # usually the value column
    df[rent_col] = pd.to_numeric(df[rent_col], errors="coerce")

    df["state_fips"] = df["GISJOIN"].str[1:3]
    df["county_fips"] = df["GISJOIN"].str[3:6]
    df["county_fips_full"] = df["state_fips"] + df["county_fips"]

    df = df.rename(columns={rent_col: "median_gross_rent"})
    df["year"] = year
    df["source"] = f"nhgis_{year}"
    return df[["county_fips_full", "state_fips", "county_fips",
               "median_gross_rent", "year", "source"]]


def get_county_rent(years: List[int], api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch ACS/Census rent data for all requested years."""
    results = []
    for year in years:
        if year not in RENT_CONFIG:
            print(f"⚠️ No config for {year}, skipping.")
            continue

        cfg = RENT_CONFIG[year]
        url = f"https://api.census.gov/data/{cfg['dataset']}"
        params = {"get": f"NAME,{cfg['var']}", "for": "county:*"}
        if api_key:
            params["key"] = api_key

        resp = requests.get(url, params=params)
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
        df["county_fips_full"] = df["state_fips"].str.zfill(2) + df["county_fips"].str.zfill(3)
        df["year"] = year
        df["source"] = cfg["dataset"]
        results.append(df)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def deduplicate_rent(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure one record per county-year, preferring ACS5 over ACS1."""
    df = df.copy()
    df = df.sort_values(["county_fips_full", "year", "source"])
    # assign preference (acs5 > acs1 > decennial)
    df["source_priority"] = df["source"].apply(
        lambda s: 1 if "acs5" in s else (2 if "acs1" in s else 3)
    )
    df = df.sort_values(["county_fips_full", "year", "source_priority"])
    df = df.drop_duplicates(subset=["county_fips_full", "year"], keep="first")

    # final aggregate safety net
    df = (
        df.groupby(["county_fips_full", "year"], as_index=False)
          .agg({
              "median_gross_rent": "mean",
              "state_fips": "first",
              "county_fips": "first",
              "source": "first"
          })
    )
    return df


def save_outputs(df: pd.DataFrame):
    """Write CSV, Parquet, and profile JSON."""
    os.makedirs(os.path.dirname(PATHS["output_csv"]), exist_ok=True)
    os.makedirs(PATHS["quality_dir"], exist_ok=True)

    out_pq = PATHS["output_csv"].replace(".csv", ".parquet")
    df.to_parquet(out_pq, index=False, compression="snappy")
    df.to_csv(PATHS["output_csv"], index=False)
    print(f"✅ Saved {len(df):,} records → {out_pq}")

    prof = {
        "total_years": df["year"].nunique(),
        "years": sorted(df["year"].unique().tolist()),
        "total_counties": df["county_fips_full"].nunique(),
        "sources": df["source"].value_counts().to_dict(),
    }
    prof_path = os.path.join(PATHS["quality_dir"], "rent_profile.json")
    with open(prof_path, "w") as f:
        json.dump(prof, f, indent=4)
    print(f"📊 Profile saved → {prof_path}")


# =====================================
# MAIN
# =====================================
if __name__ == "__main__":
    print("🏠 Processing county-level rent data...")
    api_key = get_API_KEY()

    # Offline NHGIS decennial
    rent1980 = _load_nhgis_rent(os.path.join(ROOT, "data_1980.csv"), 1980)
    rent1990 = _load_nhgis_rent(os.path.join(ROOT, "median_rent_1990.csv"), 1990)

    # Census API (2000–2023)
    rent_api = get_county_rent(YEARS, api_key)

    # Combine & normalize
    rent_df = pd.concat([rent1980, rent1990, rent_api], ignore_index=True)
    rent_df["county_fips_full"] = rent_df["county_fips_full"].astype(str).str.zfill(5)
    rent_df = rent_df[["county_fips_full", "state_fips", "county_fips",
                       "median_gross_rent", "year", "source"]]
    print(f"Raw combined: {len(rent_df):,} rows")

    rent_df = deduplicate_rent(rent_df)
    print(f"After deduplication: {len(rent_df):,} rows")

    save_outputs(rent_df)
