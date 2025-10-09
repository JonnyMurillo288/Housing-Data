#!/usr/bin/env python3
import os
import json
import requests
import pandas as pd
from typing import Optional, List

YEARS = list(range(1980, 2024))
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
PATHS = {
    "output_csv": os.path.join(ROOT, "data", "processed", "home_value_at_county.csv"),
    "quality_dir": os.path.join(ROOT, "data", "quality"),
    "CENSUS_API_KEY": os.path.join(ROOT, "census_api.txt")
}

# --------------------
# CONFIG
# --------------------
HOME_VALUE_CONFIG = {
    2000: {"dataset": "2000/dec/sf3", "var": "H085001"},
}
for y in range(2005, 2009):
    HOME_VALUE_CONFIG[y] = {"dataset": f"{y}/acs/acs1", "var": "B25077_001E"}
for y in range(2009, 2024):
    HOME_VALUE_CONFIG[y] = {"dataset": f"{y}/acs/acs5", "var": "B25077_001E"}


# --------------------
# UTILITIES
# --------------------
def get_API_KEY() -> Optional[str]:
    if os.path.isfile(PATHS["CENSUS_API_KEY"]):
        with open(PATHS["CENSUS_API_KEY"]) as f:
            return f.read().strip()
    return None


def _load_nhgis(path: str, year: int, col_name: str) -> pd.DataFrame:
    """Generic NHGIS loader for 1980 & 1990."""
    df = pd.read_csv(path, dtype=str)
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    df["state_fips"] = df["GISJOIN"].str[1:3]
    df["county_fips"] = df["GISJOIN"].str[3:6]
    df["county_fips_full"] = df["state_fips"] + df["county_fips"]
    df["year"] = year
    df["source"] = f"nhgis_{year}"
    df = df.rename(columns={col_name: "median_home_value"})
    return df[["county_fips_full", "state_fips", "county_fips", "median_home_value", "year", "source"]]


def get_county_home_value(years: List[int], api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch county-level median home values from Census API for all given years."""
    all_results = []
    for year in years:
        if year not in HOME_VALUE_CONFIG:
            print(f"⚠️ No config for {year}, skipping.")
            continue

        cfg = HOME_VALUE_CONFIG[year]
        url = f"https://api.census.gov/data/{cfg['dataset']}"
        params = {"get": f"NAME,{cfg['var']}", "for": "county:*"}
        if api_key:
            params["key"] = api_key

        resp = requests.get(url, params=params)
        if resp.status_code == 204:
            print(f"⚠️ No data for {year}")
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
        df["county_fips_full"] = df["state_fips"].str.zfill(2) + df["county_fips"].str.zfill(3)
        df["year"] = year
        df["source"] = cfg["dataset"]
        all_results.append(df)

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    return combined


def deduplicate_home_values(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one home value per county-year."""
    # Prefer ACS5 over ACS1 if both exist
    df = df.sort_values(["county_fips_full", "year", "source"])
    df["source_priority"] = df["source"].apply(
        lambda s: 1 if "acs5" in s else (2 if "acs1" in s else 3)
    )
    df = df.sort_values(["county_fips_full", "year", "source_priority"])
    df = df.drop_duplicates(subset=["county_fips_full", "year"], keep="first")

    # Aggregate safety check if still multiple
    df = (
        df.groupby(["county_fips_full", "year"], as_index=False)
        .agg({
            "median_home_value": "mean",
            "state_fips": "first",
            "county_fips": "first",
            "source": "first"
        })
    )
    return df


def save_outputs(df: pd.DataFrame):
    os.makedirs(os.path.dirname(PATHS["output_csv"]), exist_ok=True)
    os.makedirs(PATHS["quality_dir"], exist_ok=True)

    out_pq = PATHS["output_csv"].replace(".csv", ".parquet")
    df.to_parquet(out_pq, index=False, compression="snappy")
    df.to_csv(PATHS["output_csv"], index=False)
    print(f"✅ Saved {len(df)} rows to {out_pq}")

    profile = {
        "total_years": df["year"].nunique(),
        "years": sorted(df["year"].unique().tolist()),
        "total_counties": df["county_fips_full"].nunique(),
        "sources": df["source"].value_counts().to_dict(),
    }
    prof_path = os.path.join(PATHS["quality_dir"], "home_value_profile.json")
    with open(prof_path, "w") as f:
        json.dump(profile, f, indent=4)
    print(f"📋 Profile saved to {prof_path}")


# --------------------
# MAIN
# --------------------
if __name__ == "__main__":
    print("🏡 Processing median home value data…")
    api_key = get_API_KEY()

    home1980 = _load_nhgis(os.path.join(ROOT, "data_1980.csv"), 1980, "C8J001")
    home1990 = _load_nhgis(os.path.join(ROOT, "median_home_value_1990.csv"), 1990, "FCL001")
    api_df = get_county_home_value(YEARS, api_key)

    full = pd.concat([home1980, home1990, api_df], ignore_index=True)
    full["county_fips_full"] = full["county_fips_full"].astype(str).str.zfill(5)
    full = full[["county_fips_full", "state_fips", "county_fips", "median_home_value", "year", "source"]]

    print(f"Raw combined: {len(full):,} rows")
    full = deduplicate_home_values(full)
    print(f"After deduplication: {len(full):,} rows")

    save_outputs(full)
