#!/usr/bin/env python3
import os
import sys
import re
import json
import difflib
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd, YearEnd

# Try geopandas; handle environments without it.
try:
    import geopandas as gpd
except Exception as e:  # noqa: F841
    gpd = None


# -----------------------------
# Config
# -----------------------------
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
PATHS = {
    "input_csv": os.path.join(ROOT, "data",'processed',"income_hpi_home_rent_at_county.parquet"),
    "shapefiles_dir": os.path.join(ROOT, "shapefiles"),
    "processed_dir": os.path.join(ROOT, "data", "processed"),
    "geo_dir": os.path.join(ROOT, "data", "geo"),
    "quality_dir": os.path.join(ROOT, "data", "quality"),
    "fig_maps_dir": os.path.join(ROOT, "figures", "maps"),
}

TARGET_CRS = "EPSG:4326"

'''
This file is going to take in the combined income and house price data at the county level
and perform some data quality checks, 
- Create Housing affordability index (HAI)
    - Ideally we can get median HH income and median home price for the same year
    - Or if we can't get median Home Price and are stuck with HPI we can transform the HH Income into an index as well
    - HAI = (Median Household Income / Median Home Price) * 100
    - IndexedHAI = (Indexed Median Household Income / Indexed Median Home Price) * 100
    - We can use the base year of the HPI as the base year for the income index

'''

def get_period_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''function for finding the period columns of a dataframe'''
    return

def data_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    # Check for missing values
    missing_values = df.isnull().sum()
    print("Missing values in each column:\n", missing_values)

    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")

    # Check data types
    print("Data types:\n", df.dtypes)

    # Additional checks can be added here

    return df

def get_monthly_income(df: pd.DataFrame) -> pd.DataFrame:
    # Assuming df has a column 'median_household_income' which is annual income
    # This function will convert it to monthly income so we can compare to rent in an understandble number
    if 'median_household_income' not in df.columns:
        raise ValueError("DataFrame must contain 'median_household_income' column")

    # Convert annual income to monthly income
    df['median_monthly_income'] = df['median_household_income'] / 12

    return df

def calculate_hai(df: pd.DataFrame) -> pd.DataFrame:
    # Assuming df has columns 'median_household_income' and 'median_home_price'
    if 'median_household_income' not in df.columns or 'median_home_value' not in df.columns:
        raise ValueError("DataFrame must contain 'median_household_income' and 'median_home_value' columns")

    # Calculate HAI
    df['HAI'] = (df['median_home_value'] / df['median_household_income'])

    # Handle potential division by zero or NaN values
    df['HAI'].replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def calculate_rai(df: pd.DataFrame) -> pd.DataFrame:
    # Assuming df has columns 'median_gross_rent' and 'median_home_price'
    if 'median_gross_rent' not in df.columns or 'median_monthly_income' not in df.columns:
        raise ValueError("DataFrame must contain 'median_gross_rent' and 'median_home_value' columns")

    # Calculate RAI
    df['RAI'] = (df['median_monthly_income'] / df['median_gross_rent'])

    # Handle potential division by zero or NaN values
    df['RAI'].replace([np.inf, -np.inf], np.nan, inplace=True)

    return df

def calculate_indexed_hai(df: pd.DataFrame) -> pd.DataFrame:
    df['HAI_Index'] = (df['median_household_income_indexed'] / df['hpi_value_indexed']) * 100
    
    # Handle potential division by zero or NaN values
    df['HAI_Index'].replace([np.inf, -np.inf], np.nan, inplace=True)

    return df

    
def index_variable(df: pd.DataFrame, variable_to_index:str, groups_column: Optional[str]) -> pd.DataFrame:
    ''' This function will index a variable based on a start year 
    INPUTS:
    
    variable_to_index - column name of value to index
    start_year - year to start the index
    
    '''
    if df[variable_to_index].dtypes not in [int,float]:
        raise ValueError(f'{variable_to_index} not a numeric')
    
    year_col = 'year' # TODO: Change to find the period column function when I need this functionality
    if "Year" in df.columns.tolist():   
        year_col = "Year"

    df['base_value'] = df.sort_values(year_col).groupby(groups_column)[variable_to_index].transform("first")
    df[f'{variable_to_index}_indexed'] = df[variable_to_index] / df['base_value']
    return df    

def main():
    # Load data
    # df = pd.read_csv(PATHS["input_csv"], dtype={"fips": str})
    df = pd.read_parquet(PATHS["input_csv"])
    print(f"Data loaded with shape: {df.shape}")
    
    # Data quality checks
    # df = data_quality_checks(df)

    # Calculate HAI
    # df = calculate_hai(df)
    print(df.columns)
    df = index_variable(df,'median_household_income','county_fips_full')
    # df = index_variable(df,'hpi_value','county_fips_full') # do we need to create a standardized hpi value as well? This is so we can see relatively how out of line the growth in HH income is over the growth in HPI
    # Because we have in 1990 median home price
    # We have in 1990 median income. Why not compare that over this time period instead of the full as weird assumptions fuck
    
    # df = calculate_indexed_hai(df)
    df = get_monthly_income(df)
    df = calculate_rai(df)
    df = calculate_hai(df)
    
    print(df.columns)

    # Save processed data
    processed_path = os.path.join(PATHS["processed_dir"], "hpi_income_metrics_processed.csv")
    df.to_csv(processed_path, index=False)
    pq_out = os.path.join(PATHS['processed_dir'],'hpi_income_metrics.parquet')
    df.to_parquet(pq_out)
    print(f"Processed data saved to {processed_path}")
    
if __name__ == '__main__':
    main()