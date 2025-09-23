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
    "input_csv": os.path.join(ROOT, "hpi_at_county.csv"),
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