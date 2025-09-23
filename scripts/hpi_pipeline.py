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

''' This file is going to process the HPI data at the county level
Some issues that we will need to solve is:
- Ensuring the correct merge with the counties TIGER shapefile
- Ensuring the correct date format
- Ensuring the correct data types
- For each county, the base year index is 100, 
    - We need to apply to each year, the median home price to that county by looking it up
    - OOr we need a standardized 100 for the latest year available, and then scale all other years accordingly
    - Or we ignore that fact and let the data be based on a certain year
    - 
'''