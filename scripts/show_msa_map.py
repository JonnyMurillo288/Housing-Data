#!/usr/bin/env python3
import os
import sys
import argparse
import json
from typing import List, Optional, Tuple

import numpy as np

# Try optional libs
try:
    import geopandas as gpd  # type: ignore
except Exception:
    gpd = None  # type: ignore

try:
    import folium  # type: ignore
    from branca.colormap import LinearColormap  # type: ignore
except Exception as e:
    print("This script requires folium and branca. Install via: pip install folium branca")
    sys.exit(1)


ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_GEOJSON = os.path.join(ROOT, "data", "geo", "msa_timeseries_latest.geojson")
DEFAULT_OUTPUT = os.path.join(ROOT, "figures", "maps", "msa_choropleth_latest.html")

PREFERRED_VALUE_COLS = [
    "index_base100",
    "index_sa",
    "index_nsa",
    "yoy",
    "chg1",
]


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def load_geo(path: str):
    if gpd is not None:
        try:
            return gpd.read_file(path)
        except Exception as e:
            print(f"Failed to read GeoJSON with geopandas: {e}. Falling back to json library.")
    with open(path, "r") as f:
        gj = json.load(f)
    # Minimal shim to work like a GeoDataFrame for this script
    features = gj.get("features", [])
    props = [feat.get("properties", {}) for feat in features]
    return props  # list of dicts


def is_numeric(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def detect_value_column(props: List[dict], user_choice: Optional[str] = None) -> Optional[str]:
    if not props:
        return None
    keys = set().union(*[set(p.keys()) for p in props])
    # User-specified
    if user_choice and user_choice in keys:
        # ensure it's numeric enough
        vals = [p.get(user_choice) for p in props]
        nums = [float(v) for v in vals if v is not None and is_numeric(v)]
        if len(nums) > 0:
            return user_choice
    # Preferred list
    for k in PREFERRED_VALUE_COLS:
        if k in keys:
            vals = [p.get(k) for p in props]
            nums = [float(v) for v in vals if v is not None and is_numeric(v)]
            if len(nums) > 0:
                return k
    # Fallback: first numeric-ish column
    for k in keys:
        vals = [p.get(k) for p in props]
        nums = [float(v) for v in vals if v is not None and is_numeric(v)]
        if len(nums) > 0 and k not in ("cbsa_code", "msa_name"):
            return k
    return None


def extract_values(props: List[dict], value_col: str) -> np.ndarray:
    vals = []
    for p in props:
        v = p.get(value_col)
        try:
            v = float(v)
        except Exception:
            v = np.nan
        vals.append(v)
    return np.array(vals, dtype=float)


def build_map(geo_path: str, output_html: str, value_col: Optional[str] = None,
              center: Tuple[float, float] = (39.5, -98.35), zoom_start: int = 4):
    # Load as GeoDataFrame if available for easier binding
    gdf = None
    props_only = None
    try:
        if gpd is not None:
            gdf = gpd.read_file(geo_path)
        else:
            props_only = load_geo(geo_path)
    except Exception:
        props_only = load_geo(geo_path)

    # Establish properties list for column detection
    if gdf is not None:
        props = gdf.drop(columns=[c for c in gdf.columns if c == "geometry"])  # type: ignore
        props = props.to_dict(orient="records")  # type: ignore
    else:
        props = props_only or []

    if not props:
        print("No features in GeoJSON to render.")
        sys.exit(2)

    # Determine value column
    val_col = detect_value_column(props, user_choice=value_col)
    if val_col is None:
        print("Could not determine a numeric value column to color. Provide --value-col explicitly.")
        print(f"Available keys: {sorted(set().union(*[set(p.keys()) for p in props]))}")
        sys.exit(3)

    # Compute color scale bounds (robust to outliers)
    values = extract_values(props, val_col)
    # handle all-NaN
    if np.all(np.isnan(values)):
        print(f"Column '{val_col}' contains no numeric values.")
        sys.exit(4)

    vmin = np.nanpercentile(values, 2)
    vmax = np.nanpercentile(values, 98)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0

    cmap = LinearColormap(
        colors=["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
        vmin=vmin,
        vmax=vmax,
    )
    cmap.caption = f"{val_col}"

    m = folium.Map(location=center, zoom_start=zoom_start, tiles="cartodbpositron")

    # Load raw geojson dict for folium
    with open(geo_path, "r") as f:
        gj = json.load(f)

    def style_fn(feature):
        p = feature.get("properties", {})
        v = p.get(val_col)
        try:
            v = float(v)
        except Exception:
            v = None
        color = cmap(v) if v is not None and np.isfinite(v) else "#cccccc"
        return {
            "fillColor": color,
            "color": "#555555",
            "weight": 0.5,
            "fillOpacity": 0.8,
        }

    # Determine tooltip fields available
    sample_keys = set(props[0].keys())
    tooltip_fields = [k for k in ["cbsa_code", "msa_name", val_col, "yoy", "chg1"] if k in sample_keys]
    tooltip_aliases = [
        "CBSA", "MSA Name", val_col.upper(), "YoY", "Change (1)"
    ][: len(tooltip_fields)]

    gj_layer = folium.GeoJson(
        gj,
        name=f"MSA Choropleth ({val_col})",
        style_function=style_fn,
        highlight_function=lambda x: {"weight": 2, "color": "#000", "fillOpacity": 0.9},
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            sticky=True,
            localize=True,
        ) if tooltip_fields else None,
    )
    gj_layer.add_to(m)

    cmap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    ensure_dir(output_html)
    m.save(output_html)
    print(f"Map saved to: {output_html}")


def parse_args():
    p = argparse.ArgumentParser(description="Render interactive MSA choropleth from GeoJSON")
    p.add_argument("--geo", default=DEFAULT_GEOJSON, help="Path to GeoJSON input (default: data/geo/msa_timeseries_latest.geojson)")
    p.add_argument("--value-col", default=None, help="Property name to color by (default: auto-detect, prefers index_base100)")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output HTML path (default: figures/maps/msa_choropleth_latest.html)")
    p.add_argument("--center", default="39.5,-98.35", help="Map center lat,lon (default: 39.5,-98.35)")
    p.add_argument("--zoom", type=int, default=4, help="Initial zoom level (default: 4)")
    args = p.parse_args()

    try:
        lat, lon = [float(x) for x in args.center.split(",")]
    except Exception:
        lat, lon = 39.5, -98.35
    return args, (lat, lon)


if __name__ == "__main__":
    args, center = parse_args()
    if not os.path.isfile(args.geo):
        print(f"GeoJSON not found: {args.geo}\nRun the pipeline first: python3 scripts/msa_pipeline.py")
        sys.exit(1)
    build_map(args.geo, args.output, value_col=args.value_col, center=center, zoom_start=args.zoom)
