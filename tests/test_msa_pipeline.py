import os
import json
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# -----------------------------
# Helpers
# -----------------------------

def reimport_pipeline():
    """Load the pipeline module directly from its file path to avoid package import issues."""
    import sys
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "msa_pipeline.py"
    spec = importlib.util.spec_from_file_location("msa_pipeline_under_test", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to create module spec for msa_pipeline")
    mp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mp)
    return mp


def set_paths_to_tmp(mp, tmp_path: Path):
    """Patch PATHS in the module to point to a temp directory tree."""
    root = tmp_path
    paths = {
        "input_csv": str(root / "hpi_master.csv"),
        "shapefiles_dir": str(root / "shapefiles"),
        "processed_dir": str(root / "data" / "processed"),
        "geo_dir": str(root / "data" / "geo"),
        "quality_dir": str(root / "data" / "quality"),
        "fig_maps_dir": str(root / "figures" / "maps"),
    }
    # ensure dirs exist that are directories
    (root / "shapefiles").mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (root / "data" / "geo").mkdir(parents=True, exist_ok=True)
    (root / "data" / "quality").mkdir(parents=True, exist_ok=True)
    (root / "figures" / "maps").mkdir(parents=True, exist_ok=True)

    # Patch module constants
    mp.PATHS = paths
    return paths


# -----------------------------
# Unit tests for utilities
# -----------------------------

def test_standardize_and_snake_case():
    mp = reimport_pipeline()
    df = pd.DataFrame(columns=["CBSA Code", "MSA-Name", "Yr "])
    out = mp.standardize_columns(df)
    assert list(out.columns) == ["cbsa_code", "msa_name", "yr"]


def test_normalize_cbsa_code_various():
    mp = reimport_pipeline()
    s = pd.Series(["123", "01234", "abc", None, "12345-678", "  9876 "])  # note: 9876 -> 09876
    out = mp.normalize_cbsa_code(s)
    assert out.iloc[0] == "00123"
    assert out.iloc[1] == "01234"
    assert pd.isna(out.iloc[2])
    assert pd.isna(out.iloc[3])
    assert out.iloc[4] == "12345"
    assert out.iloc[5] == "09876"


def test_construct_period_from_components_month_quarter_year():
    mp = reimport_pipeline()
    df = pd.DataFrame(
        {
            "frequency": ["monthly", "quarterly", "annual"],
            "yr": [2021, 2020, 2019],
            "period": [2, 1, 1],
        }
    )
    dt = mp.construct_period_from_components(df, "frequency", "yr", "period")
    # monthly: 2021-02-01
    assert str(pd.to_datetime(dt.iloc[0]).date()) == "2021-02-01"
    # quarterly (Q1 -> March end)
    assert str(pd.to_datetime(dt.iloc[1]).date()) == "2020-03-31"
    # annual -> Dec 31
    assert str(pd.to_datetime(dt.iloc[2]).date()) == "2019-12-31"


def test_choose_primary_metric_prefers_index_sa():
    mp = reimport_pipeline()
    df = pd.DataFrame(
        {
            "cbsa_code": ["12345"],
            "msa_name": ["City"],
            "frequency": ["monthly"],
            "yr": [2020],
            "period": [1],
            "index_sa": [100.0],
            "index_nsa": [200.0],
            "hpi": [300.0],
            "non_numeric": ["x"],
        }
    )
    metric = mp.choose_primary_metric(df, exclude_cols=["cbsa_code", "msa_name", "frequency", "yr", "period"])
    assert metric == "index_sa"


def test_compute_derived_and_yoy_monthly():
    mp = reimport_pipeline()
    df = pd.DataFrame(
        {
            "cbsa_code": ["11111", "11111"],
            "period": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "value": [100.0, 110.0],
        }
    )
    derived = mp.compute_derived_metrics(df.rename(columns={"value": "val"}), "cbsa_code", "period", "val")
    # index_base100 should be 100 then 110/100*100=110
    assert list(np.round(derived["index_base100"], 2)) == [100.0, 110.0]
    # chg1 should be NaN then 0.1
    assert pd.isna(derived.loc[0, "chg1"]) and pytest.approx(derived.loc[1, "chg1"], rel=1e-6) == 0.1

    # Now monthly YoY (12 periods). With only 2 rows, both NaN remains.
    added = mp.add_yoy_by_freq(derived, "cbsa_code", "period", "val", "M")
    assert added["yoy"].isna().all()


# -----------------------------
# Integration-style tests for main()
# -----------------------------

def test_main_missing_input_csv_exits_1(tmp_path, monkeypatch):
    mp = reimport_pipeline()
    set_paths_to_tmp(mp, tmp_path)
    # Ensure geopandas branch is skipped cleanly regardless
    mp.gpd = None
    with pytest.raises(SystemExit) as exc:
        mp.main()
    assert exc.value.code == 1


def test_main_missing_time_columns_exits_2_and_writes_qa(tmp_path, monkeypatch):
    mp = reimport_pipeline()
    paths = set_paths_to_tmp(mp, tmp_path)
    mp.gpd = None

    # Create minimal CSV without required time columns
    pd.DataFrame(
        {
            "cbsa": ["12345", "23456"],
            "place_name": ["Metro A", "Metro B"],
            "some_metric": ["1", "2"],
        }
    ).to_csv(paths["input_csv"], index=False)

    with pytest.raises(SystemExit) as exc:
        mp.main()
    assert exc.value.code == 2

    qa_path = Path(paths["quality_dir"]) / "qa_error.json"
    assert qa_path.exists()
    qa = json.loads(qa_path.read_text())
    assert qa.get("error") == "missing_time_columns"
    assert "profile" in qa and isinstance(qa["profile"], dict)


def test_main_minimal_success_without_shapefile(tmp_path):
    mp = reimport_pipeline()
    paths = set_paths_to_tmp(mp, tmp_path)

    # Force geopandas unavailable to exercise the missing shapefile path cleanly
    mp.gpd = None

    # Build a minimal, valid monthly dataset
    data = pd.DataFrame(
        {
            "cbsa": ["12345", "12345", "23456", "23456"],
            "place_name": ["Metro A", "Metro A", "Metro B", "Metro B"],
            "frequency": ["monthly", "monthly", "monthly", "monthly"],
            "yr": [2020, 2020, 2020, 2020],
            "period": [1, 2, 1, 2],
            "index_sa": [100, 110, 200, 210],
        }
    )
    data.to_csv(paths["input_csv"], index=False)

    # Run pipeline
    mp.main()

    # Verify processed outputs exist (parquet if available, else csv fallback)
    proc = Path(paths["processed_dir"])    
    long_parq = proc / "msa_timeseries_long.parquet"
    long_csv = proc / "msa_timeseries_long.csv"
    wide_parq = proc / "msa_timeseries_wide.parquet"
    wide_csv = proc / "msa_timeseries_wide.csv"
    latest_csv = proc / "msa_latest.csv"

    assert latest_csv.exists(), "latest CSV not produced"
    assert long_parq.exists() or long_csv.exists(), "long dataset not produced"
    assert wide_parq.exists() or wide_csv.exists(), "wide dataset not produced"

    # Quality profile exists
    prof_path = Path(paths["quality_dir"]) / "profile.json"
    assert prof_path.exists()
    prof = json.loads(prof_path.read_text())
    assert prof.get("metric_col") == "index_sa"
    assert "freq_aliases" in prof and "M" in prof["freq_aliases"]

    # Geometry phase should be skipped and unmatched_geometry.json written
    unmatched = Path(paths["quality_dir"]) / "unmatched_geometry.json"
    assert unmatched.exists()
    meta = json.loads(unmatched.read_text())
    assert meta.get("missing_shapefile") is True
