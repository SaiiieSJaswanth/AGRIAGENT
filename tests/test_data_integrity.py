"""
Tests for Data Integrity, Source Manifests, and Security in AGRIAGENT.
"""
import os
import json
import pytest
import pandas as pd
from pathlib import Path

from src.config.settings import (
    RAW_REAL_DIR, PROCESSED_DIR,
    YIELD_DATA, SOIL_HEALTH_DATA, DISTRICT_RAINFALL_DATA,
    MANDI_PRICES_DATA, STATE_DISTRICT_MAP_PATH
)

def test_download_manifest_exists_and_valid():
    manifest_path = RAW_REAL_DIR / "download_manifest.json"
    assert manifest_path.exists(), "download_manifest.json must exist in raw/real_sources"
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "apy_crop_production.csv" in data
    assert "crop_recommendation_benchmark.csv" not in data
    assert "imd_district_rainfall_normal.csv" in data
    assert "soil_health_card_samples.csv" in data
    assert "agmarknet_mandi_prices_sample.csv" in data

def test_no_pickle_deserialization_in_download_script():
    download_script = Path("scripts/download_real_data.py")
    assert download_script.exists()
    content = download_script.read_text(encoding="utf-8")
    assert "import pickle" not in content, "Security vulnerability: 'import pickle' must not be in download script"
    assert "pd.read_pickle" not in content, "Security vulnerability: 'pd.read_pickle' must not be in download script"

def test_synthetic_crop_benchmarks_are_not_in_production():
    assert not Path("data/raw/dataset1.csv").exists()
    assert not Path("data/raw/real_sources/crop_recommendation_benchmark.csv").exists()
    assert not Path("data/processed/clean_crop_recommendation.csv").exists()
    assert not Path("models/crop/model.joblib").exists()

def test_clean_apy_yield_schema():
    assert YIELD_DATA.exists()
    df = pd.read_csv(YIELD_DATA)
    assert len(df) > 100000, "APY yield dataset must contain >100,000 cleaned records"
    for col in ["State_Name", "Crop", "Season", "Area", "Crop_Year", "Yield_tonnes_per_ha"]:
        assert col in df.columns, f"Missing expected column: {col}"
    assert (df["Area"] > 0).all(), "All records must have positive area"
    assert (df["Yield_tonnes_per_ha"] >= 0).all(), "Yield must be non-negative"

def test_district_rainfall_normals_schema():
    assert DISTRICT_RAINFALL_DATA.exists()
    df = pd.read_csv(DISTRICT_RAINFALL_DATA)
    assert len(df) >= 600, "IMD rainfall dataset should cover >= 600 districts"
    assert "ANNUAL" in df.columns
    assert (df["ANNUAL"] > 0).all(), "Annual rainfall normal must be positive"

def test_state_district_hierarchy_integrity():
    assert STATE_DISTRICT_MAP_PATH.exists()
    with open(STATE_DISTRICT_MAP_PATH, "r", encoding="utf-8") as f:
        hierarchy = json.load(f)
    assert len(hierarchy) >= 30, "Hierarchy should cover at least 30 Indian states and union territories"
    assert "Karnataka" in hierarchy
    assert "Maharashtra" in hierarchy
    assert len(hierarchy["Karnataka"]) >= 20, "Karnataka must have multiple districts"
