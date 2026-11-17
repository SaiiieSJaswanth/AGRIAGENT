"""
Tests for Production Machine Learning Models in AGRIAGENT.
Verifies loading, inference pipelines, and metadata for all 4 verified models:
1. Crop Recommendation Shadow ML (Random Forest)
2. Soil Health Assessment ML (Random Forest, zero target leakage)
3. Fertilizer Regime ML (Random Forest)
4. Irrigation Schedule ML (Decision Tree)
"""
import json
import pytest
from pathlib import Path

from src.engine.inference import (
    get_model, recommend_crops_from_history, predict_crop_recommendation_ml_hidden
)
from src.models.soil_fertility import predict_soil_health_ml
from src.models.fertilizer import predict_fertilizer_ml_regime
from src.models.irrigation import predict_irrigation_ml_class
from src.config.settings import MODEL_DIR

def test_crop_shortlist_uses_historical_local_records():
    res = recommend_crops_from_history(
        state="Karnataka", district="Bagalkote", season="Kharif", top_k=5
    )
    assert res["status"] == "success"
    assert res["recommendation_type"] == "historical_regional_crop_shortlist"
    assert res["top_crop"]
    assert 1 <= len(res["ranked_recommendations"]) <= 5
    assert "2015" in res["dataset_provenance"]
    assert "confidence_percent" not in res["ranked_recommendations"][0]
    assert res["evidence_scope"] in {
        "district_and_season", "district_all_recorded_seasons",
        "state_and_season_fallback", "state_all_recorded_seasons_fallback"
    }

def test_crop_shortlist_validates_inputs():
    with pytest.raises(ValueError):
        recommend_crops_from_history(state="", district="Bagalkote", season="Kharif")
    with pytest.raises(ValueError):
        recommend_crops_from_history(state="Karnataka", district="Bagalkote", season="Kharif", top_k=0)

def test_crop_shadow_ml_model_loads_and_predicts():
    model = get_model("crop_recommendation")
    assert model is not None, "Crop shadow ML model must load successfully"

    res = predict_crop_recommendation_ml_hidden(
        state="Karnataka",
        district="Bagalkote",
        season="Kharif",
        crop_year=2015
    )
    assert res["status"] == "success"
    assert "predicted_crop" in res
    assert res["predicted_crop"] is not None

def test_crop_shadow_ml_metadata():
    meta_path = MODEL_DIR / "crop_recommendation" / "metadata.json"
    assert meta_path.exists()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta.get("model") == "RandomForestClassifier" or meta.get("model_type") == "RandomForestClassifier"
    assert "test_metrics" in meta
    assert meta["test_metrics"]["accuracy"] > 0.50

def test_soil_health_ml_loads_and_predicts():
    model = get_model("soil_health")
    assert model is not None, "Soil health model must load successfully"

    res = predict_soil_health_ml(
        ph=6.8, ec=0.35, s=12.0, zn=0.8, fe=5.0, mn=3.0, cu=0.4
    )
    assert res["status"] == "success"
    assert res["predicted_fertility_class"] in ["Low", "Medium", "High"]

def test_soil_health_ml_metadata():
    meta_path = MODEL_DIR / "soil_health" / "metadata.json"
    assert meta_path.exists()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta.get("model") == "RandomForestClassifier" or meta.get("model_type") == "RandomForestClassifier"
    assert "test_metrics" in meta
    assert meta["test_metrics"]["accuracy"] > 0.70

def test_fertilizer_ml_loads_and_predicts():
    model = get_model("fertilizer")
    assert model is not None, "Fertilizer model must load successfully"

    res = predict_fertilizer_ml_regime(
        ph=6.8, ec=0.35, oc=0.60, p_soil=16.0, k_soil=180.0,
        rec_n=100.0, rec_p=50.0, rec_k=50.0
    )
    assert res["status"] == "success"
    assert "predicted_regime" in res
    assert res["predicted_regime"] is not None

def test_fertilizer_ml_metadata():
    meta_path = MODEL_DIR / "fertilizer" / "metadata.json"
    assert meta_path.exists()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta.get("model") == "RandomForestClassifier" or meta.get("model_type") == "RandomForestClassifier"
    assert "test_metrics" in meta
    assert meta["test_metrics"]["accuracy"] > 0.90

def test_irrigation_ml_loads_and_predicts():
    model = get_model("irrigation")
    assert model is not None, "Irrigation decision tree must load successfully"

    res = predict_irrigation_ml_class(
        soil_cap=150.0, kc=0.85, crop_water_req=600.0,
        rainfall_norm=750.0, water_deficit=50.0
    )
    assert res["status"] == "success"
    assert "predicted_class" in res
    assert res["predicted_class"] is not None

def test_irrigation_ml_metadata():
    meta_path = MODEL_DIR / "irrigation" / "metadata.json"
    assert meta_path.exists()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta.get("model") == "DecisionTreeClassifier" or meta.get("model_type") == "DecisionTreeClassifier"
    assert "test_metrics" in meta
    assert meta["test_metrics"]["accuracy"] > 0.90
