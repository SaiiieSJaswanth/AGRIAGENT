"""
Tests for Agronomic Rule Engines in AGRIAGENT.
Verifies ICAR-IISS soil fertility ratings (Option A), STCR fertilizer calculations,
irrigation water-balance assessment, and multi-dimensional risk scoring.
"""
import pytest
from src.models.soil_fertility import evaluate_soil_health, rate_ph, rate_ec
from src.models.fertilizer import calculate_fertilizer_guidance
from src.models.irrigation import evaluate_irrigation_needs
from src.models.risk import evaluate_comprehensive_risk

def test_soil_fertility_low_medium_high_ratings():
    # Test Low fertility profile
    low_res = evaluate_soil_health(ph=6.5, ec=0.3, oc=0.30, p=8.0, k=80.0)
    assert low_res["soil_fertility_class"] == "Low"
    assert "Organic Carbon (Low)" in low_res["deficiencies"]
    assert "Available Phosphorus (Low)" in low_res["deficiencies"]
    assert "Available Potassium (Low)" in low_res["deficiencies"]

    # Test Medium fertility profile
    med_res = evaluate_soil_health(ph=7.0, ec=0.4, oc=0.60, p=18.0, k=200.0)
    assert med_res["soil_fertility_class"] == "Medium"

    # Test High fertility profile
    high_res = evaluate_soil_health(ph=7.2, ec=0.5, oc=0.90, p=35.0, k=350.0)
    assert high_res["soil_fertility_class"] == "High"

def test_soil_ph_and_salinity_evaluations():
    acidic = rate_ph(5.2)
    assert "Acidic" in acidic["status"]
    assert "lime" in acidic["management"].lower()

    alkaline = rate_ph(8.8)
    assert "Alkaline" in alkaline["status"]
    assert "gypsum" in alkaline["management"].lower()

    saline = rate_ec(2.5)
    assert "Injurious" in saline["status"] or "Saline" in saline["status"]

def test_fertilizer_stcr_adjustments():
    # Low soil fertility should trigger +25% adjustment over base RDF
    low_guidance = calculate_fertilizer_guidance("rice", n_rating="Low", p_rating="Low", k_rating="Low")
    base_n = low_guidance["base_rdf_kg_ha"]["N"]
    rec_n = low_guidance["adjusted_recommendation_kg_ha"]["fertilizer_n_requirement_kg_ha"]
    assert rec_n == pytest.approx(base_n * 1.25, rel=1e-2)

    # High soil fertility should trigger 75% reduction
    high_guidance = calculate_fertilizer_guidance("rice", n_rating="High", p_rating="High", k_rating="High")
    rec_n_high = high_guidance["adjusted_recommendation_kg_ha"]["fertilizer_n_requirement_kg_ha"]
    assert rec_n_high == pytest.approx(base_n * 0.75, rel=1e-2)

def test_irrigation_needs_evaluation():
    # Low deficit scenario: Rice with abundant monsoon rainfall
    wet_res = evaluate_irrigation_needs("rice", season="Kharif", seasonal_rainfall_mm=1300.0)
    assert "Low" in wet_res["irrigation_requirement_class"] or "No immediate" in wet_res["irrigation_requirement_class"]

    # High deficit scenario: Sugarcane in dry season
    dry_res = evaluate_irrigation_needs("sugarcane", season="Summer", seasonal_rainfall_mm=100.0, irrigation_available=False)
    assert "High" in dry_res["irrigation_requirement_class"] or "Moderate" in dry_res["irrigation_requirement_class"]
    assert len(dry_res["alerts"]) > 0, "Should alert when high deficit crop has no irrigation"

def test_risk_scoring_dimensions():
    # Arid + Debt profile
    risk = evaluate_comprehensive_risk(
        annual_rainfall_mm=450.0,
        ph=8.8,
        ec=2.4,
        water_deficit_mm=400.0,
        irrigation_available=False,
        existing_loan=True,
        loan_required=True
    )
    assert risk["overall"] == "High"
    assert risk["dimensions"]["drought_risk"] == "High"
    assert risk["dimensions"]["financial_risk"] == "High"
    assert risk["dimensions"]["soil_degradation_risk"] == "High"
