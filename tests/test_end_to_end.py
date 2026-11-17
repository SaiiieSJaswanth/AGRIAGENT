"""
End-to-End System Tests for AGRIAGENT.
Verifies complete farmer workflow execution, 5 active modules, report serialization,
PDF generation, and strict exclusion of removed modules.
"""
import pytest
from pathlib import Path
from src.agr_agent.farmer_profile import FarmerProfile
from src.engine.orchestrator import run_agriagent
from src.reports.terminal_report import render_report
from src.reports.json_report import save_report
from src.reports.pdf_report import generate_farmer_pdf_report

def test_full_farm_advisory_execution(tmp_path):
    profile = FarmerProfile(
        state="Karnataka",
        district="Bagalkote",
        soil_type="Black Soil (Regur)",
        land_area_acres=5.0,
        current_crop="Maize",
        season="Kharif",
        irrigation_available=True,
        loan_required=True,
        insurance_required=True,
        nitrogen_kg_ha=110.0,
        phosphorus_kg_ha=45.0,
        potassium_kg_ha=120.0,
        soil_ph=7.2,
        organic_carbon_percent=0.62
    )

    report = run_agriagent(profile=profile)
    assert report is not None
    assert "farmer_profile" in report
    assert "location_intelligence" in report
    assert "crop_recommendation" in report
    assert "soil_health" in report
    assert "fertilizer_guidance" in report
    assert "irrigation_guidance" in report
    assert "support_matches" in report
    assert len(report["support_matches"]) > 0

    # Verify Module 1: Historical crop shortlist
    crop_rec = report["crop_recommendation"]
    assert crop_rec["status"] == "success"
    assert crop_rec["recommendation_type"] == "historical_regional_crop_shortlist"
    assert len(crop_rec["ranked_recommendations"]) > 0

    # Verify Module 2: Soil Health
    soil_res = report["soil_health"]
    assert "soil_fertility_class" in soil_res
    assert "parameter_evaluations" in soil_res

    # Verify Module 3: Fertilizer STCR & ML Regime
    fert_res = report["fertilizer_guidance"]
    assert "commercial_fertilizer_quantities" in fert_res
    assert "urea_kg" in fert_res["commercial_fertilizer_quantities"]
    assert "dap_kg" in fert_res["commercial_fertilizer_quantities"]
    assert "mop_kg" in fert_res["commercial_fertilizer_quantities"]

    # Verify Module 4: Irrigation Advisory
    irrig_res = report["irrigation_guidance"]
    assert "crop_water_requirement_mm" in irrig_res
    assert "critical_growth_stages" in irrig_res

    # Verify Removed Modules are NOT present in active report
    assert "crop_yield" not in report
    assert "mandi_prices" not in report
    assert "risk_assessment" not in report
    assert "ml_results" not in report

    # Verify JSON saving
    out_file = tmp_path / "test_report.json"
    save_report(report, str(out_file))
    assert out_file.exists()
    assert out_file.stat().st_size > 1000

    # Verify PDF generation
    pdf_file = tmp_path / "test_report.pdf"
    generate_farmer_pdf_report(report, str(pdf_file))
    assert pdf_file.exists()
    assert pdf_file.stat().st_size > 5000

def test_state_specific_scheme_filtering():
    ka_profile = FarmerProfile(state="Karnataka", district="Bagalkote", insurance_required=True, loan_required=True)
    ka_report = run_agriagent(profile=ka_profile)
    ka_schemes = {s["scheme_name"] for s in ka_report["support_matches"]}
    assert "Pradhan Mantri Fasal Bima Yojana (PMFBY)" in ka_schemes
    assert "PM-KISAN" in ka_schemes

    # Ensure state schemes for other states are not matched
    for s in ka_report["support_matches"]:
        if s.get("scheme_level") == "STATE":
            assert s.get("state") == "Karnataka", f"State scheme leak: {s['scheme_name']} for {s.get('state')} matched Karnataka farmer"
