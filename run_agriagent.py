"""
AGRIAGENT Interactive CLI Application
Provides intelligent real-data agricultural recommendations, soil fertility ratings,
crop-specific fertilizer doses, irrigation schedules, and government scheme guidance.
Generates terminal display, structured JSON report, and printable vector PDF report.
"""
import json
from pathlib import Path
from typing import List, Optional

from src.config.settings import (
    STATE_DISTRICT_MAP_PATH, STATE_CROP_CATALOG_PATH, REPORT_DIR
)
from src.data.crop_catalog import load_state_crop_catalog, crops_for_state
from src.agr_agent.farmer_profile import FarmerProfile
from src.engine.orchestrator import run_agriagent
from src.reports.terminal_report import render_report
from src.reports.json_report import save_report
from src.reports.pdf_report import generate_farmer_pdf_report

def load_hierarchy():
    if Path(STATE_DISTRICT_MAP_PATH).exists():
        with open(STATE_DISTRICT_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def choose_from_list(title: str, options: List[str]) -> str:
    options = list(options)
    print(f"\n{title}")
    print("-" * 58)
    for i, value in enumerate(options, 1):
        print(f"  {i:>2}. {value}")
    while True:
        raw = input(f"\nSelect {title.lower()} [1-{len(options)}]: ").strip()
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        except ValueError:
            pass
        for opt in options:
            if raw.lower() == opt.lower():
                return opt
        print(f"Invalid selection. Please enter a number between 1 and {len(options)}, or the exact name.")

def optional_float(prompt: str, min_val: float = 0.0, max_val: float = 100000.0) -> Optional[float]:
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return None
        try:
            value = float(raw)
            if min_val <= value <= max_val:
                return value
            print(f"Value must be between {min_val} and {max_val}.")
        except ValueError:
            print("Please enter a valid numeric value, or press Enter to skip.")

def yes_no(prompt: str, default: bool = True) -> bool:
    d_str = "Y/n" if default else "y/N"
    while True:
        v = input(f"{prompt} [{d_str}]: ").strip().lower()
        if v == "":
            return default
        if v in {"y", "yes"}:
            return True
        if v in {"n", "no"}:
            return False
        print("Please enter y or n.")

def main():
    print("=" * 68)
    print("                    AGRIAGENT")
    print("      Intelligent Farm Advisory & ML Decision Support System")
    print("=" * 68)
    print("Empowering Indian farmers with genuine agronomic data, verified")
    print("climatological normals, machine learning models, and welfare access.\n")

    hierarchy = load_hierarchy()
    states = sorted(list(hierarchy.keys()))
    if not states:
        states = ["Andhra Pradesh", "Karnataka", "Maharashtra", "Punjab", "Tamil Nadu", "Uttar Pradesh", "West Bengal"]

    # 1. State Selection
    state = choose_from_list("STATE / UNION TERRITORY", states)

    # Display Common State Crops from Ministry Catalog
    crop_catalog = load_state_crop_catalog(STATE_CROP_CATALOG_PATH) if Path(STATE_CROP_CATALOG_PATH).exists() else []
    state_crops = crops_for_state(crop_catalog, state)
    if state_crops:
        print("\nMAJOR CROPS GROWN IN THIS STATE (Ministry Catalog)")
        print("-" * 58)
        for i, item in enumerate(state_crops[:8], 1):
            print(f"  {i:>2}. {item['crop']} [{item.get('crop_category', 'General')}]")

    # 2. District Selection (Restricted strictly to the selected state)
    districts = hierarchy.get(state, ["General"])
    district = choose_from_list(f"DISTRICT IN {state.upper()}", districts)

    # 3. Soil Type Selection
    soil_types = [
        "Alluvial Soil", "Black Soil (Regur)", "Red Soil",
        "Laterite Soil", "Desert / Arid Soil", "Peaty / Marshy Soil", "Loamy Soil"
    ]
    soil_type = choose_from_list("PRIMARY SOIL TYPE", soil_types)

    # 4. Season Selection
    seasons = ["Kharif (Monsoon)", "Rabi (Winter)", "Whole Year / Annual", "Summer (Zaid)"]
    season_raw = choose_from_list("AGRICULTURAL SEASON", seasons)
    season = {
        "Kharif (Monsoon)": "Kharif",
        "Rabi (Winter)": "Rabi",
        "Whole Year / Annual": "Whole Year",
        "Summer (Zaid)": "Summer",
    }[season_raw]

    # 5. Crop of Interest
    print("\nCROP OF INTEREST (Optional)")
    print("-" * 58)
    crop_input = input("Enter crop (e.g. Rice, Wheat, Maize, Cotton) or press Enter for ML recommendation: ").strip()
    current_crop = crop_input.title() if crop_input else None

    # 6. Farm Scale & Irrigation
    print("\nFARM DETAILS")
    print("-" * 58)
    area_acres = optional_float("Landholding in acres [Default 2.5]: ", 0.01, 10000.0) or 2.5
    irrigation_available = yes_no("Do you have assured irrigation access (well, tube-well, canal)?", default=True)

    # 7. Optional Soil Test Card Measurements
    print("\nSOIL TEST VALUES (Optional - Press Enter to use regional typical values)")
    print("-" * 58)
    ph_val = optional_float("  Soil pH [e.g. 6.5 - 7.5]: ", 3.0, 11.0)
    oc_val = optional_float("  Organic Carbon % [e.g. 0.50 - 0.75%]: ", 0.05, 5.0)
    n_val = optional_float("  Available Nitrogen (kg/ha) [e.g. 150 - 280]: ", 10.0, 500.0)
    p_val = optional_float("  Available Phosphorus (kg/ha) [e.g. 10 - 25]: ", 1.0, 150.0)
    k_val = optional_float("  Available Potassium (kg/ha) [e.g. 108 - 280]: ", 10.0, 600.0)

    # 8. Government Support & Scheme Information
    print("\nGOVERNMENT SUPPORT & WELFARE INFORMATION")
    print("-" * 58)
    ownership = input("Land ownership (owned / leased / sharecropper) [Default 'owned']: ").strip() or "owned"
    farmer_cat = input("Farmer category (small/marginal/large/SC/ST/women) [optional]: ").strip() or None
    loan_req = yes_no("Do you require agricultural credit / KCC loan support?", default=False)
    existing_loan = yes_no("Do you currently have an outstanding agricultural bank loan?", default=False)
    equip_req = yes_no("Do you require farm machinery / equipment subsidy?", default=False)
    insure_req = yes_no("Do you require crop insurance (PMFBY)?", default=True)
    subsidy_req = yes_no("Do you require organic / fertilizer subsidies?", default=False)

    profile = FarmerProfile(
        state=state,
        district=district,
        soil_type=soil_type,
        land_area_acres=area_acres,
        land_ownership=ownership,
        farmer_category=farmer_cat,
        current_crop=current_crop,
        season=season,
        irrigation_available=irrigation_available,
        existing_agricultural_loan=existing_loan,
        loan_required=loan_req,
        equipment_required=equip_req,
        irrigation_required=not irrigation_available,
        insurance_required=insure_req,
        subsidy_required=subsidy_req,
        nitrogen_kg_ha=n_val,
        phosphorus_kg_ha=p_val,
        potassium_kg_ha=k_val,
        soil_ph=ph_val,
        organic_carbon_percent=oc_val
    )

    print("\nExecuting AGRIAGENT 5-module real-evidence intelligence pipeline...")
    result = run_agriagent(profile=profile)
    render_report(result)

    # Save JSON Report
    from src.config.settings import OUTPUT_DIR
    predictions_dir = REPORT_DIR / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = REPORT_DIR / "latest_agriagent_report.json"
    save_report(result, str(json_path))
    save_report(result, str(predictions_dir / "latest_agriagent_report.json"))
    save_report(result, str(OUTPUT_DIR / "latest_agriagent_report.json"))

    # Save Printable Vector PDF Report
    pdf_path = REPORT_DIR / "agriagent_farmer_report.pdf"
    generate_farmer_pdf_report(result, str(pdf_path))
    generate_farmer_pdf_report(result, str(predictions_dir / "agriagent_farmer_report.pdf"))
    generate_farmer_pdf_report(result, str(OUTPUT_DIR / "agriagent_farmer_report.pdf"))

    print("\n" + "=" * 70)
    print("SUCCESS: Reports generated successfully!")
    print(f"  [PDF Report]  : {pdf_path.resolve()}")
    print(f"  [PDF Backup]  : {(predictions_dir / 'agriagent_farmer_report.pdf').resolve()}")
    print(f"  [PDF Backup]  : {(OUTPUT_DIR / 'agriagent_farmer_report.pdf').resolve()}")
    print(f"  [JSON Report] : {json_path.resolve()}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
