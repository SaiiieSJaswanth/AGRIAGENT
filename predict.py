"""
AGRIAGENT Direct ML Prediction CLI
Provides direct command-line execution for the core agricultural intelligence models:
- Crop recommendation (Hybrid regional shortlist & internal ML)
- Soil fertility assessment (ICAR-IISS standard & ML validation)
- Crop-specific fertilizer dosing (STCR & ML regime)
- Irrigation requirement classification (Decision Tree ML & FAO-56 water balance)
"""
import sys
import argparse
from typing import Dict, Any

from src.engine.inference import recommend_crops_from_history
from src.models.soil_fertility import evaluate_soil_health
from src.models.fertilizer import calculate_fertilizer_guidance
from src.models.irrigation import evaluate_irrigation_needs

def run_crop_recommendation(args):
    print("=" * 64)
    print("           AGRIAGENT REGIONAL CROP SHORTLIST")
    print("=" * 64)
    res = recommend_crops_from_history(
        state=args.state, district=args.district, season=args.season, top_k=args.top_k
    )
    print(f"\nTop Regional Match     : {res.get('top_crop')}")
    print(f"Evidence Scope         : {res.get('evidence_scope')}")
    print("Top Ranked Crop Shortlist:")
    for i, item in enumerate(res.get("ranked_recommendations", [])[:args.top_k], 1):
        if "cultivated_area_ha" in item:
            print(f"  {i}. {item['crop']:<18} Mean Area: {item['cultivated_area_ha']:,.1f} ha | Records: {item['observations']}")
        else:
            print(f"  {i}. {item['crop']:<18} [{item.get('crop_category', 'General')}]")
    print(f"\nDataset Provenance     : {res.get('dataset_provenance')}")
    print(f"Limitations Note       : {res.get('limitations')}\n")

def run_soil_evaluation(args):
    print("=" * 64)
    print("        AGRIAGENT SOIL FERTILITY EVALUATION (ICAR-IISS)")
    print("=" * 64)
    res = evaluate_soil_health(
        ph=args.ph, ec=args.ec, oc=args.oc,
        p=args.p, k=args.k, s=args.s, zn=args.zn,
        fe=args.fe, mn=args.mn, cu=args.cu
    )
    print(f"\nSoil Fertility Class : {res.get('soil_fertility_class')}")
    if res.get("fertility_index"):
        print(f"Fertility Index      : {res.get('fertility_index')} / 3.00")
    print("Evaluations          :")
    for param, item in res.get("parameter_evaluations", {}).items():
        status = item.get("status") or item.get("rating", "Normal")
        print(f"  - {param:<4}: {item.get('value')} {item.get('unit')} [{status}]")
    if res.get("deficiencies"):
        print(f"Deficiencies Detected: {', '.join(res['deficiencies'])}")
    for adv in res.get("management_advisories", []):
        print(f"Advisory             : {adv}")
    print()

def run_fertilizer_guidance(args):
    print("=" * 64)
    print("         AGRIAGENT CROP-SPECIFIC FERTILIZER ADVISORY")
    print("=" * 64)
    res = calculate_fertilizer_guidance(
        crop_name=args.crop,
        n_rating=args.n_rating,
        p_rating=args.p_rating,
        k_rating=args.k_rating,
        land_area_acres=args.acres,
        ph=args.ph,
        ec=args.ec,
        oc=args.oc,
        p_soil=args.p,
        k_soil=args.k
    )
    print(f"\nCrop                 : {res.get('crop')}")
    print(f"ML Regime Class      : {res.get('ml_regime_classification')}")
    print(f"Land Area            : {res.get('land_area_acres')} acres ({res.get('land_area_ha')} ha)")
    print("Per-Hectare Commercial Dose:")
    for prod, val in res.get("commercial_fertilizer_per_ha", {}).items():
        print(f"  - {prod:<26}: {val:.1f}")
    print(f"Farm Total Dose      :")
    for prod, val in res.get("total_farm_requirement", {}).items():
        print(f"  - {prod:<12}: {val:.1f} kg")
    print(f"Application Schedule : {res.get('application_schedule')}\n")

def run_irrigation_guidance(args):
    print("=" * 64)
    print("        AGRIAGENT INTELLIGENT IRRIGATION ADVISORY")
    print("=" * 64)
    res = evaluate_irrigation_needs(
        crop_name=args.crop,
        season=args.season,
        seasonal_rainfall_mm=args.rainfall,
        irrigation_available=args.irrigation,
        soil_type=args.soil
    )
    print(f"\nTarget Crop          : {res.get('crop')} ({res.get('season')})")
    print(f"Soil Type            : {res.get('soil_type')}")
    print(f"ML Requirement Class : {res.get('ml_irrigation_class')}")
    print(f"Crop Water Demand    : {res.get('crop_water_requirement_mm')} mm")
    print(f"Expected Rainfall    : {res.get('expected_seasonal_rainfall_mm')} mm")
    print(f"Estimated Deficit    : {res.get('estimated_net_deficit_mm')} mm")
    print(f"Critical Stages      : {res.get('critical_growth_stages')}")
    print(f"Advisory             : {res.get('advisory')}\n")

def run_full_advisory(args):
    print("=" * 64)
    print("        AGRIAGENT COMPREHENSIVE FARMER ADVISORY")
    print("=" * 64)
    from src.agr_agent.farmer_profile import FarmerProfile
    from src.engine.orchestrator import run_agriagent
    from src.reports.terminal_report import render_report
    from src.reports.json_report import save_report
    from src.reports.pdf_report import generate_farmer_pdf_report
    from src.config.settings import REPORT_DIR

    profile = FarmerProfile(
        state=args.state,
        district=args.district,
        season=args.season,
        land_area_acres=args.area,
        soil_ph=args.ph,
        electrical_conductivity_ds_m=args.ec,
        organic_carbon_percent=args.oc,
        phosphorus_kg_ha=args.p,
        potassium_kg_ha=args.k,
        sulfur_ppm=args.s,
        zinc_ppm=args.zn,
        iron_ppm=args.fe,
        manganese_ppm=args.mn,
        copper_ppm=args.cu,
        social_category=args.category,
        gender=args.gender,
        irrigation_available=args.irrigation
    )

    report = run_agriagent(profile)
    print(render_report(report))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    predictions_dir = REPORT_DIR / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    from src.config.settings import OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = REPORT_DIR / "latest_agriagent_report.json"
    save_report(report, str(json_path))
    save_report(report, str(predictions_dir / "latest_agriagent_report.json"))
    save_report(report, str(OUTPUT_DIR / "latest_agriagent_report.json"))

    pdf_path = REPORT_DIR / "agriagent_farmer_report.pdf"
    generate_farmer_pdf_report(report, str(pdf_path))
    generate_farmer_pdf_report(report, str(predictions_dir / "agriagent_farmer_report.pdf"))
    generate_farmer_pdf_report(report, str(OUTPUT_DIR / "agriagent_farmer_report.pdf"))

    print("\n" + "=" * 70)
    print("SUCCESS: Reports generated successfully!")
    print(f"  [PDF Report]  : {pdf_path.resolve()}")
    print(f"  [PDF Backup]  : {(predictions_dir / 'agriagent_farmer_report.pdf').resolve()}")
    print(f"  [PDF Backup]  : {(OUTPUT_DIR / 'agriagent_farmer_report.pdf').resolve()}")
    print(f"  [JSON Report] : {json_path.resolve()}")
    print("=" * 70 + "\n")

def main():
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        sys.argv.insert(1, "full")

    parser = argparse.ArgumentParser(description="AGRIAGENT Direct Agricultural ML CLI")
    subparsers = parser.add_subparsers(dest="task", help="Prediction and advisory task")

    # Full Advisory parser
    full_p = subparsers.add_parser("full", help="Generate complete 5-module advisory report")
    full_p.add_argument("--state", type=str, default="Karnataka", help="State name")
    full_p.add_argument("--district", type=str, default="Belagavi", help="District name")
    full_p.add_argument("--season", type=str, default="Kharif", help="Season")
    full_p.add_argument("--area", type=float, default=2.5, help="Landholding in acres")
    full_p.add_argument("--ph", type=float, default=6.8, help="Soil pH")
    full_p.add_argument("--ec", type=float, default=0.35, help="Soil EC")
    full_p.add_argument("--oc", type=float, default=0.65, help="Soil OC %")
    full_p.add_argument("--p", type=float, default=18.0, help="Soil P kg/ha")
    full_p.add_argument("--k", type=float, default=220.0, help="Soil K kg/ha")
    full_p.add_argument("--s", type=float, default=12.0, help="Soil S mg/kg")
    full_p.add_argument("--zn", type=float, default=0.65, help="Soil Zn mg/kg")
    full_p.add_argument("--fe", type=float, default=8.5, help="Soil Fe mg/kg")
    full_p.add_argument("--mn", type=float, default=2.1, help="Soil Mn mg/kg")
    full_p.add_argument("--cu", type=float, default=1.8, help="Soil Cu mg/kg")
    full_p.add_argument("--category", type=str, default="General", help="Social category")
    full_p.add_argument("--gender", type=str, default="Male", help="Farmer gender")
    full_p.add_argument("--irrigation", action="store_true", default=True, help="Irrigation available")

    # Crop shortlist parser
    crop_p = subparsers.add_parser("crop", help="List crops historically cultivated in a state/district/season")
    crop_p.add_argument("--state", type=str, default="Karnataka", help="State name")
    crop_p.add_argument("--district", type=str, default="Bagalkote", help="District name")
    crop_p.add_argument("--season", type=str, default="Kharif", help="Season (Kharif, Rabi, Summer, Whole Year)")
    crop_p.add_argument("--top_k", type=int, default=5, help="Number of crops to list (1-20)")

    # Soil Health Parser
    soil_p = subparsers.add_parser("soil", help="Evaluate soil health card parameters (ICAR-IISS standard)")
    soil_p.add_argument("--ph", type=float, default=6.8, help="Soil pH")
    soil_p.add_argument("--ec", type=float, default=0.35, help="Electrical conductivity in dS/m")
    soil_p.add_argument("--oc", type=float, default=0.65, help="Organic carbon percent")
    soil_p.add_argument("--p", type=float, default=18.0, help="Available Phosphorus in kg/ha")
    soil_p.add_argument("--k", type=float, default=220.0, help="Available Potassium in kg/ha")
    soil_p.add_argument("--s", type=float, default=12.0, help="Available Sulfur in mg/kg")
    soil_p.add_argument("--zn", type=float, default=0.65, help="Available Zinc in mg/kg")
    soil_p.add_argument("--fe", type=float, default=8.5, help="Available Iron in mg/kg")
    soil_p.add_argument("--mn", type=float, default=2.1, help="Available Manganese in mg/kg")
    soil_p.add_argument("--cu", type=float, default=1.8, help="Available Copper in mg/kg")

    # Fertilizer Parser
    fert_p = subparsers.add_parser("fertilizer", help="Crop-specific fertilizer dose calculation and ML regime")
    fert_p.add_argument("--crop", type=str, default="Rice", help="Crop name")
    fert_p.add_argument("--acres", type=float, default=2.5, help="Landholding in acres")
    fert_p.add_argument("--n_rating", type=str, default="Medium", choices=["Low", "Medium", "High"], help="Soil N Rating")
    fert_p.add_argument("--p_rating", type=str, default="Medium", choices=["Low", "Medium", "High"], help="Soil P Rating")
    fert_p.add_argument("--k_rating", type=str, default="Medium", choices=["Low", "Medium", "High"], help="Soil K Rating")
    fert_p.add_argument("--ph", type=float, default=6.8, help="Soil pH")
    fert_p.add_argument("--ec", type=float, default=0.35, help="Soil EC")
    fert_p.add_argument("--oc", type=float, default=0.65, help="Soil OC %")
    fert_p.add_argument("--p", type=float, default=18.0, help="Soil P kg/ha")
    fert_p.add_argument("--k", type=float, default=220.0, help="Soil K kg/ha")

    # Irrigation Parser
    irrig_p = subparsers.add_parser("irrigation", help="Intelligent irrigation requirement classification")
    irrig_p.add_argument("--crop", type=str, default="Rice", help="Crop name")
    irrig_p.add_argument("--season", type=str, default="Kharif", help="Agricultural season")
    irrig_p.add_argument("--rainfall", type=float, default=750.0, help="Expected seasonal rainfall in mm")
    irrig_p.add_argument("--soil", type=str, default="Loamy Soil", help="Soil type")
    irrig_p.add_argument("--irrigation", action="store_true", default=True, help="Irrigation access available")

    args = parser.parse_args()

    if args.task == "full":
        run_full_advisory(args)
    elif args.task == "crop":
        run_crop_recommendation(args)
    elif args.task == "soil":
        run_soil_evaluation(args)
    elif args.task == "fertilizer":
        run_fertilizer_guidance(args)
    elif args.task == "irrigation":
        run_irrigation_guidance(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
