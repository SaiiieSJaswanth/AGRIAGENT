"""
Terminal Report Renderer for AGRIAGENT
Produces clean, structured, faculty-demonstration-ready outputs
for the FIVE core farmer-facing agricultural intelligence modules.
"""
from typing import Dict, Any

def render_report(report: Dict[str, Any]):
    p = report.get("farmer_profile", {})
    loc = report.get("location_intelligence", {})
    crop_rec = report.get("crop_recommendation", {})
    soil_res = report.get("soil_health", {})
    fert_res = report.get("fertilizer_guidance", {})
    irrig_res = report.get("irrigation_guidance", {})
    matches = report.get("support_matches", [])
    docs = report.get("documents", [])

    print("\n" + "=" * 72)
    print("                      AGRIAGENT ADVISORY REPORT")
    print("      Intelligent Real-World Agricultural Intelligence System")
    print("=" * 72)

    # 1. Farmer & Location Context
    print("\n[1] FARMER & LOCATION CONTEXT")
    print("-" * 72)
    print(f"  State / UT         : {p.get('state', 'N/A')}")
    print(f"  District           : {p.get('district', 'N/A')}")
    print(f"  Soil Type          : {p.get('soil_type', 'N/A')}")
    area_acres = p.get('land_area_acres')
    if area_acres is not None and float(area_acres) > 0:
        area_ha = float(area_acres) * 0.404686
        print(f"  Landholding        : {float(area_acres):.2f} acres ({area_ha:.2f} hectares)")
    print(f"  Irrigation Access  : {'Yes (Assured / Supplemental)' if p.get('irrigation_available') else 'No (Rainfed)'}")
    print(f"  Selected Season    : {p.get('season', 'Kharif')}")
    if p.get('current_crop'):
        print(f"  Target Crop        : {p.get('current_crop')}")

    # Climatological normal rainfall
    rf = loc.get("rainfall_normals")
    if rf:
        print(f"  IMD Normal Rainfall: Annual: {rf.get('annual_normal_mm')} mm | Monsoon (Jun-Sep): {rf.get('kharif_normal_mm')} mm | Post-Monsoon (Oct-Dec): {rf.get('rabi_normal_mm')} mm")
    state_crops = loc.get("state_crop_catalog", [])
    if state_crops:
        crop_names = [f"{c['crop']} [{c.get('crop_category', 'General')}]" for c in state_crops[:6]]
        print(f"  State Crop Catalog : {', '.join(crop_names)}")

    # 2. MODULE 1: INTELLIGENT CROP RECOMMENDATION
    print("\n[2] MODULE 1: INTELLIGENT CROP RECOMMENDATION (HYBRID ENGINE)")
    print("-" * 72)
    if crop_rec.get("status") == "success":
        print(f"  Top Historical Match   : {crop_rec.get('top_crop')}")
        print(f"  Evidence Scope         : {crop_rec.get('evidence_scope', 'Regional crop statistics')}")
        if crop_rec.get("evidence_year_range"):
            print(f"  Historical Time Period : {crop_rec['evidence_year_range'][0]} - {crop_rec['evidence_year_range'][1]}")
        print("  Top 5 Recommended Crops (Ranked by Historical Area & Agronomic Evidence):")
        for i, item in enumerate(crop_rec.get("ranked_recommendations", [])[:5], 1):
            if "cultivated_area_ha" in item:
                cat_tag = " [Catalog Verified]" if item.get("also_listed_in_state_catalog") else ""
                print(f"    {i}. {item['crop']:<18} Mean Area: {item['cultivated_area_ha']:,.1f} ha/rec | Records: {item['observations']} | Latest Year: {item['latest_year']}{cat_tag}")
            else:
                print(f"    {i}. {item['crop']:<18} [{item.get('crop_category', 'General')}] (State Agriculture Catalog)")
        print(f"  Data Provenance        : {crop_rec.get('dataset_provenance')}")
        print("  Hidden ML Execution    : Experimental Random Forest Classifier executed internally in shadow mode.")
        print(f"  Advisory Note          : Shortlist reflects historical regional cultivation; verify local variety with district KVK.")
    else:
        print(f"  Crop Recommendation    : {crop_rec.get('error', 'Unavailable')}")

    # 3. MODULE 2: COMPREHENSIVE SOIL HEALTH ASSESSMENT
    print("\n[3] MODULE 2: COMPREHENSIVE SOIL HEALTH ASSESSMENT (ICAR-IISS STANDARDS)")
    print("-" * 72)
    print(f"  Overall Fertility Rating : {soil_res.get('soil_fertility_class', 'Medium')}")
    if soil_res.get("fertility_index"):
        print(f"  Composite Fertility Index: {soil_res.get('fertility_index')} / 3.00")
    params = soil_res.get("parameter_evaluations", {})
    if params:
        print("  Measured Parameter Diagnostic Summary:")
        for k in ["pH", "EC", "OC", "P", "K", "Zn", "S", "Fe"]:
            if k in params:
                info = params[k]
                status_str = info.get("status", info.get("rating", "Normal"))
                print(f"    - {k:<4}: {info.get('value')} {info.get('unit', '')} -> Status: {status_str}")
    deficiencies = soil_res.get("deficiencies", [])
    if deficiencies:
        print(f"  Detected Deficiencies    : {', '.join(deficiencies)}")
    advisories = soil_res.get("management_advisories", [])
    if advisories:
        print("  Soil Amelioration Tips   :")
        for adv in advisories[:3]:
            print(f"    * {adv}")

    # 4. MODULE 3: CROP-SPECIFIC FERTILIZER RECOMMENDATION
    print("\n[4] MODULE 3: CROP-SPECIFIC FERTILIZER ADVISORY (STCR + ML REGIME)")
    print("-" * 72)
    print(f"  Selected Crop          : {fert_res.get('crop')}")
    print(f"  ML Regime Class        : {fert_res.get('ml_regime_classification')}")
    base_rdf = fert_res.get("base_rdf_kg_ha", {})
    adj_rec = fert_res.get("adjusted_recommendation_kg_ha", {})
    print(f"  Standard Base RDF      : N: {base_rdf.get('N', 0)} kg/ha | P2O5: {base_rdf.get('P2O5', 0)} kg/ha | K2O: {base_rdf.get('K2O', 0)} kg/ha")
    print(f"  Soil-Adjusted Nutrients: N: {adj_rec.get('fertilizer_n_requirement_kg_ha', 0)} kg/ha | P2O5: {adj_rec.get('fertilizer_p2o5_requirement_kg_ha', 0)} kg/ha | K2O: {adj_rec.get('fertilizer_k2o_requirement_kg_ha', 0)} kg/ha")
    comm_per_ha = fert_res.get("commercial_fertilizer_per_ha", {})
    farm_tot = fert_res.get("total_farm_requirement", {})
    print(f"  Commercial Equivalents (Rate per Hectare):")
    for prod, val in comm_per_ha.items():
        print(f"    - {prod:<26}: {val:.1f}")
    if farm_tot:
        print(f"  Total Farm Dose (for {fert_res.get('land_area_acres')} acres / {fert_res.get('land_area_ha')} ha):")
        for prod, val in farm_tot.items():
            print(f"    - {prod:<12}: {val:.1f} kg")
    if fert_res.get("nitrogen_double_counting_safeguard"):
        print(f"  Double-Counting Check  : {fert_res.get('nitrogen_double_counting_safeguard')}")
    print(f"  Application Schedule   : {fert_res.get('application_schedule')}")
    warnings = fert_res.get("safety_warnings", [])
    if warnings:
        print("  Safety Warnings        :")
        for w in warnings[:2]:
            print(f"    ! {w}")

    # 5. MODULE 4: INTELLIGENT IRRIGATION RECOMMENDATION
    print("\n[5] MODULE 4: INTELLIGENT IRRIGATION ADVISORY (DECISION TREE ML)")
    print("-" * 72)
    print(f"  ML Requirement Class   : {irrig_res.get('ml_irrigation_class')}")
    print(f"  Crop Evapotranspiration: {irrig_res.get('crop_water_requirement_mm')} mm across growing season")
    print(f"  Expected IMD Rainfall  : {irrig_res.get('expected_seasonal_rainfall_mm')} mm ({irrig_res.get('rainfall_data_nature')})")
    print(f"  Estimated Water Deficit: {irrig_res.get('estimated_net_deficit_mm')} mm")
    print(f"  Critical Growth Stages : {irrig_res.get('critical_growth_stages')}")
    print(f"  Advisory Guidance      : {irrig_res.get('advisory')}")
    alerts = irrig_res.get("alerts", [])
    if alerts:
        for al in alerts:
            print(f"  [ALERT] {al}")
    conservation = irrig_res.get("water_conservation_tips", [])
    if conservation:
        print("  Water Conservation Tips:")
        for tip in conservation[:2]:
            print(f"    > {tip}")

    # 6. MODULE 5: GOVERNMENT AGRICULTURAL SCHEME INTELLIGENCE
    print("\n[6] MODULE 5: GOVERNMENT WELFARE SCHEME INTELLIGENCE")
    print("-" * 72)
    if matches:
        print(f"  Matched Schemes        : {len(matches)} potentially relevant schemes identified")
        for i, m in enumerate(matches[:5], 1):
            s_name = m.get("scheme_name", "Scheme")
            level = m.get("scheme_level", "CENTRAL")
            print(f"\n  {i}. {s_name} [{level}]")
            print(f"     Benefits    : {m.get('benefits_summary', m.get('description', 'N/A'))}")
            print(f"     Eligibility : {m.get('eligibility_summary', 'Check official portal guidelines')}")
            print(f"     Portal URL  : {m.get('official_source_url', m.get('source_url', 'N/A'))}")
            if m.get("application_process"):
                print(f"     How to Apply: {m.get('application_process')[:90]}...")
    else:
        print("  No direct scheme match found for the supplied farmer profile.")

    if docs:
        print(f"\n  Consolidated Document Checklist: {', '.join(docs[:8])}")

    # Limitations
    print("\n" + "=" * 72)
    print("                      LIMITATIONS & DISCLAIMERS")
    print("-" * 72)
    for lim in report.get("limitations", [])[:4]:
        print(f"  * {lim}")
    print("=" * 72 + "\n")
