"""
Vector PDF Report Generator for AGRIAGENT
Generates a multi-page, publication-quality agricultural advisory PDF report
using matplotlib's vector PDF backend (no external binary dependencies required).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Dict, Any

def generate_farmer_pdf_report(report: Dict[str, Any], output_pdf_path: str):
    """
    Creates a comprehensive 3-page printable agricultural report in PDF format.
    Page 1: Farmer Context & Module 1 (Crop Recommendation) & Module 2 (Soil Health)
    Page 2: Module 3 (Fertilizer Advisory) & Module 4 (Irrigation Guidance)
    Page 3: Module 5 (Government Welfare Schemes), Document Checklist & Limitations
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)
    
    p = report.get("farmer_profile", {})
    loc = report.get("location_intelligence", {})
    crop_rec = report.get("crop_recommendation", {})
    soil_res = report.get("soil_health", {})
    fert_res = report.get("fertilizer_guidance", {})
    irrig_res = report.get("irrigation_guidance", {})
    matches = report.get("support_matches", [])
    docs = report.get("documents", [])
    limitations = report.get("limitations", [])

    with PdfPages(output_pdf_path) as pdf:
        # =====================================================================
        # PAGE 1: Overview, Location Context, Crop & Soil Health Intelligence
        # =====================================================================
        fig1 = plt.figure(figsize=(8.27, 11.69)) # Standard A4 dimensions
        fig1.patch.set_facecolor("#f9fafb")

        # Header Banner
        plt.figtext(0.5, 0.95, "AGRIAGENT AGRICULTURAL ADVISORY REPORT", 
                    fontsize=16, weight="bold", ha="center", color="#1b5e20")
        plt.figtext(0.5, 0.93, "Verified Decision Support & Hybrid Machine Learning System for India", 
                    fontsize=10, ha="center", color="#424242", style="italic")
        plt.plot([0.08, 0.92], [0.915, 0.915], color="#2e7d32", lw=2, transform=fig1.transFigure)

        # 1. Farmer Profile Summary Box
        plt.figtext(0.08, 0.88, "1. FARMER PROFILE & REGIONAL CONTEXT", fontsize=11, weight="bold", color="#1565c0")
        profile_text = (
            f"State / UT: {p.get('state', 'N/A')}   |   District: {p.get('district', 'N/A')}   |   Primary Soil Type: {p.get('soil_type', 'N/A')}\n"
            f"Landholding: {p.get('land_area_acres', '2.5')} Acres ({float(p.get('land_area_acres', 2.5))*0.404686:.2f} ha)   |   Season: {p.get('season', 'Kharif')}   |   Irrigation: {'Assured' if p.get('irrigation_available') else 'Rainfed'}\n"
            f"Target/Current Crop: {p.get('current_crop', 'Rice')}"
        )
        plt.figtext(0.08, 0.82, profile_text, fontsize=9, color="#212121", 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cfd8dc"))

        # 2. Module 1: Crop Recommendation (Hybrid Engine)
        plt.figtext(0.08, 0.77, "2. MODULE 1: INTELLIGENT CROP RECOMMENDATION (HYBRID ENGINE)", fontsize=11, weight="bold", color="#1565c0")
        ranked_crops = crop_rec.get("ranked_recommendations", [])[:5]
        crop_lines = [f"Top Regional Match: {crop_rec.get('top_crop', 'Rice')}   |   Scope: {crop_rec.get('evidence_scope', 'Regional Statistics')}"]
        crop_lines.append("\nTop 5 Ranked Recommendations (Historical APY Area + Ministry Catalog):")
        for i, it in enumerate(ranked_crops, 1):
            if "cultivated_area_ha" in it:
                crop_lines.append(f"  {i}. {it['crop']} — Mean Area: {it['cultivated_area_ha']:,.1f} ha/rec | Records: {it['observations']} | Latest Year: {it['latest_year']}")
            else:
                crop_lines.append(f"  {i}. {it['crop']} [{it.get('crop_category', 'General')}] (State Agriculture Catalog)")
        crop_lines.append("\n* Note: Evaluated alongside an internal shadow Random Forest Classifier; output remains historical regional evidence.")
        plt.figtext(0.08, 0.58, "\n".join(crop_lines), fontsize=9, color="#212121",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cfd8dc"))

        # 3. Module 2: Soil Health Assessment (ICAR-IISS Standard)
        plt.figtext(0.08, 0.53, "3. MODULE 2: SOIL HEALTH & NUTRIENT STATUS (ICAR-IISS STANDARDS)", fontsize=11, weight="bold", color="#1565c0")
        soil_lines = [
            f"Overall Soil Fertility: {soil_res.get('soil_fertility_class', 'Medium')}   |   Fertility Index: {soil_res.get('fertility_index', 'N/A')} / 3.00",
            "\nParameter Diagnostic Evaluation:"
        ]
        params = soil_res.get("parameter_evaluations", {})
        for k in ["pH", "EC", "OC", "P", "K", "Zn", "S", "Fe"]:
            if k in params:
                info = params[k]
                status_str = info.get("status", info.get("rating", "Normal"))
                soil_lines.append(f"  • {k:<4}: {info.get('value')} {info.get('unit', '')} -> Status: {status_str}")
        defic = soil_res.get("deficiencies", [])
        if defic:
            soil_lines.append(f"\nIdentified Deficiencies: {', '.join(defic)}")
        advis = soil_res.get("management_advisories", [])
        if advis:
            soil_lines.append("Management Recommendations:")
            for adv in advis[:2]:
                soil_lines.append(f"  * {adv}")
        plt.figtext(0.08, 0.20, "\n".join(soil_lines), fontsize=8.5, color="#212121",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cfd8dc"))

        plt.figtext(0.5, 0.04, "Page 1 of 3 — AGRIAGENT Decision Support System", fontsize=8, ha="center", color="#757575")
        plt.axis("off")
        pdf.savefig(fig1)
        plt.close(fig1)

        # =====================================================================
        # PAGE 2: Module 3 (Fertilizer Advisory) & Module 4 (Irrigation Guidance)
        # =====================================================================
        fig2 = plt.figure(figsize=(8.27, 11.69))
        fig2.patch.set_facecolor("#f9fafb")

        plt.figtext(0.5, 0.95, "AGRIAGENT: AGRONOMIC NUTRIENT & WATER MANAGEMENT", 
                    fontsize=14, weight="bold", ha="center", color="#1b5e20")
        plt.plot([0.08, 0.92], [0.93, 0.93], color="#2e7d32", lw=1.5, transform=fig2.transFigure)

        # 4. Module 3: Fertilizer Guidance
        plt.figtext(0.08, 0.89, "4. MODULE 3: CROP-SPECIFIC FERTILIZER RECOMMENDATION (STCR + ML)", fontsize=11, weight="bold", color="#1565c0")
        base_rdf = fert_res.get("base_rdf_kg_ha", {})
        adj_rec = fert_res.get("adjusted_recommendation_kg_ha", {})
        comm_per_ha = fert_res.get("commercial_fertilizer_per_ha", {})
        farm_tot = fert_res.get("total_farm_requirement", {})

        fert_lines = [
            f"Target Crop: {fert_res.get('crop')}   |   ML Predicted Regime: {fert_res.get('ml_regime_classification')}",
            f"Base RDF (kg/ha): N: {base_rdf.get('N', 0)} | P2O5: {base_rdf.get('P2O5', 0)} | K2O: {base_rdf.get('K2O', 0)}",
            f"Soil-Adjusted Requirement (kg/ha): N: {adj_rec.get('fertilizer_n_requirement_kg_ha', 0)} | P2O5: {adj_rec.get('fertilizer_p2o5_requirement_kg_ha', 0)} | K2O: {adj_rec.get('fertilizer_k2o_requirement_kg_ha', 0)}",
            "\nRecommended Commercial Formulations:"
        ]
        for prod, val in comm_per_ha.items():
            fert_lines.append(f"  • {prod}: {val:.1f}")
        if farm_tot:
            fert_lines.append(f"\nTotal Quantity for Farmer's Land ({fert_res.get('land_area_acres')} Acres):")
            for prod, val in farm_tot.items():
                fert_lines.append(f"  • {prod}: {val:.1f} kg")
        fert_lines.append(f"\nDouble-Counting Protection: {fert_res.get('nitrogen_double_counting_safeguard', 'DAP N deducted')}")
        fert_lines.append(f"Application Schedule: {fert_res.get('application_schedule', 'Splits recommended')}")
        warnings = fert_res.get("safety_warnings", [])
        if warnings:
            fert_lines.append("Safety Precautions:")
            for w in warnings[:2]:
                fert_lines.append(f"  ! {w}")
        plt.figtext(0.08, 0.54, "\n".join(fert_lines), fontsize=8.5, color="#212121",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cfd8dc"))

        # 5. Module 4: Irrigation Guidance
        plt.figtext(0.08, 0.49, "5. MODULE 4: INTELLIGENT IRRIGATION RECOMMENDATION (DECISION TREE ML)", fontsize=11, weight="bold", color="#1565c0")
        irrig_lines = [
            f"ML Irrigation Class: {irrig_res.get('ml_irrigation_class')}   |   Model: Decision Tree Classifier",
            f"Crop Water Demand: {irrig_res.get('crop_water_requirement_mm')} mm   |   Expected Seasonal Rainfall: {irrig_res.get('expected_seasonal_rainfall_mm')} mm",
            f"Estimated Net Water Deficit: {irrig_res.get('estimated_net_deficit_mm')} mm across growing period",
            f"Critical Growth Stages: {irrig_res.get('critical_growth_stages')}",
            f"\nAgronomic Advisory: {irrig_res.get('advisory')}"
        ]
        alerts = irrig_res.get("alerts", [])
        if alerts:
            for al in alerts:
                irrig_lines.append(f"ALERT: {al}")
        conserve = irrig_res.get("water_conservation_tips", [])
        if conserve:
            irrig_lines.append("\nWater Conservation Recommendations:")
            for c in conserve[:3]:
                irrig_lines.append(f"  > {c}")
        plt.figtext(0.08, 0.16, "\n".join(irrig_lines), fontsize=8.5, color="#212121",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cfd8dc"))

        plt.figtext(0.5, 0.04, "Page 2 of 3 — AGRIAGENT Decision Support System", fontsize=8, ha="center", color="#757575")
        plt.axis("off")
        pdf.savefig(fig2)
        plt.close(fig2)

        # =====================================================================
        # PAGE 3: Module 5 (Government Welfare Schemes) & Official Verification
        # =====================================================================
        fig3 = plt.figure(figsize=(8.27, 11.69))
        fig3.patch.set_facecolor("#f9fafb")

        plt.figtext(0.5, 0.95, "AGRIAGENT: GOVERNMENT WELFARE ACCESS & VERIFICATION", 
                    fontsize=14, weight="bold", ha="center", color="#1b5e20")
        plt.plot([0.08, 0.92], [0.93, 0.93], color="#2e7d32", lw=1.5, transform=fig3.transFigure)

        # 6. Module 5: Welfare Schemes
        plt.figtext(0.08, 0.89, "6. MODULE 5: GOVERNMENT AGRICULTURAL SCHEME INTELLIGENCE", fontsize=11, weight="bold", color="#1565c0")
        scheme_lines = [f"Total Potentially Relevant Schemes Identified: {len(matches)}"]
        for i, m in enumerate(matches[:5], 1):
            s_name = m.get("scheme_name", "Scheme")
            level = m.get("scheme_level", "CENTRAL")
            scheme_lines.append(f"\n{i}. {s_name} [{level}]")
            scheme_lines.append(f"   Benefits: {m.get('benefits_summary', m.get('description', 'N/A'))}")
            scheme_lines.append(f"   Eligibility: {m.get('eligibility_summary', 'Check portal')}")
            scheme_lines.append(f"   Official Portal: {m.get('official_source_url', m.get('source_url', 'N/A'))}")
        plt.figtext(0.08, 0.46, "\n".join(scheme_lines), fontsize=8, color="#212121",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cfd8dc"))

        # 7. Document Checklist & Limitations
        plt.figtext(0.08, 0.41, "7. CONSOLIDATED DOCUMENT CHECKLIST & SYSTEM LIMITATIONS", fontsize=11, weight="bold", color="#1565c0")
        doc_lines = []
        if docs:
            doc_lines.append(f"Essential Documents for Application: {', '.join(docs[:8])}\n")
        doc_lines.append("Important Scientific & Legal Disclaimers:")
        for lim in limitations[:5]:
            doc_lines.append(f"  * {lim}")
        plt.figtext(0.08, 0.16, "\n".join(doc_lines), fontsize=8, color="#212121",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cfd8dc"))

        # Footer Signature
        plt.figtext(0.5, 0.08, "Report generated automatically by AGRIAGENT Decision Support Platform | Genuine Open-Data Verified", 
                    fontsize=8, ha="center", color="#37474f", weight="bold")
        plt.figtext(0.5, 0.04, "Page 3 of 3 — AGRIAGENT Decision Support System", fontsize=8, ha="center", color="#757575")
        plt.axis("off")
        pdf.savefig(fig3)
        plt.close(fig3)

    print(f"Vector PDF Advisory Report generated successfully: {output_pdf_path}")
    return output_pdf_path
