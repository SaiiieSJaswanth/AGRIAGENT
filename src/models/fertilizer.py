"""
Agronomic & Machine Learning Fertilizer Advisory Engine for AGRIAGENT
Combines a trained ML Classifier (RandomForest on STCR data) with dynamic
agronomic dosage calculation based on ICAR / State Agricultural University standards.
"""
import os
import joblib
import pandas as pd
from typing import Dict, Any, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(ROOT, "models", "fertilizer", "model.joblib")

# Standard ICAR / State Agricultural University RDF recommendations (kg/ha: N, P2O5, K2O)
STANDARD_CROP_RDF = {
    "rice": {"N": 100.0, "P2O5": 50.0, "K2O": 50.0, "splits": "N in 3 splits (50% basal, 25% active tillering, 25% panicle initiation); P & K 100% basal"},
    "wheat": {"N": 120.0, "P2O5": 60.0, "K2O": 40.0, "splits": "1/2 N + full P & K at sowing; remaining 1/2 N topdressed at first irrigation (CRI stage)"},
    "maize": {"N": 120.0, "P2O5": 60.0, "K2O": 40.0, "splits": "N in 3 splits (basal, knee-high, tasseling); full P & K as basal"},
    "cotton": {"N": 120.0, "P2O5": 60.0, "K2O": 60.0, "splits": "N in 3-4 splits (basal, vegetative, squaring, flowering); P & K basal"},
    "sugarcane": {"N": 250.0, "P2O5": 115.0, "K2O": 115.0, "splits": "N in 3 splits (planting, 45 DAP, 90 DAP); full P & K at planting"},
    "groundnut": {"N": 25.0, "P2O5": 50.0, "K2O": 75.0, "splits": "Full N, P, K at sowing; apply Gypsum @ 400 kg/ha at pegging (35-40 DAS)"},
    "mustard": {"N": 80.0, "P2O5": 40.0, "K2O": 40.0, "splits": "1/2 N + full P, K and Sulfur basal; 1/2 N at flowering stage"},
    "soybean": {"N": 30.0, "P2O5": 60.0, "K2O": 40.0, "splits": "Basal application at sowing with Bradyrhizobium seed inoculation"},
    "chickpea": {"N": 20.0, "P2O5": 40.0, "K2O": 20.0, "splits": "Full dose basal at sowing; Rhizobium & PSB seed treatment"},
    "pigeonpeas": {"N": 25.0, "P2O5": 50.0, "K2O": 20.0, "splits": "Full dose basal at sowing; Rhizobium inoculation recommended"},
    "blackgram": {"N": 20.0, "P2O5": 40.0, "K2O": 20.0, "splits": "Full dose basal at sowing"},
    "mungbean": {"N": 20.0, "P2O5": 40.0, "K2O": 20.0, "splits": "Full dose basal at sowing"},
    "banana": {"N": 200.0, "P2O5": 60.0, "K2O": 300.0, "splits": "Split applications per plant at 2nd, 3rd, 4th, and 5th month after planting"},
    "potato": {"N": 150.0, "P2O5": 80.0, "K2O": 100.0, "splits": "1/2 N + full P & K basal at planting; 1/2 N during earthing up (30 DAP)"},
    "tomato": {"N": 120.0, "P2O5": 60.0, "K2O": 60.0, "splits": "1/3 N + all P & K basal; 2/3 N in two equal splits at 30 and 45 days after transplanting"},
    "default": {"N": 80.0, "P2O5": 40.0, "K2O": 40.0, "splits": "1/2 N + full P & K basal; 1/2 N topdressed at active vegetative stage"}
}

def predict_fertilizer_ml_regime(
    ph: float = 6.8, ec: float = 0.35, oc: float = 0.60,
    p_soil: float = 16.0, k_soil: float = 180.0,
    rec_n: float = 100.0, rec_p: float = 50.0, rec_k: float = 50.0
) -> Dict[str, Any]:
    """
    Executes the trained Random Forest Classifier to predict the optimal fertilizer regime class.
    """
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            input_df = pd.DataFrame([{
                "Soil_pH": float(ph),
                "Soil_EC": float(ec),
                "Soil_OC_percent": float(oc),
                "Soil_P_kg_ha": float(p_soil),
                "Soil_K_kg_ha": float(k_soil),
                "Recommended_N_kg_ha": float(rec_n),
                "Recommended_P2O5_kg_ha": float(rec_p),
                "Recommended_K2O_kg_ha": float(rec_k)
            }])
            pred_class = model.predict(input_df)[0]
            probs = model.predict_proba(input_df)[0]
            classes = model.classes_
            prob_dict = {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
            return {
                "status": "success",
                "predicted_regime": str(pred_class),
                "regime_probabilities": prob_dict,
                "model": "RandomForestClassifier (STCR-Trained)"
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "predicted_regime": "Urea_DAP_MOP_Balanced"}
    return {"status": "fallback", "predicted_regime": "Urea_DAP_MOP_Balanced"}

def calculate_fertilizer_guidance(
    crop_name: str,
    n_rating: str = "Medium",
    p_rating: str = "Medium",
    k_rating: str = "Medium",
    land_area_acres: float = 2.5,
    ph: float = 6.8,
    ec: float = 0.35,
    oc: float = 0.60,
    p_soil: float = 16.0,
    k_soil: float = 180.0
) -> Dict[str, Any]:
    """
    Computes fertilizer guidance using Soil Test Crop Response (STCR) logic
    and couples it with the trained ML classifier recommendation.
    """
    crop_clean = str(crop_name).lower().strip()
    rdf = STANDARD_CROP_RDF.get(crop_clean, STANDARD_CROP_RDF["default"])

    multiplier_map = {"Low": 1.25, "Medium": 1.00, "High": 0.75}
    n_mult = multiplier_map.get(n_rating, 1.00)
    p_mult = multiplier_map.get(p_rating, 1.00)
    k_mult = multiplier_map.get(k_rating, 1.00)

    n_rec = round(rdf["N"] * n_mult, 1)
    p_rec = round(rdf["P2O5"] * p_mult, 1)
    k_rec = round(rdf["K2O"] * k_mult, 1)

    # ML Classifier Prediction
    ml_result = predict_fertilizer_ml_regime(
        ph=ph, ec=ec, oc=oc, p_soil=p_soil, k_soil=k_soil,
        rec_n=n_rec, rec_p=p_rec, rec_k=k_rec
    )

    # Convert land area from acres to hectares
    land_area_ha = round(land_area_acres * 0.404686, 2)

    # Calculate commercial fertilizer equivalents (Urea 46% N, DAP 18% N & 46% P2O5, MOP 60% K2O)
    # Accounting for DAP nitrogen contribution to prevent double-counting
    dap_kg_ha = round(p_rec / 0.46, 1)
    n_from_dap = dap_kg_ha * 0.18
    n_remaining = max(0.0, n_rec - n_from_dap)
    urea_kg_ha = round(n_remaining / 0.46, 1)
    mop_kg_ha = round(k_rec / 0.60, 1)

    # Farm total quantities scaled to farmer's landholding
    urea_total = round(urea_kg_ha * land_area_ha, 1)
    dap_total = round(dap_kg_ha * land_area_ha, 1)
    mop_total = round(mop_kg_ha * land_area_ha, 1)

    return {
        "crop": crop_name,
        "method": "ICAR Recommended Dose of Fertilizer (RDF) adjusted by Soil Test Crop Response (STCR) + ML Classifier",
        "ml_regime_classification": ml_result.get("predicted_regime", "Urea_DAP_MOP_Balanced"),
        "ml_status": ml_result.get("status", "success"),
        "land_area_acres": land_area_acres,
        "land_area_ha": land_area_ha,
        "base_rdf_kg_ha": {"N": rdf["N"], "P2O5": rdf["P2O5"], "K2O": rdf["K2O"]},
        "adjusted_recommendation_kg_ha": {
            "fertilizer_n_requirement_kg_ha": n_rec,
            "fertilizer_p2o5_requirement_kg_ha": p_rec,
            "fertilizer_k2o_requirement_kg_ha": k_rec
        },
        "commercial_fertilizer_per_ha": {
            "Urea (46% N) [kg/ha]": urea_kg_ha,
            "DAP (18-46-0) [kg/ha]": dap_kg_ha,
            "MOP (60% K2O) [kg/ha]": mop_kg_ha
        },
        "commercial_fertilizer_quantities": {
            "urea_kg": urea_total,
            "dap_kg": dap_total,
            "mop_kg": mop_total
        },
        "total_farm_requirement": {
            "Urea [kg]": urea_total,
            "DAP [kg]": dap_total,
            "MOP [kg]": mop_total
        },
        "nitrogen_double_counting_safeguard": f"DAP contributed {round(n_from_dap, 1)} kg/ha N, deducted before calculating Urea requirement.",
        "application_schedule": rdf["splits"],
        "safety_warnings": [
            "Never mix Urea with lime or alkaline substances directly.",
            "Apply DAP and MOP at the root zone (basal), not as broadcast on wet foliage.",
            "Split Nitrogen doses to minimize volatilization and leaching losses."
        ],
        "sources": "ICAR-IISS Soil Test Crop Response (STCR) & State Agricultural University Package of Practices."
    }

class FertilizerModel:
    """Compatibility wrapper providing standard interface."""
    def predict(self, row=None, crop="rice", n_rating="Medium", p_rating="Medium", k_rating="Medium"):
        if row is not None and hasattr(row, "to_dict"):
            r = row.to_dict() if not hasattr(row, "iloc") else row.iloc[0].to_dict()
            crop = r.get("crop", r.get("recommended_crop_1", crop))
        guidance = calculate_fertilizer_guidance(crop, n_rating, p_rating, k_rating)
        return guidance["adjusted_recommendation_kg_ha"]
