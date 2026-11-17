"""
Intelligent Irrigation Advisory Engine for AGRIAGENT
Combines a trained Decision Tree Classifier (trained on FAO-56 / IMD Normals water balance)
with growth-stage-specific agronomic irrigation scheduling and conservation recommendations.
"""
import os
import joblib
import pandas as pd
from typing import Dict, Any, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(ROOT, "models", "irrigation", "model.joblib")

CROP_WATER_REQUIREMENTS_MM = {
    "rice": {"water_req_mm": 1250.0, "critical_stages": "Tillering, Panicle Initiation, Flowering, Milk Stage"},
    "wheat": {"water_req_mm": 500.0, "critical_stages": "Crown Root Initiation (CRI - 21 DAS), Tillering, Late Jointing, Flowering, Milk Stage"},
    "maize": {"water_req_mm": 600.0, "critical_stages": "Tasseling, Silking, Early Grain Fill"},
    "cotton": {"water_req_mm": 800.0, "critical_stages": "Squaring, Flowering, Boll Development"},
    "sugarcane": {"water_req_mm": 1800.0, "critical_stages": "Formative Stage (60-130 DAP), Grand Growth Period"},
    "groundnut": {"water_req_mm": 550.0, "critical_stages": "Flowering, Pegging, Pod Development"},
    "mustard": {"water_req_mm": 350.0, "critical_stages": "Rosette stage and Siliqua formation stage"},
    "soybean": {"water_req_mm": 500.0, "critical_stages": "Flowering and Pod filling stage"},
    "chickpea": {"water_req_mm": 300.0, "critical_stages": "Pre-flowering, Pod Development"},
    "pigeonpeas": {"water_req_mm": 400.0, "critical_stages": "Branching, Flowering, Pod Filling"},
    "blackgram": {"water_req_mm": 300.0, "critical_stages": "Flowering, Pod Formation"},
    "mungbean": {"water_req_mm": 300.0, "critical_stages": "Flowering, Pod Formation"},
    "banana": {"water_req_mm": 1500.0, "critical_stages": "Continuous high moisture; avoid water stress during shooting"},
    "potato": {"water_req_mm": 450.0, "critical_stages": "Stolon formation, Tuber initiation and enlargement"},
    "tomato": {"water_req_mm": 600.0, "critical_stages": "Post-transplanting establishment, Flowering, Fruit enlargement"},
    "default": {"water_req_mm": 550.0, "critical_stages": "Flowering and seed/grain filling stages"}
}

SOIL_WATER_HOLDING_CAPACITY = {
    "alluvial soil": 160.0,
    "black soil (regur)": 200.0,
    "red soil": 120.0,
    "laterite soil": 100.0,
    "desert / arid soil": 70.0,
    "peaty / marshy soil": 180.0,
    "loamy soil": 150.0,
    "default": 150.0
}

def predict_irrigation_ml_class(
    soil_cap: float = 150.0,
    kc: float = 0.85,
    crop_water_req: float = 600.0,
    rainfall_norm: float = 750.0,
    water_deficit: float = 50.0
) -> Dict[str, Any]:
    """
    Executes the trained Decision Tree Classifier to predict the irrigation requirement class.
    """
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            input_df = pd.DataFrame([{
                "Soil_Water_Capacity_mm_m": float(soil_cap),
                "Growth_Stage_Kc": float(kc),
                "Crop_Water_Requirement_mm": float(crop_water_req),
                "Seasonal_Rainfall_Normal_mm": float(rainfall_norm),
                "Water_Deficit_mm": float(water_deficit)
            }])
            pred = model.predict(input_df)[0]
            probs = model.predict_proba(input_df)[0]
            classes = model.classes_
            prob_dict = {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
            return {
                "status": "success",
                "predicted_class": str(pred),
                "probabilities": prob_dict,
                "model": "DecisionTreeClassifier (FAO-56 / IMD Normals)"
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "predicted_class": "Moderate irrigation requirement"}
    return {"status": "fallback", "predicted_class": "Moderate irrigation requirement"}

def evaluate_irrigation_needs(
    crop_name: str,
    season: str = "Kharif",
    seasonal_rainfall_mm: Optional[float] = None,
    irrigation_available: bool = True,
    soil_type: str = "Loamy Soil",
    growth_stage_kc: float = 0.85
) -> Dict[str, Any]:
    """
    Computes irrigation requirement class using the trained Decision Tree Classifier
    coupled with agronomic water-balance assessment.
    """
    crop_clean = str(crop_name).lower().strip()
    crop_info = CROP_WATER_REQUIREMENTS_MM.get(crop_clean, CROP_WATER_REQUIREMENTS_MM["default"])
    req_mm = crop_info["water_req_mm"]

    # Soil water holding capacity
    soil_cap = SOIL_WATER_HOLDING_CAPACITY.get(str(soil_type).lower().strip(), SOIL_WATER_HOLDING_CAPACITY["default"])

    # Seasonal rainfall fallback if not supplied
    if seasonal_rainfall_mm is None:
        rainfall_est = 750.0 if "kharif" in season.lower() else (150.0 if "rabi" in season.lower() else 50.0)
    else:
        rainfall_est = float(seasonal_rainfall_mm)

    effective_rf = max(0.0, rainfall_est * 0.75)
    stage_water_demand = (req_mm / 4.0) * growth_stage_kc
    stage_effective_rf = effective_rf / 4.0
    net_deficit = max(0.0, (stage_water_demand - stage_effective_rf) - (soil_cap * 0.25))

    # Execute ML Model
    ml_result = predict_irrigation_ml_class(
        soil_cap=soil_cap,
        kc=growth_stage_kc,
        crop_water_req=req_mm,
        rainfall_norm=rainfall_est,
        water_deficit=net_deficit
    )
    req_class = ml_result.get("predicted_class", "Moderate irrigation requirement")

    if req_class == "No immediate irrigation required":
        advisory = "Seasonal rainfall and soil moisture reservoir are sufficient. No immediate irrigation required."
    elif req_class == "Low irrigation requirement":
        advisory = f"Light supplemental irrigation recommended ({net_deficit:.0f} mm deficit) during dry spells."
    elif req_class == "Moderate irrigation requirement":
        advisory = f"Regular supplemental irrigation required ({net_deficit:.0f} mm deficit) during critical growth stages."
    else:
        advisory = f"High irrigation requirement ({net_deficit:.0f} mm deficit). Assured micro-irrigation (drip/sprinkler) strongly recommended."

    alerts = []
    if not irrigation_available and req_class in ["Moderate irrigation requirement", "High irrigation requirement"]:
        alerts.append("CRITICAL RISK: Farmer has no irrigation access for a crop experiencing a severe rainfall deficit. High vulnerability to moisture stress.")

    water_conservation_tips = [
        "Use organic mulching (crop residue / straw) to reduce surface soil evaporation by up to 30%.",
        "Adopt drip or micro-sprinkler irrigation to achieve 40-50% water savings compared to flood irrigation.",
        "Schedule irrigation during early morning or evening hours to minimize evaporative loss.",
        "Construct farm ponds or check dams for in-situ rainwater harvesting under PMKSY."
    ]

    return {
        "crop": crop_name,
        "season": season,
        "soil_type": soil_type,
        "soil_water_holding_capacity_mm_m": soil_cap,
        "method": "Hybrid Decision Tree ML Classifier + FAO-56 Agronomic Water Balance",
        "irrigation_requirement_class": req_class,
        "ml_irrigation_class": req_class,
        "ml_status": ml_result.get("status", "success"),
        "crop_water_requirement_mm": req_mm,
        "expected_seasonal_rainfall_mm": round(rainfall_est, 1),
        "rainfall_data_nature": "Climatological historical normal (IMD 1951-2000), not a live weather forecast",
        "estimated_net_deficit_mm": round(net_deficit, 1),
        "critical_growth_stages": crop_info["critical_stages"],
        "advisory": advisory,
        "water_conservation_tips": water_conservation_tips,
        "alerts": alerts,
        "sources": "FAO-56 Crop Evapotranspiration Guidelines & IMD District Climatological Normals."
    }
