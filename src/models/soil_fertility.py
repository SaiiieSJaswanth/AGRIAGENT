"""
Soil Fertility and Soil Health Evaluation Engine for AGRIAGENT
Implements ICAR-IISS (Indian Institute of Soil Science) and Soil Health Card
official agronomic rating limits for soil parameters and nutrient management.
"""
from typing import Dict, Any, Optional

# ICAR-IISS / DAC&FW Official Standard Rating Thresholds
NUTRIENT_THRESHOLDS = {
    "OC": {"unit": "%", "low": 0.50, "high": 0.75, "name": "Organic Carbon"},
    "P": {"unit": "kg/ha", "low": 10.0, "high": 25.0, "name": "Available Phosphorus"},
    "K": {"unit": "kg/ha", "low": 108.0, "high": 280.0, "name": "Available Potassium"},
    "S": {"unit": "mg/kg", "critical": 10.0, "name": "Available Sulfur"},
    "Zn": {"unit": "mg/kg", "critical": 0.60, "name": "Available Zinc"},
    "Fe": {"unit": "mg/kg", "critical": 4.50, "name": "Available Iron"},
    "Mn": {"unit": "mg/kg", "critical": 2.00, "name": "Available Manganese"},
    "Cu": {"unit": "mg/kg", "critical": 0.20, "name": "Available Copper"}
}

def rate_ph(ph: float) -> Dict[str, str]:
    if ph < 5.5:
        return {"status": "Strongly Acidic", "management": "Application of agricultural lime (CaCO3) recommended; risk of Al/Mn toxicity."}
    elif ph < 6.5:
        return {"status": "Moderately Acidic", "management": "Slightly acidic; suitable for tea, potato, rice; monitor micronutrient availability."}
    elif ph <= 7.5:
        return {"status": "Neutral (Optimal)", "management": "Optimal soil reaction for maximum nutrient availability for most field crops."}
    elif ph <= 8.5:
        return {"status": "Moderately Alkaline", "management": "Alkaline; risk of P and micronutrient fixation (Zn, Fe, Mn)."}
    else:
        return {"status": "Strongly Alkaline / Sodic", "management": "High alkalinity; application of gypsum (CaSO4) and organic matter recommended."}

def rate_ec(ec: float) -> Dict[str, str]:
    if ec < 1.0:
        return {"status": "Normal (Non-saline)", "management": "Salinity within safe range; no restriction on crop selection."}
    elif ec <= 2.0:
        return {"status": "Critical / Moderately Saline", "management": "Critical threshold; yields of sensitive crops may be restricted. Ensure leaching."}
    else:
        return {"status": "Injurious / Highly Saline", "management": "Severe salinity hazard; select salt-tolerant crops (e.g. barley, cotton, sugarbeet)."}

def rate_primary_nutrient(value: float, low_th: float, high_th: float) -> Dict[str, Any]:
    if value < low_th:
        return {"rating": "Low", "score": 1, "adjustment": "Apply 125% of standard Recommended Dose of Fertilizer (RDF)."}
    elif value <= high_th:
        return {"rating": "Medium", "score": 2, "adjustment": "Apply 100% of standard Recommended Dose of Fertilizer (RDF)."}
    else:
        return {"rating": "High", "score": 3, "adjustment": "Apply 75-80% of standard Recommended Dose of Fertilizer (RDF); avoid luxury consumption."}

def rate_micronutrient(value: float, critical_limit: float) -> Dict[str, str]:
    if value < critical_limit:
        return {"rating": "Deficient", "status": "Deficient"}
    return {"rating": "Sufficient", "status": "Sufficient"}

def predict_soil_health_ml(
    ph: float = 7.2,
    ec: float = 0.35,
    s: float = 12.0,
    zn: float = 0.65,
    fe: float = 8.5,
    mn: float = 2.1,
    cu: float = 1.8
) -> Dict[str, Any]:
    """Execute internal Random Forest soil fertility classifier trained on SHC lab data (zero leakage)."""
    try:
        import os, joblib, pandas as pd
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "soil_health", "model.joblib")
        if not os.path.exists(model_path):
            return {"status": "error", "error": f"Model artifact not found at {model_path}"}
        rf_model = joblib.load(model_path)
        shc_row = pd.DataFrame([{
            "pH": float(ph),
            "EC": float(ec),
            "S": float(s),
            "Zn": float(zn),
            "Fe": float(fe),
            "Mn": float(mn),
            "Cu": float(cu)
        }])
        ml_pred = rf_model.predict(shc_row)[0]
        ml_probs = rf_model.predict_proba(shc_row)[0]
        return {
            "status": "success",
            "predicted_fertility_class": str(ml_pred),
            "class_probabilities": {str(c): round(float(p), 4) for c, p in zip(rf_model.classes_, ml_probs)},
            "display_policy": "Hidden internal ML validation signal; farmer receives ICAR-IISS standard evaluation."
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

def evaluate_soil_health(
    ph: Optional[float] = None,
    ec: Optional[float] = None,
    oc: Optional[float] = None,
    p: Optional[float] = None,
    k: Optional[float] = None,
    s: Optional[float] = None,
    zn: Optional[float] = None,
    fe: Optional[float] = None,
    mn: Optional[float] = None,
    cu: Optional[float] = None
) -> Dict[str, Any]:
    """
    Evaluates soil test parameters according to ICAR-IISS standard fertility limits.
    Returns composite soil fertility class, parameter ratings, and agronomic management advice.
    """
    results: Dict[str, Any] = {
        "method": "Rule-based agronomic evaluation (ICAR-IISS / Soil Health Card Standards)",
        "parameter_evaluations": {},
        "deficiencies": [],
        "management_advisories": []
    }

    scores = []

    # pH Assessment
    if ph is not None:
        ph_eval = rate_ph(ph)
        results["parameter_evaluations"]["pH"] = {
            "value": ph,
            "unit": "pH scale",
            **ph_eval
        }
        if ph < 6.5 or ph > 7.5:
            results["management_advisories"].append(f"Soil pH ({ph:.2f}) is {ph_eval['status']}: {ph_eval['management']}")

    # EC Assessment
    if ec is not None:
        ec_eval = rate_ec(ec)
        results["parameter_evaluations"]["EC"] = {
            "value": ec,
            "unit": "dS/m",
            **ec_eval
        }
        if ec >= 1.0:
            results["management_advisories"].append(f"Electrical Conductivity ({ec:.2f} dS/m) is {ec_eval['status']}: {ec_eval['management']}")

    # Organic Carbon Assessment
    if oc is not None:
        oc_eval = rate_primary_nutrient(oc, NUTRIENT_THRESHOLDS["OC"]["low"], NUTRIENT_THRESHOLDS["OC"]["high"])
        results["parameter_evaluations"]["OC"] = {
            "value": oc,
            "unit": "%",
            **oc_eval
        }
        scores.append(oc_eval["score"])
        if oc_eval["rating"] == "Low":
            results["deficiencies"].append("Organic Carbon (Low)")
            results["management_advisories"].append("Organic Carbon is low: incorporate FYM / compost @ 5-10 t/ha, green manuring, or biochar.")

    # Phosphorus Assessment
    if p is not None:
        p_eval = rate_primary_nutrient(p, NUTRIENT_THRESHOLDS["P"]["low"], NUTRIENT_THRESHOLDS["P"]["high"])
        results["parameter_evaluations"]["P"] = {
            "value": p,
            "unit": "kg/ha",
            **p_eval
        }
        scores.append(p_eval["score"])
        if p_eval["rating"] == "Low":
            results["deficiencies"].append("Available Phosphorus (Low)")
            results["management_advisories"].append(f"Available Phosphorus is low ({p:.1f} kg/ha): apply single super phosphate (SSP) or DAP as basal dose.")

    # Potassium Assessment
    if k is not None:
        k_eval = rate_primary_nutrient(k, NUTRIENT_THRESHOLDS["K"]["low"], NUTRIENT_THRESHOLDS["K"]["high"])
        results["parameter_evaluations"]["K"] = {
            "value": k,
            "unit": "kg/ha",
            **k_eval
        }
        scores.append(k_eval["score"])
        if k_eval["rating"] == "Low":
            results["deficiencies"].append("Available Potassium (Low)")
            results["management_advisories"].append(f"Available Potassium is low ({k:.1f} kg/ha): apply muriate of potash (MOP) @ recommended basal/split dose.")

    # Micronutrients Assessment
    micro_params = [("S", s), ("Zn", zn), ("Fe", fe), ("Mn", mn), ("Cu", cu)]
    for param_name, param_val in micro_params:
        if param_val is not None:
            crit = NUTRIENT_THRESHOLDS[param_name]["critical"]
            eval_res = rate_micronutrient(param_val, crit)
            results["parameter_evaluations"][param_name] = {
                "value": param_val,
                "unit": NUTRIENT_THRESHOLDS[param_name]["unit"],
                "critical_limit": crit,
                **eval_res
            }
            if eval_res["rating"] == "Deficient":
                results["deficiencies"].append(f"{NUTRIENT_THRESHOLDS[param_name]['name']} (Deficient)")
                if param_name == "Zn":
                    results["management_advisories"].append("Zinc deficiency detected: apply ZnSO4 @ 25 kg/ha as basal or 0.5% foliar spray.")
                elif param_name == "S":
                    results["management_advisories"].append("Sulfur deficiency detected: apply gypsum or elemental sulfur @ 20-25 kg/ha.")
                elif param_name == "Fe":
                    results["management_advisories"].append("Iron deficiency detected: foliar spray of 1% FeSO4 solution.")

    # Overall Fertility Class Computation (Option A)
    if scores:
        mean_score = sum(scores) / len(scores)
        if mean_score < 1.67:
            overall_class = "Low"
        elif mean_score <= 2.33:
            overall_class = "Medium"
        else:
            overall_class = "High"
        results["soil_fertility_class"] = overall_class
        results["fertility_index"] = round(mean_score, 2)
    else:
        results["soil_fertility_class"] = "Indeterminate (Nutrient parameters not provided)"
        results["fertility_index"] = None

    # Internal Shadow ML Model Execution (Random Forest Classifier on SHC lab data)
    ml_eval = predict_soil_health_ml(
        ph=float(ph if ph is not None else 7.2),
        ec=float(ec if ec is not None else 0.35),
        s=float(s if s is not None else 12.0),
        zn=float(zn if zn is not None else 0.65),
        fe=float(fe if fe is not None else 8.5),
        mn=float(mn if mn is not None else 2.1),
        cu=float(cu if cu is not None else 1.8)
    )
    results["internal_ml_soil_assessment"] = ml_eval

    return results
