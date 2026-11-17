"""
Multi-Dimensional Agricultural Risk Assessment Engine for AGRIAGENT
Evaluates climate/rainfall risk, soil reaction/salinity hazards,
irrigation dependency, and financial vulnerability using verified parameters.
"""
from typing import Dict, Any, Optional

RISK_ORDER = {"Low": 0, "Moderate": 1, "High": 2}
RISK_LABEL = {0: "Low", 1: "Moderate", 2: "High"}

def evaluate_comprehensive_risk(
    annual_rainfall_mm: Optional[float] = None,
    ph: Optional[float] = None,
    ec: Optional[float] = None,
    water_deficit_mm: Optional[float] = None,
    irrigation_available: bool = True,
    existing_loan: bool = False,
    loan_required: bool = False
) -> Dict[str, Any]:
    dimensions = {}
    factors = []

    # 1. Climate / Drought Risk
    if annual_rainfall_mm is not None:
        if annual_rainfall_mm < 600.0:
            dimensions["drought_risk"] = "High"
            factors.append("Low normal rainfall (<600 mm): high drought risk.")
        elif annual_rainfall_mm < 1000.0:
            dimensions["drought_risk"] = "Moderate"
        else:
            dimensions["drought_risk"] = "Low"

        # Flood / Excess rainfall risk
        if annual_rainfall_mm > 2500.0:
            dimensions["flood_risk"] = "High"
            factors.append("High normal rainfall (>2500 mm): high flood / waterlogging risk.")
        elif annual_rainfall_mm > 1800.0:
            dimensions["flood_risk"] = "Moderate"
        else:
            dimensions["flood_risk"] = "Low"
    else:
        dimensions["drought_risk"] = "Moderate"
        dimensions["flood_risk"] = "Low"

    # 2. Soil Degradation / Salinity / Acidity Risk
    soil_risk = "Low"
    if ph is not None:
        if ph < 5.5 or ph > 8.5:
            soil_risk = "High"
            factors.append(f"Severe soil reaction constraint (pH {ph:.2f}): high acidity or alkalinity hazard.")
        elif ph < 6.0 or ph > 8.0:
            soil_risk = "Moderate"
    if ec is not None:
        if ec >= 2.0:
            soil_risk = "High"
            factors.append(f"Severe soil salinity hazard (EC {ec:.2f} dS/m).")
        elif ec >= 1.0 and soil_risk != "High":
            soil_risk = "Moderate"
    dimensions["soil_degradation_risk"] = soil_risk

    # 3. Water Stress Risk
    if water_deficit_mm is not None and water_deficit_mm > 300.0 and not irrigation_available:
        dimensions["water_stress_risk"] = "High"
        factors.append("Severe water deficit without access to irrigation.")
    elif water_deficit_mm is not None and water_deficit_mm > 150.0 and not irrigation_available:
        dimensions["water_stress_risk"] = "Moderate"
    else:
        dimensions["water_stress_risk"] = "Low"

    # 4. Financial Risk
    if existing_loan and loan_required:
        dimensions["financial_risk"] = "High"
        factors.append("Existing agricultural debt combined with additional credit requirement.")
    elif existing_loan or loan_required:
        dimensions["financial_risk"] = "Moderate"
    else:
        dimensions["financial_risk"] = "Low"

    # Overall Risk
    known_scores = [RISK_ORDER.get(v, 1) for v in dimensions.values()]
    overall_score = max(known_scores) if known_scores else 1
    overall = RISK_LABEL[overall_score]

    return {
        "dimensions": dimensions,
        "overall": overall,
        "risk_factors": factors,
        "method": "Multi-dimensional agronomic and financial risk assessment"
    }

def analyze_risk(row: Any) -> Dict[str, Any]:
    """Backward-compatibility interface."""
    if isinstance(row, dict):
        d = row
    elif hasattr(row, "to_dict"):
        d = row.to_dict() if not hasattr(row, "iloc") else row.iloc[0].to_dict()
    else:
        d = {}

    annual_rainfall = d.get("annual_rainfall_mm")
    ph = d.get("ph")
    ec = d.get("electrical_conductivity_ds_m", d.get("EC"))
    irrigation_avail = d.get("irrigation_available", True)
    existing_loan = d.get("existing_agricultural_loan", False)
    loan_req = d.get("loan_required", False)

    return evaluate_comprehensive_risk(
        annual_rainfall_mm=annual_rainfall,
        ph=ph,
        ec=ec,
        irrigation_available=irrigation_avail,
        existing_loan=existing_loan,
        loan_required=loan_req
    )
