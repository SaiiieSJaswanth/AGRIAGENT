"""
Unified Production Inference Engine for AGRIAGENT
Loads genuine trained models with verified schemas, provides rigorous input validation,
and maps real agricultural inputs to correct models, rule engines, and factual data lookups.
"""
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional
import json
import joblib
import pandas as pd
import numpy as np

from src.config.settings import (
    MODEL_DIR, PROCESSED_DIR, KNOWLEDGE_PATH, STATE_SCHEMES_PATH,
    DISTRICT_RAINFALL_DATA, MANDI_PRICES_DATA, YIELD_DATA, STATE_CROP_CATALOG_PATH
)
from src.models.soil_fertility import evaluate_soil_health
from src.models.fertilizer import calculate_fertilizer_guidance
from src.models.irrigation import evaluate_irrigation_needs
from src.models.risk import evaluate_comprehensive_risk
from src.data.crop_catalog import load_state_crop_catalog, crops_for_state

# Model artifact paths
YIELD_MODEL_PATH = MODEL_DIR / "yield" / "model.joblib"
SOIL_MODEL_PATH = MODEL_DIR / "soil_health" / "model.joblib"

_CACHED_MODELS: Dict[str, Any] = {}
_CACHED_DATA: Dict[str, pd.DataFrame] = {}

def get_model(task: str):
    """Loads and caches production model artifacts."""
    if task not in _CACHED_MODELS:
        path = MODEL_DIR / task / "model.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Production model artifact missing for task '{task}': {path}. "
                "Please run 'python train_pipeline.py' to train production models."
            )
        _CACHED_MODELS[task] = joblib.load(path)
    return _CACHED_MODELS[task]

def load_model(task: str):
    """Backward-compatible model loader."""
    return get_model(task)


def predict_crop_recommendation_ml_hidden(state: str, district: str, season: str, crop_year: Optional[int] = None) -> Dict[str, Any]:
    """Run the experimental APY-trained crop classifier without exposing its output.

    This prediction is internal-only. The farmer-facing crop list must always come
    from recommend_crops_from_history(), and this function must never print anything.
    """
    state = str(state or "").strip()
    district = str(district or "").strip()
    season = str(season or "").strip()
    if not state or not district or not season:
        raise ValueError("State, district and season are required for internal crop ML inference.")

    # The model was trained on APY records through 2015; use that same latest
    # available year for a consistent inference feature rather than inventing a year.
    if "apy" not in _CACHED_DATA:
        if not YIELD_DATA.exists():
            raise FileNotFoundError(f"Historical crop records not found: {YIELD_DATA}")
        _CACHED_DATA["apy"] = pd.read_csv(YIELD_DATA)
    if crop_year is not None:
        latest_year = int(crop_year)
    else:
        latest_year = int(pd.to_numeric(_CACHED_DATA["apy"]["Crop_Year"], errors="coerce").max())

    model = get_model("crop_recommendation")
    features = pd.DataFrame([{
        "State_Name": state,
        "District_Name": district,
        "Season": season,
        "Crop_Year": latest_year,
        "Area": 1000.0
    }])
    predicted_crop = str(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    confidence = float(max(probabilities)) if len(probabilities) else None

    # Save internal diagnostics to a file, never to stdout/CMD or farmer report.
    from datetime import datetime, timezone
    from src.config.settings import REPORT_DIR
    log_dir = REPORT_DIR / "internal"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "crop_ml_internal.log").open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"{datetime.now(timezone.utc).isoformat()} | state={state} | district={district} "
            f"| season={season} | model_prediction={predicted_crop} "
            f"| model_probability={confidence if confidence is not None else 'NA'}\n"
        )

    return {"status": "success", "predicted_crop": predicted_crop, "max_probability": confidence}

def recommend_crops_from_history(
    state: str, district: str, season: str, top_k: int = 5
) -> Dict[str, Any]:
    """
    Return a transparent regional crop shortlist from historical Indian crop records.

    This is NOT an ML suitability prediction. It ranks crops actually recorded in the
    DESAgri/APY historical dataset by cultivated area in the selected district/season,
    falling back to district-wide or state-level records when coverage is sparse.
    The included APY data currently spans 1997-2015, so results are historical evidence.
    """
    if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
        raise ValueError("top_k must be an integer between 1 and 20")
    state = str(state or "").strip()
    district = str(district or "").strip()
    season_text = str(season or "").strip().casefold()
    season_map = {
        "kharif (monsoon)": "Kharif", "kharif": "Kharif",
        "rabi (winter)": "Rabi", "rabi": "Rabi",
        "summer (zaid)": "Summer", "summer": "Summer", "zaid": "Summer",
        "whole year / annual": "Whole Year", "whole year": "Whole Year",
        "annual": "Whole Year", "whole": "Whole Year",
    }
    season_value = season_map.get(season_text, str(season).strip())
    if not state or not district:
        raise ValueError("Both state and district are required for regional crop recommendations.")

    if "apy" not in _CACHED_DATA:
        if not YIELD_DATA.exists():
            raise FileNotFoundError(f"Historical crop records not found: {YIELD_DATA}")
        _CACHED_DATA["apy"] = pd.read_csv(YIELD_DATA)
    apy = _CACHED_DATA["apy"].copy()
    required = {"State_Name", "District_Name", "Crop_Year", "Season", "Crop", "Area", "Production"}
    if not required.issubset(apy.columns):
        raise ValueError(f"Historical crop dataset is missing columns: {sorted(required - set(apy.columns))}")

    state_rows = apy[apy["State_Name"].astype(str).str.strip().str.casefold() == state.casefold()]
    source_district = district
    district_rows = state_rows[
        state_rows["District_Name"].astype(str).str.strip().str.casefold() == district.casefold()
    ]
    # District spellings vary between state lists and historical APY files (e.g.
    # Bagalkote vs Bagalkot). Resolve only a high-confidence unique near-match.
    if district_rows.empty and not state_rows.empty:
        names = sorted(state_rows["District_Name"].dropna().astype(str).str.strip().unique().tolist())
        scored = sorted(
            ((SequenceMatcher(None, district.casefold(), name.casefold()).ratio(), name) for name in names),
            reverse=True
        )
        if scored and scored[0][0] >= 0.88 and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.04):
            source_district = scored[0][1]
            district_rows = state_rows[
                state_rows["District_Name"].astype(str).str.strip().str.casefold() == source_district.casefold()
            ]
    season_rows = district_rows[
        district_rows["Season"].astype(str).str.strip().str.casefold() == season_value.casefold()
    ]

    # Prefer local, season-specific evidence. Fall back only when the exact slice is sparse.
    if season_rows["Crop"].nunique() >= 3:
        selected, scope = season_rows, "district_and_season"
    elif district_rows["Crop"].nunique() >= 3:
        selected, scope = district_rows, "district_all_recorded_seasons"
    else:
        state_season = state_rows[
            state_rows["Season"].astype(str).str.strip().str.casefold() == season_value.casefold()
        ]
        if state_season["Crop"].nunique() >= 3:
            selected, scope = state_season, "state_and_season_fallback"
        elif state_rows["Crop"].nunique() >= 3:
            selected, scope = state_rows, "state_all_recorded_seasons_fallback"
        else:
            catalog = load_state_crop_catalog(STATE_CROP_CATALOG_PATH)
            catalog_rows = crops_for_state(catalog, state)
            ranked = [{
                "crop": item["crop"], "evidence_scope": "state_crop_catalog",
                "crop_category": item.get("crop_category", "General"),
                "evidence_note": "Listed as a crop grown in this state; not ranked for soil or season suitability."
            } for item in catalog_rows[:top_k]]
            return {
                "status": "success" if ranked else "unavailable",
                "recommendation_type": "state_crop_reference",
                "top_crop": ranked[0]["crop"] if ranked else None,
                "ranked_recommendations": ranked,
                "state": state, "district": district, "season": season_value,
                "dataset_provenance": "State crop knowledge catalogue; indicative state-level crop list",
                "limitations": "No adequate matching district/state crop-production records were available. Confirm local suitability with the district KVK or state agriculture university."
            }

    # Use the five latest years available in the selected slice, rather than letting older
    # cultivation patterns dominate. APY coverage in this packaged snapshot ends in 2015.
    selected = selected.copy()
    selected["Crop_Year"] = pd.to_numeric(selected["Crop_Year"], errors="coerce")
    selected["Area"] = pd.to_numeric(selected["Area"], errors="coerce").fillna(0)
    selected["Production"] = pd.to_numeric(selected["Production"], errors="coerce").fillna(0)
    latest_available_year = int(selected["Crop_Year"].max())
    recent = selected[selected["Crop_Year"] >= latest_available_year - 4]
    if recent["Crop"].nunique() >= 3:
        selected = recent
    grouped = selected.groupby("Crop", dropna=True).agg(
        cultivated_area_ha=("Area", "mean"),
        recorded_production=("Production", "mean"),
        observations=("Crop_Year", "count"),
        latest_year=("Crop_Year", "max")
    ).reset_index()
    grouped = grouped.sort_values(
        ["cultivated_area_ha", "observations", "recorded_production"],
        ascending=[False, False, False]
    ).head(top_k)

    catalog = load_state_crop_catalog(STATE_CROP_CATALOG_PATH)
    catalog_names = {str(item["crop"]).strip().casefold() for item in crops_for_state(catalog, state)}
    ranked = []
    for _, row in grouped.iterrows():
        crop_name = str(row["Crop"]).strip()
        ranked.append({
            "crop": crop_name,
            "cultivated_area_ha": round(float(row["cultivated_area_ha"]), 2),
            "recorded_production": round(float(row["recorded_production"]), 2),
            "observations": int(row["observations"]),
            "latest_year": int(row["latest_year"]),
            "also_listed_in_state_catalog": crop_name.casefold() in catalog_names,
            "evidence_scope": scope,
            "evidence_note": "Ranked by recorded cultivated area in historical government crop statistics; not a guaranteed suitability score."
        })

    return {
        "status": "success" if ranked else "unavailable",
        "recommendation_type": "historical_regional_crop_shortlist",
        "top_crop": ranked[0]["crop"] if ranked else None,
        "ranked_recommendations": ranked,
        "state": state, "district": district, "matched_source_district": source_district, "season": season_value,
        "evidence_scope": scope,
        "evidence_year_range": [int(selected["Crop_Year"].min()), latest_available_year],
        "dataset_provenance": "Historical district/season crop area and production statistics (DESAgri/APY mirror; packaged data 1997-2015)",
        "ranking_inputs": ["state", "district", "season"],
        "not_used_for_crop_ranking": ["soil type", "soil NPK", "soil pH", "current weather"],
        "ranking_method": "Descending mean recorded cultivated area per observation over the latest five available years in the selected evidence slice; ties broken by observation count and mean recorded production.",
        "limitations": "This shortlist reflects crops historically cultivated in the selected region and season, not a promise of profitability or farm-specific suitability. The APY data does not include farm-level soil type or soil-test measurements, so those inputs do not change the crop ranking; they are used by separate soil/fertilizer advisory components. The packaged historical series ends in 2015. Validate current choices against the latest district KVK/state agriculture university advisory, irrigation, soil test, variety, pest pressure, and market access. Google search results can differ by query and source; exact result matching cannot be guaranteed."
    }

def predict_crop_yield(
    state: str,
    crop: str,
    season: str,
    area_hectares: float,
    crop_year: int = 2015
) -> Dict[str, Any]:
    """
    Predicts expected crop yield in tonnes per hectare using genuine chronological model.
    Features: ['State_Name', 'Crop', 'Season', 'Area', 'Crop_Year']
    """
    if area_hectares <= 0:
        raise ValueError(f"Farm area must be greater than 0 hectares. Received: {area_hectares}")

    input_df = pd.DataFrame([{
        "State_Name": str(state).strip(),
        "Crop": str(crop).strip(),
        "Season": str(season).strip(),
        "Area": float(area_hectares),
        "Crop_Year": int(crop_year)
    }])

    model = get_model("yield")
    pred_yield = float(np.clip(model.predict(input_df)[0], 0, None))
    total_production = round(pred_yield * area_hectares, 2)

    # Contextual historical benchmark lookup
    historical_benchmark = None
    if YIELD_DATA.exists():
        try:
            if "apy" not in _CACHED_DATA:
                _CACHED_DATA["apy"] = pd.read_csv(YIELD_DATA)
            apy = _CACHED_DATA["apy"]
            sub = apy[(apy["State_Name"].str.lower() == state.lower()) & (apy["Crop"].str.lower() == crop.lower())]
            if len(sub) > 0:
                historical_benchmark = {
                    "historical_state_mean_yield": round(float(sub["Yield_tonnes_per_ha"].mean()), 2),
                    "historical_state_max_yield": round(float(sub["Yield_tonnes_per_ha"].max()), 2),
                    "historical_state_min_yield": round(float(sub["Yield_tonnes_per_ha"].min()), 2),
                    "recorded_observations": int(len(sub))
                }
        except Exception:
            pass

    return {
        "status": "success",
        "crop": crop,
        "state": state,
        "season": season,
        "area_hectares": area_hectares,
        "predicted_yield_tonnes_per_ha": round(pred_yield, 2),
        "estimated_total_production_tonnes": total_production,
        "historical_state_benchmark": historical_benchmark,
        "model_type": type(model.named_steps["model"]).__name__ if hasattr(model, "named_steps") else type(model).__name__,
        "dataset_provenance": "Ministry of Agriculture & Farmers Welfare / DESAgri Historical APY Statistics (1997-2015)",
        "limitations": "Model trained on district/season historical data (1997-2011) and tested chronologically (2012-2015). Reflects historical statistical trends, not micro-climate weather events."
    }

def get_district_rainfall(state: str, district: str) -> Optional[Dict[str, Any]]:
    """Retrieves verified IMD monthly and seasonal normal rainfall from clean_district_rainfall.csv."""
    if not DISTRICT_RAINFALL_DATA.exists():
        return None
    if "rainfall" not in _CACHED_DATA:
        _CACHED_DATA["rainfall"] = pd.read_csv(DISTRICT_RAINFALL_DATA)
    rf = _CACHED_DATA["rainfall"]

    mask = (rf["STATE_UT_NAME"].str.lower() == state.lower()) & (rf["DISTRICT"].str.lower() == district.lower())
    match = rf[mask]
    if len(match) == 0:
        # Fallback to district matching
        match = rf[rf["DISTRICT"].str.lower() == district.lower()]

    if len(match) > 0:
        row = match.iloc[0]
        return {
            "source": "India Meteorological Department (IMD) District Rainfall Normals",
            "state": str(row["STATE_UT_NAME"]),
            "district": str(row["DISTRICT"]),
            "annual_normal_mm": round(float(row["ANNUAL"]), 1),
            "kharif_normal_mm": round(float(row.get("Jun-Sep", row.get("JJAS", row["ANNUAL"] * 0.7))), 1),
            "rabi_normal_mm": round(float(row.get("Oct-Dec", row.get("OND", row["ANNUAL"] * 0.15))), 1),
            "monthly_normals_mm": {
                m: round(float(row[m]), 1) for m in ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
                if m in row
            }
        }
    return None

def get_mandi_prices(state: str, district: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves wholesale mandi prices from clean_mandi_prices.csv."""
    if not MANDI_PRICES_DATA.exists():
        return None
    if "mandi" not in _CACHED_DATA:
        _CACHED_DATA["mandi"] = pd.read_csv(MANDI_PRICES_DATA)
    mp = _CACHED_DATA["mandi"]

    sub = mp[mp["State"].str.lower() == state.lower()]
    if district and len(sub) > 0:
        d_sub = sub[sub["District"].str.lower() == district.lower()]
        if len(d_sub) > 0:
            sub = d_sub

    if len(sub) > 0:
        return {
            "source": "AGMARKNET / Directorate of Marketing & Inspection",
            "commodity": "Rice / Paddy (Historical wholesale sample)",
            "observation_period": "2012 Historical Mandi Transactions",
            "modal_price_mean_rs_qtl": round(float(sub["Modal_Price_Rs_Qtl"].mean()), 2),
            "modal_price_min_rs_qtl": round(float(sub["Modal_Price_Rs_Qtl"].min()), 2),
            "modal_price_max_rs_qtl": round(float(sub["Modal_Price_Rs_Qtl"].max()), 2),
            "markets_reporting": int(sub["Market"].nunique()) if "Market" in sub else 1,
            "recorded_transactions": int(len(sub)),
            "limitations": "Prices are historical 2012 wholesale AGMARKNET modal rates for benchmarking and price volatility analysis; not live spot prices."
        }
    return None
