from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class FarmerProfile:
    state: str
    district: str
    soil_type: str = "Medium Soil"
    land_area_acres: Optional[float] = None
    land_ownership: Optional[str] = None
    farmer_category: Optional[str] = None
    income_category: Optional[str] = None
    farming_experience_years: Optional[float] = None
    current_crop: Optional[str] = None
    season: str = "Kharif"
    irrigation_available: Optional[bool] = None
    existing_agricultural_loan: Optional[bool] = None
    loan_required: bool = False
    loan_purpose: Optional[str] = None
    equipment_required: bool = False
    irrigation_required: bool = False
    insurance_required: bool = False
    subsidy_required: bool = False
    # Optional Soil Health Card measurements
    nitrogen_kg_ha: Optional[float] = None
    phosphorus_kg_ha: Optional[float] = None
    potassium_kg_ha: Optional[float] = None
    soil_ph: Optional[float] = None
    organic_carbon_percent: Optional[float] = None
    electrical_conductivity_ds_m: Optional[float] = None
    sulfur_ppm: Optional[float] = None
    zinc_ppm: Optional[float] = None
    iron_ppm: Optional[float] = None
    manganese_ppm: Optional[float] = None
    copper_ppm: Optional[float] = None
    social_category: Optional[str] = None
    gender: Optional[str] = None

    def to_dict(self):
        return asdict(self)
