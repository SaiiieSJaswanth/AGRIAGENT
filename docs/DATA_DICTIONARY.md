# AGRIAGENT Data Dictionary (Verified Production Package)

## 1. Historical APY Crop Production (`data/processed/clean_apy_yield.csv`)

| Field | Meaning | Unit / Note |
|---|---|---|
| `State_Name` | State/Union Territory | Historical source spelling (33 States/UTs) |
| `District_Name` | District | Historical source spelling (646 Districts) |
| `Crop_Year` | Agricultural Census Year | 1997–2015 in this snapshot |
| `Season` | Cultivation Season | Kharif, Rabi, Summer, Whole Year, Autumn, Winter |
| `Crop` | Crop Name | Historical census crop category (56 distinct crops) |
| `Area` | Cultivated Area | Hectares (ha) |
| `Production` | Recorded Crop Production | Metric tonnes (or bales for cotton/jute) |
| `Yield_tonnes_per_ha` | Historical Derived Yield | Metric tonnes per hectare (`Production / Area`) |

*Production Use:* Historical regional crop ranking (Module 1) and shadow Random Forest crop classification training.

---

## 2. IMD District Rainfall Normals (`data/processed/clean_district_rainfall.csv`)

| Field | Meaning | Unit / Note |
|---|---|---|
| `STATE_UT_NAME` | State/UT Name | 35 States and Union Territories |
| `DISTRICT` | District Name | 641 verified districts |
| `JAN` – `DEC` | Monthly Climatological Precipitation Normal | Millimeters (mm) derived from 100-year rain-gauge observations |
| `ANNUAL` | Annual Total Rainfall Normal | Millimeters (mm) |
| `Jan-Feb`, `Mar-May`, `Jun-Sep`, `Oct-Dec` | Seasonal Precipitation Aggregates | Winter, Pre-Monsoon, Monsoon (Kharif), Post-Monsoon (Rabi) in mm |

*Production Use:* FAO-56 Penman-Monteith water deficit calculation and irrigation scheduling (Module 4).

---

## 3. Soil Health Card Laboratory Sample (`data/processed/clean_soil_fertility.csv`)

| Field | Meaning | Unit / Reference Limit |
|---|---|---|
| `pH` | Soil Reaction | Standard pH scale (3.0 – 11.0; Neutral: 6.5 – 7.5) |
| `EC` | Electrical Conductivity | dS/m (Safe: < 1.0; Critical: 1.0 – 2.0; Injurious: > 2.0) |
| `OC` | Organic Carbon | Percent % (Low: < 0.50; Medium: 0.50 – 0.75; High: > 0.75) |
| `P` | Available Phosphorus | kg/ha (Low: < 10.0; Medium: 10.0 – 25.0; High: > 25.0) |
| `K` | Available Potassium | kg/ha (Low: < 108.0; Medium: 108.0 – 280.0; High: > 280.0) |
| `S` | Available Sulfur | mg/kg (Critical Limit: 10.0 mg/kg) |
| `Zn` | Available Zinc | mg/kg (Critical Limit: 0.60 mg/kg) |
| `Fe` | Available Iron | mg/kg (Critical Limit: 4.50 mg/kg) |
| `Mn` | Available Manganese | mg/kg (Critical Limit: 2.00 mg/kg) |
| `Cu` | Available Copper | mg/kg (Critical Limit: 0.20 mg/kg) |
| `Fertility_Class` | Overall Fertility Index | Low, Medium, High (Derived from primary macro-nutrients) |

*Production Use:* Multi-nutrient deficiency diagnosis (Module 2) and leakage-free Random Forest fertility classification.

---

## 4. Fertilizer Recommendation Dataset (`data/processed/clean_fertilizer_recommendation.csv`)

| Field | Meaning | Unit / Note |
|---|---|---|
| `Soil_pH` | Soil pH | Continuous float |
| `Soil_EC` | Soil Electrical Conductivity | dS/m |
| `Soil_OC_percent` | Soil Organic Carbon | Percent % |
| `Soil_P_kg_ha` | Available Soil Phosphorus | kg/ha |
| `Soil_K_kg_ha` | Available Soil Potassium | kg/ha |
| `Recommended_N_kg_ha` | STCR Adjusted Nitrogen Dose | kg/ha |
| `Recommended_P2O5_kg_ha` | STCR Adjusted Phosphorus Dose | kg/ha |
| `Recommended_K2O_kg_ha` | STCR Adjusted Potash Dose | kg/ha |
| `Fertilizer_Regime` | Optimal Commercial NPK Regime | Categorical class: Balanced, Sulfur-Enriched, Potash-Rich, Legume, Saline-Tolerant |

*Production Use:* Training the Random Forest Fertilizer Regime Classifier (Module 3).

---

## 5. Irrigation Scheduling Dataset (`data/processed/clean_irrigation_recommendation.csv`)

| Field | Meaning | Unit / Note |
|---|---|---|
| `Soil_Water_Capacity_mm_m` | Available Water Holding Capacity | mm water per meter soil depth |
| `Growth_Stage_Kc` | FAO-56 Crop Coefficient | Ratio (0.4 – 1.25) across phenological stages |
| `Crop_Water_Requirement_mm` | Stage Water Requirement | mm |
| `Seasonal_Rainfall_Normal_mm` | District Seasonal Rainfall | mm |
| `Water_Deficit_mm` | Net Evapotranspiration Deficit | mm ($\max(0, ET_c - P_{\text{eff}} - S_{\text{buffer}})$) |
| `Irrigation_Class` | Schedule Recommendation | Categorical class: No immediate, Low, Moderate, High requirement |

*Production Use:* Training the Interpretable Decision Tree Irrigation Classifier (Module 4).

---

## 6. State Crop Catalogue (`data/knowledge/state_crop_catalog.csv`)

417 verified state-crop-season suitability associations across 28 Indian States compiled from State Agricultural Universities (SAUs) and ICAR Package of Practices.

---

## 7. Government Welfare Schemes (`data/knowledge/schemes.json` & `state_schemes.json`)

140+ active Central and State agricultural support schemes with statutory eligibility rules, benefit descriptions, document checklists, and official application URLs.

---

## Excluded Synthetic Sources

The synthetic Cartesian grid `dataset1.csv` and simulated benchmarks are permanently excluded from this package. No production prediction path reads them.
