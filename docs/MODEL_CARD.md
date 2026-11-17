# AGRIAGENT Comprehensive Model Card
**Academic Level: B.Tech in Artificial Intelligence & Data Science**  
**Standards Compliance: Mitchell et al. Model Card Specifications**

---

## 1. System Purpose & Architecture
AGRIAGENT is a defensible, multi-disciplinary agricultural decision-support system built for Indian agro-climatic zones. It integrates verified empirical records from DESAgri (APY 1997–2015), the India Meteorological Department (IMD), and the Indian Council of Agricultural Research (ICAR-IISS / ICAR-NBSS&LUP).

The operational system delivers **exactly 5 active farmer-facing modules** backed by **4 verified machine learning models**, deterministic agronomic engines, and statutory welfare scheme matching.

---

## 2. Model 1: Crop Recommendation Historical Classifier (Shadow Mode)
- **Algorithm:** Scikit-learn Pipeline (`OneHotEncoder` + `RandomForestClassifier`, $n=100$, max_depth=15).
- **Artifact:** `models/crop_recommendation/model.joblib` | `metadata.json`
- **Predictor Features:** `State_Name` (categorical), `District_Name` (categorical), `Season` (categorical), `Crop_Year` (numeric), `Area` (numeric).
- **Target Variable:** `Dominant_Crop` (historically dominant cultivated crop in each State-District-Season-Year cell).
- **Training Samples:** 41,507 records | **Testing Samples:** 10,377 records.
- **Evaluation Metrics:** Test Accuracy = **82.35%**, Precision (macro) = **0.812**, Recall (macro) = **0.824**, F1-Score (macro) = **0.816**.
- **Deployment Policy:** **Hidden Shadow Execution**. The model executes internally and logs predictions to `reports/internal/crop_ml_internal.log`. To prevent machine learning hallucinations from misinforming smallholders, farmer-facing recommendations are derived from empirical DESAgri historical acreage rankings cross-verified against the 417-record State Crop Catalog.

---

## 3. Model 2: Soil Health Assessment Classifier
- **Algorithm:** Scikit-learn Pipeline (`StandardScaler` + `RandomForestClassifier`, $n=100$, max_depth=10).
- **Artifact:** `models/soil_health/model.joblib` | `metadata.json`
- **Predictor Features:** `pH`, `EC`, `S`, `Zn`, `Fe`, `Mn`, `Cu` (7 non-target laboratory chemical parameters).
- **Target Variable:** `Fertility_Index` (Low / Medium / High overall nutrient sufficiency).
- **Zero Target Leakage:** Primary macronutrients (Organic Carbon, Phosphorus, Potassium) were omitted from model inputs because the ground-truth fertility class is derived from NPK sufficiency thresholds.
- **Training Samples:** 856 laboratory records | **Testing Samples:** 214 laboratory records.
- **Evaluation Metrics:** Test Accuracy = **78.04%**, Precision (weighted) = **0.778**, Recall (weighted) = **0.780**, F1-Score (weighted) = **0.776**.
- **Agronomic Role:** Validates laboratory sample consistency alongside official ICAR-IISS deterministic nutrient ratings.

---

## 4. Model 3: Fertilizer Regime Recommendation Classifier
- **Algorithm:** Scikit-learn Pipeline (`StandardScaler` + `RandomForestClassifier`, $n=100$, max_depth=12).
- **Artifact:** `models/fertilizer/model.joblib` | `metadata.json`
- **Predictor Features:** `Soil_pH`, `Soil_EC`, `Soil_OC_percent`, `Soil_P_kg_ha`, `Soil_K_kg_ha`, `Recommended_N_kg_ha`, `Recommended_P2O5_kg_ha`, `Recommended_K2O_kg_ha`.
- **Target Variable:** `Fertilizer_Regime` (Commercial NPK management class: Balanced, Legume, Potash-Rich, Sulfur-Enriched, Saline-Tolerant).
- **Training Samples:** 11,984 records | **Testing Samples:** 2,996 records.
- **Evaluation Metrics:** Test Accuracy = **97.43%**, Precision (weighted) = **0.975**, Recall (weighted) = **0.974**, F1-Score (weighted) = **0.974**.
- **Precision Safeguard:** Recommends commercial dosages for Urea, DAP, and MOP while dynamically subtracting the 18% elemental Nitrogen content of DAP to prevent nitrate toxicity.

---

## 5. Model 4: Irrigation Schedule Decision Tree Classifier
- **Algorithm:** Scikit-learn `DecisionTreeClassifier` (max_depth=5, min_samples_leaf=10).
- **Artifact:** `models/irrigation/model.joblib` | `metadata.json`
- **Predictor Features:** `Soil_Water_Capacity_mm_m`, `Growth_Stage_Kc`, `Crop_Water_Requirement_mm`, `Seasonal_Rainfall_Normal_mm`, `Water_Deficit_mm`.
- **Target Variable:** `Irrigation_Class` (High requirement, Moderate requirement, Low requirement, No immediate irrigation).
- **Training Samples:** 12,307 records | **Testing Samples:** 3,077 records.
- **Evaluation Metrics:** Test Accuracy = **98.86%**, Precision (weighted) = **0.989**, Recall (weighted) = **0.989**, F1-Score (weighted) = **0.989**.
- **Interpretability:** Fully visualizable decision tree (`decision_tree_visualization.png`) mapping physical evapotranspiration deficit to actionable irrigation schedules.

---

## 6. Removed Modules Justification
- **Crop Yield Prediction:** Permanently removed from active user workflows because district aggregate data cannot account for micro-climatic shocks ($R^2 \approx 0.28$).
- **Mandi Price Intelligence:** Permanently removed because bundled AGMARKNET data reflects 2012 historical wholesale prices.
- **Agricultural Risk Assessment:** Replaced with deterministic FAO-56 water deficit modeling and soil limiting factor advisories.

---

## 7. Diagnostic Visualizations (32 Plots @ 300 DPI)
All 32 diagnostic plots are serialized in `outputs/` and mirrored in `reports/metrics/` (8 plots per model: Feature Boxplots, Confusion Matrix, Covariance Heatmap, Feature Importance, Learning Curve, SHAP Summary, Class Prediction Analysis, Class Distribution).
