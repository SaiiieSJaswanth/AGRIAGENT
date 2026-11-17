# AGRIAGENT Reproducibility and Execution Guide
**Complete Verification of 5 Active Modules, 4 Machine Learning Models, and 32 Diagnostic Visualizations**

---

## 1. Environment Setup

Create an isolated virtual environment with Python 3.10+ (tested on Python 3.13):

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Install exact pinned dependencies
pip install -r requirements.txt
```

---

## 2. Interactive Terminal Advisory CLI

Launch the interactive guided farmer interface:

```bash
python run_agriagent.py
```

Follow the prompts:
- Select State (e.g., `Karnataka`, `Maharashtra`, `Punjab`, etc.)
- Select District (dynamically filtered to the selected State)
- Select Season (`Kharif`, `Rabi`, `Summer`, `Whole Year`)
- Enter Soil Test values (pH, EC, Organic Carbon, Phosphorus, Potassium, Micronutrients)
- Enter Farm Landholding (Acres)
- Select Category and Gender for statutory scheme filtering

Outputs:
1. ANSI-formatted color terminal dashboard.
2. Machine-readable JSON export saved to `outputs/latest_agriagent_report.json`.
3. Multi-page vector PDF report generated and saved to `outputs/agriagent_farmer_report.pdf`.

---

## 3. Automated Headless Prediction CLI

Run non-interactive inference with custom arguments:

```bash
python predict.py --state "Karnataka" --district "Belagavi" --season "Kharif" --area 2.5 \
                  --ph 6.8 --ec 0.45 --oc 0.65 --p 22.0 --k 180.0 --s 14.0 --zn 0.75 \
                  --fe 5.2 --mn 3.1 --cu 0.35 --category "General" --gender "Male"
```

Generates complete advisory results and outputs JSON and PDF reports to `outputs/`.

---

## 4. Full Automated Test Suite Execution

Execute the complete pytest verification suite (all unit tests, integration tests, agronomic engines, and visualization assertions):

```bash
pytest tests/ -v
```

Expected Output:
```
============================= 31 passed in 10.22s =============================
```

---

## 5. Model Retraining and Visualizations Regeneration

To regenerate all 4 machine learning model binaries and re-render all 32 diagnostic 300-DPI PNG plots:

```bash
python scripts/train_and_generate_32_visualizations.py
```

This script:
1. Loads the genuine empirical datasets (`clean_apy_yield.csv`, `clean_soil_fertility.csv`, `clean_fertilizer_recommendation.csv`, `clean_irrigation_recommendation.csv`).
2. Trains and serializes all 4 models:
   - `models/crop_recommendation/model.joblib`
   - `models/soil_health/model.joblib`
   - `models/fertilizer/model.joblib`
   - `models/irrigation/model.joblib`
3. Writes evaluation metadata to `models/*/metadata.json`.
4. Renders exactly 8 diagnostic plots per model (32 plots total) at 300 DPI into `outputs/<module>/` and mirrors them into `reports/metrics/<module>/`.

---

## 6. Real Dataset Verification

All real datasets reside in `data/processed/`:
- `clean_apy_yield.csv`: 233,516 rows (DESAgri agricultural census 1997–2015).
- `clean_district_rainfall.csv`: 641 districts (IMD rainfall normals).
- `clean_soil_fertility.csv`: 1,070 lab records (Soil Health Card).
- `clean_fertilizer_recommendation.csv`: 14,980 rows (ICAR-STCR formulation).
- `clean_irrigation_recommendation.csv`: 15,384 rows (FAO-56 water balance).
- `state_crop_catalog.csv`: 417 verified records across 28 states.
- `schemes.json` & `state_schemes.json`: 140+ active central and state welfare programs.
