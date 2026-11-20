# 🌾 AGRIAGENT — 3D Agricultural Intelligence & Multi-Model ML Advisory Platform

[![Python 3.13](https://img.shields.io/badge/Python-3.13.6-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.0-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Three.js](https://img.shields.io/badge/Three.js-WebGL_3D-000000.svg?style=flat&logo=three.js&logoColor=white)](https://threejs.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000.svg?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Tests](https://img.shields.io/badge/Tests-31%2F31%20Passing%20(100%25)-22c55e.svg?style=flat&logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An enterprise-grade hybrid agricultural decision support system integrating **four specialized Machine Learning models**, agronomic stoichiometric constraints, and a **calm interactive 3D WebGL diorama** to deliver end-to-end advisories for Indian farmers.

---

## 🏛 System Architecture

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │           3D INTERACTIVE WEB CLIENT (Three.js)         │
                                  │   [3D Agricultural Diorama]   [4-Step Dynamic Wizard]  │
                                  │   [Real-Time ICAR Sliders]    [5-Theme Dashboard]      │
                                  └───────────────────────────┬────────────────────────────┘
                                                              │
                                                              ▼
                                  ┌────────────────────────────────────────────────────────┐
                                  │          FLASK MULTI-MODEL ML REST API GATEWAY         │
                                  │       (http://127.0.0.1:5000 / Dedicated Endpoints)    │
                                  └───────┬──────────────┬──────────────┬──────────────┬───┘
                                          │              │              │              │
                     ┌────────────────────┘              │              │              └──────────────────┐
                     ▼                                   ▼              ▼                                 ▼
   ┌──────────────────────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────────────────┐
   │             MODULE 1             │ │     MODULE 2     │ │     MODULE 3     │ │             MODULE 4             │
   │        Crop Recommendation       │ │   Soil Health    │ │  STCR Fertilizer │ │      Irrigation Scheduling       │
   │      (Random Forest Model)       │ │  (Zero-Leakage)  │ │ (Random Forest)  │ │     (FAO-56 Water Deficit)       │
   │  233,516 APY Census Records      │ │  1,070 SHC Labs  │ │ 14,980 Regimes   │ │  2,500 Climatological Scenarios  │
   └──────────────────────────────────┘ └──────────────────┘ └──────────────────┘ └──────────────────────────────────┘
                     │                                   │              │                                 │
                     └────────────────────┬──────────────┴──────────────┴─────────────────────────────────┘
                                          │
                                          ▼
                                  ┌────────────────────────────────────────────────────────┐
                                  │     CENTRAL & STATE WELFARE SCHEME MATCHING ENGINE     │
                                  │   PM-KISAN, PMFBY, PMKSY, KCC, SMAM & 140 State Portals │
                                  └───────────────────────────┬────────────────────────────┘
                                                              │
                                                              ▼
                                  ┌────────────────────────────────────────────────────────┐
                                  │         VECTOR REPORT GENERATION & PDF EXPORT          │
                                  │     (Printable Farmer Advisory & Structured JSON)      │
                                  └────────────────────────────────────────────────────────┘
```

---

## 🔬 Four Core Machine Learning Modules

### 1. Module 1: Multi-Class Crop Selection Model
- **Algorithm**: `RandomForestClassifier` (100 Trees, Depth=12, Entropy criterion)
- **Training Data**: 233,516 historical agricultural census records from the Ministry of Agriculture (DESAgri APY Dataset).
- **Features**: State, District, Season, Crop Year, Cultivated Area.
- **Validation**: Cross-referenced against the State Agricultural Crop Catalog to eliminate non-commercial or geographically unviable crops.
- **Metrics**: **88.4% Accuracy**, **0.8715 Macro F1-Score**.

### 2. Module 2: Zero-Target-Leakage Soil Health Classifier
- **Algorithm**: `RandomForestClassifier` (80 Trees, Balanced Class Weights)
- **Training Data**: 1,070 district-level laboratory soil testing records from the National Soil Health Card Scheme.
- **Leakage Prevention**: To strictly eliminate artificial feature leakage, Organic Carbon (OC), Available Phosphorus (P), and Potassium (K) are **withheld from the direct feature set**. The classifier utilizes Soil pH, Electrical Conductivity (EC), Sulfur, and Micronutrients (Zinc, Iron, Manganese, Copper) to predict overall soil fertility status (`Low`, `Medium`, `High`).
- **Corrective Guidance**: Automatically computes Agricultural Lime ($CaCO_3$) for acidic soils ($pH < 5.5$) and Agricultural Gypsum ($CaSO_4$) for sodic alkaline soils ($pH > 8.5$).
- **Metrics**: **84.1% Accuracy**, **0.8290 Macro F1-Score**.

### 3. Module 3: STCR Fertilizer Advisory & Stoichiometric DAP Deduction
- **Algorithm**: `RandomForestClassifier` trained on 14,980 ICAR Soil Test Crop Response (STCR) calibration rules.
- **Dynamic STCR Multipliers**:
  $$\text{Adjusted Nutrients} = \text{RDF} \times \text{Multiplier}(\text{Soil Nutrients})$$
  *(Low: 125%, Medium: 100%, High: 75%)*
- **Stoichiometric Nitrogen Safeguard**:
  Commercial Diammonium Phosphate (DAP 18-46-0) supplies 46% $P_2O_5$ and **18% elemental Nitrogen**. The engine calculates:
  $$\text{DAP Required (kg/ha)} = \frac{\text{Adjusted } P_2O_5}{0.46}$$
  $$\text{Nitrogen from DAP (kg/ha)} = \text{DAP} \times 0.18$$
  $$\text{Net Urea Required (kg/ha)} = \frac{\max(0, \text{Adjusted N} - \text{Nitrogen from DAP})}{0.46}$$
  This prevents severe toxic nitrate over-application and saves the farmer expenditure.
- **Metrics**: **92.3% Accuracy**, **0.9180 Macro F1-Score**.

### 4. Module 4: Dynamic Crop Evapotranspiration & Irrigation Deficit Model
- **Algorithm**: `DecisionTreeClassifier` implementing the FAO-56 Penman-Monteith crop water balance equation.
- **Formulation**:
  $$\text{ET}_c = K_c \times \text{ET}_0$$
  $$\text{Net Irrigation Deficit} = \max(0, \text{ET}_c - \text{Effective Seasonal Rainfall})$$
- **Soil Readily Available Water (RAW)**: Schedules irrigation intervals (4 to 12 days) based on soil physical texture (Sandy, Loamy, Black, Alluvial, Clayey) and identifies critical developmental growth stages.
- **Metrics**: **95.2% Accuracy**, **0.9480 Macro F1-Score**.

---

## 🌐 3D Interactive Web Application

- **Modern Clean White Theme**: Styled with clean light glassmorphism, high-contrast typography, and emerald agricultural accents.
- **Calm by Default**: Eliminates dizzying autonomous rotations or spinning squares.
- **Touchpad & Touch Controls**: Dragging anywhere on your touchpad or mouse smoothly rotates and inspects the 3D diorama in 3D space with fluid damping.
- **5 Thematic 3D Dioramas**:
  - **Crops**: 3D golden-green rice crops, wheat stalks, and harvest sheaves.
  - **Soil**: Multi-layer soil cutaway (Topsoil, Subsoil, Rock Horizon), green tree, deep roots, and warm Sun.
  - **Fertilizer**: 3D Nitrogen/Phosphorus/Potassium atoms, 3 tilted electron orbital rings with orbiting electrons, and fertilizer pellets.
  - **Irrigation**: Glowing golden Sun, lush grass meadow, shade tree, and crystal-blue water canal with water droplets.
  - **Government Schemes**: Classical Secretariat Building with colonnade pillars, grand steps, triangular pediment, dome, and golden Rupee emblem (`₹`).
- **Verified Government Portals**: Clicking **"Apply on Official Portal ↗"** opens the real government application portals (PM-KISAN, PMFBY, PMKSY, KCC, SMAM, Rythu Bharosa, etc.) directly in a new tab.

---

## 📂 Repository Structure

```
AGRIAGENT/
├── app.py                      # Flask 3D Web Application & REST API Server (737 lines)
├── train_pipeline.py           # ML Model Training Pipeline across all 4 modules (759 lines)
├── train.py                    # Training pipeline alias
├── run_agriagent.py            # Interactive CLI Inference Engine (559 lines)
├── run.py                      # Interactive CLI alias
├── predict.py                  # Standalone CLI inference utility
├── run_website.bat             # 1-Click launcher for 3D Web Platform
├── push_to_github.bat          # 1-Click GitHub sync utility
├── requirements.txt            # Python dependencies
├── data/                       # Enriched agricultural datasets with State & District
│   ├── processed/              # clean_apy_yield.csv, clean_soil_fertility.csv, etc.
│   └── knowledge/              # schemes.json, state_schemes.json (140 portal URLs)
├── models/                     # Serialized Scikit-learn joblib model artifacts
├── templates/                  # 3D Web Application Client (index.html)
├── reports/                    # Generated advisory PDF reports and case study analysis
├── docs/                       # Data dictionaries, model cards, and methodology specs
├── research_papers/            # Literature survey IEEE papers
├── government_schemes_evidence/# Scraped evidence and extraction code
├── src/                        # Core Python agronomic modules and orchestrator
└── tests/                      # Automated unit test suite (31 tests, 100% pass)
```

---

## 🚀 Quick Start Guide

### 1. Setup Environment
```bash
git clone https://github.com/SaiiieSJaswanth/AGRIAGENT.git
cd AGRIAGENT
pip install -r requirements.txt
```

### 2. Launch 3D Web Platform
Simply double-click **`run_website.bat`** or run:
```bash
python app.py
```
Open your browser at: **`http://127.0.0.1:5000`**

### 3. Run Interactive CLI Advisory
```bash
python run_agriagent.py
```

### 4. Retrain All 4 Machine Learning Models
```bash
python train_pipeline.py
```

### 5. Run Verification Test Suite
```bash
pytest tests
```
*Output: `================ 31 passed in 14.14s ================`*

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
