# AGRIAGENT — Current Data Sources and Provenance

This file describes the data sources retained in the production project. The former synthetic `dataset1.csv` and the simulated crop-recommendation benchmark are not included and are not used by the application.

## Production/reference data

| File | Publisher/source basis | Coverage | Production use | Important limitation |
|---|---|---|---|---|
| `data/processed/clean_apy_yield.csv` | Historical district-wise crop area, production and season statistics attributed to DESAgri / Ministry of Agriculture & Farmers Welfare; packaged from a public GitHub mirror | 1997–2015; 233,516 cleaned rows | Regional crop shortlist; historical yield model and benchmarks | Historical district aggregates, not current-year or farm-level observations. The mirror should be cross-checked against the official portal before high-stakes use. |
| `data/processed/clean_district_rainfall.csv` | IMD district rainfall normals, distributed through a public GitHub mirror | 641 records; long-term normals | District rainfall context and irrigation/risk calculations | Climate normals are not live weather or a current forecast. |
| `data/raw/real_sources/imd_subdivision_rainfall_1901_2015.csv` | IMD historical rainfall series via public mirror | 1901–2015; 4,116 rows | Historical reference file; not currently a direct input to crop ranking | Old time series; not current rainfall. |
| `data/processed/clean_soil_fertility.csv` | Soil Health Card portal-derived laboratory parameter sample via public mirror | 1,070 cleaned samples | Reference/validation data; production soil status is calculated by transparent rules | Sample provenance/geographic representativeness is limited; generated rule labels are not expert-confirmed ground truth. |
| `data/processed/clean_mandi_prices.csv` | AGMARKNET/DMI historical rice-market transaction sample via public mirror | 89,013 cleaned rows | Historical wholesale price context | Sample is historical (2012), not a live market quote. |
| `data/knowledge/state_crop_catalog.csv` | Curated state crop catalogue based on cited ICAR/state agriculture profiles and references | 417 state/crop entries | State-level crop reference and fallback when historical data is sparse | Indicative, not exhaustive; it is not a district-specific suitability ranking. |
| `data/knowledge/schemes.json`, `state_schemes.json` | Curated central/state government scheme references | Knowledge-base entries | Potential scheme matching | Eligibility and current availability must be verified on official government portals. |

## Crop recommendation method

The app does **not** use a synthetic crop-classification model. It ranks crops recorded in the APY history for the selected district and season by mean recorded cultivated area over the latest five available years in the selected slice. If district-season coverage is sparse, it explicitly falls back to district-wide or state-level historical records. When records are unavailable, it uses the state crop catalogue as a reference list and does not claim a suitability score.

The packaged APY series ends in 2015. Therefore the result is a **historical regional crop shortlist**, not a guarantee of present-day suitability, profitability, or a farm-specific agronomic recommendation. Check current advisories from the district Krishi Vigyan Kendra (KVK), state agriculture university, and relevant official agriculture department. Google search results vary with wording and source; exact result matching cannot be guaranteed.

## Public source locations

- DESAgri: https://desagri.gov.in/
- Government Open Data Platform: https://data.gov.in/
- IMD: https://mausam.imd.gov.in/
- IMD hydromet portal: https://hydro.imd.gov.in/hydrometweb/
- Soil Health Card portal: https://soilhealth.dac.gov.in/
- AGMARKNET: https://agmarknet.gov.in/
- APY public mirror used in this packaged snapshot: https://raw.githubusercontent.com/ankitaS11/Crop-Yield-Prediction-in-India-using-ML/master/crop_production.csv
- IMD district-normal mirror used in this snapshot: https://raw.githubusercontent.com/baburao-analyst/rain-prediction-india-dashboard/main/data/district%20wise%20rainfall%20normal.csv
- IMD historical-rainfall mirror used in this snapshot: https://raw.githubusercontent.com/baburao-analyst/rain-prediction-india-dashboard/main/data/rainfall%20in%20india%201901-2015.csv
- Soil Health Card sample mirror: https://raw.githubusercontent.com/deepanshu-yadav/soil_data_analysis/master/soil_health_data
- AGMARKNET sample mirror: https://raw.githubusercontent.com/iancovert/Agmarknet/master/Rice/2012.csv

## Security note

Downloaded data are handled as plain-text CSV. The downloader refuses to unpickle remote binary files. Do not run arbitrary downloaded pickle/joblib files from untrusted sources.
