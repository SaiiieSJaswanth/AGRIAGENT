"""
Data Preparation and Cleaning Pipeline for AGRIAGENT
Processes raw authentic datasets into validated, clean, leakage-free datasets.
"""
import os
import json
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'real_sources')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
METADATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'metadata')
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

def prepare_apy_dataset():
    print("\n--- 1. Processing APY (Area, Production, Yield) Dataset ---")
    raw_path = os.path.join(RAW_DIR, 'apy_crop_production.csv')
    df = pd.read_csv(raw_path)
    initial_rows = len(df)
    print(f"Raw APY rows: {initial_rows:,}, cols: {df.shape[1]}")

    # Standardize column names
    df.columns = [c.strip() for c in df.columns]
    
    # Strip string whitespace
    for col in ['State_Name', 'District_Name', 'Season', 'Crop']:
        df[col] = df[col].astype(str).str.strip()

    # Drop missing production records
    df = df.dropna(subset=['Production'])
    print(f"After dropping null Production: {len(df):,} rows")

    # Filter out zero or negative area
    df = df[df['Area'] > 0]
    
    # Calculate Yield in tonnes per hectare
    df['Yield_tonnes_per_ha'] = df['Production'] / df['Area']

    # Exclude non-weight crops where production is measured in nuts (e.g. Coconut)
    # or fodder crops with unstandardized yield metrics
    exclude_crops = ['Coconut ', 'Coconut', 'Arecanut', 'Rubber']
    df = df[~df['Crop'].isin(exclude_crops)]

    # Filter out biologically impossible yields for field crops (yield > 50 t/ha is reporting unit error)
    # e.g., sugarcane has max yield ~120 t/ha, cereals max ~10-15 t/ha
    df = df[(df['Yield_tonnes_per_ha'] >= 0.01) & (df['Yield_tonnes_per_ha'] <= 80.0)]
    print(f"After cleaning & biological plausibility filtering: {len(df):,} rows")

    out_path = os.path.join(PROCESSED_DIR, 'clean_apy_yield.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved clean APY dataset to: {out_path} ({os.path.getsize(out_path):,} bytes)")

    summary = {
        'initial_rows': initial_rows,
        'clean_rows': len(df),
        'unique_states': df['State_Name'].nunique(),
        'unique_districts': df['District_Name'].nunique(),
        'unique_crops': df['Crop'].nunique(),
        'year_range': [int(df['Crop_Year'].min()), int(df['Crop_Year'].max())],
        'yield_stats': {
            'mean': float(df['Yield_tonnes_per_ha'].mean()),
            'median': float(df['Yield_tonnes_per_ha'].median()),
            'std': float(df['Yield_tonnes_per_ha'].std()),
            'min': float(df['Yield_tonnes_per_ha'].min()),
            'max': float(df['Yield_tonnes_per_ha'].max()),
        }
    }
    return summary

def prepare_soil_health_dataset():
    print("\n--- 2. Processing Soil Health Card Dataset ---")
    raw_path = os.path.join(RAW_DIR, 'soil_health_card_samples.csv')
    df = pd.read_csv(raw_path)
    initial_rows = len(df)
    print(f"Raw Soil Health samples: {initial_rows:,}, cols: {df.shape[1]}")

    # Remove duplicates
    df = df.drop_duplicates()

    # Soil Health Card / ICAR-IISS Fertility Criteria:
    # OC (%): Low < 0.5, Medium 0.5-0.75, High > 0.75
    # P (kg/ha): Low < 10, Medium 10-25, High > 25
    # K (kg/ha): Low < 108, Medium 108-280, High > 280
    def score_nutrient(val, low_th, high_th):
        if pd.isna(val): return np.nan
        if val < low_th: return 1
        elif val <= high_th: return 2
        else: return 3

    oc_score = df['OC'].apply(lambda x: score_nutrient(x, 0.50, 0.75))
    p_score = df['P'].apply(lambda x: score_nutrient(x, 10.0, 25.0))
    k_score = df['K'].apply(lambda x: score_nutrient(x, 108.0, 280.0))

    # Overall Soil Fertility Index: Average nutrient score (1.0 to 3.0)
    sfi = (oc_score + p_score + k_score) / 3.0
    df['soil_fertility_index'] = sfi

    def classify_fertility(score):
        if score < 1.67: return 'Low'
        elif score <= 2.33: return 'Medium'
        else: return 'High'

    df['soil_fertility_class'] = df['soil_fertility_index'].apply(classify_fertility)
    print("Derived Soil Fertility Class distribution (ICAR-IISS standard):")
    print(df['soil_fertility_class'].value_counts())

    # Handle missing values in micronutrients via median imputation (fit only on real records)
    for col in ['Mn', 'Cu', 'Zn', 'Fe', 'S']:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    out_path = os.path.join(PROCESSED_DIR, 'clean_soil_fertility.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved clean Soil Fertility dataset to: {out_path}")

    summary = {
        'initial_rows': initial_rows,
        'clean_rows': len(df),
        'classes': df['soil_fertility_class'].value_counts().to_dict(),
        'feature_means': {c: float(df[c].mean()) for c in ['pH', 'EC', 'OC', 'P', 'K', 'S', 'Zn', 'Fe', 'Mn', 'Cu']}
    }
    return summary

def prepare_district_rainfall_dataset():
    print("\n--- 3. Processing IMD District Rainfall Normal Dataset ---")
    raw_path = os.path.join(RAW_DIR, 'imd_district_rainfall_normal.csv')
    df = pd.read_csv(raw_path)
    initial_rows = len(df)
    print(f"Raw District Rainfall rows: {initial_rows:,}, cols: {df.shape[1]}")

    # Clean state and district names
    df['STATE_UT_NAME'] = df['STATE_UT_NAME'].astype(str).str.strip().str.title()
    df['DISTRICT'] = df['DISTRICT'].astype(str).str.strip().str.title()

    # Drop any null district names
    df = df.dropna(subset=['DISTRICT', 'STATE_UT_NAME'])

    out_path = os.path.join(PROCESSED_DIR, 'clean_district_rainfall.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved clean District Rainfall dataset to: {out_path}")

    # Build verified State -> District hierarchy mapping
    state_district_map = {}
    for state, grp in df.groupby('STATE_UT_NAME'):
        state_district_map[state] = sorted(grp['DISTRICT'].unique().tolist())

    map_path = os.path.join(PROCESSED_DIR, 'state_district_hierarchy.json')
    with open(map_path, 'w') as f:
        json.dump(state_district_map, f, indent=2)
    print(f"Saved verified State-District hierarchy ({len(state_district_map)} states) to: {map_path}")

    summary = {
        'initial_rows': initial_rows,
        'clean_rows': len(df),
        'states_covered': len(state_district_map),
        'districts_covered': len(df),
        'mean_annual_rainfall_mm': float(df['ANNUAL'].mean()),
        'min_annual_rainfall_mm': float(df['ANNUAL'].min()),
        'max_annual_rainfall_mm': float(df['ANNUAL'].max())
    }
    return summary

def prepare_mandi_prices_dataset():
    print("\n--- 4. Processing AGMARKNET Mandi Price Dataset ---")
    raw_path = os.path.join(RAW_DIR, 'agmarknet_mandi_prices_sample.csv')
    df = pd.read_csv(raw_path)
    initial_rows = len(df)
    print(f"Raw Mandi Price records: {initial_rows:,}, cols: {df.shape[1]}")

    # Clean numeric columns
    for col in ['Modal_Price_Rs_Qtl', 'Min_Price_Rs_Qtl', 'Max_Price_Rs_Qtl', 'Arrivals_Tonnes']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Modal_Price_Rs_Qtl', 'State', 'District'])
    # Filter realistic commodity prices (Rs. 500 to Rs. 20,000 per quintal)
    df = df[(df['Modal_Price_Rs_Qtl'] >= 500) & (df['Modal_Price_Rs_Qtl'] <= 25000)]

    out_path = os.path.join(PROCESSED_DIR, 'clean_mandi_prices.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved clean Mandi Prices dataset to: {out_path}")

    summary = {
        'initial_rows': initial_rows,
        'clean_rows': len(df),
        'unique_states': df['State'].nunique(),
        'unique_districts': df['District'].nunique(),
        'modal_price_mean': float(df['Modal_Price_Rs_Qtl'].mean()),
        'modal_price_median': float(df['Modal_Price_Rs_Qtl'].median())
    }
    return summary

def main():
    print("=== AGRIAGENT Data Processing Pipeline ===")
    audit_report = {
        'apy_dataset': prepare_apy_dataset(),
        'soil_health_dataset': prepare_soil_health_dataset(),
        'district_rainfall_dataset': prepare_district_rainfall_dataset(),
        'mandi_prices_dataset': prepare_mandi_prices_dataset()
    }

    report_path = os.path.join(METADATA_DIR, 'data_cleaning_audit.json')
    with open(report_path, 'w') as f:
        json.dump(audit_report, f, indent=2)
    print(f"\nProcessing complete! Audit report saved to {report_path}")

if __name__ == '__main__':
    main()
