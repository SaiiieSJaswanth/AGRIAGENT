from pathlib import Path
import pandas as pd

def test_historical_crop_records_load():
    path = Path("data/processed/clean_apy_yield.csv")
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) > 100000
    assert {"State_Name", "District_Name", "Crop_Year", "Season", "Crop", "Area", "Production"}.issubset(df.columns)
    assert df["Crop_Year"].max() == 2015

def test_removed_synthetic_dataset_is_absent():
    assert not Path("data/raw/dataset1.csv").exists()
