from pathlib import Path
import pandas as pd
from src.engine.inference import recommend_crops_from_history

def test_crop_shortlist_comes_from_real_historical_records():
    result = recommend_crops_from_history("Karnataka", "Bagalkote", "Kharif")
    assert result["status"] == "success"
    assert result["dataset_provenance"].startswith("Historical district/season crop area")
    assert all("confidence_percent" not in row for row in result["ranked_recommendations"])

def test_no_synthetic_data_file_or_legacy_switch():
    assert not Path("data/raw/dataset1.csv").exists()
    content = Path("train_pipeline.py").read_text(encoding="utf-8")
    assert "--legacy" not in content
