import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Transparent, non-target-derived ratios only when source columns exist.
    if {"sand_percent", "clay_percent"}.issubset(out.columns):
        out["sand_clay_ratio"] = out["sand_percent"] / out["clay_percent"].replace(0, pd.NA)
    if {"soil_moisture_fc_percent", "soil_moisture_wp_percent"}.issubset(out.columns):
        out["available_moisture_range"] = out["soil_moisture_fc_percent"] - out["soil_moisture_wp_percent"]
    return out
