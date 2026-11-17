import pandas as pd
import numpy as np

class LocalContext:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def resolve(self, state, district, soil_type):
        checks = [
            ("state_district_soil", (self.df.state == state) & (self.df.district == district) & (self.df.soil_type == soil_type)),
            ("state_soil", (self.df.state == state) & (self.df.soil_type == soil_type)),
            ("state", self.df.state == state),
            ("soil", self.df.soil_type == soil_type),
        ]
        for level, mask in checks:
            subset = self.df.loc[mask]
            if not subset.empty:
                return level, subset
        return "global", self.df

    def summary(self, subset, global_df):
        numeric = [c for c in [
            "ph", "organic_carbon_percent", "annual_rainfall_mm",
            "water_availability_index", "soil_health_index",
            "nitrogen_kg_ha", "phosphorus_kg_ha", "potassium_kg_ha"
        ] if c in subset.columns]
        out = {"matching_records": int(len(subset)), "metrics": {}}
        for c in numeric:
            out["metrics"][c] = {
                "local_mean": float(subset[c].mean()),
                "global_mean": float(global_df[c].mean()),
            }
        for c in ["erosion_risk", "flood_risk", "drought_risk", "land_degradation_risk"]:
            if c in subset:
                out["metrics"][c] = subset[c].value_counts().to_dict()
        return out
