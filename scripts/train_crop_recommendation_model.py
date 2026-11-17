"""Train the experimental APY-based crop recommendation classifier.

The model predicts the historically dominant crop for a State-District-Season-Year
record. Its inference is executed internally only; farmer-facing recommendations
continue to use the transparent historical APY area-ranking function.
"""
from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "clean_apy_yield.csv"
MODEL_DIR = ROOT / "models" / "crop_recommendation"
MODEL_PATH = MODEL_DIR / "model.joblib"
META_PATH = MODEL_DIR / "metadata.json"
FEATURES = ["State_Name", "District_Name", "Season", "Crop_Year"]
CATEGORICAL = ["State_Name", "District_Name", "Season"]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Required APY dataset not found: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    required = set(FEATURES + ["Crop", "Area"])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"APY dataset is missing required columns: {sorted(missing)}")

    data["Crop_Year"] = pd.to_numeric(data["Crop_Year"], errors="coerce")
    data["Area"] = pd.to_numeric(data["Area"], errors="coerce")
    data = data.dropna(subset=FEATURES + ["Crop", "Area"])
    # Label = crop with the largest recorded cultivated area in each local-year slice.
    grouped = (data.groupby(FEATURES + ["Crop"], as_index=False)["Area"].sum()
               .sort_values("Area", ascending=False)
               .drop_duplicates(subset=FEATURES, keep="first"))
    grouped["Crop_Year"] = grouped["Crop_Year"].astype(int)

    train = grouped[grouped["Crop_Year"] <= 2011]
    test = grouped[(grouped["Crop_Year"] >= 2012) & (grouped["Crop_Year"] <= 2015)]
    if train.empty or test.empty:
        raise ValueError("Chronological train/test split is empty; inspect APY year coverage.")

    def make_model():
        prep = ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("year", "passthrough", ["Crop_Year"]),
        ])
        clf = RandomForestClassifier(
            n_estimators=120, class_weight="balanced_subsample",
            min_samples_leaf=2, random_state=42, n_jobs=-1,
        )
        return Pipeline([("preprocess", prep), ("classifier", clf)])

    evaluation_model = make_model()
    evaluation_model.fit(train[FEATURES], train["Crop"])
    predictions = evaluation_model.predict(test[FEATURES])
    accuracy = float(accuracy_score(test["Crop"], predictions))
    macro_f1 = float(f1_score(test["Crop"], predictions, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(test["Crop"], predictions, average="weighted", zero_division=0))

    final_model = make_model()
    final_model.fit(grouped[FEATURES], grouped["Crop"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    metadata = {
        "model": "RandomForestClassifier",
        "task": "experimental crop recommendation benchmark",
        "label_definition": "Largest total cultivated area crop within each State-District-Season-Crop_Year group",
        "features": FEATURES,
        "dataset": "data/processed/clean_apy_yield.csv",
        "split": "Chronological: train <=2011; test 2012-2015",
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "final_training_rows": int(len(grouped)),
        "final_crop_classes": int(grouped["Crop"].nunique()),
        "final_training_year_range": [int(grouped["Crop_Year"].min()), int(grouped["Crop_Year"].max())],
        "display_policy": "Internal experimental prediction only; never shown in farmer-facing terminal report; does not alter APY historical ranking.",
        "caution": "This learns historical dominant-crop patterns, not agronomic suitability or guaranteed farm-level outcomes."
    }
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Experimental crop classifier trained and saved.")
    print(f"Chronological test accuracy: {accuracy:.4f}; macro-F1: {macro_f1:.4f}; weighted-F1: {weighted_f1:.4f}")
    print("Predictions remain hidden during normal farmer-facing application runs.")


if __name__ == "__main__":
    main()
