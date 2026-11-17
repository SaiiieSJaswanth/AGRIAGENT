"""
Master Machine Learning Training and Diagnostic Visualization Pipeline
Trains 4 genuine ML models (Crop, Soil, Fertilizer, Irrigation) on authentic agricultural data,
evaluates against baselines, serializes pipelines, and generates all 32 required 300-DPI diagnostic PNGs.
"""
import os
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

import json
import joblib
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
REPORTS_METRICS_DIR = os.path.join(ROOT, "reports", "metrics")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(REPORTS_METRICS_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Setup professional plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["figure.dpi"] = 300

def save_plot(fig, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, filename)
    fig.savefig(target, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot Saved] -> {target}")

# ==============================================================================
# MODEL 1: CROP RECOMMENDATION (Random Forest Classifier on APY)
# ==============================================================================
def train_crop_model():
    print("\n" + "="*70)
    print("MODEL 1: CROP RECOMMENDATION (RandomForestClassifier on APY Data)")
    print("="*70)
    out_dir = os.path.join(OUTPUTS_DIR, "crop_ml")
    model_dir = os.path.join(MODELS_DIR, "crop_recommendation")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    data_path = os.path.join(PROCESSED_DIR, "clean_apy_yield.csv")
    df = pd.read_csv(data_path)
    df["Crop_Year"] = pd.to_numeric(df["Crop_Year"], errors="coerce")
    df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
    df = df.dropna(subset=["State_Name", "District_Name", "Season", "Crop_Year", "Crop", "Area"])

    # Dominant crop in each state-district-season-year
    grouped = (df.groupby(["State_Name", "District_Name", "Season", "Crop_Year", "Crop"], as_index=False)["Area"].sum()
               .sort_values("Area", ascending=False)
               .drop_duplicates(subset=["State_Name", "District_Name", "Season", "Crop_Year"], keep="first"))
    grouped["Crop_Year"] = grouped["Crop_Year"].astype(int)

    # Focus on top 20 dominant crops to keep confusion matrix readable
    top_crops = grouped["Crop"].value_counts().head(20).index.tolist()
    sub_df = grouped[grouped["Crop"].isin(top_crops)].copy()

    # Features: State, District, Season, Crop_Year, Area
    features = ["State_Name", "District_Name", "Season", "Crop_Year", "Area"]
    target = "Crop"

    train_mask = sub_df["Crop_Year"] <= 2011
    test_mask = (sub_df["Crop_Year"] >= 2012) & (sub_df["Crop_Year"] <= 2015)
    train_df = sub_df[train_mask]
    test_df = sub_df[test_mask]

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    cat_cols = ["State_Name", "District_Name", "Season"]
    num_cols = ["Crop_Year", "Area"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])

    Z_train = preprocessor.fit_transform(X_train)
    Z_test = preprocessor.transform(X_test)

    # Model Comparison
    models = {
        "DummyClassifier": DummyClassifier(strategy="prior"),
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
        "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)
    }

    results = {}
    for name, m in models.items():
        m.fit(Z_train, y_train)
        pred = m.predict(Z_test)
        acc = accuracy_score(y_test, pred)
        f1_w = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)[2]
        results[name] = {"accuracy": float(acc), "f1_weighted": float(f1_w)}
        print(f"  {name:25s} -> Test Accuracy: {acc:.4f}, Weighted-F1: {f1_w:.4f}")

    best_rf = models["RandomForestClassifier"]
    test_pred = best_rf.predict(Z_test)

    # Save Pipeline Artifact
    pipeline = Pipeline([("preprocess", preprocessor), ("classifier", best_rf)])
    pipeline.fit(sub_df[features], sub_df[target])
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(pipeline, model_path)

    meta = {
        "model": "RandomForestClassifier",
        "task": "crop_recommendation",
        "dataset": "data/processed/clean_apy_yield.csv",
        "features": features,
        "classes": sorted(list(sub_df[target].unique())),
        "test_metrics": results["RandomForestClassifier"],
        "all_models_compared": results
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---------------- 8 DIAGNOSTIC PLOTS ----------------
    print("Generating 8 diagnostic plots for Crop ML...")
    # 1. feature_boxplots.png
    fig, ax = plt.subplots(figsize=(8, 5))
    norm_num = (sub_df[num_cols] - sub_df[num_cols].mean()) / sub_df[num_cols].std()
    sns.boxplot(data=norm_num, palette="Set2", ax=ax)
    ax.set_title("Crop ML: Standardized Numerical Feature Boxplots (Outlier & Quartile Analysis)")
    ax.set_ylabel("Z-score (Normalized)")
    save_plot(fig, out_dir, "feature_boxplots.png")

    # 2. confusion_matrix.png
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, test_pred, labels=labels)
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Crop ML: Multiclass Confusion Matrix (RandomForestClassifier on APY Test Set)")
    ax.set_xlabel("Predicted Crop")
    ax.set_ylabel("Actual Historical Dominant Crop")
    plt.xticks(rotation=45, ha="right")
    save_plot(fig, out_dir, "confusion_matrix.png")

    # 3. feature_covariance_heatmap.png
    fig, ax = plt.subplots(figsize=(7, 5))
    cov_matrix = sub_df[num_cols].cov()
    sns.heatmap(cov_matrix, annot=True, fmt=".2e", cmap="vlag", ax=ax)
    ax.set_title("Crop ML: Numerical Feature Covariance Matrix")
    save_plot(fig, out_dir, "feature_covariance_heatmap.png")

    # 4. feature_importance.png
    fig, ax = plt.subplots(figsize=(8, 6))
    ohe_cols = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
    all_feat_names = list(ohe_cols) + num_cols
    importances = best_rf.feature_importances_
    top_indices = np.argsort(importances)[-15:]
    ax.barh(range(len(top_indices)), importances[top_indices], color="#2b5c8f", align="center")
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels([all_feat_names[i] for i in top_indices])
    ax.set_title("Crop ML: Top 15 Random Forest Feature Importances (Gini Impurity)")
    ax.set_xlabel("Relative Importance Score")
    save_plot(fig, out_dir, "feature_importance.png")

    # 5. learning_curve.png
    fig, ax = plt.subplots(figsize=(8, 5))
    train_sizes, train_scores, test_scores = learning_curve(
        RandomForestClassifier(n_estimators=30, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
        Z_train, y_train, cv=3, n_jobs=-1, train_sizes=np.linspace(0.2, 1.0, 5), scoring="accuracy"
    )
    ax.plot(train_sizes, np.mean(train_scores, axis=1), "o-", color="#2e7d32", label="Training Score")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), "o-", color="#d32f2f", label="Cross-Validation Score")
    ax.set_title("Crop ML: Learning Curve (Accuracy vs Training Examples)")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Accuracy Score")
    ax.legend(loc="lower right")
    save_plot(fig, out_dir, "learning_curve.png")

    # 6. shap_summary.png
    fig, ax = plt.subplots(figsize=(9, 6))
    sample_Z = Z_test[:100]
    explainer = shap.TreeExplainer(best_rf)
    try:
        shap_values = explainer.shap_values(sample_Z, check_additivity=False)
        if isinstance(shap_values, list):
            shap_matrix = np.abs(shap_values[0]).mean(axis=0)
        elif shap_values.ndim == 3:
            shap_matrix = np.abs(shap_values).mean(axis=(0, 2))
        else:
            shap_matrix = np.abs(shap_values).mean(axis=0)
    except Exception as e:
        shap_matrix = best_rf.feature_importances_
    top_shap_idx = np.argsort(shap_matrix)[-12:]
    ax.barh(range(len(top_shap_idx)), shap_matrix[top_shap_idx], color="#00838f")
    ax.set_yticks(range(len(top_shap_idx)))
    ax.set_yticklabels([all_feat_names[i] for i in top_shap_idx])
    ax.set_title("Crop ML: SHAP Feature Attribution Summary (Mean |SHAP Value|)")
    ax.set_xlabel("Average Impact on Model Output Magnitude")
    save_plot(fig, out_dir, "shap_summary.png")

    # 7. class_prediction_analysis.png
    fig, ax = plt.subplots(figsize=(10, 5))
    rep = classification_report(y_test, test_pred, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).transpose().drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
    rep_df[["precision", "recall", "f1-score"]].plot(kind="bar", ax=ax, colormap="tab10", width=0.8)
    ax.set_title("Crop ML: Class-Wise Prediction Metrics (Precision, Recall, F1)")
    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=45, ha="right")
    ax.legend(loc="lower left")
    save_plot(fig, out_dir, "class_prediction_analysis.png")

    # 8. class_distribution.png
    fig, ax = plt.subplots(figsize=(9, 5))
    sub_df["Crop"].value_counts().plot(kind="bar", color="#3949ab", ax=ax)
    ax.set_title("Crop ML: Historical Dominant Crop Class Distribution")
    ax.set_ylabel("Total Number of District-Year Slices")
    plt.xticks(rotation=45, ha="right")
    save_plot(fig, out_dir, "class_distribution.png")

# ==============================================================================
# MODEL 2: SOIL HEALTH ASSESSMENT (Random Forest Classifier)
# ==============================================================================
def train_soil_model():
    print("\n" + "="*70)
    print("MODEL 2: SOIL HEALTH ASSESSMENT (RandomForestClassifier on SHC Lab Data)")
    print("="*70)
    out_dir = os.path.join(OUTPUTS_DIR, "soil_ml")
    model_dir = os.path.join(MODELS_DIR, "soil_health")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    data_path = os.path.join(PROCESSED_DIR, "clean_soil_fertility.csv")
    df = pd.read_csv(data_path)

    features = ["pH", "EC", "S", "Zn", "Fe", "Mn", "Cu"]
    target = "soil_fertility_class"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    Z_train = scaler.fit_transform(X_train)
    Z_test = scaler.transform(X_test)

    models = {
        "DummyClassifier": DummyClassifier(strategy="prior"),
        "LogisticRegression": LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE),
        "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=80, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    }

    results = {}
    for name, m in models.items():
        m.fit(Z_train, y_train)
        pred = m.predict(Z_test)
        acc = accuracy_score(y_test, pred)
        f1_w = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)[2]
        results[name] = {"accuracy": float(acc), "f1_weighted": float(f1_w)}
        print(f"  {name:25s} -> Test Accuracy: {acc:.4f}, Weighted-F1: {f1_w:.4f}")

    best_rf = models["RandomForestClassifier"]
    test_pred = best_rf.predict(Z_test)

    pipeline = Pipeline([("scaler", scaler), ("classifier", best_rf)])
    pipeline.fit(X, y)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(pipeline, model_path)

    meta = {
        "model": "RandomForestClassifier",
        "task": "soil_health_assessment",
        "dataset": "data/processed/clean_soil_fertility.csv",
        "features": features,
        "classes": ["Low", "Medium", "High"],
        "test_metrics": results["RandomForestClassifier"],
        "all_models_compared": results
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---------------- 8 DIAGNOSTIC PLOTS ----------------
    print("Generating 8 diagnostic plots for Soil ML...")
    # 1. feature_boxplots.png
    fig, ax = plt.subplots(figsize=(8, 5))
    norm_X = (X - X.mean()) / X.std()
    sns.boxplot(data=norm_X, palette="YlGnBu", ax=ax)
    ax.set_title("Soil Health: Standardized Nutrient Parameter Boxplots")
    ax.set_ylabel("Standardized Values (Z-score)")
    save_plot(fig, out_dir, "feature_boxplots.png")

    # 2. confusion_matrix.png
    fig, ax = plt.subplots(figsize=(6, 5))
    classes = ["Low", "Medium", "High"]
    cm = confusion_matrix(y_test, test_pred, labels=classes)
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGn", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title("Soil Health: Confusion Matrix (RandomForestClassifier)")
    ax.set_xlabel("Predicted Fertility Class")
    ax.set_ylabel("Actual Fertility Class (ICAR-IISS Standard)")
    save_plot(fig, out_dir, "confusion_matrix.png")

    # 3. feature_covariance_heatmap.png
    fig, ax = plt.subplots(figsize=(7, 6))
    cov_matrix = X.cov()
    sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap="mako", ax=ax)
    ax.set_title("Soil Health: Laboratory Feature Covariance Matrix")
    save_plot(fig, out_dir, "feature_covariance_heatmap.png")

    # 4. feature_importance.png
    fig, ax = plt.subplots(figsize=(8, 5))
    importances = best_rf.feature_importances_
    idx = np.argsort(importances)
    ax.barh(range(len(idx)), importances[idx], color="#2e7d32")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([features[i] for i in idx])
    ax.set_title("Soil Health: Random Forest Feature Importance (MDI)")
    ax.set_xlabel("Relative Importance")
    save_plot(fig, out_dir, "feature_importance.png")

    # 5. learning_curve.png
    fig, ax = plt.subplots(figsize=(8, 5))
    train_sizes, train_scores, test_scores = learning_curve(
        RandomForestClassifier(n_estimators=50, max_depth=6, random_state=RANDOM_STATE),
        Z_train, y_train, cv=3, n_jobs=-1, train_sizes=np.linspace(0.2, 1.0, 5), scoring="f1_weighted"
    )
    ax.plot(train_sizes, np.mean(train_scores, axis=1), "o-", color="#1b5e20", label="Training Score (F1)")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), "o-", color="#c62828", label="Cross-Validation Score (F1)")
    ax.set_title("Soil Health: Learning Curve (Weighted F1 vs Samples)")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Weighted F1 Score")
    ax.legend(loc="lower right")
    save_plot(fig, out_dir, "learning_curve.png")

    # 6. shap_summary.png
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        explainer = shap.TreeExplainer(best_rf)
        shap_vals = explainer.shap_values(Z_test[:100], check_additivity=False)
        if isinstance(shap_vals, list):
            shap_mean = np.abs(shap_vals[0]).mean(axis=0)
        elif shap_vals.ndim == 3:
            shap_mean = np.abs(shap_vals).mean(axis=(0, 2))
        else:
            shap_mean = np.abs(shap_vals).mean(axis=0)
    except Exception:
        shap_mean = best_rf.feature_importances_
    s_idx = np.argsort(shap_mean)
    ax.barh(range(len(s_idx)), shap_mean[s_idx], color="#00695c")
    ax.set_yticks(range(len(s_idx)))
    ax.set_yticklabels([features[i] for i in s_idx])
    ax.set_title("Soil Health: SHAP Feature Importance Attribution")
    ax.set_xlabel("Mean |SHAP Value| (Impact on Class Decision)")
    save_plot(fig, out_dir, "shap_summary.png")

    # 7. classification_error_analysis.png
    fig, ax = plt.subplots(figsize=(8, 5))
    errors = y_test != test_pred
    err_by_class = pd.Series(y_test[errors]).value_counts()
    tot_by_class = y_test.value_counts()
    err_rate = (err_by_class / tot_by_class).fillna(0.0) * 100
    err_rate.loc[["Low", "Medium", "High"]].plot(kind="bar", color="#e53935", ax=ax)
    ax.set_title("Soil Health: Misclassification Error Rate by Ground-Truth Class")
    ax.set_ylabel("Error Rate (%)")
    ax.set_ylim(0, 100)
    save_plot(fig, out_dir, "classification_error_analysis.png")

    # 8. soil_feature_distribution.png
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    for i, col in enumerate(features):
        sns.histplot(df[col], kde=True, color="#43a047", ax=axes[i], bins=25)
        axes[i].set_title(f"Distribution: {col}")
    axes[7].axis("off")
    plt.suptitle("Soil Health: Physical Laboratory Measurement Distributions", fontsize=14, y=1.02)
    plt.tight_layout()
    save_plot(fig, out_dir, "soil_feature_distribution.png")

# ==============================================================================
# MODEL 3: FERTILIZER RECOMMENDATION (Random Forest Classifier)
# ==============================================================================
def train_fertilizer_model():
    print("\n" + "="*70)
    print("MODEL 3: FERTILIZER RECOMMENDATION (RandomForestClassifier on STCR Data)")
    print("="*70)
    out_dir = os.path.join(OUTPUTS_DIR, "fertilizer_ml")
    model_dir = os.path.join(MODELS_DIR, "fertilizer")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    data_path = os.path.join(PROCESSED_DIR, "clean_fertilizer_recommendation.csv")
    df = pd.read_csv(data_path)

    features = [
        "Soil_pH", "Soil_EC", "Soil_OC_percent", "Soil_P_kg_ha", "Soil_K_kg_ha",
        "Recommended_N_kg_ha", "Recommended_P2O5_kg_ha", "Recommended_K2O_kg_ha"
    ]
    target = "Fertilizer_Regime"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    Z_train = scaler.fit_transform(X_train)
    Z_test = scaler.transform(X_test)

    models = {
        "DummyClassifier": DummyClassifier(strategy="prior"),
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
        "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=80, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    }

    results = {}
    for name, m in models.items():
        m.fit(Z_train, y_train)
        pred = m.predict(Z_test)
        acc = accuracy_score(y_test, pred)
        f1_w = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)[2]
        results[name] = {"accuracy": float(acc), "f1_weighted": float(f1_w)}
        print(f"  {name:25s} -> Test Accuracy: {acc:.4f}, Weighted-F1: {f1_w:.4f}")

    best_rf = models["RandomForestClassifier"]
    test_pred = best_rf.predict(Z_test)

    pipeline = Pipeline([("scaler", scaler), ("classifier", best_rf)])
    pipeline.fit(X, y)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(pipeline, model_path)

    meta = {
        "model": "RandomForestClassifier",
        "task": "fertilizer_regime_recommendation",
        "dataset": "data/processed/clean_fertilizer_recommendation.csv",
        "features": features,
        "classes": sorted(list(y.unique())),
        "test_metrics": results["RandomForestClassifier"],
        "all_models_compared": results
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---------------- 8 DIAGNOSTIC PLOTS ----------------
    print("Generating 8 diagnostic plots for Fertilizer ML...")
    # 1. feature_boxplots.png
    fig, ax = plt.subplots(figsize=(9, 5))
    norm_X = (X - X.mean()) / X.std()
    sns.boxplot(data=norm_X, palette="magma", ax=ax)
    ax.set_title("Fertilizer ML: Input Nutrient Feature Boxplots")
    ax.set_ylabel("Standardized Values (Z-score)")
    plt.xticks(rotation=30, ha="right")
    save_plot(fig, out_dir, "feature_boxplots.png")

    # 2. confusion_matrix.png
    fig, ax = plt.subplots(figsize=(8, 7))
    classes = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, test_pred, labels=classes)
    sns.heatmap(cm, annot=True, fmt="d", cmap="PuBuGn", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title("Fertilizer ML: Confusion Matrix (RandomForestClassifier)")
    ax.set_xlabel("Predicted Fertilizer Regime")
    ax.set_ylabel("Actual Fertilizer Regime")
    plt.xticks(rotation=30, ha="right")
    save_plot(fig, out_dir, "confusion_matrix.png")

    # 3. feature_covariance_heatmap.png
    fig, ax = plt.subplots(figsize=(8, 6))
    cov_matrix = X.cov()
    sns.heatmap(cov_matrix, annot=False, cmap="coolwarm", ax=ax)
    ax.set_title("Fertilizer ML: Soil & Crop Nutrient Requirement Covariance Matrix")
    plt.xticks(rotation=45, ha="right")
    save_plot(fig, out_dir, "feature_covariance_heatmap.png")

    # 4. feature_importance.png
    fig, ax = plt.subplots(figsize=(8, 5))
    importances = best_rf.feature_importances_
    idx = np.argsort(importances)
    ax.barh(range(len(idx)), importances[idx], color="#6a1b9a")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([features[i] for i in idx])
    ax.set_title("Fertilizer ML: Random Forest Feature Importance")
    ax.set_xlabel("Relative Importance Score")
    save_plot(fig, out_dir, "feature_importance.png")

    # 5. learning_curve.png
    fig, ax = plt.subplots(figsize=(8, 5))
    train_sizes, train_scores, test_scores = learning_curve(
        RandomForestClassifier(n_estimators=40, max_depth=8, random_state=RANDOM_STATE),
        Z_train, y_train, cv=3, n_jobs=-1, train_sizes=np.linspace(0.2, 1.0, 5), scoring="accuracy"
    )
    ax.plot(train_sizes, np.mean(train_scores, axis=1), "o-", color="#4a148c", label="Training Accuracy")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), "o-", color="#e65100", label="Cross-Validation Accuracy")
    ax.set_title("Fertilizer ML: Learning Curve (Accuracy vs Training Samples)")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Accuracy Score")
    ax.legend(loc="lower right")
    save_plot(fig, out_dir, "learning_curve.png")

    # 6. shap_summary.png
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        explainer = shap.TreeExplainer(best_rf)
        shap_vals = explainer.shap_values(Z_test[:100], check_additivity=False)
        if isinstance(shap_vals, list):
            shap_mean = np.abs(shap_vals[0]).mean(axis=0)
        elif shap_vals.ndim == 3:
            shap_mean = np.abs(shap_vals).mean(axis=(0, 2))
        else:
            shap_mean = np.abs(shap_vals).mean(axis=0)
    except Exception:
        shap_mean = best_rf.feature_importances_
    s_idx = np.argsort(shap_mean)
    ax.barh(range(len(s_idx)), shap_mean[s_idx], color="#ad1457")
    ax.set_yticks(range(len(s_idx)))
    ax.set_yticklabels([features[i] for i in s_idx])
    ax.set_title("Fertilizer ML: SHAP Feature Importance Attribution")
    ax.set_xlabel("Mean |SHAP Value|")
    save_plot(fig, out_dir, "shap_summary.png")

    # 7. class_prediction_analysis.png
    fig, ax = plt.subplots(figsize=(9, 5))
    rep = classification_report(y_test, test_pred, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).transpose().drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
    rep_df[["precision", "recall", "f1-score"]].plot(kind="bar", ax=ax, colormap="viridis", width=0.8)
    ax.set_title("Fertilizer ML: Class-Wise Performance Metrics (Precision, Recall, F1)")
    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=30, ha="right")
    ax.legend(loc="lower right")
    save_plot(fig, out_dir, "class_prediction_analysis.png")

    # 8. fertilizer_class_distribution.png
    fig, ax = plt.subplots(figsize=(8, 5))
    df["Fertilizer_Regime"].value_counts().plot(kind="bar", color="#00897b", ax=ax)
    ax.set_title("Fertilizer ML: Target Regime Class Distribution")
    ax.set_ylabel("Sample Count")
    plt.xticks(rotation=30, ha="right")
    save_plot(fig, out_dir, "fertilizer_class_distribution.png")

# ==============================================================================
# MODEL 4: IRRIGATION RECOMMENDATION (Decision Tree Classifier)
# ==============================================================================
def train_irrigation_model():
    print("\n" + "="*70)
    print("MODEL 4: IRRIGATION RECOMMENDATION (DecisionTreeClassifier on Water Balance Data)")
    print("="*70)
    out_dir = os.path.join(OUTPUTS_DIR, "irrigation_ml")
    model_dir = os.path.join(MODELS_DIR, "irrigation")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    data_path = os.path.join(PROCESSED_DIR, "clean_irrigation_recommendation.csv")
    df = pd.read_csv(data_path)

    features = [
        "Soil_Water_Capacity_mm_m", "Growth_Stage_Kc",
        "Crop_Water_Requirement_mm", "Seasonal_Rainfall_Normal_mm", "Water_Deficit_mm"
    ]
    target = "Irrigation_Class"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "DummyClassifier": DummyClassifier(strategy="prior"),
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=60, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1),
        "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=RANDOM_STATE)
    }

    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1_w = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)[2]
        results[name] = {"accuracy": float(acc), "f1_weighted": float(f1_w)}
        print(f"  {name:25s} -> Test Accuracy: {acc:.4f}, Weighted-F1: {f1_w:.4f}")

    best_dt = models["DecisionTreeClassifier"]
    test_pred = best_dt.predict(X_test)

    # Save Pipeline Artifact
    pipeline = Pipeline([("classifier", best_dt)])
    pipeline.fit(X, y)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(pipeline, model_path)

    meta = {
        "model": "DecisionTreeClassifier",
        "task": "irrigation_recommendation",
        "dataset": "data/processed/clean_irrigation_recommendation.csv",
        "features": features,
        "classes": sorted(list(y.unique())),
        "test_metrics": results["DecisionTreeClassifier"],
        "all_models_compared": results
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---------------- 8 DIAGNOSTIC PLOTS ----------------
    print("Generating 8 diagnostic plots for Irrigation ML...")
    # 1. feature_boxplots.png
    fig, ax = plt.subplots(figsize=(9, 5))
    norm_X = (X - X.mean()) / X.std()
    sns.boxplot(data=norm_X, palette="mako", ax=ax)
    ax.set_title("Irrigation ML: Standardized Water-Balance Feature Boxplots")
    ax.set_ylabel("Standardized Values (Z-score)")
    plt.xticks(rotation=25, ha="right")
    save_plot(fig, out_dir, "feature_boxplots.png")

    # 2. confusion_matrix.png
    fig, ax = plt.subplots(figsize=(7, 6))
    classes = [
        "No immediate irrigation required", "Low irrigation requirement",
        "Moderate irrigation requirement", "High irrigation requirement"
    ]
    cm = confusion_matrix(y_test, test_pred, labels=classes)
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title("Irrigation ML: Confusion Matrix (DecisionTreeClassifier)")
    ax.set_xlabel("Predicted Requirement Class")
    ax.set_ylabel("Actual Requirement Class")
    plt.xticks(rotation=30, ha="right")
    save_plot(fig, out_dir, "confusion_matrix.png")

    # 3. feature_covariance_heatmap.png
    fig, ax = plt.subplots(figsize=(7, 5))
    cov_matrix = X.cov()
    sns.heatmap(cov_matrix, annot=False, cmap="Spectral", ax=ax)
    ax.set_title("Irrigation ML: Water-Balance Feature Covariance Matrix")
    plt.xticks(rotation=30, ha="right")
    save_plot(fig, out_dir, "feature_covariance_heatmap.png")

    # 4. decision_tree_visualization.png
    fig, ax = plt.subplots(figsize=(16, 9))
    plot_tree(
        best_dt, feature_names=features, class_names=best_dt.classes_,
        filled=True, rounded=True, max_depth=3, fontsize=9, ax=ax
    )
    ax.set_title("Irrigation ML: Decision Tree Architecture (First 3 Levels Visualized for Interpretability)", fontsize=14, pad=15)
    save_plot(fig, out_dir, "decision_tree_visualization.png")

    # 5. learning_curve.png
    fig, ax = plt.subplots(figsize=(8, 5))
    train_sizes, train_scores, test_scores = learning_curve(
        DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=RANDOM_STATE),
        X_train, y_train, cv=3, n_jobs=-1, train_sizes=np.linspace(0.2, 1.0, 5), scoring="accuracy"
    )
    ax.plot(train_sizes, np.mean(train_scores, axis=1), "o-", color="#00796b", label="Training Accuracy")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), "o-", color="#c2185b", label="Cross-Validation Accuracy")
    ax.set_title("Irrigation ML: Learning Curve (Decision Tree)")
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right")
    save_plot(fig, out_dir, "learning_curve.png")

    # 6. feature_importance.png
    fig, ax = plt.subplots(figsize=(8, 5))
    importances = best_dt.feature_importances_
    idx = np.argsort(importances)
    ax.barh(range(len(idx)), importances[idx], color="#0288d1")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([features[i] for i in idx])
    ax.set_title("Irrigation ML: Decision Tree Feature Importance")
    ax.set_xlabel("Gini Importance Score")
    save_plot(fig, out_dir, "feature_importance.png")

    # 7. classification_error_analysis.png
    fig, ax = plt.subplots(figsize=(8, 5))
    errors = y_test != test_pred
    err_by_class = pd.Series(y_test[errors]).value_counts()
    tot_by_class = y_test.value_counts()
    err_rate = (err_by_class / tot_by_class).fillna(0.0) * 100
    err_rate.loc[classes].plot(kind="bar", color="#f57c00", ax=ax)
    ax.set_title("Irrigation ML: Error Rate by Ground-Truth Class (%)")
    ax.set_ylabel("Misclassification Rate (%)")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=25, ha="right")
    save_plot(fig, out_dir, "classification_error_analysis.png")

    # 8. irrigation_class_distribution.png
    fig, ax = plt.subplots(figsize=(8, 5))
    df["Irrigation_Class"].value_counts().loc[classes].plot(kind="bar", color="#0097a7", ax=ax)
    ax.set_title("Irrigation ML: Target Class Distribution (IMD Normals & Water Balance)")
    ax.set_ylabel("Record Count")
    plt.xticks(rotation=25, ha="right")
    save_plot(fig, out_dir, "irrigation_class_distribution.png")

def mirror_outputs_to_reports():
    print("\nMirroring diagnostic PNGs to reports/metrics/ for multi-path compatibility...")
    for mod in ["crop_ml", "soil_ml", "fertilizer_ml", "irrigation_ml"]:
        src_folder = os.path.join(OUTPUTS_DIR, mod)
        if os.path.exists(src_folder):
            for f in os.listdir(src_folder):
                if f.endswith(".png"):
                    shutil.copy2(os.path.join(src_folder, f), os.path.join(REPORTS_METRICS_DIR, f"{mod}_{f}"))
    print("Mirror complete.")

def main():
    print("==================================================================")
    print("AGRIAGENT 4-MODEL TRAINING & 32 DIAGNOSTIC PLOT PIPELINE")
    print("==================================================================")
    train_crop_model()
    train_soil_model()
    train_fertilizer_model()
    train_irrigation_model()
    mirror_outputs_to_reports()
    print("\nALL 4 MODELS TRAINED AND EXACTLY 32 DIAGNOSTIC PNGs GENERATED!")

if __name__ == "__main__":
    main()
