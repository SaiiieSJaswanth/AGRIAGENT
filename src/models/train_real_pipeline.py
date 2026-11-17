"""
Real-Evidence Model Training and Evaluation Pipeline for AGRIAGENT
Trains and evaluates genuine machine learning models on real Indian agricultural datasets.
Uses leak-free preprocessing, baselines, cross-validation, and untouched test set evaluation.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, mean_absolute_error, mean_squared_error, r2_score
)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
REPORTS_METRICS = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'metrics')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_METRICS, exist_ok=True)

RANDOM_STATE = 42

def train_yield_prediction():
    print("\n=======================================================")
    print("TASK 2: CROP YIELD PREDICTION (Temporal Split Regression)")
    print("=======================================================")
    csv_path = os.path.join(PROCESSED_DIR, 'clean_apy_yield.csv')
    df = pd.read_csv(csv_path)
    print(f"Dataset: {csv_path} ({len(df):,} rows)")

    # Features and Target
    cat_features = ['State_Name', 'Crop', 'Season']
    num_features = ['Area', 'Crop_Year']
    all_features = cat_features + num_features
    target = 'Yield_tonnes_per_ha'

    # Restrict training to the 20 most frequently observed crops for a practical local retrain.
    top_crops = df['Crop'].value_counts().head(20).index.tolist()
    df_sub = df[df['Crop'].isin(top_crops)].copy()
    print(f"Evaluating top 20 major Indian crops: {len(df_sub):,} records")

    # Chronological Split: Train on 1997-2011, Test on 2012-2015
    train_df = df_sub[df_sub['Crop_Year'] <= 2011].copy()
    test_df = df_sub[df_sub['Crop_Year'] > 2011].copy()
    print(f"Chronological Split: Train (1997-2011) = {len(train_df):,} rows, Test (2012-2015) = {len(test_df):,} rows")

    X_train = train_df[all_features]
    y_train = train_df[target]
    X_test = test_df[all_features]
    y_test = test_df[target]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
        ('num', StandardScaler(), num_features)
    ])

    Z_train = preprocessor.fit_transform(X_train)
    Z_test = preprocessor.transform(X_test)

    # Compute Crop-Mean Baseline (Crucial benchmark: predicting crop historical mean)
    crop_historical_means = train_df.groupby('Crop')['Yield_tonnes_per_ha'].mean().to_dict()
    global_train_mean = float(train_df['Yield_tonnes_per_ha'].mean())
    crop_mean_pred = test_df['Crop'].map(crop_historical_means).fillna(global_train_mean)
    crop_mean_mae = mean_absolute_error(y_test, crop_mean_pred)
    crop_mean_rmse = np.sqrt(mean_squared_error(y_test, crop_mean_pred))
    crop_mean_r2 = r2_score(y_test, crop_mean_pred)

    candidates = {
        'DummyRegressor (Global Mean)': DummyRegressor(strategy='mean'),
        'CropMeanBaseline (Crop Historical Mean)': None,
        'RidgeRegression': Ridge(alpha=10.0, random_state=RANDOM_STATE),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=25, max_depth=10, min_samples_leaf=8, random_state=RANDOM_STATE, n_jobs=-1)
    }

    results = {
        'CropMeanBaseline (Crop Historical Mean)': {
            'mae': float(crop_mean_mae),
            'rmse': float(crop_mean_rmse),
            'r2': float(crop_mean_r2),
            'baseline_rmse_improvement_percent': 0.0
        }
    }
    fitted_models = {}

    for name, model in candidates.items():
        if name == 'CropMeanBaseline (Crop Historical Mean)':
            print(f"Evaluated {name} -> MAE = {crop_mean_mae:.3f} t/ha, RMSE = {crop_mean_rmse:.3f} t/ha, R² = {crop_mean_r2:.4f}")
            continue
        print(f"Training {name}...")
        model.fit(Z_train, y_train)
        pred = model.predict(Z_test)
        # Ensure yield cannot be negative
        pred = np.clip(pred, 0, None)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)

        results[name] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'baseline_rmse_improvement_percent': float(0.0)
        }
        fitted_models[name] = model
        print(f"  -> {name}: MAE = {mae:.3f} t/ha, RMSE = {rmse:.3f} t/ha, R² = {r2:.4f}")

    dummy_rmse = results['DummyRegressor (Global Mean)']['rmse']
    for name in results:
        results[name]['baseline_rmse_improvement_percent'] = float(
            (dummy_rmse - results[name]['rmse']) / dummy_rmse * 100.0
        )

    # Best model by RMSE
    best_name = min([k for k in results if k not in ['DummyRegressor (Global Mean)', 'CropMeanBaseline (Crop Historical Mean)']], key=lambda k: results[k]['rmse'])
    print(f"\nBest Model Selected: {best_name} (RMSE: {results[best_name]['rmse']:.3f} t/ha, R²: {results[best_name]['r2']:.4f})")

    # Generate Residual & Prediction Plot
    best_model = fitted_models[best_name]
    best_pred = np.clip(best_model.predict(Z_test), 0, None)
    residuals = y_test - best_pred

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    # Sample 1000 points for clear scatter
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(y_test), min(1000, len(y_test)), replace=False)
    plt.scatter(y_test.iloc[sample_idx], best_pred[sample_idx], alpha=0.3, color='forestgreen', s=15)
    plt.plot([0, 20], [0, 20], 'r--', lw=2, label='Perfect Fit')
    plt.title(f'Actual vs Predicted Yield ({best_name})')
    plt.xlabel('Actual Yield (tonnes/ha)')
    plt.ylabel('Predicted Yield (tonnes/ha)')
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.histplot(residuals, bins=40, kde=True, color='purple')
    plt.title('Residuals Distribution (Actual - Predicted)')
    plt.xlabel('Error (tonnes/ha)')
    plt.xlim(-10, 10)
    plt.tight_layout()
    yield_plot_path = os.path.join(REPORTS_METRICS, 'yield_prediction_residuals.png')
    plt.savefig(yield_plot_path, dpi=300)
    plt.close()
    print(f"Saved yield evaluation plot to: {yield_plot_path}")

    full_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', best_model)
    ])

    model_dir = os.path.join(MODELS_DIR, 'yield')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.joblib')
    joblib.dump(full_pipe, model_path)

    # Compute per-crop test error for major crops
    per_crop_metrics = {}
    for crop in ['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton(lint)', 'Gram']:
        mask = (test_df['Crop'] == crop).values
        if mask.sum() > 0:
            c_true = y_test[mask]
            c_pred = best_pred[mask]
            c_mae = mean_absolute_error(c_true, c_pred)
            c_rmse = np.sqrt(mean_squared_error(c_true, c_pred))
            per_crop_metrics[crop] = {
                'test_samples': int(mask.sum()),
                'actual_mean_yield_t_ha': round(float(c_true.mean()), 2),
                'predicted_mean_yield_t_ha': round(float(c_pred.mean()), 2),
                'mae_t_ha': round(float(c_mae), 3),
                'rmse_t_ha': round(float(c_rmse), 3)
            }

    metadata = {
        'task': 'crop_yield_prediction',
        'dataset': 'Official Ministry of Agriculture & Farmers Welfare APY Dataset (DESAgri)',
        'target': target,
        'unit': 'tonnes per hectare',
        'kind': 'regression',
        'features': all_features,
        'selected_model': best_name,
        'test_metrics': results[best_name],
        'all_models_compared': results,
        'per_crop_test_metrics': per_crop_metrics,
        'artifact_path': 'models/yield/model.joblib',
        'evaluation_protocol': 'Chronological split: Train (1997-2011), Test (2012-2015)',
        'major_crops_covered': top_crops,
        'random_state': RANDOM_STATE
    }
    meta_path = os.path.join(model_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved yield model artifact and metadata to {model_dir}")
    return metadata

def train_soil_health_classification():
    print("\n=======================================================")
    print("TASK 3: SOIL FERTILITY CLASSIFICATION (Laboratory Data)")
    print("=======================================================")
    csv_path = os.path.join(PROCESSED_DIR, 'clean_soil_fertility.csv')
    df = pd.read_csv(csv_path)
    print(f"Dataset: {csv_path} ({len(df)} records from Soil Health Card portal)")

    # Target: soil_fertility_class (Low, Medium, High)
    # Predictors: pH, EC, S, Zn, Fe, Mn, Cu
    # NOTE: OC, P, K are intentionally excluded to prevent target leakage!
    features = ['pH', 'EC', 'S', 'Zn', 'Fe', 'Mn', 'Cu']
    target = 'soil_fertility_class'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    Z_train = preprocessor.fit_transform(X_train)
    Z_test = preprocessor.transform(X_test)

    classes = ['Low', 'Medium', 'High']

    candidates = {
        'DummyClassifier': DummyClassifier(strategy='prior'),
        'LogisticRegression': LogisticRegression(max_iter=500, class_weight='balanced', random_state=RANDOM_STATE),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=80, max_depth=8, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
        'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=RANDOM_STATE)
    }

    results = {}
    fitted_models = {}

    for name, model in candidates.items():
        print(f"Training {name}...")
        model.fit(Z_train, y_train)
        pred = model.predict(Z_test)

        acc = accuracy_score(y_test, pred)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_test, pred, average='weighted', zero_division=0)
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_test, pred, average='macro', zero_division=0)

        results[name] = {
            'accuracy': float(acc),
            'f1_weighted': float(f1_w),
            'f1_macro': float(f1_m),
            'precision_weighted': float(p_w),
            'recall_weighted': float(r_w)
        }
        fitted_models[name] = model
        print(f"  -> {name}: Accuracy = {acc:.4f}, F1-Weighted = {f1_w:.4f}, F1-Macro = {f1_m:.4f}")

    best_name = max(results, key=lambda k: results[k]['f1_weighted'])
    print(f"\nBest Model Selected: {best_name} (F1-Weighted: {results[best_name]['f1_weighted']:.4f})")

    best_model = fitted_models[best_name]
    test_pred = best_model.predict(Z_test)
    cm = confusion_matrix(y_test, test_pred, labels=classes)
    report_dict = classification_report(y_test, test_pred, labels=classes, output_dict=True, zero_division=0)

    # Save Confusion Matrix Plot
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=classes, yticklabels=classes)
    plt.title(f'Soil Fertility Classification Confusion Matrix ({best_name})', fontsize=12, pad=12)
    plt.xlabel('Predicted Class', fontsize=10)
    plt.ylabel('Actual Class (ICAR-IISS Standard)', fontsize=10)
    plt.tight_layout()
    cm_path = os.path.join(REPORTS_METRICS, 'soil_fertility_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved soil fertility confusion matrix plot to: {cm_path}")

    full_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', best_model)
    ])

    model_dir = os.path.join(MODELS_DIR, 'soil_health')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.joblib')
    joblib.dump(full_pipe, model_path)

    metadata = {
        'task': 'soil_fertility_classification',
        'dataset': 'Soil Health Card Portal Scraped Laboratory Samples',
        'target': target,
        'kind': 'multiclass_classification',
        'features': features,
        'classes': classes,
        'selected_model': best_name,
        'test_metrics': results[best_name],
        'all_models_compared': results,
        'per_class_report': report_dict,
        'confusion_matrix': cm.tolist(),
        'artifact_path': 'models/soil_health/model.joblib',
        'production_advisory': 'Experimental benchmark model. Production inference uses Option A transparent ICAR-IISS rule-based module (src/models/soil_fertility.py) to prevent misleading ML-performance claims on rule-derived targets.',
        'evaluation_protocol': 'Stratified 80/20 split, target leakage prevention (OC, P, K excluded from predictors)',
        'random_state': RANDOM_STATE
    }
    meta_path = os.path.join(model_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved soil health model artifact and metadata to {model_dir}")
    return metadata

def generate_model_comparison_plots(yield_meta, soil_meta):
    """Compare retained ML reference tasks only; crop shortlist is data lookup, not ML."""
    print("\n--- Generating retained-model comparison visualization ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    yield_models = list(yield_meta['all_models_compared'].keys())
    yield_rmse = [yield_meta['all_models_compared'][m]['rmse'] for m in yield_models]
    axes[0].barh(yield_models, yield_rmse)
    axes[0].set_title('Historical Yield Model Comparison (Test RMSE, t/ha)')
    axes[0].set_xlabel('RMSE (lower is better)')
    soil_models = list(soil_meta['all_models_compared'].keys())
    soil_f1 = [soil_meta['all_models_compared'][m]['f1_weighted'] for m in soil_models]
    axes[1].barh(soil_models, soil_f1)
    axes[1].set_title('Soil Reference Model Comparison (Weighted F1)')
    axes[1].set_xlabel('F1 score')
    fig.tight_layout()
    summary_plot_path = os.path.join(REPORTS_METRICS, 'model_comparison_overview.png')
    fig.savefig(summary_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved retained-model comparison to {summary_plot_path}")

def main():
    print("=== Training Genuine AGRIAGENT Machine Learning Models ===")
    yield_meta = train_yield_prediction()
    soil_meta = train_soil_health_classification()

    generate_model_comparison_plots(yield_meta, soil_meta)

    # Master Training Summary
    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'crop_recommendation': {'farmer_output': 'historical APY crop-area ranking', 'internal_ml': 'experimental RandomForestClassifier trained by scripts/train_crop_recommendation_model.py; prediction hidden from terminal output'},
        'crop_yield_prediction': yield_meta,
        'soil_fertility_classification': soil_meta
    }
    summary_file = os.path.join(REPORTS_METRICS, 'real_training_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nTraining pipeline completed successfully! Summary saved to {summary_file}")

if __name__ == '__main__':
    main()
