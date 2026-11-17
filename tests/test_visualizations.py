"""
Tests for 32 Diagnostic PNG Visualizations in AGRIAGENT.
Verifies the existence, non-zero file size, and correct format of all 32 diagnostic plots
across outputs/ and reports/metrics/.
"""
from pathlib import Path
import pytest

MODULE_PLOTS = {
    "crop_ml": [
        "feature_boxplots.png",
        "confusion_matrix.png",
        "feature_covariance_heatmap.png",
        "feature_importance.png",
        "learning_curve.png",
        "shap_summary.png",
        "class_prediction_analysis.png",
        "class_distribution.png"
    ],
    "soil_ml": [
        "feature_boxplots.png",
        "confusion_matrix.png",
        "feature_covariance_heatmap.png",
        "feature_importance.png",
        "learning_curve.png",
        "shap_summary.png",
        "classification_error_analysis.png",
        "soil_feature_distribution.png"
    ],
    "fertilizer_ml": [
        "feature_boxplots.png",
        "confusion_matrix.png",
        "feature_covariance_heatmap.png",
        "feature_importance.png",
        "learning_curve.png",
        "shap_summary.png",
        "class_prediction_analysis.png",
        "fertilizer_class_distribution.png"
    ],
    "irrigation_ml": [
        "feature_boxplots.png",
        "confusion_matrix.png",
        "feature_covariance_heatmap.png",
        "decision_tree_visualization.png",
        "learning_curve.png",
        "feature_importance.png",
        "classification_error_analysis.png",
        "irrigation_class_distribution.png"
    ]
}

def test_all_32_plots_exist_in_outputs():
    outputs_dir = Path("outputs")
    assert outputs_dir.exists(), "outputs/ directory must exist"

    total_count = 0
    for module_name, plot_files in MODULE_PLOTS.items():
        module_dir = outputs_dir / module_name
        assert module_dir.exists(), f"outputs/{module_name} must exist"
        assert len(plot_files) == 8, f"Expected 8 plots for {module_name}"

        for plot_file in plot_files:
            file_path = module_dir / plot_file
            assert file_path.exists(), f"Missing plot: {file_path}"
            size = file_path.stat().st_size
            assert size > 5000, f"Plot {file_path} is too small or empty ({size} bytes)"
            total_count += 1

    assert total_count == 32, f"Expected 32 plots in outputs/, found {total_count}"

def test_all_32_plots_mirrored_in_reports_metrics():
    metrics_dir = Path("reports") / "metrics"
    assert metrics_dir.exists(), "reports/metrics/ directory must exist"

    total_count = 0
    for module_name, plot_files in MODULE_PLOTS.items():
        module_dir = metrics_dir / module_name
        assert module_dir.exists(), f"reports/metrics/{module_name} must exist"

        for plot_file in plot_files:
            file_path = module_dir / plot_file
            assert file_path.exists(), f"Missing mirrored plot: {file_path}"
            size = file_path.stat().st_size
            assert size > 5000, f"Mirrored plot {file_path} is too small ({size} bytes)"
            total_count += 1

    assert total_count == 32, f"Expected 32 plots in reports/metrics/, found {total_count}"
