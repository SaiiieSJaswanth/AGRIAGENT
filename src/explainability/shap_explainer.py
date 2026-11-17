from pathlib import Path
import json

def explain_pipeline(model_pipeline, row, output_path=None, max_features=8):
    # SHAP is optional at inference time because sparse one-hot preprocessing can be expensive.
    # This function records a truthful fallback when a direct SHAP explanation is unavailable.
    try:
        import shap
        estimator = model_pipeline.named_steps["model"]
        pre = model_pipeline.named_steps["preprocess"]
        transformed = pre.transform(row)
        explainer = shap.Explainer(estimator, transformed)
        values = explainer(transformed)
        vals = values.values
        if vals.ndim == 3:
            vals = vals[0].mean(axis=1)
        else:
            vals = vals[0]
        names = pre.get_feature_names_out()
        ranked = sorted(zip(names, vals), key=lambda x: abs(float(x[1])), reverse=True)[:max_features]
        result = [{"feature": str(n), "contribution": float(v)} for n, v in ranked]
    except Exception as exc:
        result = [{"feature": "SHAP unavailable", "contribution": None, "note": str(exc)}]
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return result
