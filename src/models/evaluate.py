import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

def classification_metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(p), "recall_weighted": float(r),
        "f1_weighted": float(f),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

def regression_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }

def cv_score(model, X, y, kind):
    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42) if kind == "classification" else KFold(n_splits=2, shuffle=True, random_state=42)
    scoring = "f1_weighted" if kind == "classification" else "neg_root_mean_squared_error"
    scores = cross_val_score(model, X, y, cv=splitter, scoring=scoring, n_jobs=1)
    return float(scores.mean()), float(scores.std())
