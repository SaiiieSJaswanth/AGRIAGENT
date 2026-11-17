from pathlib import Path
import json
import numpy as np

def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def json_default(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if hasattr(obj, "isoformat"): return obj.isoformat()
    return str(obj)

def save_json(data, path):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=json_default)

def normalize_text(value):
    return " ".join(str(value).strip().lower().split())
