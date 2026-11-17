import json
from pathlib import Path

def load_schemes(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scheme knowledge base not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Scheme knowledge base must be a JSON list.")
    return data
