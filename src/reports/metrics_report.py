from src.utils.helpers import save_json
def save_metrics(records, path="reports/metrics/model_registry.json"):
    save_json(records, path)
