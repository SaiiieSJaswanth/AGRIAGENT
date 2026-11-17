import joblib
class SoilHealthModel:
    def __init__(self, artifact="models/soil_health/model.joblib"):
        self.model = joblib.load(artifact)
    def predict(self, row):
        return str(self.model.predict(row)[0])
