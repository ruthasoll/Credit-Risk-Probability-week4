# src/predict.py

import joblib
import pandas as pd
import os

class CreditRiskModel:
    def __init__(self, model_path: str = "models/best_model.pkl"):
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = joblib.load(model_path)
        
    def predict(self, input_data: pd.DataFrame) -> dict:
        """
        Predict risk probability.
        input_data should be a DataFrame with the same features as training data.
        """
        # Ensure input columns match model expectation (handling this robustly is complex in production without a schema registry)
        # For simplicity, we assume input_data is pre-processed or we might need to include a pre-processing pipeline in the model object.
        
        # NOTE: Ideally the model pipeline should include feature engineering, but here we separated them.
        # This implies the input_data must be already processed numeric features.
        # OR we need to load the preprocessing pipeline.
        
        probabilities = self.model.predict_proba(input_data)[:, 1]
        results = [
            {"risk_probability": prob, "risk_label": "High Risk" if prob > 0.5 else "Low Risk"}
            for prob in probabilities
        ]
        return results

if __name__ == "__main__":
    # Test run
    # This assumes we have some dummy processed data available or pass it manually
    print("Inference script ready.")
