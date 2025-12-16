# src/api/main.py

from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CreditRiskRequest, PredictionResponse
from src.predict import CreditRiskModel
import pandas as pd
import os

app = FastAPI(title="Credit Risk Scoring API", version="1.0")

# Load model on startup
model_path = os.getenv("MODEL_PATH", "models/best_model.pkl")
# Assuming absolute path or relative to running directory. 
# In Docker, we might set WORKDIR.

try:
    risk_model = CreditRiskModel(model_path)
except Exception as e:
    print(f"Warning: Could not load model at startup: {e}")
    risk_model = None

@app.get("/")
def read_root():
    return {"message": "Welcome to Credit Risk Scoring API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(request: CreditRiskRequest):
    if not risk_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert request to DataFrame
    input_data = pd.DataFrame([request.dict()])
    
    # Ensure columns match (simple validation)
    # real world: match exact columns, fill missing with defaults or error
    
    try:
        result = risk_model.predict(input_data)[0] # single prediction
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
