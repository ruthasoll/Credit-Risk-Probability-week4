# src/api/pydantic_models.py

from pydantic import BaseModel
from typing import List, Optional

class CreditRiskRequest(BaseModel):
    total_transaction_amount: float
    avg_transaction_amount: float
    transaction_count: float
    std_transaction_amount: float
    avg_transaction_hour: float
    avg_transaction_day: float
    avg_transaction_month: float
    Recency: float
    Frequency: float
    Monetary: float
    # Add other features if necessary, or use extra='allow'
    
    class Config:
        extra = "allow" 

class PredictionResponse(BaseModel):
    risk_probability: float
    risk_label: str
