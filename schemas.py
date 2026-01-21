from pydantic import BaseModel
from typing import List

class Transaction(BaseModel):
    transaction_date: str
    amount: float
    transaction_type: str

class PredictionResponse(BaseModel):
    risk_score: float
    risk_bucket: str
    expected_loss: float
