from fastapi import FastAPI
import pickle
import os
import numpy as np
import pandas as pd

app = FastAPI()

# ------------------------------------
# SAFE PATH HANDLING (CRITICAL FIX)
# ------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

MODEL_FEATURES = [
    "cash_runway",
    "revenue_expense_ratio",
    "total_overdue",
    "overdue_severity",
    "payment_delay_change",
    "sales_3m_avg",
    "expense_3m_avg",
    "sales_growth",
    "expense_growth",
    "growth_gap",
    "sales_volatility",
]

@app.get("/")
def health():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    X = df[MODEL_FEATURES]
    X_scaled = scaler.transform(X)
    risk = model.predict_proba(X_scaled)[0][1]
    return {"risk_score": round(risk * 100, 1)}
