# ==================================================
# model.py
# SME Cash-Flow Risk Prediction Engine
# ==================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
def prepare_monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: transaction-level dataframe
    Output: monthly SME-level features
    """

    df = df.copy()

    # Safety
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["transaction_date", "amount"])

    # Month
    df["month"] = df["transaction_date"].dt.to_period("M").astype(str)

    # Aggregate monthly cash flows
    monthly = (
        df.groupby(["sme_id", "month"])
        .agg(
            cash_in=("amount", lambda x: x[x > 0].sum()),
            cash_out=("amount", lambda x: abs(x[x < 0].sum()))
        )
        .reset_index()
    )

    monthly["net_cashflow"] = monthly["cash_in"] - monthly["cash_out"]

    # Core financial features
    monthly["cash_runway"] = monthly["cash_in"] / (monthly["cash_out"] + 1)
    monthly["revenue_expense_ratio"] = monthly["cash_in"] / (monthly["cash_out"] + 1)

    monthly["sales_growth"] = (
        monthly.groupby("sme_id")["cash_in"].pct_change().fillna(0)
    )

    monthly["expense_growth"] = (
        monthly.groupby("sme_id")["cash_out"].pct_change().fillna(0)
    )

    monthly["growth_gap"] = monthly["sales_growth"] - monthly["expense_growth"]

    monthly["sales_volatility"] = (
        monthly.groupby("sme_id")["cash_in"]
        .rolling(3)
        .std()
        .reset_index(0, drop=True)
        .fillna(0)
    )

    monthly["sales_3m_avg"] = (
        monthly.groupby("sme_id")["cash_in"]
        .rolling(3)
        .mean()
        .reset_index(0, drop=True)
        .fillna(0)
    )

    monthly["expense_3m_avg"] = (
        monthly.groupby("sme_id")["cash_out"]
        .rolling(3)
        .mean()
        .reset_index(0, drop=True)
        .fillna(0)
    )

    # Placeholder credit features (future upgrade)
    monthly["total_overdue"] = 0
    monthly["overdue_severity"] = 0
    monthly["payment_delay_change"] = 0

    return monthly


# --------------------------------------------------
# TARGET CREATION (AUTO-SAFE)
# --------------------------------------------------
def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates cash_flow_stress label
    """

    df = df.copy()

    df["cash_flow_stress"] = (df["net_cashflow"] < 0).astype(int)

    # 🔥 Single-class safety fix
    if df["cash_flow_stress"].nunique() < 2:
        df.loc[df.index[-1], "cash_flow_stress"] = 1
        if len(df) > 1:
            df.loc[df.index[0], "cash_flow_stress"] = 0

    return df


# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------
def train_model(df: pd.DataFrame):
    """
    Trains logistic regression model
    Returns trained model + scaler
    """

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
        "sales_volatility"
    ]

    X = df[MODEL_FEATURES]
    y = df["cash_flow_stress"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    model.fit(X_scaled, y)

    return model, scaler, MODEL_FEATURES


# --------------------------------------------------
# RISK SCORING
# --------------------------------------------------
def score_risk(df: pd.DataFrame, model, scaler, features):
    """
    Adds risk_score and risk_bucket
    """

    X = df[features]
    X_scaled = scaler.transform(X)

    df = df.copy()
    df["risk_probability"] = model.predict_proba(X_scaled)[:, 1]
    df["risk_score"] = (df["risk_probability"] * 100).round(1)

    def bucket(score):
        if score < 30:
            return "Low Risk"
        elif score < 60:
            return "Medium Risk"
        return "High Risk"

    df["risk_bucket"] = df["risk_score"].apply(bucket)

    return df


# --------------------------------------------------
# WHAT-IF SIMULATOR
# --------------------------------------------------
def simulate_what_if(
    latest_row: pd.Series,
    model,
    scaler,
    features,
    collection_improvement=0,
    expense_reduction=0
) -> float:
    """
    Returns simulated risk score
    """

    row = latest_row.copy()

    row["sales_3m_avg"] *= (1 + collection_improvement / 100)
    row["expense_3m_avg"] *= (1 - expense_reduction / 100)

    row["revenue_expense_ratio"] = (
        row["sales_3m_avg"] / max(row["expense_3m_avg"], 1)
    )

    row["growth_gap"] = row["sales_growth"] - row["expense_growth"]

    X_sim = pd.DataFrame([row[features]])
    X_sim_scaled = scaler.transform(X_sim)

    return round(model.predict_proba(X_sim_scaled)[0][1] * 100, 1)
