import streamlit as st
import requests

st.set_page_config(page_title="SME Cash Flow Risk Predictor")

st.title("💰 SME Cash Flow Risk Predictor")

st.write("Enter business metrics to predict cash flow stress (30–60 days ahead).")

# -----------------------------
# USER INPUTS
# -----------------------------
cash_runway = st.number_input("Cash Runway (months)", 0.0, 24.0, 6.0)
revenue_expense_ratio = st.number_input("Revenue / Expense Ratio", 0.0, 5.0, 1.2)
total_overdue = st.number_input("Total Overdue Amount", 0.0, 1_000_000.0, 50000.0)
overdue_severity = st.slider("Overdue Severity (0–1)", 0.0, 1.0, 0.4)
payment_delay_change = st.number_input("Payment Delay Change (days)", -30.0, 60.0, 5.0)

sales_3m_avg = st.number_input("Avg Sales (last 3 months)", 0.0, 1_000_000.0, 200000.0)
expense_3m_avg = st.number_input("Avg Expenses (last 3 months)", 0.0, 1_000_000.0, 180000.0)

sales_growth = st.number_input("Sales Growth (%)", -100.0, 200.0, 5.0)
expense_growth = st.number_input("Expense Growth (%)", -100.0, 200.0, 8.0)

growth_gap = sales_growth - expense_growth
sales_volatility = st.slider("Sales Volatility (0–1)", 0.0, 1.0, 0.3)

# -----------------------------
# API CALL
# -----------------------------
if st.button("🔍 Predict Cash Flow Risk"):
    payload = {
        "cash_runway": cash_runway,
        "revenue_expense_ratio": revenue_expense_ratio,
        "total_overdue": total_overdue,
        "overdue_severity": overdue_severity,
        "payment_delay_change": payment_delay_change,
        "sales_3m_avg": sales_3m_avg,
        "expense_3m_avg": expense_3m_avg,
        "sales_growth": sales_growth,
        "expense_growth": expense_growth,
        "growth_gap": growth_gap,
        "sales_volatility": sales_volatility
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload,
            timeout=5
        )

        if response.status_code == 200:
            risk = response.json()["risk_score"]

            st.success(f"⚠️ Cash Flow Risk Score: **{risk}%**")

            if risk < 30:
                st.write("🟢 Low risk — cash flow is healthy.")
            elif risk < 60:
                st.write("🟡 Moderate risk — monitor closely.")
            else:
                st.write("🔴 High risk — action required immediately.")

        else:
            st.error("API error. Check FastAPI logs.")

    except Exception as e:
        st.error(f"Could not connect to API: {e}")
