# CashFlow-Risk-Predictor

# 💰 SME Cash-Flow Risk Predictor

A **product-ready dashboard** that predicts cash-flow stress for SMEs **30–60 days in advance**, along with actionable recommendations.  
Built with Python, Streamlit, FastAPI, and machine learning.

---

## 🔹 Project Overview

Many SMEs fail not because they are unprofitable, but due to **cash-flow mismanagement**. This project helps businesses **anticipate cash-flow stress** and make informed decisions before issues arise.

The system provides:

- **Cash-Flow Risk Score (0–100)**
- **Risk Level:** Low / Medium / High
- **Expected Shortfall:** Estimated cash shortage if no action is taken
- **Explainability:** Top drivers of risk in simple language
- **What-If Simulator:** Test business actions like improving collections or reducing expenses
- **PDF Report Generation:** Export actionable reports

---

## 🔹 Tech Stack

- **Python** – pandas, numpy, scikit-learn
- **Streamlit** – Frontend dashboard
- **FastAPI** – Backend API for predictions
- **ReportLab** – PDF generation
- **Uvicorn** – Server for FastAPI

---

## 🔹 Features

1. **Interactive Dashboard:** Upload your transaction CSV and see risk analysis.
2. **Risk Scoring:** Calculates probability of cash-flow stress using a Logistic Regression model.
3. **Business Impact:** Shows expected shortfall in INR and actionable suggestions.
4. **What-If Simulator:** Adjust collections and expenses to see how risk changes.
5. **Explainability:** Shows top drivers impacting cash-flow risk.
6. **PDF Report:** Generates a professional, client-ready report with insights.

---

## 🔹 How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/SME-Cashflow-Predictor.git
cd SME-Cashflow-Predictor
