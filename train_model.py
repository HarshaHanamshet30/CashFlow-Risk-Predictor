import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# -----------------------------------
# DUMMY TRAINING DATA (SAFE & GENERAL)
# -----------------------------------
np.random.seed(42)

data = pd.DataFrame({
    "cash_runway": np.random.rand(100),
    "revenue_expense_ratio": np.random.rand(100),
    "total_overdue": np.random.randint(0, 5, 100),
    "overdue_severity": np.random.rand(100),
    "payment_delay_change": np.random.rand(100),
    "sales_3m_avg": np.random.rand(100),
    "expense_3m_avg": np.random.rand(100),
    "sales_growth": np.random.randn(100),
    "expense_growth": np.random.randn(100),
    "growth_gap": np.random.randn(100),
    "sales_volatility": np.random.rand(100),
})

# Target (0 = safe, 1 = risky)
y = (data["cash_runway"] < 0.5).astype(int)

# -----------------------------------
# TRAIN MODEL
# -----------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

model = LogisticRegression(class_weight="balanced")
model.fit(X_scaled, y)

# -----------------------------------
# SAVE FILES
# -----------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ model.pkl and scaler.pkl saved successfully")
