"""
03_classification_model.py
Predicts customer churn using Logistic Regression and Random Forest,
compares them on accuracy / precision / recall / ROC-AUC, and prints
a plain-English explanation of the top 3 predictive features.

Run: python 03_classification_model.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Load + prepare features
# ---------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent / "data" / "telco_churn_clean.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Churn"])
y = df["Churn"]

# One-hot encode all categorical columns (includes tenure_group).
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------
# 2. Logistic Regression (needs scaled features)
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
log_reg.fit(X_train_scaled, y_train)
lr_pred = log_reg.predict(X_test_scaled)
lr_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------------------
# 3. Random Forest (no scaling needed)
# ---------------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, random_state=RANDOM_STATE, class_weight="balanced"
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------------
# 4. Compare metrics
# ---------------------------------------------------------------------
def report(name, y_true, y_pred, y_proba):
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred), 3),
        "recall": round(recall_score(y_true, y_pred), 3),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 3),
    }

results = pd.DataFrame([
    report("Logistic Regression", y_test, lr_pred, lr_proba),
    report("Random Forest", y_test, rf_pred, rf_proba),
])

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results.to_string(index=False))

print("\nDetailed classification report — Random Forest:")
print(classification_report(y_test, rf_pred, target_names=["Retained", "Churned"]))

# ---------------------------------------------------------------------
# 5. Top 3 predictive features (Random Forest importances)
# ---------------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
top3 = importances.head(3)

print("=" * 60)
print("TOP 3 PREDICTIVE FEATURES (Random Forest)")
print("=" * 60)
print(top3, "\n")

print("Plain-English explanation:")
explanations = {
    "tenure": "How long a customer has been with the company. Newer customers "
              "are consistently far more likely to churn — retention risk is "
              "front-loaded in the first year.",
    "MonthlyCharges": "How much a customer pays per month. Higher bills correlate "
                       "with higher churn, suggesting price sensitivity or a "
                       "perceived value gap on premium plans.",
    "TotalCharges": "Cumulative lifetime spend. This mostly tracks tenure — "
                     "customers with low lifetime spend are early-tenure, "
                     "high-risk customers.",
    "Contract_Two year": "Being on a two-year contract. These customers are "
                          "locked in and structurally far less likely to churn "
                          "than month-to-month customers.",
    "Contract_One year": "Being on a one-year contract — a middle-ground "
                          "retention effect versus month-to-month.",
    "num_services": "How many add-on services (security, backup, streaming, etc.) "
                     "a customer has. More services = more switching cost = lower churn.",
}
for feat in top3.index:
    base_feat = feat if feat in explanations else feat.split("_")[0]
    text = explanations.get(feat, explanations.get(base_feat, "Key driver of churn risk in this dataset."))
    print(f"- {feat}: {text}")
