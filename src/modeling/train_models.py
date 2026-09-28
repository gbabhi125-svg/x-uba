#!/usr/bin/env python3
"""
X-UBA: Explainable User and Entity Behavior Analytics
       Using Heterogeneous Security Telemetry

Module: Model Training (XGBoost Risk Classification + SHAP Explainability)

Trains THREE separate XGBoost models on identity_features:
  1. is_anomaly     (binary: 0/1)
  2. risk_level      (4-class: LOW/MEDIUM/HIGH/CRITICAL)
  3. threat_type     (6-class: NORMAL/PRIVILEGE_CREEP/ACCOUNT_TAKEOVER/
                       BRUTE_FORCE/DATA_EXFILTRATION/INSIDER_THREAT)

Adds SHAP TreeExplainer on the risk_level model (the dashboard's primary
model) so every prediction can be explained feature-by-feature — this is
the "Explainable" half of X-UBA's name.

Run from ANYWHERE — this script anchors itself to the project root.
Requires: data/processed/train_features.csv, data/processed/test_features.csv
Output: models/*.joblib , reports/model_metrics.json , reports/shap_feature_importance.csv
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    precision_score, recall_score
)
import joblib
import json
import os
from pathlib import Path

# ── Anchor to project root regardless of where this script is invoked from ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("=" * 70)
print("PHASE 1: XGBoost Risk Classification Training")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

train_df = pd.read_csv("data/processed/train_features.csv")
test_df = pd.read_csv("data/processed/test_features.csv")

print(f"\n✅ Loaded train: {train_df.shape}, test: {test_df.shape}")

CATEGORICAL_COLS = ["department", "job_title", "privilege_level"]
TARGET_COLS = ["is_anomaly", "risk_level", "threat_type"]
DROP_COLS = ["identity_id"] + TARGET_COLS

# ═══════════════════════════════════════════════════════════════════════════
# ENCODE CATEGORICALS (fit on combined train+test to avoid unseen-category errors)
# ═══════════════════════════════════════════════════════════════════════════

encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    le.fit(combined)
    train_df[col + "_enc"] = le.transform(train_df[col].astype(str))
    test_df[col + "_enc"] = le.transform(test_df[col].astype(str))
    encoders[col] = le

feature_cols = [c for c in train_df.columns if c not in DROP_COLS and c not in CATEGORICAL_COLS]

X_train = train_df[feature_cols].astype(float)
X_test = test_df[feature_cols].astype(float)

print(f"✅ Feature columns ({len(feature_cols)}): {feature_cols}")

results = {}


def evaluate(y_true, y_pred, target_names=None):
    """Compute a standard metrics bundle for the report."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "classification_report": classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }


# ═══════════════════════════════════════════════════════════════════════════
# MODEL 1: is_anomaly (BINARY)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "-" * 70)
print("MODEL 1: is_anomaly (binary classification)")
print("-" * 70)

y_train_anom = train_df["is_anomaly"]
y_test_anom = test_df["is_anomaly"]

model_anomaly = xgb.XGBClassifier(
    n_estimators=250, max_depth=5, learning_rate=0.08,
    subsample=0.85, colsample_bytree=0.85,
    eval_metric="logloss", random_state=42
)
model_anomaly.fit(X_train, y_train_anom)
pred_anom = model_anomaly.predict(X_test)

results["is_anomaly"] = evaluate(y_test_anom, pred_anom, target_names=["NOT_ANOMALY", "ANOMALY"])
joblib.dump(model_anomaly, "models/xgb_is_anomaly.joblib")

print(f"Accuracy: {results['is_anomaly']['accuracy']}  |  F1 (macro): {results['is_anomaly']['f1_macro']}")

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 2: risk_level (4-CLASS) — the primary dashboard model
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "-" * 70)
print("MODEL 2: risk_level (4-class: LOW/MEDIUM/HIGH/CRITICAL)")
print("-" * 70)

le_risk = LabelEncoder()
y_train_risk = le_risk.fit_transform(train_df["risk_level"])
y_test_risk = le_risk.transform(test_df["risk_level"])

model_risk = xgb.XGBClassifier(
    n_estimators=350, max_depth=6, learning_rate=0.06,
    subsample=0.85, colsample_bytree=0.85,
    objective="multi:softprob", num_class=len(le_risk.classes_),
    eval_metric="mlogloss", random_state=42
)
model_risk.fit(X_train, y_train_risk)
pred_risk = model_risk.predict(X_test)

results["risk_level"] = evaluate(y_test_risk, pred_risk, target_names=le_risk.classes_.tolist())
results["risk_level"]["classes"] = le_risk.classes_.tolist()
joblib.dump(model_risk, "models/xgb_risk_level.joblib")
joblib.dump(le_risk, "models/le_risk_level.joblib")

print(f"Accuracy: {results['risk_level']['accuracy']}  |  F1 (macro): {results['risk_level']['f1_macro']}")
print(f"Per-class F1: {[(c, round(results['risk_level']['classification_report'][c]['f1-score'], 3)) for c in le_risk.classes_]}")

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 3: threat_type (6-CLASS)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "-" * 70)
print("MODEL 3: threat_type (6-class)")
print("-" * 70)

le_threat = LabelEncoder()
y_train_threat = le_threat.fit_transform(train_df["threat_type"])
y_test_threat = le_threat.transform(test_df["threat_type"])

model_threat = xgb.XGBClassifier(
    n_estimators=350, max_depth=6, learning_rate=0.06,
    subsample=0.85, colsample_bytree=0.85,
    objective="multi:softprob", num_class=len(le_threat.classes_),
    eval_metric="mlogloss", random_state=42
)
model_threat.fit(X_train, y_train_threat)
pred_threat = model_threat.predict(X_test)

results["threat_type"] = evaluate(y_test_threat, pred_threat, target_names=le_threat.classes_.tolist())
results["threat_type"]["classes"] = le_threat.classes_.tolist()
joblib.dump(model_threat, "models/xgb_threat_type.joblib")
joblib.dump(le_threat, "models/le_threat_type.joblib")

print(f"Accuracy: {results['threat_type']['accuracy']}  |  F1 (macro): {results['threat_type']['f1_macro']}")

# ═══════════════════════════════════════════════════════════════════════════
# SAVE FEATURE METADATA (needed later for inference / counterfactual engine)
# ═══════════════════════════════════════════════════════════════════════════

joblib.dump({
    "feature_cols": feature_cols,
    "categorical_encoders": encoders
}, "models/feature_metadata.joblib")

# ═══════════════════════════════════════════════════════════════════════════
# SHAP EXPLAINABILITY (on risk_level model — the one the dashboard shows)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "-" * 70)
print("SHAP: Building explainer for risk_level model")
print("-" * 70)

import shap

explainer = shap.TreeExplainer(model_risk)
sample = X_test.iloc[:min(500, len(X_test))]
shap_values = explainer.shap_values(sample)

# Handle both possible SHAP output shapes across versions:
# - list of arrays (one per class), each (n_samples, n_features)
# - single 3D array (n_samples, n_features, n_classes)
if isinstance(shap_values, list):
    mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
elif shap_values.ndim == 3:
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
else:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_importance = pd.DataFrame({
    "feature": feature_cols,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

shap_importance.to_csv("reports/shap_feature_importance.csv", index=False)

# Save the explainer itself so the live dashboard can call it per-identity
joblib.dump(explainer, "models/shap_explainer_risk_level.joblib")

print("Top 10 risk-driving features (global SHAP importance):")
print(shap_importance.head(10).to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# SAVE FULL METRICS REPORT
# ═══════════════════════════════════════════════════════════════════════════

with open("reports/model_metrics.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# Human-readable summary for the project report
summary_lines = ["MCA CAPSTONE — MODEL EVALUATION SUMMARY", "=" * 60, ""]

for model_name in ["is_anomaly", "risk_level", "threat_type"]:
    r = results[model_name]
    summary_lines.append(f"MODEL: {model_name}")
    summary_lines.append(f"  Accuracy:          {r['accuracy']}")
    summary_lines.append(f"  Precision (macro): {r['precision_macro']}")
    summary_lines.append(f"  Recall (macro):    {r['recall_macro']}")
    summary_lines.append(f"  F1 (macro):        {r['f1_macro']}")
    if "classes" in r:
        summary_lines.append(f"  Classes: {r['classes']}")
        for cls in r["classes"]:
            cr = r["classification_report"].get(cls, {})
            summary_lines.append(
                f"    {cls:20s} precision={cr.get('precision',0):.3f}  "
                f"recall={cr.get('recall',0):.3f}  f1={cr.get('f1-score',0):.3f}  "
                f"support={int(cr.get('support',0))}"
            )
    summary_lines.append("")

summary_lines.append("TOP 10 SHAP FEATURES (risk_level model)")
for _, row in shap_importance.head(10).iterrows():
    summary_lines.append(f"  {row['feature']:30s} {row['mean_abs_shap']:.4f}")

summary_text = "\n".join(summary_lines)
with open("reports/model_evaluation_summary.txt", "w") as f:
    f.write(summary_text)

print("\n" + "=" * 70)
print("✅ PHASE 1 COMPLETE")
print("=" * 70)
print(f"\nModels saved to:  models/")
print(f"Reports saved to: reports/")
print("\n" + summary_text)