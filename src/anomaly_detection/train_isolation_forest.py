#!/usr/bin/env python3
"""
X-UBA: Explainable User and Entity Behavior Analytics
       Using Heterogeneous Security Telemetry

Module: Anomaly Detection (Phase 2) - Isolation Forest

Trains an UNSUPERVISED Isolation Forest on the same feature set used by
the Phase 1 XGBoost model, then evaluates it against the ground-truth
is_anomaly labels and directly compares its performance to the supervised
XGBoost model. This comparison is a genuine research contribution point:
it shows quantitatively how much a purely behavioral, label-free detector
underperforms a supervised model trained on the same features - which is
exactly the kind of gap X-UBA's hybrid design (Isolation Forest + XGBoost
+ SHAP explainability) is meant to address.

Run from ANYWHERE - this script anchors itself to the project root.
Requires: data/processed/train_features.csv, test_features.csv,
          models/feature_metadata.joblib (from Phase 1)
Output: models/isolation_forest.joblib
        reports/isolation_forest_scores.csv
        reports/phase2_anomaly_detection_summary.txt
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
)
import joblib
import json
import os
from pathlib import Path

# -- Anchor to project root regardless of where this script is invoked from --
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("=" * 70)
print("PHASE 2: Isolation Forest - Unsupervised Anomaly Detection")
print("=" * 70)

# =============================================================================
# LOAD DATA + REUSE PHASE 1's FEATURE ENCODING (apples-to-apples comparison)
# =============================================================================

train_df = pd.read_csv("data/processed/train_features.csv")
test_df = pd.read_csv("data/processed/test_features.csv")

metadata_path = "models/feature_metadata.joblib"
if not os.path.exists(metadata_path):
    raise FileNotFoundError(
        "models/feature_metadata.joblib not found. Run Phase 1 "
        "(src/modeling/train_models.py) first - Phase 2 reuses its "
        "feature encoding so both models are compared on identical inputs."
    )

metadata = joblib.load(metadata_path)
feature_cols = metadata["feature_cols"]
encoders = metadata["categorical_encoders"]

CATEGORICAL_COLS = list(encoders.keys())

for col in CATEGORICAL_COLS:
    le = encoders[col]
    train_df[col + "_enc"] = le.transform(train_df[col].astype(str))
    test_df[col + "_enc"] = le.transform(test_df[col].astype(str))

X_train = train_df[feature_cols].astype(float)
X_test = test_df[feature_cols].astype(float)

y_test_true = test_df["is_anomaly"].values

print(f"\n[OK] Loaded train: {X_train.shape}, test: {X_test.shape}")
print(f"[OK] Reusing {len(feature_cols)} features from Phase 1 (identical to XGBoost input)")

# =============================================================================
# TRAIN ISOLATION FOREST (unsupervised - never sees is_anomaly during fit)
# =============================================================================

train_anomaly_rate = train_df["is_anomaly"].mean()
print(f"\nObserved anomaly rate in training data: {train_anomaly_rate:.4f} "
      f"(used only to set 'contamination', not as a training label)")

iso_forest = IsolationForest(
    n_estimators=300,
    contamination=train_anomaly_rate,
    max_samples="auto",
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train)

raw_pred = iso_forest.predict(X_test)
y_pred_anomaly = (raw_pred == -1).astype(int)

anomaly_score_raw = iso_forest.score_samples(X_test)
anomaly_score = -anomaly_score_raw
anomaly_score_normalized = (
    (anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min()) * 100
).round(2)

joblib.dump(iso_forest, "models/isolation_forest.joblib")

# =============================================================================
# EVALUATE
# =============================================================================

metrics_if = {
    "accuracy": round(float(accuracy_score(y_test_true, y_pred_anomaly)), 4),
    "precision": round(float(precision_score(y_test_true, y_pred_anomaly, zero_division=0)), 4),
    "recall": round(float(recall_score(y_test_true, y_pred_anomaly, zero_division=0)), 4),
    "f1": round(float(f1_score(y_test_true, y_pred_anomaly, zero_division=0)), 4),
    "confusion_matrix": confusion_matrix(y_test_true, y_pred_anomaly).tolist()
}

print("\n" + "-" * 70)
print("ISOLATION FOREST RESULTS (unsupervised)")
print("-" * 70)
print(f"Accuracy:  {metrics_if['accuracy']}")
print(f"Precision: {metrics_if['precision']}")
print(f"Recall:    {metrics_if['recall']}")
print(f"F1:        {metrics_if['f1']}")
print(f"Confusion matrix [[TN,FP],[FN,TP]]: {metrics_if['confusion_matrix']}")

# =============================================================================
# LOAD PHASE 1's XGBoost is_anomaly RESULTS FOR DIRECT COMPARISON
# =============================================================================

comparison = {"isolation_forest": metrics_if}

phase1_metrics_path = "reports/model_metrics.json"
if os.path.exists(phase1_metrics_path):
    with open(phase1_metrics_path) as f:
        phase1 = json.load(f)
    xgb_metrics = phase1.get("is_anomaly", {})
    comparison["xgboost_supervised"] = {
        "accuracy": xgb_metrics.get("accuracy"),
        "precision": xgb_metrics.get("precision_macro"),
        "recall": xgb_metrics.get("recall_macro"),
        "f1": xgb_metrics.get("f1_macro"),
    }
    print("\n" + "-" * 70)
    print("COMPARISON: Unsupervised (Isolation Forest) vs Supervised (XGBoost)")
    print("-" * 70)
    print(f"{'Metric':<12} {'Isolation Forest':<20} {'XGBoost (supervised)':<20}")
    for m in ["accuracy", "precision", "recall", "f1"]:
        print(f"{m:<12} {metrics_if[m]:<20} {comparison['xgboost_supervised'].get(m, 'N/A'):<20}")
else:
    print("\n[WARN] reports/model_metrics.json not found - run Phase 1 first for the comparison table.")

# =============================================================================
# SAVE PER-IDENTITY ANOMALY SCORES (needed later for risk fusion / dashboard)
# =============================================================================

scores_df = pd.DataFrame({
    "identity_id": test_df["identity_id"].values,
    "isolation_forest_anomaly_score": anomaly_score_normalized,
    "isolation_forest_flagged": y_pred_anomaly,
    "true_is_anomaly": y_test_true
})
scores_df.to_csv("reports/isolation_forest_scores.csv", index=False)

# =============================================================================
# WRITE HUMAN-READABLE SUMMARY
# =============================================================================

lines = [
    "X-UBA - PHASE 2: ISOLATION FOREST ANOMALY DETECTION",
    "=" * 60,
    "",
    f"Training anomaly rate (contamination parameter used): {train_anomaly_rate:.4f}",
    "",
    "ISOLATION FOREST (unsupervised) RESULTS ON TEST SET",
    f"  Accuracy:  {metrics_if['accuracy']}",
    f"  Precision: {metrics_if['precision']}",
    f"  Recall:    {metrics_if['recall']}",
    f"  F1:        {metrics_if['f1']}",
    f"  Confusion matrix [[TN,FP],[FN,TP]]: {metrics_if['confusion_matrix']}",
    ""
]

if "xgboost_supervised" in comparison:
    lines += [
        "COMPARISON: UNSUPERVISED vs SUPERVISED",
        f"{'Metric':<12} {'Isolation Forest':<20} {'XGBoost (supervised)':<20}",
    ]
    for m in ["accuracy", "precision", "recall", "f1"]:
        lines.append(f"{m:<12} {metrics_if[m]:<20} {comparison['xgboost_supervised'].get(m, 'N/A'):<20}")
    lines.append("")
    lines.append(
        "INTERPRETATION: Isolation Forest sees only feature patterns, never the "
        "is_anomaly label. Its lower scores versus XGBoost quantify the value of "
        "supervised learning when labeled incident data exists, while still "
        "providing a label-free safety net for detecting NOVEL anomaly types "
        "the supervised model was never trained to recognize."
    )

summary_text = "\n".join(lines)
with open("reports/phase2_anomaly_detection_summary.txt", "w") as f:
    f.write(summary_text)

with open("reports/phase2_metrics.json", "w") as f:
    json.dump(comparison, f, indent=2, default=str)

print("\n" + "=" * 70)
print("[OK] PHASE 2 COMPLETE")
print("=" * 70)
print("\nModel saved to:  models/isolation_forest.joblib")
print("Scores saved to: reports/isolation_forest_scores.csv")
print("Summary saved to: reports/phase2_anomaly_detection_summary.txt")
