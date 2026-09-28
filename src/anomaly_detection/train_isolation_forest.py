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
        reports/isolation_forest_all_scores.csv  (all 10,000 identities)
        reports/phase2_anomaly_detection_summary.txt
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
)
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    SEED, init, banner, section, write_text, write_json, read_json, load_features,
    load_metadata, encode
)

# Anchor to project root regardless of where this script is invoked from
init()

banner("X-UBA | MCA MAJOR PROJECT | PHASE 2: Isolation Forest (unsupervised)")

# =============================================================================
# LOAD DATA + REUSE PHASE 1's FEATURE ENCODING (apples-to-apples comparison)
# =============================================================================

full_df, train_df, test_df = load_features()

# Phase 2 reuses Phase 1's feature encoding so both models see identical inputs
metadata = load_metadata()
feature_cols = metadata["feature_cols"]

X_train = encode(train_df, metadata)
X_test = encode(test_df, metadata)
X_full = encode(full_df, metadata)

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
    random_state=SEED,
    n_jobs=-1
)
iso_forest.fit(X_train)

raw_pred = iso_forest.predict(X_test)
y_pred_anomaly = (raw_pred == -1).astype(int)

# Higher = more anomalous. Normalised to 0-100 using the TRAINING score range
# (clipped), so the same identity always gets the same score whichever set it is in.
train_raw = -iso_forest.score_samples(X_train)
lo, hi = train_raw.min(), train_raw.max()


def normalise(raw):
    return (np.clip((raw - lo) / (hi - lo), 0, 1) * 100).round(2)


anomaly_score_normalized = normalise(-iso_forest.score_samples(X_test))

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

section("ISOLATION FOREST RESULTS (unsupervised)")
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
    phase1 = read_json(phase1_metrics_path)
    xgb_metrics = phase1.get("is_anomaly", {})
    # Anomaly-class metrics, so both columns measure the same thing
    xgb_anom_cls = xgb_metrics.get("classification_report", {}).get("ANOMALY", {})
    comparison["xgboost_supervised"] = {
        "accuracy": xgb_metrics.get("accuracy"),
        "precision": round(float(xgb_anom_cls.get("precision", float("nan"))), 4),
        "recall": round(float(xgb_anom_cls.get("recall", float("nan"))), 4),
        "f1": round(float(xgb_anom_cls.get("f1-score", float("nan"))), 4),
    }
    section("COMPARISON: Unsupervised (Isolation Forest) vs Supervised (XGBoost)")
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

# All 10,000 identities (label-free, so scoring training rows is legitimate)
all_scores_df = pd.DataFrame({
    "identity_id": full_df["identity_id"].values,
    "isolation_forest_anomaly_score": normalise(-iso_forest.score_samples(X_full)),
    "isolation_forest_flagged": (iso_forest.predict(X_full) == -1).astype(int),
})
all_scores_df.to_csv("reports/isolation_forest_all_scores.csv", index=False)

# =============================================================================
# WRITE HUMAN-READABLE SUMMARY
# =============================================================================

lines = [
    "X-UBA - MCA MAJOR PROJECT - PHASE 2: ISOLATION FOREST ANOMALY DETECTION",
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
        "NOTE: precision/recall/F1 in both columns are for the ANOMALY class "
        "(is_anomaly = 1), so the comparison is like-for-like."
    )
    lines.append("")
    lines.append(
        "INTERPRETATION: Isolation Forest sees only feature patterns, never the "
        "is_anomaly label. Its lower scores versus XGBoost quantify the value of "
        "supervised learning when labeled incident data exists, while still "
        "providing a label-free safety net for detecting NOVEL anomaly types "
        "the supervised model was never trained to recognize."
    )

summary_text = "\n".join(lines)
write_text("reports/phase2_anomaly_detection_summary.txt", summary_text)
write_json("reports/phase2_metrics.json", comparison)

print()
banner("[OK] PHASE 2 COMPLETE")
print("\nModel saved to:  models/isolation_forest.joblib")
print("Scores saved to: reports/isolation_forest_scores.csv")
print("Summary saved to: reports/phase2_anomaly_detection_summary.txt")
