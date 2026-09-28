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
        reports/identity_risk_scores.csv   (out-of-fold scores for ALL identities)
        reports/shap_identity_top_factors.csv (per-identity "why risky" factors)

Why out-of-fold scores: later phases (fusion, attack simulator, dashboard)
need a model score for every one of the 10,000 identities. Scoring the
training rows with a model that was fitted on them would be optimistic, so
each identity is scored by a model trained on the OTHER 4/5 of the data.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    SEED, CATEGORICAL_COLS, TARGET_COLS, init, banner, section, write_text, write_json,
    load_features, risk_index
)

# Anchor to project root regardless of where this script is invoked from
init()

banner("X-UBA | MCA MAJOR PROJECT | PHASE 1: XGBoost Risk Classification + SHAP")

# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

full_df, train_df, test_df = load_features()

print(f"\n[OK] Loaded train: {train_df.shape}, test: {test_df.shape}")

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

print(f"[OK] Feature columns ({len(feature_cols)}): {feature_cols}")

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

section("MODEL 1: is_anomaly (binary classification)")

y_train_anom = train_df["is_anomaly"]
y_test_anom = test_df["is_anomaly"]

model_anomaly = xgb.XGBClassifier(
    n_estimators=250, max_depth=5, learning_rate=0.08,
    subsample=0.85, colsample_bytree=0.85,
    eval_metric="logloss", random_state=SEED
)
model_anomaly.fit(X_train, y_train_anom)
pred_anom = model_anomaly.predict(X_test)

results["is_anomaly"] = evaluate(y_test_anom, pred_anom, target_names=["NOT_ANOMALY", "ANOMALY"])
joblib.dump(model_anomaly, "models/xgb_is_anomaly.joblib")

print(f"Accuracy: {results['is_anomaly']['accuracy']}  |  F1 (macro): {results['is_anomaly']['f1_macro']}")

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 2: risk_level (4-CLASS) — the primary dashboard model
# ═══════════════════════════════════════════════════════════════════════════

section("MODEL 2: risk_level (4-class: LOW/MEDIUM/HIGH/CRITICAL)")

le_risk = LabelEncoder()
y_train_risk = le_risk.fit_transform(train_df["risk_level"])
y_test_risk = le_risk.transform(test_df["risk_level"])

model_risk = xgb.XGBClassifier(
    n_estimators=350, max_depth=6, learning_rate=0.06,
    subsample=0.85, colsample_bytree=0.85,
    objective="multi:softprob", num_class=len(le_risk.classes_),
    eval_metric="mlogloss", random_state=SEED
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

section("MODEL 3: threat_type (6-class)")

le_threat = LabelEncoder()
y_train_threat = le_threat.fit_transform(train_df["threat_type"])
y_test_threat = le_threat.transform(test_df["threat_type"])

model_threat = xgb.XGBClassifier(
    n_estimators=350, max_depth=6, learning_rate=0.06,
    subsample=0.85, colsample_bytree=0.85,
    objective="multi:softprob", num_class=len(le_threat.classes_),
    eval_metric="mlogloss", random_state=SEED
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

section("SHAP: Building explainer for risk_level model")

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
# OUT-OF-FOLD SCORES FOR ALL 10,000 IDENTITIES (input to every later phase)
# ═══════════════════════════════════════════════════════════════════════════

section("Out-of-fold scoring of all identities (5-fold, stratified)")

for col, le in encoders.items():
    full_df[col + "_enc"] = le.transform(full_df[col].astype(str))
X_full = full_df[feature_cols].astype(float)
y_full_anom = full_df["is_anomaly"].values
y_full_risk = le_risk.transform(full_df["risk_level"])
y_full_threat = le_threat.transform(full_df["threat_type"])

oof_anom = np.zeros(len(full_df))
oof_risk = np.zeros((len(full_df), len(le_risk.classes_)))
oof_threat = np.zeros((len(full_df), len(le_threat.classes_)))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for fold, (tr, va) in enumerate(skf.split(X_full, full_df["risk_level"]), start=1):
    m = xgb.XGBClassifier(**model_anomaly.get_params())
    m.fit(X_full.iloc[tr], y_full_anom[tr])
    oof_anom[va] = m.predict_proba(X_full.iloc[va])[:, 1]

    m = xgb.XGBClassifier(**model_risk.get_params())
    m.fit(X_full.iloc[tr], y_full_risk[tr])
    oof_risk[va] = m.predict_proba(X_full.iloc[va])

    m = xgb.XGBClassifier(**model_threat.get_params())
    m.fit(X_full.iloc[tr], y_full_threat[tr])
    oof_threat[va] = m.predict_proba(X_full.iloc[va])
    print(f"  fold {fold}/5 done")

test_ids = set(test_df["identity_id"])
scores_df = pd.DataFrame({
    "identity_id": full_df["identity_id"],
    "split": np.where(full_df["identity_id"].isin(test_ids), "test", "train"),
    "xgb_anomaly_proba": oof_anom.round(5),
    "xgb_risk_index": risk_index(oof_risk, le_risk.classes_).round(3),
    "xgb_risk_level_pred": le_risk.classes_[oof_risk.argmax(axis=1)],
    "xgb_threat_pred": le_threat.classes_[oof_threat.argmax(axis=1)],
    "true_is_anomaly": y_full_anom,
    "true_risk_level": full_df["risk_level"],
    "true_threat_type": full_df["threat_type"],
})
scores_df.to_csv("reports/identity_risk_scores.csv", index=False)

oof_metrics = {
    "is_anomaly_f1_binary": round(float(f1_score(y_full_anom, (oof_anom >= 0.5).astype(int))), 4),
    "risk_level_accuracy": round(float(accuracy_score(y_full_risk, oof_risk.argmax(axis=1))), 4),
    "risk_level_f1_macro": round(float(f1_score(y_full_risk, oof_risk.argmax(axis=1), average="macro")), 4),
}
results["out_of_fold_all_identities"] = oof_metrics
print(f"[OK] OOF metrics over all {len(full_df)} identities: {oof_metrics}")

# ═══════════════════════════════════════════════════════════════════════════
# PER-IDENTITY SHAP FACTORS ("WHY is this identity risky?")
# Explains the is_anomaly model: one signed value per feature, in log-odds.
# ═══════════════════════════════════════════════════════════════════════════

section("SHAP: per-identity top risk factors (is_anomaly model)")

anom_explainer = shap.TreeExplainer(model_anomaly)
sv_all = np.asarray(anom_explainer.shap_values(X_full))
if sv_all.ndim == 3:  # some SHAP versions return (n, features, 2)
    sv_all = sv_all[:, :, 1]

TOP_K = 5
top_rows = []
feat_arr = np.array(feature_cols)
for i in range(len(full_df)):
    order = np.argsort(-sv_all[i])[:TOP_K]
    row = {"identity_id": full_df["identity_id"].iat[i]}
    for rank, j in enumerate(order, start=1):
        row[f"factor_{rank}"] = feat_arr[j]
        row[f"factor_{rank}_value"] = float(X_full.iat[i, j])
        row[f"factor_{rank}_shap"] = round(float(sv_all[i, j]), 4)
    top_rows.append(row)
pd.DataFrame(top_rows).to_csv("reports/shap_identity_top_factors.csv", index=False)
joblib.dump(anom_explainer, "models/shap_explainer_is_anomaly.joblib")
print(f"[OK] Top-{TOP_K} SHAP factors saved for {len(top_rows)} identities")

# ═══════════════════════════════════════════════════════════════════════════
# SAVE FULL METRICS REPORT
# ═══════════════════════════════════════════════════════════════════════════

write_json("reports/model_metrics.json", results)

# Human-readable summary for the project report
summary_lines = ["X-UBA - MCA MAJOR PROJECT - PHASE 1 MODEL EVALUATION SUMMARY", "=" * 60, ""]

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
write_text("reports/model_evaluation_summary.txt", summary_text)

print()
banner("[OK] PHASE 1 COMPLETE")
print("\nModels saved to:  models/")
print("Reports saved to: reports/")
print("\n" + summary_text)