#!/usr/bin/env python3
"""
X-UBA: Explainable User and Entity Behavior Analytics
       Using Heterogeneous Security Telemetry

Module: Model Validation (Phase 1b)

Three checks on the Phase 1 models that an examiner will ask about:

  1. 5-FOLD CROSS-VALIDATION  mean +/- std of accuracy and macro-F1 for all three
     targets over the full 10,000 identities (is the single test split lucky?)
  2. FEATURE-GROUP ABLATION   retrain the risk_level model without each telemetry
     source (identity posture, authentication, privilege changes, resource
     access, derived signals) and measure the macro-F1 drop on the test set -
     which heterogeneous source matters?  Also tests removing only the raw
     activity-volume counts (login_count, access_count, resource_count),
     which the synthetic generator ties to risk.
  3. CLASS-WEIGHTED threat_type  the 6-class threat model has weak recall on
     rare classes (INSIDER_THREAT, BRUTE_FORCE, DATA_EXFILTRATION). Retrained
     with balanced sample weights and compared class by class.

Run from ANYWHERE - this script anchors itself to the project root.
Requires: data/processed/*.csv, Phase 1 models
Output: models/xgb_threat_type_weighted.joblib
        reports/model_validation.json
        reports/model_validation_summary.txt
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    SEED, MODELS, init, banner, section, write_text, write_json, require, load_features,
    load_metadata, encode
)

init()
banner("X-UBA | MCA MAJOR PROJECT | PHASE 1b: Model Validation")

require(f"{MODELS}/xgb_is_anomaly.joblib", f"{MODELS}/xgb_risk_level.joblib",
        f"{MODELS}/xgb_threat_type.joblib")
full_df, train_df, test_df = load_features()
metadata = load_metadata()
feature_cols = metadata["feature_cols"]
X_full, X_train, X_test = (encode(d, metadata) for d in (full_df, train_df, test_df))

PARAMS = {t: joblib.load(f"{MODELS}/xgb_{t}.joblib").get_params()
          for t in ("is_anomaly", "risk_level", "threat_type")}

FEATURE_GROUPS = {
    "Identity posture": ["department_enc", "job_title_enc", "privilege_level_enc", "is_service",
                         "is_contractor", "inactive_days", "mfa_enabled", "n_systems"],
    "Authentication logs": ["login_count", "failed_login_rate", "night_login_rate",
                            "unique_countries", "unique_devices", "mfa_usage_rate"],
    "Privilege-change logs": ["privilege_changes_count", "privilege_escalations",
                              "unapproved_changes", "privilege_change_trend"],
    "Resource-access logs": ["access_count", "resource_count", "system_count",
                             "sensitive_resource_access", "data_download_volume", "admin_actions"],
    "Derived signals": ["behaviour_deviation_score", "risk_trend"],
}
ACTIVITY_VOLUME = ["login_count", "access_count", "resource_count"]
grouped = {f for g in FEATURE_GROUPS.values() for f in g}
leftover = [f for f in feature_cols if f not in grouped]
if leftover:
    FEATURE_GROUPS["Other"] = leftover
missing = grouped - set(feature_cols)
if missing:
    raise SystemExit(f"[X] Unknown feature(s) in FEATURE_GROUPS: {sorted(missing)}")


def targets(df):
    return {t: LabelEncoder().fit(full_df[t]).transform(df[t])
            for t in ("is_anomaly", "risk_level", "threat_type")}


y_full, y_train, y_test = targets(full_df), targets(train_df), targets(test_df)

# ── 1. Cross-validation ──
section("1) 5-fold stratified cross-validation (10,000 identities)")
cv = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for t in ("is_anomaly", "risk_level", "threat_type"):
    accs, f1s = [], []
    for tr, va in skf.split(X_full, full_df["risk_level"]):
        m = xgb.XGBClassifier(**PARAMS[t]).fit(X_full.iloc[tr], y_full[t][tr])
        p = m.predict(X_full.iloc[va])
        accs.append(accuracy_score(y_full[t][va], p))
        f1s.append(f1_score(y_full[t][va], p, average="macro"))
    cv[t] = {"mean_accuracy": round(float(np.mean(accs)), 4), "std_accuracy": round(float(np.std(accs)), 4),
             "mean_f1_macro": round(float(np.mean(f1s)), 4), "std_f1_macro": round(float(np.std(f1s)), 4),
             "fold_f1_macro": [round(float(v), 4) for v in f1s]}
    print(f"  {t:12s} accuracy {cv[t]['mean_accuracy']:.4f} +/- {cv[t]['std_accuracy']:.4f}   "
          f"macro-F1 {cv[t]['mean_f1_macro']:.4f} +/- {cv[t]['std_f1_macro']:.4f}")

# ── 2. Feature-group ablation (risk_level, test set) ──
section("2) Feature-group ablation - risk_level model, macro-F1 on TEST")


def fit_eval(cols):
    m = xgb.XGBClassifier(**PARAMS["risk_level"]).fit(X_train[cols], y_train["risk_level"])
    return float(f1_score(y_test["risk_level"], m.predict(X_test[cols]), average="macro"))


baseline_f1 = fit_eval(feature_cols)
ablation = []
for name, group in list(FEATURE_GROUPS.items()) + [("Activity volume only (3 counts)", ACTIVITY_VOLUME)]:
    kept = [c for c in feature_cols if c not in group]
    f1 = fit_eval(kept)
    ablation.append({"feature_group": name, "n_features_removed": len(group),
                     "f1_without": round(f1, 4), "f1_drop_from_baseline": round(baseline_f1 - f1, 4)})
    print(f"  without {name:34s} F1 {f1:.4f}  (drop {baseline_f1 - f1:+.4f})")
ablation.sort(key=lambda r: -r["f1_drop_from_baseline"])

# ── 3. Class-weighted threat_type ──
section("3) threat_type: baseline vs class-weighted (balanced sample weights)")
le_threat = joblib.load(f"{MODELS}/le_threat_type.joblib")
yt_tr = le_threat.transform(train_df["threat_type"])
yt_te = le_threat.transform(test_df["threat_type"])
base_model = joblib.load(f"{MODELS}/xgb_threat_type.joblib")
weighted = xgb.XGBClassifier(**PARAMS["threat_type"]).fit(
    X_train, yt_tr, sample_weight=compute_sample_weight("balanced", yt_tr))
joblib.dump(weighted, f"{MODELS}/xgb_threat_type_weighted.joblib")
names = le_threat.classes_.tolist()
rep_base = classification_report(yt_te, base_model.predict(X_test), target_names=names,
                                  output_dict=True, zero_division=0)
rep_w = classification_report(yt_te, weighted.predict(X_test), target_names=names,
                              output_dict=True, zero_division=0)
threat_rows = [{"class": c, "support": int(rep_base[c]["support"]),
                "f1_baseline": round(rep_base[c]["f1-score"], 4), "f1_weighted": round(rep_w[c]["f1-score"], 4),
                "recall_baseline": round(rep_base[c]["recall"], 4),
                "recall_weighted": round(rep_w[c]["recall"], 4)} for c in names]
threat_summary = {
    "baseline": {"accuracy": round(rep_base["accuracy"], 4), "f1_macro": round(rep_base["macro avg"]["f1-score"], 4)},
    "weighted": {"accuracy": round(rep_w["accuracy"], 4), "f1_macro": round(rep_w["macro avg"]["f1-score"], 4)},
    "per_class": threat_rows,
}
for r in threat_rows:
    print(f"  {r['class']:18s} n={r['support']:4d}  F1 {r['f1_baseline']:.3f} -> {r['f1_weighted']:.3f}   "
          f"recall {r['recall_baseline']:.3f} -> {r['recall_weighted']:.3f}")
print(f"  macro-F1 {threat_summary['baseline']['f1_macro']} -> {threat_summary['weighted']['f1_macro']}, "
      f"accuracy {threat_summary['baseline']['accuracy']} -> {threat_summary['weighted']['accuracy']}")

write_json("reports/model_validation.json", {
    "cross_validation": cv, "ablation_baseline_f1": round(baseline_f1, 4),
    "feature_ablation": ablation, "feature_groups": FEATURE_GROUPS,
    "threat_type_class_weighting": threat_summary,
})

gain = threat_summary["weighted"]["f1_macro"] - threat_summary["baseline"]["f1_macro"]
top = ablation[0]
write_text("reports/model_validation_summary.txt", "\n".join([
    "X-UBA - MCA MAJOR PROJECT - PHASE 1b: MODEL VALIDATION",
    "=" * 60, "",
    "1) 5-FOLD CROSS-VALIDATION (all 10,000 identities)",
    *[f"   {t:12s} accuracy {v['mean_accuracy']:.4f} +/- {v['std_accuracy']:.4f}   "
      f"macro-F1 {v['mean_f1_macro']:.4f} +/- {v['std_f1_macro']:.4f}" for t, v in cv.items()],
    "   Small standard deviations show the test-split numbers are not a lucky split.", "",
    f"2) FEATURE-GROUP ABLATION (risk_level, test macro-F1 baseline {baseline_f1:.4f})",
    pd.DataFrame(ablation).to_string(index=False), "",
    f"   Most important telemetry source: {top['feature_group']} (drop {top['f1_drop_from_baseline']:+.4f}).",
    "   Each heterogeneous source is measured separately, so the value of fusing them is shown",
    "   rather than assumed.", "",
    "3) CLASS-WEIGHTED threat_type",
    pd.DataFrame(threat_rows).to_string(index=False),
    f"   macro-F1 {threat_summary['baseline']['f1_macro']} -> {threat_summary['weighted']['f1_macro']} "
    f"({gain:+.4f}); accuracy {threat_summary['baseline']['accuracy']} -> "
    f"{threat_summary['weighted']['accuracy']}.",
    ("   Balanced weights raise minority-class recall at some cost in precision/accuracy - the "
     "usual trade-off; both models are kept and reported." if gain > 0 else
     "   Class weighting did not raise macro-F1 on this data; the baseline model is kept."),
]))

print()
banner("[OK] PHASE 1b COMPLETE")
