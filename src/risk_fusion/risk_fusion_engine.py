#!/usr/bin/env python3
"""
X-UBA: Explainable User and Entity Behavior Analytics
       Using Heterogeneous Security Telemetry

Module: Multi-Signal Risk Fusion (Research Gap 1)

The architecture's "Risk Fusion" stage: a Logistic Regression meta-model
stacks four independent signals into one calibrated risk score.

    S1  XGBoost anomaly probability     (Phase 1, out-of-fold)   supervised
    S2  Isolation Forest anomaly score  (Phase 2)                unsupervised
    S3  Temporal trend score            (Phase 4)                time-based
    S4  Blast radius score              (Phase 5)                graph-based

Protocol (no leakage):
  * S1 is out-of-fold, so no identity is scored by a model that saw its label
  * the meta-model is fitted on TRAIN identities only
  * every signal (and the fusion) gets its F1-optimal threshold on TRAIN,
    then precision / recall / F1 / AUC are measured on the held-out TEST set
  * a 1,000-sample bootstrap on TEST gives a 95% CI for the F1 gain of the
    fusion over the best single signal

A second variant ("fusion + policy") also adds the rule-based SOD severity and
compliance gap signals (Phases 8a/8b), reported separately because those rules
look at some of the same attributes the synthetic labels were built from.

Run from ANYWHERE - this script anchors itself to the project root.
Requires: outputs of Phases 1, 2, 4, 5, 8a, 8b
Output: models/risk_fusion.joblib
        reports/fused_risk_scores.csv
        reports/risk_fusion_summary.txt
        reports/risk_fusion_metrics.json
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    SEED, init, banner, section, write_text, write_json, require, risk_band
)

init()
banner("X-UBA | MCA MAJOR PROJECT | GAP 1: Multi-Signal Risk Fusion")

SIGNALS = {
    "xgb_anomaly_proba": "S1 XGBoost (supervised)",
    "isolation_forest_anomaly_score": "S2 Isolation Forest (unsupervised)",
    "temporal_trend_score": "S3 Temporal trend",
    "blast_radius_score": "S4 Graph blast radius",
}
POLICY_SIGNALS = {"sod_severity_score": "S5 SOD severity", "compliance_gap": "S6 Compliance gap"}

require("reports/identity_risk_scores.csv", "reports/isolation_forest_all_scores.csv",
        "reports/temporal_risk_trajectories.csv", "reports/blast_radius.csv",
        "reports/sod_identity_summary.csv", "reports/compliance_gaps.csv")
d = pd.read_csv("reports/identity_risk_scores.csv")[
    ["identity_id", "split", "xgb_anomaly_proba", "true_is_anomaly"]]
d = d.merge(pd.read_csv("reports/isolation_forest_all_scores.csv")[
    ["identity_id", "isolation_forest_anomaly_score"]], on="identity_id")
d = d.merge(pd.read_csv("reports/temporal_risk_trajectories.csv")[
    ["identity_id", "temporal_trend_score"]], on="identity_id")
d = d.merge(pd.read_csv("reports/blast_radius.csv")[["identity_id", "blast_radius_score"]],
            on="identity_id")
d = d.merge(pd.read_csv("reports/sod_identity_summary.csv")[
    ["identity_id", "sod_severity_score"]], on="identity_id")
cg = pd.read_csv("reports/compliance_gaps.csv")[["identity_id", "compliance_score"]]
cg["compliance_gap"] = 100 - cg["compliance_score"]
d = d.merge(cg[["identity_id", "compliance_gap"]], on="identity_id")

train = d[d["split"] == "train"]
test = d[d["split"] == "test"]
y_tr, y_te = train["true_is_anomaly"].values, test["true_is_anomaly"].values


def best_threshold(scores, y):
    """F1-optimal threshold, chosen on TRAIN only."""
    cands = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 197)))
    f1s = [f1_score(y, scores >= t, zero_division=0) for t in cands]
    return float(cands[int(np.argmax(f1s))])


def evaluate(name, s_tr, s_te):
    t = best_threshold(s_tr, y_tr)
    pred = s_te >= t
    return {
        "signal": name,
        "threshold": round(t, 4),
        "precision": round(float(precision_score(y_te, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_te, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_te, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_te, s_te)), 4),
        "pr_auc": round(float(average_precision_score(y_te, s_te)), 4),
    }, pred


def fit_fusion(cols):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))
    model.fit(train[cols], y_tr)
    return model


rows, preds = [], {}
for col, label in SIGNALS.items():
    r, p = evaluate(label, train[col].values, test[col].values)
    rows.append(r)
    preds[label] = p

fusion_cols = list(SIGNALS)
fusion = fit_fusion(fusion_cols)
r, p_fusion = evaluate("FUSION (S1-S4)", fusion.predict_proba(train[fusion_cols])[:, 1],
                       fusion.predict_proba(test[fusion_cols])[:, 1])
rows.append(r)

policy_cols = fusion_cols + list(POLICY_SIGNALS)
fusion_policy = fit_fusion(policy_cols)
r_pol, _ = evaluate("FUSION + policy (S1-S6)", fusion_policy.predict_proba(train[policy_cols])[:, 1],
                    fusion_policy.predict_proba(test[policy_cols])[:, 1])
rows.append(r_pol)

results = pd.DataFrame(rows)
single = results[results["signal"].isin(SIGNALS.values())]
best_single = single.loc[single["f1"].idxmax()]
fusion_row = results[results["signal"] == "FUSION (S1-S4)"].iloc[0]

# ── Bootstrap CI for the F1 gain on TEST ──
rng = np.random.default_rng(SEED)
p_best = preds[best_single["signal"]]
gains = []
for _ in range(1000):
    idx = rng.integers(0, len(y_te), len(y_te))
    gains.append(f1_score(y_te[idx], p_fusion[idx], zero_division=0) -
                 f1_score(y_te[idx], p_best[idx], zero_division=0))
ci_lo, ci_hi = np.percentile(gains, [2.5, 97.5])
gain = fusion_row["f1"] - best_single["f1"]

corr = d[list(SIGNALS)].corr().round(3)
coefs = {SIGNALS[c]: round(float(v), 4) for c, v in zip(fusion_cols, fusion[-1].coef_[0])}

# ── Fused score for every identity (dashboard) ──
d["fused_risk_score"] = (fusion.predict_proba(d[fusion_cols])[:, 1] * 100).round(2)
d["fused_risk_band"] = risk_band(d["fused_risk_score"])
d["fused_flag"] = (d["fused_risk_score"] / 100 >= fusion_row["threshold"]).astype(int)
d[["identity_id", "split"] + fusion_cols + ["fused_risk_score", "fused_risk_band", "fused_flag",
                                            "true_is_anomaly"]].to_csv(
    "reports/fused_risk_scores.csv", index=False)
joblib.dump({"model": fusion, "signals": fusion_cols, "threshold": fusion_row["threshold"]},
            "models/risk_fusion.joblib")

section("Held-out TEST results (thresholds tuned on TRAIN)")
print(results.to_string(index=False))
print(f"\nBest single signal: {best_single['signal']} (F1 {best_single['f1']})")
print(f"Fusion F1 gain: {gain:+.4f}  (95% bootstrap CI {ci_lo:+.4f} to {ci_hi:+.4f})")
print(f"Meta-model coefficients (standardised): {coefs}")
print("\nSignal correlation matrix:")
print(corr.to_string())

significant = ci_lo > 0
metrics = {
    "results": results.to_dict(orient="records"),
    "best_single_signal": best_single["signal"],
    "f1_gain_vs_best_single": round(float(gain), 4),
    "f1_gain_ci95": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
    "gain_significant": bool(significant),
    "meta_coefficients": coefs,
    "signal_correlation": corr.to_dict(),
}
write_json("reports/risk_fusion_metrics.json", metrics)
write_text("reports/risk_fusion_summary.txt", "\n".join([
    "X-UBA - MCA MAJOR PROJECT - GAP 1: MULTI-SIGNAL RISK FUSION",
    "=" * 60, "",
    "Meta-model: standardised Logistic Regression over 4 signals, fitted on TRAIN identities;",
    "thresholds tuned on TRAIN; all numbers below are on the held-out TEST set (2,000).", "",
    results.to_string(index=False), "",
    f"Best single signal: {best_single['signal']} (F1 {best_single['f1']})",
    f"Fusion (S1-S4) F1: {fusion_row['f1']}  ->  gain {gain:+.4f}, "
    f"95% bootstrap CI [{ci_lo:+.4f}, {ci_hi:+.4f}]",
    ("The gain is statistically significant (CI excludes 0)." if significant else
     "The CI includes 0: on this dataset fusion is NOT significantly better than the best "
     "single signal. Reported honestly - XGBoost already captures most of the information "
     "the other signals carry."), "",
    f"Meta-model coefficients: {coefs}", "",
    "Signal correlations (low correlation = complementary information):", corr.to_string(), "",
    f"FUSION + policy (adds SOD + compliance rules): F1 {r_pol['f1']}, ROC-AUC {r_pol['roc_auc']}. "
    "Shown separately: those rules inspect attributes the synthetic labels were built from.",
]))

print()
banner("[OK] GAP 1 (RISK FUSION) COMPLETE")
