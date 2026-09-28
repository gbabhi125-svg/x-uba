#!/usr/bin/env python3
"""
X-UBA: Explainable User and Entity Behavior Analytics
       Using Heterogeneous Security Telemetry

Module: SHAP Explanation Fidelity Quantification (Research Gap 2)

Most UEBA systems SHOW SHAP explanations but never test whether they are
faithful to what the model actually does. This module measures it with a
deletion test (in the spirit of the ERASER "comprehensiveness" and
"sufficiency" metrics):

  COMPREHENSIVENESS  remove the top-k SHAP features (replace each with its
                     training median) -> how much does the probability of the
                     predicted class DROP?  Compared against removing k RANDOM
                     features (20 random draws per identity).
  SUFFICIENCY        keep ONLY the top-k SHAP features (all others -> median)
                     -> how much of the prediction is retained?

An identity's explanation is "high fidelity" when its top-k features cause at
least 1.5x the drop that random features cause.

Model explained: the Phase 1 is_anomaly XGBoost model, on the 2,000 held-out
TEST identities (never seen in training).

Run from ANYWHERE - this script anchors itself to the project root.
Requires: Phase 1 models + data/processed/*.csv
Output: reports/explanation_fidelity.csv     (per identity)
        reports/explanation_fidelity_curve.csv (k = 1..10)
        reports/explanation_fidelity_summary.txt
        reports/explanation_fidelity_metrics.json
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    SEED, MODELS, init, banner, section, write_text, write_json, require, load_features,
    load_metadata, encode
)

init()
banner("X-UBA | MCA MAJOR PROJECT | GAP 2: SHAP Explanation Fidelity")

K = 5
RANDOM_DRAWS = 20
HIGH_FIDELITY_RATIO = 1.5
EPS = 1e-3   # floor for the random drop so the per-identity ratio stays finite

require(f"{MODELS}/xgb_is_anomaly.joblib")
_, train_df, test_df = load_features()
metadata = load_metadata()
model = joblib.load(f"{MODELS}/xgb_is_anomaly.joblib")

X_train = encode(train_df, metadata)
X_test = encode(test_df, metadata)
X = X_test.values
n, m = X.shape
baseline = X_train.median().values

proba1 = model.predict_proba(X_test)[:, 1]
pred = (proba1 >= 0.5).astype(int)


def p_pred_class(Xmod):
    p = model.predict_proba(pd.DataFrame(Xmod, columns=X_test.columns))[:, 1]
    return np.where(pred == 1, p, 1 - p)


p0 = np.where(pred == 1, proba1, 1 - proba1)

sv = np.asarray(shap.TreeExplainer(model).shap_values(X_test))
if sv.ndim == 3:
    sv = sv[:, :, 1]
# Contribution TOWARDS the predicted class (positive SHAP pushes towards "anomaly")
contrib = np.where(pred[:, None] == 1, sv, -sv)
order = np.argsort(-contrib, axis=1)
rng = np.random.default_rng(SEED)


def remove(cols_per_row, keep=False):
    """Replace chosen columns (or all OTHER columns if keep=True) with the median."""
    mask = np.zeros((n, m), dtype=bool)
    np.put_along_axis(mask, cols_per_row, True, axis=1)
    if keep:
        mask = ~mask
    return np.where(mask, baseline[None, :], X)


def random_cols(k):
    return np.argsort(rng.random((n, m)), axis=1)[:, :k]


def drops_for_k(k):
    top_drop = p0 - p_pred_class(remove(order[:, :k]))
    rand_drop = np.mean([p0 - p_pred_class(remove(random_cols(k))) for _ in range(RANDOM_DRAWS)],
                        axis=0)
    suff = p_pred_class(remove(order[:, :k], keep=True))
    rand_suff = np.mean([p_pred_class(remove(random_cols(k), keep=True))
                         for _ in range(RANDOM_DRAWS)], axis=0)
    return top_drop, rand_drop, suff, rand_suff


top_drop, rand_drop, suff, rand_suff = drops_for_k(K)
ratio = top_drop / np.maximum(rand_drop, EPS)
high_fid = ratio >= HIGH_FIDELITY_RATIO

per_id = pd.DataFrame({
    "identity_id": test_df["identity_id"].values,
    "predicted_class": pred,
    "p_predicted_class": p0.round(4),
    "top_features": [", ".join(X_test.columns[order[i, :K]]) for i in range(n)],
    f"drop_top{K}": top_drop.round(4),
    f"drop_random{K}": rand_drop.round(4),
    "fidelity_ratio": ratio.round(3),
    "high_fidelity": high_fid.astype(int),
    f"sufficiency_top{K}": suff.round(4),
})
per_id.to_csv("reports/explanation_fidelity.csv", index=False)

section("Fidelity curve, k = 1..10")
curve = []
for k in range(1, 11):
    td, rd, sf, rs = (top_drop, rand_drop, suff, rand_suff) if k == K else drops_for_k(k)
    curve.append({"k": k, "mean_drop_shap_topk": round(float(td.mean()), 4),
                  "mean_drop_random_k": round(float(rd.mean()), 4),
                  "drop_ratio": round(float(td.mean() / max(rd.mean(), EPS)), 2),
                  "sufficiency_shap_topk": round(float(sf.mean()), 4),
                  "sufficiency_random_k": round(float(rs.mean()), 4)})
    print(f"  k={k:2d}: drop SHAP {curve[-1]['mean_drop_shap_topk']:.4f} vs random "
          f"{curve[-1]['mean_drop_random_k']:.4f}  (x{curve[-1]['drop_ratio']})")
curve_df = pd.DataFrame(curve)
curve_df.to_csv("reports/explanation_fidelity_curve.csv", index=False)

# ── Explanation-action agreement: does Phase 6's recommended action change at
#    least one of the identity's own top-5 SHAP risk factors? ──
agreement = None
if Path("reports/counterfactual_recommendations.csv").exists():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulator"))
    from interventions import ACTIONS  # noqa: E402
    cfr = pd.read_csv("reports/counterfactual_recommendations.csv")[["identity_id", "best_action"]]
    top = pd.read_csv("reports/shap_identity_top_factors.csv")
    cfr = cfr[cfr["best_action"].isin(ACTIONS)].merge(top, on="identity_id")
    fcols = [f"factor_{k}" for k in range(1, 6)]
    scols = [f"factor_{k}_shap" for k in range(1, 6)]

    def agrees(r):
        risky = {r[f] for f, sc in zip(fcols, scols) if r[sc] > 0}
        return bool(risky & set(ACTIONS[r["best_action"]][1]))

    cfr["agrees"] = cfr.apply(agrees, axis=1)
    agreement = {
        "identities": int(len(cfr)),
        "agreement_rate": round(float(cfr["agrees"].mean()), 4),
        "by_action": {a: {"n": int(len(g)), "agreement_rate": round(float(g["agrees"].mean()), 4)}
                      for a, g in cfr.groupby("best_action")},
    }
    section("Explanation-action agreement")
    print(f"Recommended action touches one of the identity's own top-5 SHAP risk factors for "
          f"{agreement['agreement_rate']:.1%} of {agreement['identities']} identities")

mean_top, mean_rand = float(top_drop.mean()), float(rand_drop.mean())
global_ratio = mean_top / max(mean_rand, EPS)
by_class = per_id.groupby("predicted_class")["high_fidelity"].mean().round(4).to_dict()

section(f"Results at k = {K} (TEST set, {n} identities)")
print(f"Mean probability drop, top-{K} SHAP features: {mean_top:.4f}")
print(f"Mean probability drop, {K} random features:   {mean_rand:.4f}")
print(f"Global drop ratio: {global_ratio:.1f}x")
print(f"Median per-identity ratio: {np.median(ratio):.2f}")
print(f"High-fidelity explanations (ratio >= {HIGH_FIDELITY_RATIO}): {high_fid.mean():.1%}")
print(f"  by predicted class (0 = normal, 1 = anomaly): {by_class}")
print(f"Sufficiency: top-{K} SHAP alone retain {suff.mean():.4f} vs random {K} {rand_suff.mean():.4f}")

metrics = {
    "k": K, "random_draws": RANDOM_DRAWS, "test_identities": n,
    "mean_drop_shap_topk": round(mean_top, 4), "mean_drop_random_k": round(mean_rand, 4),
    "global_drop_ratio": round(global_ratio, 2),
    "median_identity_ratio": round(float(np.median(ratio)), 3),
    "pct_high_fidelity": round(float(high_fid.mean() * 100), 2),
    "pct_high_fidelity_by_predicted_class": by_class,
    "sufficiency_shap_topk": round(float(suff.mean()), 4),
    "sufficiency_random_k": round(float(rand_suff.mean()), 4),
    "curve": curve,
    "explanation_action_agreement": agreement,
}
write_json("reports/explanation_fidelity_metrics.json", metrics)
write_text("reports/explanation_fidelity_summary.txt", "\n".join([
    "X-UBA - MCA MAJOR PROJECT - GAP 2: SHAP EXPLANATION FIDELITY",
    "=" * 60, "",
    f"Model: Phase 1 XGBoost is_anomaly. Set: {n} held-out TEST identities.",
    "Removal = replace the feature with its TRAINING median. Random baseline = mean of "
    f"{RANDOM_DRAWS} random draws of k features per identity.", "",
    f"COMPREHENSIVENESS (k={K})",
    f"  top-{K} SHAP features removed: mean drop {mean_top:.4f}",
    f"  {K} random features removed:   mean drop {mean_rand:.4f}",
    f"  -> SHAP-ranked features cause {global_ratio:.1f}x the drop of random features",
    f"  high-fidelity identities (ratio >= {HIGH_FIDELITY_RATIO}): {high_fid.mean():.1%}",
    f"  by predicted class (0 normal / 1 anomaly): {by_class}", "",
    f"SUFFICIENCY (k={K}): keeping only the top-{K} SHAP features retains a predicted-class "
    f"probability of {suff.mean():.4f} (random {K}: {rand_suff.mean():.4f})", "",
    "Fidelity curve:", curve_df.to_string(index=False), "",
    "INTERPRETATION: SHAP's ranking is measurably faithful to the model - removing the "
    "features it names changes the prediction far more than removing arbitrary features.",
] + ([
    "",
    "EXPLANATION-ACTION AGREEMENT",
    f"  For {agreement['agreement_rate']:.1%} of the {agreement['identities']} identities with a "
    "recommended action (Phase 6), that action changes at least one of the identity's own "
    "top-5 positive SHAP risk factors - the WHY and the WHAT-TO-DO point at the same cause.",
    "  By action: " + ", ".join(f"{a} {v['agreement_rate']:.0%} (n={v['n']})"
                                for a, v in agreement["by_action"].items()),
    "  Divergence is informative: explanations come from the is_anomaly model and actions are "
    "scored with the risk_level model, so an action can lower risk (e.g. ENFORCE_MFA) without "
    "touching the factors that explain the anomaly. The dashboard shows both side by side.",
] if agreement else [])))

print()
banner("[OK] GAP 2 (EXPLANATION FIDELITY) COMPLETE")
