#!/usr/bin/env python3
"""
X-UBA: Explainable User and Entity Behavior Analytics
       Using Heterogeneous Security Telemetry

Module: Counterfactual Risk-Reduction Engine (Phase 6) - "what if?"

For every identity the model currently rates HIGH or CRITICAL, this engine
applies realistic security actions to a COPY of its features, re-scores it
with the Phase 1 risk model, and measures the predicted risk reduction:

  ENFORCE_MFA              mfa_enabled = 1, MFA used on >= 95% of logins
  DOWNGRADE_PRIVILEGE      admin -> power-user, power-user -> user
  RECERTIFY_STALE_ACCESS   dormant account re-certified (inactive_days <= 30)
  REDUCE_SYSTEM_FOOTPRINT  access trimmed to at most 3 systems
  REVOKE_UNAPPROVED        unapproved privilege changes rolled back
  COMBINED                 every applicable action together

An action only applies where it changes something (e.g. MFA cannot be
"enforced" on an identity that already has it). Derived features are
recomputed after every change (src/simulator/interventions.py), so the
model never sees an inconsistent feature vector.

Risk index = expected value of the 4-class risk_level probabilities
(LOW 0, MEDIUM 33, HIGH 67, CRITICAL 100).

Run from ANYWHERE - this script anchors itself to the project root.
Requires: Phase 1 models + reports/identity_risk_scores.csv
Output: reports/counterfactual_recommendations.csv
        reports/phase6_counterfactual_summary.txt
        reports/phase6_metrics.json
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    MODELS, init, banner, section, write_text, write_json, require, load_features,
    load_metadata, risk_band
)
from interventions import ACTIONS, score as _score  # noqa: E402

init()
banner("X-UBA | MCA MAJOR PROJECT | PHASE 6: Counterfactual Risk-Reduction Engine")

require(f"{MODELS}/xgb_risk_level.joblib", f"{MODELS}/le_risk_level.joblib",
        "reports/identity_risk_scores.csv", "reports/shap_feature_importance.csv")

full_df, _, _ = load_features()
metadata = load_metadata()
model = joblib.load(f"{MODELS}/xgb_risk_level.joblib")
le_risk = joblib.load(f"{MODELS}/le_risk_level.joblib")
scores = pd.read_csv("reports/identity_risk_scores.csv")
shap_imp = pd.read_csv("reports/shap_feature_importance.csv")

def score(d):
    return _score(d, model, le_risk.classes_, metadata)


# ── Targets: identities currently predicted HIGH / CRITICAL (out-of-fold) ──
target_ids = scores.loc[scores["xgb_risk_level_pred"].isin(["HIGH", "CRITICAL"]), "identity_id"]
base = full_df[full_df["identity_id"].isin(target_ids)].reset_index(drop=True)
baseline = score(base)
print(f"\n[OK] {len(base)} identities predicted HIGH/CRITICAL; mean baseline risk index "
      f"{baseline.mean():.2f}")

out = pd.DataFrame({"identity_id": base["identity_id"], "baseline_risk": baseline.round(2),
                    "baseline_band": risk_band(baseline)})
combined = base.copy()
combined_applies = np.zeros(len(base), dtype=bool)
action_stats = []
for name, (fn, _, _) in ACTIONS.items():
    d = base.copy()
    applies = fn(d).values
    after = score(d)
    delta = np.where(applies, after - baseline, np.nan)
    out[f"delta_{name}"] = np.round(delta, 2)
    fn(combined)
    combined_applies |= applies
    action_stats.append({
        "action": name,
        "applicable_identities": int(applies.sum()),
        "mean_reduction_when_applicable": round(float(-np.nanmean(delta)), 2) if applies.any() else 0.0,
        "mean_reduction_over_all_targets": round(float(-np.nan_to_num(delta).mean()), 2),
    })

after_combined = score(combined)
out["delta_COMBINED"] = np.where(combined_applies, after_combined - baseline, np.nan).round(2)
out["risk_after_combined"] = np.where(combined_applies, after_combined, baseline).round(2)
out["band_after_combined"] = risk_band(out["risk_after_combined"])

delta_cols = [f"delta_{a}" for a in ACTIONS]
best_delta = out[delta_cols].min(axis=1, skipna=True)
has_action = out[delta_cols].notna().any(axis=1)
out["best_action"] = np.where(
    has_action, out[delta_cols].fillna(np.inf).idxmin(axis=1).str.replace("delta_", "", regex=False),
    "NONE")
out["best_action_reduction"] = np.where(has_action, -best_delta, 0).round(2)
out["risk_after_best_action"] = (out["baseline_risk"] - out["best_action_reduction"]).round(2)
out["band_after_best_action"] = risk_band(out["risk_after_best_action"])
out = out.sort_values("baseline_risk", ascending=False)
out.to_csv("reports/counterfactual_recommendations.csv", index=False)

stats_df = pd.DataFrame(action_stats).sort_values("mean_reduction_over_all_targets", ascending=False)
best_counts = out["best_action"].value_counts().to_dict()

# ── Convergent validation with SHAP: are the most effective actions the ones
#    that touch the features SHAP ranks highest? ──
shap_rank = {f: i + 1 for i, f in enumerate(shap_imp["feature"])}
stats_df["best_shap_rank_of_touched_features"] = [
    min(shap_rank.get(f, 999) for f in ACTIONS[a][1]) for a in stats_df["action"]]
rank_corr = stats_df["mean_reduction_over_all_targets"].rank(ascending=False).corr(
    stats_df["best_shap_rank_of_touched_features"].rank(), method="spearman")

high_before = out["baseline_band"].isin(["HIGH", "CRITICAL"]).sum()
leave_best = (out["baseline_band"].isin(["HIGH", "CRITICAL"]) &
              ~out["band_after_best_action"].isin(["HIGH", "CRITICAL"])).sum()
leave_comb = (out["baseline_band"].isin(["HIGH", "CRITICAL"]) &
              ~out["band_after_combined"].isin(["HIGH", "CRITICAL"])).sum()

section("Average predicted risk reduction per action (risk-index points)")
print(stats_df.to_string(index=False))
print(f"\nMost frequent best single action: {best_counts}")
print(f"Mean reduction - best single action: {out['best_action_reduction'].mean():.2f}; "
      f"combined: {-out['delta_COMBINED'].mean():.2f}")
print(f"Identities leaving HIGH/CRITICAL band: best action {leave_best}/{high_before}, "
      f"combined {leave_comb}/{high_before}")
print(f"Spearman(action effectiveness, SHAP rank of touched features) = {rank_corr:.3f}")

metrics = {
    "target_identities": int(len(out)),
    "mean_baseline_risk": round(float(out["baseline_risk"].mean()), 2),
    "actions": stats_df.to_dict(orient="records"),
    "best_action_counts": best_counts,
    "mean_reduction_best_action": round(float(out["best_action_reduction"].mean()), 2),
    "mean_reduction_combined": round(float(-out["delta_COMBINED"].mean()), 2),
    "leave_high_band_best_action": int(leave_best),
    "leave_high_band_combined": int(leave_comb),
    "high_band_before": int(high_before),
    "spearman_effectiveness_vs_shap_rank": round(float(rank_corr), 3),
}
write_json("reports/phase6_metrics.json", metrics)

lines = [
    "X-UBA - MCA MAJOR PROJECT - PHASE 6: COUNTERFACTUAL RISK-REDUCTION ENGINE",
    "=" * 60, "",
    f"Identities analysed (predicted HIGH/CRITICAL): {len(out)}",
    f"Mean baseline risk index: {metrics['mean_baseline_risk']}", "",
    "Average predicted risk reduction per action:", stats_df.to_string(index=False), "",
    f"Most frequent best single action: {best_counts}",
    f"Mean reduction with the best single action: {metrics['mean_reduction_best_action']} points",
    f"Mean reduction with all applicable actions combined: {metrics['mean_reduction_combined']} points",
    f"Identities moved out of the HIGH/CRITICAL band: best single action {leave_best}/{high_before}, "
    f"combined {leave_comb}/{high_before}", "",
    f"Convergent validation: Spearman correlation between action effectiveness and the SHAP "
    f"rank of the features each action changes = {rank_corr:.3f} (positive = the explanations "
    "and the recommended actions agree).",
    "",
    "NOTE: reductions are the MODEL's predicted change (a what-if on the trained model), "
    "not an observed outcome; they show which lever the model is most sensitive to.",
]
write_text("reports/phase6_counterfactual_summary.txt", "\n".join(lines))

print()
banner("[OK] PHASE 6 COMPLETE")
