#!/usr/bin/env python3
"""
X-UBA: Explainable User and Entity Behavior Analytics
       Using Heterogeneous Security Telemetry

Module: Graph-Aware Remediation Optimizer

A security team can only fix so many accounts per review cycle. Given a fixed
budget of 200 identities (one action each), which ones should be fixed?

  NAIVE (behaviour-only)  rank by the model-predicted risk reduction of each
                          identity's best action (what Phase 6 recommends)
  IMPACT-AWARE            rank by the reduction in EXPECTED ATTACK RISK
                          (kill-chain likelihood x blast radius, Phase 7), with
                          the blast radius RECOMPUTED on the privilege graph
                          after the action (e.g. a privilege downgrade removes
                          admin reach; trimming to 3 systems removes the rest)
  BALANCED                both reductions scaled to 0-1 and added (equal weight)

Both strategies are scored on the same three outcomes:
  * total model-risk reduction           (what a behaviour-only view values)
  * total expected attack risk removed   (organisation-wide exposure)
  * critical-resource access paths removed

Candidates: every identity predicted HIGH/CRITICAL plus every QUIET_RISK identity.

Run from ANYWHERE - this script anchors itself to the project root.
Requires: outputs of Phases 1, 5, 7 and Quiet Risk; data/raw/*.csv
Output: reports/remediation_candidates.csv
        reports/remediation_plan.csv
        reports/remediation_optimizer_summary.txt
        reports/remediation_optimizer_metrics.json
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_SRC))
sys.path.insert(0, str(ROOT_SRC / "simulator"))
sys.path.insert(0, str(ROOT_SRC / "graph_analysis"))
from common import (  # noqa: E402
    MODELS, init, banner, section, write_text, write_json, read_json, require, load_features,
    load_metadata
)
from interventions import ACTIONS, GRAPH_ACTIONS, graph_after_action, score  # noqa: E402
from killchain import kill_chain  # noqa: E402
from reach import ReachModel  # noqa: E402

init()
banner("X-UBA | MCA MAJOR PROJECT | REMEDIATION OPTIMIZER (fixed budget)")

BUDGET = 200

require("reports/identity_risk_scores.csv", "reports/blast_radius.csv",
        "reports/attack_simulation.csv", "reports/quiet_risk.csv", "reports/phase5_metrics.json",
        f"{MODELS}/xgb_risk_level.joblib")
full_df, _, _ = load_features()
metadata = load_metadata()
model = joblib.load(f"{MODELS}/xgb_risk_level.joblib")
classes = joblib.load(f"{MODELS}/le_risk_level.joblib").classes_
raw_max = read_json("reports/phase5_metrics.json")["blast_radius_raw_max"]
rm = ReachModel()
n_systems_total = len(rm.systems)

scores = pd.read_csv("reports/identity_risk_scores.csv")
quiet = pd.read_csv("reports/quiet_risk.csv")[["identity_id", "quadrant"]]
attack = pd.read_csv("reports/attack_simulation.csv")[
    ["identity_id", "expected_attack_risk", "blast_radius_score", "reachable_systems",
     "reachable_critical", "reachable_high", "is_contractor_expired"]]

cand_ids = set(scores.loc[scores["xgb_risk_level_pred"].isin(["HIGH", "CRITICAL"]), "identity_id"]) | \
    set(quiet.loc[quiet["quadrant"] == "QUIET_RISK", "identity_id"])
base = full_df[full_df["identity_id"].isin(cand_ids)].reset_index(drop=True)
base = base.merge(attack, on="identity_id").merge(quiet, on="identity_id")
base_risk = score(base, model, classes, metadata)
print(f"\n[OK] {len(base)} candidates (predicted HIGH/CRITICAL + QUIET_RISK), budget {BUDGET}")


rows = []
for name, (fn, _, _) in ACTIONS.items():
    after = base.copy()
    applies = fn(after).values
    risk_after = score(after, model, classes, metadata)
    # Graph-side effects of the action
    for col in ("blast_radius_score", "reachable_systems", "reachable_critical", "reachable_high"):
        after[col] = base[col]
    if name in GRAPH_ACTIONS:
        for i in np.flatnonzero(applies):
            g = graph_after_action(rm, base.at[i, "identity_id"], name, base.at[i, "privilege_level"],
                                   raw_max)
            for col, v in g.items():
                after.at[i, col] = v
    ear_after = kill_chain(after, n_systems_total)["expected_attack_risk"].values
    for i in np.flatnonzero(applies):
        rows.append({
            "identity_id": base.at[i, "identity_id"], "quadrant": base.at[i, "quadrant"],
            "action": name,
            "risk_reduction": round(float(base_risk[i] - risk_after[i]), 3),
            "ear_before": float(base.at[i, "expected_attack_risk"]),
            "ear_reduction": round(float(base.at[i, "expected_attack_risk"] - ear_after[i]), 4),
            "critical_paths_removed": int(base.at[i, "reachable_critical"] - after.at[i, "reachable_critical"]),
        })

cands = pd.DataFrame(rows)
# Balanced objective: each reduction scaled by its largest value, then summed
cands["balanced_value"] = (cands["risk_reduction"] / cands["risk_reduction"].max() +
                           cands["ear_reduction"] / cands["ear_reduction"].max()).round(5)
cands.to_csv("reports/remediation_candidates.csv", index=False)


def plan(value_col):
    """Best action per identity by value_col, then the top-BUDGET identities."""
    best = cands.sort_values([value_col, "identity_id"], ascending=[False, True]) \
        .drop_duplicates("identity_id")
    return best.head(BUDGET)


strategies = {"NAIVE (behaviour-only)": plan("risk_reduction"),
              "IMPACT-AWARE (graph + kill chain)": plan("ear_reduction"),
              "BALANCED (both, equal weight)": plan("balanced_value")}
org_ear = float(attack["expected_attack_risk"].sum())
results = []
for sname, p in strategies.items():
    results.append({
        "strategy": sname,
        "identities_fixed": len(p),
        "total_risk_reduction": round(float(p["risk_reduction"].sum()), 2),
        "total_ear_removed": round(float(p["ear_reduction"].sum()), 2),
        "pct_org_ear_removed": round(float(p["ear_reduction"].sum()) / org_ear * 100, 2),
        "critical_paths_removed": int(p["critical_paths_removed"].sum()),
        "quiet_risk_identities_fixed": int((p["quadrant"] == "QUIET_RISK").sum()),
        "action_mix": p["action"].value_counts().to_dict(),
    })
res = pd.DataFrame(results)
naive, aware, balanced = results
overlap = len(set(strategies["NAIVE (behaviour-only)"]["identity_id"]) &
              set(strategies["IMPACT-AWARE (graph + kill chain)"]["identity_id"]))
ear_gain = (aware["total_ear_removed"] / naive["total_ear_removed"] - 1) * 100 \
    if naive["total_ear_removed"] else float("nan")
risk_cost = (aware["total_risk_reduction"] / naive["total_risk_reduction"] - 1) * 100 \
    if naive["total_risk_reduction"] else float("nan")

plan_df = pd.concat([p.assign(strategy=s) for s, p in strategies.items()])
plan_df.to_csv("reports/remediation_plan.csv", index=False)

section(f"Budget = {BUDGET} identities, one action each")
print(res.drop(columns=["action_mix"]).to_string(index=False))
print(f"\nImpact-aware removes {ear_gain:+.1f}% more expected attack risk, at {risk_cost:+.1f}% "
      f"model-risk reduction; the two plans share {overlap} identities.")
for r in results:
    print(f"  {r['strategy']}: {r['action_mix']}")

metrics = {
    "budget": BUDGET, "candidates": int(cands["identity_id"].nunique()),
    "org_expected_attack_risk": round(org_ear, 2),
    "strategies": results, "overlap_identities": overlap,
    "ear_gain_pct_impact_vs_naive": round(float(ear_gain), 2),
    "risk_reduction_change_pct_impact_vs_naive": round(float(risk_cost), 2),
    "balanced_keeps_pct_of_naive_risk_reduction":
        round(balanced["total_risk_reduction"] / naive["total_risk_reduction"] * 100, 1),
    "balanced_keeps_pct_of_impact_ear": round(balanced["total_ear_removed"] / aware["total_ear_removed"] * 100, 1),
}
write_json("reports/remediation_optimizer_metrics.json", metrics)
write_text("reports/remediation_optimizer_summary.txt", "\n".join([
    "X-UBA - MCA MAJOR PROJECT - GRAPH-AWARE REMEDIATION OPTIMIZER",
    "=" * 60, "",
    f"Budget: {BUDGET} identities per review cycle, one action each. Candidates: "
    f"{metrics['candidates']} (predicted HIGH/CRITICAL + QUIET_RISK).",
    f"Organisation-wide expected attack risk before remediation: {org_ear:.1f}", "",
    res.drop(columns=["action_mix"]).to_string(index=False), "",
    *[f"{r['strategy']} action mix: {r['action_mix']}" for r in results], "",
    f"Impact-aware vs naive: {ear_gain:+.1f}% expected attack risk removed, {risk_cost:+.1f}% "
    f"model-risk reduction. Plans share {overlap} of {BUDGET} identities.",
    f"Balanced plan keeps {balanced['total_risk_reduction'] / naive['total_risk_reduction']:.0%} of the "
    f"naive risk reduction and {balanced['total_ear_removed'] / aware['total_ear_removed']:.0%} of the "
    "impact-aware attack-risk removal.", "",
    "INTERPRETATION: each single-objective plan wins on its own objective, and the two plans "
    "barely overlap - so the choice of objective decides WHO gets fixed. "
    "Ranking fixes by the behavioural model alone optimises the model's score; "
    "ranking by graph-recomputed expected attack risk targets the accounts whose compromise "
    "would do the most damage. The difference between the two plans is the value of making "
    "remediation impact-aware. Both use the same model and the same actions; only the "
    "objective changes. Reductions are model/simulation estimates, not observed outcomes.",
]))

print()
banner("[OK] REMEDIATION OPTIMIZER COMPLETE")
