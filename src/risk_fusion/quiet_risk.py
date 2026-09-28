#!/usr/bin/env python3
"""
X-UBA: Explainable User and Entity Behavior Analytics
       Using Heterogeneous Security Telemetry

Module: Quiet Risk - Cross-Layer Disagreement Analysis

X-UBA scores every identity on two independent layers:

  BEHAVIOURAL  what the identity DOES   - Phase 1 out-of-fold risk index
  STRUCTURAL   what the identity COULD DO / which rules it breaks
               = mean percentile of blast radius (Phase 5), SOD severity
                 (Phase 8a) and compliance gap (Phase 8b)

Where the layers agree, the answer is easy. Where they DISAGREE is the
interesting part:

  QUIET_RISK        behaviourally quiet (not predicted HIGH/CRITICAL) but in
                    the top 20% structurally - dangerous accounts that a
                    behaviour-only detector would never surface
  NOISY_BENIGN      behaviourally flagged but structurally unremarkable
  CONSENSUS_RISK    both layers high
  CONSENSUS_BENIGN  both layers low

Disagreement score = structural percentile - behavioural percentile (0-100).

Run from ANYWHERE - this script anchors itself to the project root.
Requires: outputs of Phases 1, 5, 8a, 8b
Output: reports/quiet_risk.csv
        reports/quiet_risk_summary.txt
        reports/quiet_risk_metrics.json
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    RISK_ORDER, init, banner, section, write_text, write_json, require
)

init()
banner("X-UBA | MCA MAJOR PROJECT | QUIET RISK: Cross-Layer Disagreement")

STRUCTURAL_TOP_PCT = 20      # "structurally high" = top 20% of the organisation
BEHAV_HIGH = ["HIGH", "CRITICAL"]

require("reports/identity_risk_scores.csv", "reports/blast_radius.csv",
        "reports/sod_identity_summary.csv", "reports/compliance_gaps.csv")
d = pd.read_csv("reports/identity_risk_scores.csv")[
    ["identity_id", "xgb_risk_index", "xgb_risk_level_pred", "true_is_anomaly", "true_risk_level"]]
d = d.merge(pd.read_csv("reports/blast_radius.csv")[
    ["identity_id", "blast_radius_score", "reachable_critical"]], on="identity_id")
d = d.merge(pd.read_csv("reports/sod_identity_summary.csv")[
    ["identity_id", "department", "sod_severity_score", "sod_rules"]], on="identity_id")
d = d.merge(pd.read_csv("reports/compliance_gaps.csv")[
    ["identity_id", "privilege_level", "compliance_score"]], on="identity_id")
d["sod_rules"] = d["sod_rules"].fillna("")

d["behavioural_pct"] = (d["xgb_risk_index"].rank(pct=True) * 100).round(2)
d["structural_score"] = ((d["blast_radius_score"].rank(pct=True) +
                          d["sod_severity_score"].rank(pct=True) +
                          (100 - d["compliance_score"]).rank(pct=True)) / 3 * 100).round(2)
d["structural_pct"] = (d["structural_score"].rank(pct=True) * 100).round(2)
struct_high = d["structural_pct"] > 100 - STRUCTURAL_TOP_PCT
behav_high = d["xgb_risk_level_pred"].isin(BEHAV_HIGH)
d["quadrant"] = np.select(
    [behav_high & struct_high, ~behav_high & struct_high, behav_high & ~struct_high],
    ["CONSENSUS_RISK", "QUIET_RISK", "NOISY_BENIGN"], default="CONSENSUS_BENIGN")
d["disagreement_score"] = (d["structural_pct"] - d["behavioural_pct"]).round(2)
d = d.sort_values("disagreement_score", ascending=False)
d.to_csv("reports/quiet_risk.csv", index=False)

quad = d.groupby("quadrant").agg(
    identities=("identity_id", "count"),
    true_anomaly_rate=("true_is_anomaly", "mean"),
    mean_blast_radius=("blast_radius_score", "mean"),
    mean_sod_severity=("sod_severity_score", "mean"),
    mean_compliance=("compliance_score", "mean"),
    reach_critical_rate=("reachable_critical", lambda s: (s > 0).mean()),
).reindex(["CONSENSUS_RISK", "QUIET_RISK", "NOISY_BENIGN", "CONSENSUS_BENIGN"]).round(4)
quiet = d[d["quadrant"] == "QUIET_RISK"]
benign = d[d["quadrant"] == "CONSENSUS_BENIGN"]
quiet_truth = quiet["true_risk_level"].value_counts(normalize=True).reindex(RISK_ORDER, fill_value=0)
quiet_rules = quiet["sod_rules"].str.split("; ").explode()
quiet_rules = quiet_rules[quiet_rules != ""].value_counts()

section("Quadrants")
print(quad.to_string())
print(f"\nQUIET_RISK identities: {len(quiet)}  | true-anomaly rate {quiet['true_is_anomaly'].mean():.3f} "
      f"vs CONSENSUS_BENIGN {benign['true_is_anomaly'].mean():.3f}")
print(f"Quiet-risk true risk mix: {(quiet_truth * 100).round(1).to_dict()}")
print(f"Most common SOD rules in quiet risk: {quiet_rules.head(3).to_dict()}")

metrics = {
    "structural_top_pct": STRUCTURAL_TOP_PCT,
    "quadrant_counts": {q: int(n) for q, n in quad["identities"].items()},
    "quadrants": quad.reset_index().to_dict(orient="records"),
    "quiet_risk_count": int(len(quiet)),
    "quiet_risk_pct": round(len(quiet) / len(d) * 100, 2),
    "quiet_true_anomaly_rate": round(float(quiet["true_is_anomaly"].mean()), 4),
    "consensus_benign_true_anomaly_rate": round(float(benign["true_is_anomaly"].mean()), 4),
    "quiet_true_risk_mix": (quiet_truth * 100).round(2).to_dict(),
    "quiet_top_sod_rules": {k: int(v) for k, v in quiet_rules.head(6).items()},
    "quiet_reach_critical_rate": round(float((quiet["reachable_critical"] > 0).mean()), 4),
}
write_json("reports/quiet_risk_metrics.json", metrics)
write_text("reports/quiet_risk_summary.txt", "\n".join([
    "X-UBA - MCA MAJOR PROJECT - QUIET RISK: CROSS-LAYER DISAGREEMENT",
    "=" * 60, "",
    "Behavioural layer: Phase 1 out-of-fold risk (HIGH/CRITICAL = behaviourally high).",
    "Structural layer: mean percentile of blast radius, SOD severity and compliance gap;",
    f"top {STRUCTURAL_TOP_PCT}% = structurally high.", "",
    quad.to_string(), "",
    f"QUIET_RISK: {len(quiet)} identities ({metrics['quiet_risk_pct']}%) the behavioural model rates "
    "LOW/MEDIUM although their structural exposure is in the top 20%.",
    f"  ground-truth anomaly rate {quiet['true_is_anomaly'].mean():.1%} vs "
    f"{benign['true_is_anomaly'].mean():.1%} for consensus-benign identities",
    f"  {metrics['quiet_reach_critical_rate']:.0%} can reach a CRITICAL resource; most common policy "
    f"breaches: {', '.join(f'{k} ({v})' for k, v in quiet_rules.head(3).items())}", "",
    "INTERPRETATION: a behaviour-only UEBA system ranks these accounts as safe. Their risk is "
    "latent - it is in WHAT they could do if compromised, not in what they currently do - so "
    "they are prime candidates for access review even without suspicious activity. The "
    "ground-truth label measures behavioural anomaly, so a low anomaly rate here is expected "
    "and is exactly why a second, structural layer is needed.",
]))

print()
banner("[OK] QUIET RISK COMPLETE")
