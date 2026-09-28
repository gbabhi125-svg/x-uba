#!/usr/bin/env python3
"""
X-UBA: Explainable User and Entity Behavior Analytics
       Using Heterogeneous Security Telemetry

Module: AI Identity Attack Simulator (Phase 7) - flagship module

Joins every earlier phase into ONE report per identity:

    WHY     is it risky?            Phase 1 risk score + SHAP top factors
    WHO     does it behave like?    Phase 3 behavioural cluster
    WHEN    is it getting worse?    Phase 4 temporal trajectory
    WHAT    could an attacker do?   kill-chain simulation (below)
    IMPACT  how far could it spread? Phase 5 blast radius + attack path
    ACTION  what should we do?      Phase 6 best counterfactual action

Kill-chain simulation: four stage probabilities (initial access, privilege
escalation, lateral movement, data impact) derived only from observable
posture and graph reach - rules documented in src/simulator/killchain.py.
Expected attack risk = end-to-end likelihood x blast radius (0-100 scale).

Run from ANYWHERE - this script anchors itself to the project root.
    python src/simulator/attack_simulator.py                   (all identities)
    python src/simulator/attack_simulator.py --identity U00042 (print one report)
Requires: outputs of Phases 1, 3, 4, 5 and 6
Output: reports/attack_simulation.csv   (one "identity 360" row per identity)
        reports/attack_reports/top_risk_identities.txt
        reports/phase7_attack_simulator_summary.txt
        reports/phase7_metrics.json
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    RAW, init, banner, section, write_text, write_json, require, load_features
)
from killchain import kill_chain  # noqa: E402

MIN_USEFUL_REDUCTION = 1.0   # risk-index points; smaller changes are not recommended

INPUTS = ["reports/identity_risk_scores.csv", "reports/shap_identity_top_factors.csv",
          "reports/identity_clusters.csv", "reports/temporal_risk_trajectories.csv",
          "reports/blast_radius.csv", "reports/counterfactual_recommendations.csv"]

FEATURE_LABELS = {
    "login_count": "high login volume", "resource_count": "many distinct resources",
    "access_count": "high resource-access volume", "privilege_level_enc": "privilege level",
    "inactive_days": "days inactive", "n_systems": "number of systems",
    "mfa_usage_rate": "MFA usage rate", "mfa_enabled": "MFA enabled",
    "unique_devices": "number of devices", "privilege_changes_count": "privilege changes",
    "failed_login_rate": "failed-login rate", "night_login_rate": "night-login rate",
    "unique_countries": "login countries", "sensitive_resource_access": "sensitive accesses",
    "data_download_volume": "download volume", "unapproved_changes": "unapproved changes",
    "is_contractor": "contractor", "is_service": "service account",
    "system_count": "systems accessed", "admin_actions": "admin-type actions",
    "privilege_escalations": "privilege grants", "behaviour_deviation_score": "behaviour deviation",
    "privilege_change_trend": "privilege-change trend", "risk_trend": "risk trend",
    "department_enc": "department", "job_title_enc": "job title",
}


def build_table():
    require(*INPUTS, f"{RAW}/identities.csv", f"{RAW}/systems.csv")
    full_df, _, _ = load_features()
    ident = pd.read_csv(f"{RAW}/identities.csv",
                        usecols=["identity_id", "is_contractor_expired", "home_country"])
    n_systems_total = len(pd.read_csv(f"{RAW}/systems.csv"))

    t = full_df[["identity_id", "department", "job_title", "privilege_level", "mfa_enabled",
                 "is_contractor", "is_service", "inactive_days", "failed_login_rate",
                 "unique_countries", "unapproved_changes"]].merge(ident, on="identity_id")
    t = t.merge(pd.read_csv(INPUTS[0])[["identity_id", "split", "xgb_anomaly_proba",
                                        "xgb_risk_index", "xgb_risk_level_pred",
                                        "xgb_threat_pred", "true_is_anomaly",
                                        "true_risk_level", "true_threat_type"]], on="identity_id")
    t = t.merge(pd.read_csv(INPUTS[1]), on="identity_id")
    t = t.merge(pd.read_csv(INPUTS[2])[["identity_id", "cluster_name"]], on="identity_id")
    t = t.merge(pd.read_csv(INPUTS[3])[["identity_id", "trajectory", "early_warning",
                                        "temporal_trend_score"]], on="identity_id")
    t = t.merge(pd.read_csv(INPUTS[4])[["identity_id", "reachable_resources", "reachable_systems",
                                        "reachable_critical", "reachable_high",
                                        "blast_radius_score", "attack_path"]], on="identity_id")
    cf = pd.read_csv(INPUTS[5])[["identity_id", "best_action", "best_action_reduction",
                                 "risk_after_best_action", "delta_COMBINED",
                                 "risk_after_combined"]]
    t = t.merge(cf, on="identity_id", how="left")
    t["best_action"] = t["best_action"].fillna("MONITOR (not predicted HIGH/CRITICAL)")
    t["attack_path"] = t["attack_path"].fillna("")

    # ── Kill-chain stage probabilities (rules in src/simulator/killchain.py) ──
    kc = kill_chain(t, n_systems_total)
    t[kc.columns] = kc
    t["attack_rank"] = t["expected_attack_risk"].rank(ascending=False, method="first").astype(int)
    return t.sort_values("attack_rank")


def why_text(r):
    parts = []
    for k in range(1, 4):
        f = r.get(f"factor_{k}")
        if isinstance(f, str) and r.get(f"factor_{k}_shap", 0) > 0:
            # Encoded categoricals are shown as their real value, e.g. "job title = CISO"
            value = r[f[:-4]] if f.endswith("_enc") else f"{r[f'factor_{k}_value']:g}"
            parts.append(f"{FEATURE_LABELS.get(f, f)} = {value} "
                         f"(SHAP +{r[f'factor_{k}_shap']:.2f})")
    return "; ".join(parts) if parts else "no strong positive risk factor"


def report(r):
    lines = [
        f"IDENTITY {r['identity_id']}  |  {r['department']} / {r['job_title']}  |  "
        f"privilege: {r['privilege_level']}  |  MFA: {'yes' if r['mfa_enabled'] else 'NO'}",
        f"  WHY     risk index {r['xgb_risk_index']:.1f}/100 -> predicted {r['xgb_risk_level_pred']} "
        f"({r['xgb_threat_pred']}); top factors: {why_text(r)}",
        f"  WHO     behaviour cluster: {r['cluster_name']}",
        f"  WHEN    trajectory {r['trajectory']}"
        f"{'  ** EARLY WARNING **' if r['early_warning'] else ''} "
        f"(trend score {r['temporal_trend_score']:.0f}/100)",
        f"  WHAT    kill chain: access {r['p_initial_access']:.2f} -> escalate "
        f"{r['p_privilege_escalation']:.2f} -> lateral {r['p_lateral_movement']:.2f} -> impact "
        f"{r['p_data_impact']:.2f}  =  likelihood {r['attack_likelihood']:.4f}",
        f"  IMPACT  blast radius {r['blast_radius_score']:.1f}/100: {r['reachable_resources']} "
        f"resources ({r['reachable_critical']} critical) on {r['reachable_systems']} systems",
        f"          attack path: {r['attack_path'] or 'n/a'}",
    ]
    if r["best_action"].startswith("MONITOR"):
        lines.append(f"  ACTION  {r['best_action']}")
    elif r["best_action_reduction"] >= MIN_USEFUL_REDUCTION:
        lines.append(f"  ACTION  {r['best_action']}: risk {r['xgb_risk_index']:.1f} -> "
                     f"{r['risk_after_best_action']:.1f} (-{r['best_action_reduction']:.1f})")
    elif pd.notna(r["delta_COMBINED"]) and -r["delta_COMBINED"] >= MIN_USEFUL_REDUCTION:
        lines.append(f"  ACTION  no single action is enough; apply ALL applicable controls: risk "
                     f"{r['xgb_risk_index']:.1f} -> {r['risk_after_combined']:.1f} "
                     f"(-{-r['delta_COMBINED']:.1f})")
    else:
        lines.append("  ACTION  posture fixes do not lower the model score - escalate for manual "
                     "investigation / access review")
    lines.append(f"  RESULT  expected attack risk {r['expected_attack_risk']:.2f} "
                 f"(rank {r['attack_rank']} of {TOTAL})")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="X-UBA attack simulator")
    ap.add_argument("--identity", help="print the full report for one identity, e.g. U00042")
    args = ap.parse_args()

    init()
    banner("X-UBA | MCA MAJOR PROJECT | PHASE 7: AI Identity Attack Simulator")
    table = build_table()
    TOTAL = len(table)

    if args.identity:
        row = table[table["identity_id"] == args.identity.upper()]
        if row.empty:
            raise SystemExit(f"[X] Identity {args.identity} not found (valid: U00000 - U{TOTAL - 1:05d})")
        print("\n" + report(row.iloc[0]))
        sys.exit(0)

    table.to_csv("reports/attack_simulation.csv", index=False)
    os.makedirs("reports/attack_reports", exist_ok=True)
    top = table.head(25)
    write_text("reports/attack_reports/top_risk_identities.txt",
               "X-UBA - TOP 25 IDENTITIES BY EXPECTED ATTACK RISK\n" + "=" * 60 + "\n\n" +
               "\n\n".join(report(r) for _, r in top.iterrows()))

    # ── Validation against ground truth (never used by the simulator) ──
    y = table["true_is_anomaly"]
    auc_attack = roc_auc_score(y, table["expected_attack_risk"])
    auc_likelihood = roc_auc_score(y, table["attack_likelihood"])
    top100_precision = float(table.head(100)["true_is_anomaly"].mean())
    top100_critical = float((table.head(100)["true_risk_level"] == "CRITICAL").mean())
    corr = table["expected_attack_risk"].corr(table["xgb_risk_index"], method="spearman")

    section("Top 5 identities by expected attack risk")
    for _, r in table.head(5).iterrows():
        print(report(r) + "\n")

    section("Validation")
    print(f"AUC expected attack risk vs true anomaly: {auc_attack:.4f}")
    print(f"AUC attack likelihood only vs true anomaly: {auc_likelihood:.4f}")
    print(f"Top-100 simulated targets: {top100_precision:.0%} true anomalies, "
          f"{top100_critical:.0%} truly CRITICAL")
    print(f"Spearman(expected attack risk, Phase 1 risk index): {corr:.3f}")

    metrics = {
        "identities_simulated": TOTAL,
        "auc_expected_attack_risk": round(float(auc_attack), 4),
        "auc_attack_likelihood": round(float(auc_likelihood), 4),
        "top100_true_anomaly_rate": round(top100_precision, 4),
        "top100_true_critical_rate": round(top100_critical, 4),
        "spearman_vs_phase1_risk": round(float(corr), 4),
        "mean_stage_probabilities": {c: round(float(table[c].mean()), 4) for c in
                                     ["p_initial_access", "p_privilege_escalation",
                                      "p_lateral_movement", "p_data_impact"]},
    }
    write_json("reports/phase7_metrics.json", metrics)
    write_text("reports/phase7_attack_simulator_summary.txt", "\n".join([
        "X-UBA - MCA MAJOR PROJECT - PHASE 7: AI IDENTITY ATTACK SIMULATOR",
        "=" * 60, "",
        f"Identities simulated: {TOTAL} (one WHY/WHO/WHEN/WHAT/IMPACT/ACTION report each)",
        f"Mean stage probabilities: {metrics['mean_stage_probabilities']}", "",
        f"AUC of expected attack risk vs ground-truth anomaly: {auc_attack:.4f}",
        f"AUC of posture-only attack likelihood:              {auc_likelihood:.4f}",
        f"Top-100 simulated targets: {top100_precision:.0%} true anomalies, "
        f"{top100_critical:.0%} truly CRITICAL",
        f"Spearman correlation with the Phase 1 risk index: {corr:.3f}", "",
        "The simulator never sees a label: stage probabilities come from observable posture "
        "(MFA, privilege, dormancy, failed logins) and graph reach. Agreement with the "
        "ground truth is therefore independent evidence that it ranks the right identities.",
        "", "Full reports for the top 25: reports/attack_reports/top_risk_identities.txt",
        "Single identity:  python src/simulator/attack_simulator.py --identity U00042",
    ]))

    print()
    banner("[OK] PHASE 7 COMPLETE")
