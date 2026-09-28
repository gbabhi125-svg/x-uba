#!/usr/bin/env python3
"""
X-UBA: Explainable User and Entity Behavior Analytics
       Using Heterogeneous Security Telemetry

Module: Organisational Anomaly Detection (Phase 8c) - department level

Two views of "normal" inside an organisation:

  1. DEPARTMENT vs ORGANISATION - does any department carry abnormal risk?
     Each department's averages (model risk, admin share, MFA share, blast
     radius, SOD and compliance) are z-scored against the other departments;
     |z| >= 2 is flagged. A chi-square test checks whether the ground-truth
     anomaly rate actually depends on department.

  2. IDENTITY vs ITS DEPARTMENT PEERS - who behaves unlike their colleagues?
     Modified z-score (Iglewicz & Hoaglin: 0.6745 x (x - median) / MAD) of
     activity volumes inside each department; |z| > 3.5 is a peer outlier.

Run from ANYWHERE - this script anchors itself to the project root.
Requires: Phase 1, 5, 8a, 8b outputs
Output: reports/department_risk_profile.csv
        reports/department_peer_outliers.csv
        reports/phase8c_org_summary.txt
        reports/phase8c_metrics.json
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    init, banner, section, write_text, write_json, require, load_features
)

init()
banner("X-UBA | MCA MAJOR PROJECT | PHASE 8c: Organisational Anomaly Detection")

PEER_FEATURES = ["login_count", "access_count", "data_download_volume",
                 "sensitive_resource_access", "privilege_changes_count", "failed_login_rate"]
PEER_Z = 3.5
DEPT_Z = 2.0

require("reports/identity_risk_scores.csv", "reports/blast_radius.csv",
        "reports/sod_identity_summary.csv", "reports/compliance_gaps.csv")
full_df, _, _ = load_features()
df = full_df.merge(pd.read_csv("reports/identity_risk_scores.csv")[
    ["identity_id", "xgb_risk_index"]], on="identity_id")
df = df.merge(pd.read_csv("reports/blast_radius.csv")[["identity_id", "blast_radius_score"]],
              on="identity_id")
df = df.merge(pd.read_csv("reports/sod_identity_summary.csv")[
    ["identity_id", "sod_violation_count"]], on="identity_id")
df = df.merge(pd.read_csv("reports/compliance_gaps.csv")[["identity_id", "compliance_score"]],
              on="identity_id")

# ── 1. Department vs organisation ──
dept = df.groupby("department").agg(
    identities=("identity_id", "count"),
    avg_model_risk=("xgb_risk_index", "mean"),
    true_anomaly_rate=("is_anomaly", "mean"),
    pct_admin=("privilege_level", lambda s: (s == "admin").mean()),
    pct_mfa=("mfa_enabled", "mean"),
    avg_blast_radius=("blast_radius_score", "mean"),
    sod_rate=("sod_violation_count", lambda s: (s > 0).mean()),
    avg_compliance=("compliance_score", "mean"),
)
metric_cols = [c for c in dept.columns if c != "identities"]
z = (dept[metric_cols] - dept[metric_cols].mean()) / dept[metric_cols].std(ddof=0).replace(0, np.nan)
for c in metric_cols:
    dept[f"z_{c}"] = z[c].round(2)
dept["flagged_metrics"] = z.apply(
    lambda r: ", ".join(f"{c} (z={r[c]:+.1f})" for c in metric_cols if abs(r[c]) >= DEPT_Z), axis=1)
dept = dept.round(4).sort_values("avg_model_risk", ascending=False)
dept.to_csv("reports/department_risk_profile.csv")

chi2, p_value, _, _ = chi2_contingency(pd.crosstab(df["department"], df["is_anomaly"]))
risk_range = (dept["avg_model_risk"].min(), dept["avg_model_risk"].max())

# ── 2. Identity vs department peers (modified z-score) ──
peer_z = pd.DataFrame(index=df.index)
for f in PEER_FEATURES:
    med = df.groupby("department")[f].transform("median")
    mad = df.groupby("department")[f].transform(lambda s: (s - s.median()).abs().median())
    peer_z[f] = 0.6745 * (df[f] - med) / mad.replace(0, np.nan)
peer_z = peer_z.fillna(0)
df["max_peer_z"] = peer_z.abs().max(axis=1).round(2)
df["peer_outlier_features"] = peer_z.apply(
    lambda r: ", ".join(f"{f} (z={r[f]:+.1f})" for f in PEER_FEATURES if abs(r[f]) > PEER_Z), axis=1)
outliers = df[df["max_peer_z"] > PEER_Z].sort_values("max_peer_z", ascending=False)
outliers[["identity_id", "department", "job_title", "max_peer_z", "peer_outlier_features",
          "xgb_risk_index", "risk_level", "is_anomaly"]].to_csv(
    "reports/department_peer_outliers.csv", index=False)

base_rate = float(df["is_anomaly"].mean())
outlier_rate = float(outliers["is_anomaly"].mean()) if len(outliers) else 0.0
novel = outliers[outliers["xgb_risk_index"] < 50]

section("Department risk profile")
print(dept[["identities", "avg_model_risk", "true_anomaly_rate", "pct_admin", "pct_mfa",
            "avg_compliance", "flagged_metrics"]].to_string())
print(f"\nChi-square (department vs anomaly): chi2={chi2:.2f}, p={p_value:.4f}")
section("Peer outliers (identity vs own department)")
print(f"Peer outliers: {len(outliers)}  | true-anomaly rate {outlier_rate:.3f} vs base {base_rate:.3f}")
print(f"Peer outliers the Phase 1 model scores below 50 (possible novel cases): {len(novel)}")

metrics = {
    "departments": int(len(dept)),
    "avg_model_risk_range": [round(float(risk_range[0]), 2), round(float(risk_range[1]), 2)],
    "departments_flagged": int((dept["flagged_metrics"] != "").sum()),
    "chi2_department_vs_anomaly": round(float(chi2), 3),
    "chi2_p_value": round(float(p_value), 4),
    "peer_outliers": int(len(outliers)),
    "peer_outlier_true_anomaly_rate": round(outlier_rate, 4),
    "base_anomaly_rate": round(base_rate, 4),
    "peer_outliers_low_model_risk": int(len(novel)),
}
write_json("reports/phase8c_metrics.json", metrics)

independent = p_value >= 0.05
write_text("reports/phase8c_org_summary.txt", "\n".join([
    "X-UBA - MCA MAJOR PROJECT - PHASE 8c: ORGANISATIONAL ANOMALY DETECTION",
    "=" * 60, "",
    "1) DEPARTMENT vs ORGANISATION",
    dept[["identities", "avg_model_risk", "true_anomaly_rate", "pct_admin", "pct_mfa",
          "avg_blast_radius", "sod_rate", "avg_compliance", "flagged_metrics"]].to_string(), "",
    f"Average model risk ranges only {risk_range[0]:.1f} - {risk_range[1]:.1f} across departments.",
    f"Chi-square test, department vs ground-truth anomaly: chi2={chi2:.2f}, p={p_value:.4f} -> "
    + ("NO significant dependence." if independent else "significant dependence."),
    ("This is expected: the generator assigns department independently of the risk drivers "
     "(privilege, dormancy, MFA), and department is not a strong model feature, so the model "
     "does not learn department as a spurious risk signal. Real organisations with "
     "department-specific patterns (e.g. Finance month-end spikes) would show sharper "
     "departmental anomalies." if independent else
     "Departments differ in underlying risk; flagged metrics above show where."), "",
    "2) IDENTITY vs DEPARTMENT PEERS (modified z-score > 3.5)",
    f"Peer outliers: {len(outliers)}",
    f"Ground-truth anomaly rate among peer outliers: {outlier_rate:.1%} (base rate {base_rate:.1%})",
    f"Peer outliers with Phase 1 risk < 50 (candidates for analyst review): {len(novel)}",
]))

print()
banner("[OK] PHASE 8c COMPLETE")
