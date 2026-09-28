#!/usr/bin/env python3
"""
X-UBA: Explainable User and Entity Behavior Analytics
       Using Heterogeneous Security Telemetry

Module: Compliance Gap Analysis (Phase 8b) - NIST SP 800-53 / GDPR Art. 32

Maps each identity's observable state to specific control requirements and
reports where the organisation falls short. The mapping is an ASSESSMENT
AID (it shows which records would draw an auditor's attention) - it is not
a certification of compliance.

  CONTROL        REQUIREMENT (summary)                       GAP WHEN                     WEIGHT
  AC-2(3)        disable accounts inactive beyond a period   inactive > 90 days            15
  AC-2(2)        remove temporary accounts when expired      expired contractor present    20
  AC-2(4)/AC-2   account changes authorised and audited      pending/unapproved change     15
  AC-6           least privilege                             admin on >= 6 systems         15
  IA-2(1)        MFA for privileged accounts                 admin/power-user, no MFA      20
  IA-2(2)        MFA for non-privileged accounts             user, no MFA                   5
  GDPR Art.32    security of processing personal data        bulk export (> 20 units) of   15
                                                             high/critical data, or
                                                             critical-data access w/o MFA

Compliance score = 100 - sum of weights of failed controls (minimum 0).

Run from ANYWHERE - this script anchors itself to the project root.
Requires: data/raw/*.csv
Output: reports/compliance_gaps.csv            (per identity, one column per control)
        reports/compliance_by_department.csv
        reports/phase8b_compliance_summary.txt
        reports/phase8b_metrics.json
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    RAW, RISK_ORDER, init, banner, section, write_text, write_json, require
)

init()
banner("X-UBA | MCA MAJOR PROJECT | PHASE 8b: Compliance Gap Analysis (NIST / GDPR)")

CONTROLS = {
    "AC-2(3)_inactive_account": ("NIST AC-2(3) Disable inactive accounts", 15),
    "AC-2(2)_expired_temporary": ("NIST AC-2(2) Remove expired temporary accounts", 20),
    "AC-2(4)_unapproved_change": ("NIST AC-2(4) Authorised account changes", 15),
    "AC-6_least_privilege": ("NIST AC-6 Least privilege", 15),
    "IA-2(1)_mfa_privileged": ("NIST IA-2(1) MFA for privileged accounts", 20),
    "IA-2(2)_mfa_standard": ("NIST IA-2(2) MFA for non-privileged accounts", 5),
    "GDPR_Art32_processing": ("GDPR Art.32 Security of processing", 15),
}

require(f"{RAW}/identities.csv", f"{RAW}/privilege_changes.csv", f"{RAW}/resource_access.csv")
ident = pd.read_csv(f"{RAW}/identities.csv")
privs = pd.read_csv(f"{RAW}/privilege_changes.csv")
access = pd.read_csv(f"{RAW}/resource_access.csv")

unapproved_ids = set(privs.loc[privs["approval_status"].isin(["pending", "unapproved"]),
                               "identity_id"])
sens = access["sensitivity"].isin(["high", "critical"])
bulk_ids = set(access.loc[sens & access["action"].isin(["download", "export"]) &
                          (access["data_volume"] > 20), "identity_id"])
critical_ids = set(access.loc[access["sensitivity"] == "critical", "identity_id"])
no_mfa_ids = set(ident.loc[ident["mfa_enabled"] == 0, "identity_id"])

g = ident[["identity_id", "department", "privilege_level", "risk_level", "is_anomaly"]].copy()
g["AC-2(3)_inactive_account"] = ident["days_inactive"] > 90
g["AC-2(2)_expired_temporary"] = ident["is_contractor_expired"] == 1
g["AC-2(4)_unapproved_change"] = ident["identity_id"].isin(unapproved_ids)
g["AC-6_least_privilege"] = (ident["has_admin"] == 1) & (ident["n_systems"] >= 6)
g["IA-2(1)_mfa_privileged"] = (ident["privilege_level"] != "user") & (ident["mfa_enabled"] == 0)
g["IA-2(2)_mfa_standard"] = (ident["privilege_level"] == "user") & (ident["mfa_enabled"] == 0)
g["GDPR_Art32_processing"] = (ident["identity_id"].isin(bulk_ids) |
                              ident["identity_id"].isin(critical_ids & no_mfa_ids))

ctrl_cols = list(CONTROLS)
weights = np.array([CONTROLS[c][1] for c in ctrl_cols])
g[ctrl_cols] = g[ctrl_cols].astype(int)
g["gap_count"] = g[ctrl_cols].sum(axis=1)
g["compliance_score"] = np.maximum(100 - g[ctrl_cols].values @ weights, 0)
g["failed_controls"] = g[ctrl_cols].apply(
    lambda r: "; ".join(CONTROLS[c][0] for c in ctrl_cols if r[c]), axis=1)
g.to_csv("reports/compliance_gaps.csv", index=False)

by_dept = g.groupby("department").agg(
    identities=("identity_id", "count"),
    avg_compliance_score=("compliance_score", "mean"),
    pct_with_gap=("gap_count", lambda s: (s > 0).mean() * 100),
    **{c: (c, "sum") for c in ctrl_cols}).round(2).sort_values("avg_compliance_score")
by_dept.to_csv("reports/compliance_by_department.csv")

control_table = pd.DataFrame({
    "control": [CONTROLS[c][0] for c in ctrl_cols],
    "weight": weights,
    "identities_failing": [int(g[c].sum()) for c in ctrl_cols],
    "pct_failing": [round(g[c].mean() * 100, 1) for c in ctrl_cols],
}).sort_values("identities_failing", ascending=False)

pct_any = (g["gap_count"] > 0).mean() * 100
avg_score = g["compliance_score"].mean()
score_by_level = g.groupby("risk_level")["compliance_score"].mean().reindex(RISK_ORDER).round(1)
score_corr = g["compliance_score"].corr(ident["threat_score"])

section("Control failures")
print(control_table.to_string(index=False))
print(f"\nIdentities with >= 1 gap: {pct_any:.1f}%    Average compliance score: {avg_score:.1f}/100")
print(f"Average score by true risk level: {score_by_level.to_dict()}")
print(f"Correlation of compliance score with ground-truth threat score: {score_corr:.3f}")
print("\nLeast compliant departments:")
print(by_dept[["identities", "avg_compliance_score", "pct_with_gap"]].head(3).to_string())

metrics = {
    "pct_identities_with_gap": round(float(pct_any), 2),
    "average_compliance_score": round(float(avg_score), 2),
    "fully_compliant_identities": int((g["gap_count"] == 0).sum()),
    "controls": control_table.to_dict(orient="records"),
    "score_by_risk_level": score_by_level.to_dict(),
    "corr_score_vs_threat": round(float(score_corr), 4),
    "department_scores": by_dept["avg_compliance_score"].to_dict(),
}
write_json("reports/phase8b_metrics.json", metrics)
write_text("reports/phase8b_compliance_summary.txt", "\n".join([
    "X-UBA - MCA MAJOR PROJECT - PHASE 8b: COMPLIANCE GAP ANALYSIS (NIST SP 800-53 / GDPR)",
    "=" * 60, "",
    f"Identities assessed: {len(g)}   Controls: {len(ctrl_cols)}",
    f"Identities with at least one gap: {pct_any:.1f}%",
    f"Fully compliant identities: {metrics['fully_compliant_identities']}",
    f"Average compliance score: {avg_score:.1f}/100", "",
    "Control failures:", control_table.to_string(index=False), "",
    f"Average compliance score by ground-truth risk level: {score_by_level.to_dict()}",
    f"Correlation with ground-truth threat score: {score_corr:.3f} (negative = riskier "
    "identities are less compliant, as expected)", "",
    "Department view:", by_dept[["identities", "avg_compliance_score", "pct_with_gap"]].to_string(),
    "",
    "INTERPRETATION: the high share of identities with a gap reflects identity sprawl "
    "(dormant accounts and missing MFA are common in the synthetic estate), which is the "
    "problem X-UBA targets. This is an assessment aid, not a compliance certification.",
]))

print()
banner("[OK] PHASE 8b COMPLETE")
