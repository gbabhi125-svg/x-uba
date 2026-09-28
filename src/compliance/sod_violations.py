#!/usr/bin/env python3
"""
X-UBA: Explainable User and Entity Behavior Analytics
       Using Heterogeneous Security Telemetry

Module: Separation of Duties (SOD) & Access-Policy Violations (Phase 8a)

Deterministic, auditable rules evaluated on the raw IAM tables. Unlike the ML
phases, every flag here can be traced to exact records, which is what an
auditor needs.

  ID   RULE                         SEVERITY  EVIDENCE
  R1   ADMIN_WITHOUT_MFA            HIGH      identities: admin & MFA off
  R2   EXPIRED_CONTRACTOR_ACTIVE    CRITICAL  identities: contract expired, account active
  R3   PRIVILEGED_SERVICE_ACCOUNT   HIGH      identities: service account with admin rights
  R4   UNAPPROVED_ADMIN_GRANT       CRITICAL  privilege_changes: admin granted, not approved
  R5   TOXIC_DEV_PROD_COMBINATION   HIGH      resource_access: writes code (GitHub) AND
                                              changes production data (PROD_DB write/SQL)
  R6   IDENTITY_ADMIN_DATA_EXPORT   CRITICAL  admin on an identity system (AD / Azure AD /
                                              Okta) AND exports data from PROD_DB / Data_Lake
                                              (can create accounts AND remove data)

Run from ANYWHERE - this script anchors itself to the project root.
Requires: data/raw/*.csv
Output: reports/sod_violations.csv          (one row per violation, with evidence)
        reports/sod_identity_summary.csv    (per-identity counts + severity score)
        reports/phase8a_sod_summary.txt
        reports/phase8a_metrics.json
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    RAW, RISK_ORDER, init, banner, section, write_text, write_json, require
)

init()
banner("X-UBA | MCA MAJOR PROJECT | PHASE 8a: Separation of Duties Violations")

SEVERITY_POINTS = {"HIGH": 2, "CRITICAL": 3}
IDENTITY_SYSTEMS = {"Active_Directory", "Azure_AD", "Okta"}
DATA_STORES = {"PROD_DB", "Data_Lake"}

require(f"{RAW}/identities.csv", f"{RAW}/privilege_changes.csv", f"{RAW}/resource_access.csv")
ident = pd.read_csv(f"{RAW}/identities.csv")
privs = pd.read_csv(f"{RAW}/privilege_changes.csv")
access = pd.read_csv(f"{RAW}/resource_access.csv")
ok = access[access["status"] == "success"]

violations = []


def add(rule, severity, ids, evidence):
    for iid, ev in zip(ids, evidence):
        violations.append({"identity_id": iid, "rule": rule, "severity": severity, "evidence": ev})


# R1
m = (ident["has_admin"] == 1) & (ident["mfa_enabled"] == 0)
add("ADMIN_WITHOUT_MFA", "HIGH", ident.loc[m, "identity_id"],
    ["privilege_level=admin, mfa_enabled=0"] * m.sum())

# R2
m = (ident["is_contractor_expired"] == 1) & (ident["is_active"] == 1)
add("EXPIRED_CONTRACTOR_ACTIVE", "CRITICAL", ident.loc[m, "identity_id"],
    [f"contract expired, account active (last seen {d} days ago)"
     for d in ident.loc[m, "days_inactive"]])

# R3
m = (ident["is_service_account"] == 1) & (ident["has_admin"] == 1)
add("PRIVILEGED_SERVICE_ACCOUNT", "HIGH", ident.loc[m, "identity_id"],
    ["service account holds admin privilege"] * m.sum())

# R4
bad = privs[(privs["new_privilege"] == "admin") & privs["change_type"].isin(["grant", "modify"]) &
            privs["approval_status"].isin(["unapproved", "pending"])]
bad_g = bad.groupby("identity_id").agg(
    n=("change_id", "count"), ids=("change_id", lambda s: ", ".join(s[:3])),
    systems=("system_name", lambda s: ", ".join(sorted(set(s))[:3]))).reset_index()
add("UNAPPROVED_ADMIN_GRANT", "CRITICAL", bad_g["identity_id"],
    [f"{n} admin change(s) not approved on {sy} (e.g. {i})"
     for n, i, sy in zip(bad_g["n"], bad_g["ids"], bad_g["systems"])])

# R5
code_writers = set(ok.loc[(ok["system_name"] == "GitHub") & (ok["action"] == "write"), "identity_id"])
prod_changers = set(ok.loc[(ok["system_name"] == "PROD_DB") &
                           ok["action"].isin(["write", "sql_query"]), "identity_id"])
toxic = sorted(code_writers & prod_changers)
add("TOXIC_DEV_PROD_COMBINATION", "HIGH", toxic,
    ["writes to GitHub AND writes/queries PROD_DB"] * len(toxic))

# R6
id_admins = set(privs.loc[(privs["new_privilege"] == "admin") &
                          privs["change_type"].isin(["grant", "modify"]) &
                          privs["system_name"].isin(IDENTITY_SYSTEMS), "identity_id"])
exporters = set(ok.loc[ok["system_name"].isin(DATA_STORES) &
                       ok["action"].isin(["export", "download"]), "identity_id"])
combo = sorted(id_admins & exporters)
add("IDENTITY_ADMIN_DATA_EXPORT", "CRITICAL", combo,
    ["admin on an identity system AND exports from PROD_DB/Data_Lake"] * len(combo))

viol = pd.DataFrame(violations)
viol.to_csv("reports/sod_violations.csv", index=False)

# ── Per-identity summary ──
viol["points"] = viol["severity"].map(SEVERITY_POINTS)
per_id = viol.groupby("identity_id").agg(
    sod_violation_count=("rule", "count"),
    sod_severity_score=("points", "sum"),
    sod_rules=("rule", lambda s: "; ".join(sorted(s)))).reset_index()
summary = ident[["identity_id", "department", "risk_level", "is_anomaly"]].merge(
    per_id, on="identity_id", how="left")
summary["sod_violation_count"] = summary["sod_violation_count"].fillna(0).astype(int)
summary["sod_severity_score"] = summary["sod_severity_score"].fillna(0).astype(int)
summary["sod_rules"] = summary["sod_rules"].fillna("")
summary.to_csv("reports/sod_identity_summary.csv", index=False)

# ── Report ──
rule_counts = viol.groupby(["rule", "severity"]).size().reset_index(name="identities")
rule_counts = rule_counts.sort_values("identities", ascending=False)
flagged = summary[summary["sod_violation_count"] > 0]
rate_by_level = (summary.assign(flag=summary["sod_violation_count"] > 0)
                 .groupby("risk_level")["flag"].mean().reindex(RISK_ORDER) * 100).round(1)
anom_flagged = float(flagged["is_anomaly"].mean())
anom_clean = float(summary.loc[summary["sod_violation_count"] == 0, "is_anomaly"].mean())

section("Violations by rule")
print(rule_counts.to_string(index=False))
print(f"\nIdentities with >= 1 violation: {len(flagged)} ({len(flagged) / len(summary):.1%})")
print(f"Identities with >= 2 violations: {(summary['sod_violation_count'] >= 2).sum()}")
print(f"% flagged by true risk level: {rate_by_level.to_dict()}")
print(f"True-anomaly rate: flagged {anom_flagged:.3f} vs not flagged {anom_clean:.3f}")

metrics = {
    "total_violations": int(len(viol)),
    "identities_flagged": int(len(flagged)),
    "pct_identities_flagged": round(len(flagged) / len(summary) * 100, 2),
    "identities_with_2plus": int((summary["sod_violation_count"] >= 2).sum()),
    "by_rule": {r: int(n) for r, n in zip(rule_counts["rule"], rule_counts["identities"])},
    "pct_flagged_by_risk_level": rate_by_level.to_dict(),
    "true_anomaly_rate_flagged": round(anom_flagged, 4),
    "true_anomaly_rate_not_flagged": round(anom_clean, 4),
}
write_json("reports/phase8a_metrics.json", metrics)
write_text("reports/phase8a_sod_summary.txt", "\n".join([
    "X-UBA - MCA MAJOR PROJECT - PHASE 8a: SEPARATION OF DUTIES VIOLATIONS",
    "=" * 60, "",
    f"Rules evaluated: 6   Total violations: {len(viol)}",
    f"Identities with >= 1 violation: {len(flagged)} ({metrics['pct_identities_flagged']}%)",
    f"Identities with >= 2 violations: {metrics['identities_with_2plus']}", "",
    "Violations by rule:", rule_counts.to_string(index=False), "",
    f"% of identities flagged, by ground-truth risk level: {rate_by_level.to_dict()}",
    f"Ground-truth anomaly rate: flagged {anom_flagged:.1%} vs not flagged {anom_clean:.1%}", "",
    "Every violation row in reports/sod_violations.csv carries its evidence (record IDs, "
    "systems), so each flag can be verified by an auditor without trusting a model.",
]))

print()
banner("[OK] PHASE 8a COMPLETE")
