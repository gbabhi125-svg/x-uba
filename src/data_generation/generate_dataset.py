#!/usr/bin/env python3
"""
X-UBA: Explainable User and Entity Behavior Analytics
       Using Heterogeneous Security Telemetry

Module: Data Generation
Produces 6 raw tables + 3 processed feature tables at 10,000-identity scale,
simulating heterogeneous security telemetry (identity, authentication,
privilege, and resource-access logs) as would be pulled from a real
hybrid IAM environment (AD, Azure AD, AWS IAM, Okta, Salesforce, etc.).

Run from ANYWHERE — this script anchors itself to the project root,
so: python src/data_generation/generate_dataset.py   (from repo root)
or:  python generate_dataset.py                       (from this folder)
both work identically.

Output: data/raw/*.csv , data/processed/*.csv
"""

import pandas as pd
import numpy as np
import random
import os
from pathlib import Path
from datetime import datetime, timedelta

# ── Anchor to project root regardless of where this script is invoked from ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

np.random.seed(42)
random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

N_IDENTITIES = 10000
N_LOGIN_EVENTS = 300000
N_PRIVILEGE_CHANGES = 30000
N_RESOURCE_ACCESS = 150000

START_DATE = datetime(2025, 9, 16)
END_DATE = datetime(2026, 9, 16)
DAYS_SPAN = (END_DATE - START_DATE).days

DEPARTMENTS = ["Sales", "Marketing", "Security", "Engineer", "Compliance",
               "Legal", "HR", "IT", "Finance", "Operations"]

ROLE_BY_DEPT = {
    "Sales": ["Sales Rep", "Sales Dir", "Sales Mgr"],
    "Marketing": ["Marketing user", "Content Mktg", "Marketing power-us"],
    "Security": ["Security user", "CISO", "Security power-us"],
    "Engineer": ["Software user", "Engineer power-us", "DevOps user", "Tech Lead"],
    "Compliance": ["Compliance user", "Compliance power-us", "Recruiter user"],
    "Legal": ["Legal user"],
    "HR": ["HR Director", "HR Man", "Recruiter user"],
    "IT": ["IT Suppo user", "Network user", "IT Manag user", "System Admin"],
    "Finance": ["Financial user", "Controller user"],
    "Operations": ["Operatio user", "Ops Man user", "Supply Ch user"]
}

PRIVILEGE_LEVELS = ["user", "power-user", "admin"]
COUNTRIES = ["India", "United States", "United Kingdom", "Germany", "Canada",
             "Poland", "Singapore", "Australia", "Netherlands", "Brazil"]
DEVICE_TYPES = ["Laptop-{:02d}", "MobilePhone-{:02d}", "Desktop-{:02d}", "Tablet-{:02d}"]

SYSTEM_DEFS = [
    ("SYS001", "Active_Directory", "identity", "medium"),
    ("SYS002", "Azure_AD", "identity", "medium"),
    ("SYS003", "AWS_IAM", "cloud", "high"),
    ("SYS004", "GitHub", "devtools", "medium"),
    ("SYS005", "Jira", "productivity", "low"),
    ("SYS006", "Salesforce", "crm", "high"),
    ("SYS007", "EMAIL", "productivity", "medium"),
    ("SYS008", "PROD_DB", "database", "critical"),
    ("SYS009", "Data_Lake", "cloud", "critical"),
    ("SYS010", "VPN", "network", "high"),
    ("SYS011", "BI_Tool", "productivity", "medium"),
    ("SYS012", "Confluence", "productivity", "low"),
    ("SYS013", "GCP", "cloud", "high"),
    ("SYS014", "HRIS", "identity", "high"),
    ("SYS015", "Okta", "identity", "medium"),
]

RESOURCE_TYPE_BY_SYSTEM_TYPE = {
    "identity": ["GroupPolicy", "UserObject", "AccessList"],
    "cloud": ["S3Bucket", "Pipeline", "ComputeInstance", "IAMRole"],
    "devtools": ["Repo", "Secret", "Workflow"],
    "productivity": ["Page", "Document", "Dashboard"],
    "crm": ["Account", "Contact", "Opportunity"],
    "database": ["Schema", "Table", "Backup"],
    "network": ["Tunnel", "Certificate"],
}

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

print("=" * 70)
print("MCA CAPSTONE DATASET GENERATOR — Target: 10,000+ identities")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# 1. SYSTEMS
# ═══════════════════════════════════════════════════════════════════════════

systems_df = pd.DataFrame(SYSTEM_DEFS, columns=["system_id", "system_name", "system_type", "sensitivity"])
systems_df.to_csv("data/raw/systems.csv", index=False)
print(f"✅ systems.csv           -> {len(systems_df)} rows")

# ═══════════════════════════════════════════════════════════════════════════
# 2. RESOURCES
# ═══════════════════════════════════════════════════════════════════════════

resources = []
resource_counter = 0
for _, srow in systems_df.iterrows():
    n_res = random.randint(15, 35)
    rtypes = RESOURCE_TYPE_BY_SYSTEM_TYPE.get(srow["system_type"], ["Object"])
    for _ in range(n_res):
        resources.append({
            "resource_id": f"RES{resource_counter:05d}",
            "resource_name": f"{srow['system_name']}_{random.choice(rtypes)}_{resource_counter}",
            "system_id": srow["system_id"],
            "system_name": srow["system_name"],
            "resource_type": random.choice(rtypes),
            "sensitivity": random.choices(
                ["low", "medium", "high", "critical"],
                weights=[0.3, 0.35, 0.25, 0.10]
            )[0]
        })
        resource_counter += 1

resources_df = pd.DataFrame(resources)
resources_df.to_csv("data/raw/resources.csv", index=False)
print(f"✅ resources.csv         -> {len(resources_df)} rows")

# ═══════════════════════════════════════════════════════════════════════════
# 3. IDENTITIES (core risk ground-truth lives here)
# ═══════════════════════════════════════════════════════════════════════════

identities = []
for i in range(N_IDENTITIES):
    identity_id = f"U{i:05d}"
    dept = random.choice(DEPARTMENTS)
    role = random.choice(ROLE_BY_DEPT[dept])
    name_first = f"user{i}"
    name_last = f"ln{i}"
    name = f"{name_first}.{name_last}"
    email = f"{name}@company.com"

    privilege_level = random.choices(PRIVILEGE_LEVELS, weights=[0.50, 0.32, 0.18])[0]
    has_admin = int(privilege_level == "admin")
    is_contractor = int(random.random() < 0.18)
    is_contractor_expired = int(is_contractor and random.random() < 0.35)
    is_service_account = int(random.random() < 0.10)
    n_systems = random.randint(1, 8)
    home_country = random.choices(COUNTRIES, weights=[0.35, 0.2, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02])[0]
    mfa_enabled = int(random.random() < 0.68)
    hire_days_ago = random.randint(1, 2500)
    hire_date = (END_DATE - timedelta(days=hire_days_ago)).strftime("%Y-%m-%d")
    days_inactive = min(int(np.random.exponential(65)), 400)
    is_active = int(days_inactive < 180)

    # ── Ground-truth risk construction (deterministic + explainable) ──
    risk_score = 0
    threat_type = "NORMAL"

    if has_admin and days_inactive > 60:
        risk_score += 35
        threat_type = "PRIVILEGE_CREEP"
    if privilege_level == "power-user" and days_inactive > 90:
        risk_score += 18
        threat_type = "PRIVILEGE_CREEP" if threat_type == "NORMAL" else threat_type
    if is_contractor_expired:
        risk_score += 30
        threat_type = "PRIVILEGE_CREEP" if threat_type == "NORMAL" else threat_type
    if not mfa_enabled and privilege_level != "user":
        risk_score += 22
        threat_type = "ACCOUNT_TAKEOVER" if threat_type == "NORMAL" else threat_type
    if not mfa_enabled and privilege_level == "user":
        risk_score += 8
    if is_service_account and days_inactive > 45:
        risk_score += 18
    if days_inactive > 150:
        risk_score += 20
        threat_type = "INSIDER_THREAT" if (threat_type == "NORMAL" and random.random() < 0.3) else threat_type
    if n_systems >= 6:
        risk_score += 10
    if random.random() < 0.07:  # brute-force / exfil ground truth injection
        risk_score += random.randint(35, 55)
        threat_type = random.choice(["BRUTE_FORCE", "DATA_EXFILTRATION"])

    risk_score += np.random.randint(0, 15)
    risk_score = int(min(max(risk_score, 0), 100))

    if risk_score >= 55:
        risk_level = "CRITICAL"
    elif risk_score >= 35:
        risk_level = "HIGH"
    elif risk_score >= 18:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    is_anomaly = int(risk_score >= 40)
    anomaly_duration_days = random.randint(1, 60) if is_anomaly else -1

    identities.append({
        "identity_id": identity_id,
        "name": name,
        "email": email,
        "department": dept,
        "job_title": role,
        "role": role,
        "privilege_level": privilege_level,
        "has_admin": has_admin,
        "is_contractor": is_contractor,
        "is_contractor_expired": is_contractor_expired,
        "is_service_account": is_service_account,
        "n_systems": n_systems,
        "home_country": home_country,
        "mfa_enabled": mfa_enabled,
        "hire_date": hire_date,
        "days_inactive": days_inactive,
        "is_active": is_active,
        "threat_type": threat_type,
        "threat_score": risk_score,
        "risk_level": risk_level,
        "is_anomaly": is_anomaly,
        "anomaly_duration_days": anomaly_duration_days
    })

identities_df = pd.DataFrame(identities)
identities_df.to_csv("data/raw/identities.csv", index=False)
print(f"✅ identities.csv        -> {len(identities_df)} rows")
print(f"   Risk distribution: {identities_df['risk_level'].value_counts().to_dict()}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. LOGIN EVENTS (vectorized for speed at 300k rows)
# ═══════════════════════════════════════════════════════════════════════════

id_array = identities_df["identity_id"].values
risk_array = identities_df["threat_score"].values
home_country_array = identities_df["home_country"].values
mfa_array = identities_df["mfa_enabled"].values

# Weight event allocation: riskier / more active identities get more logins
weights = 1 + (risk_array / 20.0)
weights = weights / weights.sum()

chosen_idx = np.random.choice(len(id_array), size=N_LOGIN_EVENTS, p=weights)

login_rows = []
for k, idx in enumerate(chosen_idx):
    iid = id_array[idx]
    base_risk = risk_array[idx]
    home_ctry = home_country_array[idx]
    mfa_on = mfa_array[idx]

    ts = START_DATE + timedelta(
        days=random.randint(0, DAYS_SPAN),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )
    hour = ts.hour
    time_class = "night" if hour < 6 else ("off_hours" if hour >= 20 or hour < 8 else "business_hours")

    # Anomalous behavior scales with underlying identity risk
    is_anom_roll = random.random() < (0.03 + base_risk / 400.0)
    country = home_ctry if (not is_anom_roll or random.random() < 0.7) else random.choice(COUNTRIES)
    device_id = random.choice(DEVICE_TYPES).format(random.randint(0, 9))
    source_ip = f"10.{random.randint(0,9)}.{random.randint(0,50)}.{random.randint(1,254)}"
    mfa_used = int(mfa_on and random.random() < 0.95)
    status = "success" if random.random() < (0.93 if not is_anom_roll else 0.6) else "failed"

    anomaly_type = "NONE"
    is_anomaly = 0
    if is_anom_roll:
        is_anomaly = 1
        if status == "failed":
            anomaly_type = "BRUTE_FORCE"
        elif country != home_ctry:
            anomaly_type = "IMPOSSIBLE_TRAVEL"
        elif time_class in ("night", "off_hours"):
            anomaly_type = "OFF_HOURS_ACCESS"
        else:
            anomaly_type = random.choice(["UNUSUAL_DEVICE", "MFA_BYPASS_ATTEMPT"])

    login_rows.append((f"LGN{k:06d}", iid, ts.strftime("%Y-%m-%d %H:%M:%S"), country,
                        device_id, source_ip, status, mfa_used, time_class, is_anomaly, anomaly_type))

login_df = pd.DataFrame(login_rows, columns=[
    "event_id", "identity_id", "timestamp", "country", "device_id", "source_ip",
    "status", "mfa_used", "time_class", "is_anomaly", "anomaly_type"
])
login_df.to_csv("data/raw/login_events.csv", index=False)
print(f"✅ login_events.csv      -> {len(login_df)} rows")

# ═══════════════════════════════════════════════════════════════════════════
# 5. PRIVILEGE CHANGES
# ═══════════════════════════════════════════════════════════════════════════

chosen_idx_pc = np.random.choice(len(id_array), size=N_PRIVILEGE_CHANGES, p=weights)
priv_rows = []
change_types = ["grant", "revoke", "modify"]
approvers = [f"APR{n:04d}" for n in range(1, 200)]

for k, idx in enumerate(chosen_idx_pc):
    iid = id_array[idx]
    base_risk = risk_array[idx]
    ts = START_DATE + timedelta(days=random.randint(0, DAYS_SPAN), hours=random.randint(0, 23))
    ctype = random.choices(change_types, weights=[0.45, 0.25, 0.30])[0]
    old_priv = random.choice(PRIVILEGE_LEVELS)
    new_priv = random.choice(PRIVILEGE_LEVELS) if ctype != "grant" else random.choices(
        PRIVILEGE_LEVELS, weights=[0.3, 0.4, 0.3])[0]
    sys_name = random.choice(systems_df["system_name"].tolist())

    is_anom_roll = random.random() < (0.02 + base_risk / 500.0)
    approval_status = "approved"
    if is_anom_roll:
        approval_status = random.choices(["approved", "pending", "unapproved"], weights=[0.5, 0.2, 0.3])[0]

    anomaly_type = "NONE"
    is_anomaly = 0
    if is_anom_roll:
        is_anomaly = 1
        if new_priv == "admin" and old_priv == "user":
            anomaly_type = "UNAUTHORIZED_ESCALATION"
        elif approval_status == "unapproved":
            anomaly_type = "ORPHANED_GRANT"
        else:
            anomaly_type = "PRIVILEGE_CREEP"

    priv_rows.append((f"PRV{k:05d}", iid, ts.strftime("%Y-%m-%d %H:%M:%S"), ctype, old_priv,
                       new_priv, sys_name, random.choice(approvers), approval_status, is_anomaly, anomaly_type))

priv_df = pd.DataFrame(priv_rows, columns=[
    "change_id", "identity_id", "timestamp", "change_type", "old_privilege", "new_privilege",
    "system_name", "approved_by", "approval_status", "is_anomaly", "anomaly_type"
])
priv_df.to_csv("data/raw/privilege_changes.csv", index=False)
print(f"✅ privilege_changes.csv -> {len(priv_df)} rows")

# ═══════════════════════════════════════════════════════════════════════════
# 6. RESOURCE ACCESS
# ═══════════════════════════════════════════════════════════════════════════

res_ids = resources_df["resource_id"].values
res_names = resources_df["resource_name"].values
res_systems = resources_df["system_name"].values
res_sensitivity = resources_df["sensitivity"].values

chosen_idx_ra = np.random.choice(len(id_array), size=N_RESOURCE_ACCESS, p=weights)
actions = ["read", "write", "download", "export", "share", "api_call", "sql_query", "search"]

access_rows = []
for k, idx in enumerate(chosen_idx_ra):
    iid = id_array[idx]
    base_risk = risk_array[idx]
    ridx = random.randint(0, len(res_ids) - 1)

    ts = START_DATE + timedelta(days=random.randint(0, DAYS_SPAN), hours=random.randint(0, 23))
    action = random.choices(actions, weights=[0.4, 0.15, 0.15, 0.08, 0.05, 0.07, 0.06, 0.04])[0]
    sensitivity = res_sensitivity[ridx]

    is_anom_roll = random.random() < (0.02 + base_risk / 450.0)
    data_volume = 0.0
    if action in ("download", "export"):
        data_volume = round(np.random.exponential(5) * (3 if is_anom_roll else 1), 2)

    status = "success" if random.random() < 0.97 else "denied"
    anomaly_type = "NONE"
    is_anomaly = 0
    if is_anom_roll:
        is_anomaly = 1
        if action in ("download", "export") and data_volume > 20:
            anomaly_type = "BULK_EXPORT"
        elif sensitivity in ("high", "critical"):
            anomaly_type = "SENSITIVE_DATA_EXFILTRATION"
        else:
            anomaly_type = "UNUSUAL_RESOURCE_ACCESS"

    access_rows.append((f"RSA{k:06d}", iid, ts.strftime("%Y-%m-%d %H:%M:%S"), res_ids[ridx],
                         res_names[ridx], res_systems[ridx], sensitivity, action, data_volume,
                         status, is_anomaly, anomaly_type))

access_df = pd.DataFrame(access_rows, columns=[
    "access_id", "identity_id", "timestamp", "resource_id", "resource_name", "system_name",
    "sensitivity", "action", "data_volume", "status", "is_anomaly", "anomaly_type"
])
access_df.to_csv("data/raw/resource_access.csv", index=False)
print(f"✅ resource_access.csv   -> {len(access_df)} rows")

# ═══════════════════════════════════════════════════════════════════════════
# 7. FEATURE ENGINEERING -> identity_features.csv
# ═══════════════════════════════════════════════════════════════════════════

print("\n🔧 Building identity_features (aggregating raw tables)...")

login_agg = login_df.groupby("identity_id").agg(
    login_count=("event_id", "count"),
    failed_login_rate=("status", lambda s: (s == "failed").mean()),
    night_login_rate=("time_class", lambda s: (s == "night").mean()),
    unique_countries=("country", "nunique"),
    unique_devices=("device_id", "nunique"),
    mfa_usage_rate=("mfa_used", "mean")
).reset_index()

priv_agg = priv_df.groupby("identity_id").agg(
    privilege_changes_count=("change_id", "count"),
    privilege_escalations=("change_type", lambda s: (s == "grant").sum()),
    unapproved_changes=("approval_status", lambda s: (s == "unapproved").sum())
).reset_index()

access_agg = access_df.groupby("identity_id").agg(
    access_count=("access_id", "count"),
    resource_count=("resource_id", "nunique"),
    system_count=("system_name", "nunique"),
    sensitive_resource_access=("sensitivity", lambda s: s.isin(["high", "critical"]).sum()),
    data_download_volume=("data_volume", "sum"),
    admin_actions=("action", lambda s: s.isin(["export", "sql_query", "api_call"]).sum())
).reset_index()

features = identities_df[[
    "identity_id", "department", "job_title", "privilege_level", "is_service_account",
    "is_contractor", "days_inactive", "mfa_enabled", "n_systems", "risk_level", "threat_type",
    "is_anomaly"
]].rename(columns={"is_service_account": "is_service", "days_inactive": "inactive_days"})

features = features.merge(login_agg, on="identity_id", how="left")
features = features.merge(priv_agg, on="identity_id", how="left")
features = features.merge(access_agg, on="identity_id", how="left")
features = features.fillna(0)

# Derived signals
features["behaviour_deviation_score"] = (
    features["failed_login_rate"] * 0.3 +
    features["night_login_rate"] * 0.2 +
    (features["unique_countries"] > 2).astype(int) * 0.25 +
    (features["unapproved_changes"] > 0).astype(int) * 0.25
).round(4)

features["privilege_change_trend"] = (
    features["privilege_escalations"] - features["privilege_changes_count"] * 0.5
).round(4)

features["risk_trend"] = (
    features["behaviour_deviation_score"] * 0.5 + features["privilege_change_trend"].clip(lower=0) * 0.1
).round(4)

identity_features_df = features.copy()
identity_features_df.to_csv("data/processed/identity_features.csv", index=False)
print(f"✅ identity_features.csv -> {len(identity_features_df)} rows, {len(identity_features_df.columns)} columns")

# ═══════════════════════════════════════════════════════════════════════════
# 8. TRAIN/TEST SPLIT (80/20, stratified by risk_level)
# ═══════════════════════════════════════════════════════════════════════════

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    identity_features_df,
    test_size=0.2,
    random_state=42,
    stratify=identity_features_df["risk_level"]
)

train_df.to_csv("data/processed/train_features.csv", index=False)
test_df.to_csv("data/processed/test_features.csv", index=False)
print(f"✅ train_features.csv    -> {len(train_df)} rows")
print(f"✅ test_features.csv     -> {len(test_df)} rows")

# ═══════════════════════════════════════════════════════════════════════════
# README SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

summary = f"""IAM Dataset Summary — MCA Capstone
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RAW TABLES (data/raw/)
- systems.csv            {len(systems_df)} rows
- resources.csv          {len(resources_df)} rows
- identities.csv         {len(identities_df)} rows
- login_events.csv       {len(login_df)} rows
- privilege_changes.csv  {len(priv_df)} rows
- resource_access.csv    {len(access_df)} rows

PROCESSED TABLES (data/processed/)
- identity_features.csv  {len(identity_features_df)} rows, {len(identity_features_df.columns)} cols
- train_features.csv     {len(train_df)} rows (80%)
- test_features.csv      {len(test_df)} rows (20%)

RISK DISTRIBUTION (identities.csv)
{identities_df['risk_level'].value_counts().to_string()}

THREAT TYPE DISTRIBUTION
{identities_df['threat_type'].value_counts().to_string()}

Random seed: 42 (fully reproducible)
"""

with open("data/processed/README_data_summary.txt", "w") as f:
    f.write(summary)

print("\n" + "=" * 70)
print("✅ DATASET GENERATION COMPLETE")
print("=" * 70)
print(summary)