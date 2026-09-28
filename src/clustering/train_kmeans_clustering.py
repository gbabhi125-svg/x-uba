#!/usr/bin/env python3
"""
X-UBA: Explainable User and Entity Behavior Analytics
       Using Heterogeneous Security Telemetry

Module: Behavioural Clustering (Phase 3) - K-Means

Groups all 10,000 identities into behavioural peer groups WITHOUT using any
label. Each cluster is then named automatically from what makes its centroid
different from the organisation average (e.g. "Stale + Admin-Heavy + MFA-Gap"),
and profiled against the ground-truth risk levels AFTER fitting - so the risk
labels describe the clusters but never shape them.

k is fixed at 4 for interpretability (four named peer groups on the
dashboard); the silhouette score for k = 2..8 is reported alongside so the
choice is transparent.

Run from ANYWHERE - this script anchors itself to the project root.
Requires: data/processed/identity_features.csv
Output: models/kmeans_clustering.joblib
        reports/identity_clusters.csv
        reports/cluster_profiles.csv
        reports/phase3_clustering_summary.txt
        reports/phase3_metrics.json
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    SEED, PRIVILEGE_LEVELS, RISK_ORDER, init, banner, section, write_text, write_json,
    load_features
)

init()
banner("X-UBA | MCA MAJOR PROJECT | PHASE 3: K-Means Behavioural Clustering")

N_CLUSTERS = 4

# Behavioural / posture features only. No target column is used for fitting.
CLUSTER_FEATURES = [
    "privilege_ordinal", "is_service", "is_contractor", "inactive_days", "mfa_enabled",
    "n_systems", "login_count", "failed_login_rate", "night_login_rate",
    "unique_countries", "unique_devices", "mfa_usage_rate", "privilege_changes_count",
    "unapproved_changes", "access_count", "sensitive_resource_access",
    "data_download_volume",
]
# Heavy-tailed counts are log-scaled so a few extreme identities don't dominate
LOG_FEATURES = ["inactive_days", "login_count", "privilege_changes_count", "access_count",
                "sensitive_resource_access", "data_download_volume", "unapproved_changes"]

full_df, _, _ = load_features()
df = full_df.copy()
df["privilege_ordinal"] = df["privilege_level"].map({p: i for i, p in enumerate(PRIVILEGE_LEVELS)})

X = df[CLUSTER_FEATURES].astype(float).copy()
for c in LOG_FEATURES:
    X[c] = np.log1p(X[c])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\n[OK] Clustering {len(df)} identities on {len(CLUSTER_FEATURES)} behavioural features")

# ── Silhouette sweep (transparency for the choice of k) ──
section("Silhouette score by k (sampled, 4,000 identities)")
silhouettes = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_scaled)
    silhouettes[k] = round(float(silhouette_score(
        X_scaled, km.labels_, sample_size=4000, random_state=SEED)), 4)
    print(f"  k={k}: silhouette={silhouettes[k]}")
best_k = max(silhouettes, key=silhouettes.get)

# ── Final model ──
kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=SEED).fit(X_scaled)
df["cluster_id"] = kmeans.labels_
dist = kmeans.transform(X_scaled)
df["distance_to_centroid"] = dist[np.arange(len(df)), kmeans.labels_].round(4)


# ── Automatic naming from centroid vs organisation average ──
def name_cluster(g, org):
    tags = []
    if g["is_service"].mean() > 0.5:
        tags.append("Service Accounts")
    if g["inactive_days"].median() > 1.5 * org["inactive_days"].median():
        tags.append("Stale")
    if (g["privilege_level"] == "admin").mean() > 1.5 * (org["privilege_level"] == "admin").mean():
        tags.append("Admin-Heavy")
    if g["mfa_enabled"].mean() < org["mfa_enabled"].mean() - 0.15:
        tags.append("MFA-Gap")
    if g["is_contractor"].mean() > 1.5 * org["is_contractor"].mean():
        tags.append("Contractor-Heavy")
    if g["login_count"].mean() > 1.2 * org["login_count"].mean():
        tags.append("High-Activity")
    if g["login_count"].mean() < 0.85 * org["login_count"].mean():
        tags.append("Low-Activity")
    if g["n_systems"].mean() > 1.2 * org["n_systems"].mean():
        tags.append("Broad-Access")
    high_share = g["risk_level"].isin(["HIGH", "CRITICAL"]).mean()
    if high_share >= 0.5:
        tags.append("High-Risk")
    elif (g["risk_level"] == "LOW").mean() >= 0.6:
        tags.append("Low-Risk")
    return " + ".join(tags) if tags else "Baseline"


profiles = []
for cid, g in df.groupby("cluster_id"):
    risk_mix = g["risk_level"].value_counts(normalize=True).reindex(RISK_ORDER, fill_value=0)
    profiles.append({
        "cluster_id": int(cid),
        "cluster_name": name_cluster(g, df),
        "n_identities": len(g),
        "pct_admin": round((g["privilege_level"] == "admin").mean() * 100, 1),
        "pct_mfa_enabled": round(g["mfa_enabled"].mean() * 100, 1),
        "pct_service": round(g["is_service"].mean() * 100, 1),
        "pct_contractor": round(g["is_contractor"].mean() * 100, 1),
        "median_inactive_days": float(g["inactive_days"].median()),
        "mean_login_count": round(g["login_count"].mean(), 1),
        "mean_n_systems": round(g["n_systems"].mean(), 2),
        "pct_true_anomaly": round(g["is_anomaly"].mean() * 100, 1),
        **{f"pct_{r.lower()}": round(risk_mix[r] * 100, 1) for r in RISK_ORDER},
    })
profiles_df = pd.DataFrame(profiles).sort_values("pct_true_anomaly", ascending=False)

# Guarantee unique names (two clusters can share the same tag set)
seen = {}
for i, row in profiles_df.iterrows():
    n = row["cluster_name"]
    seen[n] = seen.get(n, 0) + 1
    if seen[n] > 1:
        profiles_df.at[i, "cluster_name"] = f"{n} ({seen[n]})"
name_map = dict(zip(profiles_df["cluster_id"], profiles_df["cluster_name"]))
df["cluster_name"] = df["cluster_id"].map(name_map)

profiles_df.to_csv("reports/cluster_profiles.csv", index=False)
df[["identity_id", "cluster_id", "cluster_name", "distance_to_centroid"]].to_csv(
    "reports/identity_clusters.csv", index=False)

joblib.dump({"scaler": scaler, "kmeans": kmeans, "features": CLUSTER_FEATURES,
             "log_features": LOG_FEATURES, "cluster_names": name_map},
            "models/kmeans_clustering.joblib")

# ── Report ──
section("Cluster profiles (risk mix shown AFTER clustering - labels were not used)")
cols = ["cluster_id", "cluster_name", "n_identities", "pct_admin", "pct_mfa_enabled",
        "median_inactive_days", "pct_true_anomaly", "pct_critical", "pct_high", "pct_low"]
print(profiles_df[cols].to_string(index=False))

anomaly_spread = profiles_df["pct_true_anomaly"].max() - profiles_df["pct_true_anomaly"].min()
metrics = {
    "n_clusters": N_CLUSTERS,
    "silhouette_by_k": silhouettes,
    "silhouette_chosen_k": silhouettes[N_CLUSTERS],
    "best_silhouette_k": best_k,
    "anomaly_rate_spread_pct_points": round(float(anomaly_spread), 1),
    "clusters": profiles_df.to_dict(orient="records"),
}
write_json("reports/phase3_metrics.json", metrics)

lines = [
    "X-UBA - MCA MAJOR PROJECT - PHASE 3: K-MEANS BEHAVIOURAL CLUSTERING",
    "=" * 60, "",
    f"Identities clustered: {len(df)}   Features: {len(CLUSTER_FEATURES)} (no labels used)",
    f"k = {N_CLUSTERS} (fixed for interpretability); silhouette at k={N_CLUSTERS}: "
    f"{silhouettes[N_CLUSTERS]}",
    "Silhouette by k: " + ", ".join(f"k={k}:{v}" for k, v in silhouettes.items()),
    f"Highest silhouette: k={best_k}. Silhouette values in this range indicate "
    "overlapping (not sharply separated) behaviour groups, which is typical of IAM data.",
    "", "CLUSTER PROFILES", profiles_df[cols].to_string(index=False), "",
    f"The true-anomaly rate differs by {anomaly_spread:.1f} percentage points between the "
    "riskiest and safest cluster, although no label was used to form the clusters: "
    "unsupervised peer groups separate risky from normal behaviour.",
]
write_text("reports/phase3_clustering_summary.txt", "\n".join(lines))

print()
banner("[OK] PHASE 3 COMPLETE")
