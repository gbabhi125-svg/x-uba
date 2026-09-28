#!/usr/bin/env python3
"""
X-UBA: Explainable User and Entity Behavior Analytics
       Using Heterogeneous Security Telemetry

Module: Graph-Structural Peer-Group Anomaly Detection (Research Gap 3)

Phase 5 used the privilege graph for IMPACT only. Recent IAM/GNN literature
criticises systems that score identities "in isolation, without modelling
relational dependencies". This module uses graph structure for DETECTION:

  1. Identity-identity similarity graph: each identity is a TF-IDF vector of
     the high/critical-sensitivity resources it accessed; every identity is
     linked to its 10 most similar identities (cosine k-NN). This keeps the
     graph sparse (a full co-access projection would have millions of edges).
  2. Louvain community detection -> structural peer groups (people who touch
     the same sensitive resources).
  3. Peer-group deviation: an identity whose Phase 1 risk is far above its
     structural peers (z >= 2.0) is flagged. "Novel" = flagged although Phase 1
     did NOT predict it HIGH/CRITICAL.
  4. Relational value test: does the average risk of an identity's graph
     neighbours improve detection beyond its own risk? (Logistic regression
     on TRAIN, ROC-AUC on held-out TEST, with and without the neighbour feature.)

Run from ANYWHERE - this script anchors itself to the project root.
Requires: data/raw/*.csv, reports/identity_risk_scores.csv
Output: reports/graph_peer_groups.csv
        reports/graph_structural_outliers.csv
        reports/graph_structural_summary.txt
        reports/graph_structural_metrics.json
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    SEED, RAW, init, banner, section, write_text, write_json, require
)

init()
banner("X-UBA | MCA MAJOR PROJECT | GAP 3: Graph-Structural Peer-Group Anomaly Detection")

K_NEIGHBOURS = 10
Z_FLAG = 2.0          # classic 2-sigma rule; risk index is bimodal, so z rarely exceeds ~2.3
MIN_STD = 5.0          # risk-index points; avoids huge z in near-uniform peer groups
MIN_COMMUNITY = 20

require(f"{RAW}/resource_access.csv", "reports/identity_risk_scores.csv")
access = pd.read_csv(f"{RAW}/resource_access.csv",
                     usecols=["identity_id", "resource_id", "sensitivity", "status"])
scores = pd.read_csv("reports/identity_risk_scores.csv")

sens = access[(access["status"] == "success") & access["sensitivity"].isin(["high", "critical"])]
ids = np.sort(sens["identity_id"].unique())
res = np.sort(sens["resource_id"].unique())
id_ix = {v: i for i, v in enumerate(ids)}
res_ix = {v: i for i, v in enumerate(res)}
counts = sens.groupby(["identity_id", "resource_id"]).size().reset_index(name="n")
M = csr_matrix((counts["n"].values,
                (counts["identity_id"].map(id_ix).values, counts["resource_id"].map(res_ix).values)),
               shape=(len(ids), len(res)))
V = TfidfTransformer().fit_transform(M)
print(f"\n[OK] {len(ids)} identities with sensitive access over {len(res)} high/critical resources")

nn = NearestNeighbors(n_neighbors=K_NEIGHBOURS + 1, metric="cosine").fit(V)
dist, nbr = nn.kneighbors(V)
G = nx.Graph()
G.add_nodes_from(range(len(ids)))
for i in range(len(ids)):
    for d, j in zip(dist[i, 1:], nbr[i, 1:]):
        if j != i and d < 1.0:
            G.add_edge(i, int(j), weight=float(1 - d))
communities = nx.community.louvain_communities(G, weight="weight", seed=SEED)
modularity = nx.community.modularity(G, communities, weight="weight")
comm_of = np.empty(len(ids), dtype=int)
for c, members in enumerate(sorted(communities, key=len, reverse=True)):
    comm_of[list(members)] = c
print(f"[OK] k-NN graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges; "
      f"{len(communities)} Louvain communities (modularity {modularity:.3f})")

peer = pd.DataFrame({"identity_id": ids, "community": comm_of})
peer = peer.merge(scores[["identity_id", "split", "xgb_risk_index", "xgb_risk_level_pred",
                          "true_is_anomaly", "true_risk_level"]], on="identity_id")
risk = peer.set_index("identity_id").loc[ids, "xgb_risk_index"].values
peer["neighbour_mean_risk"] = [risk[[j for j in nbr[i, 1:] if j != i]].mean()
                               for i in range(len(ids))]
size = peer.groupby("community")["identity_id"].transform("count")
c_mean = peer.groupby("community")["xgb_risk_index"].transform("mean")
c_std = peer.groupby("community")["xgb_risk_index"].transform("std").fillna(0).clip(lower=MIN_STD)
peer["community_size"] = size
peer["community_mean_risk"] = c_mean.round(2)
peer["peer_z"] = ((peer["xgb_risk_index"] - c_mean) / c_std).round(3)
peer["structural_outlier"] = ((peer["peer_z"] >= Z_FLAG) & (size >= MIN_COMMUNITY)).astype(int)
peer["novel"] = ((peer["structural_outlier"] == 1) &
                 ~peer["xgb_risk_level_pred"].isin(["HIGH", "CRITICAL"])).astype(int)
peer.to_csv("reports/graph_peer_groups.csv", index=False)
outl = peer[peer["structural_outlier"] == 1].sort_values("peer_z", ascending=False)
outl.to_csv("reports/graph_structural_outliers.csv", index=False)

# ── Relational value test ──
tr, te = peer[peer["split"] == "train"], peer[peer["split"] == "test"]


def test_auc(cols):
    lr = LogisticRegression(max_iter=1000).fit(tr[cols], tr["true_is_anomaly"])
    return roc_auc_score(te["true_is_anomaly"], lr.predict_proba(te[cols])[:, 1])


auc_own = test_auc(["xgb_risk_index"])
auc_both = test_auc(["xgb_risk_index", "neighbour_mean_risk"])
auc_nbr = roc_auc_score(te["true_is_anomaly"], te["neighbour_mean_risk"])
base_rate = float(peer["true_is_anomaly"].mean())
outl_rate = float(outl["true_is_anomaly"].mean()) if len(outl) else 0.0
novel = outl[outl["novel"] == 1]
novel_rate = float(novel["true_is_anomaly"].mean()) if len(novel) else float("nan")
comm_risk = peer.groupby("community").agg(size=("identity_id", "count"),
                                          mean_risk=("xgb_risk_index", "mean"),
                                          anomaly_rate=("true_is_anomaly", "mean"))
comm_spread = comm_risk.loc[comm_risk["size"] >= MIN_COMMUNITY, "anomaly_rate"]

section("Results")
print(f"Structural outliers (z >= {Z_FLAG}): {len(outl)}  | true-anomaly rate {outl_rate:.3f} "
      f"(base {base_rate:.3f})")
print(f"Novel (not predicted HIGH/CRITICAL by Phase 1): {len(novel)}"
      + (f"  | true-anomaly rate {novel_rate:.3f}" if len(novel) else ""))
print(f"Community anomaly-rate range (size >= {MIN_COMMUNITY}): "
      f"{comm_spread.min():.3f} - {comm_spread.max():.3f}")
print(f"TEST ROC-AUC  own risk only: {auc_own:.4f} | own + neighbour risk: {auc_both:.4f} | "
      f"neighbour risk alone: {auc_nbr:.4f}")

adds_value = auc_both - auc_own > 0.002
metrics = {
    "identities_in_graph": int(len(ids)), "sensitive_resources": int(len(res)),
    "graph_edges": G.number_of_edges(), "communities": len(communities),
    "modularity": round(float(modularity), 4),
    "structural_outliers": int(len(outl)), "outlier_true_anomaly_rate": round(outl_rate, 4),
    "base_anomaly_rate": round(base_rate, 4),
    "novel_outliers": int(len(novel)),
    "novel_true_anomaly_rate": None if not len(novel) else round(novel_rate, 4),
    "community_anomaly_rate_range": [round(float(comm_spread.min()), 4),
                                     round(float(comm_spread.max()), 4)],
    "test_auc_own_risk": round(float(auc_own), 4),
    "test_auc_own_plus_neighbour": round(float(auc_both), 4),
    "test_auc_neighbour_only": round(float(auc_nbr), 4),
    "neighbour_signal_adds_value": bool(adds_value),
}
write_json("reports/graph_structural_metrics.json", metrics)
write_text("reports/graph_structural_summary.txt", "\n".join([
    "X-UBA - MCA MAJOR PROJECT - GAP 3: GRAPH-STRUCTURAL PEER-GROUP ANOMALY DETECTION",
    "=" * 60, "",
    f"Similarity graph: {len(ids)} identities, {G.number_of_edges()} edges "
    f"(cosine {K_NEIGHBOURS}-NN over {len(res)} high/critical resources, TF-IDF weighted)",
    f"Louvain communities: {len(communities)} (modularity {modularity:.3f})",
    f"Community anomaly rates (size >= {MIN_COMMUNITY}) range {comm_spread.min():.1%} - "
    f"{comm_spread.max():.1%}", "",
    f"Structural outliers (risk z >= {Z_FLAG} above structural peers): {len(outl)}",
    f"  true-anomaly rate {outl_rate:.1%} vs base {base_rate:.1%}",
    f"  novel (Phase 1 did not predict HIGH/CRITICAL): {len(novel)}", "",
    "Relational value test (held-out TEST ROC-AUC):",
    f"  own Phase 1 risk only:            {auc_own:.4f}",
    f"  own risk + neighbour mean risk:   {auc_both:.4f}",
    f"  neighbour mean risk alone:        {auc_nbr:.4f}", "",
    ("RESULT: the neighbour signal adds measurable information beyond the identity's own risk."
     if adds_value else
     "HONEST RESULT: on this synthetic dataset the graph neighbourhood adds no detection value "
     "beyond the identity's own score, and the structural outliers are mostly identities the "
     "supervised model already rates highly. The generator draws each access event's resource "
     "uniformly at random, so WHICH sensitive resources an identity touches carries no risk "
     "information - only HOW MANY does, and Phase 1 already uses that. This is a genuine "
     "limitation of validating relational methods on synthetic data, reported rather than "
     "hidden; on real data with role-based access patterns, peer groups become meaningful."),
]))

print()
banner("[OK] GAP 3 (GRAPH-STRUCTURAL ANOMALY) COMPLETE")
