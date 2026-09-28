#!/usr/bin/env python3
"""
X-UBA: Explainable User and Entity Behavior Analytics
       Using Heterogeneous Security Telemetry

Module: Privilege Graph + Blast Radius (Phase 5) - NetworkX

Builds one directed graph of the whole IAM estate:

    identity --accessed--> resource --part_of--> system
    identity --holds privilege on--> system  (from granted/modified privileges)

and answers the question "if THIS identity were compromised, how much could
an attacker reach?" - its blast radius.

Reach rules (documented, deterministic):
  * direct reach   : every resource the identity has actually accessed
  * escalated reach: on a system where the identity holds admin rights
                     (admin privilege level, or a granted "admin" privilege),
                     an attacker can reach EVERY resource of that system
  * power-user     : on systems where it has power-user rights, all resources
                     up to "high" sensitivity (not "critical")

Blast radius score = sum of sensitivity weights of reachable resources
(low 1, medium 2, high 4, critical 8), scaled linearly to 0-100. The attack path
is the shortest path from the identity to its most sensitive reachable
resource.

Run from ANYWHERE - this script anchors itself to the project root.
Requires: data/raw/*.csv
Output: models/privilege_graph.pkl
        reports/blast_radius.csv
        reports/phase5_graph_summary.txt
        reports/phase5_metrics.json
"""

import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    RAW, RISK_ORDER, init, banner, section, write_text, write_json, require
)
from reach import ReachModel, SENS_WEIGHT, PRIV_RANK  # noqa: E402

init()
banner("X-UBA | MCA MAJOR PROJECT | PHASE 5: Privilege Graph + Blast Radius")

require(f"{RAW}/identities.csv", f"{RAW}/resources.csv", f"{RAW}/systems.csv",
        f"{RAW}/resource_access.csv", f"{RAW}/privilege_changes.csv")

rm = ReachModel()
identities, resources, systems = rm.identities, rm.resources, rm.systems

# ── Build the graph ──
G = nx.DiGraph()
for r in systems.itertuples(index=False):
    G.add_node(f"SYS:{r.system_name}", kind="system", sensitivity=r.sensitivity)
for r in resources.itertuples(index=False):
    G.add_node(f"RES:{r.resource_id}", kind="resource", sensitivity=r.sensitivity,
               name=r.resource_name, weight_value=SENS_WEIGHT[r.sensitivity])
    G.add_edge(f"RES:{r.resource_id}", f"SYS:{r.system_name}", relation="part_of")
for r in identities.itertuples(index=False):
    G.add_node(f"ID:{r.identity_id}", kind="identity", privilege_level=r.privilege_level)
G.add_edges_from(
    (f"ID:{i}", f"RES:{res}", {"relation": "accessed", "count": int(n)})
    for i, res, n in rm.acc_counts.itertuples(index=False)
)
inv_rank = {v: k for k, v in PRIV_RANK.items()}
G.add_edges_from(
    (f"ID:{i}", f"SYS:{s}", {"relation": "privilege", "level": inv_rank[k]})
    for i, s, k in rm.held.itertuples(index=False)
)

print(f"\n[OK] Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── Reach computation (rules in src/graph_analysis/reach.py) ──
rows = []
for r in identities.itertuples(index=False):
    reach, admin_systems, direct = rm.reach(r.identity_id)
    rows.append({"identity_id": r.identity_id, "privilege_level": r.privilege_level,
                 **rm.summarize(r.identity_id, reach, admin_systems, direct)})

blast = pd.DataFrame(rows)
# Linear scale against the widest reach in the estate (100 = can reach everything
# the most-privileged identity can). A log scale was tried and compressed scores.
blast["blast_radius_score"] = (blast["blast_radius_raw"] / blast["blast_radius_raw"].max() * 100).round(2)
blast = blast.merge(identities[["identity_id", "threat_score", "risk_level", "is_anomaly"]],
                    on="identity_id", how="left")
blast.to_csv("reports/blast_radius.csv", index=False)

with open("models/privilege_graph.pkl", "wb") as f:
    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

# ── Validation ──
by_level = blast.groupby("risk_level")["blast_radius_score"].mean().reindex(RISK_ORDER).round(2)
corr = blast["blast_radius_score"].corr(blast["threat_score"])
corr_spearman = blast["blast_radius_score"].corr(blast["threat_score"], method="spearman")
monotonic = bool(np.all(np.diff(by_level.values) > 0))

section("Validation: does blast radius rise with true risk?")
print(by_level.to_string())
print(f"Monotonic increase LOW -> CRITICAL: {monotonic}")
print(f"Pearson r with ground-truth threat_score:  {corr:.4f}")
print(f"Spearman rho with ground-truth threat_score: {corr_spearman:.4f}")
top = blast.nlargest(5, "blast_radius_score")[
    ["identity_id", "privilege_level", "reachable_resources", "reachable_critical",
     "blast_radius_score", "risk_level"]]
print("\nTop 5 identities by blast radius:")
print(top.to_string(index=False))

metrics = {
    "graph_nodes": G.number_of_nodes(), "graph_edges": G.number_of_edges(),
    "mean_blast_by_risk_level": by_level.to_dict(),
    "monotonic_by_risk_level": monotonic,
    "pearson_r_threat_score": round(float(corr), 4),
    "spearman_rho_threat_score": round(float(corr_spearman), 4),
    "mean_reachable_resources": round(float(blast["reachable_resources"].mean()), 2),
    "identities_reaching_critical": int((blast["reachable_critical"] > 0).sum()),
    "blast_radius_raw_max": int(blast["blast_radius_raw"].max()),
}
write_json("reports/phase5_metrics.json", metrics)

lines = [
    "X-UBA - MCA MAJOR PROJECT - PHASE 5: PRIVILEGE GRAPH + BLAST RADIUS",
    "=" * 60, "",
    f"Graph: {G.number_of_nodes()} nodes (identities, resources, systems), "
    f"{G.number_of_edges()} edges (access, part-of, privilege)",
    f"Identities that can reach at least one CRITICAL resource: "
    f"{metrics['identities_reaching_critical']}",
    f"Average reachable resources per identity: {metrics['mean_reachable_resources']}",
    "", "Average blast radius score by ground-truth risk level:", by_level.to_string(), "",
    f"Monotonic LOW -> CRITICAL: {monotonic}",
    f"Pearson r = {corr:.4f}, Spearman rho = {corr_spearman:.4f} with ground-truth threat score",
    "",
    "NOTE: the blast radius is computed ONLY from the graph (access + privilege edges); "
    "the risk labels are used afterwards to validate it. Part of the correlation comes "
    "from the generator giving riskier identities more events (hence more edges).",
    "", "Top 5 identities by blast radius:", top.to_string(index=False),
]
write_text("reports/phase5_graph_summary.txt", "\n".join(lines))

print()
banner("[OK] PHASE 5 COMPLETE")
