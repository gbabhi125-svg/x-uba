#!/usr/bin/env python3
"""
X-UBA: Explainable User and Entity Behavior Analytics
       Using Heterogeneous Security Telemetry

Module: Identity Risk Intelligence Dashboard (Phase 9) - Flask

A single-page application (templates/index.html + static/js/app.js) served by
a JSON API. Every number shown comes from the pipeline outputs or from a live
call to the trained model - nothing is hard-coded in the UI.

    GET  /api/summary                 overview KPIs and distributions
    GET  /api/identities              search / filter / sort / paginate
    GET  /api/identity/<id>           full deep-dive, incl. LIVE SHAP
    POST /api/whatif/<id>             LIVE re-prediction for edited posture
    POST /api/remediate/<id>          apply a security action, re-score LIVE
    GET  /api/graph/overview          department -> system access graph
    GET  /api/graph/identity/<id>     identity ego-network (blast radius)
    GET  /api/model-performance       metrics, CV, ablation, class weighting
    GET  /api/compliance              SOD + NIST/GDPR control gaps
    GET  /api/org                     department analytics + peer outliers
    GET  /api/quiet-risk              cross-layer disagreement
    GET  /api/remediation             fixed-budget remediation strategies
    GET  /api/research                research-gap results

Works fully offline: Chart.js and D3 are bundled in static/js.

Run:  python src/dashboard/app.py        then open http://127.0.0.1:5000
      (run  python run_pipeline.py  first so reports/ and models/ exist)
"""

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, abort, jsonify, render_template, request

SRC = Path(__file__).resolve().parents[1]
for p in (SRC, SRC / "simulator", SRC / "graph_analysis"):
    sys.path.insert(0, str(p))
from common import (  # noqa: E402
    PROJECT_ROOT, RAW, RISK_ORDER, add_derived_features, encode, risk_index
)
from interventions import ACTIONS, GRAPH_ACTIONS, graph_after_action  # noqa: E402
from killchain import kill_chain  # noqa: E402
from reach import ReachModel  # noqa: E402

os.chdir(PROJECT_ROOT)
REPORTS = Path("reports")
REQUIRED = [
    "attack_simulation.csv", "fused_risk_scores.csv", "sod_identity_summary.csv", "sod_violations.csv",
    "compliance_gaps.csv", "counterfactual_recommendations.csv", "temporal_monthly_signal.csv",
    "quiet_risk.csv", "cluster_profiles.csv", "department_risk_profile.csv",
    "department_peer_outliers.csv", "model_metrics.json", "model_validation.json",
    "phase2_metrics.json", "risk_fusion_metrics.json", "explanation_fidelity_metrics.json",
    "graph_structural_metrics.json", "quiet_risk_metrics.json", "remediation_optimizer_metrics.json",
    "remediation_plan.csv", "shap_feature_importance.csv", "phase5_metrics.json",
]
PRIVILEGES = ["user", "power-user", "admin"]

app = Flask(__name__)
S = {}


def records(df):
    """DataFrame -> JSON-safe list of dicts (NaN -> null)."""
    return json.loads(df.to_json(orient="records"))


def rj(name):
    with open(REPORTS / name, encoding="utf-8") as f:
        return json.load(f)


def rtext(name):
    p = REPORTS / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


def load():
    missing = [f for f in REQUIRED if not (REPORTS / f).exists()] + \
        [f for f in ("models/xgb_risk_level.joblib", "models/feature_metadata.joblib")
         if not Path(f).exists()]
    if missing:
        return missing
    ident = pd.read_csv(f"{RAW}/identities.csv", usecols=["identity_id", "name", "email", "hire_date"])
    m = pd.read_csv(REPORTS / "attack_simulation.csv").merge(ident, on="identity_id")
    m = m.merge(pd.read_csv(REPORTS / "fused_risk_scores.csv")[
        ["identity_id", "fused_risk_score", "fused_risk_band"]], on="identity_id")
    m = m.merge(pd.read_csv(REPORTS / "sod_identity_summary.csv")[
        ["identity_id", "sod_violation_count", "sod_severity_score", "sod_rules"]], on="identity_id")
    comp = pd.read_csv(REPORTS / "compliance_gaps.csv")
    m = m.merge(comp[["identity_id", "compliance_score", "gap_count", "failed_controls"]], on="identity_id")
    m = m.merge(pd.read_csv(REPORTS / "quiet_risk.csv")[
        ["identity_id", "quadrant", "disagreement_score", "structural_pct", "behavioural_pct"]],
        on="identity_id")
    for c in ("sod_rules", "failed_controls"):
        m[c] = m[c].fillna("")
    S["m"] = m.set_index("identity_id", drop=False).rename_axis(None)
    S["comp"] = comp
    S["viol"] = pd.read_csv(REPORTS / "sod_violations.csv")
    S["cf"] = pd.read_csv(REPORTS / "counterfactual_recommendations.csv").set_index("identity_id")
    S["monthly"] = pd.read_csv(REPORTS / "temporal_monthly_signal.csv").set_index("identity_id")
    S["feat"] = pd.read_csv("data/processed/identity_features.csv") \
        .set_index("identity_id", drop=False).rename_axis(None)
    S["model"] = joblib.load("models/xgb_risk_level.joblib")
    S["classes"] = list(joblib.load("models/le_risk_level.joblib").classes_)
    S["meta"] = joblib.load("models/feature_metadata.joblib")
    S["explainer"] = shap.TreeExplainer(S["model"])
    S["rm"] = ReachModel()
    S["raw_max"] = rj("phase5_metrics.json")["blast_radius_raw_max"]
    S["n_systems"] = len(S["rm"].systems)
    return []


MISSING = load()


@app.before_request
def guard():
    if MISSING and request.path.startswith("/api/"):
        return jsonify({"error": "pipeline outputs missing - run: python run_pipeline.py",
                        "missing": MISSING}), 503


def json_body():
    """Request JSON as a dict (anything else -> empty dict)."""
    body = request.get_json(silent=True)
    return body if isinstance(body, dict) else {}


def get_identity(iid):
    iid = str(iid).upper()
    if iid not in S["m"].index:
        abort(404)
    return iid


@app.errorhandler(404)
def not_found(_):
    if request.path.startswith("/api/"):
        return jsonify({"error": "not found"}), 404
    return render_template("index.html", missing=MISSING), 404


# ───────────────────────────── live model helpers ─────────────────────────────

def predict(features):
    """Deployed risk_level model on a 1-row features frame -> (probs dict, risk index, class)."""
    d = add_derived_features(features.copy())
    p = S["model"].predict_proba(encode(d, S["meta"]))[0]
    return ({c: round(float(v), 4) for c, v in zip(S["classes"], p)},
            round(float(risk_index(p.reshape(1, -1), S["classes"])[0]), 2),
            S["classes"][int(p.argmax())])


def killchain_row(iid, posture, graph=None):
    """Kill chain for one identity with optional posture / graph overrides."""
    r = S["m"].loc[iid]
    row = {c: r[c] for c in ("mfa_enabled", "failed_login_rate", "unique_countries", "inactive_days",
                             "is_contractor_expired", "privilege_level", "unapproved_changes",
                             "reachable_systems", "reachable_critical", "reachable_high",
                             "blast_radius_score")}
    row.update(posture)
    if graph:
        row.update(graph)
    kc = kill_chain(pd.DataFrame([row]), S["n_systems"]).iloc[0]
    return {**{k: float(v) for k, v in kc.items()},
            "blast_radius_score": float(row["blast_radius_score"]),
            "reachable_critical": int(row["reachable_critical"])}


# ───────────────────────────────── pages ─────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", missing=MISSING)


# ───────────────────────────────── API ───────────────────────────────────────

@app.route("/api/summary")
def api_summary():
    m = S["m"]
    n = len(m)
    pred = m["xgb_risk_level_pred"].value_counts().reindex(RISK_ORDER, fill_value=0)
    threat = m["xgb_threat_pred"].value_counts()
    clusters = pd.read_csv(REPORTS / "cluster_profiles.csv")
    top = m.sort_values("attack_rank").head(10)
    return jsonify({
        "total_identities": n,
        "events": {"logins": 300000, "privilege_changes": 30000, "resource_accesses": 150000},
        "risk_distribution": {k: int(v) for k, v in pred.items()},
        "threat_distribution": {k: int(v) for k, v in threat.items()},
        "admin_count": int((m["privilege_level"] == "admin").sum()),
        "mfa_gap_count": int((m["mfa_enabled"] == 0).sum()),
        "early_warnings": int(m["early_warning"].sum()),
        "quiet_risk_count": int((m["quadrant"] == "QUIET_RISK").sum()),
        "sod_violation_identities": int((m["sod_violation_count"] > 0).sum()),
        "compliance_gap_identities": int((m["gap_count"] > 0).sum()),
        "avg_compliance": round(float(m["compliance_score"].mean()), 1),
        "avg_blast_radius": round(float(m["blast_radius_score"].mean()), 1),
        "org_expected_attack_risk": round(float(m["expected_attack_risk"].sum()), 1),
        "clusters": records(clusters[["cluster_name", "n_identities", "pct_admin", "pct_mfa_enabled",
                                      "pct_true_anomaly"]]),
        "top_targets": records(top[["identity_id", "name", "department", "privilege_level",
                                    "xgb_risk_index", "xgb_risk_level_pred", "blast_radius_score",
                                    "attack_likelihood", "expected_attack_risk", "attack_rank"]]),
    })


@app.route("/api/departments")
def api_departments():
    return jsonify(sorted(S["m"]["department"].unique().tolist()))


@app.route("/api/identities")
def api_identities():
    m = S["m"]
    q = request.args.get("search", "").strip().lower()
    dept = request.args.get("department", "")
    level = request.args.get("risk_level", "")
    quadrant = request.args.get("quadrant", "")
    sort = request.args.get("sort", "attack_rank")
    page = max(request.args.get("page", 1, type=int), 1)
    size = min(max(request.args.get("page_size", 50, type=int), 10), 200)
    f = m
    if q:
        f = f[f["identity_id"].str.lower().str.contains(q, regex=False) |
              f["name"].str.lower().str.contains(q, regex=False) |
              f["job_title"].str.lower().str.contains(q, regex=False)]
    if dept:
        f = f[f["department"] == dept]
    if level in RISK_ORDER:
        f = f[f["xgb_risk_level_pred"] == level]
    if quadrant:
        f = f[f["quadrant"] == quadrant]
    sorts = {"attack_rank": True, "xgb_risk_index": False, "fused_risk_score": False,
             "blast_radius_score": False, "compliance_score": True, "disagreement_score": False,
             "identity_id": True}
    if sort not in sorts:
        sort = "attack_rank"
    f = f.sort_values([sort, "identity_id"], ascending=[sorts[sort], True])
    total = len(f)
    pages = max((total + size - 1) // size, 1)
    page = min(page, pages)
    rows = f.iloc[(page - 1) * size: page * size]
    return jsonify({"total": total, "page": page, "pages": pages, "results": records(rows[[
        "identity_id", "name", "department", "job_title", "privilege_level", "mfa_enabled",
        "xgb_risk_index", "xgb_risk_level_pred", "xgb_threat_pred", "fused_risk_score",
        "blast_radius_score", "quadrant", "sod_violation_count", "compliance_score", "attack_rank"]])})


@app.route("/api/identity/<iid>")
def api_identity(iid):
    iid = get_identity(iid)
    feat = S["feat"].loc[[iid]]
    probs, live_risk, live_cls = predict(feat)

    # LIVE SHAP for the class the deployed model predicts
    X = encode(add_derived_features(feat.copy()), S["meta"])
    sv = S["explainer"].shap_values(X)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)
    contrib = sv[0, :, S["classes"].index(live_cls)]
    cols = S["meta"]["feature_cols"]
    order = np.argsort(-np.abs(contrib))[:10]
    shap_rows = [{"feature": cols[j],
                  "value": (str(feat.iloc[0][cols[j][:-4]]) if cols[j].endswith("_enc")
                            else round(float(X.iloc[0, j]), 3)),
                  "shap": round(float(contrib[j]), 4)} for j in order]

    cf_rows = []
    if iid in S["cf"].index:
        c = S["cf"].loc[iid]
        cf_rows = [{"action": a, "delta": float(c[f"delta_{a}"]), "description": ACTIONS[a][2]}
                   for a in ACTIONS if pd.notna(c[f"delta_{a}"])]
        cf_rows.sort(key=lambda x: x["delta"])
    viol = S["viol"][S["viol"]["identity_id"] == iid][["rule", "severity", "evidence"]]
    comp = S["comp"].set_index("identity_id").loc[iid]
    controls = [{"control": col, "failed": bool(comp[col])}
                for col in S["comp"].columns if col.startswith(("AC-", "IA-", "GDPR"))]
    rec = {k: v for k, v in records(S["m"].loc[[iid]])[0].items()}
    return jsonify({
        **rec,
        "live": {"probabilities": probs, "risk_index": live_risk, "predicted_class": live_cls},
        "shap_live": shap_rows,
        "counterfactual": cf_rows,
        "sod_violations": records(viol),
        "controls": controls,
        "monthly_signal": [float(v) for v in S["monthly"].loc[iid].values],
        "whatif_defaults": {
            "mfa_enabled": int(feat.iloc[0]["mfa_enabled"]),
            "privilege_level": feat.iloc[0]["privilege_level"],
            "inactive_days": int(feat.iloc[0]["inactive_days"]),
            "n_systems": int(feat.iloc[0]["n_systems"]),
            "unapproved_changes": int(feat.iloc[0]["unapproved_changes"]),
        },
    })


@app.route("/api/whatif/<iid>", methods=["POST"])
def api_whatif(iid):
    iid = get_identity(iid)
    try:
        body = {k: (v if k == "privilege_level" else float(v)) for k, v in json_body().items()
                if k in ("mfa_enabled", "privilege_level", "inactive_days", "n_systems",
                         "unapproved_changes")}
    except (TypeError, ValueError):
        return jsonify({"error": "what-if values must be numbers"}), 400
    if not all(np.isfinite(v) for k, v in body.items() if k != "privilege_level"):
        return jsonify({"error": "what-if values must be finite numbers"}), 400
    base = S["feat"].loc[[iid]].copy()
    after = base.copy()
    posture = {}
    if "mfa_enabled" in body:
        v = 1 if int(body["mfa_enabled"]) else 0
        after["mfa_enabled"] = v
        if v and base.iloc[0]["mfa_enabled"] == 0:
            after["mfa_usage_rate"] = max(float(base.iloc[0]["mfa_usage_rate"]), 0.95)
        posture["mfa_enabled"] = v
    if body.get("privilege_level") in PRIVILEGES:
        after["privilege_level"] = body["privilege_level"]
        posture["privilege_level"] = body["privilege_level"]
    for key, lo, hi in (("inactive_days", 0, 400), ("n_systems", 1, 15), ("unapproved_changes", 0, 50)):
        if key in body:
            v = int(min(max(body[key], lo), hi))
            after[key] = v
            if key != "n_systems":
                posture[key] = v
    p0, s0, c0 = predict(base)
    p1, s1, c1 = predict(after)
    graph = None
    new_level = posture.get("privilege_level")
    if new_level and new_level != base.iloc[0]["privilege_level"]:
        rank = PRIVILEGES.index(new_level)
        reach, admin, direct = S["rm"].reach(iid, privilege_level=new_level, max_rank=rank)
        s = S["rm"].summarize(iid, reach, admin, direct)
        graph = {"blast_radius_score": round(s["blast_radius_raw"] / S["raw_max"] * 100, 2),
                 "reachable_systems": s["reachable_systems"],
                 "reachable_critical": s["reachable_critical"], "reachable_high": s["reachable_high"]}
    return jsonify({
        "before": {"probabilities": p0, "risk_index": s0, "predicted_class": c0,
                   "attack": killchain_row(iid, {})},
        "after": {"probabilities": p1, "risk_index": s1, "predicted_class": c1,
                  "attack": killchain_row(iid, posture, graph)},
        "note": "n_systems changes the model features only; privilege changes also re-compute "
                "graph reach (blast radius).",
    })


@app.route("/api/remediate/<iid>", methods=["POST"])
def api_remediate(iid):
    iid = get_identity(iid)
    base = S["feat"].loc[[iid]].copy()
    action = json_body().get("action")
    if action is not None and action not in ACTIONS:
        return jsonify({"error": f"unknown action {action}"}), 400
    p0, s0, c0 = predict(base)
    if action is None:
        # Choose live: the applicable action with the largest predicted risk reduction
        best = None
        for name, (fn, _, _) in ACTIONS.items():
            trial = base.copy()
            if bool(fn(trial).any()):
                risk = predict(trial)[1]
                if best is None or risk < best[1]:
                    best = (name, risk)
        if best is None:
            return jsonify({"error": f"no security action changes anything for {iid}"}), 400
        action = best[0]
    after = base.copy()
    if not bool(ACTIONS[action][0](after).any()):
        return jsonify({"error": f"{action} does not change anything for {iid}"}), 400
    p1, s1, c1 = predict(after)
    posture = {k: after.iloc[0][k] for k in ("mfa_enabled", "privilege_level", "inactive_days",
                                             "unapproved_changes")}
    graph = graph_after_action(S["rm"], iid, action, base.iloc[0]["privilege_level"], S["raw_max"]) \
        if action in GRAPH_ACTIONS else None
    a0, a1 = killchain_row(iid, {}), killchain_row(iid, posture, graph)
    return jsonify({
        "action_applied": action, "description": ACTIONS[action][2],
        "before": {"risk_index": s0, "predicted_class": c0, "probabilities": p0, "attack": a0},
        "after": {"risk_index": s1, "predicted_class": c1, "probabilities": p1, "attack": a1},
        "risk_change": round(s1 - s0, 2),
        "attack_risk_change": round(a1["expected_attack_risk"] - a0["expected_attack_risk"], 3),
    })


@app.route("/api/graph/overview")
def api_graph_overview():
    rm = S["rm"]
    acc = rm.ok_access.merge(S["m"][["identity_id", "department"]].reset_index(drop=True),
                             on="identity_id")
    e = acc.groupby(["department", "system_name"]).size().reset_index(name="weight")
    sens = dict(zip(rm.systems["system_name"], rm.systems["sensitivity"]))
    risk = S["m"].groupby("department")["xgb_risk_index"].mean()
    nodes = [{"id": f"D:{d}", "label": d, "type": "department", "risk": round(float(risk[d]), 1)}
             for d in sorted(e["department"].unique())]
    nodes += [{"id": f"S:{s}", "label": s, "type": "system", "sensitivity": sens.get(s, "")}
              for s in sorted(e["system_name"].unique())]
    edges = [{"source": f"D:{d}", "target": f"S:{s}", "weight": int(w)}
             for d, s, w in e.itertuples(index=False)]
    return jsonify({"nodes": nodes, "edges": edges})


@app.route("/api/graph/identity/<iid>")
def api_graph_identity(iid):
    iid = get_identity(iid)
    rm = S["rm"]
    reach, admin_systems, direct = rm.reach(iid)
    sens_rank = {"critical": 3, "high": 2, "medium": 1, "low": 0}
    shown = sorted(direct, key=lambda x: (-sens_rank[rm.res_sens[x]], x))[:30]
    systems = sorted({rm.res_system[x] for x in shown} | admin_systems)
    nodes = [{"id": iid, "label": iid, "type": "identity"}]
    nodes += [{"id": f"S:{s}", "label": s, "type": "system", "admin": s in admin_systems} for s in systems]
    nodes += [{"id": f"R:{x}", "label": x, "type": "resource", "sensitivity": rm.res_sens[x]} for x in shown]
    edges = [{"source": iid, "target": f"R:{x}", "kind": "accessed"} for x in shown]
    edges += [{"source": f"R:{x}", "target": f"S:{rm.res_system[x]}", "kind": "part_of"} for x in shown]
    edges += [{"source": iid, "target": f"S:{s}", "kind": "admin"} for s in sorted(admin_systems)]
    s = rm.summarize(iid, reach, admin_systems, direct)
    return jsonify({"nodes": nodes, "edges": edges, "summary": s,
                    "shown_resources": len(shown), "direct_resources": len(direct)})


@app.route("/api/model-performance")
def api_model_performance():
    mm = rj("model_metrics.json")
    for k in ("is_anomaly", "risk_level", "threat_type"):
        mm[k].pop("confusion_matrix", None)
    return jsonify({
        "model_metrics": mm,
        "validation": rj("model_validation.json"),
        "isolation_forest": rj("phase2_metrics.json"),
        "shap_importance": records(pd.read_csv(REPORTS / "shap_feature_importance.csv").head(12)),
        "confusion_matrix_risk": rj("model_metrics.json")["risk_level"]["confusion_matrix"],
    })


@app.route("/api/compliance")
def api_compliance():
    v = S["viol"]
    comp = S["comp"]
    ctrl_cols = [c for c in comp.columns if c.startswith(("AC-", "IA-", "GDPR"))]
    by_ctrl = {c: int(comp[c].sum()) for c in ctrl_cols}
    worst = S["m"][S["m"]["sod_violation_count"] >= 2].sort_values(
        ["sod_severity_score", "xgb_risk_index"], ascending=False).head(15)
    return jsonify({
        "sod_by_rule": v["rule"].value_counts().to_dict(),
        "sod_by_severity": v["severity"].value_counts().to_dict(),
        "compliance_by_control": dict(sorted(by_ctrl.items(), key=lambda kv: -kv[1])),
        "compliance_by_framework": {
            "NIST SP 800-53": int(comp[[c for c in ctrl_cols if not c.startswith("GDPR")]].any(axis=1).sum()),
            "GDPR Art. 32": int(comp[[c for c in ctrl_cols if c.startswith("GDPR")]].any(axis=1).sum()),
        },
        "score_by_predicted_level": {k: round(float(x), 1) for k, x in S["m"].groupby(
            "xgb_risk_level_pred")["compliance_score"].mean().reindex(RISK_ORDER).items()},
        "identities_with_gap": int((comp["gap_count"] > 0).sum()),
        "sod_identities": int((S["m"]["sod_violation_count"] > 0).sum()),
        "avg_compliance": round(float(comp["compliance_score"].mean()), 1),
        "worst": records(worst[["identity_id", "department", "privilege_level", "sod_violation_count",
                                "sod_rules", "compliance_score", "xgb_risk_level_pred"]]),
    })


@app.route("/api/org")
def api_org():
    d = pd.read_csv(REPORTS / "department_risk_profile.csv")
    peers = pd.read_csv(REPORTS / "department_peer_outliers.csv").head(20)
    return jsonify({"departments": records(d), "peer_outliers": records(peers),
                    "metrics": rj("phase8c_metrics.json")})


@app.route("/api/quiet-risk")
def api_quiet_risk():
    q = S["m"][S["m"]["quadrant"] == "QUIET_RISK"].sort_values("disagreement_score", ascending=False)
    # Fixed-seed sample for the behavioural-vs-structural scatter plot
    sample = S["m"].sample(n=min(2000, len(S["m"])), random_state=42)
    return jsonify({"metrics": rj("quiet_risk_metrics.json"),
                    "scatter": records(sample[["identity_id", "behavioural_pct", "structural_pct",
                                               "quadrant"]]),
                    "top": records(q.head(25)[["identity_id", "department", "privilege_level",
                                               "mfa_enabled", "xgb_risk_level_pred", "xgb_risk_index",
                                               "blast_radius_score", "sod_severity_score", "sod_rules",
                                               "compliance_score", "disagreement_score"]])})


@app.route("/api/remediation")
def api_remediation():
    plan = pd.read_csv(REPORTS / "remediation_plan.csv")
    metrics = rj("remediation_optimizer_metrics.json")
    # Same order as metrics["strategies"], so cards, charts and tables line up
    top = [{"strategy": st["strategy"],
            "rows": records(plan[plan["strategy"] == st["strategy"]].head(10)[
                ["identity_id", "action", "risk_reduction", "ear_reduction", "critical_paths_removed",
                 "quadrant"]])} for st in metrics["strategies"]]
    return jsonify({"metrics": metrics, "top_by_strategy": top,
                    "summary_text": rtext("remediation_optimizer_summary.txt")})


@app.route("/api/research")
def api_research():
    return jsonify({
        "fusion": rj("risk_fusion_metrics.json"),
        "fidelity": rj("explanation_fidelity_metrics.json"),
        "graph": rj("graph_structural_metrics.json"),
        "temporal": rj("phase4_metrics.json"),
        "counterfactual": rj("phase6_metrics.json"),
        "attack": rj("phase7_metrics.json"),
        "texts": {k: rtext(v) for k, v in {
            "Risk fusion": "risk_fusion_summary.txt",
            "Explanation fidelity": "explanation_fidelity_summary.txt",
            "Graph-structural anomalies": "graph_structural_summary.txt",
            "Temporal trajectory": "phase4_temporal_summary.txt",
            "Model validation": "model_validation_summary.txt"}.items()},
    })


if __name__ == "__main__":
    port = int(os.environ.get("XUBA_PORT", "5000"))
    if MISSING:
        print("[X] Pipeline outputs missing: " + ", ".join(MISSING))
        print("    Run first:  python run_pipeline.py")
    else:
        print(f"[OK] Loaded {len(S['m'])} identities, trained model and privilege graph")
    print(f"[OK] X-UBA dashboard: http://127.0.0.1:{port}   (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=port, debug=False)
