#!/usr/bin/env python3
"""
X-UBA: Explainable User and Entity Behavior Analytics
       Using Heterogeneous Security Telemetry

Module: Security Dashboard (Phase 9) - Flask

A read-only web UI over the pipeline's outputs (reports/*.csv, *.json):

    /                      overview: KPIs, risk mix, model metrics, top targets
    /identities            searchable, filterable identity list
    /identity/<id>         full WHY / WHO / WHEN / WHAT / IMPACT / ACTION report
    /research              the three research-gap results + per-phase metrics
    /api/identity/<id>     the same identity report as JSON

Works fully offline (no CDN, no JavaScript framework).

Run:  python src/dashboard/app.py        then open http://127.0.0.1:5000
      (run  python run_pipeline.py  first so reports/ exists)
"""

import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, abort, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PROJECT_ROOT, RISK_ORDER  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
REQUIRED = ["attack_simulation.csv", "fused_risk_scores.csv", "sod_identity_summary.csv",
            "sod_violations.csv", "compliance_gaps.csv", "counterfactual_recommendations.csv",
            "temporal_monthly_signal.csv", "cluster_profiles.csv",
            "department_risk_profile.csv", "model_metrics.json"]
PAGE_SIZE = 50

app = Flask(__name__)


def _read_json(name):
    p = REPORTS / name
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_data():
    missing = [f for f in REQUIRED if not (REPORTS / f).exists()]
    if missing:
        return None, missing
    a = pd.read_csv(REPORTS / "attack_simulation.csv")
    a = a.merge(pd.read_csv(REPORTS / "fused_risk_scores.csv")[
        ["identity_id", "fused_risk_score", "fused_risk_band"]], on="identity_id", how="left")
    a = a.merge(pd.read_csv(REPORTS / "sod_identity_summary.csv")[
        ["identity_id", "sod_violation_count", "sod_rules"]], on="identity_id", how="left")
    a = a.merge(pd.read_csv(REPORTS / "compliance_gaps.csv")[
        ["identity_id", "compliance_score", "failed_controls"]], on="identity_id", how="left")
    a["sod_rules"] = a["sod_rules"].fillna("")
    a["failed_controls"] = a["failed_controls"].fillna("")
    d = {
        "ids": a.set_index("identity_id", drop=False),
        "violations": pd.read_csv(REPORTS / "sod_violations.csv"),
        "cf": pd.read_csv(REPORTS / "counterfactual_recommendations.csv").set_index("identity_id"),
        "monthly": pd.read_csv(REPORTS / "temporal_monthly_signal.csv").set_index("identity_id"),
        "clusters": pd.read_csv(REPORTS / "cluster_profiles.csv"),
        "departments": pd.read_csv(REPORTS / "department_risk_profile.csv"),
        "shap": pd.read_csv(REPORTS / "shap_feature_importance.csv")
        if (REPORTS / "shap_feature_importance.csv").exists() else pd.DataFrame(),
        "m": {k: _read_json(f) for k, f in {
            "model": "model_metrics.json", "p2": "phase2_metrics.json", "p3": "phase3_metrics.json",
            "p4": "phase4_metrics.json", "p5": "phase5_metrics.json", "p6": "phase6_metrics.json",
            "p7": "phase7_metrics.json", "p8a": "phase8a_metrics.json",
            "p8b": "phase8b_metrics.json", "p8c": "phase8c_metrics.json",
            "g1": "risk_fusion_metrics.json", "g2": "explanation_fidelity_metrics.json",
            "g3": "graph_structural_metrics.json"}.items()},
    }
    return d, []


DATA, MISSING = load_data()


def clean(v):
    """Make pandas / numpy values safe for templates and JSON."""
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "item"):
        v = v.item()
        if isinstance(v, float) and math.isnan(v):
            return None
    return v


@app.before_request
def need_data():
    if DATA is None and request.endpoint != "static":
        return render_template("error.html", missing=MISSING), 503


@app.template_filter("band")
def band_class(level):
    return {"CRITICAL": "b-crit", "HIGH": "b-high", "MEDIUM": "b-med", "LOW": "b-low"}.get(
        str(level), "b-none")


@app.template_filter("pct")
def pct(v, digits=1):
    return "-" if v is None else f"{float(v) * 100:.{digits}f}%"


@app.template_filter("num")
def num(v, digits=1):
    return "-" if v is None else f"{float(v):,.{digits}f}"


@app.route("/")
def overview():
    ids, m = DATA["ids"], DATA["m"]
    level_counts = ids["xgb_risk_level_pred"].value_counts().reindex(RISK_ORDER, fill_value=0)
    kpis = {
        "identities": len(ids),
        "critical": int(level_counts["CRITICAL"]),
        "high": int(level_counts["HIGH"]),
        "early_warnings": int(ids["early_warning"].sum()),
        "sod_flagged": int((ids["sod_violation_count"].fillna(0) > 0).sum()),
        "avg_compliance": float(ids["compliance_score"].mean()),
    }
    top = ids.sort_values("attack_rank").head(15).to_dict(orient="records")
    models = []
    for key, label in (("is_anomaly", "Anomaly (binary)"), ("risk_level", "Risk level (4-class)"),
                       ("threat_type", "Threat type (6-class)")):
        r = m["model"].get(key, {})
        models.append({"name": label, "accuracy": r.get("accuracy"), "f1": r.get("f1_macro"),
                       "precision": r.get("precision_macro"), "recall": r.get("recall_macro")})
    return render_template(
        "overview.html", kpis=kpis, levels=level_counts.to_dict(), top=top, models=models,
        clusters=DATA["clusters"].to_dict(orient="records"),
        departments=DATA["departments"].to_dict(orient="records"),
        shap=DATA["shap"].head(10).to_dict(orient="records"), m=m)


@app.route("/identities")
def identities():
    df = DATA["ids"]
    q = request.args.get("q", "").strip().upper()
    level = request.args.get("level", "")
    dept = request.args.get("dept", "")
    sort = request.args.get("sort", "attack_rank")
    if q:
        df = df[df["identity_id"].str.contains(q, regex=False) |
                df["job_title"].str.upper().str.contains(q, regex=False)]
    if level in RISK_ORDER:
        df = df[df["xgb_risk_level_pred"] == level]
    if dept:
        df = df[df["department"] == dept]
    sorts = {"attack_rank": ("attack_rank", True), "risk": ("xgb_risk_index", False),
             "blast": ("blast_radius_score", False), "compliance": ("compliance_score", True),
             "id": ("identity_id", True)}
    col, asc = sorts.get(sort, sorts["attack_rank"])
    df = df.sort_values(col, ascending=asc)
    total = len(df)
    pages = max(1, math.ceil(total / PAGE_SIZE))
    page = min(max(request.args.get("page", 1, type=int), 1), pages)
    rows = df.iloc[(page - 1) * PAGE_SIZE: page * PAGE_SIZE].to_dict(orient="records")
    return render_template("identities.html", rows=rows, total=total, page=page, pages=pages,
                           q=request.args.get("q", ""), level=level, dept=dept, sort=sort,
                           levels=RISK_ORDER,
                           depts=sorted(DATA["ids"]["department"].unique()))


def identity_payload(identity_id):
    identity_id = identity_id.upper()
    if identity_id not in DATA["ids"].index:
        abort(404)
    r = {k: clean(v) for k, v in DATA["ids"].loc[identity_id].to_dict().items()}
    factors = []
    for k in range(1, 6):
        f = r.get(f"factor_{k}")
        if f:
            factors.append({"feature": f, "value": r.get(f"factor_{k}_value"),
                            "shap": r.get(f"factor_{k}_shap") or 0.0})
    max_shap = max([abs(f["shap"]) for f in factors] + [1e-9])
    for f in factors:
        f["width"] = round(abs(f["shap"]) / max_shap * 100, 1)
    actions = []
    if identity_id in DATA["cf"].index:
        c = DATA["cf"].loc[identity_id]
        for col in [x for x in c.index if x.startswith("delta_")]:
            if pd.notna(c[col]):
                actions.append({"action": col.replace("delta_", ""), "delta": float(c[col])})
        actions.sort(key=lambda a: a["delta"])
    monthly = [float(x) for x in DATA["monthly"].loc[identity_id].values] \
        if identity_id in DATA["monthly"].index else []
    viol = DATA["violations"]
    violations = viol[viol["identity_id"] == identity_id].to_dict(orient="records")
    return r, factors, actions, monthly, violations


@app.route("/identity/<identity_id>")
def identity(identity_id):
    r, factors, actions, monthly, violations = identity_payload(identity_id)
    peak = max(monthly + [1e-9])
    spark = " ".join(f"{i * 30 + 15},{95 - v / peak * 80:.1f}" for i, v in enumerate(monthly))
    stages = [("Initial access", r["p_initial_access"]),
              ("Privilege escalation", r["p_privilege_escalation"]),
              ("Lateral movement", r["p_lateral_movement"]), ("Data impact", r["p_data_impact"])]
    return render_template("identity.html", r=r, factors=factors, actions=actions,
                           monthly=monthly, spark=spark, stages=stages, violations=violations,
                           controls=[c for c in r["failed_controls"].split("; ") if c])


@app.route("/api/identity/<identity_id>")
def identity_api(identity_id):
    r, factors, actions, monthly, violations = identity_payload(identity_id)
    return jsonify({"identity": r, "shap_factors": factors, "counterfactual_actions": actions,
                    "monthly_signal": monthly,
                    "sod_violations": [{k: clean(v) for k, v in x.items()} for x in violations]})


@app.route("/research")
def research():
    return render_template("research.html", m=DATA["m"])


@app.errorhandler(404)
def not_found(_):
    return render_template("error.html", missing=None), 404


if __name__ == "__main__":
    port = int(os.environ.get("XUBA_PORT", "5000"))
    if DATA is None:
        print("[X] Pipeline outputs missing: " + ", ".join(MISSING))
        print("    Run first:  python run_pipeline.py")
    else:
        print(f"[OK] Loaded {len(DATA['ids'])} identities from reports/")
    print(f"[OK] X-UBA dashboard: http://127.0.0.1:{port}   (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=port, debug=False)
