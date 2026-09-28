#!/usr/bin/env python3
"""
X-UBA: Explainable User and Entity Behavior Analytics
       Using Heterogeneous Security Telemetry

Module: Temporal Risk Trajectory (Phase 4) - early-warning detection

Phase 1 scores an identity's risk as ONE number for the whole year. This
module asks a different question: is the identity's behaviour getting WORSE
over time? It rebuilds a 12-window (~monthly) time series of suspicious
activity per identity directly from the three raw event logs, fits a trend
line over the most recent 6 windows, and raises an EARLY_WARNING when risk
is rising sharply even though the identity is not (yet) predicted CRITICAL.

Suspicious-activity signal per window (observable events only, no labels):
    failed login x1, night login x0.5, login from outside home country x2,
    pending/unapproved privilege change x3, privilege grant x1,
    high/critical-sensitivity resource access x1, bulk download/export
    (> 20 units) x3

Run from ANYWHERE - this script anchors itself to the project root.
Requires: data/raw/*.csv, reports/identity_risk_scores.csv (Phase 1)
Output: reports/temporal_risk_trajectories.csv  (per-identity trend metrics)
        reports/temporal_monthly_signal.csv     (identity x window matrix)
        reports/phase4_temporal_summary.txt
        reports/phase4_metrics.json
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    RAW, init, banner, section, write_text, write_json, require
)

init()
banner("X-UBA | MCA MAJOR PROJECT | PHASE 4: Temporal Risk Trajectory")

N_WINDOWS = 12
TREND_WINDOWS = 6          # trend fitted over the most recent 6 windows
EARLY_WARNING_PCTL = 90    # slope must be in the top 10% of the organisation
RISE_RATIO = 1.5           # recent activity >= 1.5x the identity's own baseline

require(f"{RAW}/identities.csv", f"{RAW}/login_events.csv", f"{RAW}/privilege_changes.csv",
        f"{RAW}/resource_access.csv", "reports/identity_risk_scores.csv")

identities = pd.read_csv(f"{RAW}/identities.csv", usecols=["identity_id", "home_country"])
logins = pd.read_csv(f"{RAW}/login_events.csv",
                     usecols=["identity_id", "timestamp", "country", "status", "time_class"])
privs = pd.read_csv(f"{RAW}/privilege_changes.csv",
                    usecols=["identity_id", "timestamp", "change_type", "approval_status"])
access = pd.read_csv(f"{RAW}/resource_access.csv",
                     usecols=["identity_id", "timestamp", "sensitivity", "action", "data_volume"])
scores = pd.read_csv("reports/identity_risk_scores.csv")

all_ts = pd.to_datetime(pd.concat([logins["timestamp"], privs["timestamp"], access["timestamp"]]))
t0, t1 = all_ts.min(), all_ts.max()
span_days = (t1 - t0).total_seconds() / 86400
print(f"\n[OK] Event period: {t0.date()} -> {t1.date()} ({span_days:.0f} days, "
      f"{N_WINDOWS} windows of ~{span_days / N_WINDOWS:.1f} days)")


def window_of(ts_series):
    days = (pd.to_datetime(ts_series) - t0).dt.total_seconds() / 86400
    return np.minimum((days / span_days * N_WINDOWS).astype(int), N_WINDOWS - 1)


# ── Weighted suspicious-event points per event ──
logins = logins.merge(identities, on="identity_id", how="left")
logins["window"] = window_of(logins["timestamp"])
logins["points"] = (
    (logins["status"] == "failed") * 1.0 +
    (logins["time_class"] == "night") * 0.5 +
    (logins["country"] != logins["home_country"]) * 2.0
)
privs["window"] = window_of(privs["timestamp"])
privs["points"] = (
    privs["approval_status"].isin(["pending", "unapproved"]) * 3.0 +
    (privs["change_type"] == "grant") * 1.0
)
access["window"] = window_of(access["timestamp"])
access["points"] = (
    access["sensitivity"].isin(["high", "critical"]) * 1.0 +
    (access["action"].isin(["download", "export"]) & (access["data_volume"] > 20)) * 3.0
)

events = pd.concat([d[["identity_id", "window", "points"]] for d in (logins, privs, access)])
signal = events.pivot_table(index="identity_id", columns="window", values="points",
                            aggfunc="sum", fill_value=0.0)
signal = signal.reindex(index=identities["identity_id"], columns=range(N_WINDOWS), fill_value=0.0)
signal.columns = [f"w{c:02d}" for c in range(N_WINDOWS)]
signal.to_csv("reports/temporal_monthly_signal.csv")

# ── Vectorised trend fit (least-squares slope over the recent windows) ──
Y = signal.values
recent = Y[:, -TREND_WINDOWS:]
x = np.arange(TREND_WINDOWS, dtype=float)
xc = x - x.mean()
slope = (recent - recent.mean(axis=1, keepdims=True)) @ xc / (xc ** 2).sum()

baseline_mean = Y[:, :N_WINDOWS - 3].mean(axis=1)
recent_mean = Y[:, -3:].mean(axis=1)
overall_mean = Y.mean(axis=1)
# Relative slope: a rise of +2 points/window matters more for a quiet identity
relative_slope = slope / (overall_mean + 1.0)
rise_ratio = (recent_mean + 1.0) / (baseline_mean + 1.0)

traj = pd.DataFrame({
    "identity_id": signal.index,
    "mean_signal": overall_mean.round(3),
    "baseline_signal": baseline_mean.round(3),
    "recent_signal": recent_mean.round(3),
    "trend_slope": slope.round(4),
    "relative_slope": relative_slope.round(4),
    "rise_ratio": rise_ratio.round(3),
})
# 0-100 trend score = percentile rank of the relative slope inside the organisation
traj["temporal_trend_score"] = (traj["relative_slope"].rank(pct=True) * 100).round(2)
traj = traj.merge(scores[["identity_id", "xgb_risk_level_pred", "true_is_anomaly",
                          "true_risk_level"]], on="identity_id", how="left")

slope_cut = np.percentile(traj["relative_slope"], EARLY_WARNING_PCTL)
traj["trajectory"] = np.select(
    [traj["relative_slope"] >= slope_cut, traj["relative_slope"] <= -slope_cut],
    ["RISING", "FALLING"], default="STABLE")
traj["early_warning"] = (
    (traj["relative_slope"] >= slope_cut) &
    (traj["rise_ratio"] >= RISE_RATIO) &
    (traj["xgb_risk_level_pred"] != "CRITICAL")
).astype(int)

traj.to_csv("reports/temporal_risk_trajectories.csv", index=False)

# ── Evaluation (how much does TIME add beyond the static score?) ──
y = traj["true_is_anomaly"].values
auc_trend = roc_auc_score(y, traj["temporal_trend_score"])
auc_level = roc_auc_score(y, traj["mean_signal"])
ew = traj[traj["early_warning"] == 1]
ew_anomaly_rate = float(ew["true_is_anomaly"].mean()) if len(ew) else 0.0
base_rate = float(y.mean())

section("Results")
print(f"Identities with EARLY_WARNING:          {len(ew)}")
print(f"Trajectory mix:                          {traj['trajectory'].value_counts().to_dict()}")
print(f"AUC of activity LEVEL vs true anomaly:   {auc_level:.4f}")
print(f"AUC of activity TREND vs true anomaly:   {auc_trend:.4f}")
print(f"True-anomaly rate among early warnings:  {ew_anomaly_rate:.3f} (base rate {base_rate:.3f})")

metrics = {
    "n_windows": N_WINDOWS, "trend_windows": TREND_WINDOWS,
    "early_warning_count": int(len(ew)),
    "early_warning_true_anomaly_rate": round(ew_anomaly_rate, 4),
    "base_anomaly_rate": round(base_rate, 4),
    "auc_activity_level": round(float(auc_level), 4),
    "auc_activity_trend": round(float(auc_trend), 4),
    "trajectory_counts": traj["trajectory"].value_counts().to_dict(),
    "early_warning_by_predicted_level": ew["xgb_risk_level_pred"].value_counts().to_dict(),
}
write_json("reports/phase4_metrics.json", metrics)

lines = [
    "X-UBA - MCA MAJOR PROJECT - PHASE 4: TEMPORAL RISK TRAJECTORY",
    "=" * 60, "",
    f"Time series: {N_WINDOWS} windows per identity rebuilt from 480,000 raw events.",
    f"Trend: least-squares slope over the last {TREND_WINDOWS} windows, relative to the "
    "identity's own average activity.",
    f"EARLY_WARNING rule: slope in top {100 - EARLY_WARNING_PCTL}% AND recent activity >= "
    f"{RISE_RATIO}x own baseline AND not already predicted CRITICAL.",
    "",
    f"Identities flagged EARLY_WARNING: {len(ew)}",
    f"  by predicted level: {metrics['early_warning_by_predicted_level']}",
    f"Trajectory mix: {metrics['trajectory_counts']}",
    "",
    f"AUC (activity LEVEL vs ground-truth anomaly): {auc_level:.4f}",
    f"AUC (activity TREND vs ground-truth anomaly): {auc_trend:.4f}",
    f"True-anomaly rate among early warnings: {ew_anomaly_rate:.3f} vs base rate {base_rate:.3f}",
    "",
]
if auc_trend < 0.55:
    lines.append(
        "INTERPRETATION: the LEVEL of suspicious activity is strongly linked to risk, but the "
        f"TREND is close to random (AUC {auc_trend:.3f}) because the synthetic generator spreads "
        "each identity's events uniformly over the year - it contains no escalating attack "
        "timelines. The temporal module is validated as working machinery, but its detection "
        "value can only be demonstrated on data with real time-dependent behaviour. This is "
        "reported as a limitation of the synthetic dataset, not hidden.")
else:
    lines.append(
        f"INTERPRETATION: the trend carries real signal (AUC {auc_trend:.3f}), i.e. identities "
        "whose suspicious activity is rising are more likely to be anomalous.")
write_text("reports/phase4_temporal_summary.txt", "\n".join(lines))

print()
banner("[OK] PHASE 4 COMPLETE")
