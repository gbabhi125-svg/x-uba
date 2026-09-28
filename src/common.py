#!/usr/bin/env python3
"""
X-UBA: Explainable User and Entity Behavior Analytics
       Using Heterogeneous Security Telemetry

Shared helpers used by every pipeline module, so all phases load data,
encode features and compute risk scores in exactly the same way.

Every module does:
    from common import ...        (after putting src/ on sys.path)
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEED = 42

RAW = "data/raw"
PROCESSED = "data/processed"
MODELS = "models"
REPORTS = "reports"

CATEGORICAL_COLS = ["department", "job_title", "privilege_level"]
TARGET_COLS = ["is_anomaly", "risk_level", "threat_type"]

RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
# Expected-value weights that turn the 4-class risk_level probabilities into
# one continuous 0-100 risk index (used by every downstream module).
RISK_WEIGHTS = {"LOW": 0.0, "MEDIUM": 100 / 3, "HIGH": 200 / 3, "CRITICAL": 100.0}

PRIVILEGE_LEVELS = ["user", "power-user", "admin"]


def init():
    """Anchor to the project root and make sure output folders exist."""
    os.chdir(PROJECT_ROOT)
    for d in (RAW, PROCESSED, MODELS, REPORTS):
        os.makedirs(d, exist_ok=True)


def banner(title):
    print("=" * 70)
    print(title)
    print("=" * 70)


def section(title):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def require(*paths):
    """Fail fast with a clear message if an earlier phase has not been run."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "[X] Missing required input(s):\n    " + "\n    ".join(missing) +
            "\n    Run the earlier phases first:  python run_pipeline.py"
        )


# ── Feature engineering shared by the generator and the counterfactual engine ──

def add_derived_features(df):
    """Derived behavioural signals. Recomputed whenever a base feature changes."""
    df["behaviour_deviation_score"] = (
        df["failed_login_rate"] * 0.3 +
        df["night_login_rate"] * 0.2 +
        (df["unique_countries"] > 2).astype(int) * 0.25 +
        (df["unapproved_changes"] > 0).astype(int) * 0.25
    ).round(4)
    df["privilege_change_trend"] = (
        df["privilege_escalations"] - df["privilege_changes_count"] * 0.5
    ).round(4)
    df["risk_trend"] = (
        df["behaviour_deviation_score"] * 0.5 + df["privilege_change_trend"].clip(lower=0) * 0.1
    ).round(4)
    return df


# ── Data loading / encoding ──

def load_features():
    require(f"{PROCESSED}/identity_features.csv", f"{PROCESSED}/train_features.csv",
            f"{PROCESSED}/test_features.csv")
    full = pd.read_csv(f"{PROCESSED}/identity_features.csv")
    train = pd.read_csv(f"{PROCESSED}/train_features.csv")
    test = pd.read_csv(f"{PROCESSED}/test_features.csv")
    return full, train, test


def load_metadata():
    import joblib
    require(f"{MODELS}/feature_metadata.joblib")
    return joblib.load(f"{MODELS}/feature_metadata.joblib")


def encode(df, metadata):
    """Return the numeric model matrix (same columns, same order as Phase 1)."""
    df = df.copy()
    for col, le in metadata["categorical_encoders"].items():
        df[col + "_enc"] = le.transform(df[col].astype(str))
    return df[metadata["feature_cols"]].astype(float)


def risk_index(proba, classes):
    """Collapse risk_level class probabilities into a 0-100 risk index."""
    w = np.array([RISK_WEIGHTS[c] for c in classes])
    return np.asarray(proba) @ w


def risk_band(score):
    """Map a 0-100 risk index back to a named band (same cut-points as the index)."""
    score = np.asarray(score, dtype=float)
    return np.select(
        [score >= 75, score >= 50, score >= 25], ["CRITICAL", "HIGH", "MEDIUM"], default="LOW"
    )


def minmax_100(x):
    x = np.asarray(x, dtype=float)
    span = x.max() - x.min()
    if span == 0:
        return np.zeros_like(x)
    return (x - x.min()) / span * 100
