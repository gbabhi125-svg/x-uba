#!/usr/bin/env python3
"""
X-UBA: security actions ("interventions") shared by the counterfactual engine,
the remediation optimizer and the live dashboard, so every part of the system
applies exactly the same transformation to an identity's features.

Each action takes a features DataFrame (modified in place) and returns a
boolean Series: True where the action actually changes something.
"""

import numpy as np

from common import add_derived_features, encode, risk_index

PRIV_DOWN = {"admin": "power-user", "power-user": "user", "user": "user"}


def enforce_mfa(d):
    applies = d["mfa_enabled"] == 0
    d.loc[applies, "mfa_enabled"] = 1
    d.loc[applies, "mfa_usage_rate"] = np.maximum(d.loc[applies, "mfa_usage_rate"], 0.95)
    return applies


def downgrade_privilege(d):
    applies = d["privilege_level"] != "user"
    d.loc[applies, "privilege_level"] = d.loc[applies, "privilege_level"].map(PRIV_DOWN)
    return applies


def recertify_stale(d):
    applies = d["inactive_days"] > 30
    d.loc[applies, "inactive_days"] = 30
    return applies


def reduce_footprint(d):
    applies = d["n_systems"] > 3
    ratio = (3 / d.loc[applies, "n_systems"]).clip(upper=1)
    d.loc[applies, "n_systems"] = 3
    d.loc[applies, "system_count"] = np.minimum(d.loc[applies, "system_count"], 3)
    for c in ("resource_count", "access_count", "sensitive_resource_access"):
        d.loc[applies, c] = (d.loc[applies, c] * ratio).round()
    return applies


def revoke_unapproved(d):
    applies = d["unapproved_changes"] > 0
    d.loc[applies, "unapproved_changes"] = 0
    return applies


# name -> (function, model features the action changes, plain-English description)
ACTIONS = {
    "ENFORCE_MFA": (enforce_mfa, ["mfa_enabled", "mfa_usage_rate"],
                    "Turn on MFA and require it on >= 95% of logins"),
    "DOWNGRADE_PRIVILEGE": (downgrade_privilege, ["privilege_level_enc"],
                            "Step privilege down one level (admin -> power-user -> user)"),
    "RECERTIFY_STALE_ACCESS": (recertify_stale, ["inactive_days"],
                               "Re-certify a dormant account (inactive days capped at 30)"),
    "REDUCE_SYSTEM_FOOTPRINT": (reduce_footprint, ["n_systems", "system_count", "resource_count",
                                                   "access_count", "sensitive_resource_access"],
                                "Trim access to at most 3 systems"),
    "REVOKE_UNAPPROVED": (revoke_unapproved, ["unapproved_changes"],
                          "Roll back unapproved privilege changes"),
}


def score(features, model, classes, metadata):
    """0-100 risk index for a features DataFrame (derived features recomputed)."""
    d = add_derived_features(features.copy())
    return risk_index(model.predict_proba(encode(d, metadata)), classes)


def apply_actions(features, names):
    """Apply the named actions to a copy; returns (new_features, names_that_applied)."""
    d = features.copy()
    applied = []
    for n in names:
        if n in ACTIONS and bool(ACTIONS[n][0](d).any()):
            applied.append(n)
    return d, applied


GRAPH_ACTIONS = ("DOWNGRADE_PRIVILEGE", "REDUCE_SYSTEM_FOOTPRINT")
FOOTPRINT_SYSTEMS = 3


def graph_after_action(rm, iid, action, privilege_level, raw_max):
    """Blast-radius columns after an action, recomputed on the privilege graph.

    rm is a graph_analysis.reach.ReachModel; returns None for actions that do not
    change graph reach (MFA, re-certification, revoking unapproved changes).
    """
    if action == "DOWNGRADE_PRIVILEGE":
        new_level = PRIV_DOWN[privilege_level]
        rank = {"user": 0, "power-user": 1, "admin": 2}[new_level]
        reach, admin, direct = rm.reach(iid, privilege_level=new_level, max_rank=rank)
    elif action == "REDUCE_SYSTEM_FOOTPRINT":
        reach, admin, direct = rm.reach(iid, keep_systems=rm.top_systems(iid, FOOTPRINT_SYSTEMS))
    else:
        return None
    s = rm.summarize(iid, reach, admin, direct)
    return {"blast_radius_score": round(s["blast_radius_raw"] / raw_max * 100, 2),
            "reachable_systems": s["reachable_systems"], "reachable_critical": s["reachable_critical"],
            "reachable_high": s["reachable_high"]}
