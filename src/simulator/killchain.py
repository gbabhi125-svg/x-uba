#!/usr/bin/env python3
"""
X-UBA: kill-chain model shared by the attack simulator (Phase 7), the
remediation optimizer and the dashboard.

For a compromised-identity scenario each stage gets a probability derived only
from observable posture (no labels):

  1. INITIAL ACCESS        0.05 base  +0.25 no MFA  +0.5 x failed-login rate
                           +0.10 logins from >2 countries  +0.10 dormant (>90 d)
                           +0.15 expired contractor still present
  2. PRIVILEGE ESCALATION  admin 0.90 / power-user 0.50 / user 0.20
                           +0.10 if it has unapproved privilege changes
  3. LATERAL MOVEMENT      reachable systems / total systems
  4. DATA IMPACT           0.85 if a CRITICAL resource is reachable,
                           0.50 if only HIGH, else 0.15

Each stage is clipped to 0.01-0.95; likelihood = product of the 4 stages;
expected attack risk = likelihood x blast radius (0-100).
"""

import numpy as np
import pandas as pd

STAGES = ["p_initial_access", "p_privilege_escalation", "p_lateral_movement", "p_data_impact"]
REQUIRED_COLUMNS = ["mfa_enabled", "failed_login_rate", "unique_countries", "inactive_days",
                    "is_contractor_expired", "privilege_level", "unapproved_changes",
                    "reachable_systems", "reachable_critical", "reachable_high",
                    "blast_radius_score"]


def kill_chain(t, n_systems_total):
    """Return a DataFrame with the 4 stage probabilities, likelihood and expected risk."""
    p1 = (0.05 + 0.25 * (t["mfa_enabled"] == 0) + 0.5 * t["failed_login_rate"]
          + 0.10 * (t["unique_countries"] > 2) + 0.10 * (t["inactive_days"] > 90)
          + 0.15 * t["is_contractor_expired"])
    p2 = (t["privilege_level"].map({"admin": 0.90, "power-user": 0.50, "user": 0.20})
          + 0.10 * (t["unapproved_changes"] > 0))
    p3 = t["reachable_systems"] / n_systems_total
    p4 = np.select([t["reachable_critical"] > 0, t["reachable_high"] > 0], [0.85, 0.50], 0.15)
    out = pd.DataFrame(index=t.index)
    for name, p in zip(STAGES, (p1, p2, p3, p4)):
        out[name] = np.clip(p, 0.01, 0.95).round(4)
    out["attack_likelihood"] = (out["p_initial_access"] * out["p_privilege_escalation"] *
                                out["p_lateral_movement"] * out["p_data_impact"]).round(5)
    out["expected_attack_risk"] = (out["attack_likelihood"] * t["blast_radius_score"]).round(3)
    return out
