#!/usr/bin/env python3
"""
X-UBA: attacker-reach model shared by Phase 5 (blast radius), the remediation
optimizer and the dashboard.

Reach rules (documented, deterministic):
  * direct reach   : every resource the identity has successfully accessed
  * admin rights   : on a system where the identity holds admin (its own
                     privilege level on systems it uses, or an admin grant),
                     an attacker reaches EVERY resource of that system
  * power-user     : on such systems, all resources up to "high" sensitivity

`ReachModel.reach()` accepts overrides, so a hypothetical security action
(lower privilege, fewer systems) can be re-evaluated without rebuilding the graph.
"""

import pandas as pd

from common import RAW

SENS_WEIGHT = {"low": 1, "medium": 2, "high": 4, "critical": 8}
POWER_USER_REACH = {"low", "medium", "high"}
PRIV_RANK = {"user": 0, "power-user": 1, "admin": 2}


class ReachModel:
    def __init__(self):
        self.identities = pd.read_csv(f"{RAW}/identities.csv")
        self.resources = pd.read_csv(f"{RAW}/resources.csv")
        self.systems = pd.read_csv(f"{RAW}/systems.csv")
        access = pd.read_csv(f"{RAW}/resource_access.csv",
                             usecols=["identity_id", "resource_id", "system_name", "status"])
        privs = pd.read_csv(f"{RAW}/privilege_changes.csv",
                            usecols=["identity_id", "change_type", "new_privilege", "system_name"])

        self.ok_access = access[access["status"] == "success"]
        self.acc_counts = self.ok_access.groupby(["identity_id", "resource_id"]).size() \
            .reset_index(name="n")
        # Highest privilege ever granted/modified to per (identity, system). Revocations are
        # ignored on purpose: orphaned grants are exactly what IAM audits find.
        held = privs[privs["change_type"].isin(["grant", "modify"])].copy()
        held["rank"] = held["new_privilege"].map(PRIV_RANK)
        self.held = held.groupby(["identity_id", "system_name"])["rank"].max().reset_index()

        self.res_by_system = self.resources.groupby("system_name")["resource_id"].apply(set).to_dict()
        self.res_sens = dict(zip(self.resources["resource_id"], self.resources["sensitivity"]))
        self.res_system = dict(zip(self.resources["resource_id"], self.resources["system_name"]))
        self.direct = self.acc_counts.groupby("identity_id")["resource_id"].apply(set).to_dict()
        self.sys_touched = self.ok_access.groupby("identity_id")["system_name"].apply(set).to_dict()
        self.sys_usage = self.ok_access.groupby(["identity_id", "system_name"]).size()
        self.held_by_id = self.held.groupby("identity_id").apply(
            lambda g: dict(zip(g["system_name"], g["rank"])), include_groups=False).to_dict()
        self.privilege = dict(zip(self.identities["identity_id"], self.identities["privilege_level"]))

    def top_systems(self, iid, k):
        """The k systems an identity uses most (ties broken by name)."""
        if iid not in self.sys_touched:
            return set()
        use = self.sys_usage.loc[iid].sort_index().sort_values(ascending=False, kind="stable")
        return set(use.index[:k])

    def reach(self, iid, privilege_level=None, max_rank=2, keep_systems=None):
        """Set of reachable resource IDs, optionally under a hypothetical action.

        privilege_level : override the identity's own privilege level
        max_rank        : cap every held grant at this rank (0 user, 1 power-user, 2 admin)
        keep_systems    : only keep access / rights on these systems
        """
        plevel = privilege_level or self.privilege[iid]
        direct = set(self.direct.get(iid, set()))
        touched = set(self.sys_touched.get(iid, set()))
        held = {s: min(r, max_rank) for s, r in self.held_by_id.get(iid, {}).items()}
        if keep_systems is not None:
            direct = {x for x in direct if self.res_system[x] in keep_systems}
            touched &= keep_systems
            held = {s: r for s, r in held.items() if s in keep_systems}

        admin_systems = {s for s, r in held.items() if r == 2}
        power_systems = {s for s, r in held.items() if r == 1}
        # The identity's own privilege level applies on every system it actively uses
        if plevel == "admin":
            admin_systems |= touched
        elif plevel == "power-user":
            power_systems |= touched

        reach = set(direct)
        for s in admin_systems:
            reach |= self.res_by_system.get(s, set())
        for s in power_systems - admin_systems:
            reach |= {x for x in self.res_by_system.get(s, set())
                      if self.res_sens[x] in POWER_USER_REACH}
        return reach, admin_systems, direct

    def summarize(self, iid, reach, admin_systems, direct):
        counts = {k: 0 for k in SENS_WEIGHT}
        for x in reach:
            counts[self.res_sens[x]] += 1
        raw = sum(SENS_WEIGHT[k] * v for k, v in counts.items())
        path = ""
        if reach:
            # Attack path to the most sensitive reachable resource (prefer direct access)
            target = max(reach, key=lambda x: (SENS_WEIGHT[self.res_sens[x]], x in direct, x))
            if target in direct:
                path = f"{iid} -> {target} ({self.res_sens[target]})"
            else:
                path = (f"{iid} -> [admin/power rights on {self.res_system[target]}] -> "
                        f"{target} ({self.res_sens[target]})")
        return {
            "reachable_resources": len(reach),
            "direct_resources": len(direct),
            "reachable_systems": len({self.res_system[x] for x in reach}),
            "admin_systems": len(admin_systems),
            "reachable_critical": counts["critical"],
            "reachable_high": counts["high"],
            "blast_radius_raw": raw,
            "attack_path": path,
        }
