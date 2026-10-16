"""Deterministic synthetic organization with planted hidden dependencies.

The generator writes a SQLite database and a ground-truth file. Every planted
case is a leaver account that looks safe to delete but is not (except decoys,
which really are safe). The ground truth is what the agent is measured against.

The case types themselves live in ``cases.py``, one registered function each.
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path

from .cases import NOW, REGISTRY, OrgBuilder, _ts

__all__ = ["CASE_TYPES", "EXPECTED", "NOW", "generate"]

DEPARTMENTS = ["finance", "engineering", "sales", "support", "hr", "legal", "data", "security"]

SCHEMA = """
CREATE TABLE identities (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL,
    department TEXT, hr_status TEXT, account_status TEXT NOT NULL, owner_id TEXT
);
CREATE TABLE usage_events (identity_id TEXT NOT NULL, ts TEXT NOT NULL, resource TEXT NOT NULL, caller TEXT NOT NULL);
CREATE TABLE dependencies (source TEXT NOT NULL, target_id TEXT NOT NULL, kind TEXT NOT NULL, detail TEXT);
CREATE TABLE tickets (id TEXT PRIMARY KEY, identity_id TEXT NOT NULL, ts TEXT NOT NULL, author TEXT, text TEXT NOT NULL);
CREATE INDEX idx_usage ON usage_events(identity_id, ts);
CREATE INDEX idx_dep ON dependencies(target_id);
CREATE INDEX idx_ticket ON tickets(identity_id);
"""

# Derived from the registry in cases.py; kept as names for existing imports.
CASE_TYPES = list(REGISTRY)
EXPECTED = {name: case.expected_verdict for name, case in REGISTRY.items()}


def generate(path: str | Path, n_identities: int = 500, seed: int = 7) -> dict:
    rng = random.Random(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)

    per_type = max(1, n_identities // 250)
    n_cases = per_type * len(REGISTRY)
    n_decoys = n_cases
    n_leavers = n_cases + n_decoys
    n_services = max(n_cases, n_identities // 7)
    n_humans = n_identities - n_services
    if n_humans <= n_leavers:
        raise ValueError("n_identities too small for the planted cases")

    org = OrgBuilder(rng)

    humans = []
    for i in range(n_humans):
        hid = f"u{i:05d}"
        dept = rng.choice(DEPARTMENTS)
        leaver = i < n_leavers
        org.add_identity((hid, "human", f"user-{i:05d}", dept, "left" if leaver else "employed", "active", None))
        humans.append(hid)
        if not leaver:
            for _ in range(rng.randint(2, 4)):
                org.add_usage(hid, _ts(rng.uniform(0, 20)), rng.choice(["crm", "wiki", "mail", "git"]), "interactive")

    leavers = humans[:n_leavers]
    rng.shuffle(leavers)
    case_ids, decoy_ids = leavers[:n_cases], leavers[n_cases:]

    # Quiet background service accounts (owned by active employees).
    for j in range(n_services):
        sid = f"svc{j:05d}"
        owner = humans[n_leavers + rng.randrange(n_humans - n_leavers)]
        org.add_identity((sid, "service", f"svc-{rng.choice(['etl', 'report', 'sync', 'cron'])}-{j}", rng.choice(DEPARTMENTS), None, "active", owner))
        org.services.append(sid)
        for _ in range(3):
            org.add_usage(sid, _ts(rng.uniform(0, 10)), "api", f"service:{sid}")

    cases = []
    it = iter(case_ids)
    for case in REGISTRY.values():
        for _ in range(per_type):
            target = next(it)
            cases.append({
                "request_id": f"REQ-{len(cases) + 1:04d}", "identity_id": target, "action": "delete",
                "case_type": case.name, "expected_verdict": case.expected_verdict,
                "hidden_dependencies": case.plant(org, target),
            })

    for target in decoy_ids:
        org.add_ticket(target, rng.uniform(30, 60), "hr-bot", "Employee left the company.")
        cases.append({
            "request_id": f"REQ-{len(cases) + 1:04d}", "identity_id": target, "action": "delete",
            "case_type": "decoy_safe", "expected_verdict": "safe_to_delete", "hidden_dependencies": [],
        })

    db.executemany("INSERT INTO identities VALUES (?,?,?,?,?,?,?)", org.identities)
    db.executemany("INSERT INTO usage_events VALUES (?,?,?,?)", org.usage)
    db.executemany("INSERT INTO dependencies VALUES (?,?,?,?)", org.deps)
    db.executemany("INSERT INTO tickets VALUES (?,?,?,?,?)", org.tickets)
    db.commit()
    db.close()
    rng.shuffle(cases)
    return {"seed": seed, "n_identities": n_identities, "now": NOW.isoformat(), "cases": cases}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--identities", type=int, default=500)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="data/org.db")
    p.add_argument("--truth", default="bench/ground_truth.json")
    a = p.parse_args()
    truth = generate(a.out, a.identities, a.seed)
    Path(a.truth).parent.mkdir(parents=True, exist_ok=True)
    Path(a.truth).write_text(json.dumps(truth, indent=2))
    print(f"{a.identities} identities, {len(truth['cases'])} change requests -> {a.out}")


if __name__ == "__main__":
    main()
