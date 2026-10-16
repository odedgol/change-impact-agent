"""Naive policy: delete any leaver with no usage in 90 days.

This is the control the agent has to beat. It shows how many planted hidden
dependencies a plain staleness rule would break.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from .tools import EvidenceTools


def naive_verdict(tools: EvidenceTools, identity_id: str) -> str:
    return "safe_to_delete" if tools.get_usage(identity_id, 90)["event_count"] == 0 else "disable_first"


def run(db: str, truth_path: str) -> dict:
    tools = EvidenceTools(db)
    with open(truth_path) as f:
        truth = json.load(f)
    broken, by_type = 0, Counter()
    for c in truth["cases"]:
        v = naive_verdict(tools, c["identity_id"])
        if v == "safe_to_delete" and c["expected_verdict"] != "safe_to_delete":
            broken += 1
            by_type[c["case_type"]] += 1
    risky = sum(1 for c in truth["cases"] if c["expected_verdict"] != "safe_to_delete")
    return {"requests": len(truth["cases"]), "risky": risky, "broken_by_naive_delete": broken, "by_type": dict(by_type)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/org.db")
    p.add_argument("--truth", default="bench/ground_truth.json")
    a = p.parse_args()
    print(json.dumps(run(a.db, a.truth), indent=2))
