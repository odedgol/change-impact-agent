import json
import sqlite3

import pytest

from cia.baseline import run as run_baseline
from cia.generator import CASE_TYPES, generate
from cia.tools import EvidenceTools


@pytest.fixture(scope="module")
def org(tmp_path_factory):
    d = tmp_path_factory.mktemp("org")
    db = d / "org.db"
    truth = generate(db, n_identities=500, seed=7)
    (d / "truth.json").write_text(json.dumps(truth))
    return str(db), truth, str(d / "truth.json")


def test_deterministic(tmp_path):
    a = generate(tmp_path / "a.db", 500, seed=3)
    b = generate(tmp_path / "b.db", 500, seed=3)
    assert a == b


def test_case_mix(org):
    _, truth, _ = org
    types = [c["case_type"] for c in truth["cases"]]
    for t in CASE_TYPES:
        assert types.count(t) == 2
    assert types.count("decoy_safe") == 10


def test_every_planted_dependency_is_discoverable_with_tools(org):
    """If the tools cannot surface it, no agent could find it."""
    db, truth, _ = org
    tools = EvidenceTools(db)
    for c in truth["cases"]:
        if not c["hidden_dependencies"]:
            continue
        evidence = json.dumps({
            "deps": tools.get_dependencies(c["identity_id"]),
            "tickets": tools.get_tickets(c["identity_id"]),
            "usage": tools.get_usage(c["identity_id"], 365),
        })
        for h in c["hidden_dependencies"]:
            ref = h["ref"]
            hit = ref in evidence or (ref == "shared by finance" and "finance" in evidence) or (
                ref == "break-glass" and "Break-glass" in evidence)
            assert hit, (c["case_type"], h)


def test_tools_are_read_only(org):
    db, _, _ = org
    tools = EvidenceTools(db)
    with pytest.raises(sqlite3.OperationalError):
        tools._db.execute("DELETE FROM identities")


def test_decoys_have_no_evidence(org):
    db, truth, _ = org
    tools = EvidenceTools(db)
    for c in truth["cases"]:
        if c["case_type"] == "decoy_safe":
            assert tools.get_usage(c["identity_id"], 365)["event_count"] == 0
            assert tools.get_dependencies(c["identity_id"]) == []


def test_naive_policy_breaks_things(org):
    db, _, truth_path = org
    result = run_baseline(db, truth_path)
    assert result["broken_by_naive_delete"] > 0
    assert result["broken_by_naive_delete"] <= result["risky"]


def test_scales_to_twenty_thousand(tmp_path):
    truth = generate(tmp_path / "big.db", n_identities=20_000, seed=1)
    assert len(truth["cases"]) == 80 * 5 * 2
