import json
import sqlite3

import pytest

from cia.baseline import run as run_baseline
from cia.generator import CASE_TYPES, generate
from cia.sources import MAX_DEPTH
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
    assert types.count("decoy_safe") == 2 * len(CASE_TYPES)


def test_every_planted_dependency_is_discoverable_with_tools(org):
    """If the tools cannot surface it, no agent could find it."""
    db, truth, _ = org
    tools = EvidenceTools(db)
    for c in truth["cases"]:
        if not c["hidden_dependencies"]:
            continue
        evidence = json.dumps(
            {
                "deps": tools.get_dependencies(c["identity_id"], MAX_DEPTH),
                "tickets": tools.get_tickets(c["identity_id"]),
                "usage": tools.get_usage(c["identity_id"], 365),
            }
        )
        for h in c["hidden_dependencies"]:
            ref = h["ref"]
            hit = (
                ref in evidence
                or (ref == "shared by finance" and "finance" in evidence)
                or (ref == "break-glass" and "Break-glass" in evidence)
            )
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
    assert len(truth["cases"]) == 80 * len(CASE_TYPES) * 2


def test_a_credential_chain_is_found_only_by_following_hops(org):
    """user <- key <- secret <- pipeline <- team: the pipeline is three hops away."""
    db, truth, _ = org
    tools = EvidenceTools(db)
    chains = [c for c in truth["cases"] if c["case_type"] == "credential_chain"]
    assert chains
    for c in chains:
        pipeline = c["hidden_dependencies"][0]["ref"]
        assert tools.get_usage(c["identity_id"], 365)["event_count"] == 0
        assert pipeline not in json.dumps(tools.get_dependencies(c["identity_id"]))
        rows = tools.get_dependencies(c["identity_id"], depth=4)
        assert [r["hop"] for r in rows] == [1, 2, 3, 4]
        assert [r["kind"] for r in rows] == [
            "owns_credential",
            "stored_in",
            "read_by",
            "owned_by",
        ]
        assert rows[3]["source"] == "team:data-platform"
        assert rows[3]["via"] == [rows[0]["source"], rows[1]["source"], pipeline]


def test_naive_policy_deletes_every_credential_chain(org):
    db, truth, truth_path = org
    chains = sum(c["case_type"] == "credential_chain" for c in truth["cases"])
    assert run_baseline(db, truth_path)["by_type"]["credential_chain"] == chains
