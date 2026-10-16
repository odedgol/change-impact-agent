from typing import get_args

import pytest

from cia import cases
from cia.cases import REGISTRY, case_type
from cia.generator import CASE_TYPES, EXPECTED, generate
from cia.models import Verdict


def test_registry_keeps_the_week1_order():
    """Order is part of the generated output (it fixes which leaver gets which case)."""
    assert list(REGISTRY) == ["nightly_job", "orphaned_service_account", "policy_exception",
                              "shared_account", "break_glass"]
    assert list(REGISTRY) == CASE_TYPES


def test_every_case_type_has_a_valid_verdict():
    assert set(EXPECTED.values()) <= set(get_args(Verdict))
    assert "safe_to_delete" not in EXPECTED.values()  # only decoys are safe


def test_duplicate_names_are_rejected():
    with pytest.raises(ValueError):
        case_type("nightly_job", expected="ask_owner")(lambda org, target: [])


def test_a_new_case_type_needs_no_change_to_generate(tmp_path, monkeypatch):
    """Open/closed: registering a planter is enough for it to appear in the org."""
    def doc_only(org, target):
        org.add_ticket(target, 10, "wiki", "Runbook says this account rotates the vault key.")
        return [{"kind": "ticket_note", "ref": "runbook"}]

    monkeypatch.setitem(cases.REGISTRY, "doc_only_dependency",
                        cases.CaseType("doc_only_dependency", "ask_owner", doc_only))
    truth = generate(tmp_path / "org.db", n_identities=500, seed=7)
    planted = [c for c in truth["cases"] if c["case_type"] == "doc_only_dependency"]
    assert len(planted) == 2
    assert all(c["hidden_dependencies"] == [{"kind": "ticket_note", "ref": "runbook"}] for c in planted)


def test_every_planter_returns_hidden_dependencies(tmp_path):
    truth = generate(tmp_path / "org.db", n_identities=500, seed=7)
    for c in truth["cases"]:
        assert bool(c["hidden_dependencies"]) == (c["case_type"] != "decoy_safe")
