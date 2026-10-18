"""The executor: dry run, approval, disable first, undo, idempotency, an audit entry for each."""

import hashlib
from datetime import datetime

import pytest

from cia.audit import JsonlAuditSink, verify
from cia.executor import (
    Approval,
    Executor,
    ExecutorError,
    SyntheticTarget,
    TargetSystem,
)
from cia.generator import generate
from cia.models import ChangeRequest

AT = datetime(2026, 10, 1, 12, 0, 0)


@pytest.fixture(scope="module")
def org_db(tmp_path_factory):
    db = tmp_path_factory.mktemp("org") / "org.db"
    generate(db, n_identities=500, seed=7)
    return db


@pytest.fixture
def run(org_db, tmp_path):
    target = SyntheticTarget.copy_of(org_db, tmp_path / "target.db")
    audit = JsonlAuditSink(tmp_path / "audit.jsonl")
    return target, audit, Executor(target, audit, clock=lambda: AT)


REQ = ChangeRequest(request_id="REQ-0001", identity_id="u00001")


def approve(*actions, request_id="REQ-0001"):
    return Approval(request_id=request_id, actions=actions, by="alice", at=AT)


def outcomes(audit):
    return [(e["action"], e["outcome"]) for e in audit.entries()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_synthetic_target_is_a_target_system(run):
    assert isinstance(run[0], TargetSystem)


def test_a_dry_run_changes_nothing(run):
    target, audit, ex = run
    steps = ex.plan(REQ, ("disable", "delete"))
    assert [(s["action"], s["from"], s["to"], s["ok"]) for s in steps] == [
        ("disable", "active", "disabled", True),
        ("delete", "disabled", "missing", True),
    ]
    assert target.status("u00001") == "active"
    assert outcomes(audit) == [("plan", "dry_run")]


def test_disable_then_restore_gives_the_account_back(run):
    target, audit, ex = run
    ex.disable(REQ, approve("disable"))
    assert target.status("u00001") == "disabled"
    ex.restore(REQ)
    assert target.status("u00001") == "active"
    assert outcomes(audit) == [("disable", "done"), ("restore", "done")]


def test_the_same_action_twice_acts_once(run):
    _, audit, ex = run
    first = ex.disable(REQ, approve("disable"))
    again = ex.disable(REQ, approve("disable"))
    assert again == first
    assert outcomes(audit) == [("disable", "done")]


def test_delete_needs_this_requests_disable_first(run):
    target, audit, ex = run
    with pytest.raises(ExecutorError, match="disable"):
        ex.delete(REQ, approve("disable", "delete"))
    assert target.status("u00001") == "active"
    assert outcomes(audit) == [("delete", "refused")]


def test_disable_then_delete_removes_the_account(run):
    target, _, ex = run
    ex.disable(REQ, approve("disable", "delete"))
    ex.delete(REQ, approve("disable", "delete"))
    assert target.status("u00001") == "missing"


@pytest.mark.parametrize(
    "approval",
    [approve(), approve("delete"), approve("disable", request_id="REQ-0002")],
)
def test_an_action_without_its_approval_is_refused(run, approval):
    target, audit, ex = run
    with pytest.raises(ExecutorError):
        ex.disable(REQ, approval)
    assert target.status("u00001") == "active"
    assert outcomes(audit) == [("disable", "refused")]


def test_a_delete_that_was_not_approved_is_refused(run):
    target, _, ex = run
    ex.disable(REQ, approve("disable"))
    with pytest.raises(ExecutorError, match="not approved"):
        ex.delete(REQ, approve("disable"))
    assert target.status("u00001") == "disabled"


def test_restore_undoes_only_its_own_disable(run):
    target, _, ex = run
    target.disable("u00001")  # disabled by someone else
    with pytest.raises(ExecutorError, match="disabled nothing"):
        ex.restore(REQ)
    assert target.status("u00001") == "disabled"


def test_an_account_already_changed_elsewhere_is_left_alone(run):
    target, audit, ex = run
    target.delete("u00001")
    with pytest.raises(ExecutorError, match="missing"):
        ex.disable(REQ, approve("disable"))
    assert outcomes(audit) == [("disable", "refused")]


def test_every_entry_says_who_and_when_and_the_chain_holds(run):
    _, audit, ex = run
    ex.plan(REQ, ("disable",))
    ex.disable(REQ, approve("disable"))
    ex.restore(REQ)
    log = audit.entries()
    assert verify(log) == []
    assert all(e["at"] == "2026-10-01T12:00:00" for e in log)
    assert [e.get("by") for e in log] == [None, "alice", "executor"]
    assert {e["key"] for e in log} == {
        "REQ-0001:plan",
        "REQ-0001:disable",
        "REQ-0001:restore",
    }


def test_the_evidence_database_is_never_written(org_db, run):
    before = digest(org_db)
    _, _, ex = run
    ex.disable(REQ, approve("disable", "delete"))
    ex.delete(REQ, approve("disable", "delete"))
    assert digest(org_db) == before
