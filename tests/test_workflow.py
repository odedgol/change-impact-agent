"""The change workflow on the synthetic org, with the rule model and a fake clock.

Nothing changes before a person approves; a decline or a guardrail block changes
nothing; use during the quiet period restores; only safe_to_delete ever deletes;
a restart between the waits resumes from the checkpoint and never acts twice.
"""

from datetime import datetime, timedelta

import pytest

from cia.audit import JsonlAuditSink, verify
from cia.bench import RuleLLM
from cia.executor import Executor, SyntheticTarget
from cia.generator import NOW, generate
from cia.investigator import Investigator
from cia.models import ChangeRequest
from cia.sources import SyntheticSource
from cia.workflow import ChangeWorkflow, sqlite_checkpointer, used_since


@pytest.fixture(scope="module")
def org(tmp_path_factory):
    db = tmp_path_factory.mktemp("org") / "org.db"
    truth = generate(db, n_identities=500, seed=7)
    return db, truth["cases"]


class Clock:
    def __init__(self):
        self.now = NOW

    def __call__(self):
        return self.now


class UsedAfter(SyntheticSource):
    """The org, plus one sign-in at ``when``: something still uses the account."""

    def __init__(self, db, when):
        super().__init__(str(db))
        self.when = when

    def get_usage(self, identity_id, days=90):
        return {**super().get_usage(identity_id, days), "last_seen": self.when}


class Rig:
    """One org copy, audit log and checkpoint file; each call is a new process."""

    def __init__(self, db, tmp_path):
        self.db, self.tmp = db, tmp_path
        self.clock = Clock()
        self.target = SyntheticTarget.copy_of(db, tmp_path / "target.db")
        self.audit = JsonlAuditSink(tmp_path / "audit.jsonl")

    def __call__(self, source=None):
        inv = Investigator(
            source or SyntheticSource(str(self.db)), RuleLLM(), clock=self.clock
        )
        ex = Executor(self.target, self.audit, clock=self.clock)
        cp = sqlite_checkpointer(self.tmp / "checkpoints.db")
        return ChangeWorkflow(inv, ex, cp, clock=self.clock)


@pytest.fixture
def make(org, tmp_path):
    return Rig(org[0], tmp_path)


def case(org, case_type):
    c = next(c for c in org[1] if c["case_type"] == case_type)
    return ChangeRequest(request_id=c["request_id"], identity_id=c["identity_id"])


def actions(audit):
    return [(e["action"], e["outcome"]) for e in audit.entries()]


def test_it_waits_for_a_person_and_changes_nothing_before(org, make):
    req = case(org, "decoy_safe")
    s = make().start(req)
    assert s["verdict"] == "safe_to_delete"
    assert s["waiting_for"] == "approval" and s["allowed"] == ["disable", "delete"]
    assert make.target.status(req.identity_id) == "active"
    assert actions(make.audit) == [("guardrail", "allowed")]


def test_a_decline_changes_nothing(org, make):
    req = case(org, "decoy_safe")
    flow = make()
    flow.start(req)
    s = flow.answer(req.request_id, [], by="alice")
    assert s["outcome"] == "declined" and s["waiting_for"] is None
    assert make.target.status(req.identity_id) == "active"


def test_approved_delete_runs_dry_run_disable_quiet_period_then_delete(org, make):
    req = case(org, "decoy_safe")
    flow = make()
    flow.start(req)
    s = flow.answer(req.request_id, ["disable", "delete"], by="alice")
    assert s["waiting_for"] == "quiet_period"
    assert make.target.status(req.identity_id) == "disabled"
    s = flow.resume(req.request_id, NOW + timedelta(days=3))  # too early: wait again
    assert s["waiting_for"] == "quiet_period" and s["outcome"] is None
    s = flow.resume(req.request_id, NOW + timedelta(days=14))
    assert s["outcome"] == "deleted"
    assert make.target.status(req.identity_id) == "missing"
    assert actions(make.audit) == [
        ("guardrail", "allowed"),
        ("approval", "approved"),
        ("plan", "dry_run"),
        ("disable", "done"),
        ("recheck", "quiet"),
        ("delete", "done"),
    ]
    assert verify(make.audit.entries()) == []


def test_approving_only_the_disable_keeps_the_account_disabled(org, make):
    req = case(org, "decoy_safe")
    flow = make()
    flow.start(req)
    flow.answer(req.request_id, ["disable"], by="alice")
    s = flow.resume(req.request_id, NOW + timedelta(days=15))
    assert s["outcome"] == "kept_disabled"
    assert make.target.status(req.identity_id) == "disabled"


def test_disable_first_never_deletes_even_if_a_person_asks(org, make):
    req = case(org, "nightly_job")
    flow = make()
    s = flow.start(req)
    assert s["verdict"] == "disable_first" and s["allowed"] == ["disable"]
    s = flow.answer(req.request_id, ["disable", "delete"], by="alice")
    assert s["outcome"] == "declined" and "delete" in s["reason"]
    assert make.target.status(req.identity_id) == "active"


@pytest.mark.parametrize("case_type", ["break_glass", "credential_chain"])
def test_ask_owner_is_blocked_before_any_person_is_asked(org, make, case_type):
    req = case(org, case_type)
    s = make().start(req)
    assert s["verdict"] == "ask_owner"
    assert s["outcome"] == "blocked" and s["waiting_for"] is None
    assert actions(make.audit) == [("guardrail", "blocked")]
    assert make.target.status(req.identity_id) == "active"


def test_use_during_the_quiet_period_restores_the_account(org, make):
    req = case(org, "decoy_safe")
    flow = make(UsedAfter(org[0], (NOW + timedelta(days=2)).isoformat()))
    flow.start(req)
    flow.answer(req.request_id, ["disable", "delete"], by="alice")
    s = flow.resume(req.request_id, NOW + timedelta(days=14))
    assert s["outcome"] == "restored"
    assert make.target.status(req.identity_id) == "active"
    assert ("delete", "done") not in actions(make.audit)


def test_a_restart_between_the_waits_resumes_and_never_acts_twice(org, make):
    req = case(org, "decoy_safe")
    make().start(req)
    make().answer(req.request_id, ["disable", "delete"], by="alice")  # a new process
    s = make().resume(req.request_id, NOW + timedelta(days=14))  # and another
    assert s["outcome"] == "deleted"
    assert actions(make.audit).count(("disable", "done")) == 1


def test_a_run_cannot_be_resumed_for_the_wrong_wait(org, make):
    req = case(org, "decoy_safe")
    flow = make()
    flow.start(req)
    with pytest.raises(ValueError, match="not waiting for quiet_period"):
        flow.resume(req.request_id, NOW)
    with pytest.raises(KeyError):
        flow.status("REQ-9999")


def test_use_counts_only_after_the_disable():
    since = datetime(2026, 10, 1, 12)
    assert not used_since({"last_seen": None}, since)
    assert not used_since({"last_seen": "2026-09-30T02:00:00"}, since)
    assert used_since({"last_seen": "2026-10-02T02:00:00"}, since)
