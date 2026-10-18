"""The change workflow on the synthetic org, with the rule model and a fake clock.

Nothing changes before a person approves; a decline or a guardrail block changes
nothing; use during the quiet period restores; only safe_to_delete ever deletes;
a restart between the waits resumes from the checkpoint and never acts twice.
"""

import json
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


class Live(SyntheticSource):
    """The org, plus tickets written after the report, or a ticket system that is down."""

    def __init__(self, db):
        super().__init__(str(db))
        self.new_tickets: list[dict] = []
        self.down = False

    def get_tickets(self, identity_id):
        if self.down:
            raise ConnectionError("ticket system timed out")
        return [*super().get_tickets(identity_id), *self.new_tickets]


def ticket(n, author="finance-lead", text="We still use this login for the close."):
    return {
        "id": f"T-NEW-{n}",
        "ts": "2026-10-02T09:00:00",
        "author": author,
        "text": text,
    }


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

    def __call__(self, source=None, events=None):
        inv = Investigator(
            source or SyntheticSource(str(self.db)),
            RuleLLM(),
            clock=self.clock,
            events=events,
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
        ("verify", "unchanged"),
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


def run_count(events):
    return sum(e["type"] == "request" for e in events)


def test_unchanged_evidence_keeps_the_report_without_calling_the_model(org, make):
    req = case(org, "decoy_safe")
    events = []
    flow = make(events=events.append)
    flow.start(req)
    make.clock.now = NOW + timedelta(days=3)  # the person answers three days later
    s = flow.answer(req.request_id, ["disable"], by="alice")
    assert s["waiting_for"] == "quiet_period"
    assert run_count(events) == 1  # one investigation, no second model run
    assert ("verify", "unchanged") in actions(make.audit)


def test_a_new_ticket_before_approval_means_a_new_investigation(org, make):
    req = case(org, "decoy_safe")
    live = Live(org[0])
    events = []
    flow = make(live, events.append)
    flow.start(req)
    live.new_tickets.append(ticket(1))  # finance writes in after the report
    s = flow.answer(req.request_id, ["disable", "delete"], by="alice")
    assert s["outcome"] == "blocked" and s["verdict"] == "ask_owner"
    assert run_count(events) == 2
    assert {"type": "evidence_changed", "tools": ["get_tickets"]} in events
    assert make.target.status(req.identity_id) == "active"


def test_a_change_that_keeps_the_verdict_still_asks_the_person_again(org, make):
    req = case(org, "decoy_safe")
    live = Live(org[0])
    flow = make(live)
    flow.start(req)
    live.new_tickets.append(ticket(1, author="hr-bot", text="Laptop returned."))
    s = flow.answer(req.request_id, ["disable", "delete"], by="alice")
    assert s["waiting_for"] == "approval" and s["approved"] == []  # the old yes is void
    assert make.target.status(req.identity_id) == "active"
    s = flow.answer(req.request_id, ["disable", "delete"], by="alice")
    assert s["waiting_for"] == "quiet_period"  # our own disable is not a change


def test_evidence_that_cannot_be_read_at_approval_stops_with_no_change(org, make):
    req = case(org, "decoy_safe")
    live = Live(org[0])
    flow = make(live)
    flow.start(req)
    live.down = True
    s = flow.answer(req.request_id, ["disable"], by="alice")
    assert s["outcome"] == "failed" and "could not re-read" in s["reason"]
    assert make.target.status(req.identity_id) == "active"


@pytest.mark.parametrize("break_it", ["ticket", "down"])
def test_a_change_or_a_failed_read_during_the_quiet_period_restores(
    org, make, break_it
):
    req = case(org, "decoy_safe")
    live = Live(org[0])
    flow = make(live)
    flow.start(req)
    flow.answer(req.request_id, ["disable", "delete"], by="alice")
    if break_it == "ticket":
        live.new_tickets.append(ticket(1))
    else:
        live.down = True
    s = flow.resume(req.request_id, NOW + timedelta(days=14))
    assert s["outcome"] == "restored"
    assert make.target.status(req.identity_id) == "active"


def test_an_approval_after_the_7_day_backstop_acts_on_nothing(org, make):
    req = case(org, "decoy_safe")
    flow = make()
    flow.start(req)
    make.clock.now = NOW + timedelta(days=8)
    s = flow.answer(req.request_id, ["disable"], by="alice")
    assert s["outcome"] == "blocked" and "old" in s["reason"]
    assert make.target.status(req.identity_id) == "active"


def test_evidence_that_keeps_changing_stops_after_three_investigations(org, make):
    req = case(org, "decoy_safe")
    live = Live(org[0])
    flow = make(live)
    s = flow.start(req)
    for n in range(3):
        live.new_tickets.append(ticket(n, author="hr-bot", text=f"note {n}"))
        s = flow.answer(req.request_id, ["disable"], by="alice")
    assert s["outcome"] == "failed" and "kept changing" in s["reason"]
    assert make.target.status(req.identity_id) == "active"


# -- the change-desk command line (make change / approve / resume) -------------------------


def test_command_line_runs_a_change_end_to_end(tmp_path, capsys):
    from cia.workflow import main

    db, truth_path = tmp_path / "org.db", tmp_path / "truth.json"
    truth = generate(db, n_identities=500, seed=7)
    truth_path.write_text(json.dumps(truth))
    rid = next(c for c in truth["cases"] if c["case_type"] == "decoy_safe")[
        "request_id"
    ]
    common = ["--db", str(db), "--truth", str(truth_path)]
    common += ["--data", str(tmp_path / "data"), "--runs", str(tmp_path / "runs")]

    def run(*args):
        main([*args, *common])
        return json.loads(capsys.readouterr().out.splitlines()[-1])

    assert run("start", rid)["waiting_for"] == "approval"
    assert run("approve", rid, "--by", "ana")["waiting_for"] == "quiet_period"
    assert run("resume", rid, "--days", "14")["outcome"] == "deleted"
    assert run("status")["outcome"] == "deleted"

    lines = (tmp_path / "runs" / f"change-{rid}.jsonl").read_text().splitlines()
    types = [json.loads(line)["type"] for line in lines]
    assert types[0] == "run" and types[-1] == "outcome"
    assert types.count("run") == 1  # approve and resume append to the same trace

    with pytest.raises(SystemExit, match="not waiting for approval"):
        main(["approve", rid, *common])
    main(["reset", "--data", str(tmp_path / "data")])
    assert len(json.loads(capsys.readouterr().out)["removed"]) == 3
