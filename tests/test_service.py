"""The agent as a service: an HR event starts the workflow, a message says what it found."""

import json
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from cia.audit import JsonlAuditSink, verify
from cia.generator import generate
from cia.models import OffboardingEvent
from cia.notify import MemoryNotifier, text
from cia.service import (
    SIGNATURE_HEADER,
    Agent,
    UnknownIdentity,
    create_app,
    duration,
    signature,
)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def desk(tmp_path):
    db = tmp_path / "org.db"
    truth = generate(db, n_identities=500, seed=7)
    clock, note = Clock(), MemoryNotifier()

    def agent():
        return Agent(
            db,
            tmp_path / "data",
            tmp_path / "runs",
            notifier=note,
            start_delay=timedelta(seconds=5),
            quiet_period=timedelta(seconds=60),
            monitor_url="http://127.0.0.1:8787",
            now=clock,
        )

    def ident(case_type):
        return next(c for c in truth["cases"] if c["case_type"] == case_type)[
            "identity_id"
        ]

    return agent, clock, note, ident, tmp_path


def event(identity, eid="e1"):
    return OffboardingEvent(event_id=eid, identity_id=identity)


def test_an_hr_event_starts_the_workflow_after_the_delay(desk):
    make, clock, note, ident, tmp = desk
    agent = make()
    out = agent.receive(event(ident("decoy_safe")))
    assert out == {"status": "scheduled", "request_id": "OFF-e1", "starts_in": 5.0}
    assert agent.tick() == [] and note.posted == []  # nobody pressed anything; not yet
    clock.t += 5
    assert agent.tick() == ["started OFF-e1"]
    (msg,) = note.posted
    assert msg.title.endswith("left · safe_to_delete (medium)")
    assert msg.actions == ("disable", "delete") and not msg.done
    assert msg.link == "http://127.0.0.1:8787/#run=change-OFF-e1"
    assert "Approve delete" in text(msg)
    trace = [json.loads(x) for x in (tmp / "runs/change-OFF-e1.jsonl").open()]
    request = next(e for e in trace if e["type"] == "request")
    assert request["description"].startswith("Offboarding cleanup: delete user-")


def test_approval_then_quiet_period_then_delete(desk):
    make, clock, note, ident, tmp = desk
    agent = make()
    agent.receive(event(ident("decoy_safe")))
    clock.t += 5
    agent.tick()
    status = agent.answer("OFF-e1", ["disable", "delete"], "dana")
    assert status["waiting_for"] == "quiet_period"
    assert note.updates[-1][1].state.startswith("Disabled. Quiet period")
    clock.t += 59
    assert agent.tick() == []
    clock.t += 1
    assert agent.tick() == ["resumed OFF-e1"]
    final = note.updates[-1][1]
    assert final.done and final.state == "Outcome: deleted" and final.actions == ()
    entries = JsonlAuditSink(tmp / "data/audit.jsonl").entries()
    assert verify(entries) == []
    assert [e["by"] for e in entries if e["action"] == "delete"] == ["dana"]


def test_a_risky_account_offers_only_what_the_guardrails_allow(desk):
    make, clock, note, ident, _ = desk
    agent = make()
    agent.receive(event(ident("nightly_job"), "job"))
    agent.receive(event(ident("shared_account"), "shared"))
    clock.t += 5
    agent.tick()
    by_id = {m.request_id: m for m in note.posted}
    job, shared = by_id["OFF-job"], by_id["OFF-shared"]
    assert job.actions == ("disable",) and "disable_first" in job.title
    assert any("ev-" in line for line in job.lines)  # findings carry their evidence ids
    assert shared.actions == () and shared.done  # ask_owner: no buttons at all
    assert shared.state.startswith("Outcome: blocked")
    # more than allowed is refused, and the change ends declined: nothing runs
    refused = agent.answer("OFF-job", ["disable", "delete"], "dana")
    assert refused["outcome"] == "declined" and "approval refused" in refused["reason"]


def test_the_same_event_twice_is_one_change_and_unknown_people_are_refused(desk):
    make, _, _, ident, _ = desk
    agent = make()
    agent.receive(event(ident("decoy_safe")))
    assert agent.receive(event(ident("decoy_safe")))["status"] == "duplicate"
    with pytest.raises(UnknownIdentity):
        agent.receive(event("u99999", "e2"))


def test_a_restart_keeps_pending_starts_and_quiet_periods(desk):
    make, clock, note, ident, _ = desk
    first = make()
    first.receive(event(ident("decoy_safe"), "a"))
    first.receive(event(ident("nightly_job"), "b"))
    clock.t += 5
    first.tick()
    first.answer("OFF-a", ["disable", "delete"], "dana")
    first.receive(event(ident("break_glass"), "c"))  # still waiting to start

    again = make()  # a new process on the same data
    assert set(again.pending) == {"OFF-c"} and set(again.quiet) == {"OFF-a"}
    assert again.receive(event(ident("decoy_safe"), "a"))["status"] == "duplicate"
    clock.t += 60
    assert sorted(again.tick()) == ["resumed OFF-a", "started OFF-c"]
    assert note.updates[-1][1].state == "Outcome: deleted"


def test_http_api_checks_the_signature_and_answers(desk):
    make, clock, _, ident, _ = desk
    agent = make()
    client = TestClient(create_app(agent, secret="s3cret", tick_seconds=None))
    body = json.dumps({"event_id": "h1", "identity_id": ident("decoy_safe")}).encode()
    post = client.post
    assert post("/events/offboarding", content=body).status_code == 401
    bad = {SIGNATURE_HEADER: signature("wrong", body)}
    assert post("/events/offboarding", content=body, headers=bad).status_code == 401
    good = {SIGNATURE_HEADER: signature("s3cret", body)}
    r = post("/events/offboarding", content=body, headers=good)
    assert r.status_code == 202 and r.json()["request_id"] == "OFF-h1"
    assert post("/events/offboarding", content=body, headers=good).status_code == 200
    junk = b'{"event_id": "bad id!", "identity_id": "u1"}'
    assert (
        post(
            "/events/offboarding",
            content=junk,
            headers={SIGNATURE_HEADER: signature("s3cret", junk)},
        ).status_code
        == 422
    )
    assert client.get("/changes").json()[0]["waiting_for"] == "start"
    clock.t += 5
    agent.tick()
    answer = {"approved": ["disable"], "by": "dana"}
    r = post("/changes/OFF-h1/answer", json=answer)
    assert r.status_code == 200 and r.json()["waiting_for"] == "quiet_period"
    assert post("/changes/OFF-h1/answer", json=answer).status_code == 409  # answered
    assert client.get("/health").json() == {"status": "ok"}


def test_durations():
    assert duration("90s") == timedelta(seconds=90)
    assert duration("2m") == timedelta(minutes=2)
    assert duration("14d") == timedelta(days=14)
    assert duration("5") == timedelta(seconds=5)
    with pytest.raises(ValueError):
        duration("soon")
