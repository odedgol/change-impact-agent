"""Slack: the message and its buttons, and clicks turned into answers. No network."""

import json
from datetime import timedelta

import pytest

from cia.generator import generate
from cia.models import OffboardingEvent
from cia.notify import Message
from cia.service import Agent
from cia.slack import APPROVE, DECLINE, SlackNotifier, answer_from_click, blocks


class FakeSlack:
    def __init__(self):
        self.calls = []

    def chat_postMessage(self, **kw):  # noqa: N802 - Slack's name
        self.calls.append(("post", kw))
        return {"channel": kw["channel"], "ts": f"{len(self.calls)}.0001"}

    def chat_update(self, **kw):
        self.calls.append(("update", kw))
        return {"ok": True}


def buttons(bks):
    acts = [b for b in bks if b["type"] == "actions"]
    return (
        [(e["text"]["text"], json.loads(e["value"])) for e in acts[0]["elements"]]
        if acts
        else []
    )


def click(value, uid="U1", name="dana"):
    return {
        "user": {"id": uid, "username": name},
        "actions": [{"value": json.dumps(value)}],
    }


@pytest.fixture
def agent(tmp_path):
    db = tmp_path / "org.db"
    truth = generate(db, n_identities=500, seed=7)
    t = [1000.0]
    slack = FakeSlack()
    a = Agent(
        db,
        tmp_path / "data",
        tmp_path / "runs",
        notifier=SlackNotifier(slack, "C123"),
        start_delay=timedelta(0),
        quiet_period=timedelta(seconds=60),
        now=lambda: t[0],
    )

    def start(case_type, eid):
        ident = next(c for c in truth["cases"] if c["case_type"] == case_type)
        a.receive(OffboardingEvent(event_id=eid, identity_id=ident["identity_id"]))
        a.tick()

    return a, slack, start, t


def test_buttons_follow_the_guardrails_and_a_delete_brings_its_disable():
    safe = Message(
        "R1", "x left · safe_to_delete", "Allowed", actions=("disable", "delete")
    )
    assert buttons(blocks(safe)) == [
        ("Approve disable", {"request_id": "R1", "actions": ["disable"]}),
        (
            "Approve disable + delete",
            {"request_id": "R1", "actions": ["disable", "delete"]},
        ),
        ("Decline", {"request_id": "R1", "actions": []}),
    ]
    only = Message("R2", "t", "s", actions=("disable",))
    assert [label for label, _ in buttons(blocks(only))] == [
        "Approve disable",
        "Decline",
    ]
    assert buttons(blocks(Message("R3", "t", "Outcome: blocked"))) == []  # ask_owner


def test_model_text_is_escaped():
    m = Message("R1", "t", "s", lines=("<!channel> click <http://evil|here> & go",))
    text = json.dumps(blocks(m))
    assert "<!channel>" not in text and "&lt;!channel&gt;" in text
    assert "&amp; go" in text


def test_a_change_posts_then_updates_then_tells_the_outcome_in_its_thread(agent):
    a, slack, start, t = agent
    start("decoy_safe", "e1")
    kind, post = slack.calls[0]
    assert kind == "post" and post["channel"] == "C123"
    assert [label for label, _ in buttons(post["blocks"])] == [
        "Approve disable",
        "Approve disable + delete",
        "Decline",
    ]
    note = answer_from_click(
        click({"request_id": "OFF-e1", "actions": ["disable", "delete"]}),
        {"U1"},
        a.answer,
    )
    assert note is None
    kind, upd = slack.calls[-1]
    assert kind == "update" and upd["ts"] == "1.0001" and buttons(upd["blocks"]) == []
    t[0] += 60
    a.tick()
    (kind, thread) = slack.calls[-1]
    assert kind == "post" and thread["thread_ts"] == "1.0001"
    assert thread["text"] == "Outcome: deleted"


def test_only_approvers_may_click_and_a_second_click_is_harmless(agent):
    a, _, start, _ = agent
    start("nightly_job", "e2")
    value = {"request_id": "OFF-e2", "actions": ["disable"]}
    stranger = answer_from_click(click(value, uid="U9"), {"U1"}, a.answer)
    assert stranger and "not on the approver list" in stranger
    assert a.changes()[0]["waiting_for"] == "approval"  # nothing happened
    assert answer_from_click(click(value), {"U1"}, a.answer) is None
    again = answer_from_click(click(value), {"U1"}, a.answer)
    assert again and again.startswith("Not applied")
    assert answer_from_click(
        {"user": {"id": "U1"}, "actions": [{"value": "junk"}]}, {"U1"}, a.answer
    ) == ("This button is not one of ours.")


def test_a_forged_click_for_more_than_allowed_is_refused(agent):
    a, _, start, _ = agent
    start("nightly_job", "e3")  # disable_first: delete is not allowed
    forged = click({"request_id": "OFF-e3", "actions": ["disable", "delete"]})
    note = answer_from_click(forged, {"U1"}, a.answer)
    assert note and note.startswith("Refused: approval refused")


def test_decline_from_slack(agent):
    a, slack, start, _ = agent
    start("decoy_safe", "e4")
    assert (
        answer_from_click(
            click({"request_id": "OFF-e4", "actions": []}), {"U1"}, a.answer
        )
        is None
    )
    assert slack.calls[-1][1]["text"].startswith("Outcome: declined")
    assert APPROVE and DECLINE  # the action ids the manifest and handler share
