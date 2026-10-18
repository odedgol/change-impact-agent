"""The offboarding case: last day -> investigate -> approve -> disable -> daily loop ->
gate in code -> approve delete. The model proposes; code and people decide."""

from datetime import timedelta

import pytest

from cia.audit import verify
from cia.case import (
    ACTIONS,
    MAX_ACTIONS,
    NODE_KIND,
    ROUTE_LABELS,
    describe,
    gate,
    open_desk,
    rule_case_model,
)
from cia.chat_models import PolicyChatModel, calls
from cia.demo_org import build as build_demo
from cia.world import START, build


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    build_demo(out)
    return out


@pytest.fixture
def world_dir(demo, tmp_path):
    build(tmp_path / "world", demo)
    return tmp_path / "world"


def run(desk, days, approve=True):
    for _ in range(days):
        desk.next_day()
        for c in desk.cases():
            if c["waiting_for"] in ("approval", "delete_approval"):
                desk.answer(c["id"], approve, "hr-test")


def day_of(at: str) -> int:
    return (START.fromisoformat(at[:10]) - START).days


@pytest.fixture(scope="module")
def finished(demo, tmp_path_factory):
    """Every case of the demo, worked to the end by the rule model."""
    out = tmp_path_factory.mktemp("run") / "world"
    build(out, demo)
    events = []
    desk = open_desk(out, rule_case_model(), events.append)
    run(desk, 40)
    return desk, events


def test_every_case_ends_deleted_or_kept(finished):
    desk, _ = finished
    ends = {c["name"]: (c["outcome"], c["account"]) for c in desk.cases()}
    assert len(ends) == 7
    kept = [n for n, (o, _) in ends.items() if o == "kept_disabled"]
    assert kept == ["Noah Dubois"]  # the break-glass account
    assert ends["Noah Dubois"][1] == "disabled"
    assert all(
        o == "deleted" and a == "missing" for n, (o, a) in ends.items() if n not in kept
    )


def test_no_delete_before_the_quiet_period_and_every_action_approved(finished):
    desk, _ = finished
    entries = desk.case.audit.entries()
    assert verify(entries) == []
    done = [e for e in entries if e["outcome"] == "done"]
    assert {e["by"] for e in done} == {"hr-test"}
    disabled = {
        e["identity_id"]: day_of(e["at"]) for e in done if e["action"] == "disable"
    }
    for e in done:
        if e["action"] == "delete":
            assert day_of(e["at"]) - disabled[e["identity_id"]] >= 14


def test_the_agent_asks_owners_and_the_answers_clear_blockers(finished):
    _, events = finished
    asks = [e for e in events if e["type"] == "action" and e["tool"] == "ask_owner"]
    things = {e["input"]["thing"] for e in asks}
    assert {"job:nightly-report", "NetSuite", "FIN-221", "Okta Admin"} <= things
    assert all(
        e["result"].get("asked") for e in asks
    )  # each owner asked once per thing


def test_a_model_that_always_says_ready_is_held_by_the_gate(world_dir):
    def eager(messages, names):
        name = (names & {"submit_findings", "submit_update"}).pop()
        return calls(
            (name, {"status": "ready", "summary": "all clear", "blockers": []})
        )

    events = []
    desk = open_desk(world_dir, PolicyChatModel(policy=eager), events.append)
    run(desk, 20)
    cases = {c["name"]: c for c in desk.cases()}
    # nobody asked the job owner, so the job still uses Noah's key: never deleted
    assert cases["Noah Novak"]["account"] == "disabled"
    assert any("still uses" in g for g in cases["Noah Novak"]["gate"])
    assert (
        cases["Noah Dubois"]["outcome"] == "kept_disabled"
    )  # protected wins over ready
    held = [e for e in events if e["type"] == "daily_update" and e["overridden"]]
    assert held


def test_declining_closes_the_case_and_touches_nothing(world_dir):
    desk = open_desk(world_dir, rule_case_model())
    run(desk, 1, approve=False)
    cases = desk.cases()
    assert cases and all(c["outcome"] == "declined" for c in cases)
    assert all(c["account"] == "active" for c in cases)
    assert desk.case.audit.entries() == []


def test_safe_actions_are_capped_per_day(world_dir):
    def chatty(messages, names):
        if "submit_findings" in names:
            return calls(("submit_findings", {"summary": "", "blockers": []}))
        if len(messages) == 2:  # the system prompt and the day's brief: act 5 times
            ticket = {"summary": "check", "text": "hi"}
            return calls(*[("open_ticket", ticket)] * (MAX_ACTIONS + 2))
        return calls(
            ("submit_update", {"status": "on_hold", "summary": "", "blockers": []})
        )

    events = []
    desk = open_desk(world_dir, PolicyChatModel(policy=chatty), events.append)
    run(desk, 2)
    first = [e for e in events if e["type"] == "action"]
    assert len(first) == MAX_ACTIONS * 2  # two cases on day 0, capped each


def test_the_model_cannot_act_on_accounts():
    assert set(ACTIONS) == {
        "ask_owner",
        "comment_ticket",
        "open_ticket",
    }


def test_a_case_survives_a_restart(world_dir):
    desk = open_desk(world_dir, rule_case_model())
    run(desk, 2)
    before = {c["id"]: c["status"] for c in desk.cases()}
    again = open_desk(world_dir, rule_case_model())
    assert {c["id"]: c["status"] for c in again.cases()} == before
    run(again, 1)
    assert again.case.world.day == 2


def test_the_gate_reads_the_systems(world_dir):
    desk = open_desk(world_dir, rule_case_model())
    run(desk, 1)
    w = desk.case.world
    pid = (w.person("Noah Novak") or {})["id"]
    why = gate(w, pid, 0)
    assert any("quiet days" in g for g in why) and any("still uses" in g for g in why)
    assert START + timedelta(days=w.day) == START.fromisoformat(w.today)


def test_every_node_and_route_is_described():
    g = describe()
    for n in g["nodes"]:
        if n["kind"] not in ("start", "end"):
            assert n["id"] in NODE_KIND and n["doc"], n
    for e in g["edges"]:
        assert (e["source"], e["target"]) in ROUTE_LABELS, e


def test_code_refuses_tools_the_phase_does_not_offer(world_dir):
    def pushy(messages, names):
        if len(messages) == 2:  # the investigation: try to act, and a bad call
            return calls(
                ("ask_owner", {"owner": "x", "thing": "y", "question": "z"}),
                ("owner_of", {}),
                ("submit_update", {"status": "ready", "summary": "", "blockers": []}),
            )
        if len(messages) == 6:  # three refusals came back
            return calls(("submit_findings", {"summary": "s", "blockers": []}))
        raise AssertionError(f"one investigation only, got {len(messages)} messages")

    events = []
    desk = open_desk(world_dir, PolicyChatModel(policy=pushy), events.append)
    desk.next_day()
    assert not [e for e in events if e["type"] == "action"]
    asked = desk.case.world.pending_asks(
        (desk.case.world.person("Noah Novak") or {})["id"]
    )
    assert [a["question"][:24] for a in asked] == [
        "Today is Noah Novak's la"
    ]  # only code's question to the manager
    assert all(c["waiting_for"] == "approval" for c in desk.cases())
