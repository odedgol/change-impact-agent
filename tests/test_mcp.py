"""The MCP server: read-only answers about offboarding changes."""

import asyncio
from datetime import timedelta

import pytest

from cia.generator import generate
from cia.mcp_server import Desk, build
from cia.models import OffboardingEvent
from cia.notify import MemoryNotifier
from cia.service import Agent


@pytest.fixture
def desk(tmp_path):
    db = tmp_path / "org.db"
    truth = generate(db, n_identities=500, seed=7)
    t = [1000.0]
    agent = Agent(
        db,
        tmp_path / "data",
        tmp_path / "runs",
        notifier=MemoryNotifier(),
        start_delay=timedelta(0),
        quiet_period=timedelta(seconds=60),
        now=lambda: t[0],
    )

    def ident(case_type):
        return next(c for c in truth["cases"] if c["case_type"] == case_type)[
            "identity_id"
        ]

    agent.receive(OffboardingEvent(event_id="d1", identity_id=ident("decoy_safe")))
    agent.receive(OffboardingEvent(event_id="j1", identity_id=ident("nightly_job")))
    agent.tick()
    agent.answer("OFF-d1", ["disable", "delete"], "dana")
    t[0] += 60
    agent.tick()  # the decoy is deleted
    agent.start_delay = timedelta(hours=1)
    agent.receive(OffboardingEvent(event_id="b1", identity_id=ident("break_glass")))
    return Desk(db, tmp_path / "data", tmp_path / "runs"), ident


def test_who_left_lists_every_change_and_its_state(desk):
    d, _ = desk
    rows = {r["request_id"]: r for r in d.who_left()}
    assert set(rows) == {"OFF-d1", "OFF-j1", "OFF-b1"}
    assert rows["OFF-d1"]["state"] == "Outcome: deleted"
    assert rows["OFF-j1"]["state"].startswith("Allowed: disable")
    assert rows["OFF-b1"]["state"] == "waiting to start"
    assert rows["OFF-d1"]["name"].startswith("user-")


def test_status_by_request_identity_or_name(desk):
    d, ident = desk
    by_id = d.status(ident("nightly_job"))
    name = by_id["changes"][0]["who"].split(" (")[0]
    assert d.status(name) == by_id == d.status("OFF-j1")
    change = by_id["changes"][0]
    assert change["waiting_for"] == "approval" and "disable_first" in change["who"]
    assert any("ev-" in f for f in change["findings"])
    assert d.status("nobody")["found"] is False


def test_report_includes_the_evidence_each_finding_cites(desk):
    d, _ = desk
    r = d.report("OFF-j1")
    assert r["found"] and r["verdict"] == "disable_first"
    cited = {e for f in r["findings"] for e in f["evidence_ids"]}
    assert cited and cited == set(r["evidence"])
    assert all(v["tool"].startswith("get_") for v in r["evidence"].values())
    assert d.report("OFF-b1")["found"] is False  # not started yet


def test_audit_shows_who_approved_and_the_chain(desk):
    d, _ = desk
    a = d.audit("OFF-d1")
    assert a["chain_verified"]
    assert [
        e["action"] for e in a["entries"] if e["action"] in ("disable", "delete")
    ] == [
        "disable",
        "delete",
    ]
    assert {e["by"] for e in a["entries"] if e["action"] == "delete"} == {"dana"}


def test_the_server_offers_only_read_only_tools(desk):
    d, _ = desk
    server = build(d)

    async def go():
        tools = await server.list_tools()
        names = sorted(t.name for t in tools)
        flags = {t.name: t.annotations.read_only_hint for t in tools if t.annotations}
        result = await server.call_tool("status", {"who": "OFF-j1"})
        try:
            await server.call_tool("report", {"request_id": "../etc"})
        except Exception as e:  # noqa: BLE001 - the SDK wraps the tool's ValueError
            bad = f"{e} {e.__cause__}"
        else:
            bad = "accepted"
        return names, flags, result, bad

    names, flags, result, bad = asyncio.run(go())
    assert names == ["audit", "report", "status", "who_left"]
    assert all(flags.values()) and len(flags) == 4
    assert "disable_first" in str(result)
    assert "request id looks like" in str(bad)
