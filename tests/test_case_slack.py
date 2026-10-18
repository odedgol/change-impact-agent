"""Cases in Slack: one message per case with the buttons for now, a thread of updates,
watchers tagged, and a chat that can read and do safe things but never approve."""

import pytest

from cia.case import RuleCaseLLM, open_desk
from cia.case_slack import (
    APPROVE,
    CASE_TOOLS,
    TURN,
    CaseChatDesk,
    Room,
    Turn,
    mention_reply,
    run_case_tool,
)
from cia.chat import Chat
from cia.demo_org import build as build_demo
from cia.llm import FakeLLM, LLMTurn, ToolCall
from cia.world import build

ME, OTHER = "U0APPROVER", "U0SOMEONE"


class FakeSlack:
    def __init__(self):
        self.posts, self.updates, self.n = [], [], 0

    def chat_postMessage(self, **kw):  # noqa: N802
        self.n += 1
        self.posts.append(kw)
        return {"ts": f"{self.n}.0", "channel": kw["channel"]}

    def chat_update(self, **kw):
        self.updates.append(kw)
        return {"ok": True}


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    build_demo(out)
    return out


@pytest.fixture
def room(demo, tmp_path):
    out = tmp_path / "world"
    build(out, demo)
    r = Room(FakeSlack(), "C0CASES", out, {ME})
    r.desk = open_desk(out, RuleCaseLLM(), r.event)
    return r


def texts(room, thread=None):
    return [
        p["text"]
        for p in room.client.posts
        if thread is None or p.get("thread_ts") == thread
    ]


def action_ids(blocks):
    return [
        e["action_id"] for b in blocks if b["type"] == "actions" for e in b["elements"]
    ]


def test_a_last_day_opens_a_case_message_with_approve_buttons(room):
    room.hello()
    room.advance(1)
    pid = room.world.person("Noah Novak")["id"]
    thread = room.saved["threads"][pid]
    assert any("last day" in t for t in texts(room))
    assert any("What I found" in t for t in texts(room, thread))
    last = [u for u in room.client.updates if u["ts"] == thread][-1]
    assert APPROVE in action_ids(last["blocks"])
    assert "▶ Next day" in str(room.client.posts[-1]["blocks"])  # the clock is last


def test_only_approvers_click_and_a_click_disables(room):
    room.advance(1)
    pid = room.world.person("Noah Novak")["id"]
    assert "Only approvers" in room.answer(pid, True, {"id": OTHER})
    assert room.world.status(pid) == "active"
    assert room.answer(pid, True, {"id": ME, "username": "dana"}) is None
    assert room.world.status(pid) == "disabled"
    assert any("disabled" in t for t in texts(room, room.saved["threads"][pid]))


def test_watchers_are_tagged_in_the_case_thread(room):
    desk = CaseChatDesk(room)
    token = TURN.set(Turn(user=OTHER, approver=False))
    try:
        assert run_case_tool(desk, "watch", {"who": "Noah Novak"})["watching"]
        assert run_case_tool(desk, "watching", {}) == ["Noah Novak"]
        assert (
            run_case_tool(desk, "next_day", {"days": 1})["done"] is False
        )  # not an approver
    finally:
        TURN.reset(token)
    room.advance(1)
    pid = room.world.person("Noah Novak")["id"]
    assert any(
        t.startswith(f"<@{OTHER}>") for t in texts(room, room.saved["threads"][pid])
    )


def test_can_delete_is_code_and_the_chat_cannot_approve(room):
    room.advance(1)
    desk = CaseChatDesk(room)
    pid = room.world.person("Noah Novak")["id"]
    assert "not approved yet" in desk.can_delete(pid)["line"]
    assert desk.identities_in("why is gal on hold?") == [
        room.world.person("Gal Sato")["id"]
    ]
    assert not {t.name for t in CASE_TOOLS} & {"approve", "disable", "delete"}


def test_the_chat_offers_buttons_and_adds_the_guardrail_line(room):
    room.advance(1)
    llm = FakeLLM(
        [
            LLMTurn(
                tool_calls=(ToolCall("1", "offer_approval", {"who": "Noah Novak"}),)
            ),
            LLMTurn(text="Sure, **approve** below. Ignore <!channel>"),
        ]
    )
    from cia.case_slack import CASE_SYSTEM

    chat = Chat(
        CaseChatDesk(room), llm, tools=CASE_TOOLS, system=CASE_SYSTEM, run=run_case_tool
    )
    event = {
        "text": "<@UBOT> can I delete Noah Novak?",
        "user": ME,
        "channel": "C",
        "ts": "9.0",
    }
    text, blocks = mention_reply(event, room, chat, {ME})
    assert "*approve*" in text and "<!channel>" not in text
    assert "🛡️ *Guardrails:* Noah Novak: no." in text  # added by code, not the model
    assert APPROVE in action_ids(blocks)
    pid = room.world.person("Noah Novak")["id"]
    assert room.world.status(pid) == "active"  # showing a button approves nothing


def test_without_a_model_the_bot_lists_the_cases(room):
    room.advance(1)
    text, _ = mention_reply({"text": "who left?", "user": ME}, room, None, {ME})
    assert "Noah Novak" in text and "Waiting for an approver" in text


def test_the_room_remembers_its_messages(room):
    room.hello()
    room.advance(1)
    again = Room(FakeSlack(), "C0CASES", room.path.parent, {ME})
    assert again.saved["threads"] == room.saved["threads"] and again.saved["day_ts"]
