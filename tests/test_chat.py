"""Talk to the bot: the chat graph's read-only tools and code's last word. Scripted model."""

import json
from datetime import timedelta

import pytest

from cia.chat import FORGET_AFTER, MAX_STEPS, TOOLS, Chat, run_tool
from cia.generator import generate
from cia.llm import FakeLLM, LLMTurn, ToolCall, ToolResults, UserText
from cia.mcp_server import Desk
from cia.models import OffboardingEvent
from cia.notify import MemoryNotifier
from cia.service import Agent
from cia.workflow import sqlite_checkpointer


@pytest.fixture
def desk(tmp_path):
    db = tmp_path / "org.db"
    truth = generate(db, n_identities=500, seed=7)
    agent = Agent(
        db,
        tmp_path / "data",
        tmp_path / "runs",
        notifier=MemoryNotifier(),
        start_delay=timedelta(0),
    )

    def ident(case_type):
        return next(c for c in truth["cases"] if c["case_type"] == case_type)[
            "identity_id"
        ]

    agent.receive(OffboardingEvent(event_id="j1", identity_id=ident("nightly_job")))
    agent.receive(OffboardingEvent(event_id="d1", identity_id=ident("decoy_safe")))
    agent.tick()
    return Desk(db, tmp_path / "data", tmp_path / "runs"), ident, tmp_path


def turn(text="", *calls):
    return LLMTurn(
        text=text,
        tool_calls=tuple(ToolCall(f"c{i}", n, a) for i, (n, a) in enumerate(calls)),
    )


def test_the_chat_has_only_read_only_tools():
    assert {t.name for t in TOOLS} == {
        "who_left",
        "profile",
        "can_delete",
        "change_status",
        "report",
        "audit",
    }


def test_the_guardrail_line_wins_even_when_the_model_says_yes(desk):
    d, ident, tmp = desk
    llm = FakeLLM(
        [
            turn("", ("can_delete", {"who": "OFF-j1"})),
            turn("**Yes**, delete it right now."),
        ]
    )
    out = Chat(d, llm).answer("C1:1", "can I delete OFF-j1 yet?")
    assert out["text"].startswith("*Yes*, delete it right now.")
    assert "🛡️ *Guardrails:*" in out["text"]
    assert "not yet" in out["text"] and "Delete is not allowed" in out["text"]
    result = llm.calls[1]["messages"][-1]
    assert isinstance(result, ToolResults)
    assert result.results[0].content["data"][0]["can_delete_now"] is False


def test_a_delete_question_gets_the_line_even_without_the_tool(desk):
    d, ident, _ = desk
    llm = FakeLLM([turn("Sure, go ahead.")])
    out = Chat(d, llm).answer("C1:2", f"is it safe to remove {ident('decoy_safe')}?")
    assert "Guardrails:" in out["text"] and "allow disable, delete" in out["text"]


def test_text_from_the_model_or_a_ticket_pings_nobody(desk):
    d, _, _ = desk
    llm = FakeLLM([turn("<!channel> see <http://evil|here> & now")])
    text = Chat(d, llm).answer("C1:3", "hi")["text"]
    assert "<!channel>" not in text and "&lt;!channel&gt;" in text
    assert "<http" not in text


def test_the_step_cap_holds(desk):
    d, _, _ = desk
    loop = [turn("", ("who_left", {}))] * (MAX_STEPS + 2)
    llm = FakeLLM(loop)
    out = Chat(d, llm).answer("C1:4", "who left?")
    assert "could not finish" in out["text"]
    assert len(llm.calls) == MAX_STEPS


def test_a_thread_remembers_and_another_thread_does_not(desk):
    d, _, tmp = desk
    llm = FakeLLM([turn("one"), turn("two"), turn("three")])
    t = [1000.0]
    chat = Chat(d, llm, sqlite_checkpointer(tmp / "chat.db"), now=lambda: t[0])
    chat.answer("C1:5", "first question")
    chat.answer("C1:5", "and the next?")
    seen = [m.text for m in llm.calls[1]["messages"] if isinstance(m, UserText)]
    assert seen == ["first question", "and the next?"]
    chat.answer("C1:6", "other thread")
    assert [m.text for m in llm.calls[2]["messages"] if isinstance(m, UserText)] == [
        "other thread"
    ]


def test_a_thread_is_forgotten_after_seven_days(desk):
    d, _, tmp = desk
    llm = FakeLLM([turn("one"), turn("two")])
    t = [1000.0]
    chat = Chat(d, llm, sqlite_checkpointer(tmp / "chat.db"), now=lambda: t[0])
    chat.answer("C1:7", "old question")
    t[0] += FORGET_AFTER + 1
    chat.answer("C1:7", "new question")
    seen = [m.text for m in llm.calls[1]["messages"] if isinstance(m, UserText)]
    assert seen == ["new question"]


def test_who_left_and_profile_tools(desk):
    d, ident, _ = desk
    rows = run_tool(d, "who_left", {})
    assert rows and {"name", "last_day", "state"} <= set(rows[0])
    states = {r["id"]: r["state"] for r in rows}
    assert states[ident("nightly_job")].startswith("Allowed: disable")
    assert (
        run_tool(d, "who_left", {"days": 1}) == []
    )  # nobody left on the org's last day
    prof = run_tool(d, "profile", {"who": ident("nightly_job")})
    assert prof[0]["id"] == ident("nightly_job")
    assert run_tool(d, "profile", {"who": "nobody"})["found"] is False
    assert run_tool(d, "report", {"request_id": "../x"})["found"] is False


def test_a_profiled_person_comes_with_a_card_and_a_down_model_falls_back(desk):
    from cia.slack import chat_reply

    d, ident, _ = desk
    who = ident("nightly_job")
    llm = FakeLLM([turn("", ("profile", {"who": who})), turn("Here.")])
    event = {"text": f"<@U0BOT> tell me about {who}", "channel": "C1", "ts": "9.1"}
    text, blocks = chat_reply(event, d, Chat(d, llm))
    assert text == "Here." and any(b["type"] == "header" for b in blocks)

    class Down:
        def converse(self, *a, **k):
            raise RuntimeError("503")

    text, blocks = chat_reply(event, d, Chat(d, Down()))
    assert "not answering" in json.dumps(blocks) and blocks[1]["type"] == "header"


def test_the_log_holds_tokens_not_words(desk, tmp_path):
    d, _, _ = desk
    log = tmp_path / "chat.jsonl"
    Chat(d, FakeLLM([turn("ok")]), log=log).answer("C1:8", "secret words")
    line = log.read_text()
    assert "secret words" not in line and json.loads(line)["steps"] == 1
