import pytest

from cia.llm import (
    FakeLLM,
    LLMClient,
    LLMTurn,
    ScriptExhausted,
    ToolCall,
    ToolSpec,
    UserText,
)

TOOLS = [ToolSpec("get_usage", "Usage events for an identity", {"type": "object"})]


def test_fake_satisfies_the_interface():
    assert isinstance(FakeLLM(script=[]), LLMClient)


def test_returns_scripted_turns_in_order():
    first = LLMTurn(
        tool_calls=(ToolCall("t1", "get_usage", {"identity_id": "u-1"}),),
        stop_reason="tool_use",
    )
    second = LLMTurn(text="done")
    llm = FakeLLM(script=[first, second])
    assert llm.converse("sys", [UserText("hi")], TOOLS) is first
    assert llm.converse("sys", [UserText("hi")], TOOLS) is second


def test_records_what_it_received():
    llm = FakeLLM(script=[LLMTurn(text="ok")])
    llm.converse("system prompt", [UserText("investigate u-1")], TOOLS)
    assert llm.calls == [
        {
            "system": "system prompt",
            "messages": [UserText("investigate u-1")],
            "tools": TOOLS,
        }
    ]


def test_raises_when_the_script_runs_out():
    llm = FakeLLM(script=[LLMTurn(text="only one")])
    llm.converse("sys", [], [])
    with pytest.raises(ScriptExhausted):
        llm.converse("sys", [], [])
