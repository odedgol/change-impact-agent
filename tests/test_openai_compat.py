"""OpenAI-compatible mapping (Ollama, vLLM), tested with a stub transport: no network."""
import json

import pytest

from cia.llm import (
    AssistantTurn,
    LLMClient,
    LLMTurn,
    OpenAICompatClient,
    ToolCall,
    ToolResult,
    ToolResults,
    ToolSpec,
    UserText,
)

TOOLS = [ToolSpec("get_usage", "Usage of an identity", {
    "type": "object", "properties": {"identity_id": {"type": "string"}}, "required": ["identity_id"]})]


class StubTransport:
    def __init__(self, message, finish="tool_calls"):
        self.response = {"choices": [{"message": message, "finish_reason": finish}]}
        self.requests = []

    def __call__(self, url, body, headers):
        self.requests.append({"url": url, "body": body, "headers": headers})
        return self.response


def tool_message(arguments='{"identity_id": "u00002"}', **extra):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "get_usage", "arguments": arguments}}], **extra}


def test_satisfies_the_interface():
    assert isinstance(OpenAICompatClient(transport=StubTransport({})), LLMClient)


def test_defaults_point_at_local_ollama(monkeypatch):
    for v in ("CIA_OPENAI_MODEL", "CIA_OPENAI_BASE_URL", "CIA_OPENAI_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    stub = StubTransport(tool_message())
    OpenAICompatClient(transport=stub).converse("sys", [UserText("hi")], TOOLS)
    req = stub.requests[0]
    assert req["url"] == "http://localhost:11434/v1/chat/completions"
    assert req["body"]["model"] == "gemma4:26b-mlx" and req["body"]["temperature"] == 0
    assert "Authorization" not in req["headers"]


def test_env_and_api_key(monkeypatch):
    monkeypatch.setenv("CIA_OPENAI_BASE_URL", "https://vllm.internal/v1/")
    monkeypatch.setenv("CIA_OPENAI_MODEL", "qwen3:14b")
    monkeypatch.setenv("CIA_OPENAI_API_KEY", "k")
    stub = StubTransport(tool_message())
    OpenAICompatClient(transport=stub).converse("sys", [UserText("hi")], TOOLS)
    assert stub.requests[0]["url"] == "https://vllm.internal/v1/chat/completions"
    assert stub.requests[0]["body"]["model"] == "qwen3:14b"
    assert stub.requests[0]["headers"]["Authorization"] == "Bearer k"


def test_request_maps_system_tools_and_history():
    stub = StubTransport(tool_message())
    history = [UserText("delete u00002"),
               AssistantTurn(LLMTurn(tool_calls=(ToolCall("call_1", "get_usage", {"identity_id": "u00002"}),))),
               ToolResults((ToolResult("call_1", {"evidence_id": "ev-usage-1", "data": [1, 2]}),))]
    OpenAICompatClient(transport=stub).converse("be careful", history, TOOLS)
    body = stub.requests[0]["body"]
    assert body["tools"] == [{"type": "function", "function": {
        "name": "get_usage", "description": "Usage of an identity", "parameters": TOOLS[0].input_schema}}]
    system, user, assistant, tool = body["messages"]
    assert system == {"role": "system", "content": "be careful"}
    assert user == {"role": "user", "content": "delete u00002"}
    assert assistant["content"] is None
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"identity_id": "u00002"}
    assert tool["role"] == "tool" and tool["tool_call_id"] == "call_1"
    assert json.loads(tool["content"]) == {"evidence_id": "ev-usage-1", "data": [1, 2]}


def test_response_maps_tool_calls_and_reasoning():
    stub = StubTransport(tool_message(reasoning="Check usage first."))
    turn = OpenAICompatClient(transport=stub).converse("s", [UserText("x")], TOOLS)
    assert turn.tool_calls == (ToolCall("call_1", "get_usage", {"identity_id": "u00002"}),)
    assert turn.reasoning == "Check usage first." and turn.stop_reason == "tool_calls"


def test_text_answer_without_tools():
    stub = StubTransport({"role": "assistant", "content": "done"}, finish="stop")
    turn = OpenAICompatClient(transport=stub).converse("s", [UserText("x")], [])
    assert (turn.text, turn.tool_calls, turn.stop_reason) == ("done", (), "stop")
    assert "tools" not in stub.requests[0]["body"]


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", None])
def test_bad_arguments_are_passed_on_not_crashed(raw):
    turn = OpenAICompatClient(transport=StubTransport(tool_message(arguments=raw))).converse("s", [UserText("x")], TOOLS)
    assert "_unparsed" in turn.tool_calls[0].input or turn.tool_calls[0].input == {}


def test_an_empty_turn_is_sent_back_as_an_empty_string_not_null():
    """Ollama answers HTTP 400 to an assistant message with content null and no tool calls."""
    stub = StubTransport({"role": "assistant", "content": "ok"}, finish="stop")
    client = OpenAICompatClient(model="m", transport=stub)
    call = ToolCall("c1", "get_usage", {"identity_id": "u1"})
    client.converse("sys", [UserText("hi"), AssistantTurn(LLMTurn()), UserText("call a tool"),
                            AssistantTurn(LLMTurn(tool_calls=(call,)))], TOOLS)
    sent = [m for m in stub.requests[0]["body"]["messages"] if m["role"] == "assistant"]
    assert sent[0]["content"] == "" and "tool_calls" not in sent[0]
    assert sent[1]["content"] is None and sent[1]["tool_calls"]


def test_an_http_error_keeps_the_server_reason(monkeypatch):
    import email.message
    import io
    import urllib.error

    from cia import llm

    def refuse(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", email.message.Message(),
                                     io.BytesIO(b'{"error":"invalid message content type"}'))

    monkeypatch.setattr(llm.urllib.request, "urlopen", refuse)
    with pytest.raises(RuntimeError, match="HTTP 400 .*invalid message content type"):
        llm._http_post("http://x/v1/chat/completions", {}, {})


def test_max_tokens_from_argument_env_or_default(monkeypatch):
    monkeypatch.delenv("CIA_OPENAI_MAX_TOKENS", raising=False)
    assert OpenAICompatClient(model="m").max_tokens == 2048
    monkeypatch.setenv("CIA_OPENAI_MAX_TOKENS", "8192")
    assert OpenAICompatClient(model="m").max_tokens == 8192
    assert OpenAICompatClient(model="m", max_tokens=100).max_tokens == 100
