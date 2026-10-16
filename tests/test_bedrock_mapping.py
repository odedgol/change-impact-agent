"""Converse mapping, tested with a stub client: no network, no AWS credentials, no boto3."""
import pytest

from cia.llm import (
    AssistantTurn,
    BedrockClient,
    LLMClient,
    LLMTurn,
    ToolCall,
    ToolResult,
    ToolResults,
    ToolSpec,
    UserText,
)

TOOLS = [ToolSpec("get_dependencies", "What depends on an identity", {
    "type": "object", "properties": {"identity_id": {"type": "string"}}, "required": ["identity_id"]})]


class StubConverse:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def converse(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


def _response(content, stop="end_turn"):
    return {"output": {"message": {"role": "assistant", "content": content}}, "stopReason": stop}


def _client(stub, **kw):
    return BedrockClient(model_id="test-model", client=stub, **kw)


def test_bedrock_satisfies_the_interface():
    assert isinstance(_client(StubConverse(_response([]))), LLMClient)


def test_request_shape():
    stub = StubConverse(_response([{"text": "ok"}]))
    _client(stub).converse("You investigate IAM changes.", [UserText("Investigate u-0042")], TOOLS)
    req = stub.requests[0]
    assert req["modelId"] == "test-model"
    assert req["system"] == [{"text": "You investigate IAM changes."}]
    assert req["messages"] == [{"role": "user", "content": [{"text": "Investigate u-0042"}]}]
    assert req["inferenceConfig"] == {"temperature": 0.0, "maxTokens": 2048}
    spec = req["toolConfig"]["tools"][0]["toolSpec"]
    assert spec["name"] == "get_dependencies"
    assert spec["inputSchema"] == {"json": TOOLS[0].input_schema}


def test_no_tool_config_without_tools():
    stub = StubConverse(_response([{"text": "ok"}]))
    _client(stub).converse("sys", [UserText("hi")], [])
    assert "toolConfig" not in stub.requests[0]


def test_tool_use_response_becomes_tool_calls():
    stub = StubConverse(_response([
        {"text": "Checking dependencies."},
        {"toolUse": {"toolUseId": "tooluse_1", "name": "get_dependencies", "input": {"identity_id": "u-0042"}}},
    ], stop="tool_use"))
    turn = _client(stub).converse("sys", [UserText("hi")], TOOLS)
    assert turn == LLMTurn(text="Checking dependencies.", stop_reason="tool_use",
                           tool_calls=(ToolCall("tooluse_1", "get_dependencies", {"identity_id": "u-0042"}),))


def test_history_round_trips_to_converse_shapes():
    call = ToolCall("tooluse_1", "get_dependencies", {"identity_id": "u-0042"})
    history = [
        UserText("Investigate u-0042"),
        AssistantTurn(LLMTurn(tool_calls=(call,), stop_reason="tool_use")),
        ToolResults((ToolResult("tooluse_1", [{"source": "job-nightly", "kind": "runs_as"}]),
                     ToolResult("tooluse_2", {"error": "unknown tool"}, is_error=True))),
    ]
    stub = StubConverse(_response([{"text": "ok"}]))
    _client(stub).converse("sys", history, TOOLS)
    msgs = stub.requests[0]["messages"]
    assert msgs[1] == {"role": "assistant", "content": [
        {"toolUse": {"toolUseId": "tooluse_1", "name": "get_dependencies", "input": {"identity_id": "u-0042"}}}]}
    assert msgs[2]["role"] == "user"
    ok, err = (b["toolResult"] for b in msgs[2]["content"])
    assert ok == {"toolUseId": "tooluse_1", "status": "success",
                  "content": [{"json": {"result": [{"source": "job-nightly", "kind": "runs_as"}]}}]}
    assert err["status"] == "error"
    assert err["content"] == [{"json": {"error": "unknown tool"}}]


def test_model_id_comes_from_env(monkeypatch):
    monkeypatch.setenv("CIA_BEDROCK_MODEL_ID", "env-model")
    assert BedrockClient(client=StubConverse(_response([]))).model_id == "env-model"


def test_missing_model_id_fails_fast(monkeypatch):
    monkeypatch.delenv("CIA_BEDROCK_MODEL_ID", raising=False)
    with pytest.raises(ValueError, match="CIA_BEDROCK_MODEL_ID"):
        BedrockClient(client=StubConverse(_response([])))
