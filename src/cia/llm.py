"""The one door to a language model. The investigator only knows ``LLMClient``.

Messages, tool specs and turns are plain provider-neutral types. Each client
maps them to its own wire format: ``BedrockClient`` to the Bedrock Converse
API, ``FakeLLM`` to a script for tests. A private-cloud client (for example an
OpenAI-compatible endpoint on vLLM) would be one more mapping, with no change
to the investigator.

``boto3`` is imported only inside ``BedrockClient``, and only when no client is
injected, so the package and its tests never need AWS.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

MODEL_ID_ENV = "CIA_BEDROCK_MODEL_ID"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class LLMTurn:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: str = "end_turn"


@dataclass(frozen=True)
class UserText:
    text: str


@dataclass(frozen=True)
class AssistantTurn:
    turn: LLMTurn


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    content: Any
    is_error: bool = False


@dataclass(frozen=True)
class ToolResults:
    results: tuple[ToolResult, ...]


Message = UserText | AssistantTurn | ToolResults


@runtime_checkable
class LLMClient(Protocol):
    def converse(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMTurn: ...


class ScriptExhausted(RuntimeError):
    """FakeLLM was asked for more turns than its script holds."""


@dataclass
class FakeLLM:
    """Replays scripted turns in order and records every call it receives."""

    script: list[LLMTurn]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def converse(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMTurn:
        self.calls.append({"system": system, "messages": list(messages), "tools": list(tools)})
        if len(self.calls) > len(self.script):
            raise ScriptExhausted(f"script has {len(self.script)} turns; call {len(self.calls)} has none")
        return self.script[len(self.calls) - 1]


class BedrockClient:
    """Claude on Amazon Bedrock through the Converse API."""

    def __init__(self, model_id: str | None = None, region: str | None = None, client: Any = None,
                 temperature: float = 0.0, max_tokens: int = 2048):
        self.model_id = model_id or os.environ.get(MODEL_ID_ENV)
        if not self.model_id:
            raise ValueError(f"set {MODEL_ID_ENV} or pass model_id")
        if client is None:
            import boto3  # lazy: only real Bedrock runs need AWS

            client = boto3.client("bedrock-runtime", region_name=region)
        self._client = client
        self.inference_config = {"temperature": temperature, "maxTokens": max_tokens}

    def converse(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMTurn:
        request: dict[str, Any] = {
            "modelId": self.model_id,
            "system": [{"text": system}],
            "messages": to_converse_messages(messages),
            "inferenceConfig": self.inference_config,
        }
        if tools:
            request["toolConfig"] = to_tool_config(tools)
        return from_converse_response(self._client.converse(**request))


def to_tool_config(tools: Sequence[ToolSpec]) -> dict[str, Any]:
    return {"tools": [{"toolSpec": {"name": t.name, "description": t.description,
                                    "inputSchema": {"json": t.input_schema}}} for t in tools]}


def to_converse_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, UserText):
            out.append({"role": "user", "content": [{"text": m.text}]})
        elif isinstance(m, AssistantTurn):
            content: list[dict[str, Any]] = [{"text": m.turn.text}] if m.turn.text else []
            content += [{"toolUse": {"toolUseId": c.id, "name": c.name, "input": c.input}} for c in m.turn.tool_calls]
            out.append({"role": "assistant", "content": content})
        elif isinstance(m, ToolResults):
            out.append({"role": "user", "content": [_tool_result_block(r) for r in m.results]})
        else:
            raise TypeError(f"unknown message type: {type(m).__name__}")
    return out


def _tool_result_block(r: ToolResult) -> dict[str, Any]:
    # Converse wants a JSON object here; lists, strings and None are wrapped.
    body = r.content if isinstance(r.content, dict) else {"result": r.content}
    return {"toolResult": {"toolUseId": r.tool_call_id, "content": [{"json": body}],
                           "status": "error" if r.is_error else "success"}}


def from_converse_response(response: dict[str, Any]) -> LLMTurn:
    blocks = response["output"]["message"]["content"]
    text = "".join(b["text"] for b in blocks if "text" in b)
    calls = tuple(ToolCall(id=b["toolUse"]["toolUseId"], name=b["toolUse"]["name"], input=b["toolUse"]["input"])
                  for b in blocks if "toolUse" in b)
    return LLMTurn(text=text, tool_calls=calls, stop_reason=response.get("stopReason", "end_turn"))
