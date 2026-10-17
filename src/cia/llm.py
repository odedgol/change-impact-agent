"""The one door to a language model. The investigator only knows ``LLMClient``.

Messages, tool specs and turns are plain provider-neutral types. Each client
maps them to its own wire format: ``BedrockClient`` to the Bedrock Converse
API, ``OpenAICompatClient`` to an OpenAI-compatible chat endpoint (Ollama on a
laptop, vLLM in a private cloud), ``OpenRouterClient`` to hosted open models, ``FakeLLM`` to a script for tests. A new
provider is one more mapping, with no change to the investigator.

``boto3`` is imported only inside ``BedrockClient``, and only when no client is
injected, so the package and its tests never need AWS.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

MODEL_ID_ENV = "CIA_BEDROCK_MODEL_ID"
OPENAI_BASE_URL_ENV = "CIA_OPENAI_BASE_URL"
OPENAI_MODEL_ENV = "CIA_OPENAI_MODEL"
OPENAI_API_KEY_ENV = "CIA_OPENAI_API_KEY"
OPENAI_MAX_TOKENS_ENV = "CIA_OPENAI_MAX_TOKENS"  # reasoning models need room to think before the tool call
DEFAULT_OPENAI_BASE_URL = "http://localhost:11434/v1"  # Ollama
DEFAULT_OPENAI_MODEL = "gemma4:26b-mlx"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_MODEL_ENV = "CIA_OPENROUTER_MODEL"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "google/gemma-4-31b-it:free"


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
    reasoning: str = ""  # the model's thinking, when the provider returns it; shown, never trusted


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


Transport = Callable[[str, dict[str, Any], dict[str, str]], dict[str, Any]]


def _http_post(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float = 300) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (URL comes from config)
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:  # keep the server's reason, it is what explains a 400
        raise HTTPStatusError(e.code, f"HTTP {e.code} from {url}: {e.read()[:500].decode(errors='replace')}",
                              retry_after=e.headers.get("Retry-After") if e.headers else None) from e


class HTTPStatusError(RuntimeError):
    def __init__(self, code: int, message: str, retry_after: str | None = None):
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after


def retrying(post: Transport, tries: int = 4, wait: float = 10.0, sleep: Callable[[float], None] = time.sleep) -> Transport:
    """Retry a rate-limited (429) or briefly unavailable (502, 503) request, honouring Retry-After."""
    def call(url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        for attempt in range(tries):
            try:
                return post(url, body, headers)
            except HTTPStatusError as e:
                if e.code not in (429, 502, 503) or attempt == tries - 1:
                    raise
                ra = e.retry_after
                sleep(float(ra) if ra is not None and ra.isdigit() else wait * (attempt + 1))
        raise AssertionError("unreachable")
    return call


class OpenAICompatClient:
    """Any OpenAI-compatible ``/chat/completions`` endpoint: Ollama locally, vLLM in a private cloud.

    Standard library only. ``transport`` is injectable so tests never open a socket.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.0, max_tokens: int | None = None, transport: Transport | None = None):
        self.model = model or os.environ.get(OPENAI_MODEL_ENV) or DEFAULT_OPENAI_MODEL
        self.base_url = (base_url or os.environ.get(OPENAI_BASE_URL_ENV) or DEFAULT_OPENAI_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get(OPENAI_API_KEY_ENV)
        self.temperature = temperature
        self.max_tokens = max_tokens or int(os.environ.get(OPENAI_MAX_TOKENS_ENV) or 2048)
        self._post = transport or _http_post
        self.extra_headers: dict[str, str] = {}

    def converse(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMTurn:
        body: dict[str, Any] = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                                "messages": [{"role": "system", "content": system}, *to_openai_messages(messages)]}
        if tools:
            body["tools"] = to_openai_tools(tools)
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return from_openai_response(self._post(f"{self.base_url}/chat/completions", body, headers))


class OpenRouterClient(OpenAICompatClient):
    """Hosted open models through OpenRouter (default: Gemma 4 31B, free tier). Same wire format as Ollama.

    Needs ``OPENROUTER_API_KEY``. Free models are rate-limited, so 429s are retried with backoff.
    Prompts leave the machine: fine for the synthetic org, not for real IAM data.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None, temperature: float = 0.0,
                 max_tokens: int | None = None, transport: Transport | None = None):
        key = api_key or os.environ.get(OPENROUTER_API_KEY_ENV)
        if not key:
            raise ValueError(f"set {OPENROUTER_API_KEY_ENV} or pass api_key")
        super().__init__(model=model or os.environ.get(OPENROUTER_MODEL_ENV) or DEFAULT_OPENROUTER_MODEL,
                         base_url=OPENROUTER_BASE_URL, api_key=key, temperature=temperature,
                         max_tokens=max_tokens, transport=transport or retrying(_http_post))
        self.extra_headers = {"X-Title": "change-impact-agent"}


LLM_CHOICES = ("openai", "openrouter", "bedrock")  # openai = any OpenAI-compatible URL, Ollama by default


def make_client(llm: str, model: str | None = None, max_tokens: int | None = None) -> LLMClient:
    """The one place a CLI or the monitor turns a provider name into a client."""
    if llm == "openai":
        return OpenAICompatClient(model=model, max_tokens=max_tokens)
    if llm == "openrouter":
        return OpenRouterClient(model=model, max_tokens=max_tokens)
    if llm == "bedrock":
        return BedrockClient(model_id=model, max_tokens=max_tokens or 2048)
    raise ValueError(f"unknown llm {llm!r}; choose from {', '.join(LLM_CHOICES)}")


def to_openai_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                              "parameters": t.input_schema}} for t in tools]


def to_openai_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, UserText):
            out.append({"role": "user", "content": m.text})
        elif isinstance(m, AssistantTurn):
            # an empty turn still needs a string: Ollama rejects content null without tool calls (HTTP 400)
            msg: dict[str, Any] = {"role": "assistant", "content": m.turn.text or (None if m.turn.tool_calls else "")}
            if m.turn.tool_calls:
                msg["tool_calls"] = [{"id": c.id, "type": "function",
                                      "function": {"name": c.name, "arguments": json.dumps(c.input)}}
                                     for c in m.turn.tool_calls]
            out.append(msg)
        elif isinstance(m, ToolResults):
            out += [{"role": "tool", "tool_call_id": r.tool_call_id, "content": json.dumps(r.content, default=str)}
                    for r in m.results]
        else:
            raise TypeError(f"unknown message type: {type(m).__name__}")
    return out


def _arguments(raw: Any) -> dict[str, Any]:
    # Arguments arrive as a JSON string. Anything that is not a JSON object is passed on as-is under
    # "_unparsed", so the investigator answers with a tool error instead of the client crashing.
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {"_unparsed": raw}
    return parsed if isinstance(parsed, dict) else {"_unparsed": raw}


def from_openai_response(response: dict[str, Any]) -> LLMTurn:
    if response.get("error") or not response.get("choices"):  # OpenRouter can report a provider error in a 200
        raise RuntimeError(f"model returned no choices: {str(response.get('error') or response)[:300]}")
    choice = response["choices"][0]
    msg = choice["message"]
    calls = tuple(ToolCall(id=c["id"], name=c["function"]["name"], input=_arguments(c["function"].get("arguments")))
                  for c in msg.get("tool_calls") or ())
    return LLMTurn(text=msg.get("content") or "", tool_calls=calls, stop_reason=choice.get("finish_reason") or "stop",
                   reasoning=msg.get("reasoning") or msg.get("reasoning_content") or "")
