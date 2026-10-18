"""Chat models through LangChain (week 5, PR 5.6).

One factory for every provider, so the graphs only ever see a LangChain
``BaseChatModel`` and bind LangChain tools to it:

    openrouter  ChatOpenAI on https://openrouter.ai/api/v1 (cost reported per call)
    ollama      ChatOpenAI on Ollama's OpenAI-compatible endpoint (local, free)
    bedrock     ChatBedrockConverse from langchain-aws (optional extra; boto3 inside)

Two models without a provider, for tests, CI and the bench:

    ScriptedChatModel  replays AIMessages in order and records what it was sent
    PolicyChatModel    a fixed rule written as a function of the messages and tools
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any
from uuid import uuid4

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = {"openrouter": "google/gemma-4-26b-a4b-it", "ollama": "gemma4:26b-mlx"}


def make_chat_model(
    provider: str, model: str | None = None, max_tokens: int = 4096
) -> BaseChatModel:
    """A LangChain chat model for ``provider``; keys and URLs come from the environment."""
    model = model or DEFAULT_MODEL.get(provider)
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or "",
            base_url=OPENROUTER_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],  # type: ignore[arg-type]
            max_completion_tokens=max_tokens,
            extra_body={"usage": {"include": True}},  # OpenRouter adds the cost
            default_headers={"X-Title": "Change Impact Agent"},
        )
    if provider == "ollama":
        from langchain_openai import ChatOpenAI

        url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        return ChatOpenAI(
            model=model or "",
            base_url=f"{url}/v1",
            api_key="ollama",  # type: ignore[arg-type]
            max_completion_tokens=max_tokens,
        )
    if provider == "bedrock":
        from langchain_aws import ChatBedrockConverse  # type: ignore[import-not-found]

        return ChatBedrockConverse(model=model, max_tokens=max_tokens)  # type: ignore[call-arg]
    raise ValueError(f"unknown provider {provider!r}: openrouter, ollama or bedrock")


def usage_of(message: BaseMessage) -> dict[str, Any] | None:
    """Tokens and, when the provider reports it, cost of one model reply."""
    meta = getattr(message, "usage_metadata", None)
    if not meta:
        return None
    raw = (getattr(message, "response_metadata", None) or {}).get("token_usage") or {}
    cost = raw.get("cost") if isinstance(raw, dict) else None
    return {
        "input_tokens": meta.get("input_tokens", 0),
        "output_tokens": meta.get("output_tokens", 0),
        "cost_usd": cost if isinstance(cost, int | float) else None,
    }


class _ToolAware(BaseChatModel):
    """Binding tools keeps their schemas in the call, so a fake can see what it was offered."""

    def bind_tools(  # type: ignore[override]
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)  # type: ignore[return-value]


def offered(kwargs: dict[str, Any]) -> set[str]:
    return {t["function"]["name"] for t in kwargs.get("tools") or []}


class ScriptedChatModel(_ToolAware):
    """Replays ``script`` in order; ``calls`` records each call's messages and tool names."""

    script: list[AIMessage]
    calls: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append({"messages": list(messages), "tools": offered(kwargs)})
        if len(self.calls) > len(self.script):
            raise RuntimeError(f"the script has {len(self.script)} turns")
        return ChatResult(
            generations=[ChatGeneration(message=self.script[len(self.calls) - 1])]
        )


class PolicyChatModel(_ToolAware):
    """A fixed rule as a chat model: ``policy(messages, tool_names) -> AIMessage``."""

    policy: Callable[[list[BaseMessage], set[str]], AIMessage]
    label: str = "rule"

    @property
    def _llm_type(self) -> str:
        return "policy"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        reply = self.policy(list(messages), offered(kwargs))
        return ChatResult(generations=[ChatGeneration(message=reply)])


def calls(*named: tuple[str, dict[str, Any]], text: str = "") -> AIMessage:
    """An AIMessage that calls tools, for scripts and rules: ``calls(("case", {"who": "Noah"}))``."""
    return AIMessage(
        content=text,
        tool_calls=[
            {"name": n, "args": a, "id": f"call-{uuid4().hex[:8]}", "type": "tool_call"}
            for n, a in named
        ],
    )
