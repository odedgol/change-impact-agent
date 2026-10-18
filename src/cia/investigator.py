"""The investigator: a LangGraph state graph (``cia.graph``) that turns a change request into a RiskReport.

The model only proposes. Code owns everything that makes the report trustworthy:

- Evidence ids. Every tool result is logged as ``ev-<tool>-<n>`` by the graph and
  handed to the model with the data; the model can cite ids, never invent them.
- The step cap. At most ``max_steps`` tool calls (``submit_report`` and a turn
  with no tool call count too). Hitting the cap returns ``insufficient_evidence``
  with low confidence, never a delete.
- The citation check. A report citing an unknown id goes back once with the
  rejected ids (the retry counts against the cap). After that, findings with
  unknown ids are dropped, the unknown ids are counted in
  ``rejected_citations``, and ``safe_to_delete`` with any rejection becomes
  ``ask_owner``.
- Paging. List tools return 50-row pages with ``next_offset``; nothing is cut.
- Incomplete is not safe. ``safe_to_delete`` stands only if usage, dependencies
  and tickets were all read successfully, with no unread pages; otherwise it
  becomes ``insufficient_evidence``. A failing source is a tool error, never an
  empty result.
- Freshness. Each piece of evidence records ``collected_at`` (from ``clock``),
  the source's ``name`` and, only if the source exposes it, ``updated_at``.

``events`` receives one dict per step (request, model turn, tool call, tool error,
citation retry, step cap, report); ``JsonlEvents`` writes them as JSON lines for the terminal UI.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import IO, Any

from langchain_core.runnables import RunnableConfig

from .graph import (
    REQUIRED_FOR_DELETE,
    SUBMIT,
    build_graph,
    initial_state,
    recursion_limit,
)
from .llm import LLMClient, ToolSpec
from .models import ChangeRequest, RiskReport
from .sources import EvidenceSource

__all__ = [
    "MAX_STEPS",
    "REQUIRED_FOR_DELETE",
    "SUBMIT",
    "TOOLS",
    "Investigator",
    "JsonlEvents",
    "main",
]

MAX_STEPS = 8

SYSTEM_PROMPT = """You investigate whether an IAM identity can be safely deleted.
Use the tools to gather evidence: usage, dependencies (including consumers of accounts it owns), tickets and the owner.
get_dependencies follows dependencies hop by hop (depth 1-4, default 2); a key or secret that depends on the identity may be used further along, so look deeper when you see one.
Dormant is not the same as unused: break-glass accounts, shared logins, scheduled jobs and policy exceptions look idle.
List tools return pages; if next_offset is not null, there are more rows.
When done, call submit_report once. Verdicts:
- safe_to_delete: nothing depends on the identity.
- disable_first: something may depend on it; disabling is reversible, deleting is not.
- ask_owner: a person must decide (an owned service account, a credential still used downstream, a shared or break-glass account, unclear evidence).
- insufficient_evidence: you could not check what you needed (a tool failed, a source was unavailable).
Every finding must cite the evidence_id values returned by the tools. Never cite an id you were not given."""

EventSink = Callable[[dict[str, Any]], None]

_ID = {"type": "string", "description": "identity id, for example u00042"}
_OFFSET = {
    "type": "integer",
    "minimum": 0,
    "description": "row offset of the page, from next_offset",
}
_DEPTH = {
    "type": "integer",
    "minimum": 1,
    "maximum": 4,
    "description": "how many hops to follow (default 2)",
}

TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "get_identity",
        "The identity's record: kind, department, HR status, account status, owner id.",
        {
            "type": "object",
            "properties": {"identity_id": _ID},
            "required": ["identity_id"],
        },
    ),
    ToolSpec(
        "get_usage",
        "Usage events in a window (at least 90 days): count, last seen, top callers.",
        {
            "type": "object",
            "properties": {
                "identity_id": _ID,
                "days": {"type": "integer", "minimum": 1},
            },
            "required": ["identity_id"],
        },
    ),
    ToolSpec(
        "get_dependencies",
        "Things that depend on the identity, hop by hop: hop 1 points at it, hop 2 at a "
        "hop-1 thing (such as consumers of an account it owns). Each row has hop and via. "
        "Paged.",
        {
            "type": "object",
            "properties": {"identity_id": _ID, "offset": _OFFSET, "depth": _DEPTH},
            "required": ["identity_id"],
        },
    ),
    ToolSpec(
        "get_tickets",
        "Tickets and notes about the identity, newest first. Paged.",
        {
            "type": "object",
            "properties": {"identity_id": _ID, "offset": _OFFSET},
            "required": ["identity_id"],
        },
    ),
    ToolSpec(
        "get_owner",
        "The identity's owner record, if it has one.",
        {
            "type": "object",
            "properties": {"identity_id": _ID},
            "required": ["identity_id"],
        },
    ),
    ToolSpec(
        SUBMIT,
        "Submit the final risk report. Every finding cites evidence ids from the tool results.",
        {
            "type": "object",
            "required": ["verdict", "confidence", "findings"],
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": [
                        "safe_to_delete",
                        "disable_first",
                        "ask_owner",
                        "insufficient_evidence",
                    ],
                },
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["kind", "summary", "evidence_ids", "confidence"],
                        "properties": {
                            "kind": {"type": "string"},
                            "summary": {"type": "string"},
                            "evidence_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                            },
                            "confidence": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                            },
                        },
                    },
                },
            },
        },
    ),
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Investigator:
    """Runs the investigator graph (``cia.graph``) for one change request at a time."""

    def __init__(
        self,
        source: EvidenceSource,
        llm: LLMClient,
        max_steps: int = MAX_STEPS,
        events: EventSink | None = None,
        clock: Callable[[], datetime] = _utcnow,
        checkpointer: Any = None,
    ):
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1 (got {max_steps})")
        self.source = source
        self.llm = llm
        self.max_steps = max_steps
        self.events = events or (lambda event: None)
        self.clock = clock
        # Optional source metadata; never invented when the source does not expose it.
        self.source_name = getattr(source, "name", None) or type(source).__name__
        self.source_updated_at = getattr(source, "updated_at", None)
        # With a checkpointer every node's state is saved under the request id (thread), so a run can be
        # inspected step by step, and later paused for human approval and resumed.
        self.checkpointer = checkpointer
        self.graph = build_graph(self, checkpointer)

    def investigate(self, request: ChangeRequest) -> RiskReport:
        self.events(
            {
                "type": "request",
                "request_id": request.request_id,
                "identity_id": request.identity_id,
                "action": request.action,
                "description": request.description,
            }
        )
        config: RunnableConfig = {"recursion_limit": recursion_limit(self.max_steps)}
        if self.checkpointer is not None:
            config["configurable"] = {"thread_id": request.request_id}
        final = self.graph.invoke(initial_state(request), config)
        report = final["report"]
        self.events({"type": "report", **report.model_dump(), "steps": final["steps"]})
        return report


class JsonlEvents:
    """Writes each event as one JSON line, stamped with wall-clock time ``t``, for the UIs or a replay."""

    def __init__(self, stream: IO[str]):
        self.stream = stream

    def __call__(self, event: dict[str, Any]) -> None:
        self.stream.write(
            json.dumps({**event, "t": round(time.time(), 3)}, default=str) + "\n"
        )
        self.stream.flush()


def main(argv: list[str] | None = None, llm: LLMClient | None = None) -> None:
    from .env import load_env
    from .llm import LLM_CHOICES, make_client
    from .sources import SyntheticSource

    load_env()

    p = argparse.ArgumentParser(description="Investigate one change request.")
    p.add_argument("--db", default="data/org.db")
    p.add_argument("--identity", required=True)
    p.add_argument("--request-id", default="REQ-CLI")
    p.add_argument(
        "--events", help="write JSON-lines events to this file, or - for stdout"
    )
    p.add_argument(
        "--llm",
        choices=LLM_CHOICES,
        default="bedrock",
        help="openai = any OpenAI-compatible endpoint, Ollama by default; openrouter = hosted open models",
    )
    p.add_argument(
        "--model",
        help="model id; default from CIA_BEDROCK_MODEL_ID or CIA_OPENAI_MODEL",
    )
    a = p.parse_args(argv)
    out = None
    if a.events:
        out = sys.stdout if a.events == "-" else open(a.events, "w")  # noqa: SIM115
    try:
        if llm is None:
            llm = make_client(a.llm, a.model)
        sink = JsonlEvents(out) if out else None
        if sink:
            sink(
                {
                    "type": "run",
                    "run_id": a.request_id,
                    "llm": a.llm,
                    "model": getattr(llm, "model", None)
                    or getattr(llm, "model_id", None),
                }
            )
        source = SyntheticSource(a.db)
        inv = Investigator(
            source, llm, events=sink, clock=lambda: source.now
        )  # the org's simulated clock
        report = inv.investigate(
            ChangeRequest(request_id=a.request_id, identity_id=a.identity)
        )
    finally:
        if out not in (None, sys.stdout):
            out.close()
    if a.events != "-":
        print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
