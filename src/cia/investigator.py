"""The investigator: a tool-calling loop that turns a change request into a RiskReport.

The model only proposes. Code owns everything that makes the report trustworthy:

- Evidence ids. Every tool result is logged as ``ev-<tool>-<n>`` by this loop and
  handed to the model with the data; the model can cite ids, never invent them.
- The step cap. At most ``max_steps`` tool calls (``submit_report`` and a turn
  with no tool call count too). Hitting the cap returns ``ask_owner`` with low
  confidence, never a delete.
- The citation check. A report citing an unknown id goes back once with the
  rejected ids (the retry counts against the cap). After that, findings with
  unknown ids are dropped, the unknown ids are counted in
  ``rejected_citations``, and ``safe_to_delete`` with any rejection becomes
  ``ask_owner``.
- Paging. List tools return 50-row pages with ``next_offset``; nothing is cut.

``events`` receives one dict per step (request, model turn, tool call, tool error,
citation retry, step cap, report); ``JsonlEvents`` writes them as JSON lines for the terminal UI.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from collections.abc import Callable
from typing import IO, Any

from pydantic import ValidationError

from .llm import AssistantTurn, LLMClient, LLMTurn, Message, ToolCall, ToolResult, ToolResults, ToolSpec, UserText
from .models import ChangeRequest, Evidence, Finding, RiskReport
from .sources import EvidenceSource, paginate

MAX_STEPS = 8
SUBMIT = "submit_report"

SYSTEM_PROMPT = """You investigate whether an IAM identity can be safely deleted.
Use the tools to gather evidence: usage, dependencies (including consumers of accounts it owns), tickets and the owner.
Dormant is not the same as unused: break-glass accounts, shared logins, scheduled jobs and policy exceptions look idle.
List tools return pages; if next_offset is not null, there are more rows.
When done, call submit_report once. Verdicts:
- safe_to_delete: nothing depends on the identity.
- disable_first: something may depend on it; disabling is reversible, deleting is not.
- ask_owner: a person must decide (an owned service account, a shared or break-glass account, unclear evidence).
Every finding must cite the evidence_id values returned by the tools. Never cite an id you were not given."""

EventSink = Callable[[dict[str, Any]], None]

_ID = {"type": "string", "description": "identity id, for example u00042"}
_OFFSET = {"type": "integer", "minimum": 0, "description": "row offset of the page, from next_offset"}

TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec("get_identity", "The identity's record: kind, department, HR status, account status, owner id.",
             {"type": "object", "properties": {"identity_id": _ID}, "required": ["identity_id"]}),
    ToolSpec("get_usage", "Usage events in a window (at least 90 days): count, last seen, top callers.",
             {"type": "object", "properties": {"identity_id": _ID, "days": {"type": "integer", "minimum": 1}},
              "required": ["identity_id"]}),
    ToolSpec("get_dependencies", "Things that point at the identity, plus consumers of accounts it owns. Paged.",
             {"type": "object", "properties": {"identity_id": _ID, "offset": _OFFSET}, "required": ["identity_id"]}),
    ToolSpec("get_tickets", "Tickets and notes about the identity, newest first. Paged.",
             {"type": "object", "properties": {"identity_id": _ID, "offset": _OFFSET}, "required": ["identity_id"]}),
    ToolSpec("get_owner", "The identity's owner record, if it has one.",
             {"type": "object", "properties": {"identity_id": _ID}, "required": ["identity_id"]}),
    ToolSpec(SUBMIT, "Submit the final risk report. Every finding cites evidence ids from the tool results.",
             {"type": "object", "required": ["verdict", "confidence", "findings"], "properties": {
                 "verdict": {"type": "string", "enum": ["safe_to_delete", "disable_first", "ask_owner"]},
                 "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                 "findings": {"type": "array", "items": {
                     "type": "object", "required": ["kind", "summary", "evidence_ids", "confidence"],
                     "properties": {"kind": {"type": "string"}, "summary": {"type": "string"},
                                    "evidence_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]}}}}}}),
)

# Short names keep evidence ids readable: ev-deps-3, ev-usage-1.
_EVIDENCE_PREFIX = {"get_identity": "identity", "get_usage": "usage", "get_dependencies": "deps",
                    "get_tickets": "tickets", "get_owner": "owner"}


_REPORT_FIELDS = {"verdict", "confidence", "findings"}


class _CapReached(Exception):
    pass


class Investigator:
    def __init__(self, source: EvidenceSource, llm: LLMClient, max_steps: int = MAX_STEPS,
                 events: EventSink | None = None):
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1 (got {max_steps})")
        self.source = source
        self.llm = llm
        self.max_steps = max_steps
        self.events = events or (lambda event: None)

    def investigate(self, request: ChangeRequest) -> RiskReport:
        run = _Run(self, request)
        self.events({"type": "request", "request_id": request.request_id,
                     "identity_id": request.identity_id, "action": request.action})
        try:
            report = run.loop()
        except _CapReached:
            report = RiskReport(request_id=request.request_id, verdict="ask_owner", confidence="low")
            self.events({"type": "step_cap", "request_id": request.request_id, "steps": run.steps})
        self.events({"type": "report", **report.model_dump(), "steps": run.steps})
        return report


class _Run:
    """State of one investigation: messages, the evidence log, steps used, whether the retry is spent."""

    def __init__(self, inv: Investigator, request: ChangeRequest):
        self.inv = inv
        self.request = request
        self.messages: list[Message] = [UserText(
            f"Change request {request.request_id}: {request.action} identity {request.identity_id}. "
            "Investigate, then call submit_report.")]
        self.evidence: dict[str, Evidence] = {}
        self.steps = 0
        self.retried = False

    def take_step(self) -> None:
        if self.steps >= self.inv.max_steps:
            raise _CapReached
        self.steps += 1

    def loop(self) -> RiskReport:
        while True:
            turn = self.inv.llm.converse(SYSTEM_PROMPT, self.messages, TOOLS)
            self.messages.append(AssistantTurn(turn))
            self.inv.events({"type": "model_turn", "request_id": self.request.request_id, "step": self.steps,
                             "text": turn.text, "reasoning": turn.reasoning,
                             "tool_calls": [{"name": c.name, "input": c.input} for c in turn.tool_calls]})
            report = self.handle(turn)
            if report is not None:
                return report

    def handle(self, turn: LLMTurn) -> RiskReport | None:
        if not turn.tool_calls:
            self.take_step()
            self.messages.append(UserText("Call a tool, or call submit_report with your verdict."))
            return None
        results = []
        for call in turn.tool_calls:
            self.take_step()
            if call.name == SUBMIT:
                result = self.submit(call)
                if isinstance(result, RiskReport):
                    return result
            else:
                result = self.dispatch(call)
            results.append(result)
        self.messages.append(ToolResults(tuple(results)))
        return None

    def dispatch(self, call: ToolCall) -> ToolResult:
        fn = {"get_identity": self.inv.source.get_identity, "get_usage": self.inv.source.get_usage,
              "get_dependencies": self._paged(self.inv.source.get_dependencies),
              "get_tickets": self._paged(self.inv.source.get_tickets),
              "get_owner": self.inv.source.get_owner}.get(call.name)
        if fn is None:
            return self.error(call, f"unknown tool {call.name!r}")
        try:
            inspect.signature(fn).bind(**call.input)
            data = fn(**call.input)
        except (TypeError, ValueError) as e:
            return self.error(call, f"bad arguments for {call.name}: {e}")
        ev = Evidence(id=f"ev-{_EVIDENCE_PREFIX[call.name]}-{len(self.evidence) + 1}", tool=call.name,
                      args=dict(call.input), result=data)
        self.evidence[ev.id] = ev
        self.inv.events({"type": "tool_call", "request_id": self.request.request_id, "step": self.steps,
                         "evidence_id": ev.id, "tool": call.name, "args": ev.args, "result": data})
        return ToolResult(call.id, {"evidence_id": ev.id, "data": data})

    @staticmethod
    def _paged(list_fn: Callable[[str], list[dict]]) -> Callable[..., dict]:
        def page(identity_id: str, offset: int = 0) -> dict:
            return paginate(list_fn(identity_id), offset=offset)
        return page

    def error(self, call: ToolCall, message: str) -> ToolResult:
        self.inv.events({"type": "tool_error", "request_id": self.request.request_id, "step": self.steps,
                         "tool": call.name, "error": message})
        return ToolResult(call.id, {"error": message}, is_error=True)

    def submit(self, call: ToolCall) -> RiskReport | ToolResult:
        """The accepted report, or a tool error to send back to the model."""
        extra = set(call.input) - _REPORT_FIELDS  # request_id and rejected_citations are set by code
        if extra:
            return self.error(call, f"invalid report: unexpected fields {sorted(extra)}")
        try:
            draft = RiskReport(request_id=self.request.request_id, **call.input)
        except ValidationError as e:
            return self.error(call, f"invalid report: {e}")
        unknown = sorted({i for f in draft.findings for i in f.evidence_ids if i not in self.evidence})
        if not unknown:
            return draft
        if not self.retried:
            self.retried = True
            self.inv.events({"type": "citation_retry", "request_id": self.request.request_id, "step": self.steps,
                             "rejected_ids": unknown})
            return ToolResult(call.id, {"error": "unknown evidence ids; cite only ids the tools returned",
                                        "rejected_ids": unknown}, is_error=True)
        return self.drop_bad_citations(draft, set(unknown))

    @staticmethod
    def drop_bad_citations(draft: RiskReport, unknown: set[str]) -> RiskReport:
        kept: list[Finding] = [f for f in draft.findings if not unknown & set(f.evidence_ids)]
        rejected = sum(1 for f in draft.findings for i in f.evidence_ids if i in unknown)
        verdict, confidence = draft.verdict, draft.confidence
        if verdict == "safe_to_delete":
            verdict, confidence = "ask_owner", "low"
        return draft.model_copy(update={"findings": kept, "rejected_citations": rejected,
                                        "verdict": verdict, "confidence": confidence})


class JsonlEvents:
    """Writes each event as one JSON line, stamped with wall-clock time ``t``, for the UIs or a replay."""

    def __init__(self, stream: IO[str]):
        self.stream = stream

    def __call__(self, event: dict[str, Any]) -> None:
        self.stream.write(json.dumps({**event, "t": round(time.time(), 3)}, default=str) + "\n")
        self.stream.flush()


def main(argv: list[str] | None = None, llm: LLMClient | None = None) -> None:
    from .llm import BedrockClient, OpenAICompatClient
    from .sources import SyntheticSource

    p = argparse.ArgumentParser(description="Investigate one change request.")
    p.add_argument("--db", default="data/org.db")
    p.add_argument("--identity", required=True)
    p.add_argument("--request-id", default="REQ-CLI")
    p.add_argument("--events", help="write JSON-lines events to this file, or - for stdout")
    p.add_argument("--llm", choices=["bedrock", "openai"], default="bedrock",
                   help="openai = any OpenAI-compatible endpoint, Ollama by default")
    p.add_argument("--model", help="model id; default from CIA_BEDROCK_MODEL_ID or CIA_OPENAI_MODEL")
    a = p.parse_args(argv)
    out = None
    if a.events:
        out = sys.stdout if a.events == "-" else open(a.events, "w")  # noqa: SIM115
    try:
        if llm is None:
            llm = OpenAICompatClient(model=a.model) if a.llm == "openai" else BedrockClient(model_id=a.model)
        sink = JsonlEvents(out) if out else None
        if sink:
            sink({"type": "run", "run_id": a.request_id, "llm": a.llm,
                  "model": getattr(llm, "model", None) or getattr(llm, "model_id", None)})
        inv = Investigator(SyntheticSource(a.db), llm, events=sink)
        report = inv.investigate(ChangeRequest(request_id=a.request_id, identity_id=a.identity))
    finally:
        if out not in (None, sys.stdout):
            out.close()
    if a.events != "-":
        print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
