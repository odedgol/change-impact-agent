"""The investigator: a tool-calling loop that turns a change request into a RiskReport.

The model only proposes. Code owns everything that makes the report trustworthy:

- Evidence ids. Every tool result is logged as ``ev-<tool>-<n>`` by this loop and
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
import inspect
import json
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import IO, Any

from pydantic import ValidationError

from .llm import AssistantTurn, LLMClient, LLMTurn, Message, ToolCall, ToolResult, ToolResults, ToolSpec, UserText
from .models import ChangeRequest, Evidence, Finding, RiskReport
from .sources import EvidenceSource, paginate

MAX_STEPS = 8
SUBMIT = "submit_report"
# A delete needs all three read in full; break-glass and shared accounts show up only in tickets.
REQUIRED_FOR_DELETE = ("get_usage", "get_dependencies", "get_tickets")

SYSTEM_PROMPT = """You investigate whether an IAM identity can be safely deleted.
Use the tools to gather evidence: usage, dependencies (including consumers of accounts it owns), tickets and the owner.
Dormant is not the same as unused: break-glass accounts, shared logins, scheduled jobs and policy exceptions look idle.
List tools return pages; if next_offset is not null, there are more rows.
When done, call submit_report once. Verdicts:
- safe_to_delete: nothing depends on the identity.
- disable_first: something may depend on it; disabling is reversible, deleting is not.
- ask_owner: a person must decide (an owned service account, a shared or break-glass account, unclear evidence).
- insufficient_evidence: you could not check what you needed (a tool failed, a source was unavailable).
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
                 "verdict": {"type": "string", "enum": ["safe_to_delete", "disable_first", "ask_owner", "insufficient_evidence"]},
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


_REPORT_FIELDS = {"verdict", "confidence", "findings"}  # request_id, rejected_citations, incomplete_reason: code


class _CapReached(Exception):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Investigator:
    def __init__(self, source: EvidenceSource, llm: LLMClient, max_steps: int = MAX_STEPS,
                 events: EventSink | None = None, clock: Callable[[], datetime] = _utcnow):
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

    def investigate(self, request: ChangeRequest) -> RiskReport:
        run = _Run(self, request)
        self.events({"type": "request", "request_id": request.request_id,
                     "identity_id": request.identity_id, "action": request.action})
        try:
            report = run.loop()
        except _CapReached:
            report = RiskReport(request_id=request.request_id, verdict="insufficient_evidence", confidence="low",
                                incomplete_reason=f"step cap reached after {run.steps} steps")
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
        self.read_ok: set[str] = set()  # tools that returned data at least once
        self.unread_pages: set[str] = set()  # paged tools whose last page the model never reached
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
                             "text": turn.text, "reasoning": turn.reasoning, "stop_reason": turn.stop_reason,
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
                    return self.finish(result)
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
        except Exception as e:  # noqa: BLE001 - an unavailable source is an error, never "no rows"
            return self.error(call, f"source unavailable for {call.name}: {type(e).__name__}: {e}")
        self.track_coverage(call.name, data)
        ev = Evidence(id=f"ev-{_EVIDENCE_PREFIX[call.name]}-{len(self.evidence) + 1}", tool=call.name,
                      args=dict(call.input), result=data, collected_at=self.inv.clock(),
                      source_name=self.inv.source_name, source_updated_at=self.inv.source_updated_at)
        self.evidence[ev.id] = ev
        self.inv.events({"type": "tool_call", "request_id": self.request.request_id, "step": self.steps,
                         "evidence_id": ev.id, "tool": call.name, "args": ev.args, "result": data,
                         "collected_at": ev.collected_at, "source_name": ev.source_name,
                         "source_updated_at": ev.source_updated_at})
        return ToolResult(call.id, {"evidence_id": ev.id, "data": data})

    def track_coverage(self, tool: str, data: Any) -> None:
        self.read_ok.add(tool)
        if isinstance(data, dict) and "next_offset" in data:
            if data["next_offset"] is None:
                self.unread_pages.discard(tool)
            else:
                self.unread_pages.add(tool)

    def unverified(self) -> list[str]:
        """What a delete would rest on but this run never read in full."""
        missing = [f"{t} not read" for t in REQUIRED_FOR_DELETE if t not in self.read_ok]
        return missing + [f"{t} has unread pages" for t in sorted(self.unread_pages)]

    def finish(self, report: RiskReport) -> RiskReport:
        """A model-reported or unverified result is insufficient_evidence, never safe_to_delete."""
        if report.verdict == "insufficient_evidence" and not report.incomplete_reason:
            return report.model_copy(update={"confidence": "low", "incomplete_reason": "reported by the model"})
        gaps = self.unverified() if report.verdict == "safe_to_delete" else []
        if gaps:
            return report.model_copy(update={"verdict": "insufficient_evidence", "confidence": "low",
                                             "incomplete_reason": "safe_to_delete not verified: " + "; ".join(gaps)})
        return report

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
    from .env import load_env
    from .llm import LLM_CHOICES, make_client
    from .sources import SyntheticSource

    load_env()

    p = argparse.ArgumentParser(description="Investigate one change request.")
    p.add_argument("--db", default="data/org.db")
    p.add_argument("--identity", required=True)
    p.add_argument("--request-id", default="REQ-CLI")
    p.add_argument("--events", help="write JSON-lines events to this file, or - for stdout")
    p.add_argument("--llm", choices=LLM_CHOICES, default="bedrock",
                   help="openai = any OpenAI-compatible endpoint, Ollama by default; openrouter = hosted open models")
    p.add_argument("--model", help="model id; default from CIA_BEDROCK_MODEL_ID or CIA_OPENAI_MODEL")
    a = p.parse_args(argv)
    out = None
    if a.events:
        out = sys.stdout if a.events == "-" else open(a.events, "w")  # noqa: SIM115
    try:
        if llm is None:
            llm = make_client(a.llm, a.model)
        sink = JsonlEvents(out) if out else None
        if sink:
            sink({"type": "run", "run_id": a.request_id, "llm": a.llm,
                  "model": getattr(llm, "model", None) or getattr(llm, "model_id", None)})
        source = SyntheticSource(a.db)
        inv = Investigator(source, llm, events=sink, clock=lambda: source.now)  # the org's simulated clock
        report = inv.investigate(ChangeRequest(request_id=a.request_id, identity_id=a.identity))
    finally:
        if out not in (None, sys.stdout):
            out.close()
    if a.events != "-":
        print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
