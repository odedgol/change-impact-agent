"""The investigator as a LangGraph state graph.

    START → model ─┬─ no tool call ──→ nudge ─→ model
                   └─ tool calls ────→ tools ─┬─ more turns ─→ model
                                              └─ submit_report → check ─┬─ accepted → finish → END
                                                                        └─ sent back → tools
    any node that would take step max_steps + 1 → cap → END

The model only proposes. The nodes call small rule functions that own everything that
makes the report trustworthy: evidence ids, the step cap, the citation check with one
retry, and "incomplete is never safe". The graph keeps the order; the rules stay testable
on their own. ``python -m cia.graph`` prints the graph as Mermaid.
"""

from __future__ import annotations

import argparse
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from .llm import (
    AssistantTurn,
    LLMTurn,
    Message,
    ToolCall,
    ToolResult,
    ToolResults,
    UserText,
)
from .models import ChangeRequest, Evidence, Finding, RiskReport
from .sources import paginate

if TYPE_CHECKING:
    from .investigator import Investigator

SUBMIT = "submit_report"
# A delete needs all three read in full; break-glass and shared accounts show up only in tickets.
REQUIRED_FOR_DELETE = ("get_usage", "get_dependencies", "get_tickets")
REPORT_FIELDS = {
    "verdict",
    "confidence",
    "findings",
}  # request_id, rejected_citations, incomplete_reason, overridden: code
# Automated notices: every leaver gets one, and it says nothing about risk.
NOTICE_AUTHORS = ("hr-bot",)
NUDGE = "Call a tool, or call submit_report with your verdict."
# Short names keep evidence ids readable: ev-deps-3, ev-usage-1.
EVIDENCE_PREFIX = {
    "get_identity": "identity",
    "get_usage": "usage",
    "get_dependencies": "deps",
    "get_tickets": "tickets",
    "get_owner": "owner",
}


class RunState(TypedDict):
    """Everything one investigation knows. ``initial_state`` sets every key; nodes return an ``Update``."""

    request: ChangeRequest
    messages: list[Message]
    turn: LLMTurn | None  # the model's latest turn; None before the first one
    cursor: int  # next tool call of that turn to handle
    results: list[ToolResult]  # results of that turn so far, sent back together
    evidence: dict[str, Evidence]  # the evidence log: id -> what a tool returned
    read_ok: list[str]  # tools that returned data at least once
    unread_pages: list[str]  # paged tools whose last page the model never reached
    steps: int
    retried: bool  # the one citation retry is spent
    report: RiskReport | None


Update = dict[str, Any]  # what a node returns: only the RunState keys it changes


def latest_turn(state: RunState) -> LLMTurn:
    turn = state["turn"]
    assert turn is not None, "routed here before the model's first turn"
    return turn


# ---- rules: plain functions, no graph, tested on their own ------------------------------------------


def coverage(
    read_ok: list[str], unread: list[str], tool: str, data: Any
) -> tuple[list[str], list[str]]:
    """Which tools have been read, and which still have pages left, after ``tool`` returned ``data``."""
    read_ok = sorted({*read_ok, tool})
    if isinstance(data, dict) and "next_offset" in data:
        unread = (
            sorted({*unread, tool})
            if data["next_offset"] is not None
            else [t for t in unread if t != tool]
        )
    return read_ok, unread


def unverified(read_ok: list[str], unread: list[str]) -> list[str]:
    """What a delete would rest on but this run never read in full."""
    missing = [f"{t} not read" for t in REQUIRED_FOR_DELETE if t not in read_ok]
    return missing + [f"{t} has unread pages" for t in sorted(unread)]


def finish_report(
    report: RiskReport, read_ok: list[str], unread: list[str]
) -> RiskReport:
    """A model-reported or unverified result is insufficient_evidence, never safe_to_delete."""
    if report.verdict == "insufficient_evidence" and not report.incomplete_reason:
        return report.model_copy(
            update={"confidence": "low", "incomplete_reason": "reported by the model"}
        )
    gaps = unverified(read_ok, unread) if report.verdict == "safe_to_delete" else []
    if gaps:
        return report.model_copy(
            update={
                "verdict": "insufficient_evidence",
                "confidence": "low",
                "incomplete_reason": "safe_to_delete not verified: " + "; ".join(gaps),
            }
        )
    return report


def contradictions(evidence: dict[str, Evidence]) -> list[str]:
    """What in this run's own evidence log speaks against a delete."""
    found = []
    for ev in evidence.values():
        r = ev.result if isinstance(ev.result, dict) else {}
        if ev.tool == "get_usage" and r.get("event_count"):
            found.append(f"{ev.id} shows {r['event_count']} usage events")
        elif ev.tool == "get_dependencies" and r.get("total"):
            found.append(f"{ev.id} shows {r['total']} dependencies")
        elif ev.tool == "get_tickets":
            people = [
                t for t in r.get("rows", []) if t.get("author") not in NOTICE_AUTHORS
            ]
            if people:
                found.append(f"{ev.id} shows {len(people)} tickets from people")
    return found


def ground_safe(report: RiskReport, evidence: dict[str, Evidence]) -> RiskReport:
    """A safe_to_delete must agree with the evidence and cite usage, dependencies and
    tickets. Valid ids are not enough: they must support the verdict. Else ask_owner."""
    if report.verdict != "safe_to_delete":
        return report
    problems = contradictions(evidence)
    cited = {
        evidence[i].tool
        for f in report.findings
        for i in f.evidence_ids
        if i in evidence
    }
    missing = [t for t in REQUIRED_FOR_DELETE if t not in cited]
    if missing:
        problems.append("does not cite " + ", ".join(missing))
    if not problems:
        return report
    return report.model_copy(
        update={
            "verdict": "ask_owner",
            "confidence": "low",
            "overridden": "safe_to_delete not grounded: " + "; ".join(problems),
        }
    )


def drop_bad_citations(draft: RiskReport, unknown: set[str]) -> RiskReport:
    """After the retry: drop findings that cite unknown ids; a safe verdict with any rejection becomes ask_owner."""
    kept: list[Finding] = [
        f for f in draft.findings if not unknown & set(f.evidence_ids)
    ]
    rejected = sum(1 for f in draft.findings for i in f.evidence_ids if i in unknown)
    verdict, confidence = draft.verdict, draft.confidence
    if verdict == "safe_to_delete":
        verdict, confidence = "ask_owner", "low"
    return draft.model_copy(
        update={
            "findings": kept,
            "rejected_citations": rejected,
            "verdict": verdict,
            "confidence": confidence,
        }
    )


def cap_report(request: ChangeRequest, steps: int) -> RiskReport:
    return RiskReport(
        request_id=request.request_id,
        verdict="insufficient_evidence",
        confidence="low",
        incomplete_reason=f"step cap reached after {steps} steps",
    )


# ---- the graph -------------------------------------------------------------------------------------


def build_graph(inv: Investigator, checkpointer: Any = None):
    """Compile the investigator graph around one Investigator (its source, model, clock and event sink)."""
    from .investigator import SYSTEM_PROMPT, TOOLS

    def emit(state: RunState, event: dict[str, Any]) -> None:
        inv.events(
            {
                "type": event.pop("type"),
                "request_id": state["request"].request_id,
                **event,
            }
        )

    def error(state: RunState, step: int, call: ToolCall, message: str) -> ToolResult:
        emit(
            state,
            {"type": "tool_error", "step": step, "tool": call.name, "error": message},
        )
        return ToolResult(call.id, {"error": message}, is_error=True)

    def paged(list_fn: Callable[..., list[dict]]) -> Callable[..., dict]:
        def page(identity_id: str, offset: int = 0, **kwargs: Any) -> dict:
            return paginate(list_fn(identity_id, **kwargs), offset=offset)

        return page

    tools_by_name = {
        "get_identity": inv.source.get_identity,
        "get_usage": inv.source.get_usage,
        "get_dependencies": paged(inv.source.get_dependencies),
        "get_tickets": paged(inv.source.get_tickets),
        "get_owner": inv.source.get_owner,
    }

    # -- nodes

    def model(state: RunState) -> Update:
        """Ask the model for its next turn; it only proposes tool calls."""
        turn = inv.llm.converse(SYSTEM_PROMPT, state["messages"], TOOLS)
        emit(
            state,
            {
                "type": "model_turn",
                "step": state["steps"],
                "text": turn.text,
                "reasoning": turn.reasoning,
                "stop_reason": turn.stop_reason,
                "tool_calls": [
                    {"name": c.name, "input": c.input} for c in turn.tool_calls
                ],
            },
        )
        return {
            "turn": turn,
            "cursor": 0,
            "results": [],
            "messages": [*state["messages"], AssistantTurn(turn)],
        }

    def nudge(state: RunState) -> Update:
        """A turn with no tool call costs a step and gets a reminder."""
        return {
            "steps": state["steps"] + 1,
            "messages": [*state["messages"], UserText(NUDGE)],
        }

    def tools(state: RunState) -> Update:
        """Run this turn's read-only calls in order, up to a submit_report or the step cap."""
        calls, i = latest_turn(state).tool_calls, state["cursor"]
        steps, results, evidence = (
            state["steps"],
            list(state["results"]),
            dict(state["evidence"]),
        )
        read_ok, unread = state["read_ok"], state["unread_pages"]
        while i < len(calls) and calls[i].name != SUBMIT and steps < inv.max_steps:
            call, steps = calls[i], steps + 1
            result, ev = dispatch(state, steps, len(evidence), call)
            if ev is not None:
                evidence[ev.id] = ev
                read_ok, unread = coverage(read_ok, unread, call.name, ev.result)
            results.append(result)
            i += 1
        update: Update = {
            "cursor": i,
            "steps": steps,
            "results": results,
            "evidence": evidence,
            "read_ok": read_ok,
            "unread_pages": unread,
        }
        if i == len(calls):  # the turn is done: send every result back at once
            update.update(
                messages=[*state["messages"], ToolResults(tuple(results))], results=[]
            )
        return update

    def dispatch(
        state: RunState, step: int, n_evidence: int, call: ToolCall
    ) -> tuple[ToolResult, Evidence | None]:
        fn = tools_by_name.get(call.name)
        if fn is None:
            return error(state, step, call, f"unknown tool {call.name!r}"), None
        try:
            inspect.signature(fn).bind(**call.input)
            data = fn(**call.input)
        except (TypeError, ValueError) as e:
            return error(state, step, call, f"bad arguments for {call.name}: {e}"), None
        except Exception as e:  # noqa: BLE001 - an unavailable source is an error, never "no rows"
            return (
                error(
                    state,
                    step,
                    call,
                    f"source unavailable for {call.name}: {type(e).__name__}: {e}",
                ),
                None,
            )
        ev = Evidence(
            id=f"ev-{EVIDENCE_PREFIX[call.name]}-{n_evidence + 1}",
            tool=call.name,
            args=dict(call.input),
            result=data,
            collected_at=inv.clock(),
            source_name=inv.source_name,
            source_updated_at=inv.source_updated_at,
        )
        emit(
            state,
            {
                "type": "tool_call",
                "step": step,
                "evidence_id": ev.id,
                "tool": call.name,
                "args": ev.args,
                "result": data,
                "collected_at": ev.collected_at,
                "source_name": ev.source_name,
                "source_updated_at": ev.source_updated_at,
            },
        )
        return ToolResult(call.id, {"evidence_id": ev.id, "data": data}), ev

    def check(state: RunState) -> Update:
        """The citation check: accept the report, or send it back once with the unknown ids."""
        call, steps = latest_turn(state).tool_calls[state["cursor"]], state["steps"] + 1
        extra = set(call.input) - REPORT_FIELDS
        if extra:
            return send_back(
                state,
                steps,
                error(
                    state,
                    steps,
                    call,
                    f"invalid report: unexpected fields {sorted(extra)}",
                ),
            )
        try:
            draft = RiskReport(request_id=state["request"].request_id, **call.input)
        except ValidationError as e:
            return send_back(
                state, steps, error(state, steps, call, f"invalid report: {e}")
            )
        unknown = sorted(
            {
                i
                for f in draft.findings
                for i in f.evidence_ids
                if i not in state["evidence"]
            }
        )
        if not unknown:
            return {"steps": steps, "report": draft}
        if not state["retried"]:
            emit(
                state,
                {"type": "citation_retry", "step": steps, "rejected_ids": unknown},
            )
            back = ToolResult(
                call.id,
                {
                    "error": "unknown evidence ids; cite only ids the tools returned",
                    "rejected_ids": unknown,
                },
                is_error=True,
            )
            return {**send_back(state, steps, back), "retried": True}
        return {"steps": steps, "report": drop_bad_citations(draft, set(unknown))}

    def send_back(state: RunState, steps: int, result: ToolResult) -> Update:
        """A rejected report is one more tool result of this turn; the turn's other calls still run."""
        return {
            "steps": steps,
            "cursor": state["cursor"] + 1,
            "results": [*state["results"], result],
        }

    def finish(state: RunState) -> Update:
        """Incomplete is never safe, and safe must be grounded: an unverified safe_to_delete
        becomes insufficient_evidence; one its own evidence contradicts becomes ask_owner."""
        report = state["report"]
        assert report is not None, "check routes here only with a report"
        report = finish_report(report, state["read_ok"], state["unread_pages"])
        return {"report": ground_safe(report, state["evidence"])}

    def cap(state: RunState) -> Update:
        """The step cap: no step is left, so the run ends as insufficient_evidence, never a delete."""
        emit(state, {"type": "step_cap", "steps": state["steps"]})
        return {"report": cap_report(state["request"], state["steps"])}

    # -- routing: where each node goes next

    def after_model(state: RunState) -> str:
        if not latest_turn(state).tool_calls:
            return "cap" if state["steps"] >= inv.max_steps else "nudge"
        return "tools"

    def after_tools(state: RunState) -> str:
        calls, i = latest_turn(state).tool_calls, state["cursor"]
        if i == len(calls):
            return "model"
        if state["steps"] >= inv.max_steps:
            return "cap"
        return "check"  # calls[i] is submit_report

    def after_check(state: RunState) -> str:
        if state["report"] is not None:
            return "finish"
        return "tools"

    g = StateGraph(RunState)
    for name, fn in (
        ("model", model),
        ("nudge", nudge),
        ("tools", tools),
        ("check", check),
        ("finish", finish),
        ("cap", cap),
    ):
        g.add_node(name, fn)
    g.add_edge(START, "model")
    g.add_conditional_edges("model", after_model, ["nudge", "tools", "cap"])
    g.add_edge("nudge", "model")
    g.add_conditional_edges("tools", after_tools, ["model", "check", "cap"])
    g.add_conditional_edges("check", after_check, ["finish", "tools"])
    g.add_edge("finish", END)
    g.add_edge("cap", END)
    return g.compile(checkpointer=checkpointer)


def initial_state(request: ChangeRequest) -> RunState:
    return {
        "request": request,
        "turn": None,
        "steps": 0,
        "retried": False,
        "report": None,
        "evidence": {},
        "read_ok": [],
        "unread_pages": [],
        "cursor": 0,
        "results": [],
        "messages": [
            UserText(
                f"Change request {request.request_id}: {request.action} identity "
                f"{request.identity_id}. Investigate, then call submit_report."
            )
        ],
    }


# The only classes a checkpoint may hold; anything else is refused when a checkpoint is loaded.
STATE_TYPES = [
    ("cia.models", n) for n in ("ChangeRequest", "Evidence", "Finding", "RiskReport")
] + [
    ("cia.llm", n)
    for n in (
        "UserText",
        "ToolCall",
        "LLMTurn",
        "AssistantTurn",
        "ToolResult",
        "ToolResults",
    )
]


def memory_checkpointer():
    """An in-process checkpointer for our state types; a Postgres saver takes the same ``serde`` later."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=STATE_TYPES))


def recursion_limit(max_steps: int) -> int:
    """LangGraph's own backstop, well above our step cap: each step is at most three graph hops."""
    return 4 * max_steps + 10


def drawing_graph():
    """The compiled graph with a source and model that are never called, for drawing and describing it."""
    from .investigator import Investigator
    from .llm import FakeLLM

    class _NoSource:
        """Drawing the graph never calls a tool."""

        name = "none"

        def get_identity(self, identity_id: str) -> dict[str, Any] | None:
            raise NotImplementedError

        def get_usage(self, identity_id: str, days: int = 90) -> dict[str, Any]:
            raise NotImplementedError

        def get_dependencies(
            self, identity_id: str, depth: int = 2
        ) -> list[dict[str, Any]]:
            raise NotImplementedError

        def get_tickets(self, identity_id: str) -> list[dict[str, Any]]:
            raise NotImplementedError

        def get_owner(self, identity_id: str) -> dict[str, Any] | None:
            raise NotImplementedError

    return build_graph(Investigator(_NoSource(), FakeLLM([]))).get_graph()


def mermaid() -> str:
    return drawing_graph().draw_mermaid()


# ---- the graph as data, for the drawing in docs/index.html#engine ---------------------------------
# Nodes, edges and which edges are conditional come from LangGraph itself; each node's subtitle is its
# docstring. Only the colour of a node and the words on a route live here, and test_graph.py fails when
# a node or route has none, so the drawing always matches the code. `make graph` rewrites the docs.

NODE_KIND = {
    "model": "llm",
    "tools": "tool",
    "nudge": "node",
    "check": "rule",
    "finish": "rule",
    "cap": "rule",
}
ROUTE_LABELS = {
    ("__start__", "model"): "",
    ("model", "tools"): "tool calls",
    ("model", "nudge"): "no tool call",
    ("model", "cap"): "step cap",
    ("nudge", "model"): "reminder",
    ("tools", "model"): "results",
    ("tools", "check"): "submit_report",
    ("tools", "cap"): "step cap",
    ("check", "finish"): "accepted",
    ("check", "tools"): "sent back ×1",
    ("finish", "__end__"): "",
    ("cap", "__end__"): "",
}
DOCS_BEGIN, DOCS_END = (
    "<!-- graph:begin (written by `make graph` from src/cia/graph.py; do not edit) -->",
    "<!-- graph:end -->",
)


def describe() -> dict[str, Any]:
    """The compiled graph as plain data: nodes with kind and docstring, edges with route labels."""
    g = drawing_graph()
    nodes = []
    for node in g.nodes.values():
        fn = getattr(node.data, "func", None)
        doc = inspect.getdoc(fn) if fn is not None else None
        kind = (
            "start"
            if node.id == START
            else "end"
            if node.id == END
            else NODE_KIND.get(node.id, "")
        )
        nodes.append({"id": node.id, "kind": kind, "doc": (doc or "").split("\n")[0]})
    edges = [
        {
            "source": e.source,
            "target": e.target,
            "conditional": e.conditional,
            "label": ROUTE_LABELS.get((e.source, e.target), ""),
        }
        for e in g.edges
    ]
    return {"nodes": nodes, "edges": edges}


def docs_block(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    return f'{DOCS_BEGIN}\n<script type="application/json" id="engine-graph">{payload}</script>\n{DOCS_END}'


def write_docs(path: str | Path = "docs/index.html") -> bool:
    """Put the current graph into the docs page between the markers. Returns True if the page changed."""
    p = Path(path)
    page = p.read_text()
    start, end = page.index(DOCS_BEGIN), page.index(DOCS_END) + len(DOCS_END)
    new = page[:start] + docs_block(describe()) + page[end:]
    if new != page:
        p.write_text(new)
    return new != page


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Print the investigator graph, or write it into the docs."
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="print nodes and edges as JSON instead of Mermaid",
    )
    p.add_argument(
        "--write-docs",
        metavar="HTML",
        help="update the graph data embedded in this page",
    )
    a = p.parse_args(argv)
    if a.write_docs:
        print(
            f"{a.write_docs}: {'updated' if write_docs(a.write_docs) else 'already current'}"
        )
    elif a.json:
        print(json.dumps(describe(), ensure_ascii=False, indent=2))
    else:
        print(mermaid())


if __name__ == "__main__":
    main()
