"""The change workflow: investigate, guard, ask a person, then act reversibly.

A second LangGraph ``StateGraph`` around the investigator (``cia.graph``):

    investigate -> guard -> approve -> dry_run -> disable -> quiet -> recheck
                     |         |                               ^  |     |-> delete
                     +---------+--> stop                       +--+     |-> restore
                                                                        +-> keep

LangGraph decides the order; the rules are plain functions (``cia.guardrails``) and
the only writes go through ``cia.executor``. Two waits use ``interrupt()``: the
approval, and the quiet period after the disable. Nothing sleeps: a person, a
scheduler or ``resume`` continues the run, and the checkpointer keeps the state in
between, so a wait survives a restart (Q13). Every decision is in the audit log.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .executor import Approval, Executor
from .graph import STATE_TYPES
from .guardrails import (
    Decision,
    GuardrailError,
    Policy,
    allowed_actions,
    check_approval,
)
from .investigator import Investigator
from .models import ChangeRequest, RiskReport

QUIET_PERIOD = timedelta(days=14)  # Q14; any use while disabled ends it with a restore

Outcome = Literal[
    "blocked", "declined", "deleted", "restored", "kept_disabled", "failed"
]


class FlowState(TypedDict):
    request: ChangeRequest
    report: RiskReport | None
    reported_at: str | None  # ISO times: plain strings in every checkpoint
    allowed: list[str]
    approved: list[str]
    approver: str | None
    approved_at: str | None
    disabled_at: str | None
    now: str | None  # the time the quiet period was last resumed at
    used: bool
    outcome: Outcome | None
    reason: str | None


Update = dict[str, Any]


def initial_state(request: ChangeRequest) -> FlowState:
    return {
        "request": request,
        "report": None,
        "reported_at": None,
        "allowed": [],
        "approved": [],
        "approver": None,
        "approved_at": None,
        "disabled_at": None,
        "now": None,
        "used": False,
        "outcome": None,
        "reason": None,
    }


def used_since(usage: dict[str, Any], since: datetime) -> bool:
    """Any use after the disable means something still depends on the account."""
    last = usage.get("last_seen")
    return last is not None and datetime.fromisoformat(last) > since


def build_workflow(flow: ChangeWorkflow, checkpointer: Any = None):
    """Compile the change workflow around one ChangeWorkflow (investigator, executor, clock)."""
    ex = flow.executor

    def emit(event: dict[str, Any]) -> None:
        flow.events(event)

    def approval(state: FlowState) -> Approval:
        return Approval(
            request_id=state["request"].request_id,
            actions=tuple(state["approved"]),  # type: ignore[arg-type]
            by=state["approver"] or "",
            at=datetime.fromisoformat(state["approved_at"] or ""),
        )

    # -- nodes

    def investigate(state: FlowState) -> Update:
        """Run the investigator graph; the report is the only thing it hands on."""
        report = flow.investigator.investigate(state["request"])
        return {"report": report, "reported_at": flow.clock().isoformat()}

    def guard(state: FlowState) -> Update:
        """Code decides which actions the verdict allows (guardrails.py)."""
        assert state["report"] is not None and state["reported_at"] is not None
        decision = allowed_actions(
            state["request"],
            state["report"],
            datetime.fromisoformat(state["reported_at"]),
            flow.clock(),
            flow.policy,
        )
        emit(decision.as_event())
        ex.log(
            state["request"],
            "guardrail",
            "allowed" if decision.allowed else "blocked",
            allowed=list(decision.allowed),
            reasons=list(decision.reasons),
        )
        if not decision.allowed:
            return {
                "allowed": [],
                "outcome": "blocked",
                "reason": "; ".join(decision.reasons),
            }
        emit(
            {
                "type": "approval_requested",
                "request_id": state["request"].request_id,
                "verdict": state["report"].verdict,
                "allowed": list(decision.allowed),
            }
        )
        return {"allowed": list(decision.allowed)}

    def approve(state: FlowState) -> Update:
        """Wait for a person (interrupt). They may approve less than allowed, never more."""
        answer = interrupt(
            {"wait": "approval", "allowed": state["allowed"]}
        )  # resumes with {"approved": [...], "by": "..."}
        by = str(answer.get("by") or "unknown")
        try:
            approved = check_approval(
                Decision(tuple(state["allowed"])),  # type: ignore[arg-type]
                list(answer.get("approved", [])),
            )
        except GuardrailError as e:
            ex.log(state["request"], "approval", "refused", by=by, reason=str(e))
            emit({"type": "approval", "by": by, "approved": [], "refused": str(e)})
            return {"outcome": "declined", "reason": f"approval refused: {e}"}
        ex.log(
            state["request"],
            "approval",
            "approved" if approved else "declined",
            by=by,
            approved=list(approved),
        )
        emit({"type": "approval", "by": by, "approved": list(approved)})
        if not approved:
            return {
                "approver": by,
                "outcome": "declined",
                "reason": f"declined by {by}",
            }
        return {
            "approved": list(approved),
            "approver": by,
            "approved_at": flow.clock().isoformat(),
        }

    def dry_run(state: FlowState) -> Update:
        """Plan every approved step first; change nothing."""
        steps = ex.plan(state["request"], tuple(state["approved"]))  # type: ignore[arg-type]
        emit({"type": "action", "action": "plan", "steps": steps})
        bad = [s for s in steps if not s["ok"]]
        if bad:
            return {
                "outcome": "failed",
                "reason": f"cannot {bad[0]['action']}: account is {bad[0]['from']}",
            }
        return {}

    def disable(state: FlowState) -> Update:
        """Disable: reversible, and the start of the quiet period."""
        entry = ex.disable(state["request"], approval(state))
        emit({"type": "action", "action": "disable", "outcome": entry["outcome"]})
        return {"disabled_at": entry["at"]}

    def quiet(state: FlowState) -> Update:
        """Wait out the quiet period (interrupt); a scheduler resumes with the time."""
        until = datetime.fromisoformat(state["disabled_at"] or "") + flow.quiet_period
        answer = interrupt({"wait": "quiet_period", "until": until.isoformat()})
        return {"now": str(answer.get("now") or flow.clock().isoformat())}

    def recheck(state: FlowState) -> Update:
        """Check again: any use since the disable ends it with a restore."""
        identity = state["request"].identity_id
        since = datetime.fromisoformat(state["disabled_at"] or "")
        days = max(1, (datetime.fromisoformat(state["now"] or "") - since).days + 1)
        used = used_since(flow.investigator.source.get_usage(identity, days), since)
        ex.log(
            state["request"],
            "recheck",
            "used" if used else "quiet",
            since=state["disabled_at"],
        )
        return {"used": used}

    def delete(state: FlowState) -> Update:
        """Delete: only for safe_to_delete, only after the disable and a quiet period."""
        entry = ex.delete(state["request"], approval(state))
        emit({"type": "action", "action": "delete", "outcome": entry["outcome"]})
        return {"outcome": "deleted"}

    def restore(state: FlowState) -> Update:
        """Undo the disable: something used the account."""
        entry = ex.restore(state["request"])
        emit({"type": "action", "action": "restore", "outcome": entry["outcome"]})
        return {"outcome": "restored", "reason": "used while disabled"}

    def keep(state: FlowState) -> Update:
        """Stays disabled: disable_first never deletes; a new run is needed first."""
        return {"outcome": "kept_disabled"}

    def stop(state: FlowState) -> Update:
        """No change: blocked by the guardrails, declined, or the plan failed."""
        return {}

    # -- routes

    def after_guard(state: FlowState) -> str:
        return "approve" if state["allowed"] else "stop"

    def after_approve(state: FlowState) -> str:
        return "dry_run" if state["approved"] else "stop"

    def after_dry_run(state: FlowState) -> str:
        return "stop" if state["outcome"] == "failed" else "disable"

    def after_quiet(state: FlowState) -> str:
        until = datetime.fromisoformat(state["disabled_at"] or "") + flow.quiet_period
        return (
            "recheck"
            if datetime.fromisoformat(state["now"] or "") >= until
            else "quiet"
        )

    def after_recheck(state: FlowState) -> str:
        if state["used"]:
            return "restore"
        return "delete" if "delete" in state["approved"] else "keep"

    g = StateGraph(FlowState)
    for fn in (
        investigate,
        guard,
        approve,
        dry_run,
        disable,
        quiet,
        recheck,
        delete,
        restore,
        keep,
        stop,
    ):
        g.add_node(fn.__name__, fn)
    g.add_edge(START, "investigate")
    g.add_edge("investigate", "guard")
    g.add_conditional_edges("guard", after_guard, ["approve", "stop"])
    g.add_conditional_edges("approve", after_approve, ["dry_run", "stop"])
    g.add_conditional_edges("dry_run", after_dry_run, ["disable", "stop"])
    g.add_edge("disable", "quiet")
    g.add_conditional_edges("quiet", after_quiet, ["recheck", "quiet"])
    g.add_conditional_edges("recheck", after_recheck, ["delete", "restore", "keep"])
    for end in ("delete", "restore", "keep", "stop"):
        g.add_edge(end, END)
    return g.compile(checkpointer=checkpointer)


class ChangeWorkflow:
    """Starts, answers and resumes change requests; each one is a checkpointed thread."""

    def __init__(
        self,
        investigator: Investigator,
        executor: Executor,
        checkpointer: Any,
        policy: Policy | None = None,
        quiet_period: timedelta = QUIET_PERIOD,
        clock: Callable[[], datetime] | None = None,
        events: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.investigator, self.executor = investigator, executor
        self.policy = policy or Policy()
        self.quiet_period = quiet_period
        self.clock = clock or investigator.clock
        self.events = events or investigator.events
        self.graph = build_workflow(self, checkpointer)

    def _config(self, request_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": f"change-{request_id}"}}

    def start(self, request: ChangeRequest) -> dict[str, Any]:
        self.graph.invoke(initial_state(request), self._config(request.request_id))
        return self.status(request.request_id)

    def answer(self, request_id: str, approved: list[str], by: str) -> dict[str, Any]:
        """A person's answer to the approval wait; an empty list declines."""
        return self._resume(request_id, "approval", {"approved": approved, "by": by})

    def resume(self, request_id: str, now: datetime | None = None) -> dict[str, Any]:
        """Continue the quiet period at ``now`` (a scheduler, or ``make resume``)."""
        now = now or self.clock()
        return self._resume(request_id, "quiet_period", {"now": now.isoformat()})

    def _resume(
        self, request_id: str, wait: str, value: dict[str, Any]
    ) -> dict[str, Any]:
        current = self.status(request_id)
        if current["waiting_for"] != wait:
            raise ValueError(
                f"{request_id} is not waiting for {wait} (state: {current})"
            )
        self.graph.invoke(Command(resume=value), self._config(request_id))
        return self.status(request_id)

    def status(self, request_id: str) -> dict[str, Any]:
        snap = self.graph.get_state(self._config(request_id))
        if not snap.values:
            raise KeyError(f"no change workflow for {request_id}")
        waits = [i.value for t in snap.tasks for i in t.interrupts]
        values = snap.values
        report = values.get("report")
        return {
            "request_id": request_id,
            "verdict": report.verdict if report else None,
            "allowed": values.get("allowed", []),
            "approved": values.get("approved", []),
            "waiting_for": waits[0]["wait"] if waits else None,
            "until": waits[0].get("until") if waits else None,
            "outcome": values.get("outcome"),
            "reason": values.get("reason"),
        }


def sqlite_checkpointer(path: str | Path):
    """A checkpointer on disk, so an approval or a quiet period survives a restart (Q13).
    Same allow-listed serializer as the investigator; Postgres takes the same ``serde`` later."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(
        conn, serde=JsonPlusSerializer(allowed_msgpack_modules=STATE_TYPES)
    )
