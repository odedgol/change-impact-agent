"""The change workflow: investigate, guard, ask a person, then act reversibly.

A second LangGraph ``StateGraph`` around the investigator (``cia.graph``):

    snapshot -> investigate -> guard -> approve -> verify -> dry_run -> disable -> quiet
       ^                         |         |         |                           ^  |
       |                         +---------+---------+--> stop                   +--+
       +--- evidence changed --------------------------+                            |
                                                         delete <- recheck <--------+
                                                        restore <-+  keep <-+

No cache: ``snapshot`` fingerprints fresh reads of the five tools before each
investigation. When the person answers, ``verify`` reads them again: unchanged, the
same report stands and the model is not called; changed, it investigates again and
asks again. ``recheck`` compares them once more after the quiet period: any change,
any use, or a failed read restores the account (``cia.snapshot``).

LangGraph decides the order; the rules are plain functions (``cia.guardrails``) and
the only writes go through ``cia.executor``. Two waits use ``interrupt()``: the
approval, and the quiet period after the disable. Nothing sleeps: a person, a
scheduler or ``resume`` continues the run, and the checkpointer keeps the state in
between, so a wait survives a restart (Q13). Every decision is in the audit log.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .audit import JsonlAuditSink
from .env import load_env
from .executor import Approval, Executor, SyntheticTarget
from .graph import STATE_TYPES
from .guardrails import (
    Decision,
    GuardrailError,
    Policy,
    allowed_actions,
    check_approval,
)
from .investigator import Investigator, JsonlEvents
from .llm import LLM_CHOICES
from .models import ChangeRequest, RiskReport
from .snapshot import changed, fingerprint
from .sources import SyntheticSource

QUIET_PERIOD = timedelta(days=14)  # Q14; any use while disabled ends it with a restore
MAX_RUNS = 3  # investigations per request; evidence that keeps changing needs a person

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
    used: bool  # the recheck found a reason to restore
    fingerprints: dict[str, str]  # what the current report rests on, per tool
    changed: list[str]  # tools whose evidence changed since the snapshot
    runs: int  # investigations so far
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
        "fingerprints": {},
        "changed": [],
        "runs": 0,
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

    def evidence_changes(state: FlowState) -> tuple[list[str], str | None]:
        """Fresh reads compared with the snapshot: (changed tools, or why we could not tell)."""
        try:
            now = fingerprint(flow.investigator.source, state["request"].identity_id)
        except Exception as e:  # noqa: BLE001 - could not check is never "unchanged"
            return [], f"could not re-read the evidence: {type(e).__name__}: {e}"
        return changed(state["fingerprints"], now), None

    def finished(outcome: Outcome, reason: str | None) -> Update:
        emit({"type": "outcome", "outcome": outcome, "reason": reason})
        return {"outcome": outcome, "reason": reason}

    # -- nodes

    def snapshot(state: FlowState) -> Update:
        """Fingerprint fresh reads before investigating, so a change during the run shows later."""
        if state["runs"] >= MAX_RUNS:
            return {
                "outcome": "failed",
                "reason": f"evidence kept changing; investigated {state['runs']} times",
            }
        try:
            fp = fingerprint(flow.investigator.source, state["request"].identity_id)
        except Exception as e:  # noqa: BLE001 - a source that cannot be read stops the run
            return {"outcome": "failed", "reason": f"could not read the evidence: {e}"}
        return {
            "fingerprints": fp,
            "changed": [],
            "runs": state["runs"] + 1,
            "allowed": [],
            "approved": [],
            "approver": None,
            "approved_at": None,
        }

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

    def verify(state: FlowState) -> Update:
        """Before acting, read fresh: unchanged keeps the report (no model call); changed
        investigates again and asks again; unreadable stops."""
        diff, problem = evidence_changes(state)
        if problem:
            ex.log(state["request"], "verify", "unreadable", reason=problem)
            return {"outcome": "failed", "reason": problem}
        if diff:
            ex.log(state["request"], "verify", "changed", tools=diff)
            emit({"type": "evidence_changed", "tools": diff})
            return {"changed": diff}
        assert state["report"] is not None and state["reported_at"] is not None
        decision = allowed_actions(  # the report-age backstop, at approval time
            state["request"],
            state["report"],
            datetime.fromisoformat(state["reported_at"]),
            flow.clock(),
            flow.policy,
        )
        ex.log(state["request"], "verify", "unchanged", reasons=list(decision.reasons))
        if not set(state["approved"]) <= set(decision.allowed):
            return {"outcome": "blocked", "reason": "; ".join(decision.reasons)}
        return {}

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
        """Check again: use since the disable, changed evidence, or a failed read restores."""
        identity = state["request"].identity_id
        since = datetime.fromisoformat(state["disabled_at"] or "")
        days = max(1, (datetime.fromisoformat(state["now"] or "") - since).days + 1)
        diff, problem = evidence_changes(state)
        if problem:
            outcome, reason = "unreadable", problem
        elif used_since(flow.investigator.source.get_usage(identity, days), since):
            outcome, reason = "used", "used while disabled"
        elif diff:
            outcome, reason = "changed", f"evidence changed: {', '.join(diff)}"
        else:
            outcome, reason = "quiet", None
        ex.log(state["request"], "recheck", outcome, since=state["disabled_at"])
        return {"used": reason is not None, "reason": reason}

    def delete(state: FlowState) -> Update:
        """Delete: only for safe_to_delete, only after the disable and a quiet period."""
        entry = ex.delete(state["request"], approval(state))
        emit({"type": "action", "action": "delete", "outcome": entry["outcome"]})
        return finished("deleted", None)

    def restore(state: FlowState) -> Update:
        """Undo the disable: something used the account, or the evidence changed."""
        entry = ex.restore(state["request"])
        emit({"type": "action", "action": "restore", "outcome": entry["outcome"]})
        return finished("restored", state["reason"])

    def keep(state: FlowState) -> Update:
        """Stays disabled: disable_first never deletes; a new run is needed first."""
        return finished("kept_disabled", None)

    def stop(state: FlowState) -> Update:
        """No change: blocked by the guardrails, declined, unreadable, or the plan failed."""
        return finished(state["outcome"] or "failed", state["reason"])

    # -- routes

    def after_guard(state: FlowState) -> str:
        return "approve" if state["allowed"] else "stop"

    def after_snapshot(state: FlowState) -> str:
        return "stop" if state["outcome"] else "investigate"

    def after_approve(state: FlowState) -> str:
        return "verify" if state["approved"] else "stop"

    def after_verify(state: FlowState) -> str:
        if state["outcome"]:
            return "stop"
        return "snapshot" if state["changed"] else "dry_run"

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
        snapshot,
        investigate,
        guard,
        approve,
        verify,
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
    g.add_edge(START, "snapshot")
    g.add_conditional_edges("snapshot", after_snapshot, ["investigate", "stop"])
    g.add_edge("investigate", "guard")
    g.add_conditional_edges("guard", after_guard, ["approve", "stop"])
    g.add_conditional_edges("approve", after_approve, ["verify", "stop"])
    g.add_conditional_edges("verify", after_verify, ["dry_run", "snapshot", "stop"])
    g.add_conditional_edges("dry_run", after_dry_run, ["disable", "stop"])
    g.add_edge("disable", "quiet")
    g.add_conditional_edges("quiet", after_quiet, ["recheck", "quiet"])
    g.add_conditional_edges("recheck", after_recheck, ["delete", "restore", "keep"])
    for end in ("delete", "restore", "keep", "stop"):
        g.add_edge(end, END)
    return g.compile(checkpointer=checkpointer)


# ---- the graph as data, for the drawing in docs/index.html#change-graph (same rules as graph.py)

NODE_KIND = {
    "snapshot": "tool",
    "investigate": "llm",
    "guard": "rule",
    "approve": "wait",
    "verify": "tool",
    "dry_run": "act",
    "disable": "act",
    "quiet": "wait",
    "recheck": "tool",
    "delete": "act",
    "restore": "act",
    "keep": "node",
    "stop": "node",
}
ROUTE_LABELS = {
    ("__start__", "snapshot"): "",
    ("snapshot", "investigate"): "read",
    ("snapshot", "stop"): "unreadable or 3 runs",
    ("investigate", "guard"): "report",
    ("guard", "approve"): "something allowed",
    ("guard", "stop"): "nothing allowed",
    ("approve", "verify"): "approved",
    ("approve", "stop"): "declined",
    ("verify", "dry_run"): "unchanged",
    ("verify", "snapshot"): "changed: again",
    ("verify", "stop"): "unreadable or too old",
    ("dry_run", "disable"): "plan ok",
    ("dry_run", "stop"): "plan fails",
    ("disable", "quiet"): "",
    ("quiet", "quiet"): "not yet",
    ("quiet", "recheck"): "14 days",
    ("recheck", "delete"): "quiet, delete approved",
    ("recheck", "restore"): "used or changed",
    ("recheck", "keep"): "quiet, disable only",
    ("delete", "__end__"): "",
    ("restore", "__end__"): "",
    ("keep", "__end__"): "",
    ("stop", "__end__"): "",
}


def describe() -> dict[str, Any]:
    """The change workflow as data. Compiled around stand-ins: drawing it calls nothing."""
    from types import SimpleNamespace

    from .graph import describe_graph

    stub = SimpleNamespace(executor=None, investigator=None, events=None)
    graph = build_workflow(stub, None)  # type: ignore[arg-type]
    return describe_graph(graph.get_graph(), NODE_KIND, ROUTE_LABELS)


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

    def resume_after(self, request_id: str, wait: timedelta) -> dict[str, Any]:
        """Continue the quiet period as if ``wait`` had passed since the disable (the demo's
        simulated clock does not move by itself)."""
        disabled = self.graph.get_state(self._config(request_id)).values.get(
            "disabled_at"
        )
        if not disabled:
            raise ValueError(f"{request_id} has not been disabled")
        return self.resume(request_id, datetime.fromisoformat(disabled) + wait)

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

    def report(self, request_id: str) -> RiskReport | None:
        """The latest report of this change (after a re-investigation, the new one)."""
        return self.graph.get_state(self._config(request_id)).values.get("report")

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


# ---- command line: the demo's change desk ----------------------------------------------------------

DATA = Path("data")


def open_workflow(
    db: str | Path = "data/org.db",
    data: str | Path = DATA,
    llm: str = "fake",
    model: str | None = None,
    events: Callable[[dict[str, Any]], None] | None = None,
) -> ChangeWorkflow:
    """The workflow on the synthetic org: its own writable copy (``target.db``), the audit
    log and the checkpoints all live in ``data/``. Time is the org's simulated clock."""
    from .bench import (
        make_llm,
    )  # the rule model ("fake") or a real client; lazy: no cycle

    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    source = SyntheticSource(str(db))
    target_db = data / "target.db"
    target = (
        SyntheticTarget(target_db)
        if target_db.exists()
        else SyntheticTarget.copy_of(db, target_db)
    )

    def clock() -> datetime:
        return source.now

    inv = Investigator(source, make_llm(llm, model), events=events, clock=clock)
    ex = Executor(target, JsonlAuditSink(data / "audit.jsonl"), clock=clock)
    return ChangeWorkflow(
        inv, ex, sqlite_checkpointer(data / "checkpoints.db"), clock=clock
    )


def changes(data: str | Path = DATA) -> list[str]:
    """Every request the change desk has seen, in the order it first appeared."""
    seen: dict[str, None] = {}
    for e in JsonlAuditSink(Path(data) / "audit.jsonl").entries():
        seen.setdefault(e["request_id"], None)
    return list(seen)


def events_file(runs: str | Path, request_id: str) -> Path:
    return Path(runs) / f"change-{request_id}.jsonl"


def reset(data: str | Path = DATA) -> list[str]:
    """Forget every change: the target copy, checkpoints and audit log. The org is untouched."""
    gone = []
    for name in ("target.db", "checkpoints.db", "audit.jsonl"):
        p = Path(data) / name
        if p.exists():
            p.unlink()
            gone.append(str(p))
    return gone


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Change desk: start a change, answer it, end its quiet period."
    )
    p.add_argument(
        "command", choices=["start", "approve", "decline", "resume", "status", "reset"]
    )
    p.add_argument("request_id", nargs="?")
    p.add_argument("--actions", default="disable,delete", help="approve: what to allow")
    p.add_argument("--by", default=os.environ.get("USER", "someone"))
    p.add_argument(
        "--days", type=float, default=14, help="resume: days after the disable"
    )
    p.add_argument("--llm", default="fake", choices=["fake", *LLM_CHOICES])
    p.add_argument("--model")
    p.add_argument("--db", default="data/org.db")
    p.add_argument("--truth", default="bench/ground_truth.json")
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--runs", default="runs")
    a = p.parse_args(argv)
    load_env()
    if a.command == "reset":
        print(json.dumps({"removed": reset(a.data)}))
        return
    if a.command == "status" and not a.request_id:
        flow = open_workflow(a.db, a.data)
        for rid in changes(a.data):
            print(json.dumps(flow.status(rid)))
        return
    if not a.request_id:
        p.error(f"{a.command} needs a request id, such as REQ-0013")
    try:
        out = run_command(
            a.command,
            a.request_id,
            db=a.db,
            truth=a.truth,
            data=a.data,
            runs=a.runs,
            llm=a.llm,
            model=a.model,
            actions=[x for x in a.actions.split(",") if x],
            by=a.by,
            days=a.days,
        )
    except (ValueError, KeyError) as e:
        raise SystemExit(f"cia.workflow: {e}") from e
    print(json.dumps(out))


def run_command(
    command: str,
    request_id: str,
    *,
    db: str | Path = "data/org.db",
    truth: str | Path = "bench/ground_truth.json",
    data: str | Path = DATA,
    runs: str | Path = "runs",
    llm: str = "fake",
    model: str | None = None,
    actions: list[str] | None = None,
    by: str = "someone",
    days: float = 14,
) -> dict[str, Any]:
    """One change-desk command; its events go to ``runs/change-<request>.jsonl``.
    Shared by ``make change`` and the monitor."""
    from .bench import model_name

    path = events_file(runs, request_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        sink = JsonlEvents(f)
        flow = open_workflow(db, data, llm, model, events=sink)
        if command == "start":
            with open(truth) as t:
                cases = json.load(t)["cases"]
            case = next((c for c in cases if c["request_id"] == request_id), None)
            if case is None:
                raise ValueError(f"unknown request {request_id}")
            sink(
                {
                    "type": "run",
                    "run_id": path.stem,
                    "llm": llm,
                    "model": model_name(flow.investigator.llm, llm),
                    "case_type": case["case_type"],
                    "expected_verdict": case["expected_verdict"],
                }
            )
            return flow.start(ChangeRequest.from_case(case))
        if command == "approve":
            return flow.answer(request_id, actions or [], by)
        if command == "decline":
            return flow.answer(request_id, [], by)
        if command == "resume":
            return flow.resume_after(request_id, timedelta(days=days))
        if command == "status":
            return flow.status(request_id)
        raise ValueError(f"unknown command {command!r}")


if __name__ == "__main__":
    main()
