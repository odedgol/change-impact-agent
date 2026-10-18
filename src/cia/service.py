"""The agent as a service: an HR event starts it, a message tells people, a person approves.

    POST /events/offboarding     {event_id, identity_id, source?, at?}: a person left.
                                 202 scheduled, 200 duplicate (same event_id), 401 bad
                                 signature, 404 unknown identity
    POST /changes/<REQ>/answer   {approved: [...], by}: a person's answer; [] declines
    GET  /changes                every change the service started, with its status
    GET  /health

Nobody presses Start. ``CIA_START_DELAY`` after the event (60 s; a few seconds in the
demo) the change workflow starts on its own: investigate, guardrails, then a message
(``cia.notify``) with what the agent found and what may be approved. The answer comes
back through ``/changes/<REQ>/answer`` (Slack buttons call the same ``Agent.answer``).
After a disable, the quiet period (``CIA_QUIET_PERIOD``: 14 days; minutes in the demo)
runs on the wall clock; when it ends the workflow checks again and deletes, keeps or
restores, and the message says so.

The workflow, guardrails, executor and audit log are the same ones the CLI and the
monitor use (``cia.workflow``); this module only adds the edges: intake, timers and
messages. Every event is appended to ``data/events.jsonl``, so a restart picks up
pending starts and quiet periods. Binds to 127.0.0.1. When ``CIA_EVENT_SECRET`` is set,
events must carry ``X-CIA-Signature: sha256=<HMAC of the body>``.

    python -m cia.service serve                 # make agent
    python -m cia.service event u00034          # make hr-event ID=u00034
    python -m cia.service approve OFF-... --actions disable --by dana
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from .env import load_env
from .generator import offboarding_text
from .investigator import JsonlEvents
from .models import ChangeRequest, OffboardingEvent
from .notify import ConsoleNotifier, Notifier, message
from .sources import SyntheticSource
from .workflow import DATA, ChangeWorkflow, events_file, open_workflow

PORT = 8790
SIGNATURE_HEADER = "X-CIA-Signature"
UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def duration(text: str) -> timedelta:
    """'90s', '2m', '1h', '14d' or plain seconds."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*", text)
    if not m:
        raise ValueError(f"not a duration: {text!r} (try 90s, 2m, 14d)")
    return timedelta(seconds=float(m.group(1)) * UNITS[m.group(2) or "s"])


def signature(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class UnknownIdentity(ValueError):
    pass


class Answer(BaseModel):
    approved: list[str] = Field(default_factory=list)
    by: str = Field(pattern=r"^[A-Za-z0-9_.@ -]{1,60}$")


class Agent:
    """Intake, timers and messages around the change workflow. One change at a time."""

    def __init__(
        self,
        db: str | Path = "data/org.db",
        data: str | Path = DATA,
        runs: str | Path = "runs",
        notifier: Notifier | None = None,
        llm: str = "fake",
        model: str | None = None,
        start_delay: timedelta = timedelta(seconds=60),
        quiet_period: timedelta = timedelta(days=14),
        monitor_url: str | None = None,
        now: Callable[[], float] = time.time,
    ):
        self.db, self.data, self.runs = str(db), Path(data), Path(runs)
        self.notifier = notifier or ConsoleNotifier()
        self.llm, self.model = llm, model
        self.start_delay, self.quiet_period = start_delay, quiet_period
        self.monitor_url = monitor_url
        self.now = now
        self.log = self.data / "events.jsonl"
        self.lock = (
            threading.Lock()
        )  # one workflow step at a time: one target, one audit chain
        # rebuilt from the log on start
        self.seen: set[str] = set()
        self.pending: dict[str, tuple[OffboardingEvent, float]] = {}
        self.quiet: dict[str, float] = {}
        self.refs: dict[str, str] = {}
        self.identities: dict[str, str] = {}
        self.started: list[str] = []
        self._replay()

    def identity(self, identity_id: str) -> dict[str, Any] | None:
        """A fresh read-only connection per call: requests arrive on many threads."""
        return SyntheticSource(self.db).get_identity(identity_id)

    # -- the log: what happened, so a restart carries on --------------------------------------

    def _append(self, kind: str, **fields: Any) -> None:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("a") as f:
            f.write(json.dumps({"kind": kind, "t": self.now(), **fields}) + "\n")

    def _replay(self) -> None:
        if not self.log.exists():
            return
        for line in self.log.read_text().splitlines():
            e = json.loads(line)
            if e["kind"] == "received":
                ev = OffboardingEvent.model_validate(e["event"])
                self.seen.add(ev.event_id)
                self.pending[ev.request_id()] = (ev, e["due"])
                self.identities[ev.request_id()] = ev.identity_id
            elif e["kind"] == "started":
                self.pending.pop(e["request_id"], None)
                self.started.append(e["request_id"])
                self.refs[e["request_id"]] = e["ref"]
            elif e["kind"] == "quiet":
                self.quiet[e["request_id"]] = e["due"]
            elif e["kind"] == "finished":
                self.quiet.pop(e["request_id"], None)

    # -- the workflow, with this request's trace file -------------------------------------------

    @contextmanager
    def _flow(self, request_id: str) -> Generator[tuple[ChangeWorkflow, JsonlEvents]]:
        path = events_file(self.runs, request_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            sink = JsonlEvents(f)
            yield (
                open_workflow(self.db, self.data, self.llm, self.model, events=sink),
                sink,
            )

    def _link(self, request_id: str) -> str | None:
        if not self.monitor_url:
            return None
        return f"{self.monitor_url.rstrip('/')}/#run={events_file('', request_id).stem}"

    def _message(self, flow: ChangeWorkflow, request_id: str):
        status = flow.status(request_id)
        identity_id = self.identities.get(request_id)
        shown = {**status, "identity_id": identity_id}
        if request_id in self.quiet:  # the quiet period runs on the wall clock here
            shown["until"] = datetime.fromtimestamp(self.quiet[request_id]).isoformat(
                sep=" ", timespec="minutes"
            )
        return status, message(
            shown,
            flow.report(request_id),
            self.identity(identity_id) if identity_id else None,
            self._link(request_id),
        )

    # -- what the API calls ------------------------------------------------------------------------

    def receive(self, event: OffboardingEvent) -> dict[str, Any]:
        """Schedule the investigation; the same event twice is one change."""
        rid = event.request_id()
        with self.lock:
            if event.event_id in self.seen:
                return {"status": "duplicate", "request_id": rid}
            if self.identity(event.identity_id) is None:
                raise UnknownIdentity(f"unknown identity {event.identity_id}")
            due = self.now() + self.start_delay.total_seconds()
            self.seen.add(event.event_id)
            self.pending[rid] = (event, due)
            self.identities[rid] = event.identity_id
            self._append("received", event=event.model_dump(mode="json"), due=due)
        return {
            "status": "scheduled",
            "request_id": rid,
            "starts_in": self.start_delay.total_seconds(),
        }

    def tick(self) -> list[str]:
        """Start what is due, end quiet periods that are over. Returns what it did."""
        done = []
        now = self.now()
        for rid, (event, due) in list(self.pending.items()):
            if due <= now:
                self._start(event)
                done.append(f"started {rid}")
        for rid, due in list(self.quiet.items()):
            if due <= now:
                self._resume(rid)
                done.append(f"resumed {rid}")
        return done

    def _start(self, event: OffboardingEvent) -> None:
        rid = event.request_id()
        identity = self.identity(event.identity_id) or {}
        request = ChangeRequest(
            request_id=rid,
            identity_id=event.identity_id,
            description=offboarding_text(
                identity.get("name", event.identity_id), identity.get("department")
            ),
        )
        with self.lock, self._flow(rid) as (flow, sink):
            sink(
                {
                    "type": "run",
                    "run_id": events_file("", rid).stem,
                    "llm": self.llm,
                    "model": self.model or self.llm,
                    "source": event.source,
                }
            )
            flow.start(request)
            _, msg = self._message(flow, rid)
            ref = self.notifier.post(msg)
            self.pending.pop(rid, None)
            self.started.append(rid)
            self.refs[rid] = ref
            self._append("started", request_id=rid, ref=ref)

    def answer(self, request_id: str, approved: list[str], by: str) -> dict[str, Any]:
        """A person's answer (an empty list declines). Raises if the change does not wait for one."""
        with self.lock, self._flow(request_id) as (flow, _):
            status = flow.answer(request_id, approved, by)
            if status["waiting_for"] == "quiet_period":
                due = self.now() + self.quiet_period.total_seconds()
                self.quiet[request_id] = due
                self._append("quiet", request_id=request_id, due=due)
            elif status["outcome"]:
                self._append(
                    "finished", request_id=request_id, outcome=status["outcome"]
                )
            _, msg = self._message(flow, request_id)
            self.notifier.update(self.refs.get(request_id, request_id), msg)
            return status

    def _resume(self, request_id: str) -> None:
        with self.lock, self._flow(request_id) as (flow, _):
            # the org's clock is simulated: continue as if the whole quiet period had passed
            flow.resume_after(request_id, flow.quiet_period)
            status, msg = self._message(flow, request_id)
            self.notifier.update(self.refs.get(request_id, request_id), msg)
            self.quiet.pop(request_id, None)
            self._append("finished", request_id=request_id, outcome=status["outcome"])

    def changes(self) -> list[dict[str, Any]]:
        with self.lock:
            out = [
                {
                    "request_id": rid,
                    "identity_id": ev.identity_id,
                    "waiting_for": "start",
                    "starts_at": due,
                }
                for rid, (ev, due) in self.pending.items()
            ]
            if self.started:
                flow = open_workflow(self.db, self.data, self.llm, self.model)
                out += [flow.status(rid) for rid in self.started]
            return out


# ---- HTTP ------------------------------------------------------------------------------------------


def create_app(
    agent: Agent, secret: str | None = None, tick_seconds: float | None = 1.0
):
    """The FastAPI app. ``tick_seconds`` runs ``agent.tick`` in the background; None leaves
    it to the caller (tests)."""
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(tick_seconds or 1.0):
            try:
                agent.tick()
            except Exception as e:  # noqa: BLE001 - one failing change must not stop the timers
                print(f"cia.service: tick failed: {type(e).__name__}: {e}", flush=True)

    @asynccontextmanager
    async def lifespan(_app):
        thread = None
        if tick_seconds:
            thread = threading.Thread(target=loop, daemon=True)
            thread.start()
        yield
        stop.set()

    app = FastAPI(title="Change Impact Agent", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/events/offboarding")
    async def offboarding(request: Request):
        body = await request.body()
        if secret:
            given = request.headers.get(SIGNATURE_HEADER, "")
            if not hmac.compare_digest(given, signature(secret, body)):
                raise HTTPException(401, "bad or missing signature")
        try:
            event = OffboardingEvent.model_validate_json(body)
        except ValidationError as e:
            raise HTTPException(422, e.errors(include_url=False)) from e
        try:
            out = agent.receive(event)
        except UnknownIdentity as e:
            raise HTTPException(404, str(e)) from e
        return JSONResponse(
            out, status_code=202 if out["status"] == "scheduled" else 200
        )

    @app.post("/changes/{request_id}/answer")
    def answer(request_id: str, body: Answer) -> dict[str, Any]:
        try:
            return agent.answer(request_id, body.approved, body.by)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except ValueError as e:  # not waiting for an answer, or more than allowed
            raise HTTPException(409, str(e)) from e

    @app.get("/changes")
    def changes() -> list[dict[str, Any]]:
        return agent.changes()

    return app


# ---- command line ------------------------------------------------------------------------------------


def agent_from_env(db: str, data: str, runs: str) -> Agent:
    return Agent(
        db,
        data,
        runs,
        llm=os.environ.get("CIA_AGENT_LLM", "fake"),
        model=os.environ.get("CIA_AGENT_MODEL") or None,
        start_delay=duration(os.environ.get("CIA_START_DELAY", "60s")),
        quiet_period=duration(os.environ.get("CIA_QUIET_PERIOD", "14d")),
        monitor_url=os.environ.get("CIA_MONITOR_URL", "http://127.0.0.1:8787"),
    )


def _post(url: str, body: dict[str, Any], secret: str | None) -> tuple[int, Any]:
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if secret:
        headers[SIGNATURE_HEADER] = signature(secret, data)
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310 - our own local service
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="The agent as a service: HR events in, messages out."
    )
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=PORT)
    s.add_argument("--db", default="data/org.db")
    s.add_argument("--data", default=str(DATA))
    s.add_argument("--runs", default="runs")
    e = sub.add_parser("event", help="send a synthetic HR 'person left' event")
    e.add_argument("identity_id", nargs="?")
    e.add_argument(
        "--case", help="the first identity of a planted case type, e.g. decoy_safe"
    )
    e.add_argument("--truth", default="bench/ground_truth.json")
    e.add_argument("--event-id", default=None)
    for name in ("approve", "decline"):
        a = sub.add_parser(name)
        a.add_argument("request_id")
        a.add_argument("--by", default=os.environ.get("USER", "someone"))
        if name == "approve":
            a.add_argument("--actions", default="disable,delete")
    sub.add_parser("list")
    a = p.parse_args(argv)
    load_env()
    url = os.environ.get("CIA_AGENT_URL", f"http://127.0.0.1:{PORT}")
    secret = os.environ.get("CIA_EVENT_SECRET") or None
    if a.command == "serve":
        import uvicorn

        agent = agent_from_env(a.db, a.data, a.runs)
        print(
            f"agent on http://127.0.0.1:{a.port} · model {agent.llm}{':' + agent.model if agent.model else ''}"
            f" · starts {agent.start_delay.total_seconds():g}s after an event · quiet period {agent.quiet_period}",
            flush=True,
        )
        uvicorn.run(
            create_app(agent, secret),
            host="127.0.0.1",
            port=a.port,
            log_level="warning",
        )
        return
    if a.command == "event":
        if a.case:
            with open(a.truth) as t:
                cases = sorted(json.load(t)["cases"], key=lambda c: c["request_id"])
            a.identity_id = next(
                (c["identity_id"] for c in cases if c["case_type"] == a.case), None
            )
        if not a.identity_id:
            p.error("event needs an identity id or a known --case")
        body = {
            "event_id": a.event_id or uuid.uuid4().hex[:12],
            "identity_id": a.identity_id,
            "source": "hr-demo",
        }
        out = _post(f"{url}/events/offboarding", body, secret)
    elif a.command in ("approve", "decline"):
        actions = (
            [x for x in a.actions.split(",") if x] if a.command == "approve" else []
        )
        out = _post(
            f"{url}/changes/{a.request_id}/answer",
            {"approved": actions, "by": a.by},
            None,
        )
    else:
        with urllib.request.urlopen(f"{url}/changes", timeout=10) as r:  # noqa: S310
            out = (r.status, json.loads(r.read()))
    print(json.dumps(out[1], indent=2))
    if out[0] >= 400:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
