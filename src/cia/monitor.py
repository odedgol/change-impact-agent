"""Local run monitor: launch investigations, watch them live, browse past runs.

Standard library only. Binds to 127.0.0.1. Each run is one JSON-lines file in
``runs/`` (the same stream ``--events`` writes and the Rust UI will read), so
runs started from the terminal show up here too.

    GET  /                       the page (monitor.html)
    GET  /api/cases              change requests from the ground truth, with the expected verdict
    GET  /api/models             local Ollama models, plus OpenRouter and Bedrock when configured
    GET  /api/runs               every run, newest first, summarised
    GET  /api/runs/<id>/events   the run's events as server-sent events; follows a live run to the end
    POST /api/runs               {"request_id", "model", "llm"} starts a run, returns {"run_id"}
    GET  /api/changes            the change desk: every change request and what it waits for
    GET  /api/baseline           what the naive "no use in 90 days" rule would break
    GET  /api/bench              the newest bench result (bench/results/*.json), per case
    GET  /api/audit              the change desk's audit log and whether its hash chain holds
    POST /api/changes/<REQ>/<command>
                                 start {"llm", "model"} | approve {"actions", "by"} |
                                 decline {"by"} | resume {"days"}; returns {"run_id"}

Investigations only read the org. The change desk acts on its own writable copy
(``data/target.db``), one command at a time, and only after a person approves.
Every POST must be JSON and, when the browser sends an Origin, come from this page.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import threading
import time
import urllib.request
from collections.abc import Callable
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .audit import JsonlAuditSink, verify
from .baseline import run as run_baseline
from .bench import make_llm
from .env import load_env
from .investigator import Investigator, JsonlEvents
from .llm import (
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    LLM_CHOICES,
    MODEL_ID_ENV,
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_MODEL_ENV,
    LLMClient,
)
from .models import ChangeRequest
from .sources import SyntheticSource
from .workflow import DATA, changes, events_file, open_workflow, run_command

PAGE = Path(__file__).with_name("monitor.html")
DEMO_MODEL_ENV = "CIA_DEMO_MODEL"
RUN_ID = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
MODEL_NAME = re.compile(r"^[A-Za-z0-9_.:/@-]{1,120}$")
PERSON = re.compile(r"^[A-Za-z0-9_.@ -]{1,60}$")
CHANGE_WAITS = {  # command -> what the change must be waiting for
    "start": None,
    "approve": "approval",
    "decline": "approval",
    "resume": "quiet_period",
}
ACTIONS = ("disable", "delete")
FOLLOW_SECONDS = 600  # how long one event stream follows a live run
POLL_SECONDS = 0.25

LLMFactory = Callable[[str, str], LLMClient]


class Conflict(Exception):
    """The request is valid, but the change is not in a state that allows it (HTTP 409)."""


def default_llm_factory(llm: str, model: str) -> LLMClient:
    return make_llm(llm, model or None)  # "fake" is the offline rule model


class Monitor:
    def __init__(
        self,
        db: str,
        truth: str,
        runs_dir: str = "runs",
        llm_factory: LLMFactory = default_llm_factory,
        data_dir: str | Path = DATA,
        results_dir: str | Path = "bench/results",
    ):
        self.db = db
        self.truth = truth
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.llm_factory = llm_factory
        self.active: set[str] = set()
        self._lock = threading.Lock()
        self._desk = (
            threading.Lock()
        )  # one change command at a time: one target, one audit chain

    # -- data -------------------------------------------------------------------------------------------

    def cases(self) -> list[dict[str, Any]]:
        with open(self.truth) as f:
            truth = json.load(f)
        keep = ("request_id", "identity_id", "action", "case_type", "expected_verdict")
        return sorted(
            (
                {**{k: c[k] for k in keep}, "description": c.get("description", "")}
                for c in truth["cases"]
            ),
            key=lambda c: c["request_id"],
        )

    def baseline(self) -> dict[str, Any]:
        return run_baseline(self.db, self.truth)

    def bench(self) -> dict[str, Any] | None:
        """The newest bench result, cut to what the demo shows; None when there is none."""
        files = sorted(self.results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not files:
            return None
        result = json.loads(files[-1].read_text())
        keep = (
            "request_id",
            "case_type",
            "expected_verdict",
            "verdict",
            "outcome",
            "input_tokens",
            "output_tokens",
            "cost_usd",
        )
        return {
            "file": files[-1].name,
            "label": result.get("label"),
            "model": result.get("model"),
            "scores": result.get("scores", {}),
            "cases": [{k: c.get(k) for k in keep} for c in result.get("cases", [])],
        }

    def audit(self) -> dict[str, Any]:
        """Every audit entry of the change desk, and the problems in its hash chain."""
        entries = JsonlAuditSink(self.data_dir / "audit.jsonl").entries()
        problems = verify(entries)
        return {"entries": entries, "verified": not problems, "problems": problems}

    def models(self) -> list[dict[str, str]]:
        out = []
        base = (
            os.environ.get("CIA_OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL
        ).rstrip("/")
        try:  # Ollama lists its models at /api/tags, next to the OpenAI-compatible /v1
            with urllib.request.urlopen(
                base.removesuffix("/v1") + "/api/tags", timeout=2
            ) as r:  # noqa: S310
                names = sorted(m["name"] for m in json.load(r).get("models", []))
            out += [{"llm": "openai", "model": n} for n in names if "embed" not in n]
        except (OSError, ValueError):
            out.append(
                {"llm": "openai", "model": DEFAULT_OPENAI_MODEL, "offline": "true"}
            )
        if os.environ.get(OPENROUTER_API_KEY_ENV):
            out.append(
                {
                    "llm": "openrouter",
                    "model": os.environ.get(OPENROUTER_MODEL_ENV)
                    or DEFAULT_OPENROUTER_MODEL,
                }
            )
        if os.environ.get(MODEL_ID_ENV):
            out.append({"llm": "bedrock", "model": os.environ[MODEL_ID_ENV]})
        out.append(
            {"llm": "fake", "model": "rules"}
        )  # offline, no LLM: for demos and tests
        # `make demo` names the model to start with: an llm ("openrouter") or llm|model;
        # when that one is not available, the offline rule model
        want = os.environ.get(DEMO_MODEL_ENV, "")
        if want:
            pick = next(
                (m for m in out if want in (m["llm"], f"{m['llm']}|{m['model']}")),
                out[-1],
            )
            pick["default"] = "true"
        return out

    def path(self, run_id: str) -> Path:
        if not RUN_ID.match(run_id):
            raise ValueError("bad run id")
        return self.runs_dir / f"{run_id}.jsonl"

    def events(self, run_id: str) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in self.path(run_id).read_text().splitlines()
            if line.strip()
        ]

    def summary(self, run_id: str) -> dict[str, Any]:
        events = self.events(run_id)
        first = {e["type"]: e for e in reversed(events)}
        last = {e["type"]: e for e in events}  # a change may be investigated again
        head, req, rep = (
            first.get("run", {}),
            first.get("request", {}),
            last.get("report"),
        )
        times = [e["t"] for e in events if "t" in e]
        return {
            "run_id": run_id,
            "request_id": req.get("request_id"),
            "identity_id": req.get("identity_id"),
            "description": req.get("description") or "",
            "model": head.get("model"),
            "llm": head.get("llm"),
            "case_type": head.get("case_type"),
            "expected_verdict": head.get("expected_verdict"),
            "verdict": rep and rep["verdict"],
            "confidence": rep and rep["confidence"],
            "steps": max(  # a dry-run plan's "steps" is a list, not a count
                (
                    n
                    for e in events
                    for n in (e.get("step"), e.get("steps"))
                    if isinstance(n, int)
                ),
                default=0,
            ),
            "error": last.get("run_error", {}).get("error"),
            "outcome": last.get("outcome", {}).get("outcome"),
            "running": run_id in self.active,
            "started": times[0] if times else self.path(run_id).stat().st_mtime,
            "seconds": round(times[-1] - times[0], 1) if len(times) > 1 else None,
        }

    def runs(self) -> list[dict[str, Any]]:
        out = []
        for p in self.runs_dir.glob("*.jsonl"):
            try:
                out.append(self.summary(p.stem))
            except (ValueError, OSError, KeyError):
                continue  # a half-written or foreign file never breaks the list
        return sorted(out, key=lambda r: r["started"], reverse=True)

    # -- runs -------------------------------------------------------------------------------------------

    def start(self, request_id: str, model: str = "", llm: str = "openai") -> str:
        case = next((c for c in self.cases() if c["request_id"] == request_id), None)
        if case is None:
            raise ValueError(f"unknown request {request_id!r}")
        if llm not in ("fake", *LLM_CHOICES):
            raise ValueError(f"unknown llm {llm!r}")
        if model and not MODEL_NAME.match(model):
            raise ValueError("bad model name")
        run_id = f"{request_id}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        path = self.path(run_id)
        path.touch()  # exists before the response, so the page can open the stream at once
        with self._lock:
            self.active.add(run_id)
        threading.Thread(
            target=self._run, args=(run_id, path, case, model, llm), daemon=True
        ).start()
        return run_id

    def _run(self, run_id: str, path: Path, case: dict, model: str, llm: str) -> None:
        with path.open("w") as f:
            sink = JsonlEvents(f)
            try:
                client = self.llm_factory(llm, model)
                sink(
                    {
                        "type": "run",
                        "run_id": run_id,
                        "llm": llm,
                        "model": getattr(client, "model", None)
                        or getattr(client, "model_id", None)
                        or model,
                        "case_type": case["case_type"],
                        "expected_verdict": case["expected_verdict"],
                    }
                )
                source = SyntheticSource(self.db)
                Investigator(
                    source, client, events=sink, clock=lambda: source.now
                ).investigate(ChangeRequest.from_case(case))
            except Exception as e:  # noqa: BLE001 - any failure is shown in the UI, never swallowed
                sink({"type": "run_error", "error": f"{type(e).__name__}: {e}"})
            finally:
                with self._lock:
                    self.active.discard(run_id)

    # -- the change desk -------------------------------------------------------------------------------

    def _status(self, request_id: str) -> dict[str, Any] | None:
        if not (self.data_dir / "audit.jsonl").exists():
            return None
        try:
            return open_workflow(self.db, self.data_dir).status(request_id)
        except KeyError:
            return None

    def changes(self) -> list[dict[str, Any]]:
        """Every change, newest first, with what it waits for and its run id for the trace."""
        with self._lock:
            active = {
                r.removeprefix("change-")
                for r in self.active
                if r.startswith("change-")
            }
        ids = list(dict.fromkeys([*changes(self.data_dir), *sorted(active)]))
        out = []
        for rid in reversed(ids):
            row = self._status(rid) or {"request_id": rid}
            out.append(
                {**row, "run_id": events_file("", rid).stem, "running": rid in active}
            )
        return out

    def change(
        self,
        command: str,
        request_id: str,
        llm: str = "fake",
        model: str = "",
        actions: list[str] | None = None,
        by: str = "",
        days: float = 14,
    ) -> str:
        """Checks the command, then runs it in the background; returns the run id to follow."""
        if command not in CHANGE_WAITS:
            raise ValueError(f"unknown command {command!r}")
        if not RUN_ID.match(request_id):
            raise ValueError("bad request id")
        if llm not in ("fake", *LLM_CHOICES):
            raise ValueError(f"unknown llm {llm!r}")
        if model and not MODEL_NAME.match(model):
            raise ValueError("bad model name")
        if command in ("approve", "decline") and not PERSON.match(by):
            raise ValueError("say who answers: letters, digits, . _ @ - and spaces")
        actions = list(actions or [])
        if command == "approve" and (not actions or set(actions) - set(ACTIONS)):
            raise ValueError(f"approve one or more of {', '.join(ACTIONS)}")
        if not 0 <= days <= 3650:
            raise ValueError("days must be between 0 and 3650")
        run_id = events_file("", request_id).stem
        with self._lock:
            if run_id in self.active:
                raise Conflict(f"{request_id} is busy")
            state = self._status(request_id)
            if command == "start":
                if state:
                    raise Conflict(f"{request_id} has already started")
                if request_id not in {c["request_id"] for c in self.cases()}:
                    raise ValueError(f"unknown request {request_id!r}")
            elif not state or state["waiting_for"] != CHANGE_WAITS[command]:
                raise Conflict(
                    f"{request_id} is not waiting for {CHANGE_WAITS[command]}"
                )
            self.active.add(run_id)
        args = dict(llm=llm, model=model or None, actions=actions, by=by, days=days)
        threading.Thread(
            target=self._change, args=(run_id, command, request_id, args), daemon=True
        ).start()
        return run_id

    def _change(
        self, run_id: str, command: str, request_id: str, args: dict[str, Any]
    ) -> None:
        try:
            with self._desk:
                run_command(
                    command,
                    request_id,
                    db=self.db,
                    truth=self.truth,
                    data=self.data_dir,
                    runs=self.runs_dir,
                    **args,
                )
        except Exception as e:  # noqa: BLE001 - shown in the trace, never swallowed
            with self.path(run_id).open("a") as f:
                JsonlEvents(f)(
                    {"type": "run_error", "error": f"{type(e).__name__}: {e}"}
                )
        finally:
            with self._lock:
                self.active.discard(run_id)


def make_handler(monitor: Monitor) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "cia-monitor"

        def log_message(
            self, format: str, *args: Any
        ) -> None:  # quiet; the page is the log
            pass

        def send_json(self, body: Any, status: int = 200) -> None:
            data = json.dumps(body, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            route = self.path.split("?", 1)[0]
            try:
                if route == "/":
                    data = PAGE.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                elif route == "/api/cases":
                    self.send_json(monitor.cases())
                elif route == "/api/models":
                    self.send_json(monitor.models())
                elif route == "/api/runs":
                    self.send_json(monitor.runs())
                elif route == "/api/changes":
                    self.send_json(monitor.changes())
                elif route == "/api/baseline":
                    self.send_json(monitor.baseline())
                elif route == "/api/bench":
                    bench = monitor.bench()
                    self.send_json(
                        bench or {"error": "no bench results yet"},
                        404 if bench is None else 200,
                    )
                elif route == "/api/audit":
                    self.send_json(monitor.audit())
                elif m := re.fullmatch(r"/api/runs/([^/]+)/events", route):
                    self.stream(m.group(1))
                else:
                    self.send_json({"error": "not found"}, 404)
            except ValueError as e:
                self.send_json({"error": str(e)}, 400)
            except FileNotFoundError:
                self.send_json({"error": "no such run"}, 404)

        def same_page(self) -> bool:
            """A POST must be JSON (a plain cross-site form cannot send that without a
            preflight) and, when the browser names an Origin, come from this server."""
            ctype = self.headers.get("Content-Type", "").split(";")[0].strip()
            origin = self.headers.get("Origin")
            host = self.headers.get("Host", "")
            return ctype == "application/json" and origin in (None, f"http://{host}")

        def do_POST(self) -> None:  # noqa: N802
            if not self.same_page():
                return self.send_json({"error": "JSON from this page only"}, 403)
            try:
                body = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length") or 0))
                    or b"{}"
                )
                if not isinstance(body, dict):
                    raise ValueError("the body must be a JSON object")
                if self.path == "/api/runs":
                    run_id = monitor.start(
                        str(body.get("request_id", "")),
                        str(body.get("model", "")),
                        str(body.get("llm", "openai")),
                    )
                elif m := re.fullmatch(r"/api/changes/([^/]+)/([a-z]+)", self.path):
                    actions = body.get("actions", [])
                    run_id = monitor.change(
                        m.group(2),
                        m.group(1),
                        llm=str(body.get("llm", "fake")),
                        model=str(body.get("model", "")),
                        actions=[str(x) for x in actions]
                        if isinstance(actions, list)
                        else [],
                        by=str(body.get("by", "")),
                        days=float(body.get("days", 14)),
                    )
                else:
                    return self.send_json({"error": "not found"}, 404)
            except Conflict as e:
                return self.send_json({"error": str(e)}, HTTPStatus.CONFLICT)
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                return self.send_json({"error": str(e)}, 400)
            self.send_json(
                {"run_id": run_id},
                HTTPStatus.ACCEPTED if "/changes/" in self.path else HTTPStatus.CREATED,
            )

        def stream(self, run_id: str) -> None:
            path = monitor.path(run_id)
            if not path.exists():
                raise FileNotFoundError
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            sent, deadline = 0, time.monotonic() + FOLLOW_SECONDS
            try:
                while True:
                    text = path.read_text()
                    lines = text.splitlines()
                    complete = (
                        lines if text.endswith("\n") else lines[:-1]
                    )  # never send a half-written line
                    for line in complete[sent:]:
                        self.wfile.write(f"data: {line}\n\n".encode())
                    sent = len(complete)
                    self.wfile.flush()
                    if run_id not in monitor.active or time.monotonic() > deadline:
                        self.wfile.write(b"event: end\ndata: {}\n\n")
                        self.wfile.flush()
                        return
                    time.sleep(POLL_SECONDS)
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


def serve(
    monitor: Monitor, port: int = 8765, host: str = "127.0.0.1"
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(monitor))
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Local run monitor for the investigator.")
    p.add_argument("--db", default="data/org.db")
    p.add_argument("--truth", default="bench/ground_truth.json")
    p.add_argument("--runs", default="runs")
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args(argv)
    load_env()
    try:
        server = serve(Monitor(a.db, a.truth, a.runs, data_dir=a.data), a.port)
    except OSError as e:
        raise SystemExit(
            f"cannot listen on 127.0.0.1:{a.port} ({e.strerror}); try: make monitor PORT=8787"
        ) from e
    print(f"monitor on http://127.0.0.1:{a.port}  (runs in {a.runs}/, Ctrl-C to stop)")
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
