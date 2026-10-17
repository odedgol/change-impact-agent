"""Local run monitor: launch investigations, watch them live, browse past runs.

Standard library only. Binds to 127.0.0.1. Each run is one JSON-lines file in
``runs/`` (the same stream ``--events`` writes and the Rust UI will read), so
runs started from the terminal show up here too.

    GET  /                       the page (monitor.html)
    GET  /api/cases              change requests from the ground truth, with the expected verdict
    GET  /api/models             local Ollama models, plus Bedrock when configured
    GET  /api/runs               every run, newest first, summarised
    GET  /api/runs/<id>/events   the run's events as server-sent events; follows a live run to the end
    POST /api/runs               {"request_id", "model", "llm"} starts a run, returns {"run_id"}

The monitor is read-only towards the org: it can only start investigations,
which use the read-only evidence source.
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

from .investigator import Investigator, JsonlEvents
from .llm import (
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    MODEL_ID_ENV,
    BedrockClient,
    LLMClient,
    OpenAICompatClient,
)
from .models import ChangeRequest
from .sources import SyntheticSource

PAGE = Path(__file__).with_name("monitor.html")
RUN_ID = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
MODEL_NAME = re.compile(r"^[A-Za-z0-9_.:/@-]{1,120}$")
FOLLOW_SECONDS = 600  # how long one event stream follows a live run
POLL_SECONDS = 0.25

LLMFactory = Callable[[str, str], LLMClient]


def default_llm_factory(llm: str, model: str) -> LLMClient:
    if llm == "bedrock":
        return BedrockClient(model_id=model or None)
    return OpenAICompatClient(model=model or None)


class Monitor:
    def __init__(self, db: str, truth: str, runs_dir: str = "runs", llm_factory: LLMFactory = default_llm_factory):
        self.db = db
        self.truth = truth
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.llm_factory = llm_factory
        self.active: set[str] = set()
        self._lock = threading.Lock()

    # -- data -------------------------------------------------------------------------------------------

    def cases(self) -> list[dict[str, Any]]:
        with open(self.truth) as f:
            truth = json.load(f)
        keep = ("request_id", "identity_id", "action", "case_type", "expected_verdict")
        return sorted(({k: c[k] for k in keep} for c in truth["cases"]), key=lambda c: c["request_id"])

    def models(self) -> list[dict[str, str]]:
        out = []
        base = (os.environ.get("CIA_OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL).rstrip("/")
        try:  # Ollama lists its models at /api/tags, next to the OpenAI-compatible /v1
            with urllib.request.urlopen(base.removesuffix("/v1") + "/api/tags", timeout=2) as r:  # noqa: S310
                names = sorted(m["name"] for m in json.load(r).get("models", []))
            out += [{"llm": "openai", "model": n} for n in names if "embed" not in n]
        except (OSError, ValueError):
            out.append({"llm": "openai", "model": DEFAULT_OPENAI_MODEL, "offline": "true"})
        if os.environ.get(MODEL_ID_ENV):
            out.append({"llm": "bedrock", "model": os.environ[MODEL_ID_ENV]})
        return out

    def path(self, run_id: str) -> Path:
        if not RUN_ID.match(run_id):
            raise ValueError("bad run id")
        return self.runs_dir / f"{run_id}.jsonl"

    def events(self, run_id: str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.path(run_id).read_text().splitlines() if line.strip()]

    def summary(self, run_id: str) -> dict[str, Any]:
        events = self.events(run_id)
        first = {e["type"]: e for e in reversed(events)}
        head, req, rep = first.get("run", {}), first.get("request", {}), first.get("report")
        times = [e["t"] for e in events if "t" in e]
        return {
            "run_id": run_id, "request_id": req.get("request_id"), "identity_id": req.get("identity_id"),
            "model": head.get("model"), "llm": head.get("llm"), "case_type": head.get("case_type"),
            "expected_verdict": head.get("expected_verdict"),
            "verdict": rep and rep["verdict"], "confidence": rep and rep["confidence"],
            "steps": max((max(e.get("step", 0), e.get("steps", 0)) for e in events), default=0),
            "error": first.get("run_error", {}).get("error"),
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
        if llm not in ("openai", "bedrock"):
            raise ValueError(f"unknown llm {llm!r}")
        if model and not MODEL_NAME.match(model):
            raise ValueError("bad model name")
        run_id = f"{request_id}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        path = self.path(run_id)
        path.touch()  # exists before the response, so the page can open the stream at once
        with self._lock:
            self.active.add(run_id)
        threading.Thread(target=self._run, args=(run_id, path, case, model, llm), daemon=True).start()
        return run_id

    def _run(self, run_id: str, path: Path, case: dict, model: str, llm: str) -> None:
        with path.open("w") as f:
            sink = JsonlEvents(f)
            try:
                client = self.llm_factory(llm, model)
                sink({"type": "run", "run_id": run_id, "llm": llm,
                      "model": getattr(client, "model", None) or getattr(client, "model_id", None) or model,
                      "case_type": case["case_type"], "expected_verdict": case["expected_verdict"]})
                source = SyntheticSource(self.db)
                Investigator(source, client, events=sink, clock=lambda: source.now).investigate(
                    ChangeRequest(request_id=case["request_id"], identity_id=case["identity_id"]))
            except Exception as e:  # noqa: BLE001 - any failure is shown in the UI, never swallowed
                sink({"type": "run_error", "error": f"{type(e).__name__}: {e}"})
            finally:
                with self._lock:
                    self.active.discard(run_id)


def make_handler(monitor: Monitor) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "cia-monitor"

        def log_message(self, format: str, *args: Any) -> None:  # quiet; the page is the log
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
                elif m := re.fullmatch(r"/api/runs/([^/]+)/events", route):
                    self.stream(m.group(1))
                else:
                    self.send_json({"error": "not found"}, 404)
            except ValueError as e:
                self.send_json({"error": str(e)}, 400)
            except FileNotFoundError:
                self.send_json({"error": "no such run"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/runs":
                return self.send_json({"error": "not found"}, 404)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                run_id = monitor.start(str(body.get("request_id", "")), str(body.get("model", "")),
                                       str(body.get("llm", "openai")))
            except (ValueError, json.JSONDecodeError) as e:
                return self.send_json({"error": str(e)}, 400)
            self.send_json({"run_id": run_id}, HTTPStatus.CREATED)

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
                    complete = lines if text.endswith("\n") else lines[:-1]  # never send a half-written line
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


def serve(monitor: Monitor, port: int = 8765, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(monitor))
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Local run monitor for the investigator.")
    p.add_argument("--db", default="data/org.db")
    p.add_argument("--truth", default="bench/ground_truth.json")
    p.add_argument("--runs", default="runs")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args(argv)
    try:
        server = serve(Monitor(a.db, a.truth, a.runs), a.port)
    except OSError as e:
        raise SystemExit(f"cannot listen on 127.0.0.1:{a.port} ({e.strerror}); try: make monitor PORT=8787") from e
    print(f"monitor on http://127.0.0.1:{a.port}  (runs in {a.runs}/, Ctrl-C to stop)")
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
