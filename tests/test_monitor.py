"""Run monitor server: real HTTP on an ephemeral localhost port, FakeLLM, temp runs dir."""
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from cia.generator import generate
from cia.llm import FakeLLM, LLMTurn, ToolCall
from cia.monitor import Monitor, serve


def script_for(identity):
    return [LLMTurn(reasoning="check deps", tool_calls=(ToolCall("a", "get_dependencies", {"identity_id": identity}),)),
            LLMTurn(tool_calls=(ToolCall("b", "submit_report", {"verdict": "ask_owner", "confidence": "high", "findings": [
                {"kind": "owned", "summary": "owns a service account", "evidence_ids": ["ev-deps-1"],
                 "confidence": "high"}]}),))]


class ScriptedFactory:
    """The monitor's llm_factory in tests: a scripted run for ``identity``; model "broken" fails to start."""

    identity = "u00000"

    def __call__(self, llm, model):
        if model == "broken":
            raise RuntimeError("Connection refused")
        return FakeLLM(script_for(self.identity))


@pytest.fixture
def server(tmp_path):
    db, truth_path = tmp_path / "org.db", tmp_path / "truth.json"
    truth = generate(db, n_identities=500, seed=7)
    truth_path.write_text(json.dumps(truth))

    factory = ScriptedFactory()
    mon = Monitor(str(db), str(truth_path), str(tmp_path / "runs"), llm_factory=factory)
    srv = serve(mon, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", mon, truth, factory
    srv.shutdown()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait_done(mon, run_id):
    for _ in range(100):
        if run_id not in mon.active:
            return
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_page_and_cases(server):
    base, _, truth, _ = server
    status, ctype, body = get(base + "/")
    assert status == 200 and ctype.startswith("text/html") and b"Run Monitor" in body
    cases = json.loads(get(base + "/api/cases")[2])
    assert len(cases) == len(truth["cases"])
    assert {"request_id", "identity_id", "case_type", "expected_verdict"} <= set(cases[0])


def test_start_run_stream_and_list(server):
    base, mon, truth, factory = server
    case = next(c for c in truth["cases"] if c["case_type"] == "orphaned_service_account")
    factory.identity = case["identity_id"]
    status, body = post(base + "/api/runs", {"request_id": case["request_id"], "llm": "openai", "model": "gemma4:26b-mlx"})
    assert status == 201
    run_id = body["run_id"]
    wait_done(mon, run_id)

    raw = get(f"{base}/api/runs/{run_id}/events")[2].decode()
    events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {\"")]
    assert [e["type"] for e in events] == ["run", "request", "model_turn", "tool_call", "model_turn", "report"]
    assert events[0]["expected_verdict"] == "ask_owner" and events[2]["reasoning"] == "check deps"
    assert "event: end" in raw

    runs = json.loads(get(base + "/api/runs")[2])
    assert runs[0]["run_id"] == run_id and runs[0]["verdict"] == "ask_owner"
    assert runs[0]["expected_verdict"] == "ask_owner" and runs[0]["running"] is False and runs[0]["steps"] == 2


def test_a_failing_model_is_reported_not_hidden(server):
    base, mon, truth, _ = server
    _, body = post(base + "/api/runs", {"request_id": truth["cases"][0]["request_id"], "model": "broken"})
    wait_done(mon, body["run_id"])
    run = json.loads(get(base + "/api/runs")[2])[0]
    assert run["error"] == "RuntimeError: Connection refused" and run["verdict"] is None


@pytest.mark.parametrize("body", [{"request_id": "REQ-9999"}, {"request_id": "REQ-0001", "llm": "gpt"},
                                  {"request_id": "REQ-0001", "model": "x; rm -rf /"}])
def test_bad_start_requests_are_rejected(server, body):
    status, out = post(server[0] + "/api/runs", body)
    assert status == 400 and "error" in out


@pytest.mark.parametrize("path", ["/api/runs/..%2F..%2Fetc%2Fpasswd/events", "/api/runs/no-such-run/events", "/nope"])
def test_bad_paths(server, path):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(server[0] + path)
    assert e.value.code in (400, 404)


def test_terminal_runs_show_up_and_junk_files_are_skipped(server):
    base, mon, _, _ = server
    (mon.runs_dir / "junk.jsonl").write_text("not json\n")
    (mon.runs_dir / "u00002.jsonl").write_text(
        json.dumps({"type": "request", "request_id": "REQ-CLI", "identity_id": "u00002", "t": 1.0}) + "\n" +
        json.dumps({"type": "report", "request_id": "REQ-CLI", "verdict": "ask_owner", "confidence": "low",
                    "findings": [], "rejected_citations": 0, "t": 3.5}) + "\n")
    runs = {r["run_id"]: r for r in json.loads(get(base + "/api/runs")[2])}
    assert "junk" not in runs
    assert runs["u00002"]["verdict"] == "ask_owner" and runs["u00002"]["seconds"] == 2.5


def test_page_draws_the_flow_with_pinned_react_flow_and_filters_by_model():
    """The Flow tab uses React Flow from a CDN at exact versions; model output is rendered as text only."""
    import re

    from cia.monitor import PAGE

    page = PAGE.read_text()
    urls = re.findall(r"https://(?:esm\.sh|cdn\.jsdelivr\.net/npm)/[^\"'\s]+", page)
    assert urls and all(re.search(r"@\d+\.\d+\.\d+", u) for u in urls), urls  # every library pinned
    assert "@xyflow/react@12.11.6" in page and "react@19.2.8" in page
    assert 'id="mfilter"' in page and '["flow","Flow"]' in page
    assert "innerHTML" not in page and "dangerouslySetInnerHTML" not in page
    assert "insufficient_evidence, confidence low" in page  # the step-cap card matches PR 2.4d
