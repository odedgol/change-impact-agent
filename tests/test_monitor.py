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
    return [
        LLMTurn(
            reasoning="check deps",
            tool_calls=(ToolCall("a", "get_dependencies", {"identity_id": identity}),),
        ),
        LLMTurn(
            tool_calls=(
                ToolCall(
                    "b",
                    "submit_report",
                    {
                        "verdict": "ask_owner",
                        "confidence": "high",
                        "findings": [
                            {
                                "kind": "owned",
                                "summary": "owns a service account",
                                "evidence_ids": ["ev-deps-1"],
                                "confidence": "high",
                            }
                        ],
                    },
                ),
            )
        ),
    ]


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
    mon = Monitor(
        str(db),
        str(truth_path),
        str(tmp_path / "runs"),
        llm_factory=factory,
        data_dir=tmp_path / "data",
    )
    srv = serve(mon, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", mon, truth, factory
    srv.shutdown()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def post(url, body, headers=None):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers=headers or {"Content-Type": "application/json"},
    )
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
    assert {"request_id", "identity_id", "case_type", "expected_verdict"} <= set(
        cases[0]
    )


def test_start_run_stream_and_list(server):
    base, mon, truth, factory = server
    case = next(
        c for c in truth["cases"] if c["case_type"] == "orphaned_service_account"
    )
    factory.identity = case["identity_id"]
    status, body = post(
        base + "/api/runs",
        {"request_id": case["request_id"], "llm": "openai", "model": "gemma4:26b-mlx"},
    )
    assert status == 201
    run_id = body["run_id"]
    wait_done(mon, run_id)

    raw = get(f"{base}/api/runs/{run_id}/events")[2].decode()
    events = [
        json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {"')
    ]
    assert [e["type"] for e in events] == [
        "run",
        "request",
        "model_turn",
        "tool_call",
        "model_turn",
        "report",
    ]
    assert (
        events[0]["expected_verdict"] == "ask_owner"
        and events[2]["reasoning"] == "check deps"
    )
    assert "event: end" in raw

    runs = json.loads(get(base + "/api/runs")[2])
    assert runs[0]["run_id"] == run_id and runs[0]["verdict"] == "ask_owner"
    assert (
        runs[0]["expected_verdict"] == "ask_owner"
        and runs[0]["running"] is False
        and runs[0]["steps"] == 2
    )


def test_a_failing_model_is_reported_not_hidden(server):
    base, mon, truth, _ = server
    _, body = post(
        base + "/api/runs",
        {"request_id": truth["cases"][0]["request_id"], "model": "broken"},
    )
    wait_done(mon, body["run_id"])
    run = json.loads(get(base + "/api/runs")[2])[0]
    assert run["error"] == "RuntimeError: Connection refused" and run["verdict"] is None


@pytest.mark.parametrize(
    "body",
    [
        {"request_id": "REQ-9999"},
        {"request_id": "REQ-0001", "llm": "gpt"},
        {"request_id": "REQ-0001", "model": "x; rm -rf /"},
    ],
)
def test_bad_start_requests_are_rejected(server, body):
    status, out = post(server[0] + "/api/runs", body)
    assert status == 400 and "error" in out


@pytest.mark.parametrize(
    "path",
    [
        "/api/runs/..%2F..%2Fetc%2Fpasswd/events",
        "/api/runs/no-such-run/events",
        "/nope",
    ],
)
def test_bad_paths(server, path):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(server[0] + path)
    assert e.value.code in (400, 404)


def test_terminal_runs_show_up_and_junk_files_are_skipped(server):
    base, mon, _, _ = server
    (mon.runs_dir / "junk.jsonl").write_text("not json\n")
    (mon.runs_dir / "u00002.jsonl").write_text(
        json.dumps(
            {
                "type": "request",
                "request_id": "REQ-CLI",
                "identity_id": "u00002",
                "t": 1.0,
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "report",
                "request_id": "REQ-CLI",
                "verdict": "ask_owner",
                "confidence": "low",
                "findings": [],
                "rejected_citations": 0,
                "t": 3.5,
            }
        )
        + "\n"
    )
    runs = {r["run_id"]: r for r in json.loads(get(base + "/api/runs")[2])}
    assert "junk" not in runs
    assert runs["u00002"]["verdict"] == "ask_owner" and runs["u00002"]["seconds"] == 2.5


def test_page_draws_the_flow_with_pinned_react_flow_and_filters_by_model():
    """The Flow tab uses React Flow from a CDN at exact versions; model output is rendered as text only."""
    import re

    from cia.monitor import PAGE

    page = PAGE.read_text()
    urls = re.findall(r"https://(?:esm\.sh|cdn\.jsdelivr\.net/npm)/[^\"'\s]+", page)
    assert urls and all(re.search(r"@\d+\.\d+\.\d+", u) for u in urls), (
        urls
    )  # every library pinned
    assert "@xyflow/react@12.11.6" in page and "react@19.2.8" in page
    assert 'id="mfilter"' in page and '["flow","Flow"]' in page
    assert "innerHTML" not in page and "dangerouslySetInnerHTML" not in page
    assert (
        "insufficient_evidence, confidence low" in page
    )  # the step-cap card matches PR 2.4d
    # the Flow tab draws the LangGraph nodes of cia.graph, not the old loop
    assert "StateGraph" in page
    for node in ("model", "tools", "nudge", "check", "finish", "cap"):
        assert f'box("{node}"' in page
    assert "Investigator loop" not in page and "Temporal" not in page
    # the change desk: start, answer and end the quiet period from the page
    assert 'id="startchange"' in page and '"/api/changes/"' in page
    for command in ('send("approve"', 'send("decline"', 'send("resume"'):
        assert command in page


# -- the change desk ---------------------------------------------------------------------


def change(base, mon, rid, command, body=None):
    status, out = post(f"{base}/api/changes/{rid}/{command}", body or {})
    if status == 202:
        wait_done(mon, out["run_id"])
    return status, out


def test_change_desk_start_approve_quiet_period_delete(server):
    base, mon, truth, _ = server
    rid = next(c for c in truth["cases"] if c["case_type"] == "decoy_safe")[
        "request_id"
    ]
    assert json.loads(get(base + "/api/changes")[2]) == []

    assert change(base, mon, rid, "start") == (202, {"run_id": f"change-{rid}"})
    (row,) = json.loads(get(base + "/api/changes")[2])
    assert row["verdict"] == "safe_to_delete" and row["waiting_for"] == "approval"
    assert row["run_id"] == f"change-{rid}" and row["running"] is False

    status, _ = change(
        base, mon, rid, "approve", {"actions": ["disable", "delete"], "by": "ana"}
    )
    assert status == 202
    (row,) = json.loads(get(base + "/api/changes")[2])
    assert row["waiting_for"] == "quiet_period" and row["approved"] == [
        "disable",
        "delete",
    ]

    assert change(base, mon, rid, "resume", {"days": 14})[0] == 202
    (row,) = json.loads(get(base + "/api/changes")[2])
    assert row["outcome"] == "deleted" and row["waiting_for"] is None

    run = next(
        r
        for r in json.loads(get(base + "/api/runs")[2])
        if r["run_id"] == row["run_id"]
    )
    assert run["outcome"] == "deleted" and run["verdict"] == "safe_to_delete"
    events = mon.events(row["run_id"])
    assert [e["type"] for e in events if e["type"] in ("approval", "outcome")] == [
        "approval",
        "outcome",
    ]


def test_change_desk_refuses_out_of_order_and_bad_answers(server):
    base, mon, truth, _ = server
    rid = next(c for c in truth["cases"] if c["case_type"] == "decoy_safe")[
        "request_id"
    ]
    assert (
        change(base, mon, rid, "approve", {"actions": ["delete"], "by": "ana"})[0]
        == 409
    )
    assert change(base, mon, "REQ-9999", "start")[0] == 400
    assert change(base, mon, rid, "launch")[0] == 400
    assert change(base, mon, rid, "start")[0] == 202
    assert change(base, mon, rid, "start")[0] == 409  # already started
    assert change(base, mon, rid, "resume")[0] == 409  # still waits for a person
    for bad in (
        {"actions": ["delete"], "by": ""},  # who answers?
        {"actions": ["drop_table"], "by": "ana"},
        {"actions": [], "by": "ana"},  # an approval names what it allows
        {"actions": ["delete"], "by": "<script>"},
    ):
        assert change(base, mon, rid, "approve", bad)[0] == 400, bad
    assert json.loads(get(base + "/api/changes")[2])[0]["waiting_for"] == "approval"


@pytest.mark.parametrize(
    "headers",
    [
        {"Content-Type": "text/plain"},  # a cross-site form can send this
        {"Content-Type": "application/json", "Origin": "http://evil.example"},
    ],
)
def test_posts_only_as_json_from_this_page(server, headers):
    base, mon, truth, _ = server
    rid = truth["cases"][0]["request_id"]
    for path in (f"/api/changes/{rid}/start", "/api/runs"):
        status, out = post(base + path, {"request_id": rid}, headers)
        assert status == 403 and "error" in out
    assert not mon.active and json.loads(get(base + "/api/changes")[2]) == []


def test_post_from_the_page_itself_is_allowed(server):
    base, mon, truth, _ = server
    rid = next(c for c in truth["cases"] if c["case_type"] == "decoy_safe")[
        "request_id"
    ]
    status, out = post(
        f"{base}/api/changes/{rid}/start",
        {},
        {"Content-Type": "application/json", "Origin": base},
    )
    assert status == 202
    wait_done(mon, out["run_id"])
