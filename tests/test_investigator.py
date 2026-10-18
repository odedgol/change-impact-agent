import io
import json
import re

import pytest

from cia.generator import generate
from cia.investigator import MAX_STEPS, TOOLS, Investigator, JsonlEvents, main
from cia.llm import FakeLLM, LLMTurn, ToolCall, ToolResults
from cia.models import EVIDENCE_ID_PATTERN, ChangeRequest
from cia.sources import SyntheticSource


@pytest.fixture(scope="module")
def org(tmp_path_factory):
    db = tmp_path_factory.mktemp("org") / "org.db"
    truth = generate(db, n_identities=500, seed=7)
    return str(db), truth


def _case(truth, case_type):
    return next(c for c in truth["cases"] if c["case_type"] == case_type)


def call(name, n=0, **args):
    return LLMTurn(tool_calls=(ToolCall(id=f"t{n}", name=name, input=args),))


def submit(verdict, *cited, confidence="high", n=99):
    findings = [
        {
            "kind": "evidence",
            "summary": f"cites {i}",
            "evidence_ids": [i],
            "confidence": "high",
        }
        for i in cited
    ]
    return call(
        "submit_report", n, verdict=verdict, confidence=confidence, findings=findings
    )


def run(org, script, case_type="nightly_job", **kw):
    db, truth = org
    case = _case(truth, case_type)
    events, llm = [], FakeLLM(script)
    inv = Investigator(SyntheticSource(db), llm, events=events.append, **kw)
    report = inv.investigate(
        ChangeRequest(request_id=case["request_id"], identity_id=case["identity_id"])
    )
    return report, events, llm, case


def of(events, kind):
    return [e for e in events if e["type"] == kind]


def test_tool_calls_are_dispatched_and_logged_with_unique_ids(org):
    db, truth = org
    target = _case(truth, "nightly_job")["identity_id"]
    script = [
        LLMTurn(
            tool_calls=(
                ToolCall("a", "get_usage", {"identity_id": target, "days": 30}),
                ToolCall("b", "get_dependencies", {"identity_id": target}),
            )
        ),
        submit("disable_first", "ev-usage-1", "ev-deps-2"),
    ]
    report, events, llm, _ = run(org, script)
    ids = [e["evidence_id"] for e in of(events, "tool_call")]
    assert ids == ["ev-usage-1", "ev-deps-2"]
    assert all(re.match(EVIDENCE_ID_PATTERN, i) for i in ids)
    assert report.verdict == "disable_first" and report.rejected_citations == 0
    assert [f.evidence_ids for f in report.findings] == [["ev-usage-1"], ["ev-deps-2"]]
    # the model sees each result with the id it may cite, and the clamped window
    results = llm.calls[1]["messages"][-1]
    assert isinstance(results, ToolResults)
    assert results.results[0].content["evidence_id"] == "ev-usage-1"
    assert results.results[0].content["data"]["window_days"] == 90
    assert results.results[0].content["data"]["event_count"] > 0


def test_every_call_gets_the_system_prompt_and_tools(org):
    _, _, llm, _ = run(org, [submit("disable_first")])
    assert all(c["tools"] == list(TOOLS) and c["system"] for c in llm.calls)


def test_list_tools_are_paged_never_cut(org):
    db, truth = org
    target = _case(truth, "orphaned_service_account")["identity_id"]
    script = [
        call("get_dependencies", 1, identity_id=target, offset=0),
        call("get_dependencies", 2, identity_id=target, offset=2),
        submit("ask_owner", "ev-deps-1"),
    ]
    _, events, _, _ = run(org, script, "orphaned_service_account")
    first, second = (e["result"] for e in of(events, "tool_call"))
    assert first["total"] == 4 and first["offset"] == 0 and first["next_offset"] is None
    assert second["rows"] == first["rows"][2:]


def test_unknown_citation_triggers_exactly_one_retry_then_is_dropped_and_counted(org):
    db, truth = org
    target = _case(truth, "nightly_job")["identity_id"]
    script = [
        call("get_usage", 1, identity_id=target),
        submit("disable_first", "ev-usage-1", "ev-deps-7", n=2),
        submit("disable_first", "ev-usage-1", "ev-deps-7", n=3),
    ]
    report, events, llm, _ = run(org, script)
    assert len(llm.calls) == 3
    assert [e["rejected_ids"] for e in of(events, "citation_retry")] == [["ev-deps-7"]]
    retry = llm.calls[2]["messages"][-1].results[0]
    assert retry.is_error and retry.content["rejected_ids"] == ["ev-deps-7"]
    assert report.verdict == "disable_first"
    assert [f.evidence_ids for f in report.findings] == [["ev-usage-1"]]
    assert report.rejected_citations == 1


def test_a_corrected_retry_is_accepted(org):
    db, truth = org
    target = _case(truth, "nightly_job")["identity_id"]
    script = [
        call("get_usage", 1, identity_id=target),
        submit("disable_first", "ev-usage-9", n=2),
        submit("disable_first", "ev-usage-1", n=3),
    ]
    report, _, _, _ = run(org, script)
    assert report.rejected_citations == 0 and len(report.findings) == 1


def test_safe_to_delete_with_a_rejected_citation_becomes_ask_owner(org):
    script = [
        submit("safe_to_delete", "ev-usage-1", n=1),
        submit("safe_to_delete", "ev-usage-1", n=2),
    ]
    report, _, _, _ = run(org, script, "decoy_safe")
    assert (report.verdict, report.confidence) == ("ask_owner", "low")
    assert report.findings == [] and report.rejected_citations == 1


def test_safe_to_delete_with_valid_citations_is_kept(org):
    db, truth = org
    target = _case(truth, "decoy_safe")["identity_id"]
    script = [
        call("get_usage", 1, identity_id=target),
        call("get_dependencies", 2, identity_id=target),
        call("get_tickets", 3, identity_id=target),
        submit("safe_to_delete", "ev-usage-1", "ev-deps-2", "ev-tickets-3"),
    ]
    report, _, _, _ = run(org, script, "decoy_safe")
    assert (
        report.verdict == "safe_to_delete"
        and report.permits_delete
        and report.incomplete_reason is None
        and report.overridden is None
    )


def test_safe_to_delete_must_cite_usage_dependencies_and_tickets(org):
    db, truth = org
    target = _case(truth, "decoy_safe")["identity_id"]
    script = [
        call("get_usage", 1, identity_id=target),
        call("get_dependencies", 2, identity_id=target),
        call("get_tickets", 3, identity_id=target),
        submit("safe_to_delete", "ev-usage-1"),
    ]
    report, _, _, _ = run(org, script, "decoy_safe")
    assert report.verdict == "ask_owner" and not report.permits_delete
    assert "does not cite get_dependencies, get_tickets" in (report.overridden or "")


@pytest.mark.parametrize(
    "case_type,shows",
    [
        ("nightly_job", "usage events"),
        ("orphaned_service_account", "dependencies"),
        ("break_glass", "tickets from people"),
        ("credential_chain", "dependencies"),
    ],
)
def test_safe_to_delete_that_its_own_evidence_contradicts_becomes_ask_owner(
    org, case_type, shows
):
    """Real ids are not enough: the evidence they point at must support the verdict."""
    db, truth = org
    target = _case(truth, case_type)["identity_id"]
    script = [
        call("get_usage", 1, identity_id=target),
        call("get_dependencies", 2, identity_id=target),
        call("get_tickets", 3, identity_id=target),
        submit("safe_to_delete", "ev-usage-1", "ev-deps-2", "ev-tickets-3"),
    ]
    report, _, _, _ = run(org, script, case_type)
    assert report.verdict == "ask_owner" and report.confidence == "low"
    assert report.rejected_citations == 0 and shows in (report.overridden or "")


def test_step_cap_returns_insufficient_evidence_never_a_delete(org):
    db, truth = org
    target = _case(truth, "nightly_job")["identity_id"]
    script = [call("get_usage", i, identity_id=target) for i in range(20)] + [
        submit("safe_to_delete")
    ]
    report, events, llm, _ = run(org, script)
    assert (report.verdict, report.confidence, report.findings) == (
        "insufficient_evidence",
        "low",
        [],
    )
    assert (
        report.incomplete_reason == f"step cap reached after {MAX_STEPS} steps"
        and not report.permits_delete
    )
    assert len(of(events, "tool_call")) == MAX_STEPS
    assert of(events, "step_cap")[0]["steps"] == MAX_STEPS
    assert len(llm.calls) == MAX_STEPS + 1


def test_turns_without_tool_calls_count_against_the_cap(org):
    report, events, llm, _ = run(org, [LLMTurn(text="thinking")] * 10, max_steps=3)
    assert report.verdict == "insufficient_evidence"
    assert len(llm.calls) == 4


def test_the_retry_counts_against_the_cap(org):
    script = [
        submit("disable_first", "ev-x-1", n=1),
        submit("disable_first", "ev-x-1", n=2),
    ]
    report, events, _, _ = run(org, script, max_steps=1)
    assert len(of(events, "citation_retry")) == 1
    assert report.verdict == "insufficient_evidence" and of(events, "step_cap")


@pytest.mark.parametrize(
    "bad_call",
    [
        ToolCall("x", "delete_identity", {"identity_id": "u00001"}),
        ToolCall("x", "get_usage", {"identity_id": "u00001", "window": 30}),
        ToolCall("x", "get_tickets", {"identity_id": "u00001", "offset": -1}),
        ToolCall("x", "get_owner", {}),
    ],
)
def test_bad_tool_calls_return_a_tool_error_without_crashing(org, bad_call):
    script = [LLMTurn(tool_calls=(bad_call,)), submit("ask_owner")]
    report, events, llm, _ = run(org, script)
    result = llm.calls[1]["messages"][-1].results[0]
    assert result.is_error and "error" in result.content
    assert of(events, "tool_error") and not of(events, "tool_call")
    assert report.verdict == "ask_owner"


@pytest.mark.parametrize(
    "bad",
    [
        {"verdict": "delete_now", "confidence": "high", "findings": []},
        {
            "verdict": "safe_to_delete",
            "confidence": "high",
            "findings": [],
            "rejected_citations": 0,
        },
        {
            "verdict": "safe_to_delete",
            "confidence": "high",
            "findings": [],
            "request_id": "REQ-9999",
        },
    ],
)
def test_an_invalid_report_is_sent_back(org, bad):
    """The model can't pick a bad verdict, or set fields that code owns."""
    script = [call("submit_report", 1, **bad), submit("ask_owner")]
    report, events, llm, _ = run(org, script)
    assert llm.calls[1]["messages"][-1].results[0].is_error
    assert report.verdict == "ask_owner"


def test_events_tell_the_whole_story_in_order(org):
    db, truth = org
    target = _case(truth, "break_glass")["identity_id"]
    script = [
        call("get_tickets", 1, identity_id=target),
        submit("ask_owner", "ev-tickets-1"),
    ]
    _, events, _, case = run(org, script, "break_glass")
    assert [e["type"] for e in events] == [
        "request",
        "model_turn",
        "tool_call",
        "model_turn",
        "report",
    ]
    assert events[0]["identity_id"] == case["identity_id"]
    assert events[1]["tool_calls"] == [
        {"name": "get_tickets", "input": {"identity_id": target}}
    ]
    assert events[1]["stop_reason"] == "end_turn"  # a cut-off turn shows as "length"
    assert "Break-glass" in events[2]["result"]["rows"][0]["text"]
    assert events[-1]["verdict"] == "ask_owner"


def test_jsonl_events_and_cli(org, tmp_path):
    buf = io.StringIO()
    JsonlEvents(buf)({"type": "request", "n": 1})
    line = json.loads(buf.getvalue())
    assert line["type"] == "request" and line["n"] == 1 and line["t"] > 0

    db, truth = org
    case = _case(truth, "policy_exception")
    out = tmp_path / "events.jsonl"
    llm = FakeLLM(
        [
            call("get_dependencies", 1, identity_id=case["identity_id"]),
            submit("disable_first", "ev-deps-1"),
        ]
    )
    main(["--db", db, "--identity", case["identity_id"], "--events", str(out)], llm=llm)
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [e["type"] for e in lines if e["type"] != "model_turn"] == [
        "run",
        "request",
        "tool_call",
        "report",
    ]
    assert of(lines, "tool_call")[0]["result"]["rows"][0]["kind"] == "policy_reference"


def test_bad_step_cap_is_rejected(org):
    with pytest.raises(ValueError):
        Investigator(SyntheticSource(org[0]), FakeLLM([]), max_steps=0)
