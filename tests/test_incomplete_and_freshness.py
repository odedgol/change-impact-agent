"""Evidence freshness metadata and incomplete investigations (insufficient_evidence)."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from cia.generator import NOW, generate
from cia.investigator import REQUIRED_FOR_DELETE, Investigator
from cia.llm import FakeLLM, LLMTurn, ToolCall
from cia.models import ChangeRequest, Evidence, RiskReport
from cia.sources import SyntheticSource


@pytest.fixture(scope="module")
def org(tmp_path_factory):
    db = tmp_path_factory.mktemp("org") / "org.db"
    truth = generate(db, n_identities=500, seed=7)
    return str(db), truth


def decoy(truth):
    return next(c for c in truth["cases"] if c["case_type"] == "decoy_safe")


def call(name, n, **args):
    return LLMTurn(tool_calls=(ToolCall(f"t{n}", name, args),))


def submit(verdict, *cited, n=99):
    return call(
        "submit_report",
        n,
        verdict=verdict,
        confidence="high",
        findings=[
            {
                "kind": "evidence",
                "summary": "cited",
                "evidence_ids": [i],
                "confidence": "high",
            }
            for i in cited
        ],
    )


def investigate(source, script, case, **kw):
    events = []
    inv = Investigator(
        source, FakeLLM(script), events=events.append, clock=lambda: NOW, **kw
    )
    report = inv.investigate(
        ChangeRequest(request_id=case["request_id"], identity_id=case["identity_id"])
    )
    return report, events


def full_read(target):
    return [
        call(t, i, identity_id=target) for i, t in enumerate(REQUIRED_FOR_DELETE, 1)
    ]


# ---- contracts: backward compatible, nothing invented


def test_old_evidence_without_freshness_is_still_valid():
    ev = Evidence(id="ev-deps-1", tool="get_dependencies", args={}, result=[])
    assert (ev.collected_at, ev.source_name, ev.source_updated_at) == (None, None, None)


def test_evidence_with_freshness():
    ev = Evidence(
        id="ev-deps-1",
        tool="get_dependencies",
        args={},
        result=[],
        collected_at=NOW,
        source_name="keycloak",
        source_updated_at=NOW - timedelta(minutes=5),
    )
    assert ev.source_updated_at is not None and ev.collected_at is not None
    assert ev.source_updated_at < ev.collected_at


def test_source_update_after_collection_is_rejected():
    with pytest.raises(ValidationError):
        Evidence(
            id="ev-deps-1",
            tool="t",
            args={},
            result=[],
            collected_at=NOW,
            source_updated_at=NOW + timedelta(1),
        )


def test_mixed_naive_and_aware_times_are_rejected():
    with pytest.raises(ValidationError):
        Evidence(
            id="ev-deps-1",
            tool="t",
            args={},
            result=[],
            collected_at=NOW,
            source_updated_at=datetime(2026, 9, 1, tzinfo=UTC),
        )


def test_old_reports_still_parse_and_only_safe_permits_delete():
    old = RiskReport.model_validate_json(
        '{"request_id": "R", "verdict": "ask_owner", "confidence": "low"}'
    )
    assert old.incomplete_reason is None and not old.permits_delete
    assert RiskReport(
        request_id="R", verdict="safe_to_delete", confidence="high"
    ).permits_delete
    for v in ("disable_first", "ask_owner", "insufficient_evidence"):
        assert not RiskReport(
            request_id="R", verdict=v, confidence="high"
        ).permits_delete


# ---- freshness on real runs


def test_synthetic_evidence_records_collection_time_but_no_invented_update_time(org):
    db, truth = org
    case = decoy(truth)
    _, events = investigate(
        SyntheticSource(db),
        [*full_read(case["identity_id"]), submit("safe_to_delete")],
        case,
    )
    calls = [e for e in events if e["type"] == "tool_call"]
    assert {e["collected_at"] for e in calls} == {NOW}
    assert {e["source_name"] for e in calls} == {"synthetic"}
    assert {e["source_updated_at"] for e in calls} == {None}


def test_a_source_that_knows_its_freshness_passes_it_on(org):
    db, truth = org
    case = decoy(truth)
    source = SyntheticSource(db)
    source.name, source.updated_at = "iam-snapshot", NOW - timedelta(hours=1)
    _, events = investigate(
        source,
        [call("get_usage", 1, identity_id=case["identity_id"]), submit("ask_owner")],
        case,
    )
    ev = next(e for e in events if e["type"] == "tool_call")
    assert (ev["source_name"], ev["source_updated_at"]) == (
        "iam-snapshot",
        NOW - timedelta(hours=1),
    )


# ---- incomplete is not safe


def test_safe_to_delete_without_tickets_is_insufficient_evidence(org):
    db, truth = org
    case = decoy(truth)
    script = [
        call("get_usage", 1, identity_id=case["identity_id"]),
        call("get_dependencies", 2, identity_id=case["identity_id"]),
        submit("safe_to_delete", "ev-usage-1"),
    ]
    report, _ = investigate(SyntheticSource(db), script, case)
    assert (report.verdict, report.confidence) == ("insufficient_evidence", "low")
    assert (
        report.incomplete_reason == "safe_to_delete not verified: get_tickets not read"
    )
    assert not report.permits_delete


def test_unread_pages_block_a_delete(org):
    db, truth = org
    case = decoy(truth)

    class PagedSource(SyntheticSource):
        def get_tickets(self, identity_id):
            return [{"id": i} for i in range(120)]

    report, _ = investigate(
        PagedSource(db),
        [*full_read(case["identity_id"]), submit("safe_to_delete")],
        case,
    )
    assert report.verdict == "insufficient_evidence"
    assert (report.incomplete_reason or "").endswith("get_tickets has unread pages")


def test_reading_every_page_clears_the_gap(org):
    db, truth = org
    case = decoy(truth)

    class PagedSource(SyntheticSource):
        def get_tickets(self, identity_id):
            return [{"id": i} for i in range(60)]

    t = case["identity_id"]
    script = [
        *full_read(t),
        call("get_tickets", 9, identity_id=t, offset=50),
        submit("safe_to_delete"),
    ]
    report, _ = investigate(PagedSource(db), script, case)
    assert report.verdict == "safe_to_delete"


def test_an_unavailable_source_is_an_error_not_an_empty_result(org):
    db, truth = org
    case = decoy(truth)

    class DownSource(SyntheticSource):
        def get_tickets(self, identity_id):
            raise ConnectionError("ticket system timed out")

    report, events = investigate(
        DownSource(db),
        [*full_read(case["identity_id"]), submit("safe_to_delete")],
        case,
    )
    errors = [e for e in events if e["type"] == "tool_error"]
    assert (
        errors
        and "source unavailable for get_tickets: ConnectionError" in errors[0]["error"]
    )
    assert report.verdict == "insufficient_evidence" and "get_tickets not read" in (
        report.incomplete_reason or ""
    )


def test_failure_then_success_counts_as_read(org):
    db, truth = org
    case = decoy(truth)
    calls = {"n": 0}

    class FlakySource(SyntheticSource):
        def get_tickets(self, identity_id):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("first try")
            return super().get_tickets(identity_id)

    t = case["identity_id"]
    script = [
        *full_read(t),
        call("get_tickets", 9, identity_id=t),
        submit("safe_to_delete"),
    ]
    report, _ = investigate(FlakySource(db), script, case)
    assert report.verdict == "safe_to_delete"


def test_the_model_can_report_insufficient_evidence(org):
    db, truth = org
    case = decoy(truth)
    report, _ = investigate(
        SyntheticSource(db), [submit("insufficient_evidence")], case
    )
    assert (report.verdict, report.confidence, report.incomplete_reason) == (
        "insufficient_evidence",
        "low",
        "reported by the model",
    )


def test_cautious_verdicts_are_not_downgraded_by_missing_reads(org):
    db, truth = org
    case = decoy(truth)
    report, _ = investigate(SyntheticSource(db), [submit("disable_first")], case)
    assert report.verdict == "disable_first" and report.incomplete_reason is None


def test_the_model_cannot_set_incomplete_reason(org):
    db, truth = org
    case = decoy(truth)
    bad = call(
        "submit_report",
        1,
        verdict="ask_owner",
        confidence="low",
        findings=[],
        incomplete_reason="x",
    )
    report, events = investigate(SyntheticSource(db), [bad, submit("ask_owner")], case)
    assert any(
        e["type"] == "tool_error" and "unexpected fields" in e["error"] for e in events
    )
