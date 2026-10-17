import pytest
from pydantic import ValidationError

from cia.models import ChangeRequest, Evidence, Finding, RiskReport


def _finding(**overrides):
    data = {"kind": "hidden_dependency", "summary": "Nightly job uses this key", "evidence_ids": ["ev-deps-1"],
            "confidence": "high"}
    return Finding(**(data | overrides))


def test_finding_without_evidence_is_rejected():
    with pytest.raises(ValidationError):
        _finding(evidence_ids=[])


def test_invalid_verdict_is_rejected():
    with pytest.raises(ValidationError):
        RiskReport(request_id="REQ-0001", verdict="delete_now", confidence="high")  # pyright: ignore[reportArgumentType] - invalid on purpose


def test_invalid_confidence_is_rejected():
    with pytest.raises(ValidationError):
        _finding(confidence="certain")


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        RiskReport(request_id="REQ-0001", verdict="ask_owner", confidence="low", approved=True)  # pyright: ignore[reportCallIssue] - unknown field on purpose


def test_rejected_citations_cannot_be_negative():
    with pytest.raises(ValidationError):
        RiskReport(request_id="REQ-0001", verdict="ask_owner", confidence="low", rejected_citations=-1)


@pytest.mark.parametrize("bad_id", ["deps-3", "ev-deps", "ev-Deps-3", "ev-deps-3; drop"])
def test_evidence_id_must_match_format(bad_id):
    with pytest.raises(ValidationError):
        Evidence(id=bad_id, tool="get_dependencies", args={}, result=[])


def test_report_parses_from_model_json():
    raw = """{"request_id": "REQ-0003", "verdict": "disable_first", "confidence": "medium",
              "findings": [{"kind": "hidden_dependency", "summary": "Owner of svc-billing",
                            "evidence_ids": ["ev-deps-2", "ev-usage-1"], "confidence": "high"}]}"""
    report = RiskReport.model_validate_json(raw)
    assert report.verdict == "disable_first"
    assert report.findings[0].evidence_ids == ["ev-deps-2", "ev-usage-1"]
    assert report.rejected_citations == 0


def test_contracts_are_immutable():
    request = ChangeRequest(request_id="REQ-0001", identity_id="u-0042")
    assert request.action == "delete"
    with pytest.raises(ValidationError):
        request.identity_id = "u-0001"
