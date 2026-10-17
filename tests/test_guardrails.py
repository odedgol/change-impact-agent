"""What each verdict allows, and the cases where nothing is allowed whatever the verdict."""

from datetime import datetime, timedelta

import pytest

from cia.guardrails import (
    ALLOWED,
    Decision,
    GuardrailError,
    Policy,
    allowed_actions,
    check_approval,
)
from cia.models import ChangeRequest, RiskReport

NOW = datetime(2026, 10, 1, 12, 0, 0)
REQUEST = ChangeRequest(request_id="REQ-0001", identity_id="u00001")


def report(verdict, request_id="REQ-0001"):
    return RiskReport(request_id=request_id, verdict=verdict, confidence="high")


def decide(verdict, request=REQUEST, age=timedelta(hours=1), **kw):
    return allowed_actions(request, report(verdict), NOW - age, NOW, **kw)


@pytest.mark.parametrize(
    "verdict,allowed",
    [
        ("safe_to_delete", ("disable", "delete")),
        ("disable_first", ("disable",)),
        ("ask_owner", ()),
        ("insufficient_evidence", ()),
    ],
)
def test_each_verdict_allows_its_row_of_the_table(verdict, allowed):
    assert decide(verdict).allowed == allowed == ALLOWED[verdict]


def test_only_safe_to_delete_ever_allows_a_delete():
    for verdict, actions in ALLOWED.items():
        assert ("delete" in actions) == report(verdict).permits_delete


def test_a_disable_request_never_gets_a_delete():
    request = ChangeRequest(
        request_id="REQ-0001", identity_id="u00001", action="disable"
    )
    d = decide("safe_to_delete", request=request)
    assert d.allowed == ("disable",) and d.reasons


def test_a_protected_identity_allows_nothing():
    d = decide("safe_to_delete", policy=Policy(protected_ids=frozenset({"u00001"})))
    assert d.allowed == () and "protected" in d.reasons[0]


@pytest.mark.parametrize("hours", [25, 24.01])
def test_a_report_older_than_24_hours_allows_nothing(hours):
    d = decide("safe_to_delete", age=timedelta(hours=hours))
    assert d.allowed == () and "old" in d.reasons[0]


def test_a_report_exactly_24_hours_old_still_counts():
    assert decide("safe_to_delete", age=timedelta(hours=24)).allowed


def test_the_age_limit_is_a_setting():
    d = decide(
        "disable_first",
        age=timedelta(hours=2),
        policy=Policy(max_report_age=timedelta(hours=1)),
    )
    assert d.allowed == ()


def test_a_report_from_the_future_allows_nothing():
    assert decide("safe_to_delete", age=timedelta(hours=-1)).allowed == ()


def test_a_report_for_another_request_allows_nothing():
    d = allowed_actions(REQUEST, report("safe_to_delete", "REQ-0002"), NOW, NOW)
    assert d.allowed == () and "REQ-0002" in d.reasons[0]


def test_a_person_can_approve_less():
    d = Decision(("disable", "delete"))
    assert check_approval(d, ["disable"]) == ("disable",)
    assert check_approval(d, []) == ()
    assert check_approval(d, ["delete", "disable"]) == ("disable", "delete")


@pytest.mark.parametrize(
    "allowed,approved",
    [
        ((), ["disable"]),  # ask_owner: a person cannot add an action
        (("disable",), ["disable", "delete"]),  # disable_first: no delete
        (("disable", "delete"), ["delete"]),  # delete without disable first
        (("disable",), ["drop_table"]),
    ],
)
def test_a_person_can_never_approve_more(allowed, approved):
    with pytest.raises(GuardrailError):
        check_approval(Decision(allowed), approved)


def test_a_decision_is_an_event():
    assert decide("ask_owner").as_event() == {
        "type": "guardrail",
        "allowed": [],
        "reasons": ["ask_owner allows no action"],
    }
