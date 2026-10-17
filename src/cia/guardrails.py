"""What a finished investigation allows: plain functions, decided in code, never by the model.

The model proposes a verdict; these rules turn it into the actions that may run, and a
person may then approve some or all of them (never more). Anything unclear allows
nothing: a report for another request, a report too old to trust, a protected identity.

    safe_to_delete         -> disable, then delete
    disable_first          -> disable
    ask_owner              -> nothing (the owner is asked)
    insufficient_evidence  -> nothing (run again)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from .models import ChangeRequest, RiskReport, Verdict

Action = Literal["disable", "delete"]

ALLOWED: dict[Verdict, tuple[Action, ...]] = {
    "safe_to_delete": ("disable", "delete"),
    "disable_first": ("disable",),
    "ask_owner": (),
    "insufficient_evidence": (),
}
MAX_REPORT_AGE = timedelta(hours=24)  # a ticket or a job may have appeared since (Q15)


class GuardrailError(ValueError):
    """An approval asked for something the guardrails do not allow."""


@dataclass(frozen=True)
class Policy:
    """Settings the tenant owns. Protected identities (break-glass, the tenant's own
    admins) allow no action, whatever the verdict."""

    protected_ids: frozenset[str] = frozenset()
    max_report_age: timedelta = MAX_REPORT_AGE


@dataclass(frozen=True)
class Decision:
    """The actions that may run, in order, and why anything else was taken away."""

    allowed: tuple[Action, ...]
    reasons: tuple[str, ...] = field(default=())

    def as_event(self) -> dict:
        return {
            "type": "guardrail",
            "allowed": list(self.allowed),
            "reasons": list(self.reasons),
        }


def allowed_actions(
    request: ChangeRequest,
    report: RiskReport,
    reported_at: datetime,
    now: datetime,
    policy: Policy | None = None,
) -> Decision:
    """The actions this report allows for this request, at time ``now``."""
    policy = policy or Policy()
    if report.request_id != request.request_id:
        return Decision(
            (), (f"report is for {report.request_id}, not {request.request_id}",)
        )
    if request.identity_id in policy.protected_ids:
        return Decision((), (f"{request.identity_id} is protected",))
    age = now - reported_at
    if age < timedelta(0):
        return Decision((), ("report time is in the future",))
    if age > policy.max_report_age:
        return Decision(
            (),
            (
                f"report is {_hours(age)} old; the limit is {_hours(policy.max_report_age)}",
            ),
        )
    allowed = ALLOWED[report.verdict]
    if not allowed:
        return Decision((), (f"{report.verdict} allows no action",))
    if request.action == "disable" and "delete" in allowed:
        return Decision(("disable",), ("the request asks only to disable",))
    return Decision(allowed)


def check_approval(decision: Decision, approved: Sequence[str]) -> tuple[Action, ...]:
    """The approved actions, in the allowed order. A person can only narrow (Q12):
    anything not allowed, or a delete without its disable, is refused."""
    extra = [a for a in approved if a not in decision.allowed]
    if extra:
        raise GuardrailError(f"not allowed: {', '.join(extra)}")
    if "delete" in approved and "disable" not in approved:
        raise GuardrailError("delete needs disable first")
    return tuple(a for a in decision.allowed if a in approved)


def _hours(d: timedelta) -> str:
    return f"{d.total_seconds() / 3600:g} h"
