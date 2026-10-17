"""Data contracts between the investigator, the model and everything after it.

Pydantic rejects anything that breaks the contract before anyone sees it: a
finding must cite at least one evidence id, and a verdict must be one of three.
Evidence ids are assigned by code (``ev-<tool>-<n>``), never by the model.
Whether a cited id actually exists is checked later, against the run's evidence
log, so a bad citation drops one finding instead of failing the whole report.

Freshness fields on ``Evidence`` are optional and never invented: ``None`` means
the source did not say. ``collected_at`` is when the tool ran; ``source_updated_at``
is when the source data last changed, which is not the time of any event in it.

``insufficient_evidence`` means the investigation did not finish (step cap, a
required check never succeeded, unread pages). It is not a risk finding, and like
the two cautious verdicts it never permits a delete.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Verdict = Literal["safe_to_delete", "disable_first", "ask_owner", "insufficient_evidence"]
Confidence = Literal["low", "medium", "high"]

EVIDENCE_ID_PATTERN = r"^ev-[a-z_]+-\d+$"


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ChangeRequest(_Contract):
    request_id: str
    identity_id: str
    action: Literal["delete", "disable"] = "delete"


class Evidence(_Contract):
    """One tool result, logged by the investigator under an id it assigned."""

    id: str = Field(pattern=EVIDENCE_ID_PATTERN)
    tool: str
    args: dict[str, Any]
    result: Any
    collected_at: datetime | None = None
    source_name: str | None = None
    source_updated_at: datetime | None = None

    @model_validator(mode="after")
    def _source_not_newer_than_collection(self) -> Evidence:
        collected, updated = self.collected_at, self.source_updated_at
        if collected is None or updated is None:
            return self
        if (collected.tzinfo is None) != (updated.tzinfo is None):
            raise ValueError("collected_at and source_updated_at mix naive and timezone-aware times")
        if updated > collected:
            raise ValueError("source_updated_at is after collected_at")
        return self


class Finding(_Contract):
    kind: str
    summary: str
    evidence_ids: list[str] = Field(min_length=1)
    confidence: Confidence


class RiskReport(_Contract):
    request_id: str
    verdict: Verdict
    confidence: Confidence
    findings: list[Finding] = Field(default_factory=list)
    rejected_citations: int = Field(default=0, ge=0)
    incomplete_reason: str | None = None  # set with insufficient_evidence: what could not be verified

    @property
    def permits_delete(self) -> bool:
        """Only a finished investigation that found nothing may allow a delete."""
        return self.verdict == "safe_to_delete"
