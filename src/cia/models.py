"""Data contracts between the investigator, the model and everything after it.

Pydantic rejects anything that breaks the contract before anyone sees it: a
finding must cite at least one evidence id, and a verdict must be one of three.
Evidence ids are assigned by code (``ev-<tool>-<n>``), never by the model.
Whether a cited id actually exists is checked later, against the run's evidence
log, so a bad citation drops one finding instead of failing the whole report.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Verdict = Literal["safe_to_delete", "disable_first", "ask_owner"]
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
