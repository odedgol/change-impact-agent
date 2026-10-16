"""Planted case types: one registered function per type (Strategy + Registry).

Each planter receives the org being built and a leaver id, adds the hidden
evidence that makes the leaver unsafe to delete, and returns the
``hidden_dependencies`` recorded in the ground truth. ``generate`` runs every
registered type in registration order, so adding a case type is one new
``@case_type`` function and nothing else.

Determinism: registration order and the RNG calls inside each planter are part
of the generated output. Append new types at the end; never reorder.
"""
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .models import Verdict

NOW = datetime(2026, 10, 1, 12, 0, 0)  # fixed clock keeps runs reproducible


def _ts(days_ago: float, hour: int | None = None) -> str:
    t = NOW - timedelta(days=days_ago)
    if hour is not None:
        t = t.replace(hour=hour, minute=0, second=0)
    return t.isoformat(timespec="seconds")


@dataclass
class OrgBuilder:
    """Rows of the org being generated, plus the RNG every planter shares."""

    rng: random.Random
    identities: list[tuple] = field(default_factory=list)
    usage: list[tuple] = field(default_factory=list)
    deps: list[tuple] = field(default_factory=list)
    tickets: list[tuple] = field(default_factory=list)
    services: list[str] = field(default_factory=list)
    _index: dict[str, int] = field(default_factory=dict, repr=False)

    def add_identity(self, row: tuple) -> None:
        self._index[row[0]] = len(self.identities)
        self.identities.append(row)

    def add_usage(self, identity: str, ts: str, resource: str, caller: str) -> None:
        self.usage.append((identity, ts, resource, caller))

    def add_dependency(self, source: str, target: str, kind: str, detail: str) -> None:
        self.deps.append((source, target, kind, detail))

    def add_ticket(self, identity: str, days_ago: float, author: str, text: str) -> None:
        self.tickets.append((f"T-{len(self.tickets) + 1:06d}", identity, _ts(days_ago), author, text))

    def set_owner(self, identity: str, owner: str) -> None:
        i = self._index[identity]
        self.identities[i] = (*self.identities[i][:6], owner)


Planter = Callable[[OrgBuilder, str], list[dict]]


@dataclass(frozen=True)
class CaseType:
    name: str
    expected_verdict: Verdict
    plant: Planter


REGISTRY: dict[str, CaseType] = {}


def case_type(name: str, expected: Verdict) -> Callable[[Planter], Planter]:
    def register(fn: Planter) -> Planter:
        if name in REGISTRY:
            raise ValueError(f"case type {name!r} is already registered")
        REGISTRY[name] = CaseType(name, expected, fn)
        return fn
    return register


@case_type("nightly_job", expected="disable_first")
def nightly_job(org: OrgBuilder, target: str) -> list[dict]:
    for d in range(30):
        org.add_usage(target, _ts(d, hour=2), "warehouse", "job:nightly-report")
    org.add_dependency("pipeline:reports/nightly.yaml", target, "config_ref", "credentials: user account")
    return [{"kind": "config_ref", "ref": "pipeline:reports/nightly.yaml"}]


@case_type("orphaned_service_account", expected="ask_owner")
def orphaned_service_account(org: OrgBuilder, target: str) -> list[dict]:
    svc = org.rng.choice(org.services)
    org.set_owner(svc, target)
    org.add_dependency(svc, target, "owner_of", "service account whose only owner is the leaver")
    for consumer in org.rng.sample(["billing-api", "export-job", "alerts"], 3):
        org.add_dependency(f"service:{consumer}", svc, "consumes", "uses the service account")
    return [{"kind": "owner_of", "ref": svc}]


@case_type("policy_exception", expected="disable_first")
def policy_exception(org: OrgBuilder, target: str) -> list[dict]:
    org.add_dependency("policy:vendor-access-exception", target, "policy_reference", "named explicitly as an exception")
    return [{"kind": "policy_reference", "ref": "policy:vendor-access-exception"}]


@case_type("shared_account", expected="ask_owner")
def shared_account(org: OrgBuilder, target: str) -> list[dict]:
    org.add_usage(target, _ts(80), "ledger", "interactive")
    org.add_ticket(target, 400, "it-help", "Shared login for the finance close; do not rotate without telling finance.")
    return [{"kind": "ticket_note", "ref": "shared by finance"}]


@case_type("break_glass", expected="ask_owner")
def break_glass(org: OrgBuilder, target: str) -> list[dict]:
    org.add_ticket(target, 700, "security", "Break-glass account for IdP outage. Dormant on purpose, do not remove.")
    return [{"kind": "ticket_note", "ref": "break-glass"}]
