"""Telling people about a change, where they already are.

One pure function, ``message``, turns a change (its status, report and identity) into
the same content for every channel: a title, what the agent found with its evidence
ids, what the guardrails allow, and where to see the whole trace. A ``Notifier``
only renders and sends it: ``ConsoleNotifier`` prints it, ``MemoryNotifier`` keeps it
for tests, a Slack notifier (week 4) posts it with buttons. The buttons come only from
the guardrails' allowed actions; the model's text is shown, never obeyed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import IO, Any, Protocol

from .models import RiskReport

MAX_FINDINGS = 3


@dataclass(frozen=True)
class Message:
    request_id: str
    title: str
    state: str  # what happens next, in one line
    lines: tuple[str, ...] = ()  # findings, each with its evidence ids
    actions: tuple[str, ...] = ()  # what a person may approve, from the guardrails
    link: str | None = None
    done: bool = False  # the change reached its outcome


class Notifier(Protocol):
    def post(self, message: Message) -> str:
        """Send a new message; returns a reference to update it later."""
        ...

    def update(self, ref: str, message: Message) -> None: ...


def message(
    change: dict[str, Any],
    report: RiskReport | None,
    identity: dict[str, Any] | None,
    link: str | None = None,
) -> Message:
    """The content of a change's message, from ``ChangeWorkflow.status`` and its report."""
    rid = change["request_id"]
    who = identity or {}
    name = who.get("name") or change.get("identity_id") or rid
    title = f"{name} ({who.get('department') or 'no department'}) left"
    lines: tuple[str, ...] = ()
    if report:
        title += f" · {report.verdict} ({report.confidence})"
        lines = tuple(
            f"{f.summary} [{' '.join(f.evidence_ids)}]"
            for f in report.findings[:MAX_FINDINGS]
        )
        if report.incomplete_reason:
            lines += (f"incomplete: {report.incomplete_reason}",)
    waiting, outcome = change.get("waiting_for"), change.get("outcome")
    actions: tuple[str, ...] = ()
    if waiting == "approval":
        actions = tuple(change.get("allowed") or ())
        state = f"Allowed: {', '.join(actions)}. Approve all or less, or decline."
    elif waiting == "quiet_period":
        then = "delete" if "delete" in change.get("approved", []) else "keep disabled"
        state = (
            f"Disabled. Quiet period until {str(change.get('until'))[:16]}: "
            f"any use restores it; otherwise {then}."
        )
    elif outcome:
        reason = change.get("reason")
        state = f"Outcome: {outcome}" + (f" ({reason})" if reason else "")
    else:
        state = "Investigating."
    return Message(
        rid,
        title,
        state,
        lines,
        actions,
        link,
        done=bool(outcome) and not waiting,
    )


def text(m: Message) -> str:
    """A plain-text rendering, for the console and for logs."""
    out = [f"[{m.request_id}] {m.title}", *(f"  • {line}" for line in m.lines)]
    out.append(f"  {m.state}")
    if m.actions:
        out.append(
            f"  Buttons: {' | '.join(f'Approve {a}' for a in m.actions)} | Decline"
        )
    if m.link:
        out.append(f"  Trace: {m.link}")
    return "\n".join(out)


class ConsoleNotifier:
    """Prints every message and update: the offline stand-in for Slack."""

    def __init__(self, out: IO[str] | None = None):
        self.out = out or sys.stdout

    def post(self, message: Message) -> str:
        print(text(message), file=self.out, flush=True)
        return message.request_id

    def update(self, ref: str, message: Message) -> None:
        print(f"(update) {text(message)}", file=self.out, flush=True)


@dataclass
class MemoryNotifier:
    """Keeps every message, for tests."""

    posted: list[Message] = field(default_factory=list)
    updates: list[tuple[str, Message]] = field(default_factory=list)

    def post(self, message: Message) -> str:
        self.posted.append(message)
        return message.request_id

    def update(self, ref: str, message: Message) -> None:
        self.updates.append((ref, message))
