"""The only code that changes an identity, and only what a person approved.

``TargetSystem`` is the interface to the system that holds the accounts.
``SyntheticTarget`` is a writable copy of the generated org (``data/target.db``):
the evidence database stays read-only (``mode=ro``) and is never written.
``AwsIamTarget`` or ``KeycloakTarget`` can implement the same methods later.

The executor:
- dry-runs first: ``plan`` says what would change and changes nothing;
- runs an action only if the ``Approval`` names it for this request;
- deletes only an account it disabled itself earlier (disable first);
- can undo a disable with ``restore``;
- never runs the same action twice: each one has a key (``request_id:action``) and
  an action whose key is already ``done`` in the audit log is skipped;
- writes every outcome, refusals included, to the audit log.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .audit import AuditSink
from .guardrails import Action
from .models import ChangeRequest

Status = str  # "active", "disabled", or "missing" (deleted or never existed)


@runtime_checkable
class TargetSystem(Protocol):
    name: str

    def status(self, identity_id: str) -> Status: ...
    def disable(self, identity_id: str) -> None: ...
    def enable(self, identity_id: str) -> None: ...
    def delete(self, identity_id: str) -> None: ...


class SyntheticTarget:
    """A writable copy of the generated org, for demos and tests."""

    name = "synthetic"

    def __init__(self, db_path: str | Path):
        self._db = sqlite3.connect(db_path)

    @classmethod
    def copy_of(cls, org_db: str | Path, target_db: str | Path) -> SyntheticTarget:
        """A fresh copy of ``org_db``; the source is opened read-only."""
        Path(target_db).unlink(missing_ok=True)
        src = sqlite3.connect(f"file:{org_db}?mode=ro", uri=True)
        dst = sqlite3.connect(target_db)
        src.backup(dst)
        src.close()
        dst.close()
        return cls(target_db)

    def status(self, identity_id: str) -> Status:
        row = self._db.execute(
            "SELECT account_status FROM identities WHERE id = ?", (identity_id,)
        ).fetchone()
        return row[0] if row else "missing"

    def _set(self, identity_id: str, status: str) -> None:
        with self._db:
            self._db.execute(
                "UPDATE identities SET account_status = ? WHERE id = ?",
                (status, identity_id),
            )

    def disable(self, identity_id: str) -> None:
        self._set(identity_id, "disabled")

    def enable(self, identity_id: str) -> None:
        self._set(identity_id, "active")

    def delete(self, identity_id: str) -> None:
        with self._db:
            self._db.execute("DELETE FROM identities WHERE id = ?", (identity_id,))


class ExecutorError(RuntimeError):
    """An action was refused. The refusal is in the audit log too."""


@dataclass(frozen=True)
class Approval:
    """A person's answer: which of the allowed actions may run for one request."""

    request_id: str
    actions: tuple[Action, ...]
    by: str
    at: datetime


# what each action does to the account: (needed status before, status after)
STEPS: dict[str, tuple[Status, Status]] = {
    "disable": ("active", "disabled"),
    "restore": ("disabled", "active"),
    "delete": ("disabled", "missing"),
}


class Executor:
    def __init__(
        self,
        target: TargetSystem,
        audit: AuditSink,
        clock: Callable[[], datetime] = datetime.now,
    ):
        self.target, self.audit, self.clock = target, audit, clock

    def plan(
        self, request: ChangeRequest, actions: tuple[Action, ...]
    ) -> list[dict[str, Any]]:
        """Dry run: what each action would change. Nothing changes on the target."""
        status = self.target.status(request.identity_id)
        steps = []
        for action in actions:
            before, after = STEPS[action]
            steps.append(
                {
                    "action": action,
                    "identity_id": request.identity_id,
                    "from": status,
                    "to": after,
                    "ok": status == before,
                }
            )
            status = after if status == before else status
        self._log(request, "plan", "dry_run", steps=steps)
        return steps

    def disable(self, request: ChangeRequest, approval: Approval) -> dict[str, Any]:
        return self._run(request, "disable", approval)

    def delete(self, request: ChangeRequest, approval: Approval) -> dict[str, Any]:
        if not self._done(request, "disable"):
            self._refuse(
                request, "delete", "delete needs a disable by this request first"
            )
        return self._run(request, "delete", approval)

    def restore(self, request: ChangeRequest) -> dict[str, Any]:
        """Undo this request's disable. Needs no new approval: it only gives access back."""
        if not self._done(request, "disable"):
            self._refuse(request, "restore", "this request disabled nothing")
        return self._run(request, "restore", None)

    # -- internals

    def _run(
        self, request: ChangeRequest, action: str, approval: Approval | None
    ) -> dict[str, Any]:
        done = self._done(request, action)
        if done:
            return done  # idempotent: a retry or a resume never acts twice
        if approval is not None:
            if approval.request_id != request.request_id:
                self._refuse(request, action, f"approval is for {approval.request_id}")
            if action not in approval.actions:
                self._refuse(request, action, f"{action} was not approved")
        before, after = STEPS[action]
        status = self.target.status(request.identity_id)
        if status != before:
            self._refuse(request, action, f"account is {status}, expected {before}")
        getattr(self.target, "enable" if action == "restore" else action)(
            request.identity_id
        )
        return self._log(
            request,
            action,
            "done",
            by=approval.by if approval else "executor",
            status={"from": before, "to": after},
        )

    def _done(self, request: ChangeRequest, action: str) -> dict[str, Any] | None:
        key = f"{request.request_id}:{action}"
        return next(
            (
                e
                for e in self.audit.entries()
                if e.get("key") == key and e.get("outcome") == "done"
            ),
            None,
        )

    def _refuse(self, request: ChangeRequest, action: str, reason: str) -> None:
        self._log(request, action, "refused", reason=reason)
        raise ExecutorError(reason)

    def _log(
        self, request: ChangeRequest, action: str, outcome: str, **detail: Any
    ) -> dict[str, Any]:
        return self.audit.append(
            {
                "at": self.clock().isoformat(timespec="seconds"),
                "request_id": request.request_id,
                "identity_id": request.identity_id,
                "target": self.target.name,
                "action": action,
                "key": f"{request.request_id}:{action}",
                "outcome": outcome,
                **detail,
            }
        )
