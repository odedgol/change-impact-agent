"""The audit log: every guardrail decision, approval, dry run and action, append-only.

``AuditSink`` is the interface; ``JsonlAuditSink`` writes JSON lines to a file today,
and a Postgres sink with insert-only rights can implement the same two methods later.

Each entry holds the hash of the one before it (Q16), so ``verify()`` finds an entry
that was edited, removed or moved. The log is also how the executor stays idempotent:
an action whose key is already ``done`` in the log is never run twice.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

GENESIS = "0" * 64


@runtime_checkable
class AuditSink(Protocol):
    def append(self, entry: dict[str, Any]) -> dict[str, Any]: ...
    def entries(self) -> list[dict[str, Any]]: ...


def entry_hash(entry: dict[str, Any]) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, default=str).encode()
    ).hexdigest()


def verify(entries: list[dict[str, Any]]) -> list[str]:
    """Problems in the chain, empty when the log is intact."""
    problems, prev = [], GENESIS
    for i, e in enumerate(entries):
        if e.get("seq") != i:
            problems.append(f"entry {i}: seq is {e.get('seq')}")
        if e.get("prev") != prev:
            problems.append(f"entry {i}: does not follow the entry before it")
        if e.get("hash") != entry_hash(e):
            problems.append(f"entry {i}: content changed")
        prev = e.get("hash", "")
    return problems


class JsonlAuditSink:
    """One JSON object per line, opened for append only. Chained by hash."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def append(self, entry: dict[str, Any]) -> dict[str, Any]:
        log = self.entries()
        stored = {
            **entry,
            "seq": len(log),
            "prev": log[-1]["hash"] if log else GENESIS,
        }
        stored["hash"] = entry_hash(stored)
        with self.path.open("a") as f:
            f.write(json.dumps(stored, sort_keys=True, default=str) + "\n")
        return stored
