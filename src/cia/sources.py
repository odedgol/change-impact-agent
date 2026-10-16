"""Where evidence comes from. The investigator only knows ``EvidenceSource``.

``SyntheticSource`` reads the generated org; ``AwsSource`` or ``KeycloakSource``
can implement the same five methods later without touching the agent.

Every source is read-only. ``SyntheticSource`` opens SQLite with ``mode=ro``,
so even a buggy or manipulated tool call cannot change state.

List results are capped so one request can't flood the model's context. A cap
never hides data silently: a truncated list ends with a ``truncated`` marker
that says how many rows were left out, so the investigator can't mistake a cut
list for a complete one.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from .generator import NOW

MAX_ROWS = 50
MAX_CALLERS = 20


@runtime_checkable
class EvidenceSource(Protocol):
    def get_identity(self, identity_id: str) -> dict | None: ...
    def get_usage(self, identity_id: str, days: int = 90) -> dict: ...
    def get_dependencies(self, identity_id: str) -> list[dict]: ...
    def get_tickets(self, identity_id: str) -> list[dict]: ...
    def get_owner(self, identity_id: str) -> dict | None: ...


def cap_rows(rows: list[dict], max_rows: int) -> list[dict]:
    if len(rows) <= max_rows:
        return rows
    return rows[:max_rows] + [{"truncated": True, "omitted": len(rows) - max_rows}]


class SyntheticSource:
    def __init__(self, db_path: str, now: datetime = NOW, max_rows: int = MAX_ROWS, max_callers: int = MAX_CALLERS):
        self._db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self._db.row_factory = sqlite3.Row
        self.now = now
        self.max_rows = max_rows
        self.max_callers = max_callers

    def _rows(self, sql: str, *args) -> list[dict]:
        return [dict(r) for r in self._db.execute(sql, args)]

    def get_identity(self, identity_id: str) -> dict | None:
        rows = self._rows("SELECT * FROM identities WHERE id = ?", identity_id)
        return rows[0] if rows else None

    def get_usage(self, identity_id: str, days: int = 90) -> dict:
        since = (self.now - timedelta(days=days)).isoformat(timespec="seconds")
        events = self._rows(
            "SELECT ts, resource, caller FROM usage_events WHERE identity_id = ? AND ts >= ? ORDER BY ts DESC",
            identity_id, since,
        )
        counts: dict[str, int] = {}
        for e in events:
            counts[e["caller"]] = counts.get(e["caller"], 0) + 1
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: self.max_callers]
        return {"window_days": days, "event_count": len(events),
                "last_seen": events[0]["ts"] if events else None, "callers": dict(top),
                "callers_omitted": len(counts) - len(top)}

    def get_dependencies(self, identity_id: str) -> list[dict]:
        """Things that point at this identity, plus the consumers of accounts it owns."""
        direct = self._rows("SELECT source, kind, detail FROM dependencies WHERE target_id = ?", identity_id)
        for d in list(direct):
            if d["kind"] == "owner_of":
                direct += self._rows("SELECT source, kind, detail FROM dependencies WHERE target_id = ?", d["source"])
        return cap_rows(direct, self.max_rows)

    def get_tickets(self, identity_id: str) -> list[dict]:
        rows = self._rows("SELECT id, ts, author, text FROM tickets WHERE identity_id = ? ORDER BY ts DESC", identity_id)
        return cap_rows(rows, self.max_rows)

    def get_owner(self, identity_id: str) -> dict | None:
        ident = self.get_identity(identity_id)
        if ident and ident["owner_id"]:
            return self.get_identity(ident["owner_id"])
        return None
