"""Read-only evidence tools. The agent gets these and nothing that can write.

The database is opened with SQLite's read-only mode, so even a buggy or
manipulated tool call cannot change state. This is the first guardrail.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from .generator import NOW


class EvidenceTools:
    def __init__(self, db_path: str, now: datetime = NOW):
        self._db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self._db.row_factory = sqlite3.Row
        self.now = now

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
        callers: dict[str, int] = {}
        for e in events:
            callers[e["caller"]] = callers.get(e["caller"], 0) + 1
        return {"window_days": days, "event_count": len(events),
                "last_seen": events[0]["ts"] if events else None, "callers": callers}

    def get_dependencies(self, identity_id: str) -> list[dict]:
        """Things that point at this identity, plus the consumers of accounts it owns."""
        direct = self._rows("SELECT source, kind, detail FROM dependencies WHERE target_id = ?", identity_id)
        for d in list(direct):
            if d["kind"] == "owner_of":
                direct += self._rows(
                    "SELECT source, kind, detail FROM dependencies WHERE target_id = ?", d["source"])
        return direct

    def get_tickets(self, identity_id: str) -> list[dict]:
        return self._rows("SELECT id, ts, author, text FROM tickets WHERE identity_id = ? ORDER BY ts DESC", identity_id)

    def get_owner(self, identity_id: str) -> dict | None:
        ident = self.get_identity(identity_id)
        if ident and ident["owner_id"]:
            return self.get_identity(ident["owner_id"])
        return None
