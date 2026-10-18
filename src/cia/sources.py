"""Where evidence comes from. The investigator only knows ``EvidenceSource``.

``SyntheticSource`` reads the generated org; ``AwsSource`` or ``KeycloakSource``
can implement the same five methods later without touching the agent.

Every source is read-only. ``SyntheticSource`` opens SQLite with ``mode=ro``,
so even a buggy or manipulated tool call cannot change state.

Sources return complete, deterministically ordered lists. Nothing is cut here:
a cut list can hide the one row that blocks a delete (an old "do not remove"
ticket, a consumer of an owned service account). Keeping the model's context
small is the caller's job, through ``paginate``, which always says how to fetch
the next page.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from .generator import NOW

PAGE_SIZE = 50
MAX_CALLERS = 20
MIN_USAGE_DAYS = 90  # a shorter window can make a nightly or monthly job look dormant
MAX_USAGE_DAYS = 3650
DEFAULT_DEPTH = (
    2  # the identity's dependents and theirs, e.g. consumers of an owned account
)
MAX_DEPTH = 4  # deep enough for user -> key -> secret -> pipeline -> team


@runtime_checkable
class EvidenceSource(Protocol):
    def get_identity(self, identity_id: str) -> dict | None: ...
    def get_usage(self, identity_id: str, days: int = 90) -> dict: ...
    def get_dependencies(
        self, identity_id: str, depth: int = DEFAULT_DEPTH
    ) -> list[dict]: ...
    def get_tickets(self, identity_id: str) -> list[dict]: ...
    def get_owner(self, identity_id: str) -> dict | None: ...


def paginate(rows: list[dict], offset: int = 0, limit: int = PAGE_SIZE) -> dict:
    """One page of a list, with the total and the offset of the next page (None on the last page)."""
    if offset < 0 or limit < 1:
        raise ValueError(
            f"offset must be >= 0 and limit >= 1 (got offset={offset}, limit={limit})"
        )
    end = offset + limit
    return {
        "rows": rows[offset:end],
        "total": len(rows),
        "offset": offset,
        "next_offset": end if end < len(rows) else None,
    }


class SyntheticSource:
    name = "synthetic"
    # Optional freshness metadata (not part of the five-method interface). The generated org records
    # no "last updated" time, so this stays None: unknown, never invented. Event timestamps are not it.
    updated_at: datetime | None = None

    def __init__(
        self, db_path: str, now: datetime = NOW, max_callers: int = MAX_CALLERS
    ):
        if max_callers < 1:
            raise ValueError(f"max_callers must be >= 1 (got {max_callers})")
        self._db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self._db.row_factory = sqlite3.Row
        self.now = now
        self.max_callers = max_callers

    def _now(self) -> str:
        """Nothing after ``now`` exists yet: moving ``now`` forward reveals it."""
        return self.now.isoformat(timespec="seconds")

    def _rows(self, sql: str, *args) -> list[dict]:
        return [dict(r) for r in self._db.execute(sql, args)]

    def get_identity(self, identity_id: str) -> dict | None:
        rows = self._rows("SELECT * FROM identities WHERE id = ?", identity_id)
        return rows[0] if rows else None

    def get_usage(self, identity_id: str, days: int = 90) -> dict:
        """Usage in the window, aggregated in SQL. The window is clamped to [90, 3650] days, and the
        result reports the window actually used."""
        window = min(max(int(days), MIN_USAGE_DAYS), MAX_USAGE_DAYS)
        since = (self.now - timedelta(days=window)).isoformat(timespec="seconds")
        callers = self._rows(
            "SELECT caller, COUNT(*) AS n, MAX(ts) AS last FROM usage_events "
            "WHERE identity_id = ? AND ts >= ? AND ts <= ? "
            "GROUP BY caller ORDER BY n DESC, caller",
            identity_id,
            since,
            self._now(),
        )
        top = callers[: self.max_callers]
        return {
            "window_days": window,
            "event_count": sum(c["n"] for c in callers),
            "last_seen": max((c["last"] for c in callers), default=None),
            "callers": {c["caller"]: c["n"] for c in top},
            "callers_omitted": len(callers) - len(top),
        }

    def get_dependencies(
        self, identity_id: str, depth: int = DEFAULT_DEPTH
    ) -> list[dict]:
        """Things that depend on this identity, hop by hop, clamped to 1-4 hops.

        Hop 1 points at the identity, hop 2 at a hop-1 thing (the consumers of an
        owned service account), and so on. Each row names the ``target`` it points
        at, its ``hop`` and ``via``: the path from the identity to that target.
        """
        depth = min(max(int(depth), 1), MAX_DEPTH)
        rows: list[dict] = []
        paths: dict[str, list[str]] = {identity_id: []}
        frontier = [identity_id]
        for hop in range(1, depth + 1):
            found = self._rows(
                "SELECT source, kind, detail, target_id AS target FROM dependencies "
                f"WHERE target_id IN ({','.join('?' * len(frontier))}) "
                "ORDER BY target_id, kind, source",
                *frontier,
            )
            frontier = []
            for r in found:
                rows.append({**r, "hop": hop, "via": paths[r["target"]]})
                if r["source"] not in paths:
                    paths[r["source"]] = [*paths[r["target"]], r["source"]]
                    frontier.append(r["source"])
            if not frontier:
                break
        return rows

    def get_tickets(self, identity_id: str) -> list[dict]:
        return self._rows(
            "SELECT id, ts, author, text FROM tickets WHERE identity_id = ? AND ts <= ? "
            "ORDER BY ts DESC, id",
            identity_id,
            self._now(),
        )

    def get_owner(self, identity_id: str) -> dict | None:
        ident = self.get_identity(identity_id)
        if ident and ident["owner_id"]:
            return self.get_identity(ident["owner_id"])
        return None
