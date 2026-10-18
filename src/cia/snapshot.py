"""Evidence fingerprints: has anything a report rests on changed since it was made?

No cache. Each check fetches fresh from the same five tools, so there is one source of
truth: what the tools return now. Each result is reduced to the part that matters and
hashed. Same hashes: the report still holds, and no model call is needed. Any change:
investigate again. A failed fetch raises, because "could not check" is never "unchanged".

What is left out on purpose:
- ``account_status``: our own disable changes it, and that is not new evidence;
- the moving usage window: usage is read over the longest window (10 years), so old
  events do not drop out and look like a change; a new event moves ``last_seen``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .sources import MAX_DEPTH, MAX_USAGE_DAYS, EvidenceSource

TOOLS = ("get_identity", "get_usage", "get_dependencies", "get_tickets", "get_owner")
SET_BY_US = ("account_status",)


def _record(row: dict[str, Any] | None) -> dict[str, Any] | None:
    return {k: v for k, v in row.items() if k not in SET_BY_US} if row else None


def read(source: EvidenceSource, identity_id: str) -> dict[str, Any]:
    """Fresh reads of the five tools, reduced to what a decision rests on."""
    usage = source.get_usage(identity_id, MAX_USAGE_DAYS)
    return {
        "get_identity": _record(source.get_identity(identity_id)),
        "get_usage": {k: usage[k] for k in ("event_count", "last_seen", "callers")},
        "get_dependencies": source.get_dependencies(identity_id, MAX_DEPTH),
        "get_tickets": source.get_tickets(identity_id),
        "get_owner": _record(source.get_owner(identity_id)),
    }


def fingerprint(source: EvidenceSource, identity_id: str) -> dict[str, str]:
    """One hash per tool. Raises if a source fails: a failed read is not "unchanged"."""
    return {
        tool: hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()
        for tool, data in read(source, identity_id).items()
    }


def changed(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """The tools whose evidence differs; a tool missing on either side counts as changed."""
    return [t for t in TOOLS if before.get(t) is None or before.get(t) != after.get(t)]
