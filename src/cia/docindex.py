"""Document search for the agent: one ``DocIndex`` interface, two stores (week 5, PR 5.3).

    MemoryIndex    in process; tests, CI and runs without Docker
    PgVectorIndex  Postgres + pgvector (``make db-up``); the same SQL runs in a private
                   cloud or on RDS, and OpenSearch can sit behind the same interface later

Search is hybrid (Q9): words first, for exact names and keys (FIN-221, nightly-report),
and vectors for paraphrases ("who still needs this login"). The two lists are merged by
reciprocal rank fusion. Every hit carries its document id, so the agent can cite it.

Embeddings come from an ``Embedder``: Ollama ``nomic-embed-text`` locally (free), or a
hashing embedder that needs nothing and is good enough for tests (Q32).

What gets indexed comes from the simulated company: runbooks, tickets with their
comments, owners' answers, and one log summary per person per day (``index_world``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

WORD = re.compile(r"[a-z0-9][a-z0-9_.:/-]*")
RRF_K = 60  # reciprocal rank fusion: the usual constant
PG_DSN = "postgresql://cia:cia@localhost:5433/cia"


@dataclass(frozen=True)
class Doc:
    id: str
    kind: str  # runbook | ticket | reply | logs
    title: str
    text: str
    about: str | None = None  # the person it is about, when there is one
    day: int = 0


@dataclass(frozen=True)
class Hit:
    doc: Doc
    score: float


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Words and word pairs hashed into a fixed vector: no model, no network."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * self.dim
            words = WORD.findall(text.lower())
            for t in words + [
                f"{a} {b}" for a, b in zip(words, words[1:], strict=False)
            ]:
                h = int(hashlib.blake2b(t.encode(), digest_size=8).hexdigest(), 16)
                v[h % self.dim] += 1.0 if h & 1 else -1.0
            out.append(_unit(v))
        return out


class OllamaEmbedder:
    """``nomic-embed-text`` (768 numbers) from a local Ollama."""

    def __init__(self, model: str = "nomic-embed-text", url: str | None = None):
        self.model = model
        self.url = (
            url or os.environ.get("CIA_OLLAMA_URL") or "http://localhost:11434"
        ).rstrip("/")
        self.dim = 768

    def embed(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": texts}).encode()
        req = urllib.request.Request(
            f"{self.url}/api/embed", body, {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - local URL
            vectors = json.load(r)["embeddings"]
        self.dim = len(vectors[0]) if vectors else self.dim
        return [_unit(v) for v in vectors]


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _words(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


def fuse(*ranked: list[str]) -> list[tuple[str, float]]:
    """Reciprocal rank fusion of several ranked id lists."""
    score: dict[str, float] = {}
    for ids in ranked:
        for rank, i in enumerate(ids):
            score[i] = score.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
    return sorted(score.items(), key=lambda kv: -kv[1])


class DocIndex(Protocol):
    def add(self, docs: list[Doc]) -> int: ...
    def search(
        self,
        query: str,
        k: int = 5,
        about: str | None = None,
        kinds: list[str] | None = None,
    ) -> list[Hit]: ...
    def count(self) -> int: ...


class MemoryIndex:
    """The whole index in a dict; the same behaviour as ``PgVectorIndex``."""

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or HashEmbedder()
        self.docs: dict[str, Doc] = {}
        self.vectors: dict[str, list[float]] = {}

    def add(self, docs: list[Doc]) -> int:
        new = [d for d in docs if self.docs.get(d.id) != d]
        if new:
            for d, v in zip(
                new,
                self.embedder.embed([f"{d.title}\n{d.text}" for d in new]),
                strict=True,
            ):
                self.docs[d.id], self.vectors[d.id] = d, v
        return len(new)

    def search(self, query, k=5, about=None, kinds=None):
        pool = [
            d
            for d in self.docs.values()
            if (about is None or d.about in (about, None))
            and (not kinds or d.kind in kinds)
        ]
        q = self.embedder.embed([query])[0]
        by_vec = sorted(
            pool,
            key=lambda d: (
                -sum(a * b for a, b in zip(q, self.vectors[d.id], strict=True))
            ),
        )
        terms = _words(query)
        by_word = sorted(
            (d for d in pool if terms & _words(f"{d.title} {d.text}")),
            key=lambda d: -len(terms & _words(f"{d.title} {d.text}")),
        )
        fused = fuse([d.id for d in by_vec[: k * 4]], [d.id for d in by_word[: k * 4]])
        return [Hit(self.docs[i], round(s, 5)) for i, s in fused[:k]]

    def count(self) -> int:
        return len(self.docs)


class PgVectorIndex:
    """Postgres with pgvector. One table; upserts by document id; hybrid search in SQL.
    The table name goes in as an identifier and every value as a parameter."""

    def __init__(
        self,
        dsn: str | None = None,
        embedder: Embedder | None = None,
        table: str = "cia_docs",
    ):
        import psycopg  # only here: the rest of the agent runs without Postgres
        from psycopg import sql

        self.sql = sql
        self.embedder = embedder or OllamaEmbedder()
        self.t = sql.Identifier(table)
        self.conn = psycopg.connect(
            dsn or os.environ.get("CIA_PG_DSN") or PG_DSN, autocommit=True
        )
        dim = len(self.embedder.embed(["dimension check"])[0])
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.execute(
            self._q(
                "CREATE TABLE IF NOT EXISTS {t} (id TEXT PRIMARY KEY, kind TEXT, "
                "title TEXT, text TEXT, about TEXT, day INTEGER, embedding vector({dim}), "
                "words tsvector GENERATED ALWAYS AS "
                "(to_tsvector('simple', title || ' ' || text)) STORED)",
                dim=self.sql.Literal(dim),
            )
        )

    def _q(self, text: str, **parts: Any):
        return self.sql.SQL(text).format(t=self.t, **parts)  # type: ignore[arg-type]

    def add(self, docs: list[Doc]) -> int:
        if not docs:
            return 0
        have = {
            r[0]: (r[1], r[2])
            for r in self.conn.execute(
                self._q("SELECT id, title, text FROM {t} WHERE id = ANY(%s)"),
                ([d.id for d in docs],),
            )
        }
        new = [d for d in docs if have.get(d.id) != (d.title, d.text)]
        if not new:
            return 0
        vectors = self.embedder.embed([f"{d.title}\n{d.text}" for d in new])
        with self.conn.cursor() as cur:
            cur.executemany(
                self._q(
                    "INSERT INTO {t} (id, kind, title, text, about, day, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s::vector) ON CONFLICT (id) DO "
                    "UPDATE SET kind = EXCLUDED.kind, title = EXCLUDED.title, "
                    "text = EXCLUDED.text, about = EXCLUDED.about, day = EXCLUDED.day, "
                    "embedding = EXCLUDED.embedding"
                ),
                [
                    (d.id, d.kind, d.title, d.text, d.about, d.day, str(v))
                    for d, v in zip(new, vectors, strict=True)
                ],
            )
        return len(new)

    def search(self, query, k=5, about=None, kinds=None):
        where, args = [self.sql.SQL("TRUE")], []
        if about is not None:
            where.append(self.sql.SQL("(about = %s OR about IS NULL)"))
            args.append(about)
        if kinds:
            where.append(self.sql.SQL("kind = ANY(%s)"))
            args.append(kinds)
        cond = self.sql.SQL(" AND ").join(where)
        q = str(self.embedder.embed([query])[0])
        by_vec = [
            r[0]
            for r in self.conn.execute(
                self._q(
                    "SELECT id FROM {t} WHERE {cond} "
                    "ORDER BY embedding <=> %s::vector LIMIT %s",
                    cond=cond,
                ),
                (*args, q, k * 4),
            )
        ]
        tsq = _tsquery(query)
        by_word = [
            r[0]
            for r in self.conn.execute(
                self._q(
                    "SELECT id FROM {t} WHERE {cond} AND words @@ to_tsquery('simple', %s) "
                    "ORDER BY ts_rank(words, to_tsquery('simple', %s)) DESC LIMIT %s",
                    cond=cond,
                ),
                (*args, tsq, tsq, k * 4),
            )
        ]
        fused = fuse(by_vec, by_word)[:k]
        if not fused:
            return []
        rows = {
            r[0]: Doc(*r)
            for r in self.conn.execute(
                self._q(
                    "SELECT id, kind, title, text, about, day FROM {t} WHERE id = ANY(%s)"
                ),
                ([i for i, _ in fused],),
            )
        }
        return [Hit(rows[i], round(s, 5)) for i, s in fused if i in rows]

    def count(self) -> int:
        row = self.conn.execute(self._q("SELECT COUNT(*) FROM {t}")).fetchone()
        return int(row[0]) if row else 0

    def drop(self) -> None:
        self.conn.execute(self._q("DROP TABLE IF EXISTS {t}"))


def _tsquery(text: str) -> str:
    """Words joined with OR, each quoted, so no user text is ever query syntax."""
    words = sorted(w.replace("'", "") for w in _words(text) if len(w) > 1)
    return " | ".join(f"'{w}'" for w in words) or "''"


def make_index(kind: str | None = None) -> DocIndex:
    """``CIA_DOCINDEX``: ``pgvector`` (Docker, Ollama embeddings) or ``memory`` (hashing)."""
    kind = kind or os.environ.get("CIA_DOCINDEX", "memory")
    if kind == "pgvector":
        emb = (
            OllamaEmbedder()
            if os.environ.get("CIA_EMBEDDER", "ollama") == "ollama"
            else HashEmbedder()
        )
        return PgVectorIndex(embedder=emb)
    return MemoryIndex()


def world_docs(world: Any) -> list[Doc]:
    """Everything in the simulated company worth searching, up to today."""
    docs = [
        Doc(d["id"], d["kind"], d["title"], d["text"], None, d["day"])
        for d in world.docs()
        if d["kind"] == "runbook"
    ]
    leavers = world.resignations()
    for p in leavers:
        pid, name = p["id"], p["name"]
        for t in world.tickets(pid):
            notes = "\n".join(
                f"{c['date']} {c['author']}: {c['text']}" for c in t["comments"]
            )
            docs.append(
                Doc(
                    f"ticket-{t['key']}",
                    "ticket",
                    f"{t['key']} {t['summary']} ({t['status']})",
                    f"Reported by {t['reporter']} on {t['created']}. {t['text']}\n{notes}".strip(),
                    pid,
                )
            )
        for i, m in enumerate(world.messages(pid)):
            docs.append(
                Doc(
                    f"msg-{pid}-{i}",
                    "reply",
                    f"Message about {name} on {m['date']}",
                    f"{m['sender'] or 'agent'}: {m['text']}",
                    pid,
                )
            )
        by_day: dict[str, list[str]] = {}
        for log in world.logs(pid, world.day - 14):
            if log["action"] != "login" or log["via"] != "interactive":
                by_day.setdefault(log["date"], []).append(
                    f"{log['ts']} {log['action']} via {log['via']}: {log['result']} ({log['detail']})"
                )
        for date, lines in by_day.items():
            docs.append(
                Doc(
                    f"logs-{pid}-{date}",
                    "logs",
                    f"Activity of {name} on {date}",
                    "\n".join(lines),
                    pid,
                )
            )
    return docs


def index_world(world: Any, index: DocIndex) -> int:
    """Add what is new since the last call; returns how many documents changed."""
    return index.add(world_docs(world))
