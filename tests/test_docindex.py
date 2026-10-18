"""Document search: one contract for both stores; pgvector runs when CIA_PG_DSN is set."""

import os
import uuid

import pytest

from cia.demo_org import build as build_demo
from cia.docindex import Doc, HashEmbedder, MemoryIndex, fuse, index_world, world_docs
from cia.world import build as build_world

DOCS = [
    Doc(
        "rb-1",
        "runbook",
        "Moving a job off a person's key",
        "Ask the job's owner to create a service account and point the job at it.",
    ),
    Doc(
        "rb-2",
        "runbook",
        "Break-glass accounts",
        "Emergency logins for an identity provider outage; dormant on purpose.",
    ),
    Doc(
        "t-1",
        "ticket",
        "FIN-221 Keep the account until the quarter close",
        "Finance still signs in with it.",
        "u1",
    ),
    Doc("t-2", "ticket", "HR-201 Offboarding: Wei Lee", "Wei Lee resigned.", "u2"),
]


def stores():
    yield "memory", lambda: MemoryIndex(HashEmbedder())
    if os.environ.get("CIA_PG_DSN"):
        from cia.docindex import PgVectorIndex

        yield (
            "pgvector",
            lambda: PgVectorIndex(
                embedder=HashEmbedder(), table=f"t_{uuid.uuid4().hex[:10]}"
            ),
        )


@pytest.fixture(params=[name for name, _ in stores()])
def index(request):
    make = dict(stores())[request.param]
    ix = make()
    yield ix
    drop = getattr(ix, "drop", None)
    if drop:
        drop()


def test_exact_words_find_the_ticket(index):
    index.add(DOCS)
    assert index.count() == 4
    hits = index.search("FIN-221", k=2)
    assert hits[0].doc.id == "t-1"


def test_about_limits_to_one_person_and_shared_docs(index):
    index.add(DOCS)
    ids = {h.doc.id for h in index.search("account", k=10, about="u2")}
    assert "t-1" not in ids and "t-2" in ids and "rb-1" in ids


def test_kinds_filter_and_upsert_is_idempotent(index):
    assert index.add(DOCS) == 4
    assert index.add(DOCS) == 0  # nothing changed, nothing embedded again
    changed = Doc("t-1", "ticket", DOCS[2].title, "Closed: the close is done.", "u1")
    assert index.add([changed]) == 1 and index.count() == 4
    hits = index.search("quarter close", kinds=["runbook"])
    assert hits and all(h.doc.kind == "runbook" for h in hits)


def test_query_text_is_never_sql_or_query_syntax(index):
    index.add(DOCS)
    assert isinstance(index.search("'); DROP TABLE x; -- | & !"), list)
    assert index.count() == 4


def test_fusion_rewards_agreement():
    ranked = fuse(["a", "b", "c"], ["b", "a"])
    assert [i for i, _ in ranked][:2] in (["a", "b"], ["b", "a"]) and ranked[-1][
        0
    ] == "c"


def test_the_world_is_indexed_and_grows_each_day(tmp_path):
    demo = tmp_path / "demo"
    build_demo(demo)
    world = build_world(tmp_path / "world", demo)
    world.next_day()
    ix = MemoryIndex(HashEmbedder())
    first = index_world(world, ix)
    assert first > 0 and {d.kind for d in world_docs(world)} >= {
        "runbook",
        "ticket",
        "logs",
    }
    assert index_world(world, ix) == 0  # same day: nothing new
    world.next_day()
    assert index_world(world, ix) > 0  # a new day of job runs
    hits = ix.search("break-glass emergency login")
    assert hits[0].doc.id == "rb-break-glass"
