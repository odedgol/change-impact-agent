import inspect
import sqlite3
from datetime import timedelta

import pytest

from cia.generator import generate
from cia.sources import MIN_USAGE_DAYS, EvidenceSource, SyntheticSource, paginate
from cia.tools import EvidenceTools


@pytest.fixture(scope="module")
def org(tmp_path_factory):
    db = tmp_path_factory.mktemp("org") / "org.db"
    truth = generate(db, n_identities=500, seed=7)
    return str(db), truth


def _case(truth, case_type):
    return next(c for c in truth["cases"] if c["case_type"] == case_type)


def test_synthetic_source_satisfies_the_interface(org):
    assert isinstance(SyntheticSource(org[0]), EvidenceSource)


@pytest.mark.parametrize(
    "name",
    ["get_identity", "get_usage", "get_dependencies", "get_tickets", "get_owner"],
)
def test_signatures_match_the_protocol(name):
    """isinstance only checks method names; this catches a source whose parameters drift."""
    proto = inspect.signature(getattr(EvidenceSource, name))
    impl = inspect.signature(getattr(SyntheticSource, name))
    assert list(impl.parameters) == list(proto.parameters)
    assert [p.default for p in impl.parameters.values()] == [
        p.default for p in proto.parameters.values()
    ]


def test_week1_name_still_works():
    assert EvidenceTools is SyntheticSource


def test_source_is_read_only(org):
    source = SyntheticSource(org[0])
    with pytest.raises(sqlite3.OperationalError):
        source._db.execute("DELETE FROM identities")
    with pytest.raises(sqlite3.OperationalError):
        source._db.execute("UPDATE identities SET status = 'deleted'")


def test_paginate_never_hides_rows():
    rows = [{"n": i} for i in range(5)]
    first = paginate(rows, limit=2)
    assert first == {"rows": rows[:2], "total": 5, "offset": 0, "next_offset": 2}
    seen = list(first["rows"])
    page = first
    while page["next_offset"] is not None:
        page = paginate(rows, offset=page["next_offset"], limit=2)
        seen += page["rows"]
    assert seen == rows


@pytest.mark.parametrize("offset,limit", [(-1, 10), (0, 0), (0, -1)])
def test_paginate_rejects_bad_arguments(offset, limit):
    with pytest.raises(ValueError):
        paginate([{"n": 1}], offset=offset, limit=limit)


def test_bad_caller_cap_is_rejected(org):
    with pytest.raises(ValueError):
        SyntheticSource(org[0], max_callers=0)


def test_owned_account_consumers_name_their_target(org):
    db, truth = org
    case = _case(truth, "orphaned_service_account")
    deps = SyntheticSource(db).get_dependencies(case["identity_id"])
    owned = {d["source"] for d in deps if d["kind"] == "owner_of"}
    assert owned
    transitive = [d for d in deps if d["target"] != case["identity_id"]]
    assert transitive and all(d["target"] in owned for d in transitive)
    # direct rows come first, so the leaver's own links lead the list
    assert deps[0]["target"] == case["identity_id"]


def test_dependencies_are_deterministic(org):
    db, truth = org
    case = _case(truth, "orphaned_service_account")
    assert SyntheticSource(db).get_dependencies(case["identity_id"]) == SyntheticSource(
        db
    ).get_dependencies(case["identity_id"])


@pytest.mark.parametrize("days", [0, -5, 7])
def test_short_usage_windows_are_clamped(org, days):
    """A model asking for a tiny window must not make a nightly job look dormant."""
    db, truth = org
    case = _case(truth, "nightly_job")
    usage = SyntheticSource(db).get_usage(case["identity_id"], days)
    assert usage["window_days"] == MIN_USAGE_DAYS
    assert usage["event_count"] > 0


def test_usage_aggregates_match_raw_rows(org):
    db, truth = org
    case = _case(truth, "nightly_job")
    source = SyntheticSource(db)
    usage = source.get_usage(case["identity_id"], 365)
    since = (source.now - timedelta(days=365)).isoformat(timespec="seconds")
    count, last = (
        sqlite3.connect(db)
        .execute(
            "SELECT COUNT(*), MAX(ts) FROM usage_events WHERE identity_id = ? AND ts >= ?",
            (case["identity_id"], since),
        )
        .fetchone()
    )
    assert (usage["event_count"], usage["last_seen"]) == (count, last)
    assert usage["callers_omitted"] == 0
    assert sum(usage["callers"].values()) == count


@pytest.mark.parametrize(
    "depth,hops", [(0, {1}), (1, {1}), (3, {1, 2, 3}), (9, {1, 2, 3, 4})]
)
def test_dependency_depth_is_clamped_to_one_to_four_hops(org, depth, hops):
    db, truth = org
    case = _case(truth, "credential_chain")
    rows = SyntheticSource(db).get_dependencies(case["identity_id"], depth=depth)
    assert {r["hop"] for r in rows} == hops
