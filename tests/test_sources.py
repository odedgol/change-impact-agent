import sqlite3

import pytest

from cia.generator import generate
from cia.sources import EvidenceSource, SyntheticSource, cap_rows
from cia.tools import EvidenceTools


@pytest.fixture(scope="module")
def org(tmp_path_factory):
    db = tmp_path_factory.mktemp("org") / "org.db"
    truth = generate(db, n_identities=500, seed=7)
    return str(db), truth


def test_synthetic_source_satisfies_the_interface(org):
    assert isinstance(SyntheticSource(org[0]), EvidenceSource)


def test_week1_name_still_works(org):
    assert EvidenceTools is SyntheticSource


def test_source_is_read_only(org):
    source = SyntheticSource(org[0])
    with pytest.raises(sqlite3.OperationalError):
        source._db.execute("DELETE FROM identities")
    with pytest.raises(sqlite3.OperationalError):
        source._db.execute("UPDATE identities SET status = 'deleted'")


def test_cap_rows_marks_truncation():
    rows = [{"n": i} for i in range(5)]
    assert cap_rows(rows, 5) == rows
    capped = cap_rows(rows, 2)
    assert capped[:2] == rows[:2]
    assert capped[-1] == {"truncated": True, "omitted": 3}


def test_capped_dependencies_never_hide_silently(org):
    db, truth = org
    case = next(c for c in truth["cases"] if c["case_type"] == "orphaned_service_account")
    full = SyntheticSource(db).get_dependencies(case["identity_id"])
    assert len(full) >= 2
    capped = SyntheticSource(db, max_rows=1).get_dependencies(case["identity_id"])
    assert capped[0] == full[0]
    assert capped[-1] == {"truncated": True, "omitted": len(full) - 1}


def test_usage_reports_omitted_callers(org):
    db, truth = org
    case = next(c for c in truth["cases"] if c["case_type"] == "nightly_job")
    usage = SyntheticSource(db, max_callers=0).get_usage(case["identity_id"], 365)
    assert usage["callers"] == {}
    assert usage["callers_omitted"] >= 1
    assert usage["event_count"] > 0
