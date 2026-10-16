"""Golden output: the same seed must produce byte-identical orgs.

The ground truth, every test and every bench result depend on the generated
org, so a refactor of the generator must not change a single row. The hashes
below were recorded from the generator before the case-registry refactor.
To change them on purpose, say why in the PR.
"""
import hashlib
import json
import sqlite3

import pytest

from cia.generator import generate

TABLES = ["identities", "usage_events", "dependencies", "tickets"]

GOLDEN = {
    (500, 7): "6d6fe3d1f184a1cb327814f3a53b96e4da2d30684894aa6dddc36549bbfb3717",
    (2000, 1): "088dd57136f5bff199c1aff804da8e2cb5fc99b42ec7c3848b20803d410e3eab",
}


def org_digest(tmp_path, n_identities: int, seed: int) -> str:
    db = tmp_path / "org.db"
    truth = generate(db, n_identities=n_identities, seed=seed)
    h = hashlib.sha256(json.dumps(truth, sort_keys=True).encode())
    conn = sqlite3.connect(db)
    for table in TABLES:  # rowid order, so a change in insertion order also shows up
        for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid"):
            h.update(repr((table, row)).encode())
    conn.close()
    return h.hexdigest()


@pytest.mark.parametrize("n_identities,seed", list(GOLDEN))
def test_generated_org_matches_golden(tmp_path, n_identities, seed):
    assert org_digest(tmp_path, n_identities, seed) == GOLDEN[(n_identities, seed)]
