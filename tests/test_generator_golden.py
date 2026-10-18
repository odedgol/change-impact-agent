"""Golden output: the same seed must produce byte-identical orgs.

The ground truth, every test and every bench result depend on the generated
org, so a refactor of the generator must not change a single row. The hashes
were first recorded before the case types moved to commands (R1), then changed on
purpose in PR 3.1, which added the credential_chain case type, and in PR 4.2, which
added the five red-team case types, and when each request got a description. To change them on purpose, say why in the PR.
"""

import hashlib
import json
import sqlite3

import pytest

from cia.generator import generate

TABLES = ["identities", "usage_events", "dependencies", "tickets"]

GOLDEN = {
    (500, 7): "33860f9d04f9dd9cebe0b7e67c763f065ad68926058bb7059900ba018d64a054",
    (2000, 1): "9df0b4757f166953e27a1899a5b7ced7efd1e4ac6402d2d3c66a3eed13ef76bb",
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
