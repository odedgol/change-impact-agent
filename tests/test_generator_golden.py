"""Golden output: the same seed must produce byte-identical orgs.

The ground truth, every test and every bench result depend on the generated
org, so a refactor of the generator must not change a single row. The hashes
were first recorded before the case types moved to commands (R1), then changed on
purpose in PR 3.1, which added the credential_chain case type, and in PR 4.2, which
added the five red-team case types. To change them on purpose, say why in the PR.
"""

import hashlib
import json
import sqlite3

import pytest

from cia.generator import generate

TABLES = ["identities", "usage_events", "dependencies", "tickets"]

GOLDEN = {
    (500, 7): "d38eef230812fcbd95324d55fac7559f59f79bb774d499fd0749d975063a4ef8",
    (2000, 1): "0d7e61dd9e47031bff59c7dacaec8c8c29fd81bd84a3ccd4190aab5989065419",
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
