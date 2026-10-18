"""The realistic demo org: names that read like a company, the same evidence and answers."""

import json
import sqlite3

import pytest

from cia.bench import RuleLLM, run
from cia.demo_org import N_IDENTITIES, build


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    truth = build(out)
    return out, truth


def test_every_identity_has_a_real_looking_unique_name(demo):
    out, truth = demo
    conn = sqlite3.connect(out / "org.db")
    rows = conn.execute("SELECT id, kind, name FROM identities").fetchall()
    assert len(rows) == N_IDENTITIES
    assert len({name for _, _, name in rows}) == N_IDENTITIES
    assert not any(name.startswith("user-") for _, _, name in rows)
    assert all(name.startswith("svc-") for _, kind, name in rows if kind == "service")
    assert truth["style"] == "demo"
    assert json.loads((out / "truth.json").read_text())["cases"] == truth["cases"]


def test_the_lookalike_wears_a_decoys_name_with_one_letter_swapped(demo):
    _, truth = demo
    by_type = {c["case_type"]: c for c in truth["cases"]}
    look = by_type["lookalike_name"]["name"]
    decoys = [c["name"] for c in truth["cases"] if c["case_type"] == "decoy_safe"]
    assert look not in decoys
    assert any(
        len(look) == len(d) and sum(a != b for a, b in zip(look, d, strict=True)) == 1
        for d in decoys
    )


def test_ids_inside_dependency_names_become_handles(demo):
    out, truth = demo
    conn = sqlite3.connect(out / "org.db")
    ids = {i for (i,) in conn.execute("SELECT id FROM identities WHERE kind='human'")}
    for source, target in conn.execute("SELECT source, target_id FROM dependencies"):
        for text in (source, target):
            assert text in ids or not any(i in text for i in ids)
    chain = next(c for c in truth["cases"] if c["case_type"] == "credential_chain")
    handle = chain["name"].lower().replace(" ", ".")
    assert any(handle in h["ref"] for h in chain["hidden_dependencies"])


def test_descriptions_name_the_person(demo):
    _, truth = demo
    for c in truth["cases"]:
        assert c["name"] in c["description"]


def test_the_rule_bench_gives_the_same_answers_on_the_demo_org(demo):
    out, _ = demo
    result = run(str(out / "org.db"), str(out / "truth.json"), RuleLLM(), "fake", None)
    s = result["scores"]
    assert s["accuracy"] == 1.0 and s["unsafe_deletes"] == 0
    assert (s["wrong_deletes"], s["unapproved_actions"]) == (0, 0)
