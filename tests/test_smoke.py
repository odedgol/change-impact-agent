"""The smoke stream the Rust UI reads: what it covers, and that the UI's recorded copy is current."""

import io
import json
import re
from pathlib import Path

import pytest

from cia.generator import generate
from cia.smoke import EVENT_TYPES, run

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tui" / "tests" / "fixtures" / "smoke.jsonl"


@pytest.fixture(scope="module")
def events(tmp_path_factory):
    db = tmp_path_factory.mktemp("org") / "org.db"
    truth = generate(db, n_identities=500, seed=7)  # the same org as `make seed`
    out = io.StringIO()
    run(str(db), truth["cases"], out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def without_time(event):
    return {k: v for k, v in event.items() if k != "t"}


def test_smoke_reaches_every_guardrail_and_deletes_nothing_unsafe(events):
    assert {e["type"] for e in events} == set(EVENT_TYPES) - {
        "run_error"
    }  # run_error needs a crashing model
    assert sum(e["type"] == "run" for e in events) == 13
    expected = None
    for e in events:
        if e["type"] == "run":
            expected = e["expected_verdict"]
        if e["type"] == "report":
            assert not (
                e["verdict"] == "safe_to_delete" and expected != "safe_to_delete"
            )
    assert [e["outcome"] for e in events if e["type"] == "outcome"] == [
        "deleted",
        "declined",
    ]


def test_the_uis_recorded_stream_is_current(events):
    """tui/tests/fixtures/smoke.jsonl is what `make smoke` emits today; refresh it with `make tui-fixture`."""
    recorded = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
    assert [without_time(e) for e in recorded] == [without_time(e) for e in events]


def test_the_ui_parses_every_event_type():
    source = (ROOT / "tui" / "src" / "events.rs").read_text()
    variants = set(re.findall(r"^    (\w+)[ ({]", source, re.M))
    for t in EVENT_TYPES:
        assert "".join(part.title() for part in t.split("_")) in variants, t
