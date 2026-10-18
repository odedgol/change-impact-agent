"""The audit log is append-only and chained: any edit, removal or reordering shows."""

import json

import pytest

from cia.audit import AuditSink, JsonlAuditSink, verify


@pytest.fixture
def sink(tmp_path):
    s = JsonlAuditSink(tmp_path / "audit.jsonl")
    for i in range(3):
        s.append({"action": "disable", "n": i})
    return s


def rewrite(sink, entries):
    sink.path.write_text("".join(json.dumps(e) + "\n" for e in entries))


def test_entries_are_numbered_and_chained(sink):
    log = sink.entries()
    assert [e["seq"] for e in log] == [0, 1, 2]
    assert [e["prev"] for e in log[1:]] == [e["hash"] for e in log[:-1]]
    assert verify(log) == []
    assert isinstance(sink, AuditSink)


def test_an_edited_entry_is_found(sink):
    log = sink.entries()
    log[1]["action"] = "delete"
    rewrite(sink, log)
    assert any("entry 1: content changed" in p for p in verify(sink.entries()))


def test_a_removed_entry_is_found(sink):
    log = sink.entries()
    rewrite(sink, [log[0], log[2]])
    assert verify(sink.entries())


def test_reordered_entries_are_found(sink):
    log = sink.entries()
    rewrite(sink, [log[1], log[0], log[2]])
    assert verify(sink.entries())


def test_append_never_rewrites_what_is_there(sink):
    before = sink.path.read_text()
    sink.append({"action": "restore"})
    assert sink.path.read_text().startswith(before)
