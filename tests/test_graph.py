from pathlib import Path

import pytest
from langchain_core.runnables import RunnableConfig

from cia.generator import generate
from cia.graph import (
    DOCS_BEGIN,
    DOCS_END,
    REQUIRED_FOR_DELETE,
    ROUTE_LABELS,
    coverage,
    docs_block,
    drawings,
    drop_bad_citations,
    finish_report,
    memory_checkpointer,
    mermaid,
    recursion_limit,
    write_docs,
)
from cia.investigator import Investigator
from cia.llm import FakeLLM, LLMTurn, ToolCall
from cia.models import ChangeRequest, Finding, RiskReport
from cia.sources import SyntheticSource
from cia.workflow import ROUTE_LABELS as WORKFLOW_ROUTES


@pytest.fixture(scope="module")
def org(tmp_path_factory):
    db = tmp_path_factory.mktemp("org") / "org.db"
    truth = generate(db, n_identities=500, seed=7)
    case = next(c for c in truth["cases"] if c["case_type"] == "nightly_job")
    return str(db), case


def turn(*calls):
    return LLMTurn(
        tool_calls=tuple(
            ToolCall(f"t{i}", name, args) for i, (name, args) in enumerate(calls)
        )
    )


def safe(*cited):
    return RiskReport(
        request_id="R",
        verdict="safe_to_delete",
        confidence="high",
        findings=[
            Finding(kind="k", summary="s", evidence_ids=[i], confidence="high")
            for i in cited
        ],
    )


def test_graph_has_the_agreed_nodes_and_routes():
    text = mermaid()
    for node in ("model", "nudge", "tools", "check", "finish", "cap"):
        assert f"\t{node}(" in text
    for edge in (
        "__start__ --> model",
        "model -.-> tools",
        "model -.-> nudge",
        "tools -.-> check",
        "check -.-> tools",
        "check -.-> finish",
        "tools -.-> cap",
        "finish --> __end__",
    ):
        assert edge in text


def test_coverage_tracks_reads_and_unread_pages():
    read, unread = coverage([], [], "get_tickets", {"rows": [], "next_offset": 50})
    assert read == ["get_tickets"] and unread == ["get_tickets"]
    read, unread = coverage(
        read, unread, "get_tickets", {"rows": [], "next_offset": None}
    )
    assert unread == []


def test_finish_report_never_lets_an_unverified_delete_through():
    out = finish_report(safe("ev-usage-1"), ["get_usage"], [])
    assert out.verdict == "insufficient_evidence" and out.confidence == "low"
    assert "get_dependencies not read" in (out.incomplete_reason or "")
    assert (
        finish_report(safe("ev-usage-1"), list(REQUIRED_FOR_DELETE), []).verdict
        == "safe_to_delete"
    )
    assert (
        finish_report(safe(), list(REQUIRED_FOR_DELETE), ["get_tickets"]).verdict
        == "insufficient_evidence"
    )


def test_drop_bad_citations_turns_a_safe_verdict_into_ask_owner():
    out = drop_bad_citations(safe("ev-usage-1", "ev-made-up-9"), {"ev-made-up-9"})
    assert out.verdict == "ask_owner" and out.rejected_citations == 1
    assert [f.evidence_ids for f in out.findings] == [["ev-usage-1"]]


def test_recursion_limit_is_only_a_backstop(org):
    # A model that never calls a tool takes two graph hops per step; our cap must fire first.
    db, case = org
    llm = FakeLLM([LLMTurn(text="thinking")] * 20)
    report = Investigator(SyntheticSource(db), llm, max_steps=8).investigate(
        ChangeRequest(request_id="R", identity_id=case["identity_id"])
    )
    assert report.verdict == "insufficient_evidence" and "step cap" in (
        report.incomplete_reason or ""
    )
    assert recursion_limit(8) > 3 * 8


def test_checkpointer_saves_every_step_under_the_request_id(org):
    db, case = org
    target = case["identity_id"]
    script = [
        turn(
            ("get_usage", {"identity_id": target}),
            ("get_dependencies", {"identity_id": target}),
        ),
        turn(
            (
                "submit_report",
                {
                    "verdict": "disable_first",
                    "confidence": "high",
                    "findings": [
                        {
                            "kind": "job",
                            "summary": "nightly",
                            "evidence_ids": ["ev-usage-1"],
                            "confidence": "high",
                        }
                    ],
                },
            )
        ),
    ]
    inv = Investigator(
        SyntheticSource(db), FakeLLM(script), checkpointer=memory_checkpointer()
    )
    report = inv.investigate(ChangeRequest(request_id="REQ-CP", identity_id=target))
    config: RunnableConfig = {"configurable": {"thread_id": "REQ-CP"}}
    saved = inv.graph.get_state(config).values
    assert saved["report"] == report and sorted(saved["evidence"]) == [
        "ev-deps-2",
        "ev-usage-1",
    ]
    assert (
        len(list(inv.graph.get_state_history(config))) >= 5
    )  # start, model, tools, model, check, finish


@pytest.mark.parametrize(
    "name, labels",
    [("engine-graph", ROUTE_LABELS), ("workflow-graph", WORKFLOW_ROUTES)],
)
def test_every_node_and_route_has_a_drawing_entry(name, labels):
    """Adding a node or a route to either graph fails here until it has a kind or a label for the docs."""
    data = drawings()[name]
    assert all(n["kind"] for n in data["nodes"])
    assert {(e["source"], e["target"]) for e in data["edges"]} == set(labels)
    assert all(
        n["doc"] for n in data["nodes"] if n["kind"] not in ("start", "end")
    )  # each node has a docstring


def test_the_workflow_drawing_has_the_agreed_nodes():
    nodes = {n["id"]: n["kind"] for n in drawings()["workflow-graph"]["nodes"]}
    assert nodes["investigate"] == "llm"  # the investigator graph, as one node
    assert nodes["approve"] == nodes["quiet"] == "wait"  # the two interrupt() waits
    assert {k for k, v in nodes.items() if v == "act"} == {
        "dry_run",
        "disable",
        "delete",
        "restore",
    }


def test_the_docs_drawing_matches_the_code():
    """docs/index.html#engine and #workflow draw the embedded data; run `make graph` after
    changing graph.py or workflow.py."""
    page = (Path(__file__).resolve().parents[1] / "docs" / "index.html").read_text()
    embedded = page[page.index(DOCS_BEGIN) : page.index(DOCS_END) + len(DOCS_END)]
    assert embedded == docs_block(drawings()), (
        "docs/index.html is behind graph.py or workflow.py: run `make graph`"
    )


def test_write_docs_replaces_only_the_marked_block(tmp_path):
    page = tmp_path / "page.html"
    page.write_text(f"<p>before</p>\n{DOCS_BEGIN}\nold\n{DOCS_END}\n<p>after</p>\n")
    assert write_docs(page) is True and write_docs(page) is False
    text = page.read_text()
    assert (
        text.startswith("<p>before</p>")
        and text.endswith("<p>after</p>\n")
        and '"model"' in text
        and 'id="workflow-graph"' in text
    )
