"""Smoke runs for the terminal UI: a few offline investigations written as JSON lines to stdout.

    PYTHONPATH=src python -m cia.smoke | cargo run --manifest-path tui/Cargo.toml     # make smoke

One request per planted case type, played by the rule model from the bench, then two scripted runs
that reach every guardrail: a refused tool, a citation sent back once, and the step cap. No network,
no model, no keys. The events are the investigator's own, so the UI sees exactly what a real run emits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from typing import IO, Any

from .bench import RuleLLM
from .investigator import Investigator, JsonlEvents
from .llm import FakeLLM, LLMClient, LLMTurn, ToolCall
from .models import ChangeRequest
from .sources import SyntheticSource

# Every event type the investigator emits, plus the run header; the Rust UI parses this same list.
EVENT_TYPES = (
    "run",
    "request",
    "model_turn",
    "tool_call",
    "tool_error",
    "citation_retry",
    "step_cap",
    "run_error",
    "report",
)


def call(name: str, n: int = 0, **args: Any) -> ToolCall:
    return ToolCall(f"s{n}", name, args)


def guardrails_script(identity: str) -> list[LLMTurn]:
    """Tries a write tool, reads the evidence, cites an id it was never given, then fixes the citation."""

    def report(*cited: str) -> LLMTurn:
        return LLMTurn(
            reasoning="Scripted: submit.",
            tool_calls=(
                call(
                    "submit_report",
                    9,
                    verdict="ask_owner",
                    confidence="high",
                    findings=[
                        {
                            "kind": "owned_service_account",
                            "summary": "owns a service account",
                            "evidence_ids": list(cited),
                            "confidence": "high",
                        }
                    ],
                ),
            ),
        )

    return [
        LLMTurn(
            reasoning="Scripted: try a tool that does not exist.",
            tool_calls=(call("delete_identity", 1, identity_id=identity),),
        ),
        LLMTurn(
            reasoning="Scripted: read what depends on the account.",
            tool_calls=(
                call("get_dependencies", 2, identity_id=identity),
                call("get_usage", 3, identity_id=identity),
                call("get_tickets", 4, identity_id=identity),
            ),
        ),
        report("ev-deps-1", "ev-made-up-9"),
        report("ev-deps-1"),
    ]


def silent_script(turns: int) -> list[LLMTurn]:
    """A model that only thinks and never calls a tool: every turn is a nudge until the step cap."""
    return [LLMTurn(reasoning="Scripted: thinking without calling a tool.")] * turns


def plan(
    cases: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, Any], LLMClient, str]]:
    """One case per type with the rule model, then the two scripted runs."""
    picked, seen = [], set()
    for case in sorted(cases, key=lambda c: c["request_id"]):
        if case["case_type"] not in seen:
            seen.add(case["case_type"])
            picked.append((case, RuleLLM(), "rule"))
    orphan = [c for c in cases if c["case_type"] == "orphaned_service_account"][-1]
    decoy = [c for c in cases if c["case_type"] == "decoy_safe"][-1]
    picked.append(
        (
            orphan,
            FakeLLM(guardrails_script(orphan["identity_id"])),
            "FakeLLM (scripted guardrails)",
        )
    )
    picked.append((decoy, FakeLLM(silent_script(12)), "FakeLLM (never calls a tool)"))
    return picked


def run(db: str, cases: Sequence[dict[str, Any]], out: IO[str]) -> int:
    """Write every smoke run to ``out``; returns how many runs there were."""
    sink = JsonlEvents(out)
    runs = plan(cases)
    for case, llm, model in runs:
        sink(
            {
                "type": "run",
                "run_id": f"smoke-{case['request_id']}",
                "llm": "fake",
                "model": model,
                "case_type": case["case_type"],
                "expected_verdict": case["expected_verdict"],
            }
        )
        source = SyntheticSource(db)
        Investigator(
            source, llm, events=sink, clock=lambda s=source: s.now
        ).investigate(
            ChangeRequest(
                request_id=case["request_id"], identity_id=case["identity_id"]
            )
        )
    return len(runs)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Offline smoke runs as JSON-lines events, for the terminal UI."
    )
    p.add_argument("--db", default="data/org.db")
    p.add_argument("--truth", default="bench/ground_truth.json")
    a = p.parse_args(argv)
    with open(a.truth) as f:
        cases = json.load(f)["cases"]
    try:
        run(a.db, cases, sys.stdout)
    except BrokenPipeError:  # the UI quit before the stream ended; nothing left to say
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())


if __name__ == "__main__":
    main()
