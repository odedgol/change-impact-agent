"""Benchmark: run the investigator on every change request and score it against the ground truth.

    python -m cia.bench --llm fake       # plumbing check: a fixed rule plays the model
    python -m cia.bench --llm openai     # a local model through Ollama (or vLLM)
    python -m cia.bench --llm openrouter # Gemma 4 31B on OpenRouter's free tier (OPENROUTER_API_KEY)
    python -m cia.bench --llm bedrock

Each case writes its events to ``runs/bench-<model>-<time>-<request>.jsonl`` with the same run header the
monitor writes, so ``make monitor`` lists bench runs and can follow one live.

Scores, risky = the expected verdict is not ``safe_to_delete``:
    unsafe_deletes  safe_to_delete on a risky case (the hard gate: must be 0)
    recall          risky cases kept from deletion / risky cases
    precision       kept cases that were risky / kept cases
    accuracy        exact verdict matches / cases
A run that crashes counts as kept (nothing is deleted) and is listed under errors.
Results on the synthetic org show the plumbing and guardrails work; they are not production evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from .env import load_env
from .investigator import Investigator, JsonlEvents
from .llm import (
    LLM_CHOICES,
    LLMClient,
    LLMTurn,
    Message,
    ToolCall,
    ToolResults,
    ToolSpec,
    make_client,
)
from .models import ChangeRequest
from .sources import SyntheticSource

SAFE = "safe_to_delete"
READS = ("get_usage", "get_dependencies", "get_tickets")
HR_NOTICE = (
    "hr-bot"  # every leaver gets its "Employee left" ticket; it says nothing about risk
)


class RuleLLM:
    """Plays the model with a fixed rule, so the bench runs offline. A plumbing check, not a quality claim.

    Turn 1 reads usage, dependencies and tickets. Turn 2 decides: a ticket written by a person (not the
    HR leaver notice) or an owned account means ask_owner; any usage or dependency means disable_first;
    nothing at all means safe_to_delete.
    """

    model = "rule"

    def converse(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMTurn:
        if not any(isinstance(m, ToolResults) for m in messages):
            identity = re.search(r"identity (\S+)\.", messages[0].text).group(1)  # type: ignore[union-attr]
            return LLMTurn(
                tool_calls=tuple(
                    ToolCall(f"r{i}", name, {"identity_id": identity})
                    for i, name in enumerate(READS)
                )
            )
        results = [
            r.content
            for m in messages
            if isinstance(m, ToolResults)
            for r in m.results
            if not r.is_error
        ]
        return LLMTurn(
            tool_calls=(ToolCall("submit", "submit_report", self.decide(results)),)
        )

    @staticmethod
    def decide(results: list[dict[str, Any]]) -> dict[str, Any]:
        found = {}
        for r in results:
            data = r["data"]
            if r["evidence_id"].startswith("ev-tickets"):
                data = {
                    **data,
                    "rows": [t for t in data["rows"] if t.get("author") != HR_NOTICE],
                }
            if data.get("event_count") or data.get("rows"):
                found[r["evidence_id"]] = data
        kinds = {row.get("kind") for d in found.values() for row in d.get("rows", [])}
        tickets = any(e.startswith("ev-tickets") for e in found)
        verdict = (
            "ask_owner"
            if tickets or "owner_of" in kinds
            else "disable_first"
            if found
            else SAFE
        )
        cited = list(found) or [r["evidence_id"] for r in results]
        return {
            "verdict": verdict,
            "confidence": "medium",
            "findings": (
                [
                    {
                        "kind": "evidence",
                        "summary": "rule: found" if found else "rule: nothing found",
                        "evidence_ids": cited,
                        "confidence": "medium",
                    }
                ]
                if cited
                else []
            ),
        }


def make_llm(llm: str, model: str | None, max_tokens: int | None = None) -> LLMClient:
    return RuleLLM() if llm == "fake" else make_client(llm, model, max_tokens)


def model_name(client: LLMClient, fallback: str) -> str:
    return (
        getattr(client, "model", None) or getattr(client, "model_id", None) or fallback
    )


def slug(name: str) -> str:
    """A model name as a file-name part: gemma4:26b-mlx -> gemma4-26b-mlx."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")


def run_case(
    db: str, case: dict[str, Any], client: LLMClient, llm: str, events_path: Path | None
) -> dict[str, Any]:
    row = {
        k: case[k]
        for k in ("request_id", "identity_id", "case_type", "expected_verdict")
    }
    seen: list[dict[str, Any]] = []
    out = events_path.open("w") if events_path else None
    sink = JsonlEvents(out) if out else None

    def emit(event: dict[str, Any]) -> None:
        seen.append(event)
        if sink:
            sink(event)

    try:
        emit(
            {
                "type": "run",
                "run_id": events_path.stem if events_path else case["request_id"],
                "llm": llm,
                "model": model_name(client, llm),
                "case_type": case["case_type"],
                "expected_verdict": case["expected_verdict"],
            }
        )
        source = SyntheticSource(db)
        report = Investigator(
            source, client, events=emit, clock=lambda: source.now
        ).investigate(
            ChangeRequest(
                request_id=case["request_id"], identity_id=case["identity_id"]
            )
        )
        row.update(
            verdict=report.verdict,
            confidence=report.confidence,
            rejected_citations=report.rejected_citations,
            incomplete_reason=report.incomplete_reason,
        )
    except Exception as e:  # noqa: BLE001 - one failing case is recorded, the bench goes on
        emit({"type": "run_error", "error": f"{type(e).__name__}: {e}"})
        row.update(verdict=None, error=f"{type(e).__name__}: {e}")
    finally:
        if out:
            out.close()
    row["steps"] = max(
        (e.get("steps", 0) for e in seen if e["type"] == "report"), default=0
    )
    return row


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    risky = [r for r in rows if r["expected_verdict"] != SAFE]
    kept = [r for r in rows if r["verdict"] != SAFE]
    caught = [r for r in risky if r["verdict"] != SAFE]
    return {
        "cases": len(rows),
        "unsafe_deletes": len(risky) - len(caught),
        "recall": round(len(caught) / len(risky), 3) if risky else None,
        "precision": round(len(caught) / len(kept), 3) if kept else None,
        "accuracy": (
            round(
                sum(r["verdict"] == r["expected_verdict"] for r in rows) / len(rows), 3
            )
            if rows
            else None
        ),
        "false_alarms": sum(
            r["expected_verdict"] == SAFE and r["verdict"] != SAFE for r in rows
        ),
        "insufficient_evidence": sum(
            r["verdict"] == "insufficient_evidence" for r in rows
        ),
        "rejected_citations": sum(r.get("rejected_citations", 0) for r in rows),
        "errors": sum("error" in r for r in rows),
    }


def run(
    db: str,
    truth_path: str,
    client: LLMClient,
    llm: str,
    runs_dir: str | None = "runs",
    label: str | None = None,
    progress=None,
) -> dict[str, Any]:
    with open(truth_path) as f:
        cases = sorted(json.load(f)["cases"], key=lambda c: c["request_id"])
    label = (
        label
        or f"bench-{slug(model_name(client, llm))}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    folder = Path(runs_dir) if runs_dir else None
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        rows.append(
            run_case(
                db,
                case,
                client,
                llm,
                folder / f"{label}-{case['request_id']}.jsonl" if folder else None,
            )
        )
        if progress:
            progress(rows[-1])
    return {
        "label": label,
        "llm": llm,
        "model": model_name(client, llm),
        "max_tokens": getattr(client, "max_tokens", None)
        or getattr(client, "inference_config", {}).get("maxTokens"),
        "scores": score(rows),
        "cases": rows,
    }


def main(argv: list[str] | None = None, client: LLMClient | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Score the investigator against the ground truth."
    )
    p.add_argument("--db", default="data/org.db")
    p.add_argument("--truth", default="bench/ground_truth.json")
    p.add_argument("--llm", choices=["fake", *LLM_CHOICES], default="fake")
    p.add_argument(
        "--model",
        help="model id; default from CIA_OPENAI_MODEL or CIA_BEDROCK_MODEL_ID",
    )
    p.add_argument(
        "--max-tokens", type=int, help="output limit per model turn (default 2048)"
    )
    p.add_argument(
        "--runs", default="runs", help="folder for per-case events; '' writes none"
    )
    p.add_argument("--out", help="also write the full result as JSON to this file")
    a = p.parse_args(argv)
    load_env()
    client = client or make_llm(a.llm, a.model, a.max_tokens)

    def progress(r: dict[str, Any]) -> None:
        mark = (
            "ok "
            if r["verdict"] == r["expected_verdict"]
            else "UNSAFE"
            if r["verdict"] == SAFE
            else "diff"
        )
        print(
            f"{mark:6} {r['request_id']} {r['case_type']:26} expected {r['expected_verdict']:15} "
            f"got {r['verdict']} ({r['steps']} steps){'  ' + r['error'] if 'error' in r else ''}",
            flush=True,
        )

    result = run(a.db, a.truth, client, a.llm, a.runs or None, progress=progress)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    s = result["scores"]
    print(f"\nmodel: {result['model']} (max_tokens {result['max_tokens']})")
    for k, v in s.items():
        print(f"{k}: {v}")
    return 1 if s["unsafe_deletes"] else 0  # the hard gate fails the command


if __name__ == "__main__":
    sys.exit(main())
