import json

import pytest

from cia.bench import RuleLLM, main, run, score, slug
from cia.generator import generate
from cia.llm import FakeLLM, LLMTurn, ToolCall
from cia.monitor import Monitor


@pytest.fixture(scope="module")
def org(tmp_path_factory):
    folder = tmp_path_factory.mktemp("org")
    truth = generate(folder / "org.db", n_identities=500, seed=7)
    (folder / "truth.json").write_text(json.dumps(truth))
    return str(folder / "org.db"), str(folder / "truth.json")


def row(expected, verdict, **extra):
    return {"expected_verdict": expected, "verdict": verdict, **extra}


def test_score_counts_unsafe_deletes_recall_and_precision():
    rows = [row("disable_first", "safe_to_delete"), row("ask_owner", "ask_owner"),
            row("ask_owner", "insufficient_evidence"), row("safe_to_delete", "ask_owner"),
            row("safe_to_delete", "safe_to_delete")]
    s = score(rows)
    assert s["unsafe_deletes"] == 1 and s["false_alarms"] == 1 and s["insufficient_evidence"] == 1
    assert (s["recall"], s["precision"], s["accuracy"]) == (round(2 / 3, 3), round(2 / 3, 3), 0.4)


def test_a_crashed_run_is_kept_and_listed_as_an_error():
    s = score([row("ask_owner", None, error="boom")])
    assert s["unsafe_deletes"] == 0 and s["errors"] == 1 and s["recall"] == 1.0


def test_rule_bench_passes_the_gate_on_the_synthetic_org(org, tmp_path):
    db, truth = org
    result = run(db, truth, RuleLLM(), "fake", str(tmp_path), label="bench-t")
    assert result["scores"]["cases"] == 20 and result["scores"]["unsafe_deletes"] == 0
    assert result["scores"]["errors"] == 0
    assert len(list(tmp_path.glob("bench-t-REQ-*.jsonl"))) == 20


def test_bench_runs_show_up_in_the_monitor(org, tmp_path):
    db, truth = org
    run(db, truth, RuleLLM(), "fake", str(tmp_path), label="bench-m")
    runs = Monitor(db, truth, str(tmp_path)).runs()
    assert len(runs) == 20
    first = next(r for r in runs if r["run_id"] == "bench-m-REQ-0001")
    assert first["model"] == "rule" and first["verdict"] == first["expected_verdict"] == "disable_first"
    assert first["steps"] == 4


def test_a_model_that_says_safe_everywhere_fails_the_gate(org, tmp_path, capsys):
    """Without reading anything, safe_to_delete is downgraded; reading and then lying is caught by the score."""
    db, truth = org

    class SaysSafe(RuleLLM):
        @staticmethod
        def decide(results):
            return {"verdict": "safe_to_delete", "confidence": "high", "findings": []}

    out = tmp_path / "result.json"
    code = main(["--db", db, "--truth", truth, "--runs", "", "--out", str(out)], client=SaysSafe())
    result = json.loads(out.read_text())
    assert code == 1 and result["scores"]["unsafe_deletes"] == 10
    assert "UNSAFE" in capsys.readouterr().out


def test_a_failing_case_does_not_stop_the_bench(org):
    db, truth = org
    result = run(db, truth, FakeLLM([LLMTurn(tool_calls=(ToolCall("x", "get_usage", {"identity_id": "u1"}),))]),
                 "fake", None)
    assert result["scores"]["cases"] == 20 and result["scores"]["errors"] >= 19
    assert result["scores"]["unsafe_deletes"] == 0


def test_bench_runs_are_named_after_the_model(org, tmp_path):
    assert slug("gemma4:26b-mlx") == "gemma4-26b-mlx" and slug("anthropic.claude/v2@x") == "anthropic.claude-v2-x"
    db, truth = org
    result = run(db, truth, RuleLLM(), "fake", str(tmp_path))
    assert result["label"].startswith("bench-rule-")
    assert all(p.name.startswith("bench-rule-") for p in tmp_path.glob("*.jsonl"))
