# 001 — Week 2: investigator loop and RiskReport

Status: active
Owner: Oded Goldberg
Last updated: 2026-10-16

## Context
Week 1 shipped the synthetic org (`src/cia/generator.py`), read-only tools
(`src/cia/tools.py::EvidenceTools`) and the naive baseline that breaks 6 of 10 risky
cases. Week 2 adds the agent itself: two swappable interfaces, a tool-calling loop,
and a structured report where every finding cites evidence ids that code verifies.

## Goal
`investigate(request) -> RiskReport` runs a tool-calling loop over an `LLMClient`,
returns `safe_to_delete | disable_first | ask_owner` with confidence, and every
finding cites evidence ids that exist in that run's tool log. Proven end-to-end
with `FakeLLM`; one smoke run on Bedrock.

## Scope
In:
- `EvidenceSource` Protocol; `SyntheticSource` = today's `EvidenceTools`, each result tagged with an evidence id.
- `LLMClient` Protocol (`converse(messages, tools) -> LLMTurn`); `FakeLLM` (scripted turns) and `BedrockClient` (boto3 Converse).
- `Investigator`: loop, step cap, tool dispatch, tool log, final report parsing.
- Pydantic v2 models: `ChangeRequest`, `Evidence`, `Finding`, `RiskReport`, plus citation check.
- `cia.bench` runner: agent vs ground truth, reports unsafe deletes / recall / precision.
- Add `pydantic`, `ruff`; `boto3` as optional extra `[bedrock]`.

Out (week 3+): guardrails beyond citation check, Temporal, executor, audit log, FastAPI, Postgres, red-team cases.

## Assumptions
- Ground truth and generator are unchanged; tests in `tests/test_generator_and_tools.py` stay green.
- Unit tests and CI never call Bedrock; `FakeLLM` covers the loop.
- Bedrock access exists in the user's AWS account for the smoke run only.
- Tools stay read-only (`mode=ro`); no new write path is introduced.

## Open Questions
All resolved on 2026-10-16: the recommended answer was accepted for each one.

1. **Plan home conflict.** The `writing-plans` skill says `.plan/*.md` is the source of truth and "never create `docs/`"; `CLAUDE.md` says plans live in `docs/*.html` and `docs/` already exists.
   *Recommended:* `.plan/` holds the plan lifecycle (status, questions, approval); `docs/week-2.html` stays as the visual one-pager per `CLAUDE.md`, written from this plan. The skill's `docs/` and "frontend-only" rules don't apply to this repo.
2. **Evidence id format.** *Recommended:* `ev-<tool>-<n>` per run (e.g. `ev-deps-3`), assigned by the Investigator when it logs a tool result, not by the model.
3. **Rename `EvidenceTools` → `SyntheticSource`?** *Recommended:* yes, keep `EvidenceTools = SyntheticSource` alias so week-1 tests and `baseline.py` don't change.
4. **Python floor.** `pyproject.toml` says `>=3.11`, `CLAUDE.md` says 3.12+. *Recommended:* bump to `>=3.12`.
5. **Bedrock model.** *Recommended:* Claude Sonnet 5.5 via env `CIA_BEDROCK_MODEL_ID` (no hard-coded id), temperature 0, region from env.
6. **Step cap.** *Recommended:* max 8 tool calls; on cap, return `ask_owner` with confidence `low` (fail safe, never `safe_to_delete`).
7. **Report on bad citations.** *Recommended:* drop findings that cite unknown ids, count them in `rejected_citations`; if the verdict is `safe_to_delete` and any citation was rejected, downgrade to `ask_owner`.

## Steps
1. `pyproject.toml`: python `>=3.12`, deps `pydantic>=2`, extras `dev=[pytest, ruff]`, `bedrock=[boto3]`; `[tool.ruff]`; `Makefile` targets `lint`, `bench-fake`, `bench-bedrock`.
2. `src/cia/models.py`: `ChangeRequest`, `Evidence(id, tool, args, result)`, `Finding(kind, summary, evidence_ids: list[str] (min 1), confidence)`, `RiskReport(request_id, verdict, confidence, findings, rejected_citations)`.
3. `src/cia/sources.py`: `EvidenceSource` Protocol with `get_identity/get_usage/get_dependencies/get_tickets/get_owner`; move `EvidenceTools` here as `SyntheticSource`, re-export from `tools.py` for compatibility. Add row caps on list results.
4. `src/cia/llm.py`: `LLMClient` Protocol, `LLMTurn(text, tool_calls)`, `ToolSpec`; `FakeLLM(script)` returns scripted turns and records received messages; `BedrockClient` maps to `bedrock-runtime.converse` `toolConfig` / `toolUse` / `toolResult`. Import `boto3` lazily.
5. `src/cia/investigator.py`: system prompt, tool specs derived from `EvidenceSource`, loop (call LLM → dispatch tool calls → log `Evidence` with id → feed `toolResult` back) until a `submit_report` tool call or step cap; parse into `RiskReport`; run citation check (Q7).
6. `src/cia/bench.py`: runs the investigator over `bench/ground_truth.json`; prints `unsafe_deletes`, verdict accuracy, recall on hidden deps (ref found in cited evidence), precision on alarms, tool calls per case. `--llm fake|bedrock`. The fake mode uses `ScriptedPolicyLLM`: calls every tool, then reports from evidence — plumbing check, not a quality claim.
7. Tests (below), `ruff check` clean.
8. `docs/week-2.html` (plan, loop diagram, acceptance checks), then update `CLAUDE.md` status, `docs/plan.html` roadmap row, backlog row 2 → done, this plan → done.

## Validation
- [ ] `make test` passes; the 7 week-1 tests still pass unchanged.
- [ ] `make lint` (`ruff check src tests`) exits 0.
- [ ] `tests/test_models.py`: `Finding` with empty `evidence_ids` raises `ValidationError`; invalid verdict raises.
- [ ] `tests/test_llm_fake.py`: `FakeLLM` returns scripted turns in order and raises when the script runs out.
- [ ] `tests/test_investigator.py`:
  - tool calls are dispatched to the source and logged with unique evidence ids;
  - a finding citing an unknown id is dropped and counted in `rejected_citations`;
  - `safe_to_delete` with a rejected citation is downgraded to `ask_owner`;
  - hitting the step cap returns `ask_owner`, never `safe_to_delete`;
  - an unknown tool name returns a tool error to the model and does not crash.
- [ ] `tests/test_sources.py`: `SyntheticSource` satisfies `EvidenceSource` (`isinstance` with `runtime_checkable`); still read-only (DELETE raises `OperationalError`).
- [ ] `tests/test_bedrock_mapping.py`: request/response mapping to Converse shapes, tested with a stubbed client (no network).
- [ ] `make bench-fake` prints `unsafe_deletes: 0` on the default 500-identity org.
- [ ] Manual, optional: `make bench-bedrock` with `CIA_BEDROCK_MODEL_ID` set completes the 20 cases; result recorded in `docs/week-2.html` (no target claimed).

## Risks
| Risk | Mitigation |
|---|---|
| Model invents evidence ids | Ids assigned in code; citation check drops unknown ids |
| Model never calls `submit_report` | Step cap with fail-safe `ask_owner` |
| `ScriptedPolicyLLM` mistaken for agent quality | Bench labels fake runs "plumbing"; only Bedrock numbers go in results |
| Converse API shape drift | Mapping isolated in `BedrockClient`, covered by stub test; check docs via context7 before coding |
| Ticket text as prompt injection | Tools read-only now; red-team cases in week 4 |
| Rename breaks week-1 code | Alias `EvidenceTools = SyntheticSource` |

## Rollout Order
1 deps/tooling → 2 models → 3 sources → 4 llm (fake first, Bedrock last) → 5 investigator → 6 bench → 7 tests/lint green → 8 docs. One commit per step on a new branch `feat/week2-investigator` off `feat/week1-synthetic-org`.

## Rollback
All work is additive on its own branch: delete the branch, or `git revert` the step commits. The `EvidenceTools` alias keeps week-1 callers working at every step; no data migration, nothing deployed.
