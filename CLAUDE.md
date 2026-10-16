# Change Impact Agent

Portfolio project (target: Twine Security, Backend Engineer – Agentic AI). Before an IAM identity/permission is deleted or changed, an agent gathers evidence and returns a risk report (confidence + safe_to_delete / disable_first / ask_owner). A separate reversible executor runs only after human approval.

## Status
- Week 1 done: synthetic org generator, read-only evidence tools, naive baseline (breaks 6 of 10 risky cases). 7 tests pass.
- Next (week 2): `LLMClient` interface + `FakeLLM`, then the investigator tool-calling loop over Bedrock Converse, structured Pydantic `RiskReport` (every finding cites evidence ids).

## Commands
- `make seed` / `make test` / `make baseline`

## Design rules
- Simulate the data tools return, not the tools. One `EvidenceSource` interface (`SyntheticSource` now; `AwsSource`/`KeycloakSource` optional) and one `LLMClient` interface (`BedrockClient`, `FakeLLM`).
- Evidence tools are read-only (SQLite `mode=ro`). Guardrails live in code, not in the prompt.
- Executor: dry-run, disable-first, undo, idempotency; append-only audit log.
- Stack: Python 3.12+, Pydantic v2, FastAPI, boto3 Bedrock Converse, Temporal (approval signal; LLM/DB calls in activities), Postgres later, pytest, Ruff.
- Hard eval gate: 0 unsafe-delete recommendations. Targets (recall >= 90%, precision >= 80%) are proposals, not proven.

## Docs are HTML-first
- Plans, specs, design notes, weekly plans and results live in `docs/*.html`: self-contained single files (inline CSS/JS/SVG, light and dark theme, readable at phone width, no build step).
- Only `README.md` and `CLAUDE.md` stay Markdown (GitHub and Claude Code read them as Markdown).
- Before coding each week, write `docs/week-N.html` (plan, diagram, acceptance checks); update it when done. Keep each page short and visual; tables and diagrams over prose.
- Current full plan: `docs/plan.html` (PRD, class diagram, simulation, challenges and proof, skills, LinkedIn plan, roadmap).
