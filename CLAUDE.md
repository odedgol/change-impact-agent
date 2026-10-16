# Change Impact Agent

Portfolio project (target: Twine Security, Backend Engineer – Agentic AI). Before an IAM identity/permission is deleted or changed, an agent gathers evidence and returns a risk report (confidence + safe_to_delete / disable_first / ask_owner). A separate reversible executor runs only after human approval.

## Status
- Week 1 done: synthetic org generator, read-only evidence tools, naive baseline (breaks 6 of 10 risky cases). 7 tests pass.
- Week 2 in progress (plan: `docs/week-2.html`). PR 2.1 (tooling + Pydantic models) merged. PR 2.2 (`EvidenceSource`, `SyntheticSource`) merged; review fixes in a follow-up: lists paged with `next_offset` instead of cut, usage window clamped to >= 90 days, consumer rows carry `target`. 37 tests. Next: `LLMClient` + `FakeLLM`, the investigator loop, the bench.

## Commands
- `make seed` / `make test` / `make lint` / `make baseline`

## Design rules
- Simulate the data tools return, not the tools. One `EvidenceSource` interface (`SyntheticSource` now; `AwsSource`/`KeycloakSource` optional) and one `LLMClient` interface (`BedrockClient`, `FakeLLM`).
- Evidence tools are read-only (SQLite `mode=ro`). Guardrails live in code, not in the prompt.
- Executor: dry-run, disable-first, undo, idempotency; append-only audit log.
- Cloud-agnostic: AWS is one deployment target, never a dependency; the design must move to a private cloud. Cloud services sit behind an interface or open protocol (`LLMClient`, `EvidenceSource`, `AuditSink`, Postgres SQL, Temporal, OCI containers). `boto3` is imported only inside adapter classes, and each adapter has a non-AWS sibling or a fake. Mapping: `docs/index.html#portable`.
- Stack: Python 3.12+, Pydantic v2, FastAPI, boto3 Bedrock Converse, Temporal (approval signal; LLM/DB calls in activities), Postgres later, pytest, Ruff.
- Hard eval gate: 0 unsafe-delete recommendations. Targets (recall >= 90%, precision >= 80%) are proposals, not proven.

## Docs are HTML-first
- Plans, specs, design notes, weekly plans and results live in `docs/*.html`: self-contained single files (inline CSS/JS/SVG, light and dark theme, readable at phone width, no build step).
- Only `README.md` and `CLAUDE.md` stay Markdown (GitHub and Claude Code read them as Markdown).
- Before coding each week, write `docs/week-N.html` (plan, diagram, acceptance checks); update it when done. Keep each page short and visual; tables and diagrams over prose.
- **Every change updates the docs in the same PR (mandatory).** Before committing, find each `docs/*.html` section the change affects and update it: the week plan (step status, validation checks ticked, decisions), the hub (`docs/index.html`: sprints and PR table, roadmap status, tech stack, PR workflow gates, overview status), and the status lines in `CLAUDE.md`. A PR whose docs are stale is not done.
- Docs hub: `docs/index.html` (overview, system diagram, agentic workflow, guardrails, roadmap, sprints and PRs, tech stack, PR workflow, requirements). Week plans: `docs/week-N.html`, same shell and sidebar.
