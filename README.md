# change-impact-agent

An agent that investigates **what could break before an identity change is made**.

Given a request such as "delete account X", it digs through usage logs, dependency edges,
config and policy references, and ticket history, then returns a risk report with
confidence levels and one verdict: `safe_to_delete`, `disable_first`, `ask_owner`, or
`insufficient_evidence` when it could not check what it needed. Only `safe_to_delete`
permits a delete. Execution is a separate, reversible step that needs human approval.

## Status

Week 2 of 4. Done: synthetic org, read-only evidence tools, naive baseline, the
`EvidenceSource` and `LLMClient` interfaces, and the investigator as a LangGraph state
graph (code-assigned evidence ids, an 8-step cap, citation check; `make graph` draws it). It runs on a local model through Ollama, a
hosted open model through OpenRouter, or on Bedrock, and a local run monitor shows each run live. In review: the benchmark. On
the 20 synthetic cases, gemma4:26b on Ollama made 0 unsafe deletes and got all 20 right
(with an 8192-token output limit; 19 of 20 at 2048, the miss safely marked incomplete).
The same model hosted on OpenRouter also got 20 of 20, in about 3 minutes, for under 1 cent.

Results so far come from a synthetic org. They show the guardrails work as designed; they
are not evidence of production readiness.

## Why a benchmark first

Staleness rules are the usual way to clean up accounts. They break things. The generator
plants hidden dependencies that a "no usage in 90 days" rule cannot see, so the agent has
a measurable target.

| Planted case | What hides it |
| --- | --- |
| nightly_job | A leaver account still used by a nightly pipeline |
| orphaned_service_account | A service account whose only owner is the leaver, with live consumers |
| policy_exception | The leaver is named in an access policy exception |
| shared_account | A ticket says finance shares the login; usage is quarterly |
| break_glass | Dormant on purpose; only a ticket says so |
| decoy_safe | Truly safe to delete; measures false alarms |

On the default 500-identity org, the naive rule would delete 6 of the 10 risky accounts.

## Run it

```bash
pip install -e ".[dev]"
cp .env.example .env   # model settings and keys; git-ignored, the shell overrides it
make seed       # 500 identities, 20 change requests, writes bench/ground_truth.json
make baseline   # shows what the naive rule breaks
make test
make typecheck # pyright, 0 errors (also in CI)
make format    # ruff format at 88 columns; make lint checks it (also in CI)

# investigate one change request with a local model (Ollama, default gemma4:26b-mlx)
make run-local ID=REQ-0003   # events in runs/REQ-0003.jsonl
make smoke                   # Rust terminal UI (tui/, needs cargo) with 8 offline runs piped in; q quits
make graph                   # redraw docs/index.html#engine from graph.py; print Mermaid
make monitor                 # http://127.0.0.1:8765: start runs, watch them live;
                             # Flow tab: the run on the agent's five levels, with replay; Model filter

# score every change request (fails if any risky account would be deleted)
make bench-fake             # offline plumbing check, also in CI
make bench-local MAX_TOKENS=8192   # Ollama; results in bench/results/
make bench-openrouter OR_MODEL=google/gemma-4-26b-a4b-it   # OpenRouter; key in .env
```

`python -m cia.generator --identities 20000` builds the 20,000-identity org in a few seconds.

## Design notes

- Evidence tools open the database in SQLite read-only mode. The agent has no write path.
- Guardrails are code, not prompt: a failing source is a tool error, never "no rows", and
  `safe_to_delete` needs usage, dependencies and tickets read in full, else it becomes
  `insufficient_evidence`.
- Evidence records when it was collected and from which source; a source's update time is
  recorded only when the source provides it.
- Everything is deterministic: fixed seed, fixed clock.
- SQLite keeps the benchmark dependency-free. The service layer will use PostgreSQL.

## Docs
Plans and design notes are HTML: open [docs/index.html](docs/index.html) in a browser.
