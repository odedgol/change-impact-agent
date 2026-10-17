# change-impact-agent

An agent that investigates **what could break before an identity change is made**.

Given a request such as "delete account X", it digs through usage logs, dependency edges,
config and policy references, and ticket history, then returns a risk report with
confidence levels and one verdict: `safe_to_delete`, `disable_first`, `ask_owner`, or
`insufficient_evidence` when it could not check what it needed. Only `safe_to_delete`
permits a delete. Execution is a separate, reversible step that needs human approval.

## Status

Week 2 of 4. Done: synthetic org, read-only evidence tools, naive baseline, the
`EvidenceSource` and `LLMClient` interfaces, and the investigator loop (code-assigned
evidence ids, an 8-step cap, citation check). It runs on a local model through Ollama or
on Bedrock, and a local run monitor shows each run live. In review: evidence freshness
and the `insufficient_evidence` verdict. Next: the benchmark across models.

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
make seed       # 500 identities, 20 change requests, writes bench/ground_truth.json
make baseline   # shows what the naive rule breaks
make test

# investigate one change request with a local model (Ollama, default gemma4:26b-mlx)
make run-local ID=REQ-0003   # events in runs/REQ-0003.jsonl
make monitor                 # http://127.0.0.1:8765: start runs, watch them live
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
