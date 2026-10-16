# change-impact-agent

An agent that investigates **what could break before an identity change is made**.

Given a request such as "delete account X", it digs through usage logs, dependency edges,
config and policy references, and ticket history, then returns a risk report with
confidence levels and a safe recommendation: delete, disable first, or ask the owner.
Execution is a separate, reversible step that needs human approval.

## Status

Week 1 of 4: synthetic organization, read-only evidence tools, benchmark ground truth.
The investigator agent, approval workflow and executor come next.

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
```

`python -m cia.generator --identities 20000` builds the 20,000-identity org in a few seconds.

## Design notes

- Evidence tools open the database in SQLite read-only mode. The agent has no write path.
- Everything is deterministic: fixed seed, fixed clock.
- SQLite keeps the benchmark dependency-free. The service layer will use PostgreSQL.

## Docs
Plans and design notes are HTML: open [docs/plan.html](docs/plan.html) in a browser.
