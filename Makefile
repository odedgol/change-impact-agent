.PHONY: seed test lint format typecheck smoke tui-test tui-fixture baseline run-local monitor graph bench-fake bench-local bench-openrouter bench-bedrock change approve decline resume changes change-reset agent hr-event agent-approve agent-decline agent-changes mcp demo-org

seed:
	PYTHONPATH=src python -m cia.generator --identities 500 --out data/org.db --truth bench/ground_truth.json

test:
	PYTHONPATH=src python -m pytest -q

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests
	ruff check --fix src tests

typecheck:
	pyright --pythonpath "$$(command -v python)"  # the active Python: .venv locally, the runner in CI

baseline: seed
	PYTHONPATH=src python -m cia.baseline --db data/org.db --truth bench/ground_truth.json

# One request on a local model through Ollama (default gemma4:26b-mlx). Example: make run-local ID=u00002
run-local: seed
	PYTHONPATH=src python -m cia.investigator --llm openai --identity $(ID) --events runs/$(ID).jsonl

# Run monitor: launch runs, watch them live, browse runs/*.jsonl. http://127.0.0.1:$(PORT)
PORT ?= 8765
monitor: seed
	PYTHONPATH=src python -m cia.monitor --port $(PORT)

# Change desk (week 3): start a change, approve or decline it, end its quiet period. LLM=fake is the
# offline rule model; LLM=openai|openrouter|bedrock runs a real one. State in data/ (a writable copy
# of the org, checkpoints, audit log); events in runs/change-$(ID).jsonl, so the monitor shows them.
LLM ?= fake
BY ?= $(USER)
ACTIONS ?= disable,delete
DAYS ?= 14
change: seed
	PYTHONPATH=src python -m cia.workflow start $(ID) --llm $(LLM)

approve:
	PYTHONPATH=src python -m cia.workflow approve $(ID) --actions $(ACTIONS) --by $(BY)

decline:
	PYTHONPATH=src python -m cia.workflow decline $(ID) --by $(BY)

resume:
	PYTHONPATH=src python -m cia.workflow resume $(ID) --days $(DAYS)

changes:
	PYTHONPATH=src python -m cia.workflow status

change-reset:
	PYTHONPATH=src python -m cia.workflow reset

# The agent as a service (src/cia/service.py): an HR event starts the workflow on its own, a message
# says what it found, a person answers. Demo timings by default; production: START_DELAY=60s QUIET=14d.
# ORG=demo uses the realistic demo org (make demo-org): its own data/demo/ and runs/demo/.
ORG ?=
ORG_DB = $(if $(ORG),data/$(ORG)/org.db,data/org.db)
ORG_DATA = $(if $(ORG),data/$(ORG),data)
ORG_RUNS = $(if $(ORG),runs/$(ORG),runs)
ORG_TRUTH = $(if $(ORG),data/$(ORG)/truth.json,bench/ground_truth.json)
AGENT_LLM ?= fake
AGENT_MODEL ?=
START_DELAY ?= 5s
QUIET ?= 2m
agent:
	CIA_AGENT_LLM=$(AGENT_LLM) CIA_AGENT_MODEL=$(AGENT_MODEL) CIA_START_DELAY=$(START_DELAY) \
	  CIA_QUIET_PERIOD=$(QUIET) PYTHONPATH=src python -m cia.service serve \
	  --db $(ORG_DB) --data $(ORG_DATA) --runs $(ORG_RUNS)
hr-event:  # ID=u00034, CASE=nightly_job (first identity of that case type) or WHO="Noah Novak"
	PYTHONPATH=src python -m cia.service event $(ID) $(if $(CASE),--case $(CASE)) \
	  $(if $(WHO),--who "$(WHO)") --db $(ORG_DB) --truth $(ORG_TRUTH)
# The realistic demo org: 120 people with names, the same traps. Starts over (drops data/demo, runs/demo).
demo-org:
	rm -rf data/demo runs/demo
	PYTHONPATH=src python -m cia.demo_org --out data/demo
agent-approve:
	PYTHONPATH=src python -m cia.service approve $(ID) --actions $(ACTIONS) --by $(BY)
agent-decline:
	PYTHONPATH=src python -m cia.service decline $(ID) --by $(BY)
agent-changes:
	PYTHONPATH=src python -m cia.service list

# MCP server (src/cia/mcp_server.py), read-only: prints the command that adds it to Claude Code.
mcp:
	@echo 'claude mcp add cia -- $(CURDIR)/.venv/bin/python -m cia.mcp_server --db $(CURDIR)/$(ORG_DB) --data $(CURDIR)/$(ORG_DATA) --runs $(CURDIR)/$(ORG_RUNS)'

# Rust terminal UI (tui/): offline smoke runs piped in live. Interactive on a terminal; SMOKE_FLAGS=--plain
# prints text and fails if any run recommended an unsafe delete (CI). Replay a saved run:
#   cargo run --manifest-path tui/Cargo.toml -- runs/<run>.jsonl
smoke: seed
	PYTHONPATH=src python -m cia.smoke | cargo run -q --manifest-path tui/Cargo.toml -- $(SMOKE_FLAGS)

tui-test:
	cd tui && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test

# Re-record the stream tui/tests replays (tests/test_smoke.py fails when it is stale).
tui-fixture: seed
	PYTHONPATH=src python -m cia.smoke > tui/tests/fixtures/smoke.jsonl

# The investigator graph (LangGraph): writes it into docs/index.html#engine (React Flow draws it there)
# and prints it as Mermaid. Run after any change to src/cia/graph.py; test_graph.py fails until you do.
graph:
	PYTHONPATH=src python -m cia.graph --write-docs docs/index.html
	PYTHONPATH=src python -m cia.graph

# Bench: every change request, scored against the ground truth; fails if unsafe_deletes > 0.
# Per-case events go to runs/bench-*.jsonl, so `make monitor` shows them.
bench-fake: seed
	PYTHONPATH=src python -m cia.bench --llm fake

MODEL ?= gemma4:26b-mlx
MAX_TOKENS ?= 2048
bench-local: seed
	PYTHONPATH=src python -m cia.bench --llm openai --model $(MODEL) --max-tokens $(MAX_TOKENS) \
	  --out bench/results/$(subst :,-,$(MODEL))-$(MAX_TOKENS).json

# Hosted Gemma 4 31B on OpenRouter's free tier; needs OPENROUTER_API_KEY. Prompts leave the machine (synthetic data only).
# OR_MODEL=google/gemma-4-26b-a4b-it is the paid twin (about 1 cent for the whole bench) when the free pool is rate-limited.
OR_MODEL ?= google/gemma-4-31b-it:free
bench-openrouter: seed
	PYTHONPATH=src python -m cia.bench --llm openrouter --model $(OR_MODEL) --max-tokens $(MAX_TOKENS) \
	  --out bench/results/openrouter-$(subst :,-,$(subst /,-,$(OR_MODEL)))-$(MAX_TOKENS).json

# Bedrock reports tokens, not prices: give the model's USD per million tokens to price the run
bench-bedrock: seed
	PYTHONPATH=src python -m cia.bench --llm bedrock --out bench/results/bedrock.json \
	  $(if $(PRICE_IN),--price-in $(PRICE_IN) --price-out $(PRICE_OUT))
