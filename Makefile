.PHONY: seed test lint baseline run-local monitor graph bench-fake bench-local bench-openrouter bench-bedrock

seed:
	PYTHONPATH=src python -m cia.generator --identities 500 --out data/org.db --truth bench/ground_truth.json

test:
	PYTHONPATH=src python -m pytest -q

lint:
	ruff check src tests

baseline: seed
	PYTHONPATH=src python -m cia.baseline --db data/org.db --truth bench/ground_truth.json

# One request on a local model through Ollama (default gemma4:26b-mlx). Example: make run-local ID=u00002
run-local: seed
	PYTHONPATH=src python -m cia.investigator --llm openai --identity $(ID) --events runs/$(ID).jsonl

# Run monitor: launch runs, watch them live, browse runs/*.jsonl. http://127.0.0.1:$(PORT)
PORT ?= 8765
monitor: seed
	PYTHONPATH=src python -m cia.monitor --port $(PORT)

# The investigator graph (LangGraph) as Mermaid text; paste into docs or mermaid.live.
graph:
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
# OR_MODEL=google/gemma-4-26b-a4b-it is the paid twin (about 1 cent for the 20 cases) when the free pool is rate-limited.
OR_MODEL ?= google/gemma-4-31b-it:free
bench-openrouter: seed
	PYTHONPATH=src python -m cia.bench --llm openrouter --model $(OR_MODEL) --max-tokens $(MAX_TOKENS) \
	  --out bench/results/openrouter-$(subst :,-,$(subst /,-,$(OR_MODEL)))-$(MAX_TOKENS).json

bench-bedrock: seed
	PYTHONPATH=src python -m cia.bench --llm bedrock --out bench/results/bedrock.json
