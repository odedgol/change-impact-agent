.PHONY: seed test lint baseline run-local

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
