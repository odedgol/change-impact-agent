.PHONY: seed test baseline

seed:
	PYTHONPATH=src python -m cia.generator --identities 500 --out data/org.db --truth bench/ground_truth.json

test:
	PYTHONPATH=src python -m pytest -q

baseline: seed
	PYTHONPATH=src python -m cia.baseline --db data/org.db --truth bench/ground_truth.json
