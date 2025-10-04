.PHONY: install format lint run-api run-ui build-graph test

install:
	python -m venv .venv
	.\.venv\Scripts\python -m pip install --upgrade pip
	.\.venv\Scripts\pip install -e .[dev]

format:
	.\.venv\Scripts\ruff format

lint:
	.\.venv\Scripts\ruff check app scripts ui tests

build-graph:
	.\.venv\Scripts\python scripts/build_graph.py

run-api:
	.\.venv\Scripts\uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	.\.venv\Scripts\streamlit run ui/app.py

test:
	.\.venv\Scripts\pytest
