SHELL := /bin/bash
.SHELLFLAGS := -eo pipefail -c
.DEFAULT_GOAL := help

.PHONY: help install install-dev lint format test check-artifact all
help:
	@echo 'install | install-dev LIBRARY_PATH=../../apertus-common | lint | format | test | check-artifact APERTUS_ENCODING=... | all'
install:
	uv sync --locked
install-dev:
	uv venv
	uv pip install -e "$(LIBRARY_PATH)[hf]" -e . --no-deps
	uv pip install 'datasets>=4,<6' 'pyarrow>=21' 'numpy>=2' 'transformers>=5.16' 'pydantic>=2.13.4' 'jsonschema>=4.26' 'tokenizers>=0.23.2,<0.24' pytest ruff
lint:
	uv run --no-sync ruff check src tests examples
	uv run --no-sync ruff format --check src tests examples
format:
	uv run --no-sync ruff check --fix src tests examples
	uv run --no-sync ruff format src tests examples
test:
	uv run --no-sync pytest
check-artifact:
	test -d "$(APERTUS_ENCODING)"
	APERTUS_ENCODING="$(APERTUS_ENCODING)" uv run --no-sync pytest tests/test_pipeline.py -k real_artifact
all: lint test

.PHONY: install-megatron check-megatron
install-megatron:
	uv sync --locked --extra megatron
check-megatron:
	test -d "$(MEGATRON_PATH)"
	MEGATRON_PATH="$(MEGATRON_PATH)" uv run --no-sync pytest tests/test_megatron.py

.PHONY: bench-memory
bench-memory:
	uv run --no-sync python examples/bench_memory.py
