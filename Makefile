package := siso


# Convenience targets

.PHONY: sync
sync:
	uv sync


# Linting targets

.PHONY: format
format:
	uv run ruff format

.PHONY: lint
lint:
	uv run ruff check --fix


# Test targets

.PHONY: pytest
pytest:
	uv run pytest

.PHONY: mypy
mypy:
	uv run mypy

.PHONY: lint-check
lint-check:
	uv run ruff check
	uv run ruff format --check

.PHONY: test
test: pytest mypy lint-check


# Build targets (used from CI)

.PHONY: build
build:
	uv build
