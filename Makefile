package := siso


# Convenience targets

.PHONY: install
install:
	pdm install --dev


# Linting targets

.PHONY: format
format:
	pdm run ruff format

.PHONY: lint
lint:
	pdm run ruff check --fix


# Test targets

.PHONY: pytest
pytest:
	pdm run pytest

.PHONY: mypy
mypy:
	pdm run mypy

.PHONY: lint-check
lint-check:
	pdm run ruff check
	pdm run ruff format --check

.PHONY: test
test: pytest mypy lint-check


# Build targets (used from CI)

.PHONY: build
build:
	pdm build
