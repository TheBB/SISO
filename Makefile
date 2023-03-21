.PHONY: install mypy lint pytest test fmt fmtcheck

install:
	poetry install --with=dev

mypy:
	poetry run mypy siso

lint:
	poetry run ruff siso

pytest:
	poetry run pytest

fmt:
	poetry run black siso
	poetry run isort siso

fmtcheck:
	poetry run black siso --check
	poetry run isort siso --check

test: mypy pytest lint fmtcheck
