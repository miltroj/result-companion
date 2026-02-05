.PHONY: install test test-unit test-integration test-coverage test-e2e validate lint format help

install:  ## Install dependencies with dev packages
	poetry install --with=dev

test: test-unit test-integration  ## Run all tests

test-unit:  ## Run unit tests
	poetry run pytest -vv tests/unittests/

test-integration:  ## Run integration tests
	poetry run pytest -vv tests/integration/

test-coverage:  ## Run unit tests with coverage
	poetry run pytest --cov=result_companion tests/unittests

test-e2e:  ## Run Robot Framework e2e tests
	poetry run robot --loglevel TRACE tests/integration/e2e/

validate:  ## Validate developer setup with local Ollama
	poetry run result-companion analyze -o examples/run_test/output.xml -r log_with_results.html

lint:  ## Run linting checks
	poetry run black --check .
	poetry run isort --check --profile black .

format:  ## Format code
	poetry run black .
	poetry run isort --profile black .

help:  ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
