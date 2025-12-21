.PHONY: help install test clean run-examples build publish format lint check

help:
	@echo "Diet Pandas - Make Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make install       - Install package in development mode"
	@echo "  make test          - Run all tests"
	@echo "  make test-verbose  - Run tests with verbose output"
	@echo "  make coverage      - Run tests with coverage report"
	@echo "  make format        - Format code with Black and isort"
	@echo "  make lint          - Lint code with Flake8"
	@echo "  make check         - Format and lint code"
	@echo "  make run-examples  - Run example scripts"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make build         - Build distribution packages"
	@echo "  make publish-test  - Publish to TestPyPI"
	@echo "  make publish       - Publish to PyPI"

install:
	pip install -e ".[dev]"

test:
	pytest tests/

test-verbose:
	pytest tests/ -v

coverage:
	pytest tests/ --cov=dietpandas --cov-report=html --cov-report=term

run-examples:
	python scripts/examples.py

run-demo:
	python scripts/demo.py

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

lint:
	flake8 src/ tests/ scripts/

check: format lint
	@echo "Code quality checks complete!"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

build: clean
	pip install build
	python -m build

publish-test: build
	pip install twine
	twine upload --repository testpypi dist/*

publish: build
	pip install twine
	twine upload dist/*
