PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

# Pacotes em ``src/``; testes importam ``main`` na raiz.
export PYTHONPATH := $(abspath $(CURDIR)/src):$(abspath $(CURDIR))

.PHONY: help install install-dev requirements lint lint-fix format test test-fast coverage run docker-up docker-down docker-fresh clean check check-rings check-rings-strict validate-platform validate-platform-infra

help:
	@echo "Alvos principais:"
	@echo "  make install-dev   pip install -e \".[dev]\""
	@echo "  make lint          ruff check (alinha com pyproject.toml)"
	@echo "  make lint-fix      ruff check --fix"
	@echo "  make format        black em src, tests, main e DAG"
	@echo "  make test          pytest (cobertura conforme pyproject.toml)"
	@echo "  make test-fast     pytest sem cobertura (mais rápido)"
	@echo "  make coverage      mesmo fluxo de teste com relatórios de cobertura"
	@echo "  make check         lint + test-fast"
	@echo "  make check-rings   docs Fase 0 + avisos de import entre anéis"
	@echo "  make run           uvicorn local (porta 8000)"
	@echo "  make docker-up     docker compose up --build"
	@echo "  make docker-down   docker compose down"
	@echo "  make docker-fresh  down -v, prune cache/imagens locais, build --no-cache, up -d"
	@echo "  make validate-platform       validação completa Docker (plataforma, sem churn)"
	@echo "  make validate-platform-infra só infra Docker + health (sem E2E API)"
	@echo "  make clean         artefatos de build e caches locais"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

requirements:
	$(PIP) install -r requirements.txt

lint:
	$(PYTHON) -m ruff check .

lint-fix:
	$(PYTHON) -m ruff check --fix .

format:
	$(PYTHON) -m black src tests main.py airflow/dags

test:
	$(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest -q -o addopts=

coverage:
	$(PYTHON) -m pytest --cov=. --cov-report=html --cov-report=term-missing

run:
	$(PYTHON) -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker compose up -d

docker-down:
	docker compose down

# Reset total: contentores, volumes nomeados, imagens locais do compose, cache de build.
# Preserva bind mounts no host (./src, ./data/recommendation, ratings.csv, etc.).
docker-fresh:
	docker compose down -v --remove-orphans --rmi local
	docker builder prune -af
	docker compose build --no-cache --pull
	docker compose up -d --force-recreate
	@echo "Stack limpa e a subir. Acompanhe: docker compose ps"

tc02-repro:
	PYTHONPATH=src dvc repro

tc02-validate:
	PYTHONPATH=src python3 scripts/validate_env.py

tc02-promote:
	PYTHONPATH=src python3 scripts/ml/promote_registry.py

tc02-up:
	docker compose up -d --build mlflow_server worker_recommendation

check: lint test-fast

check-rings:
	python3 scripts/check_ring_imports.py

check-rings-strict:
	python3 scripts/check_ring_imports.py --strict

validate-platform:
	@test -n "$$VALIDATE_API_PASSWORD" || (echo "Defina VALIDATE_API_PASSWORD (senha do user seed em init_db/database.sql)"; exit 1)
	bash scripts/validate_platform.sh

validate-platform-infra:
	bash scripts/validate_platform.sh --infra-only --skip-local

clean:
	rm -rf build dist *.egg-info htmlcov .pytest_cache .ruff_cache .coverage coverage.xml
	-find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
