PYTHON := python3
PIP := $(PYTHON) -m pip

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

requirements:
	$(PIP) install -r requirements.txt

lint:
	ruff .

format:
	black .

test:
	pytest

coverage:
	pytest --cov=. --cov-report=html --cov-report=term-missing

run:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf build dist *.egg-info .pytest_cache __pycache__ htmlcov
	