# =============================================================================
# Atajos del proyecto.
#
# En Windows, `make` no viene instalado. Las alternativas, en orden de
# preferencia:
#   1. Los comandos `uv run ...` directamente (están todos en el README)
#   2. `./run.ps1 <target>` — equivalente en PowerShell, incluido en el repo
#   3. `winget install GnuWin32.Make` o usar Git Bash con make
# =============================================================================

.PHONY: help install train score monitor api mlflow prefect deploy-flows \
        export-model test test-cov lint format check docker-build docker-up \
        docker-down notebooks clean clean-all all

UV := uv run

help:  ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --- Instalación -------------------------------------------------------------

install:  ## Instala dependencias y los hooks de pre-commit
	uv sync --all-groups
	$(UV) pre-commit install

setup:  ## TRAS CLONAR: instala, reapunta MLflow a esta máquina y verifica (~3 min)
	uv sync --all-groups
	$(UV) python scripts/relocate_mlflow.py
	-$(UV) prefect work-pool create local-pool --type process
	$(UV) python deploy_flows.py
	$(UV) pytest -q
	$(UV) python -c "from src.models.registry import load_champion; c = load_champion(allow_local_fallback=False); print(f'campeon: {c.name} v{c.version} umbral {c.threshold:.2f}')"
	@echo ""
	@echo "LISTO. Esperado: 153 tests | campeon v1 umbral 0.42"

bootstrap:  ## ALTERNATIVA: reproduce el estado desde cero reentrenando (~12 min)
	uv sync --all-groups
	$(UV) pytest -q
	$(UV) python -m src.pipelines.training_flow
	@echo ">>> Segundo entrenamiento: la v2 empata y queda como challenger"
	$(UV) python -m src.pipelines.training_flow
	$(UV) python -m src.pipelines.batch_scoring_flow
	$(UV) python -m src.pipelines.monitoring_flow
	$(UV) python scripts/export_model.py
	-$(UV) prefect work-pool create local-pool --type process
	$(UV) python deploy_flows.py
	@echo ""
	@echo "LISTO. Esperado: Test PR-AUC 0.4909 | champion->v1, challenger->v2"
	@echo "Guion de exposicion: docs/presentacion.md"

# --- Pipelines ---------------------------------------------------------------

train:  ## Entrena el modelo (ingesta, experimentos, registro, promoción)
	$(UV) python -m src.pipelines.training_flow

train-fast:  ## Entrena sin búsqueda de hiperparámetros (para pruebas rápidas)
	$(UV) python -c "from src.pipelines.training_flow import training_pipeline; training_pipeline(skip_tuning=True)"

score:  ## Puntúa un lote y genera la lista de llamadas
	$(UV) python -m src.pipelines.batch_scoring_flow

monitor:  ## Mide drift y genera el reporte HTML
	$(UV) python -m src.pipelines.monitoring_flow

all: train score monitor  ## Ejecuta el flujo completo de punta a punta

# --- Servicios ---------------------------------------------------------------

api:  ## Levanta la API en modo desarrollo
	$(UV) uvicorn src.api.main:app --reload

# Ruta absoluta a propósito: con una relativa, MLflow abre una base de datos
# distinta según el directorio desde donde se lance.
mlflow:  ## Abre la UI de MLflow del proyecto (puerto 5001)
	$(UV) mlflow server --backend-store-uri "sqlite:///$(CURDIR)/mlflow.db" --host 127.0.0.1 --port 5001

prefect:  ## Levanta el servidor de Prefect
	$(UV) prefect server start

deploy-flows:  ## Registra los flows con sus schedules
	$(UV) python deploy_flows.py

export-model:  ## Exporta el campeón para deployment sin MLflow
	$(UV) python scripts/export_model.py

# --- Calidad -----------------------------------------------------------------

test:  ## Corre los tests
	$(UV) pytest

test-cov:  ## Corre los tests con reporte de cobertura
	$(UV) pytest --cov=src --cov-report=term-missing --cov-report=html

lint:  ## Revisa lint y formato sin modificar nada
	$(UV) ruff check src tests
	$(UV) black --check src tests

format:  ## Aplica formato y arregla lo que ruff pueda
	$(UV) ruff check --fix src tests
	$(UV) black src tests

check: lint test  ## Lo que verifica el CI

# --- Docker ------------------------------------------------------------------

docker-build:  ## Construye la imagen de la API
	docker build -t bank-term-deposit-api .

docker-up:  ## Levanta el stack (MLflow + API)
	docker compose up --build

docker-down:  ## Detiene el stack
	docker compose down

# --- Notebooks ---------------------------------------------------------------

notebooks:  ## Ejecuta los tres notebooks y guarda sus salidas
	$(UV) jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=1800 notebooks/*.ipynb

# --- Limpieza ----------------------------------------------------------------

clean:  ## Borra cachés y artefactos temporales
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-all: clean  ## Borra además datos, modelos y experimentos (deja el repo como recién clonado)
	rm -rf data/raw/* data/processed/* data/reference/* data/predictions/*
	rm -rf reports/* logs/* mlruns mlflow.db
	rm -rf models/champion models/champion_metadata.json models/model_card.md
