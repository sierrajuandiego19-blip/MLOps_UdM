# =============================================================================
# Imagen de la API de predicción.
#
# Build en dos etapas: las dependencias se resuelven en `builder` y solo el
# entorno virtual resultante pasa a la imagen final. Evita que las herramientas
# de compilación terminen en producción y mantiene la imagen alrededor de
# 1/3 del tamaño de un build en una sola etapa.
# =============================================================================

# ---------- Etapa 1: dependencias ----------
FROM python:3.11-slim AS builder

# uv resuelve e instala mucho más rápido que pip, y respeta el uv.lock
# exacto del proyecto: el contenedor obtiene las mismas versiones que local.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Los manifiestos primero: mientras no cambien, Docker reutiliza la capa de
# dependencias y el rebuild tras editar código tarda segundos.
COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# ---------- Etapa 2: runtime ----------
FROM python:3.11-slim

# Usuario sin privilegios: un proceso comprometido no debe ser root.
RUN groupadd --system --gid 1001 appuser \
    && useradd --system --uid 1001 --gid appuser --create-home appuser

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Solo lo que el servicio necesita en ejecución. Los notebooks, los tests y los
# datos crudos no entran: no aportan nada en producción y engordan la imagen.
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser pyproject.toml README.md ./

# El campeón exportado (~1 MB, generado por `scripts/export_model.py`). Con él
# dentro, el contenedor sirve predicciones sin necesitar un tracking server:
# es lo que hace viable el deployment en cloud. `models/.gitkeep` garantiza que
# el directorio exista aunque todavía no se haya exportado nada.
COPY --chown=appuser:appuser models/ ./models/

# Directorios que el proceso escribe en runtime.
RUN mkdir -p data/raw data/processed data/reference data/predictions reports logs \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# El health check es el del propio servicio: reporta `degraded` si el proceso
# vive pero no pudo cargar el modelo.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request, json, sys; \
r = json.load(urllib.request.urlopen('http://localhost:8000/health', timeout=4)); \
sys.exit(0 if r['status'] == 'healthy' else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
