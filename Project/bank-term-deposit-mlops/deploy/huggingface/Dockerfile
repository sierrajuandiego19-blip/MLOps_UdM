# =============================================================================
# Imagen para Hugging Face Spaces (SDK: docker).
#
# Diferencias respecto al Dockerfile principal, todas impuestas por la
# plataforma:
#
#   * Puerto 7860 en lugar de 8000 (es el que Spaces expone por defecto).
#   * Usuario con UID 1000 llamado `user`: Spaces ejecuta el contenedor con ese
#     UID y, si los archivos no le pertenecen, el proceso no puede escribir.
#   * `BTD_SERVING__PREFER_LOCAL_MODEL=true` fijado en la imagen: en Spaces no
#     hay tracking server de MLflow, el modelo viaja dentro del contenedor.
#   * HOME=/home/user para que las librerías que escriben caché no fallen.
#
# No se usa en local; para eso está el Dockerfile de la raíz.
# =============================================================================

# ---------- Etapa 1: dependencias ----------
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# ---------- Etapa 2: runtime ----------
FROM python:3.11-slim

# Spaces ejecuta con UID 1000. Crear el usuario con ese UID exacto evita
# problemas de permisos de escritura en runtime.
RUN useradd --create-home --uid 1000 user

ENV PATH="/app/.venv/bin:$PATH" \
    HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    BTD_SERVING__PREFER_LOCAL_MODEL=true

WORKDIR /app

COPY --from=builder --chown=user:user /app/.venv /app/.venv

COPY --chown=user:user src/ ./src/
COPY --chown=user:user configs/ ./configs/
COPY --chown=user:user models/ ./models/
COPY --chown=user:user pyproject.toml README.md ./

RUN mkdir -p data/raw data/processed data/reference data/predictions reports logs \
    && chown -R user:user /app

USER user

EXPOSE 7860

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
