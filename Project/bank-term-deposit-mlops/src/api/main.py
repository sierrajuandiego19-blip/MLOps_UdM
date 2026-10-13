"""
API REST de predicción (FastAPI).

Endpoints:

| Método | Ruta           | Para qué                                        |
|--------|----------------|-------------------------------------------------|
| GET    | `/`            | Metadatos del servicio                          |
| GET    | `/health`      | Health check (liveness/readiness)               |
| GET    | `/model-info`  | Versión y métricas del modelo servido           |
| POST   | `/predict`     | Puntúa un cliente                               |
| POST   | `/predict/batch`| Puntúa hasta 1000 clientes                     |
| POST   | `/reload`      | Recarga el campeón del registry sin reiniciar   |
| GET    | `/metrics`     | Contadores en texto plano para scraping         |

Levantar en local:
    uv run uvicorn src.api.main:app --reload

Documentación interactiva: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import JSONResponse

from src.api.schemas import (
    BatchPredictionItem,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ClientFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from src.config import load_config
from src.models.predict import PredictionService, get_prediction_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

config = load_config()

# Estado del proceso. Un contador en memoria alcanza para un servicio de un
# proceso; con varias réplicas esto iría a un backend compartido (ver
# docs/monitoring.md).
_state: dict[str, object] = {"service": None, "startup_error": None}
_counters: Counter[str] = Counter()
_latencies_ms: list[float] = []
LATENCY_WINDOW = 1000


def _service() -> PredictionService:
    """Servicio de predicción, o 503 si el modelo no pudo cargarse."""
    service = _state.get("service")
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "El modelo no está disponible. "
                f"Causa: {_state.get('startup_error')}. "
                "Ejecuta el pipeline de entrenamiento y luego POST /reload."
            ),
        )
    return service  # type: ignore[return-value]


def _record_latency(elapsed_ms: float) -> None:
    """Guarda la latencia en una ventana acotada, para no crecer sin límite."""
    _latencies_ms.append(elapsed_ms)
    if len(_latencies_ms) > LATENCY_WINDOW:
        del _latencies_ms[: len(_latencies_ms) - LATENCY_WINDOW]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Carga el modelo al arrancar.

    Si falla, el servicio **sí arranca** pero marca `degraded`: así el
    orquestador ve un proceso vivo con un health check que falla (y un motivo
    legible) en vez de un contenedor en bucle de reinicios.
    """
    try:
        _state["service"] = get_prediction_service()
        _state["startup_error"] = None
        service: PredictionService = _state["service"]  # type: ignore[assignment]
        logger.info(
            "API lista — modelo %s v%s (umbral=%.2f)",
            service.champion.name,
            service.champion.version,
            service.threshold,
        )
    except Exception as error:  # noqa: BLE001 - se reporta vía /health
        _state["service"] = None
        _state["startup_error"] = str(error)
        logger.error("El modelo no pudo cargarse al arrancar: %s", error)

    yield
    logger.info("API detenida")


app = FastAPI(
    title=config.api.title,
    version=config.api.version,
    description=(
        "Predice la probabilidad de que un cliente bancario suscriba un depósito "
        "a plazo fijo, para priorizar las llamadas del call center.\n\n"
        "Modelo servido con alias `champion` desde el Model Registry de MLflow."
    ),
    lifespan=lifespan,
)


@app.get("/", tags=["servicio"])
def root() -> dict[str, object]:
    """Metadatos del servicio y rutas disponibles."""
    return {
        "service": config.api.title,
        "version": config.api.version,
        "project": config.project.name,
        "docs": "/docs",
        "endpoints": [
            "GET /health",
            "GET /model-info",
            "GET /metrics",
            "POST /predict",
            "POST /predict/batch",
            "POST /reload",
        ],
    }


@app.get("/health", response_model=HealthResponse, tags=["servicio"])
def health() -> HealthResponse:
    """
    Health check para liveness y readiness.

    Responde 200 con `degraded` si el proceso vive pero no hay modelo, para que
    el balanceador lo saque de rotación sin matar el contenedor.
    """
    service = _state.get("service")
    if service is None:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            api_version=config.api.version,
            detail=str(_state.get("startup_error")),
        )

    champion = service.champion  # type: ignore[union-attr]
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=champion.name,
        model_version=champion.version,
        api_version=config.api.version,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["modelo"])
def model_info() -> ModelInfoResponse:
    """Versión, umbral y métricas de test del modelo que se está sirviendo."""
    return ModelInfoResponse(**_service().info())


@app.post("/predict", response_model=PredictionResponse, tags=["predicción"])
def predict(client: ClientFeatures) -> PredictionResponse:
    """
    Puntúa un cliente.

    Devuelve la probabilidad de suscripción, la decisión al umbral vigente y la
    banda de propensión. La decisión es una recomendación: el umbral es un
    parámetro de negocio, no una propiedad del modelo.
    """
    service = _service()
    started = time.perf_counter()
    try:
        result = service.predict_one(client.to_model_input())
    except Exception as error:  # noqa: BLE001
        _counters["predict_errors"] += 1
        logger.exception("Error al predecir")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al generar la predicción: {error}",
        ) from error

    elapsed_ms = (time.perf_counter() - started) * 1000
    _record_latency(elapsed_ms)
    _counters["predict_requests"] += 1
    _counters["predict_positive" if result["will_subscribe"] else "predict_negative"] += 1

    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["predicción"])
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Puntúa varios clientes en una sola petición (máximo 1000).

    Mucho más eficiente que N llamadas a `/predict`: el pipeline vectoriza el
    preprocesamiento y la inferencia sobre todo el lote.
    """
    service = _service()
    started = time.perf_counter()

    frame = pd.DataFrame([client.to_model_input() for client in request.clients])
    try:
        scored = service.predict(frame)
    except Exception as error:  # noqa: BLE001
        _counters["batch_errors"] += 1
        logger.exception("Error al predecir en batch")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al generar las predicciones: {error}",
        ) from error

    elapsed_ms = (time.perf_counter() - started) * 1000
    _record_latency(elapsed_ms)
    _counters["batch_requests"] += 1
    _counters["batch_clients"] += len(scored)

    return BatchPredictionResponse(
        predictions=[
            BatchPredictionItem(
                index=position,
                probability=round(float(row["probability"]), 6),
                will_subscribe=bool(row["will_subscribe"]),
                propensity_band=str(row["propensity_band"]),
            )
            for position, (_, row) in enumerate(scored.iterrows())
        ],
        n_clients=len(scored),
        n_to_call=int(scored["will_subscribe"].sum()),
        mean_probability=round(float(scored["probability"].mean()), 6),
        threshold_used=round(service.threshold, 4),
        model_name=service.champion.name,
        model_version=service.champion.version,
        predicted_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


@app.post("/reload", tags=["modelo"])
def reload_model() -> JSONResponse:
    """
    Recarga el modelo campeón desde el registry, sin reiniciar el proceso.

    Es el mecanismo de despliegue: el pipeline de entrenamiento promueve una
    versión nueva al alias `champion` y luego llama a este endpoint.
    """
    try:
        _state["service"] = get_prediction_service(force_reload=True)
        _state["startup_error"] = None
        service: PredictionService = _state["service"]  # type: ignore[assignment]
        _counters["reloads"] += 1
        logger.info("Modelo recargado: v%s", service.champion.version)
        return JSONResponse(
            {
                "status": "reloaded",
                "model_name": service.champion.name,
                "model_version": service.champion.version,
                "decision_threshold": round(service.threshold, 4),
            }
        )
    except Exception as error:  # noqa: BLE001
        _state["startup_error"] = str(error)
        logger.error("Falló la recarga del modelo: %s", error)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No se pudo recargar el modelo: {error}",
        ) from error


@app.get("/metrics", tags=["servicio"])
def metrics() -> Response:
    """
    Contadores del servicio en formato texto, estilo Prometheus.

    No es un cliente de Prometheus completo (sería una dependencia más para lo
    que aquí es una demostración), pero el formato es el que un scraper espera,
    así que apuntar Prometheus a esta ruta funciona tal cual.
    """
    service = _state.get("service")
    average_latency = sum(_latencies_ms) / len(_latencies_ms) if _latencies_ms else 0.0

    lines = [
        "# HELP api_predict_requests_total Peticiones a /predict",
        "# TYPE api_predict_requests_total counter",
        f"api_predict_requests_total {_counters['predict_requests']}",
        "# HELP api_predict_positive_total Predicciones por encima del umbral",
        "# TYPE api_predict_positive_total counter",
        f"api_predict_positive_total {_counters['predict_positive']}",
        "# HELP api_batch_requests_total Peticiones a /predict/batch",
        "# TYPE api_batch_requests_total counter",
        f"api_batch_requests_total {_counters['batch_requests']}",
        "# HELP api_batch_clients_total Clientes puntuados en batch",
        "# TYPE api_batch_clients_total counter",
        f"api_batch_clients_total {_counters['batch_clients']}",
        "# HELP api_errors_total Errores de predicción",
        "# TYPE api_errors_total counter",
        f"api_errors_total {_counters['predict_errors'] + _counters['batch_errors']}",
        "# HELP api_model_reloads_total Recargas del modelo",
        "# TYPE api_model_reloads_total counter",
        f"api_model_reloads_total {_counters['reloads']}",
        "# HELP api_latency_ms_avg Latencia media de inferencia en ms",
        "# TYPE api_latency_ms_avg gauge",
        f"api_latency_ms_avg {average_latency:.3f}",
        "# HELP api_model_loaded 1 si hay un modelo cargado",
        "# TYPE api_model_loaded gauge",
        f"api_model_loaded {1 if service else 0}",
    ]
    return Response(content="\n".join(lines) + "\n", media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=False,
    )
