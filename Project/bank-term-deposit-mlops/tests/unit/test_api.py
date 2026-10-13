"""
Tests de la API.

Se inyecta un modelo entrenado en datos sintéticos directamente en el estado de
la aplicación, en vez de levantar MLflow. Lo que se prueba aquí es la capa HTTP:
rutas, códigos de estado, validación y forma de las respuestas.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.api import main as api_main
from src.features.preprocess import build_model_pipeline
from src.models.predict import PredictionService
from src.models.registry import RegisteredChampion


@pytest.fixture
def client(synthetic_features, synthetic_labels, monkeypatch) -> TestClient:
    """Cliente de test con un modelo falso ya cargado."""
    pipeline = build_model_pipeline(LogisticRegression(max_iter=500, class_weight="balanced"))
    pipeline.fit(synthetic_features, synthetic_labels)

    champion = RegisteredChampion(
        model=pipeline,
        name="test-model",
        version="3",
        alias="champion",
        run_id="run-de-prueba",
        threshold=0.30,
        metrics={"test_pr_auc": 0.45, "test_roc_auc": 0.80},
    )
    service = PredictionService(champion)

    # El lifespan intentaría leer del registry; se le entrega el modelo falso.
    monkeypatch.setattr(api_main, "get_prediction_service", lambda force_reload=False: service)
    with TestClient(api_main.app) as test_client:
        yield test_client


@pytest.fixture
def degraded_client(monkeypatch) -> TestClient:
    """Cliente de test simulando que el modelo no pudo cargarse."""

    def fail(force_reload: bool = False):
        raise RuntimeError("no hay modelo campeón registrado")

    monkeypatch.setattr(api_main, "get_prediction_service", fail)
    with TestClient(api_main.app) as test_client:
        yield test_client


# --- Servicio -------------------------------------------------------------


def test_root_lists_endpoints(client):
    """La raíz describe el servicio y sus rutas."""
    response = client.get("/")
    assert response.status_code == 200
    assert "endpoints" in response.json()


def test_health_reports_healthy_with_model(client):
    """Con modelo cargado el health check es `healthy`."""
    payload = client.get("/health").json()
    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is True
    assert payload["model_version"] == "3"


def test_model_info_exposes_metrics(client):
    """`/model-info` publica versión, umbral y métricas."""
    payload = client.get("/model-info").json()
    assert payload["version"] == "3"
    assert payload["decision_threshold"] == 0.30
    assert "test_pr_auc" in payload["metrics"]


def test_metrics_endpoint_is_prometheus_text(client):
    """`/metrics` responde texto plano con cabeceras HELP/TYPE."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# HELP api_predict_requests_total" in response.text


def test_openapi_schema_available(client):
    """El esquema OpenAPI se genera: `/docs` funciona."""
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    assert "/predict/batch" in schema["paths"]


# --- Predicción individual ------------------------------------------------


def test_predict_returns_probability(client, valid_client_payload):
    """Una predicción válida devuelve 200 con el contrato completo."""
    response = client.post("/predict", json=valid_client_payload)
    assert response.status_code == 200

    payload = response.json()
    assert 0 <= payload["probability"] <= 1
    assert isinstance(payload["will_subscribe"], bool)
    assert payload["model_version"] == "3"
    assert payload["recommended_action"] in {"llamar", "no_llamar"}


def test_predict_decision_consistent_with_threshold(client, valid_client_payload):
    """La decisión debe coincidir con comparar la probabilidad al umbral."""
    payload = client.post("/predict", json=valid_client_payload).json()
    assert payload["will_subscribe"] == (payload["probability"] >= payload["threshold_used"])


def test_predict_rejects_missing_field(client, valid_client_payload):
    """Falta un campo -> 422, no un 500."""
    incomplete = {key: value for key, value in valid_client_payload.items() if key != "age"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_predict_rejects_unknown_category(client, valid_client_payload):
    """Categoría inválida -> 422 nombrando el campo."""
    response = client.post("/predict", json={**valid_client_payload, "marital": "casado"})
    assert response.status_code == 422
    assert "marital" in response.text


def test_predict_rejects_out_of_range_value(client, valid_client_payload):
    """Valor fuera de rango -> 422."""
    assert client.post("/predict", json={**valid_client_payload, "age": 200}).status_code == 422


def test_predict_rejects_extra_field(client, valid_client_payload):
    """Campo desconocido -> 422: el contrato está cerrado."""
    response = client.post("/predict", json={**valid_client_payload, "saldo": 1000})
    assert response.status_code == 422


def test_predict_rejects_empty_body(client):
    """Un body vacío no pasa la validación."""
    assert client.post("/predict", json={}).status_code == 422


# --- Predicción batch -----------------------------------------------------


def test_batch_predict_returns_one_row_per_client(client, valid_client_payload):
    """El batch devuelve exactamente una predicción por cliente."""
    response = client.post("/predict/batch", json={"clients": [valid_client_payload] * 4})
    assert response.status_code == 200

    payload = response.json()
    assert payload["n_clients"] == 4
    assert len(payload["predictions"]) == 4
    assert [item["index"] for item in payload["predictions"]] == [0, 1, 2, 3]


def test_batch_matches_single_prediction(client, valid_client_payload):
    """Batch y online deben coincidir para el mismo cliente."""
    single = client.post("/predict", json=valid_client_payload).json()["probability"]
    batched = client.post("/predict/batch", json={"clients": [valid_client_payload]}).json()
    assert batched["predictions"][0]["probability"] == pytest.approx(single, abs=1e-6)


def test_batch_aggregates_are_consistent(client, valid_client_payload):
    """Los agregados deben cuadrar con las filas devueltas."""
    payload = client.post("/predict/batch", json={"clients": [valid_client_payload] * 6}).json()
    expected_calls = sum(item["will_subscribe"] for item in payload["predictions"])
    assert payload["n_to_call"] == expected_calls


def test_batch_rejects_empty_list(client):
    """Un batch sin clientes -> 422."""
    assert client.post("/predict/batch", json={"clients": []}).status_code == 422


def test_batch_rejects_invalid_client(client, valid_client_payload):
    """Un cliente inválido invalida la petición completa."""
    response = client.post(
        "/predict/batch",
        json={"clients": [valid_client_payload, {**valid_client_payload, "job": "pirata"}]},
    )
    assert response.status_code == 422


# --- Recarga del modelo ---------------------------------------------------


def test_reload_returns_new_version(client):
    """`/reload` recarga el campeón y reporta la versión servida."""
    response = client.post("/reload")
    assert response.status_code == 200
    assert response.json()["model_version"] == "3"


# --- Modo degradado -------------------------------------------------------


def test_health_reports_degraded_without_model(degraded_client):
    """
    Sin modelo, el servicio arranca pero se declara `degraded`.

    Así el orquestador lo saca de rotación en vez de reiniciarlo en bucle.
    """
    payload = degraded_client.get("/health").json()
    assert payload["status"] == "degraded"
    assert payload["model_loaded"] is False
    assert payload["detail"]


def test_predict_returns_503_without_model(degraded_client, valid_client_payload):
    """Sin modelo, predecir devuelve 503 con una causa legible."""
    response = degraded_client.post("/predict", json=valid_client_payload)
    assert response.status_code == 503
    assert "modelo" in response.json()["detail"].lower()


def test_metrics_reports_model_not_loaded(degraded_client):
    """El gauge `api_model_loaded` refleja que no hay modelo."""
    assert "api_model_loaded 0" in degraded_client.get("/metrics").text
