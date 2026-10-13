"""Tests de los contratos de la API (validación de inputs)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import BatchPredictionRequest, ClientFeatures


def test_valid_payload_parses(valid_client_payload):
    """Un payload correcto se acepta."""
    client = ClientFeatures(**valid_client_payload)
    assert client.age == 41
    assert client.euribor3m == pytest.approx(1.313)


def test_to_model_input_uses_dataset_names(valid_client_payload, config):
    """
    La salida debe usar los nombres con punto del dataset.

    El pipeline de sklearn selecciona por nombre: si la API enviara
    `emp_var_rate` en vez de `emp.var.rate`, la feature se perdería en silencio.
    """
    payload = ClientFeatures(**valid_client_payload).to_model_input()
    assert set(payload) == set(config.feature_columns)
    assert "emp.var.rate" in payload


def test_missing_required_field_rejected(valid_client_payload):
    """Falta un campo obligatorio -> error de validación."""
    incomplete = {key: value for key, value in valid_client_payload.items() if key != "age"}
    with pytest.raises(ValidationError, match="age"):
        ClientFeatures(**incomplete)


def test_extra_field_rejected(valid_client_payload):
    """
    Un campo de más es un error, no algo a ignorar.

    Casi siempre significa un nombre mal escrito: fallar fuerte es mejor que
    puntuar sin esa feature.
    """
    with pytest.raises(ValidationError):
        ClientFeatures(**{**valid_client_payload, "ingreso_anual": 50000})


def test_unknown_category_rejected(valid_client_payload):
    """Una categoría que el modelo nunca vio se rechaza con 422."""
    with pytest.raises(ValidationError, match="job"):
        ClientFeatures(**{**valid_client_payload, "job": "astronauta"})


@pytest.mark.parametrize("age", [16, 101, -5])
def test_age_out_of_range_rejected(valid_client_payload, age):
    """La edad vive en [17, 100]."""
    with pytest.raises(ValidationError, match="age"):
        ClientFeatures(**{**valid_client_payload, "age": age})


def test_campaign_must_be_at_least_one(valid_client_payload):
    """Si hay una predicción, hubo al menos un contacto."""
    with pytest.raises(ValidationError, match="campaign"):
        ClientFeatures(**{**valid_client_payload, "campaign": 0})


def test_pdays_sentinel_accepted(valid_client_payload):
    """999 es el centinela válido de "nunca contactado"."""
    assert ClientFeatures(**{**valid_client_payload, "pdays": 999}).pdays == 999


def test_pdays_real_days_accepted(valid_client_payload):
    """Los días reales de contacto previo también son válidos."""
    assert ClientFeatures(**{**valid_client_payload, "pdays": 7}).pdays == 7


def test_pdays_intermediate_value_rejected(valid_client_payload):
    """
    Un `pdays` entre 41 y 998 no existe en el dataset.

    Suele ser un error de codificación en el sistema de origen.
    """
    with pytest.raises(ValidationError, match="pdays"):
        ClientFeatures(**{**valid_client_payload, "pdays": 500})


def test_euribor_out_of_range_rejected(valid_client_payload):
    """Un euríbor del 50% no es un escenario, es un error de datos."""
    with pytest.raises(ValidationError, match="euribor3m"):
        ClientFeatures(**{**valid_client_payload, "euribor3m": 50.0})


def test_macro_field_aliases_required(valid_client_payload):
    """Los campos macro solo se aceptan por su alias con punto."""
    payload = {key: value for key, value in valid_client_payload.items() if key != "emp.var.rate"}
    with pytest.raises(ValidationError):
        ClientFeatures(**{**payload, "emp_var_rate": -1.8})


def test_batch_request_accepts_multiple_clients(valid_client_payload):
    """La petición batch acepta varios clientes."""
    request = BatchPredictionRequest(clients=[valid_client_payload] * 5)
    assert len(request.clients) == 5


def test_batch_request_rejects_empty_list():
    """Una petición batch sin clientes no tiene sentido."""
    with pytest.raises(ValidationError):
        BatchPredictionRequest(clients=[])


def test_batch_request_enforces_size_limit(valid_client_payload):
    """El tope de 1000 protege la memoria del servicio."""
    with pytest.raises(ValidationError):
        BatchPredictionRequest(clients=[valid_client_payload] * 1001)


def test_json_schema_includes_example():
    """El esquema lleva un ejemplo: la doc de `/docs` queda usable."""
    assert "example" in ClientFeatures.model_json_schema()
