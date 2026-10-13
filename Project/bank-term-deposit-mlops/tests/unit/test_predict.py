"""
Tests del servicio de predicción.

No se toca el Model Registry: se construye un `RegisteredChampion` sobre un
modelo entrenado con datos sintéticos. Así los tests corren sin MLflow y sin
depender de que exista un modelo entrenado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.features.preprocess import build_model_pipeline
from src.models.predict import PredictionService
from src.models.registry import RegisteredChampion


@pytest.fixture
def service(synthetic_features, synthetic_labels) -> PredictionService:
    """Servicio de predicción con un modelo entrenado en datos sintéticos."""
    pipeline = build_model_pipeline(LogisticRegression(max_iter=500, class_weight="balanced"))
    pipeline.fit(synthetic_features, synthetic_labels)

    champion = RegisteredChampion(
        model=pipeline,
        name="test-model",
        version="7",
        alias="champion",
        run_id="run-de-prueba",
        threshold=0.35,
        metrics={"test_pr_auc": 0.42},
    )
    return PredictionService(champion)


def test_predict_proba_returns_valid_probabilities(service, synthetic_features):
    """Una probabilidad por fila, todas en [0, 1]."""
    probabilities = service.predict_proba(synthetic_features)
    assert probabilities.shape == (len(synthetic_features),)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_predict_applies_model_threshold(service, synthetic_features):
    """La decisión usa el umbral del modelo, no un 0.5 hardcodeado."""
    result = service.predict(synthetic_features)
    expected = (result["probability"] >= service.threshold).astype(int)

    assert result["threshold_used"].unique().tolist() == [0.35]
    pd.testing.assert_series_equal(result["will_subscribe"], expected, check_names=False)


def test_predict_accepts_threshold_override(service, synthetic_features):
    """Un umbral explícito manda sobre el del modelo."""
    strict = service.predict(synthetic_features, threshold=0.95)
    lenient = service.predict(synthetic_features, threshold=0.05)
    assert strict["will_subscribe"].sum() <= lenient["will_subscribe"].sum()


def test_predict_assigns_propensity_bands(service, synthetic_features):
    """Las bandas deben ser las cuatro definidas, sin nulos."""
    bands = set(service.predict(synthetic_features)["propensity_band"])
    assert bands <= {"muy_baja", "baja", "media", "alta"}
    assert "nan" not in bands


def test_recommended_action_matches_decision(service, synthetic_features):
    """La acción recomendada es consistente con la decisión binaria."""
    result = service.predict(synthetic_features)
    called = result.loc[result["recommended_action"] == "llamar", "will_subscribe"]
    assert (called == 1).all()


def test_missing_column_is_imputed_not_crashed(service, synthetic_features):
    """
    Si falta una feature, el servicio la imputa en vez de caerse.

    En producción un campo nuevo puede faltar temporalmente; una excepción
    tumbaría el endpoint entero.
    """
    incomplete = synthetic_features.drop(columns=["euribor3m"]).head(5)
    probabilities = service.predict_proba(incomplete)
    assert np.isfinite(probabilities).all()


def test_column_order_irrelevant(service, synthetic_features):
    """El orden de las columnas de entrada no cambia la predicción."""
    sample = synthetic_features.head(20)
    shuffled = sample[list(reversed(sample.columns))]
    np.testing.assert_allclose(service.predict_proba(sample), service.predict_proba(shuffled))


def test_predict_one_returns_full_contract(service, valid_client_payload):
    """La predicción individual devuelve todos los campos del contrato."""
    result = service.predict_one(valid_client_payload)
    assert set(result) >= {
        "probability",
        "will_subscribe",
        "propensity_band",
        "recommended_action",
        "threshold_used",
        "model_name",
        "model_version",
        "predicted_at",
    }
    assert isinstance(result["will_subscribe"], bool)
    assert 0 <= result["probability"] <= 1


def test_predict_one_matches_batch(service, valid_client_payload):
    """
    Online y batch tienen que dar el mismo número.

    Es la razón de que ambos modos compartan `PredictionService`: una
    discrepancia aquí sería un bug de training/serving skew.
    """
    single = service.predict_one(valid_client_payload)["probability"]
    batched = service.predict(pd.DataFrame([valid_client_payload]))["probability"].iloc[0]
    assert single == pytest.approx(batched, abs=1e-6)


def test_predictions_are_deterministic(service, synthetic_features):
    """El mismo input siempre da el mismo output."""
    sample = synthetic_features.head(50)
    np.testing.assert_array_equal(service.predict_proba(sample), service.predict_proba(sample))


def test_info_exposes_model_version(service, config):
    """`info()` reporta la versión servida y el contrato de features."""
    info = service.info()
    assert info["version"] == "7"
    assert info["decision_threshold"] == 0.35
    assert info["feature_columns"] == config.feature_columns


def test_champion_version_coerced_to_string():
    """
    La versión siempre se expone como string.

    MLflow la devuelve como int según el backend, y el contrato de `/health` y
    `/model-info` la declara string: sin esta normalización la API responde 500
    al serializar.
    """
    champion = RegisteredChampion(
        model=None,
        name="test-model",
        version=4,  # como lo devuelve el registry
        alias="champion",
        run_id="run-de-prueba",
        threshold="0.3",
        metrics={},
    )
    assert champion.version == "4"
    assert isinstance(champion.info()["version"], str)
    assert champion.threshold == pytest.approx(0.3)
