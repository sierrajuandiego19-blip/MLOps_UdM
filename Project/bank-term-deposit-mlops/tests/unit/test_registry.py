"""
Tests del registry y de la copia local del modelo.

La lógica de promoción (`promote_if_better`) necesita un tracking server, así
que no se prueba aquí. Lo que sí se prueba es el **round-trip de exportación**,
que es el mecanismo del que depende el deployment en cloud: si una copia
exportada no se puede volver a cargar, o carga pero predice distinto, el
contenedor en producción sirve otro modelo del que se registró.
"""

from __future__ import annotations

import json

import mlflow
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.features.preprocess import build_model_pipeline
from src.models.registry import (
    LOCAL_METADATA_FILENAME,
    RegisteredChampion,
    load_local_champion,
)


@pytest.fixture
def exported_model(tmp_path, synthetic_features, synthetic_labels):
    """Exporta a disco un modelo entrenado, igual que `export_champion`."""
    pipeline = build_model_pipeline(LogisticRegression(max_iter=500, class_weight="balanced"))
    pipeline.fit(synthetic_features, synthetic_labels)

    destination = tmp_path / "champion"
    mlflow.sklearn.save_model(
        pipeline,
        path=str(destination),
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )
    (destination / LOCAL_METADATA_FILENAME).write_text(
        json.dumps(
            {
                "name": "test-model",
                "version": "5",
                "alias": "champion",
                "run_id": "run-de-prueba",
                "threshold": 0.37,
                "metrics": {"test_pr_auc": 0.44},
            }
        ),
        encoding="utf-8",
    )
    return destination, pipeline


def test_local_champion_round_trip_preserves_predictions(exported_model, synthetic_features):
    """
    La copia local debe predecir **exactamente** lo mismo que el original.

    Es la garantía que sostiene el deployment en cloud: el contenedor sirve una
    copia exportada, no el modelo del registry.
    """
    destination, original = exported_model
    loaded = load_local_champion(destination)

    sample = synthetic_features.head(100)
    np.testing.assert_allclose(
        original.predict_proba(sample)[:, 1],
        loaded.model.predict_proba(sample)[:, 1],
        rtol=0,
        atol=0,
    )


def test_local_champion_preserves_contract(exported_model):
    """El umbral y la versión viajan con el modelo, no se pierden."""
    destination, _ = exported_model
    loaded = load_local_champion(destination)

    assert loaded.name == "test-model"
    assert loaded.version == "5"
    assert loaded.threshold == pytest.approx(0.37)
    assert loaded.metrics["test_pr_auc"] == pytest.approx(0.44)


def test_local_champion_missing_directory_raises(tmp_path):
    """Sin copia exportada, el error dice qué comando la genera."""
    with pytest.raises(FileNotFoundError, match="export_model"):
        load_local_champion(tmp_path / "no-existe")


def test_local_champion_missing_metadata_raises(tmp_path, exported_model):
    """Un directorio con modelo pero sin metadata es inservible: falta el umbral."""
    destination, _ = exported_model
    (destination / LOCAL_METADATA_FILENAME).unlink()

    with pytest.raises(FileNotFoundError):
        load_local_champion(destination)


def test_champion_info_excludes_model_object():
    """`info()` tiene que ser serializable: el modelo no puede ir en el JSON."""
    champion = RegisteredChampion(
        model=object(),
        name="test-model",
        version="1",
        alias="champion",
        run_id="run",
        threshold=0.4,
        metrics={"test_pr_auc": 0.5},
    )
    info = champion.info()
    assert "model" not in info
    assert json.loads(json.dumps(info))
