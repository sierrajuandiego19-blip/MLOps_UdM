"""Tests del feature engineering y el preprocesamiento."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.features.preprocess import (
    PDAYS_NEVER_CONTACTED,
    DomainFeatureEngineer,
    build_feature_pipeline,
    build_model_pipeline,
    build_preprocessor,
)


def test_domain_engineer_flags_never_contacted():
    """`pdays == 999` debe traducirse a "nunca contactado", no a 999 días."""
    frame = pd.DataFrame(
        {"pdays": [PDAYS_NEVER_CONTACTED, 5], "campaign": [1, 2], "previous": [0, 1]}
    )
    result = DomainFeatureEngineer().fit_transform(frame)

    assert result["was_contacted_before"].tolist() == [0, 1]
    assert result["days_since_contact"].tolist() == [0, 5]


def test_domain_engineer_drops_raw_pdays():
    """`pdays` cruda desaparece: el centinela 999 distorsionaría el escalado."""
    frame = pd.DataFrame({"pdays": [999, 3], "campaign": [1, 1], "previous": [0, 0]})
    result = DomainFeatureEngineer().fit_transform(frame)
    assert "pdays" not in result.columns


def test_campaign_intensity_normalizes_by_history():
    """La intensidad divide contactos actuales entre 1 + contactos previos."""
    frame = pd.DataFrame({"pdays": [999, 999], "campaign": [4, 4], "previous": [0, 3]})
    result = DomainFeatureEngineer().fit_transform(frame)
    assert result["campaign_intensity"].tolist() == pytest.approx([4.0, 1.0])


def test_domain_engineer_tolerates_missing_columns():
    """Si faltan columnas, las derivadas toman valores neutros en vez de fallar."""
    result = DomainFeatureEngineer().fit_transform(pd.DataFrame({"age": [30, 40]}))
    assert result["was_contacted_before"].tolist() == [0, 0]
    assert result["campaign_intensity"].tolist() == [0.0, 0.0]


def test_feature_pipeline_outputs_numeric_matrix(synthetic_features):
    """Tras el preprocesamiento todo es numérico y sin nulos."""
    matrix = build_feature_pipeline().fit_transform(synthetic_features)
    assert matrix.shape[0] == len(synthetic_features)
    assert np.isfinite(matrix).all()


def test_preprocessor_expands_categoricals(synthetic_features, config):
    """El one-hot debe producir más columnas que features de entrada."""
    matrix = build_feature_pipeline().fit_transform(synthetic_features)
    assert matrix.shape[1] > len(config.feature_columns)


def test_scaler_standardizes_numeric_features(synthetic_features):
    """Las numéricas quedan centradas: media ~0, desviación ~1."""
    pipeline = build_feature_pipeline()
    matrix = pipeline.fit_transform(synthetic_features)
    numeric_width = len(pipeline.named_steps["preprocessor"].transformers_[0][2])

    numeric_block = matrix[:, :numeric_width]
    assert np.abs(numeric_block.mean(axis=0)).max() < 1e-8
    assert np.abs(numeric_block.std(axis=0) - 1).max() < 0.05


def test_unseen_category_does_not_crash(synthetic_features):
    """
    Una categoría nueva en producción no puede tumbar el servicio.

    `handle_unknown="ignore"` la codifica como todo ceros.
    """
    pipeline = build_feature_pipeline().fit(synthetic_features)
    new_row = synthetic_features.head(1).copy()
    new_row["job"] = "astronauta"

    matrix = pipeline.transform(new_row)
    assert matrix.shape == (1, pipeline.transform(synthetic_features.head(1)).shape[1])


def test_missing_values_are_imputed(synthetic_features):
    """Los nulos se imputan; nunca llegan al estimador."""
    pipeline = build_feature_pipeline().fit(synthetic_features)
    with_nulls = synthetic_features.head(5).copy()
    with_nulls.loc[with_nulls.index[0], "euribor3m"] = np.nan
    with_nulls.loc[with_nulls.index[1], "job"] = None

    assert np.isfinite(pipeline.transform(with_nulls)).all()


def test_column_order_does_not_change_output(synthetic_features):
    """
    El `ColumnTransformer` selecciona por nombre, no por posición.

    Importa: la API construye el DataFrame desde un JSON, cuyo orden de claves
    no está garantizado.
    """
    pipeline = build_feature_pipeline().fit(synthetic_features)
    sample = synthetic_features.head(10)
    shuffled = sample[list(reversed(sample.columns))]

    np.testing.assert_allclose(pipeline.transform(sample), pipeline.transform(shuffled))


def test_model_pipeline_trains_and_predicts(synthetic_features, synthetic_labels):
    """El pipeline completo acepta el DataFrame crudo y devuelve probabilidades."""
    pipeline = build_model_pipeline(LogisticRegression(max_iter=500, class_weight="balanced"))
    pipeline.fit(synthetic_features, synthetic_labels)

    probabilities = pipeline.predict_proba(synthetic_features)[:, 1]
    assert probabilities.shape == (len(synthetic_features),)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_preprocessor_drops_unlisted_columns(synthetic_features):
    """Una columna que no está en el contrato se descarta, no se cuela al modelo."""
    pipeline = build_feature_pipeline().fit(synthetic_features)
    baseline_width = pipeline.transform(synthetic_features.head(1)).shape[1]

    with_extra = synthetic_features.head(1).copy()
    with_extra["columna_inesperada"] = 123
    assert pipeline.transform(with_extra).shape[1] == baseline_width


def test_preprocessor_is_rebuilt_independently():
    """Cada llamada devuelve una instancia nueva, sin estado compartido."""
    assert build_preprocessor() is not build_preprocessor()
