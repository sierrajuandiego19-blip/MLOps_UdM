"""Tests del ETL y la partición de datos."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.split import build_splits, clean_dataset


def test_clean_drops_leakage_column(synthetic_raw):
    """`duration` tiene que desaparecer: es la columna con leakage."""
    assert "duration" in synthetic_raw.columns
    assert "duration" not in clean_dataset(synthetic_raw).columns


def test_clean_encodes_target_as_binary(synthetic_raw, config):
    """El target `yes`/`no` se codifica a 1/0."""
    clean = clean_dataset(synthetic_raw)
    assert set(clean[config.data.target].unique()) <= {0, 1}
    assert pd.api.types.is_integer_dtype(clean[config.data.target])


def test_clean_target_encoding_preserves_counts(synthetic_raw, config):
    """El número de positivos antes y después debe coincidir."""
    expected = int((synthetic_raw[config.data.target] == config.data.positive_label).sum())
    assert int(clean_dataset(synthetic_raw)[config.data.target].sum()) == expected


def test_clean_keeps_contract_columns(synthetic_raw, config):
    """La salida son exactamente las features del contrato más el target."""
    clean = clean_dataset(synthetic_raw)
    assert list(clean.columns) == [*config.feature_columns, config.data.target]


def test_clean_is_idempotent(synthetic_raw):
    """Limpiar dos veces da el mismo resultado: el target ya numérico no se toca."""
    once = clean_dataset(synthetic_raw)
    twice = clean_dataset(once)
    pd.testing.assert_frame_equal(once, twice)


def test_clean_works_without_target(synthetic_raw, config):
    """En inferencia no hay target y el ETL debe seguir funcionando."""
    clean = clean_dataset(synthetic_raw.drop(columns=[config.data.target]))
    assert list(clean.columns) == config.feature_columns


def test_splits_partition_all_rows(synthetic_raw):
    """Las tres particiones suman el total: no se pierde ni se duplica nada."""
    splits = build_splits(synthetic_raw)
    assert sum(splits.sizes().values()) == len(synthetic_raw)


def test_splits_respect_configured_proportions(synthetic_raw, config):
    """Los tamaños deben acercarse a las proporciones configuradas."""
    splits = build_splits(synthetic_raw)
    total = len(synthetic_raw)

    assert splits.sizes()["test"] / total == pytest.approx(config.data.test_size, abs=0.02)
    assert splits.sizes()["valid"] / total == pytest.approx(config.data.valid_size, abs=0.02)


def test_splits_are_stratified(synthetic_raw):
    """
    La tasa de positivos debe ser casi igual en los tres splits.

    Con 11% de positivos, un split no estratificado puede dejar un test con una
    tasa base distinta y hacer incomparables las métricas.
    """
    rates = build_splits(synthetic_raw).positive_rates()
    assert max(rates.values()) - min(rates.values()) < 0.03


def test_splits_do_not_leak_target_into_features(synthetic_raw, config):
    """El target no puede aparecer entre las columnas de features."""
    splits = build_splits(synthetic_raw)
    for frame in (splits.x_train, splits.x_valid, splits.x_test):
        assert config.data.target not in frame.columns
        assert list(frame.columns) == config.feature_columns


def test_splits_are_reproducible(synthetic_raw):
    """Misma semilla, mismos splits: requisito de reproducibilidad."""
    first = build_splits(synthetic_raw)
    second = build_splits(synthetic_raw)

    pd.testing.assert_frame_equal(first.x_test, second.x_test)
    pd.testing.assert_series_equal(first.y_test, second.y_test)


def test_splits_are_disjoint(synthetic_raw):
    """
    Ninguna fila puede estar en dos splits a la vez.

    Se compara por el contenido completo de la fila, ya que los índices se
    resetean al particionar.
    """
    splits = build_splits(synthetic_raw)

    def signatures(frame: pd.DataFrame) -> set[tuple]:
        return {tuple(row) for row in frame.itertuples(index=False)}

    train, valid, test = (
        signatures(splits.x_train),
        signatures(splits.x_valid),
        signatures(splits.x_test),
    )
    assert not train & valid
    assert not train & test
    assert not valid & test


def test_build_splits_rejects_invalid_data(synthetic_raw):
    """Datos que no pasan la validación no llegan a particionarse."""
    with pytest.raises(ValueError, match="Validación de datos fallida"):
        build_splits(synthetic_raw.drop(columns=["euribor3m"]))
