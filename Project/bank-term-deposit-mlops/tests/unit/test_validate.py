"""Tests de la validación de datos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.validate import validate_dataframe


def test_valid_dataframe_passes(synthetic_raw):
    """Datos bien formados pasan sin errores."""
    result = validate_dataframe(synthetic_raw, require_target=True)
    assert result.is_valid
    assert result.errors == []
    assert result.n_rows == len(synthetic_raw)


def test_empty_dataframe_fails():
    """Un DataFrame vacío se rechaza."""
    result = validate_dataframe(pd.DataFrame(), require_target=True)
    assert not result.is_valid
    assert any("vacío" in error for error in result.errors)


def test_missing_feature_column_fails(synthetic_raw):
    """Falta una feature del contrato -> error bloqueante."""
    result = validate_dataframe(synthetic_raw.drop(columns=["euribor3m"]), require_target=True)
    assert not result.is_valid
    assert any("euribor3m" in error for error in result.errors)


def test_missing_target_fails_when_required(synthetic_raw):
    """Sin target no se puede entrenar."""
    result = validate_dataframe(synthetic_raw.drop(columns=["y"]), require_target=True)
    assert not result.is_valid
    assert any("target" in error for error in result.errors)


def test_missing_target_is_fine_for_inference(synthetic_raw):
    """En inferencia el target es justamente lo que no se tiene."""
    result = validate_dataframe(synthetic_raw.drop(columns=["y"]), require_target=False)
    assert result.is_valid


def test_single_class_target_fails(synthetic_raw):
    """Un target con una sola clase no permite entrenar."""
    frame = synthetic_raw.copy()
    frame["y"] = "no"
    result = validate_dataframe(frame, require_target=True)
    assert not result.is_valid
    assert any("una sola clase" in error for error in result.errors)


def test_unexpected_target_values_fail(synthetic_raw):
    """Valores de target fuera de yes/no o 0/1 se rechazan."""
    frame = synthetic_raw.copy()
    frame.loc[frame.index[:10], "y"] = "maybe"
    result = validate_dataframe(frame, require_target=True)
    assert not result.is_valid
    assert any("inesperados" in error for error in result.errors)


def test_wrong_numeric_type_fails(synthetic_raw):
    """Una numérica que llega como texto es un error de contrato."""
    frame = synthetic_raw.copy()
    frame["age"] = frame["age"].astype(str)
    result = validate_dataframe(frame, require_target=True)
    assert not result.is_valid
    assert any("age" in error for error in result.errors)


def test_out_of_range_values_warn_but_pass(synthetic_raw):
    """Un valor atípico advierte pero no bloquea: puede ser un cliente real."""
    frame = synthetic_raw.copy()
    frame.loc[frame.index[0], "age"] = 150
    result = validate_dataframe(frame, require_target=True)
    assert result.is_valid
    assert any("age" in warning for warning in result.warnings)


def test_few_nulls_warn_but_pass(synthetic_raw):
    """Pocos nulos advierten; el imputador del pipeline los resuelve."""
    frame = synthetic_raw.copy()
    frame.loc[frame.index[:5], "euribor3m"] = np.nan
    result = validate_dataframe(frame, require_target=True)
    assert result.is_valid
    assert any("euribor3m" in warning for warning in result.warnings)


def test_mostly_null_column_fails(synthetic_raw):
    """Una columna casi toda nula sí bloquea: no hay nada que imputar."""
    frame = synthetic_raw.copy()
    frame.loc[frame.index[: int(len(frame) * 0.8)], "euribor3m"] = np.nan
    result = validate_dataframe(frame, require_target=True)
    assert not result.is_valid


def test_duplicates_warn(synthetic_raw):
    """Las filas duplicadas se reportan."""
    frame = pd.concat([synthetic_raw, synthetic_raw.head(3)], ignore_index=True)
    result = validate_dataframe(frame, require_target=True)
    assert any("duplicadas" in warning for warning in result.warnings)


def test_raise_if_invalid_raises(synthetic_raw):
    """`raise_if_invalid` lanza con el detalle de los errores."""
    result = validate_dataframe(synthetic_raw.drop(columns=["y"]), require_target=True)
    with pytest.raises(ValueError, match="Validación de datos fallida"):
        result.raise_if_invalid()


def test_summary_is_serializable(synthetic_raw):
    """El resumen debe poder viajar a un artifact de Prefect."""
    summary = validate_dataframe(synthetic_raw).summary()
    assert set(summary) >= {"is_valid", "n_rows", "n_errors", "n_warnings"}
    assert isinstance(summary["n_rows"], int)
