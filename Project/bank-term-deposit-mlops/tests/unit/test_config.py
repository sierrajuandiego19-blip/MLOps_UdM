"""Tests de la carga de configuración."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config, project_root


def test_config_loads_required_sections(config):
    """Todas las secciones que el código consume deben existir."""
    for section in ("project", "data", "features", "training", "business", "mlflow", "monitoring"):
        assert section in config.keys(), f"Falta la sección '{section}'"


def test_feature_columns_are_numeric_plus_categorical(config):
    """`feature_columns` concatena ambas listas sin perder ni duplicar nada."""
    expected = len(config.features.numeric) + len(config.features.categorical)
    assert len(config.feature_columns) == expected
    assert len(set(config.feature_columns)) == expected


def test_target_not_in_feature_columns(config):
    """El target nunca puede colarse entre las features: sería leakage total."""
    assert config.data.target not in config.feature_columns


def test_duration_is_dropped(config):
    """`duration` debe estar en la lista de descartes (leakage conocido)."""
    assert "duration" in config.data.drop_columns
    assert "duration" not in config.feature_columns


def test_split_sizes_leave_room_for_training(config):
    """Train no puede quedarse sin datos."""
    assert 0 < config.data.test_size < 0.5
    assert 0 < config.data.valid_size < 0.5
    assert config.data.test_size + config.data.valid_size < 0.6


def test_monitoring_thresholds_are_ordered(config):
    """El umbral de alerta debe ser más exigente que el de advertencia."""
    assert config.monitoring.psi_warning < config.monitoring.psi_alert


def test_decision_threshold_is_a_probability(config):
    """El umbral de decisión tiene que ser una probabilidad válida."""
    assert 0 < config.training.decision_threshold < 1


def test_path_helper_creates_directory(config):
    """`path()` devuelve una ruta absoluta y crea el directorio."""
    resolved = config.path("reports")
    assert resolved.is_absolute()
    assert resolved.exists()
    assert resolved.is_dir()


def test_unknown_key_raises_attribute_error(config):
    """Una clave inexistente falla con un mensaje que la nombra."""
    with pytest.raises(AttributeError, match="no_existe"):
        _ = config.no_existe


def test_config_is_memoized():
    """`load_config` está memoizado: siempre la misma instancia."""
    assert load_config() is load_config()


def test_project_root_contains_configs():
    """La raíz detectada debe contener `configs/config.yaml`."""
    assert (project_root() / "configs" / "config.yaml").exists()


def test_mlflow_tracking_uri_is_absolute(config):
    """
    La ruta de SQLite se resuelve contra la raíz del proyecto.

    `sqlite:///mlflow.db` es relativo al directorio de trabajo: sin resolverlo,
    ejecutar desde `notebooks/` o desde un worker de Prefect abriría una base de
    datos distinta y vacía.
    """
    uri = config.mlflow_tracking_uri
    if uri.startswith("sqlite:///"):
        assert Path(uri.removeprefix("sqlite:///")).is_absolute()
        assert str(config.root.as_posix()) in uri


def test_mlflow_tracking_uri_leaves_remote_untouched(monkeypatch):
    """Una URI remota no se toca: no hay ninguna ruta que resolver."""
    load_config.cache_clear()
    monkeypatch.setenv("BTD_MLFLOW__TRACKING_URI", "http://mlflow.example.com:5000")
    try:
        assert load_config().mlflow_tracking_uri == "http://mlflow.example.com:5000"
    finally:
        load_config.cache_clear()


def test_env_override_applies(monkeypatch):
    """Las variables BTD_ sobreescriben valores anidados del YAML."""
    load_config.cache_clear()
    monkeypatch.setenv("BTD_TRAINING__OPTUNA_TRIALS", "3")
    try:
        assert load_config().training.optuna_trials == 3
    finally:
        load_config.cache_clear()
