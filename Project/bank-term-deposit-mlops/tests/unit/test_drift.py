"""Tests de la detección de drift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    detect_drift,
    population_stability_index,
    simulate_drifted_batch,
)


def test_psi_is_zero_for_identical_distributions():
    """La misma distribución contra sí misma no tiene drift."""
    series = pd.Series(np.random.default_rng(0).normal(size=5000))
    assert population_stability_index(series, series) == pytest.approx(0.0, abs=1e-6)


def test_psi_grows_with_distance():
    """Cuanto más se desplaza la distribución, mayor el PSI."""
    rng = np.random.default_rng(1)
    reference = pd.Series(rng.normal(0, 1, 5000))

    small_shift = population_stability_index(reference, pd.Series(rng.normal(0.2, 1, 5000)))
    large_shift = population_stability_index(reference, pd.Series(rng.normal(2.0, 1, 5000)))

    assert small_shift < large_shift
    assert large_shift > 0.25


def test_psi_is_non_negative():
    """El PSI es una divergencia: nunca negativo."""
    rng = np.random.default_rng(2)
    for scale in (0.5, 1.0, 3.0):
        value = population_stability_index(
            pd.Series(rng.normal(0, 1, 1000)), pd.Series(rng.normal(0, scale, 1000))
        )
        assert value >= 0


def test_psi_handles_categorical():
    """El PSI también cubre variables categóricas."""
    reference = pd.Series(["a"] * 700 + ["b"] * 300)
    identical = pd.Series(["a"] * 700 + ["b"] * 300)
    shifted = pd.Series(["a"] * 300 + ["b"] * 700)

    assert population_stability_index(reference, identical) == pytest.approx(0.0, abs=1e-6)
    assert population_stability_index(reference, shifted) > 0.25


def test_psi_handles_new_category():
    """Una categoría nueva aporta drift sin provocar log(0)."""
    reference = pd.Series(["a"] * 500 + ["b"] * 500)
    current = pd.Series(["a"] * 300 + ["b"] * 300 + ["c"] * 400)

    value = population_stability_index(reference, current)
    assert np.isfinite(value)
    assert value > 0


def test_psi_handles_constant_column():
    """Una columna constante no tiene bins: PSI 0, sin división por cero."""
    constant = pd.Series([5.0] * 100)
    assert population_stability_index(constant, constant) == 0.0


def test_psi_handles_empty_input():
    """Series vacías devuelven 0 en vez de lanzar."""
    assert population_stability_index(pd.Series(dtype=float), pd.Series([1.0, 2.0])) == 0.0


def test_detect_drift_finds_nothing_on_resample(synthetic_features):
    """Un remuestreo de los mismos datos no debe disparar alertas."""
    reference = synthetic_features
    current = reference.sample(n=400, replace=True, random_state=3)

    report = detect_drift(reference, current)
    assert report.alerts == []
    assert not report.needs_retraining


def test_detect_drift_flags_injected_shift(synthetic_features):
    """Un desplazamiento grande e intencional debe detectarse."""
    reference = synthetic_features
    current = reference.copy()
    current["euribor3m"] = current["euribor3m"] + 4.0

    report = detect_drift(reference, current)
    assert "euribor3m" in [item.feature for item in report.alerts]
    assert report.needs_retraining


def test_detect_drift_covers_all_contract_features(synthetic_features, config):
    """Se evalúan todas las features del contrato, ni más ni menos."""
    report = detect_drift(synthetic_features, synthetic_features.sample(200, replace=True))
    checked = {item.feature for item in report.features}
    assert checked == set(config.feature_columns)


def test_drift_results_sorted_by_psi(synthetic_features):
    """El reporte viene ordenado: lo más grave primero."""
    current = synthetic_features.copy()
    current["age"] = current["age"] + 25

    report = detect_drift(synthetic_features, current)
    values = [item.psi for item in report.features]
    assert values == sorted(values, reverse=True)


def test_prediction_drift_detected_when_scores_shift(synthetic_features):
    """El drift de scores se mide aparte del de features."""
    reference_scores = np.random.default_rng(4).beta(2, 8, 2000)
    current_scores = np.random.default_rng(5).beta(8, 2, 2000)

    report = detect_drift(
        synthetic_features,
        synthetic_features.sample(300, replace=True),
        reference_scores=reference_scores,
        current_scores=current_scores,
    )
    assert report.prediction_drift is not None
    assert report.prediction_drift.severity == "alert"


def test_drift_summary_is_serializable(synthetic_features):
    """El resumen debe poder volcarse a JSON."""
    import json

    report = detect_drift(synthetic_features, synthetic_features.sample(200, replace=True))
    assert json.loads(json.dumps(report.summary()))


def test_simulate_without_drift_preserves_distribution(synthetic_features):
    """Con `drift_strength=0` el lote es solo un remuestreo."""
    batch = simulate_drifted_batch(synthetic_features, n_rows=1500, drift_strength=0.0)

    assert len(batch) == 1500
    assert list(batch.columns) == list(synthetic_features.columns)
    assert batch["euribor3m"].mean() == pytest.approx(
        synthetic_features["euribor3m"].mean(), abs=0.2
    )


def test_simulate_with_drift_shifts_macro_features(synthetic_features):
    """El drift simulado mueve las macro en la dirección prevista."""
    batch = simulate_drifted_batch(synthetic_features, n_rows=1500, drift_strength=1.0)

    assert batch["euribor3m"].mean() > synthetic_features["euribor3m"].mean()
    assert batch["nr.employed"].mean() < synthetic_features["nr.employed"].mean()
    assert batch["age"].mean() < synthetic_features["age"].mean()


def test_simulate_respects_value_bounds(synthetic_features):
    """El lote simulado sigue siendo válido: nada fuera de rango."""
    batch = simulate_drifted_batch(synthetic_features, n_rows=800, drift_strength=1.0)

    assert batch["age"].between(18, 95).all()
    assert batch["campaign"].between(1, 60).all()
    assert batch["euribor3m"].between(0, 8).all()


def test_simulate_drops_target_column(synthetic_features, synthetic_labels, config):
    """El lote simulado es de inferencia: no lleva target."""
    with_target = synthetic_features.copy()
    with_target[config.data.target] = synthetic_labels.to_numpy()

    batch = simulate_drifted_batch(with_target, n_rows=100)
    assert config.data.target not in batch.columns
