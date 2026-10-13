"""Tests de las métricas de evaluación."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.evaluate import (
    business_metrics,
    classification_metrics,
    find_optimal_threshold,
    lift_at_k,
)


@pytest.fixture
def perfect_scores():
    """Un clasificador perfecto: 20% de positivos, separados sin solapamiento."""
    y_true = np.array([1] * 20 + [0] * 80)
    y_proba = np.array([0.95] * 20 + [0.05] * 80)
    return y_true, y_proba


@pytest.fixture
def random_scores():
    """Un clasificador sin información: scores constantes."""
    rng = np.random.default_rng(0)
    y_true = (rng.random(1000) < 0.11).astype(int)
    return y_true, np.full(1000, 0.11)


def test_perfect_classifier_reaches_pr_auc_one(perfect_scores):
    """Separación perfecta -> PR-AUC y ROC-AUC = 1."""
    metrics = classification_metrics(*perfect_scores, threshold=0.5)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)


def test_random_classifier_pr_auc_near_base_rate(random_scores):
    """Sin información, el PR-AUC tiende a la tasa base de positivos."""
    y_true, y_proba = random_scores
    metrics = classification_metrics(y_true, y_proba, threshold=0.5)
    assert metrics["pr_auc"] == pytest.approx(y_true.mean(), abs=0.03)
    assert metrics["roc_auc"] == pytest.approx(0.5, abs=0.02)


def test_confusion_matrix_counts_add_up(perfect_scores):
    """Los cuatro cuadrantes deben sumar el total de observaciones."""
    y_true, y_proba = perfect_scores
    metrics = classification_metrics(y_true, y_proba, threshold=0.5)
    total = (
        metrics["true_positives"]
        + metrics["false_positives"]
        + metrics["true_negatives"]
        + metrics["false_negatives"]
    )
    assert total == len(y_true)


def test_lower_threshold_increases_recall(perfect_scores):
    """Bajar el umbral nunca puede reducir el recall."""
    y_true = np.array([1] * 30 + [0] * 70)
    y_proba = np.concatenate([np.linspace(0.3, 0.9, 30), np.linspace(0.05, 0.5, 70)])

    low = classification_metrics(y_true, y_proba, threshold=0.2)
    high = classification_metrics(y_true, y_proba, threshold=0.8)
    assert low["recall"] >= high["recall"]


def test_lift_above_one_for_good_model(perfect_scores):
    """Un modelo que ordena bien tiene lift > 1 en el top-k."""
    y_true, y_proba = perfect_scores
    result = lift_at_k(y_true, y_proba, k_fraction=0.2)
    assert result["top20_lift"] > 1
    assert result["top20_calls"] == 20


def test_lift_equals_one_for_random_model(random_scores):
    """Sin capacidad de ordenar, el lift es 1: da igual a quién llamar."""
    y_true, y_proba = random_scores
    result = lift_at_k(y_true, y_proba, k_fraction=0.2)
    assert result["top20_lift"] == pytest.approx(1.0, abs=0.35)


def test_lift_recall_bounded(perfect_scores):
    """El recall del top-k es una fracción: entre 0 y 1."""
    y_true, y_proba = perfect_scores
    result = lift_at_k(y_true, y_proba, k_fraction=0.2)
    assert 0 <= result["top20_recall"] <= 1


def test_business_metrics_profit_arithmetic():
    """La utilidad = ganancias por suscripción - costo de todas las llamadas."""
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.9, 0.9, 0.1])

    result = business_metrics(y_true, y_proba, threshold=0.5)
    assert result["calls_made"] == 3
    assert result["subscriptions_won"] == 2
    assert result["subscriptions_missed"] == 0
    # 2 * 80 - 3 * 8 = 136 con los supuestos por defecto.
    assert result["profit_usd"] == pytest.approx(2 * 80.0 - 3 * 8.0)


def test_business_metrics_cost_per_acquisition_infinite_without_wins():
    """Sin suscripciones ganadas, el costo por adquisición es infinito."""
    y_true = np.array([0, 0, 0])
    y_proba = np.array([0.9, 0.9, 0.9])
    assert business_metrics(y_true, y_proba, threshold=0.5)["cost_per_acquisition_usd"] == np.inf


def test_targeting_beats_calling_everyone(perfect_scores):
    """
    El valor del modelo es ahorrar llamadas inútiles.

    Con separación perfecta, segmentar debe ganarle a llamar a toda la base.
    """
    y_true, y_proba = perfect_scores
    result = business_metrics(y_true, y_proba, threshold=0.5)
    assert result["profit_uplift_usd"] > 0


def test_optimal_threshold_is_a_probability(perfect_scores):
    """El umbral óptimo tiene que caer dentro de la grilla explorada."""
    y_true, y_proba = perfect_scores
    threshold, profit = find_optimal_threshold(y_true, y_proba)
    assert 0.05 <= threshold <= 0.95
    assert profit > 0


def test_optimal_threshold_maximizes_profit():
    """Ningún otro umbral de la grilla debe superar la utilidad del elegido."""
    rng = np.random.default_rng(7)
    y_true = (rng.random(500) < 0.2).astype(int)
    y_proba = np.clip(y_true * 0.4 + rng.random(500) * 0.5, 0, 1)

    best_threshold, best_profit = find_optimal_threshold(y_true, y_proba)
    for candidate in np.arange(0.05, 0.96, 0.05):
        profit = business_metrics(y_true, y_proba, threshold=float(candidate))["profit_usd"]
        assert profit <= best_profit + 1e-6, f"umbral {candidate} supera a {best_threshold}"


def test_metrics_accept_pandas_series(synthetic_labels):
    """Las métricas deben aceptar Series de pandas igual que arrays."""
    scores = np.linspace(0.01, 0.99, len(synthetic_labels))
    metrics = classification_metrics(synthetic_labels, scores, threshold=0.5)
    assert 0 <= metrics["pr_auc"] <= 1
