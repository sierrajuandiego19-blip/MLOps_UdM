"""
Evaluación de modelos.

Dos bloques de métricas:

* **Técnicas** — PR-AUC (métrica primaria), ROC-AUC, precision, recall, F1,
  Brier score. Con 11% de positivos el accuracy no dice nada útil: un modelo
  que siempre responde "no" acierta el 89% y no sirve para nada.
* **De negocio** — con el call center capaz de llamar a una fracción de la base,
  lo que importa es el lift sobre el top-k y la utilidad esperada en dólares.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import load_config

logger = logging.getLogger(__name__)


def classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    """
    Métricas técnicas de clasificación.

    Args:
        y_true: etiquetas reales (0/1).
        y_proba: probabilidad predicha de la clase positiva.
        threshold: umbral de decisión. Si es None usa el de configuración.

    Returns:
        Diccionario de métricas, listo para `mlflow.log_metrics`.
    """
    config = load_config()
    cutoff = config.training.decision_threshold if threshold is None else threshold

    y_true = np.asarray(y_true).astype(int)
    y_pred = (y_proba >= cutoff).astype(int)

    true_neg, false_pos, false_neg, true_pos = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "threshold": float(cutoff),
        "true_positives": int(true_pos),
        "false_positives": int(false_pos),
        "true_negatives": int(true_neg),
        "false_negatives": int(false_neg),
        "positive_rate_actual": float(y_true.mean()),
        "positive_rate_predicted": float(y_pred.mean()),
    }


def lift_at_k(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    k_fraction: float,
) -> dict[str, float]:
    """
    Lift sobre el top-k: la pregunta real del call center.

    Si solo alcanza a llamar al k% de la base, ¿cuántos suscriptores captura
    priorizando por score frente a llamar al azar?

    Args:
        y_true: etiquetas reales (0/1).
        y_proba: probabilidad predicha.
        k_fraction: fracción de la base que se alcanza a contactar.

    Returns:
        Tasa de conversión en el top-k, lift y recall capturado.
    """
    y_true = np.asarray(y_true).astype(int)
    n_total = len(y_true)
    n_target = max(1, int(round(n_total * k_fraction)))

    top_indices = np.argsort(-y_proba)[:n_target]
    conversions_in_top = int(y_true[top_indices].sum())
    total_conversions = int(y_true.sum())

    baseline_rate = y_true.mean()
    top_rate = conversions_in_top / n_target

    return {
        f"top{int(k_fraction * 100)}_conversion_rate": float(top_rate),
        f"top{int(k_fraction * 100)}_lift": (
            float(top_rate / baseline_rate) if baseline_rate > 0 else 0.0
        ),
        f"top{int(k_fraction * 100)}_recall": (
            float(conversions_in_top / total_conversions) if total_conversions > 0 else 0.0
        ),
        f"top{int(k_fraction * 100)}_calls": int(n_target),
    }


def business_metrics(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    """
    Traduce la matriz de confusión a dólares usando los supuestos de negocio.

    Cada llamada cuesta `cost_per_call_usd`; cada suscripción aporta
    `profit_per_subscription_usd`. Se compara contra la política actual del
    banco (llamar a todo el mundo).

    Args:
        y_true: etiquetas reales (0/1).
        y_proba: probabilidad predicha.
        threshold: umbral de decisión. None = el de configuración.

    Returns:
        Utilidad del modelo, de la política actual, y la ganancia incremental.
    """
    config = load_config()
    cost_per_call = config.business.cost_per_call_usd
    profit_per_subscription = config.business.profit_per_subscription_usd
    cutoff = config.training.decision_threshold if threshold is None else threshold

    y_true = np.asarray(y_true).astype(int)
    y_pred = (y_proba >= cutoff).astype(int)

    calls_made = int(y_pred.sum())
    subscriptions_won = int(((y_pred == 1) & (y_true == 1)).sum())
    subscriptions_missed = int(((y_pred == 0) & (y_true == 1)).sum())

    model_profit = subscriptions_won * profit_per_subscription - calls_made * cost_per_call
    call_everyone_profit = int(y_true.sum()) * profit_per_subscription - len(y_true) * cost_per_call

    return {
        "calls_made": calls_made,
        "subscriptions_won": subscriptions_won,
        "subscriptions_missed": subscriptions_missed,
        "profit_usd": float(model_profit),
        "profit_call_everyone_usd": float(call_everyone_profit),
        "profit_uplift_usd": float(model_profit - call_everyone_profit),
        "cost_per_acquisition_usd": (
            float(calls_made * cost_per_call / subscriptions_won)
            if subscriptions_won > 0
            else float("inf")
        ),
    }


def evaluate_model(
    model: Any,
    x: pd.DataFrame,
    y: np.ndarray | pd.Series,
    prefix: str = "",
    threshold: float | None = None,
) -> dict[str, float]:
    """
    Evalúa un modelo y devuelve todas las métricas con prefijo opcional.

    Args:
        model: pipeline entrenado con `predict_proba`.
        x: features.
        y: etiquetas reales.
        prefix: prefijo para las claves, por ejemplo "test_".
        threshold: umbral de decisión.

    Returns:
        Métricas técnicas + lift + negocio, con las claves prefijadas.
    """
    config = load_config()
    y_proba = model.predict_proba(x)[:, 1]

    metrics: dict[str, float] = {}
    metrics.update(classification_metrics(y, y_proba, threshold=threshold))
    metrics.update(lift_at_k(y, y_proba, config.business.call_capacity_fraction))
    metrics.update(business_metrics(y, y_proba, threshold=threshold))

    if prefix:
        metrics = {f"{prefix}{key}": value for key, value in metrics.items()}
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Busca el umbral que maximiza la utilidad en dólares.

    El umbral óptimo de negocio casi nunca es 0.5: depende de la asimetría
    entre el costo de una llamada y el beneficio de una suscripción.

    Args:
        y_true: etiquetas reales.
        y_proba: probabilidad predicha.
        grid: umbrales a probar. None = de 0.05 a 0.95 en pasos de 0.01.

    Returns:
        (mejor umbral, utilidad en ese umbral).
    """
    candidates = np.arange(0.05, 0.96, 0.01) if grid is None else grid
    best_threshold, best_profit = 0.5, -np.inf

    for candidate in candidates:
        profit = business_metrics(y_true, y_proba, threshold=float(candidate))["profit_usd"]
        if profit > best_profit:
            best_threshold, best_profit = float(candidate), profit

    logger.info(
        "Umbral óptimo de negocio: %.2f (utilidad = $%s)",
        best_threshold,
        f"{best_profit:,.0f}",
    )
    return best_threshold, best_profit
