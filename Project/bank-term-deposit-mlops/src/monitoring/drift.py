"""
Detección de drift de datos y de predicciones.

Tres tests, cada uno para lo que sabe medir:

* **PSI** (Population Stability Index) — para numéricas y categóricas. Es el
  estándar de facto en banca y tiene umbrales interpretables: <0.1 estable,
  0.1-0.25 cambio moderado, >0.25 cambio severo.
* **Kolmogorov-Smirnov** — para numéricas. Da un p-valor, pero con muestras
  grandes detecta diferencias irrelevantes; se reporta junto al PSI, no en
  lugar de él.
* **Chi-cuadrado** — para categóricas, misma salvedad sobre el p-valor.

Se mide drift sobre las features **y** sobre la distribución de scores del
modelo. El segundo suele ser la primera señal: resume en una sola serie el
efecto combinado de todos los cambios de entrada.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from src.config import load_config

logger = logging.getLogger(__name__)

PSI_EPSILON = 1e-6
DEFAULT_PSI_BINS = 10

DriftSeverity = Literal["stable", "warning", "alert"]


@dataclass
class FeatureDrift:
    """Resultado del análisis de drift de una feature."""

    feature: str
    feature_type: Literal["numeric", "categorical"]
    psi: float
    severity: DriftSeverity
    test_name: str
    test_statistic: float
    p_value: float
    reference_mean: float | None = None
    current_mean: float | None = None

    def as_row(self) -> dict[str, object]:
        """Fila plana para tablas y reportes."""
        return {
            "feature": self.feature,
            "tipo": self.feature_type,
            "psi": round(self.psi, 4),
            "severidad": self.severity,
            "test": self.test_name,
            "estadistico": round(self.test_statistic, 4),
            "p_valor": round(self.p_value, 6),
            "media_referencia": (
                round(self.reference_mean, 4) if self.reference_mean is not None else None
            ),
            "media_actual": round(self.current_mean, 4) if self.current_mean is not None else None,
        }


@dataclass
class DriftReport:
    """Resultado agregado de un chequeo de drift."""

    n_reference: int
    n_current: int
    features: list[FeatureDrift] = field(default_factory=list)
    prediction_drift: FeatureDrift | None = None

    @property
    def alerts(self) -> list[FeatureDrift]:
        """Features con drift severo."""
        return [item for item in self.features if item.severity == "alert"]

    @property
    def warnings(self) -> list[FeatureDrift]:
        """Features con drift moderado."""
        return [item for item in self.features if item.severity == "warning"]

    @property
    def share_drifted(self) -> float:
        """Fracción de features con drift al menos moderado."""
        if not self.features:
            return 0.0
        return (len(self.alerts) + len(self.warnings)) / len(self.features)

    @property
    def needs_retraining(self) -> bool:
        """
        Si el drift observado justifica reentrenar.

        Criterio: cualquier feature en alerta, o más de un tercio de las
        features con drift moderado. Una sola feature moderada no mueve la
        aguja; un cambio generalizado sí.
        """
        return bool(self.alerts) or self.share_drifted > 0.33

    def summary(self) -> dict[str, object]:
        """Resumen serializable del reporte."""
        return {
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "n_features_checked": len(self.features),
            "n_alerts": len(self.alerts),
            "n_warnings": len(self.warnings),
            "share_drifted": round(self.share_drifted, 4),
            "needs_retraining": self.needs_retraining,
            "alert_features": [item.feature for item in self.alerts],
            "warning_features": [item.feature for item in self.warnings],
            "max_psi": round(max((item.psi for item in self.features), default=0.0), 4),
            "prediction_drift_psi": (
                round(self.prediction_drift.psi, 4) if self.prediction_drift else None
            ),
            "prediction_drift_severity": (
                self.prediction_drift.severity if self.prediction_drift else None
            ),
        }


def _classify(psi: float) -> DriftSeverity:
    """Traduce un PSI a severidad según los umbrales configurados."""
    config = load_config()
    if psi >= config.monitoring.psi_alert:
        return "alert"
    if psi >= config.monitoring.psi_warning:
        return "warning"
    return "stable"


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = DEFAULT_PSI_BINS,
) -> float:
    """
    Calcula el PSI entre dos distribuciones.

    PSI = sum((actual% - esperado%) * ln(actual% / esperado%)) sobre los bins.

    Para numéricas los bins son cuantiles de la referencia (robustos a
    distribuciones asimétricas); para categóricas, las categorías observadas.
    Se añade un epsilon para que un bin vacío no produzca log(0).

    Args:
        reference: distribución de referencia (datos de entrenamiento).
        current: distribución observada en producción.
        bins: número de bins para variables numéricas.

    Returns:
        PSI >= 0. Cuanto más alto, más cambió la distribución.
    """
    reference = reference.dropna()
    current = current.dropna()
    if reference.empty or current.empty:
        return 0.0

    if pd.api.types.is_numeric_dtype(reference):
        # Cuantiles de la referencia como bordes. `np.unique` colapsa bordes
        # repetidos, habitual en variables muy concentradas como `previous`.
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(reference.to_numpy(), quantiles))
        if len(edges) < 2:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf

        reference_counts, _ = np.histogram(reference.to_numpy(), bins=edges)
        current_counts, _ = np.histogram(current.to_numpy(), bins=edges)
    else:
        categories = sorted(set(reference.astype(str)) | set(current.astype(str)))
        reference_counts = np.array(
            [(reference.astype(str) == category).sum() for category in categories]
        )
        current_counts = np.array(
            [(current.astype(str) == category).sum() for category in categories]
        )

    reference_share = np.clip(reference_counts / reference_counts.sum(), PSI_EPSILON, None)
    current_share = np.clip(current_counts / current_counts.sum(), PSI_EPSILON, None)

    return float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )


def _numeric_drift(name: str, reference: pd.Series, current: pd.Series) -> FeatureDrift:
    """Drift de una feature numérica: PSI + test KS."""
    psi = population_stability_index(reference, current)
    statistic, p_value = stats.ks_2samp(reference.dropna(), current.dropna())
    return FeatureDrift(
        feature=name,
        feature_type="numeric",
        psi=psi,
        severity=_classify(psi),
        test_name="kolmogorov-smirnov",
        test_statistic=float(statistic),
        p_value=float(p_value),
        reference_mean=float(reference.mean()),
        current_mean=float(current.mean()),
    )


def _categorical_drift(name: str, reference: pd.Series, current: pd.Series) -> FeatureDrift:
    """Drift de una feature categórica: PSI + chi-cuadrado."""
    psi = population_stability_index(reference, current)
    categories = sorted(set(reference.astype(str)) | set(current.astype(str)))
    table = np.array(
        [
            [(reference.astype(str) == category).sum() for category in categories],
            [(current.astype(str) == category).sum() for category in categories],
        ]
    )
    # Descarta categorías ausentes en ambos grupos: chi2 no acepta columnas de ceros.
    table = table[:, table.sum(axis=0) > 0]

    if table.shape[1] < 2:
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value, _, _ = stats.chi2_contingency(table)

    return FeatureDrift(
        feature=name,
        feature_type="categorical",
        psi=psi,
        severity=_classify(psi),
        test_name="chi-cuadrado",
        test_statistic=float(statistic),
        p_value=float(p_value),
    )


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reference_scores: np.ndarray | None = None,
    current_scores: np.ndarray | None = None,
) -> DriftReport:
    """
    Compara los datos actuales contra la referencia de entrenamiento.

    Args:
        reference: snapshot de referencia (`data/reference/reference.parquet`).
        current: lote de datos de producción.
        reference_scores: scores del modelo sobre la referencia, opcional.
        current_scores: scores del modelo sobre el lote actual, opcional.
            Con ambos se añade el drift de predicciones al reporte.

    Returns:
        `DriftReport` con el detalle por feature y el veredicto agregado.
    """
    config = load_config()
    results: list[FeatureDrift] = []

    for name in config.features.numeric:
        if name in reference.columns and name in current.columns:
            results.append(_numeric_drift(name, reference[name], current[name]))

    for name in config.features.categorical:
        if name in reference.columns and name in current.columns:
            results.append(_categorical_drift(name, reference[name], current[name]))

    prediction_drift = None
    if reference_scores is not None and current_scores is not None:
        prediction_drift = _numeric_drift(
            "prediction_score",
            pd.Series(reference_scores),
            pd.Series(current_scores),
        )

    results.sort(key=lambda item: item.psi, reverse=True)
    report = DriftReport(
        n_reference=len(reference),
        n_current=len(current),
        features=results,
        prediction_drift=prediction_drift,
    )

    logger.info(
        "Chequeo de drift: %d features | %d alertas | %d advertencias | PSI máx=%.4f",
        len(results),
        len(report.alerts),
        len(report.warnings),
        max((item.psi for item in results), default=0.0),
    )
    for item in report.alerts:
        logger.warning("DRIFT SEVERO en '%s': PSI=%.4f", item.feature, item.psi)

    return report


def simulate_drifted_batch(
    reference: pd.DataFrame,
    n_rows: int = 2000,
    drift_strength: float = 0.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Genera un lote sintético de producción, con drift controlado.

    Sirve para **demostrar que el monitoreo funciona**: sin datos nuevos reales
    no hay forma de verificar que el detector dispara cuando debe. El drift
    simulado imita un escenario plausible: un cambio de régimen
    macroeconómico (sube el euríbor, cae el empleo) más un público objetivo más
    joven por un cambio en la estrategia de la campaña.

    Args:
        reference: datos de referencia a remuestrear.
        n_rows: tamaño del lote.
        drift_strength: 0.0 = sin drift (solo remuestreo); 1.0 = cambio severo.
        seed: semilla. None = la de configuración.

    Returns:
        DataFrame con las mismas columnas que la referencia (sin el target).
    """
    config = load_config()
    rng = np.random.default_rng(config.project.seed if seed is None else seed)
    target = config.data.target

    source = reference.drop(columns=[target]) if target in reference.columns else reference
    batch = source.sample(
        n=n_rows, replace=True, random_state=rng.integers(0, 2**31)
    ).reset_index(drop=True)

    if drift_strength <= 0:
        return batch

    # Régimen macroeconómico: sube el costo del dinero, cae el empleo.
    if "euribor3m" in batch.columns:
        batch["euribor3m"] = (batch["euribor3m"] + 2.5 * drift_strength).clip(0, 8)
    if "nr.employed" in batch.columns:
        batch["nr.employed"] = batch["nr.employed"] - 120 * drift_strength
    if "emp.var.rate" in batch.columns:
        batch["emp.var.rate"] = batch["emp.var.rate"] - 1.8 * drift_strength
    if "cons.conf.idx" in batch.columns:
        batch["cons.conf.idx"] = batch["cons.conf.idx"] - 8 * drift_strength

    # Público más joven.
    if "age" in batch.columns:
        shift = rng.normal(loc=-8 * drift_strength, scale=3, size=n_rows)
        batch["age"] = (batch["age"] + shift).clip(18, 95).round().astype(int)

    # Más contactos por cliente: el call center insiste más.
    if "campaign" in batch.columns:
        extra = rng.poisson(lam=2 * drift_strength, size=n_rows)
        batch["campaign"] = (batch["campaign"] + extra).clip(1, 60)

    # Se migra del teléfono fijo al celular.
    if "contact" in batch.columns:
        flip = rng.random(n_rows) < 0.45 * drift_strength
        batch.loc[flip, "contact"] = "cellular"

    logger.info("Lote simulado: %d filas con drift_strength=%.2f", n_rows, drift_strength)
    return batch
