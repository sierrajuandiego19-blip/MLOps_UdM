"""
Servicio de predicción, compartido por la API y el batch scoring.

Un solo camino de código para ambos modos de deployment: así una predicción
online y una batch sobre el mismo cliente dan exactamente el mismo resultado.

El modelo se carga una vez y se cachea en memoria. `reload()` fuerza una
recarga, que es cómo se despliega un modelo nuevo sin reiniciar el servicio.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config
from src.models.registry import RegisteredChampion, load_champion

logger = logging.getLogger(__name__)

_CHAMPION: RegisteredChampion | None = None


class PredictionService:
    """Envuelve al modelo campeón y traduce sus scores a decisiones de negocio."""

    def __init__(self, champion: RegisteredChampion) -> None:
        self.champion = champion
        self.config = load_config()
        self.loaded_at = datetime.now(UTC)

    # -- Metadatos ---------------------------------------------------------

    @property
    def threshold(self) -> float:
        """Umbral de decisión calibrado durante el entrenamiento."""
        return self.champion.threshold

    def info(self) -> dict[str, Any]:
        """Información del modelo servido, para el endpoint `/model-info`."""
        return {
            **self.champion.info(),
            "loaded_at": self.loaded_at.isoformat(timespec="seconds"),
            "feature_columns": self.config.feature_columns,
        }

    # -- Predicción --------------------------------------------------------

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Alinea el DataFrame de entrada con el contrato de features.

        Reordena columnas y rellena las que falten con NaN, para que el
        imputador del pipeline las trate. El orden importa: un
        `ColumnTransformer` selecciona por nombre, pero XGBoost valida la forma.
        """
        expected = self.config.feature_columns
        result = frame.copy()
        for column in expected:
            if column not in result.columns:
                result[column] = np.nan
        return result[expected]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Probabilidad de suscripción para cada fila."""
        return self.champion.model.predict_proba(self._prepare(frame))[:, 1]

    def predict(self, frame: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
        """
        Puntúa un DataFrame y devuelve scores con su decisión y banda de riesgo.

        Args:
            frame: features crudas, una fila por cliente.
            threshold: umbral de decisión. None = el del modelo.

        Returns:
            DataFrame con `probability`, `will_subscribe`, `propensity_band` y
            `recommended_action`.
        """
        cutoff = self.threshold if threshold is None else threshold
        probabilities = self.predict_proba(frame)

        result = pd.DataFrame(
            {
                "probability": probabilities,
                "will_subscribe": (probabilities >= cutoff).astype(int),
                "threshold_used": cutoff,
            }
        )
        result["propensity_band"] = pd.cut(
            result["probability"],
            bins=[-0.001, 0.10, 0.30, 0.60, 1.001],
            labels=["muy_baja", "baja", "media", "alta"],
        ).astype(str)
        result["recommended_action"] = np.where(
            result["will_subscribe"] == 1,
            "llamar",
            "no_llamar",
        )
        return result

    def predict_one(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Puntúa un único cliente.

        Args:
            payload: features del cliente como diccionario.

        Returns:
            Diccionario con el score, la decisión y la versión del modelo.
        """
        scored = self.predict(pd.DataFrame([payload])).iloc[0]
        return {
            "probability": round(float(scored["probability"]), 6),
            "will_subscribe": bool(scored["will_subscribe"]),
            "propensity_band": str(scored["propensity_band"]),
            "recommended_action": str(scored["recommended_action"]),
            "threshold_used": round(float(scored["threshold_used"]), 4),
            "model_name": self.champion.name,
            "model_version": self.champion.version,
            "predicted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }


def get_prediction_service(force_reload: bool = False) -> PredictionService:
    """
    Devuelve el servicio de predicción, cargando el campeón la primera vez.

    Args:
        force_reload: vuelve a leer el modelo del registry. Es el mecanismo de
            despliegue en caliente del endpoint `/reload`.

    Returns:
        `PredictionService` listo para usar.
    """
    global _CHAMPION
    if _CHAMPION is None or force_reload:
        _CHAMPION = load_champion()
        logger.info(
            "Modelo campeón cargado: %s v%s (umbral=%.2f)",
            _CHAMPION.name,
            _CHAMPION.version,
            _CHAMPION.threshold,
        )
    return PredictionService(_CHAMPION)
