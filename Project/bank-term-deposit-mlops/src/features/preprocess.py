"""
Feature engineering y preprocesamiento.

Todo vive dentro de un `Pipeline` de scikit-learn. Esto importa: el objeto que
se registra en MLflow incluye las transformaciones, así que la API y el batch
scoring reciben el DataFrame crudo y no pueden desalinearse del entrenamiento
(el clásico training/serving skew).

Features derivadas, todas calculables en el momento de la predicción:

* ``was_contacted_before``: `pdays == 999` significa "nunca contactado". Es un
  centinela, no un número de días; como número arruina cualquier escalado.
* ``days_since_contact``: los días reales, con 0 cuando nunca hubo contacto.
* ``campaign_intensity``: contactos de esta campaña vs. históricos.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import load_config

logger = logging.getLogger(__name__)

PDAYS_NEVER_CONTACTED = 999

DERIVED_NUMERIC_FEATURES = [
    "was_contacted_before",
    "days_since_contact",
    "campaign_intensity",
]


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Agrega las features derivadas de dominio y neutraliza el centinela de `pdays`.

    Es un transformer de sklearn para que viaje dentro del pipeline serializado
    y se aplique igual en entrenamiento, en la API y en batch.
    """

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803 - convención sklearn
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        result = X.copy()

        if "pdays" in result.columns:
            never = result["pdays"] == PDAYS_NEVER_CONTACTED
            result["was_contacted_before"] = (~never).astype(int)
            result["days_since_contact"] = result["pdays"].where(~never, 0)
            # `pdays` cruda deja de aportar una vez descompuesta; mantenerla
            # con el 999 metería un outlier artificial en el escalado.
            result = result.drop(columns=["pdays"])
        else:
            result["was_contacted_before"] = 0
            result["days_since_contact"] = 0

        if {"campaign", "previous"}.issubset(result.columns):
            result["campaign_intensity"] = result["campaign"] / (1 + result["previous"])
        else:
            result["campaign_intensity"] = 0.0

        return result

    def get_feature_names_out(self, input_features=None):
        base = list(input_features or self.feature_names_in_)
        base = [name for name in base if name != "pdays"]
        return np.array(base + DERIVED_NUMERIC_FEATURES)


def build_preprocessor() -> ColumnTransformer:
    """
    Construye el `ColumnTransformer` que alimenta al clasificador.

    Numéricas: imputación por mediana + estandarización.
    Categóricas: imputación por constante + one-hot (ignorando categorías
    nuevas, para que un `job` desconocido en producción no rompa el servicio).
    """
    config = load_config()

    # Tras `DomainFeatureEngineer`, `pdays` ya no existe y aparecen las derivadas.
    numeric_features = [
        column for column in config.features.numeric if column != "pdays"
    ] + DERIVED_NUMERIC_FEATURES
    categorical_features = list(config.features.categorical)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_pipeline() -> Pipeline:
    """Feature engineering + preprocesamiento, sin el estimador final."""
    return Pipeline(
        steps=[
            ("domain_features", DomainFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
        ]
    )


def build_model_pipeline(estimator) -> Pipeline:
    """
    Pipeline completo: features -> preprocesamiento -> estimador.

    Args:
        estimator: cualquier clasificador compatible con sklearn.

    Returns:
        `Pipeline` entrenable que acepta el DataFrame crudo como entrada.
    """
    return Pipeline(
        steps=[
            ("domain_features", DomainFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def get_output_feature_names(pipeline: Pipeline) -> list[str]:
    """Nombres de las columnas que ve el estimador, útil para explicabilidad."""
    try:
        return list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    except Exception as error:  # pragma: no cover - depende de la versión de sklearn
        logger.warning("No se pudieron obtener los nombres de features: %s", error)
        return []
