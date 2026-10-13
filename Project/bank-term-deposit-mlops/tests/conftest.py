"""
Fixtures compartidas por los tests.

Los datos sintéticos permiten que la suite corra en segundos y sin red: ningún
test unitario descarga el dataset ni toca MLflow. Los tests que sí necesitan un
modelo entrenado lo entrenan sobre estos datos sintéticos, no sobre los reales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config

JOBS = ["admin.", "blue-collar", "technician", "services", "retired", "unknown"]
MARITALS = ["married", "single", "divorced", "unknown"]
EDUCATIONS = ["basic.4y", "high.school", "university.degree", "professional.course", "unknown"]
YES_NO_UNKNOWN = ["no", "yes", "unknown"]
CONTACTS = ["cellular", "telephone"]
MONTHS = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DAYS = ["mon", "tue", "wed", "thu", "fri"]
POUTCOMES = ["failure", "nonexistent", "success"]


@pytest.fixture(scope="session")
def config():
    """Configuración del proyecto."""
    return load_config()


@pytest.fixture
def synthetic_raw(request) -> pd.DataFrame:
    """
    Dataset sintético con el mismo esquema que el real, incluido `duration`.

    La señal es deliberada: la probabilidad de suscribir sube con la edad y con
    un `poutcome` de éxito, y baja con el euríbor. Así un modelo entrenado sobre
    estos datos tiene algo que aprender y los tests de métricas son
    significativos.
    """
    n_rows = getattr(request, "param", 600)
    rng = np.random.default_rng(42)

    age = rng.integers(18, 85, size=n_rows)
    poutcome = rng.choice(POUTCOMES, size=n_rows, p=[0.1, 0.85, 0.05])
    euribor = rng.uniform(0.6, 5.0, size=n_rows)

    logit = (
        -2.5
        + 0.03 * (age - 40)
        + 1.6 * (poutcome == "success")
        - 0.45 * euribor
        + rng.normal(0, 0.5, size=n_rows)
    )
    probability = 1 / (1 + np.exp(-logit))
    target = np.where(rng.random(n_rows) < probability, "yes", "no")

    return pd.DataFrame(
        {
            "age": age,
            "job": rng.choice(JOBS, size=n_rows),
            "marital": rng.choice(MARITALS, size=n_rows),
            "education": rng.choice(EDUCATIONS, size=n_rows),
            "default": rng.choice(YES_NO_UNKNOWN, size=n_rows, p=[0.78, 0.02, 0.20]),
            "housing": rng.choice(YES_NO_UNKNOWN, size=n_rows, p=[0.45, 0.53, 0.02]),
            "loan": rng.choice(YES_NO_UNKNOWN, size=n_rows, p=[0.82, 0.16, 0.02]),
            "contact": rng.choice(CONTACTS, size=n_rows),
            "month": rng.choice(MONTHS, size=n_rows),
            "day_of_week": rng.choice(DAYS, size=n_rows),
            "duration": rng.integers(10, 900, size=n_rows),
            "campaign": rng.integers(1, 10, size=n_rows),
            # Mezcla deliberada del centinela 999 con días reales.
            "pdays": np.where(rng.random(n_rows) < 0.85, 999, rng.integers(1, 30, size=n_rows)),
            "previous": rng.integers(0, 4, size=n_rows),
            "poutcome": poutcome,
            "emp.var.rate": rng.uniform(-3.4, 1.4, size=n_rows),
            "cons.price.idx": rng.uniform(92.2, 94.8, size=n_rows),
            "cons.conf.idx": rng.uniform(-50.8, -26.9, size=n_rows),
            "euribor3m": euribor,
            "nr.employed": rng.uniform(4963.6, 5228.1, size=n_rows),
            "y": target,
        }
    )


@pytest.fixture
def synthetic_features(synthetic_raw, config) -> pd.DataFrame:
    """Solo las columnas de features del contrato, sin target ni `duration`."""
    return synthetic_raw[config.feature_columns].copy()


@pytest.fixture
def synthetic_labels(synthetic_raw, config) -> pd.Series:
    """Target codificado como 0/1."""
    return (synthetic_raw[config.data.target] == config.data.positive_label).astype(int)


@pytest.fixture
def valid_client_payload() -> dict[str, object]:
    """Payload válido para la API, con los nombres de campo del dataset."""
    return {
        "age": 41,
        "job": "technician",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "thu",
        "campaign": 2,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": -1.8,
        "cons.price.idx": 92.893,
        "cons.conf.idx": -46.2,
        "euribor3m": 1.313,
        "nr.employed": 5099.1,
    }
