"""
Validación de datos.

Se ejecuta antes de entrenar y antes de cada batch de scoring: si los datos no
cumplen el contrato, el pipeline falla temprano y con un mensaje claro en vez de
producir un modelo o unas predicciones silenciosamente malas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)

# Rangos plausibles de las variables numéricas. Fuera de rango no es un error
# fatal (puede ser un cliente real atípico) pero sí una advertencia a registrar.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "age": (17, 100),
    "campaign": (0, 60),
    "pdays": (0, 999),
    "previous": (0, 10),
    "emp.var.rate": (-4, 2),
    "cons.price.idx": (90, 96),
    "cons.conf.idx": (-55, -25),
    "euribor3m": (0, 6),
    "nr.employed": (4900, 5300),
}


@dataclass
class ValidationResult:
    """Resultado de validar un DataFrame contra el contrato de datos."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0

    def raise_if_invalid(self) -> None:
        """Lanza ValueError si la validación falló."""
        if not self.is_valid:
            raise ValueError("Validación de datos fallida:\n  - " + "\n  - ".join(self.errors))

    def log(self) -> None:
        """Escribe el resultado al logger."""
        for warning in self.warnings:
            logger.warning("Validación: %s", warning)
        if self.is_valid:
            logger.info("Validación OK: %d filas, %d advertencias", self.n_rows, len(self.warnings))
        else:
            for error in self.errors:
                logger.error("Validación: %s", error)

    def summary(self) -> dict[str, object]:
        """Resumen serializable (para artifacts de Prefect / MLflow)."""
        return {
            "is_valid": self.is_valid,
            "n_rows": self.n_rows,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_dataframe(frame: pd.DataFrame, require_target: bool = True) -> ValidationResult:
    """
    Valida un DataFrame contra el contrato de datos del proyecto.

    Comprueba, en orden: que haya filas, que estén las columnas de features,
    que el target exista y sea binario (si se exige), que los tipos sean los
    esperados, y que los valores numéricos caigan en rangos plausibles.

    Args:
        frame: datos a validar.
        require_target: si la columna target debe estar presente. False para
            datos de inferencia, donde el target es justamente lo desconocido.

    Returns:
        `ValidationResult` con errores (bloqueantes) y advertencias.
    """
    config = load_config()
    errors: list[str] = []
    warnings: list[str] = []

    if frame.empty:
        return ValidationResult(is_valid=False, errors=["El DataFrame está vacío"], n_rows=0)

    # --- Columnas requeridas -------------------------------------------------
    missing_features = [column for column in config.feature_columns if column not in frame.columns]
    if missing_features:
        errors.append(f"Faltan columnas de features: {missing_features}")

    target = config.data.target
    if require_target:
        if target not in frame.columns:
            errors.append(f"Falta la columna target '{target}'")
        else:
            observed = set(frame[target].dropna().unique())
            expected = {"yes", "no"}
            if not observed.issubset(expected) and not observed.issubset({0, 1}):
                errors.append(f"Target '{target}' con valores inesperados: {sorted(observed)}")
            if frame[target].nunique() < 2:
                errors.append(f"Target '{target}' tiene una sola clase: no se puede entrenar")

    # --- Nulos ---------------------------------------------------------------
    # El dataset codifica lo desconocido como la categoría 'unknown', no como
    # NaN. Un NaN real indica un problema en el upstream.
    for column in config.feature_columns:
        if column not in frame.columns:
            continue
        null_count = int(frame[column].isna().sum())
        if null_count:
            null_share = null_count / len(frame)
            message = f"'{column}' tiene {null_count} nulos ({null_share:.1%})"
            if null_share > 0.5:
                errors.append(message + " — más de la mitad de las filas")
            else:
                warnings.append(message)

    # --- Tipos ---------------------------------------------------------------
    for column in config.features.numeric:
        if column in frame.columns and not pd.api.types.is_numeric_dtype(frame[column]):
            errors.append(f"'{column}' debería ser numérica pero es {frame[column].dtype}")

    # --- Rangos --------------------------------------------------------------
    for column, (low, high) in PLAUSIBLE_RANGES.items():
        if column not in frame.columns or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        out_of_range = int(((frame[column] < low) | (frame[column] > high)).sum())
        if out_of_range:
            warnings.append(
                f"'{column}': {out_of_range} valores fuera del rango plausible [{low}, {high}]"
            )

    # --- Duplicados ----------------------------------------------------------
    duplicate_count = int(frame.duplicated().sum())
    if duplicate_count:
        warnings.append(f"{duplicate_count} filas duplicadas")

    return ValidationResult(
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
        n_rows=len(frame),
    )
