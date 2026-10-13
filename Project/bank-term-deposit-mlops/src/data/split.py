"""
Partición train / validation / test y ETL mínimo.

Decisiones:

* Se descarta `duration` (data leakage: solo se conoce al terminar la llamada).
* El target `yes`/`no` se codifica a 1/0.
* La partición es estratificada porque solo 11% de los clientes suscriben.
* Semilla fija en `configs/config.yaml` para que cualquiera reproduzca los
  mismos splits byte a byte.
* El split de train se guarda además como `data/reference/` — es la línea base
  contra la que el módulo de monitoreo mide drift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.data.download import load_raw_dataset
from src.data.validate import validate_dataframe

logger = logging.getLogger(__name__)


@dataclass
class DataSplits:
    """Los tres splits con features y target ya separados."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_valid: pd.DataFrame
    y_valid: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series

    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.x_train),
            "valid": len(self.x_valid),
            "test": len(self.x_test),
        }

    def positive_rates(self) -> dict[str, float]:
        return {
            "train": float(self.y_train.mean()),
            "valid": float(self.y_valid.mean()),
            "test": float(self.y_test.mean()),
        }


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el ETL determinista: descarta columnas con leakage y codifica el target.

    Args:
        frame: dataset crudo.

    Returns:
        DataFrame con las features del contrato más el target como 0/1.
    """
    config = load_config()
    target = config.data.target
    result = frame.copy()

    to_drop = [column for column in config.data.drop_columns if column in result.columns]
    if to_drop:
        logger.info("Descartando columnas por data leakage: %s", to_drop)
        result = result.drop(columns=to_drop)

    # `is_numeric_dtype` en vez de `== object`: según la versión de pandas y el
    # backend de lectura, las columnas de texto llegan como `object`, `str` o
    # `string[pyarrow]`. Lo único estable es preguntar si ya es numérica.
    if target in result.columns and not pd.api.types.is_numeric_dtype(result[target]):
        positive = config.data.positive_label
        result[target] = (result[target].astype(str) == positive).astype(int)

    keep = [column for column in config.feature_columns if column in result.columns]
    if target in result.columns:
        keep = keep + [target]
    return result[keep]


def build_splits(frame: pd.DataFrame | None = None) -> DataSplits:
    """
    Construye los splits train/valid/test estratificados y reproducibles.

    Args:
        frame: dataset crudo. Si es None se carga (y descarga si falta).

    Returns:
        `DataSplits`.
    """
    config = load_config()
    raw = load_raw_dataset() if frame is None else frame

    validation = validate_dataframe(raw, require_target=True)
    validation.log()
    validation.raise_if_invalid()

    clean = clean_dataset(raw)
    target = config.data.target
    features = clean.drop(columns=[target])
    labels = clean[target]
    seed = config.project.seed

    # Primer corte: separa test del resto.
    x_rest, x_test, y_rest, y_test = train_test_split(
        features,
        labels,
        test_size=config.data.test_size,
        stratify=labels,
        random_state=seed,
    )

    # Segundo corte: valid sale de lo que queda, manteniendo la proporción
    # deseada respecto al dataset completo.
    valid_share_of_rest = config.data.valid_size / (1 - config.data.test_size)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_rest,
        y_rest,
        test_size=valid_share_of_rest,
        stratify=y_rest,
        random_state=seed,
    )

    splits = DataSplits(
        x_train=x_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        x_valid=x_valid.reset_index(drop=True),
        y_valid=y_valid.reset_index(drop=True),
        x_test=x_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )
    logger.info("Splits: %s", splits.sizes())
    logger.info(
        "Tasa de positivos: %s",
        {name: f"{rate:.4f}" for name, rate in splits.positive_rates().items()},
    )
    return splits


def save_splits(splits: DataSplits) -> dict[str, Path]:
    """
    Persiste los splits en Parquet y escribe la referencia de monitoreo.

    Args:
        splits: resultado de `build_splits`.

    Returns:
        Mapa nombre -> ruta de los archivos escritos.
    """
    config = load_config()
    processed = config.path("data_processed")
    reference = config.path("data_reference")
    target = config.data.target
    written: dict[str, Path] = {}

    pairs = {
        "train": (splits.x_train, splits.y_train),
        "valid": (splits.x_valid, splits.y_valid),
        "test": (splits.x_test, splits.y_test),
    }
    for name, (features, labels) in pairs.items():
        combined = features.copy()
        combined[target] = labels.to_numpy()
        destination = processed / f"{name}.parquet"
        combined.to_parquet(destination, index=False)
        written[name] = destination

    # Snapshot de referencia para drift: la distribución sobre la que el modelo
    # fue entrenado.
    reference_frame = splits.x_train.copy()
    reference_frame[target] = splits.y_train.to_numpy()
    reference_path = reference / "reference.parquet"
    reference_frame.to_parquet(reference_path, index=False)
    written["reference"] = reference_path

    logger.info("Splits guardados en %s", processed)
    return written


def load_splits() -> DataSplits:
    """Lee los splits ya persistidos en `data/processed/`."""
    config = load_config()
    processed = config.path("data_processed")
    target = config.data.target

    frames = {}
    for name in ("train", "valid", "test"):
        path = processed / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"No existe {path}. Ejecuta primero el flow de entrenamiento "
                "(`uv run python -m src.pipelines.training_flow`)."
            )
        frames[name] = pd.read_parquet(path)

    return DataSplits(
        x_train=frames["train"].drop(columns=[target]),
        y_train=frames["train"][target],
        x_valid=frames["valid"].drop(columns=[target]),
        y_valid=frames["valid"][target],
        x_test=frames["test"].drop(columns=[target]),
        y_test=frames["test"][target],
    )


def load_reference_data() -> pd.DataFrame:
    """Lee el snapshot de referencia usado por el monitoreo de drift."""
    config = load_config()
    path = config.path("data_reference") / "reference.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero el flow de entrenamiento para generarlo."
        )
    return pd.read_parquet(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    data_splits = build_splits()
    for name, path in save_splits(data_splits).items():
        print(f"{name:10s} -> {path}")
