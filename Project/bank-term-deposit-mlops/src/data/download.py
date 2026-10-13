"""
Ingesta del dataset UCI Bank Marketing.

El archivo de UCI es un ZIP que contiene otro ZIP. Esta capa lo descarga una
sola vez, extrae el CSV que interesa y lo deja en `data/raw/` como Parquet para
que el resto del pipeline no tenga que repetir el trabajo ni depender de la red.
"""

from __future__ import annotations

import hashlib
import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.config import load_config

logger = logging.getLogger(__name__)

RAW_FILENAME = "bank_marketing_raw.parquet"
DOWNLOAD_TIMEOUT_SECONDS = 120


def _download_bytes(url: str) -> bytes:
    """Descarga la URL completa a memoria (el archivo pesa ~1 MB)."""
    logger.info("Descargando dataset desde %s", url)
    response = requests.get(url, timeout=DOWNLOAD_TIMEOUT_SECONDS)
    response.raise_for_status()
    logger.info("Descargados %.1f KB", len(response.content) / 1024)
    return response.content


def _extract_csv(archive: bytes, outer_member: str, inner_member: str) -> bytes:
    """Extrae el CSV objetivo del ZIP anidado de UCI."""
    with zipfile.ZipFile(io.BytesIO(archive)) as outer:
        if outer_member not in outer.namelist():
            raise KeyError(
                f"'{outer_member}' no está en el ZIP descargado. Contenido: {outer.namelist()}"
            )
        nested = outer.read(outer_member)

    with zipfile.ZipFile(io.BytesIO(nested)) as inner:
        if inner_member not in inner.namelist():
            raise KeyError(
                f"'{inner_member}' no está en {outer_member}. Contenido: {inner.namelist()}"
            )
        return inner.read(inner_member)


def download_raw_dataset(force: bool = False) -> Path:
    """
    Garantiza que el dataset crudo exista localmente en Parquet.

    Args:
        force: si es True vuelve a descargar aunque el archivo ya exista.

    Returns:
        Ruta al Parquet crudo.
    """
    config = load_config()
    destination = config.path("data_raw") / RAW_FILENAME

    if destination.exists() and not force:
        logger.info("Dataset crudo ya presente en %s (se omite la descarga)", destination)
        return destination

    archive = _download_bytes(config.data.source_url)
    csv_bytes = _extract_csv(
        archive,
        outer_member=config.data.outer_zip_member,
        inner_member=config.data.inner_csv_member,
    )

    checksum = hashlib.sha256(csv_bytes).hexdigest()
    frame = pd.read_csv(io.BytesIO(csv_bytes), sep=config.data.csv_separator)
    frame.to_parquet(destination, index=False)

    logger.info(
        "Dataset guardado en %s (%d filas, %d columnas, sha256=%s)",
        destination,
        len(frame),
        frame.shape[1],
        checksum[:12],
    )
    return destination


def load_raw_dataset(force_download: bool = False) -> pd.DataFrame:
    """Carga el dataset crudo, descargándolo si hace falta."""
    return pd.read_parquet(download_raw_dataset(force=force_download))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    frame = load_raw_dataset()
    print(f"Filas: {len(frame):,} | Columnas: {frame.shape[1]}")
    print(frame["y"].value_counts(normalize=True).round(4).to_string())
