"""
Relocalización del historial de MLflow entre máquinas.

MLflow guarda la ubicación de cada artefacto como **ruta absoluta**, tanto en
`mlflow.db` como en los archivos `MLmodel`. Este proyecto versiona ese
historial a propósito (ver `docs/compartir.md`), así que al clonarlo en otra
máquina —u otro sistema operativo— hay que reescribir esas rutas o los
artefactos no se encuentran.

La lógica vive aquí, y no en `scripts/`, para que tenga tests: es código del que
depende que el proyecto funcione al compartirlo.

## Los dos casos que hay que manejar bien

**Mismo sistema operativo.** `C:/Users/ana/proyecto` -> `C:/dev/proyecto`.
Reemplazo directo.

**Entre sistemas operativos**, que es lo que pasa en CI (se versiona desde
Windows y se ejecuta en Ubuntu). Aquí aparece una trampa: la raíz de Windows no
empieza por `/` y la de POSIX sí, así que un reemplazo ingenuo sobre
`file:///C:/Users/ana/...` produce `file:////home/runner/...` — con **cuatro**
barras, que no es un URI válido. Por eso cada reemplazo se normaliza después.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

FILE_URI_PREFIXES = ("file:///", "file://", "file:")
MLRUNS_MARKER = "/mlruns/"


def normalize_file_uris(text: str) -> str:
    """
    Normaliza todo `file:` a exactamente tres barras.

    MLflow escribe `file:///C:/Users/...` en Windows y `file:///home/...` en
    POSIX: en los dos casos son **tres** barras, porque la barra inicial de la
    ruta POSIX es la tercera. Un reemplazo de raíces rompe esa invariante en los
    dos sentidos:

    * Windows -> POSIX: `file:///` + `/home/...` = `file:////home/...` (cuatro)
    * POSIX -> Windows: `file://` + `C:/Users/...` = `file://C:/Users/...` (dos)

    Ninguna de las dos la resuelve MLflow, y el fallo es silencioso.

    Args:
        text: cadena que puede contener URIs `file:`.

    Returns:
        La cadena con los prefijos normalizados a `file:///`.
    """
    return re.sub(r"file:/{2,}", "file:///", text)


def as_posix_root(path: Path) -> str:
    """Raíz del proyecto en la forma que usa MLflow: barras hacia adelante."""
    return path.resolve().as_posix()


def detect_old_root(connection: sqlite3.Connection) -> str | None:
    """
    Deduce la raíz del proyecto con la que se escribió la base de datos.

    Se lee de `experiments.artifact_location`, que tiene la forma
    `file:///<raíz>/mlruns/<id>`.

    Args:
        connection: conexión abierta a `mlflow.db`.

    Returns:
        La raíz antigua en forma POSIX, o None si no se pudo determinar.
    """
    row = connection.execute(
        "SELECT artifact_location FROM experiments "
        "WHERE artifact_location LIKE '%/mlruns/%' LIMIT 1"
    ).fetchone()
    if not row or not row[0]:
        return None

    location = str(row[0]).replace("\\", "/")
    for prefix in FILE_URI_PREFIXES:
        if location.startswith(prefix):
            location = location[len(prefix) :]
            break

    if MLRUNS_MARKER not in location:
        return None

    old_root = location.split(MLRUNS_MARKER)[0]
    # Una raíz POSIX pierde su `/` inicial al quitar el prefijo `file:///`.
    # Sin reponerlo, el reemplazo no encaja con lo que hay en la base de datos.
    if old_root and not old_root.startswith("/") and ":" not in old_root.split("/")[0]:
        old_root = "/" + old_root
    return old_root


def path_variants(old_root: str, new_root: str) -> tuple[tuple[str, str], ...]:
    """
    Las formas en que una ruta puede estar escrita, de más específica a menos.

    Se cubren las barras hacia adelante (lo normal en MLflow) y las invertidas
    (si algo la escribió en formato nativo de Windows).

    Args:
        old_root: raíz antigua, en forma POSIX.
        new_root: raíz nueva, en forma POSIX.

    Returns:
        Pares (buscar, reemplazar).
    """
    variants = [(old_root, new_root)]
    windows_old = old_root.replace("/", "\\")
    if windows_old != old_root:
        variants.append((windows_old, new_root.replace("/", "\\")))
    return tuple(variants)


def replace_root(value: str, old_root: str, new_root: str) -> str:
    """
    Reemplaza la raíz antigua por la nueva en una cadena, y normaliza el URI.

    Args:
        value: cadena original.
        old_root: raíz antigua, en forma POSIX.
        new_root: raíz nueva, en forma POSIX.

    Returns:
        La cadena con la raíz sustituida.
    """
    result = value
    for old_variant, new_variant in path_variants(old_root, new_root):
        result = result.replace(old_variant, new_variant)
    return normalize_file_uris(result)


def _text_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    """Columnas de tipo texto de una tabla."""
    columns = []
    for row in connection.execute(f"PRAGMA table_info('{table}')"):
        name, declared_type = row[1], (row[2] or "").upper()
        if any(kind in declared_type for kind in ("CHAR", "TEXT", "CLOB")) or declared_type == "":
            columns.append(name)
    return columns


def relocate_database(database: Path, old_root: str, new_root: str) -> int:
    """
    Reescribe la ruta antigua por la nueva en todas las columnas de texto.

    Se recorren todas las tablas en vez de una lista fija: MLflow añade tablas
    entre versiones, y una ruta sin actualizar rompe las cargas en silencio.

    Args:
        database: ruta a `mlflow.db`.
        old_root: raíz antigua, en forma POSIX.
        new_root: raíz nueva, en forma POSIX.

    Returns:
        Número de filas actualizadas.
    """
    total = 0
    with sqlite3.connect(database) as connection:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]

        for table in tables:
            for column in _text_columns(connection, table):
                # La sustitución se hace en Python, no con el REPLACE de SQL,
                # para poder normalizar el URI después de cambiar la raíz.
                # SQLite no tiene expresiones regulares, y un REPLACE encadenado
                # no distingue los dos sentidos (ver `normalize_file_uris`).
                conditions = " OR ".join(
                    f'"{column}" LIKE ?' for _, _ in path_variants(old_root, new_root)
                )
                parameters = [f"%{old}%" for old, _ in path_variants(old_root, new_root)]
                try:
                    rows = connection.execute(
                        f'SELECT rowid, "{column}" FROM "{table}" WHERE {conditions}',
                        parameters,
                    ).fetchall()
                except sqlite3.OperationalError:
                    # Tabla sin rowid o columna no consultable: no es donde
                    # MLflow guarda rutas, así que se puede omitir.
                    continue

                updated_here = 0
                for rowid, value in rows:
                    if not isinstance(value, str):
                        continue
                    new_value = replace_root(value, old_root, new_root)
                    if new_value != value:
                        connection.execute(
                            f'UPDATE "{table}" SET "{column}" = ? WHERE rowid = ?',
                            (new_value, rowid),
                        )
                        updated_here += 1

                if updated_here:
                    logger.info("  %s.%s: %d fila(s) actualizada(s)", table, column, updated_here)
                    total += updated_here

        connection.commit()
    return total


def relocate_mlmodel_files(mlruns_dir: Path, old_root: str, new_root: str) -> int:
    """
    Reescribe el campo `artifact_path` de los archivos `MLmodel`.

    Hay que parsear el YAML en vez de hacer un reemplazo de texto: cuando la
    ruta es larga, PyYAML la escribe como **escalar plegado en varias líneas**,
    por ejemplo

        artifact_path: file:///C:/Users/x/OneDrive - The University/Documents/MLOPS
          2026/Project/...

    Un `str.replace` sobre el texto crudo no encuentra la ruta porque está
    partida, y el archivo queda sin actualizar en silencio.

    Args:
        mlruns_dir: directorio `mlruns/`.
        old_root: raíz antigua, en forma POSIX.
        new_root: raíz nueva, en forma POSIX.

    Returns:
        Número de archivos modificados.
    """
    if not mlruns_dir.exists():
        return 0

    modified = 0
    for mlmodel in sorted(mlruns_dir.rglob("MLmodel")):
        try:
            document = yaml.safe_load(mlmodel.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            logger.warning("  no se pudo leer %s: %s", mlmodel.name, error)
            continue

        if not isinstance(document, dict):
            continue

        original = document.get("artifact_path")
        if not isinstance(original, str):
            continue

        updated = replace_root(original, old_root, new_root)
        if updated == original:
            continue

        document["artifact_path"] = updated
        # `sort_keys=False` conserva el orden original: MLflow lee el archivo
        # con yaml.safe_load y el orden le da igual, pero un diff legible
        # ayuda a cualquiera que revise el cambio.
        mlmodel.write_text(
            yaml.safe_dump(document, sort_keys=False, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
        logger.info("  %s", mlmodel.relative_to(mlruns_dir.parent))
        modified += 1

    return modified
