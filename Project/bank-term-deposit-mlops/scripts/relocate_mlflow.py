"""
Reapunta el historial de MLflow a la ruta de ESTA máquina.

## Por qué hace falta

MLflow guarda la ubicación de cada artefacto como **ruta absoluta**, tanto en
`mlflow.db` como en los archivos `MLmodel`. Por ejemplo:

    file:///C:/Users/juan/.../bank-term-deposit-mlops/mlruns/1/models/m-abc/artifacts

Eso significa que si clonas el repositorio en otra máquina —u otro usuario, u
otra carpeta— MLflow abre bien la interfaz y muestra los runs, pero **no
encuentra un solo artefacto**: los modelos no se pueden cargar y el registry
queda roto. Y el fallo es silencioso hasta que alguien intenta usar un modelo.

Este script reescribe todas esas rutas para que apunten a donde está el
proyecto ahora. Es **idempotente**: si ya están bien, no hace nada.

## Cuándo ejecutarlo

Una sola vez, justo después de clonar el repositorio:

    uv run python scripts/relocate_mlflow.py

Si no clonaste nada y entrenaste en tu propia máquina, no lo necesitas.

## Qué modifica

* `mlflow.db` — toda columna de texto que contenga la ruta antigua
  (`experiments.artifact_location`, `runs.artifact_uri`,
  `logged_models.artifact_location`, `model_versions.storage_location`, ...)
* `mlruns/**/MLmodel` — el campo `artifact_path`

Antes de tocar nada hace una copia de seguridad de `mlflow.db`.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKUP_SUFFIX = ".bak"


def _as_posix_root(path: Path) -> str:
    """Raíz del proyecto en la forma que usa MLflow: barras hacia adelante."""
    return path.resolve().as_posix()


def detect_old_root(connection: sqlite3.Connection) -> str | None:
    """
    Deduce la raíz del proyecto con la que se escribió la base de datos.

    Se lee de `experiments.artifact_location`, que tiene la forma
    `file:///<raíz>/mlruns/<id>`.

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
    for prefix in ("file:///", "file://", "file:"):
        if location.startswith(prefix):
            location = location[len(prefix) :]
            break

    marker = "/mlruns/"
    if marker not in location:
        return None
    return location.split(marker)[0]


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
                # Se cubren las dos formas en que la ruta puede estar escrita:
                # con barras hacia adelante (lo normal en MLflow) y con barras
                # invertidas (si algo la escribió en formato nativo de Windows).
                for old_variant, new_variant in (
                    (old_root, new_root),
                    (old_root.replace("/", "\\"), new_root.replace("/", "\\")),
                ):
                    cursor = connection.execute(
                        f'UPDATE "{table}" SET "{column}" = REPLACE("{column}", ?, ?) '
                        f'WHERE "{column}" LIKE ?',
                        (old_variant, new_variant, f"%{old_variant}%"),
                    )
                    if cursor.rowcount > 0:
                        logger.info(
                            "  %s.%s: %d fila(s) actualizada(s)", table, column, cursor.rowcount
                        )
                        total += cursor.rowcount
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

    variants = ((old_root, new_root), (old_root.replace("/", "\\"), new_root.replace("/", "\\")))
    modified = 0

    for mlmodel in mlruns_dir.rglob("MLmodel"):
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

        updated = original
        for old_variant, new_variant in variants:
            updated = updated.replace(old_variant, new_variant)

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Reapunta el historial de MLflow a esta máquina")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="no crear copia de seguridad de mlflow.db",
    )
    args = parser.parse_args()

    config = load_config()
    database = Path(config.mlflow_tracking_uri.removeprefix("sqlite:///"))
    mlruns_dir = PROJECT_ROOT / "mlruns"
    new_root = _as_posix_root(PROJECT_ROOT)

    if not database.exists():
        print(
            f"No existe {database}.\n"
            "No hay historial que reapuntar: entrena con\n"
            "  uv run python -m src.pipelines.training_flow",
            file=sys.stderr,
        )
        return 1

    with sqlite3.connect(database) as connection:
        old_root = detect_old_root(connection)

    if old_root is None:
        print(
            "No se pudo determinar la ruta original desde mlflow.db. "
            "Si el historial funciona, no hace falta hacer nada.",
            file=sys.stderr,
        )
        return 1

    print()
    print("=" * 70)
    print("REAPUNTAR EL HISTORIAL DE MLFLOW")
    print("=" * 70)
    print(f"  Ruta original:  {old_root}")
    print(f"  Ruta de ahora:  {new_root}")
    print("=" * 70)
    print()

    if old_root == new_root:
        print("Las rutas ya coinciden: no hay nada que cambiar.")
        print("(Este script es idempotente, se puede volver a ejecutar sin riesgo.)")
        return 0

    if not args.no_backup:
        backup = database.with_suffix(database.suffix + BACKUP_SUFFIX)
        shutil.copy2(database, backup)
        logger.info("Copia de seguridad: %s", backup.name)

    logger.info("Actualizando mlflow.db...")
    rows = relocate_database(database, old_root, new_root)

    logger.info("Actualizando archivos MLmodel...")
    files = relocate_mlmodel_files(mlruns_dir, old_root, new_root)

    print()
    print("=" * 70)
    print("LISTO")
    print("=" * 70)
    print(f"  Filas de la base de datos actualizadas: {rows}")
    print(f"  Archivos MLmodel actualizados:          {files}")
    print()
    print("Verifica que el campeon carga:")
    print('  uv run python -c "from src.models.registry import load_champion; \\')
    print('      c = load_champion(); print(c.name, c.version, c.threshold)"')
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
