"""
Reapunta el historial de MLflow a la ruta de ESTA máquina.

## Por qué hace falta

MLflow guarda la ubicación de cada artefacto como **ruta absoluta**, tanto en
`mlflow.db` como en los archivos `MLmodel`. Por ejemplo:

    file:///C:/Users/juan/.../bank-term-deposit-mlops/mlruns/1/models/m-abc/artifacts

Eso significa que si clonas el repositorio en otra máquina —u otro usuario, u
otra carpeta, u otro sistema operativo— MLflow abre bien la interfaz y muestra
los runs, pero **no encuentra un solo artefacto**: los modelos no se pueden
cargar y el registry queda roto. El fallo es silencioso hasta que alguien
intenta usar un modelo.

Este script reescribe todas esas rutas para que apunten a donde está el
proyecto ahora. Es **idempotente**: si ya están bien, no hace nada.

La lógica está en `src/models/relocate.py`, que tiene tests.

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.models.relocate import (  # noqa: E402
    as_posix_root,
    detect_old_root,
    relocate_database,
    relocate_mlmodel_files,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKUP_SUFFIX = ".bak"


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
    new_root = as_posix_root(PROJECT_ROOT)

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
