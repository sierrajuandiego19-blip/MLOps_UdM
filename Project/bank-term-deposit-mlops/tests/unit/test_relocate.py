"""
Tests de la relocalización del historial de MLflow.

Este código es load-bearing para compartir el proyecto: si falla, quien clone el
repositorio ve los runs en MLflow pero **ningún modelo carga**, y el error es
silencioso. De ahí que se prueben los dos sentidos de la conversión entre
sistemas operativos, que es lo que ocurre en CI (se versiona desde Windows y se
ejecuta en Ubuntu).
"""

from __future__ import annotations

import sqlite3

import pytest
import yaml

from src.models.relocate import (
    detect_old_root,
    normalize_file_uris,
    path_variants,
    relocate_database,
    relocate_mlmodel_files,
    replace_root,
)

WINDOWS_ROOT = "C:/Users/ana/proyecto"
POSIX_ROOT = "/home/runner/work/proyecto"


def _make_database(path, artifact_location: str) -> None:
    """Crea un `mlflow.db` mínimo con las tablas que toca la relocalización."""
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE experiments (experiment_id INTEGER, name TEXT, artifact_location TEXT)"
        )
        connection.execute("CREATE TABLE runs (run_uuid TEXT, artifact_uri TEXT)")
        connection.execute("CREATE TABLE metrics (key TEXT, value REAL)")
        connection.execute(
            "INSERT INTO experiments VALUES (1, 'exp', ?)",
            (artifact_location,),
        )
        connection.execute(
            "INSERT INTO runs VALUES ('r1', ?)",
            (artifact_location.replace("/mlruns/1", "/mlruns/1/r1/artifacts"),),
        )
        connection.execute("INSERT INTO metrics VALUES ('pr_auc', 0.49)")
        connection.commit()


# --- Normalización de URIs -------------------------------------------------


def test_normalize_collapses_four_slashes():
    """
    `file:////` no es un URI válido y MLflow no lo resuelve.

    Aparece al reemplazar una raíz de Windows (`C:/...`, sin barra inicial) por
    una POSIX (`/home/...`, con barra inicial) dentro de un `file:///`.
    """
    assert normalize_file_uris("file:////home/x/mlruns") == "file:///home/x/mlruns"


def test_normalize_collapses_many_slashes():
    """Varias barras de más también se colapsan."""
    assert normalize_file_uris("file://////home/x") == "file:///home/x"


def test_normalize_leaves_valid_uri_untouched():
    """Un URI correcto no se toca."""
    for uri in ("file:///home/x/mlruns", "file:///C:/Users/ana/mlruns"):
        assert normalize_file_uris(uri) == uri


# --- Reemplazo de la raíz --------------------------------------------------


def test_replace_windows_to_posix_produces_valid_uri():
    """
    Windows -> Linux. Es el caso de CI, y el que tenía el bug.

    Un reemplazo ingenuo da `file:////home/...` con cuatro barras.
    """
    original = f"file:///{WINDOWS_ROOT}/mlruns/1/models/m-abc/artifacts"
    result = replace_root(original, WINDOWS_ROOT, POSIX_ROOT)

    assert result == f"file://{POSIX_ROOT}/mlruns/1/models/m-abc/artifacts"
    assert "file:////" not in result
    assert result.startswith("file:///")


def test_replace_posix_to_windows_produces_valid_uri():
    """Linux -> Windows, el sentido contrario."""
    original = f"file://{POSIX_ROOT}/mlruns/1/models/m-abc/artifacts"
    result = replace_root(original, POSIX_ROOT, WINDOWS_ROOT)

    assert result == f"file:///{WINDOWS_ROOT}/mlruns/1/models/m-abc/artifacts"
    assert "file:////" not in result


def test_replace_windows_to_windows():
    """El caso simple: misma plataforma, otra carpeta."""
    original = f"file:///{WINDOWS_ROOT}/mlruns/1"
    result = replace_root(original, WINDOWS_ROOT, "D:/dev/proyecto")
    assert result == "file:///D:/dev/proyecto/mlruns/1"


def test_replace_handles_backslashes():
    """Si la ruta quedó escrita con barras invertidas, también se sustituye."""
    original = r"C:\Users\ana\proyecto\mlruns\1"
    result = replace_root(original, WINDOWS_ROOT, "D:/dev/proyecto")
    assert "ana" not in result
    assert "dev" in result


def test_replace_is_idempotent():
    """Aplicarlo dos veces da el mismo resultado."""
    original = f"file:///{WINDOWS_ROOT}/mlruns/1"
    once = replace_root(original, WINDOWS_ROOT, POSIX_ROOT)
    twice = replace_root(once, WINDOWS_ROOT, POSIX_ROOT)
    assert once == twice


def test_replace_leaves_unrelated_paths_alone():
    """Una ruta que no contiene la raíz antigua no se modifica."""
    unrelated = "file:///otra/ruta/mlruns/1"
    assert replace_root(unrelated, WINDOWS_ROOT, POSIX_ROOT) == unrelated


def test_path_variants_puts_posix_form_first():
    """
    La forma con barras hacia adelante se busca primero.

    Es la que MLflow escribe; la variante con barras invertidas es solo un
    respaldo para rutas que algo haya guardado en formato nativo de Windows.
    """
    variants = path_variants(WINDOWS_ROOT, POSIX_ROOT)
    assert variants[0] == (WINDOWS_ROOT, POSIX_ROOT)
    assert any("\\" in old for old, _ in variants[1:])


# --- Detección de la raíz antigua ------------------------------------------


def test_detect_windows_root(tmp_path):
    """Se deduce la raíz de Windows desde `artifact_location`."""
    database = tmp_path / "mlflow.db"
    _make_database(database, f"file:///{WINDOWS_ROOT}/mlruns/1")

    with sqlite3.connect(database) as connection:
        assert detect_old_root(connection) == WINDOWS_ROOT


def test_detect_posix_root_keeps_leading_slash(tmp_path):
    """
    Una raíz POSIX tiene que conservar su `/` inicial.

    Al quitar el prefijo `file:///` se pierde, y sin reponerlo el reemplazo no
    encaja con lo que hay en la base de datos.
    """
    database = tmp_path / "mlflow.db"
    _make_database(database, f"file://{POSIX_ROOT}/mlruns/1")

    with sqlite3.connect(database) as connection:
        assert detect_old_root(connection) == POSIX_ROOT


def test_detect_returns_none_without_mlruns(tmp_path):
    """Sin el marcador `/mlruns/` no se puede deducir nada."""
    database = tmp_path / "mlflow.db"
    _make_database(database, "s3://bucket/artefactos")

    with sqlite3.connect(database) as connection:
        assert detect_old_root(connection) is None


# --- Relocalización de la base de datos ------------------------------------


def test_relocate_database_updates_all_columns(tmp_path):
    """Se actualizan todas las columnas de texto, no solo una lista fija."""
    database = tmp_path / "mlflow.db"
    _make_database(database, f"file:///{WINDOWS_ROOT}/mlruns/1")

    assert relocate_database(database, WINDOWS_ROOT, POSIX_ROOT) >= 2

    with sqlite3.connect(database) as connection:
        location = connection.execute("SELECT artifact_location FROM experiments").fetchone()[0]
        run_uri = connection.execute("SELECT artifact_uri FROM runs").fetchone()[0]

    assert location == f"file://{POSIX_ROOT}/mlruns/1"
    assert run_uri == f"file://{POSIX_ROOT}/mlruns/1/r1/artifacts"
    assert "file:////" not in location
    assert "ana" not in run_uri


def test_relocate_database_leaves_no_trace_of_old_root(tmp_path):
    """Después de relocalizar no debe quedar ninguna referencia a la raíz vieja."""
    database = tmp_path / "mlflow.db"
    _make_database(database, f"file:///{WINDOWS_ROOT}/mlruns/1")
    relocate_database(database, WINDOWS_ROOT, POSIX_ROOT)

    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT COUNT(*) FROM experiments WHERE artifact_location LIKE ?",
            (f"%{WINDOWS_ROOT}%",),
        ).fetchone()[0]
    assert rows == 0


def test_relocate_database_preserves_other_data(tmp_path):
    """Las columnas que no contienen rutas no se tocan."""
    database = tmp_path / "mlflow.db"
    _make_database(database, f"file:///{WINDOWS_ROOT}/mlruns/1")
    relocate_database(database, WINDOWS_ROOT, POSIX_ROOT)

    with sqlite3.connect(database) as connection:
        value = connection.execute("SELECT value FROM metrics WHERE key='pr_auc'").fetchone()[0]
    assert value == pytest.approx(0.49)


# --- Relocalización de los MLmodel -----------------------------------------


def test_relocate_mlmodel_handles_folded_yaml(tmp_path):
    """
    El caso que el reemplazo de texto plano no detectaba.

    Cuando la ruta es larga, PyYAML parte el valor en varias líneas. Un
    `str.replace` sobre el texto crudo no la encuentra y el archivo queda sin
    actualizar **sin dar error**.
    """
    mlruns = tmp_path / "mlruns" / "1" / "models" / "m-abc" / "artifacts"
    mlruns.mkdir(parents=True)
    long_root = "C:/Users/ana/OneDrive - Una Universidad Con Nombre Largo/Documents/MLOPS 2026"
    mlmodel = mlruns / "MLmodel"
    mlmodel.write_text(
        yaml.safe_dump(
            {
                "artifact_path": f"file:///{long_root}/mlruns/1/models/m-abc/artifacts",
                "flavors": {"sklearn": {"pickled_model": "model.pkl"}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    # Se confirma que PyYAML efectivamente partió el valor en varias líneas:
    # la ruta completa ya no aparece de forma contigua en el texto crudo, que
    # es exactamente lo que hace fallar un `str.replace`.
    raw = mlmodel.read_text(encoding="utf-8")
    assert long_root not in raw, "el valor no quedó plegado; el test no prueba nada"
    assert len(raw.splitlines()) > 2

    modified = relocate_mlmodel_files(tmp_path / "mlruns", long_root, POSIX_ROOT)

    assert modified == 1
    document = yaml.safe_load(mlmodel.read_text(encoding="utf-8"))
    assert document["artifact_path"] == f"file://{POSIX_ROOT}/mlruns/1/models/m-abc/artifacts"
    assert document["flavors"]["sklearn"]["pickled_model"] == "model.pkl"


def test_relocate_mlmodel_skips_unrelated_files(tmp_path):
    """Un MLmodel que no contiene la raíz antigua no se reescribe."""
    mlruns = tmp_path / "mlruns" / "1"
    mlruns.mkdir(parents=True)
    (mlruns / "MLmodel").write_text("artifact_path: file:///otra/ruta\n", encoding="utf-8")

    assert relocate_mlmodel_files(tmp_path / "mlruns", WINDOWS_ROOT, POSIX_ROOT) == 0


def test_relocate_mlmodel_tolerates_missing_directory(tmp_path):
    """Sin directorio `mlruns/` no hay nada que hacer, y no debe fallar."""
    assert relocate_mlmodel_files(tmp_path / "no-existe", WINDOWS_ROOT, POSIX_ROOT) == 0


def test_relocate_mlmodel_tolerates_malformed_yaml(tmp_path):
    """Un MLmodel corrupto se salta con advertencia, sin tumbar el proceso."""
    mlruns = tmp_path / "mlruns" / "1"
    mlruns.mkdir(parents=True)
    (mlruns / "MLmodel").write_text("{no es: yaml: valido:", encoding="utf-8")

    assert relocate_mlmodel_files(tmp_path / "mlruns", WINDOWS_ROOT, POSIX_ROOT) == 0
