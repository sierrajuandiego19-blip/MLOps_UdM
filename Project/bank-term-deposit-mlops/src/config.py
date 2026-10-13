"""
Carga de configuración del proyecto.

Un único punto de verdad: `configs/config.yaml`. Se expone como objeto con
acceso por atributo para evitar diccionarios anidados por todo el código.

Las variables de entorno con prefijo ``BTD_`` sobreescriben valores anidados
usando ``__`` como separador, por ejemplo::

    BTD_MLFLOW__TRACKING_URI=http://localhost:5000
    BTD_TRAINING__OPTUNA_TRIALS=5

Esto permite que Docker, CI y el deployment en cloud cambien configuración sin
tocar archivos.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ENV_PREFIX = "BTD_"
ENV_NESTED_SEPARATOR = "__"


def project_root() -> Path:
    """Raíz del proyecto (carpeta que contiene `configs/`)."""
    return Path(__file__).resolve().parent.parent


class ConfigSection:
    """Vista de sólo lectura de una sección del YAML, con acceso por atributo."""

    def __init__(self, data: dict[str, Any], prefix: str = "") -> None:
        self._data = data
        self._prefix = prefix

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._data:
            raise AttributeError(
                f"La clave '{self._prefix}{name}' no existe en configs/config.yaml"
            )
        value = self._data[name]
        if isinstance(value, dict):
            return ConfigSection(value, prefix=f"{self._prefix}{name}.")
        return value

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self.__getattr__(key)
        except AttributeError:
            return default

    def as_dict(self) -> dict[str, Any]:
        """Copia del contenido como diccionario plano de Python."""
        import copy

        return copy.deepcopy(self._data)

    def keys(self):
        return self._data.keys()

    def __repr__(self) -> str:
        return f"ConfigSection({sorted(self._data)})"


class Config(ConfigSection):
    """Configuración raíz, con helpers de rutas absolutas."""

    def __init__(self, data: dict[str, Any], root: Path) -> None:
        super().__init__(data)
        self.root = root

    def path(self, key: str) -> Path:
        """Ruta absoluta de `paths.<key>`, creando el directorio si falta."""
        relative = self._data["paths"][key]
        resolved = self.root / relative
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    @property
    def feature_columns(self) -> list[str]:
        """Columnas de entrada del modelo, en orden estable (numéricas + categóricas)."""
        return list(self.features.numeric) + list(self.features.categorical)

    @property
    def mlflow_tracking_uri(self) -> str:
        """
        Tracking URI de MLflow, con las rutas locales resueltas a absoluto.

        `sqlite:///mlflow.db` es relativo al directorio de trabajo, así que
        ejecutar desde `notebooks/` o desde un worker de Prefect apuntaría a una
        base de datos distinta —y vacía—. Se ancla a la raíz del proyecto para
        que todos los procesos vean los mismos experimentos.

        Las URIs remotas (`http://`, `https://`, `databricks`) se devuelven tal
        cual: ahí no hay ninguna ruta que resolver.
        """
        uri = str(self._data["mlflow"]["tracking_uri"])

        if uri.startswith("sqlite:///"):
            path = uri.removeprefix("sqlite:///")
            if not Path(path).is_absolute():
                return f"sqlite:///{(self.root / path).as_posix()}"
        elif uri.startswith("file:"):
            path = uri.removeprefix("file:").lstrip("/")
            if not Path(path).is_absolute():
                return f"file:///{(self.root / path).as_posix()}"

        return uri


def _coerce(raw: str) -> Any:
    """Convierte un string de entorno al tipo que sugiere su contenido."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Aplica overrides desde variables de entorno con prefijo BTD_."""
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(ENV_PREFIX):
            continue
        path = env_key[len(ENV_PREFIX) :].lower().split(ENV_NESTED_SEPARATOR)
        cursor: dict[str, Any] = data
        for part in path[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[path[-1]] = _coerce(env_value)
    return data


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> Config:
    """
    Carga (y memoiza) la configuración del proyecto.

    Args:
        config_path: ruta alternativa al YAML. Si es None usa
            `configs/config.yaml` relativo a la raíz del proyecto, o la ruta de
            la variable de entorno ``BTD_CONFIG_FILE``.

    Returns:
        Objeto `Config` listo para usar.
    """
    root = project_root()
    resolved = Path(config_path or os.getenv("BTD_CONFIG_FILE", root / "configs" / "config.yaml"))
    if not resolved.is_absolute():
        resolved = root / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {resolved}")

    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    data = _apply_env_overrides(data)
    return Config(data, root=root)
