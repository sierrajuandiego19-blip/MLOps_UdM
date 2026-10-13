"""
Model Registry de MLflow: registro, promoción y carga.

Se usan **alias** (`champion`, `challenger`) en vez de los stages legacy, que
están deprecados en MLflow 3. La ventaja práctica: la API y el batch scoring
piden `models:/<nombre>@champion` y nunca necesitan saber qué número de versión
está en producción hoy.

La promoción no es automática por defecto. Un candidato pasa a `champion` solo
si supera al campeón actual en la métrica primaria por un margen mínimo
(`min_improvement`), para no promover ruido estadístico.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.config import load_config

logger = logging.getLogger(__name__)

CHALLENGER_ALIAS = "challenger"
MODEL_CARD_FILENAME = "model_card.md"
CHAMPION_METADATA_FILENAME = "champion_metadata.json"
LOCAL_METADATA_FILENAME = "metadata.json"


@dataclass
class RegisteredChampion:
    """Datos del modelo en producción, para servirlos por la API."""

    model: Any
    name: str
    version: str
    alias: str
    run_id: str
    threshold: float
    metrics: dict[str, float]

    def __post_init__(self) -> None:
        """
        Normaliza la versión a string.

        Según el backend del registry, MLflow devuelve `version` como int o
        como str. El contrato de la API la declara string, así que se fuerza
        aquí, en el único punto por el que pasan todos los modelos cargados.
        """
        self.version = str(self.version)
        self.threshold = float(self.threshold)

    def info(self) -> dict[str, Any]:
        """Resumen serializable, sin el objeto del modelo."""
        return {
            "name": self.name,
            "version": self.version,
            "alias": self.alias,
            "run_id": self.run_id,
            "decision_threshold": self.threshold,
            "metrics": self.metrics,
        }


def get_client() -> MlflowClient:
    """Cliente de MLflow apuntando al tracking URI configurado."""
    config = load_config()
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    return MlflowClient(tracking_uri=config.mlflow_tracking_uri)


def register_model(
    model_uri: str,
    tags: dict[str, str] | None = None,
) -> str:
    """
    Registra un logged model como nueva versión en el registry.

    Args:
        model_uri: URI del modelo a registrar. En MLflow 3 es el `models:/m-...`
            que devuelve `log_model`; `runs:/<run_id>/model` ya no resuelve
            porque el modelo dejó de guardarse como artifact del run.
        tags: tags a adjuntar a la versión (métricas clave, dataset, etc.).

    Returns:
        El número de versión creado, como string.
    """
    config = load_config()
    client = get_client()
    model_name = config.mlflow.registered_model_name

    try:
        client.create_registered_model(
            model_name,
            description=(
                "Clasificador de propensión a suscribir un depósito a plazo fijo. "
                "Entrada: datos del cliente y contexto de campaña. "
                "Salida: probabilidad de suscripción."
            ),
        )
        logger.info("Modelo registrado creado: '%s'", model_name)
    except MlflowException:
        logger.debug("El modelo registrado '%s' ya existe", model_name)

    version = mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.info("Registrada la versión %s de '%s'", version.version, model_name)

    for key, value in (tags or {}).items():
        client.set_model_version_tag(model_name, version.version, key, str(value))

    # Siempre string: según el backend, MLflow devuelve la versión como int o
    # como str, y el contrato de la API la declara string.
    return str(version.version)


def set_alias(version: str, alias: str) -> None:
    """Apunta un alias a una versión concreta del modelo."""
    config = load_config()
    client = get_client()
    client.set_registered_model_alias(config.mlflow.registered_model_name, alias, version)
    logger.info("Alias '%s' -> versión %s", alias, version)


def get_champion_version() -> str | None:
    """Versión actualmente marcada como `champion`, o None si no hay ninguna."""
    config = load_config()
    client = get_client()
    try:
        version = client.get_model_version_by_alias(
            config.mlflow.registered_model_name, config.mlflow.champion_alias
        )
        return str(version.version)
    except MlflowException:
        return None


def _metric_from_version(version: str, metric_key: str) -> float | None:
    """Lee una métrica del run que originó una versión del modelo."""
    config = load_config()
    client = get_client()
    try:
        model_version = client.get_model_version(config.mlflow.registered_model_name, version)
        run = client.get_run(model_version.run_id)
        return run.data.metrics.get(metric_key)
    except MlflowException as error:
        logger.warning("No se pudo leer la métrica de la versión %s: %s", version, error)
        return None


def promote_if_better(
    candidate_version: str,
    metric_key: str | None = None,
    min_improvement: float = 0.005,
) -> dict[str, Any]:
    """
    Promueve el candidato a `champion` solo si mejora al campeón actual.

    Si no hay campeón, el candidato lo es por defecto (primer despliegue).
    Si no mejora lo suficiente, queda como `challenger`: sigue registrado y
    disponible para shadow testing, pero no sirve tráfico.

    Args:
        candidate_version: versión recién registrada.
        metric_key: métrica de comparación. None = `test_<primary_metric>`.
        min_improvement: mejora relativa mínima exigida (0.005 = 0.5%).

    Returns:
        Decisión tomada, con las métricas comparadas.
    """
    config = load_config()
    champion_alias = config.mlflow.champion_alias
    key = metric_key or f"test_{config.training.primary_metric}"

    candidate_metric = _metric_from_version(candidate_version, key)
    current_champion = get_champion_version()

    if current_champion is None:
        set_alias(candidate_version, champion_alias)
        logger.info("No había campeón: la versión %s lo es ahora", candidate_version)
        return {
            "promoted": True,
            "reason": "first_model",
            "champion_version": candidate_version,
            "previous_champion_version": None,
            "metric_key": key,
            "candidate_metric": candidate_metric,
            "champion_metric": None,
        }

    if current_champion == candidate_version:
        return {
            "promoted": True,
            "reason": "already_champion",
            "champion_version": candidate_version,
            "previous_champion_version": current_champion,
            "metric_key": key,
            "candidate_metric": candidate_metric,
            "champion_metric": candidate_metric,
        }

    champion_metric = _metric_from_version(current_champion, key)

    if candidate_metric is None or champion_metric is None:
        set_alias(candidate_version, CHALLENGER_ALIAS)
        return {
            "promoted": False,
            "reason": "missing_metrics",
            "champion_version": current_champion,
            "previous_champion_version": current_champion,
            "metric_key": key,
            "candidate_metric": candidate_metric,
            "champion_metric": champion_metric,
        }

    required = champion_metric * (1 + min_improvement)
    if candidate_metric >= required:
        set_alias(candidate_version, champion_alias)
        set_alias(current_champion, CHALLENGER_ALIAS)
        logger.info(
            "Promovida la versión %s a campeón (%s: %.4f > %.4f)",
            candidate_version,
            key,
            candidate_metric,
            champion_metric,
        )
        return {
            "promoted": True,
            "reason": "outperformed_champion",
            "champion_version": candidate_version,
            "previous_champion_version": current_champion,
            "metric_key": key,
            "candidate_metric": candidate_metric,
            "champion_metric": champion_metric,
        }

    set_alias(candidate_version, CHALLENGER_ALIAS)
    logger.info(
        "La versión %s NO supera al campeón (%s: %.4f < %.4f requerido). Queda como challenger.",
        candidate_version,
        key,
        candidate_metric,
        required,
    )
    return {
        "promoted": False,
        "reason": "did_not_beat_champion",
        "champion_version": current_champion,
        "previous_champion_version": current_champion,
        "metric_key": key,
        "candidate_metric": candidate_metric,
        "champion_metric": champion_metric,
    }


def export_champion(destination: Path | None = None) -> Path:
    """
    Guarda una copia autocontenida del campeón en disco.

    Sirve para servir el modelo donde no hay tracking server: un contenedor en
    cloud, por ejemplo. Escribe el modelo en formato MLflow más un
    `metadata.json` con la versión y el umbral, de modo que la copia local lleva
    el mismo contrato que la del registry.

    Args:
        destination: directorio destino. None = `serving.local_model_dir`.

    Returns:
        Ruta del directorio escrito.
    """
    import shutil

    config = load_config()
    target = Path(destination) if destination else config.root / config.serving.local_model_dir

    champion = load_champion(allow_local_fallback=False)

    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    mlflow.sklearn.save_model(
        champion.model,
        path=str(target),
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )
    (target / LOCAL_METADATA_FILENAME).write_text(
        json.dumps(
            {
                "name": champion.name,
                "version": champion.version,
                "alias": champion.alias,
                "run_id": champion.run_id,
                "threshold": champion.threshold,
                "metrics": champion.metrics,
                "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Campeón exportado a %s (versión %s)", target, champion.version)
    return target


def load_local_champion(source: Path | None = None) -> RegisteredChampion:
    """
    Carga el campeón desde la copia autocontenida en disco.

    Args:
        source: directorio a leer. None = `serving.local_model_dir`.

    Returns:
        `RegisteredChampion` equivalente al del registry.

    Raises:
        FileNotFoundError: si no existe la copia exportada.
    """
    config = load_config()
    directory = Path(source) if source else config.root / config.serving.local_model_dir
    metadata_path = directory / LOCAL_METADATA_FILENAME

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No hay un modelo exportado en {directory}. "
            "Genéralo con `uv run python scripts/export_model.py`."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = mlflow.sklearn.load_model(str(directory))
    logger.info(
        "Modelo cargado del directorio local %s (versión %s)", directory, metadata["version"]
    )

    return RegisteredChampion(
        model=model,
        name=metadata["name"],
        version=metadata["version"],
        alias=metadata.get("alias", config.mlflow.champion_alias),
        run_id=metadata.get("run_id", "unknown"),
        threshold=metadata["threshold"],
        metrics=metadata.get("metrics", {}),
    )


def load_champion(allow_local_fallback: bool = True) -> RegisteredChampion:
    """
    Carga el modelo campeón junto con su umbral de decisión y sus métricas.

    El umbral se lee del run que generó el modelo: forma parte del contrato del
    modelo tanto como sus pesos, y hardcodearlo en la API lo desincronizaría del
    modelo en el primer reentrenamiento.

    Args:
        allow_local_fallback: si al fallar el registry se intenta la copia
            exportada en `serving.local_model_dir`. Es lo que permite servir el
            modelo en un contenedor sin tracking server.

    Returns:
        `RegisteredChampion` listo para predecir.

    Raises:
        RuntimeError: si no hay campeón ni en el registry ni en disco.
    """
    config = load_config()
    model_name = config.mlflow.registered_model_name
    alias = config.mlflow.champion_alias

    # En un deployment sin MLflow, consultar el registry solo añade latencia de
    # arranque y un error que hay que atrapar.
    if allow_local_fallback and config.serving.prefer_local_model:
        return load_local_champion()

    client = get_client()
    try:
        model_version = client.get_model_version_by_alias(model_name, alias)
    except MlflowException as error:
        if allow_local_fallback:
            try:
                return load_local_champion()
            except FileNotFoundError:
                pass
        raise RuntimeError(
            f"No hay un modelo con alias '{alias}' en '{model_name}', ni una copia "
            f"exportada en '{config.serving.local_model_dir}'. "
            "Ejecuta el flow de entrenamiento: "
            "`uv run python -m src.pipelines.training_flow`"
        ) from error

    model_uri = f"models:/{model_name}@{alias}"
    logger.info("Cargando modelo campeón desde %s", model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    run = client.get_run(model_version.run_id)
    metrics = run.data.metrics
    threshold = metrics.get("optimal_threshold", config.training.decision_threshold)

    return RegisteredChampion(
        model=model,
        name=model_name,
        version=str(model_version.version),
        alias=alias,
        run_id=model_version.run_id,
        threshold=float(threshold),
        metrics={
            key: round(float(value), 6)
            for key, value in metrics.items()
            if key.startswith(("test_", "valid_", "cv_"))
        },
    )


def write_champion_metadata(summary: dict[str, Any]) -> Path:
    """
    Escribe el metadata del campeón a `models/champion_metadata.json`.

    Es la red de seguridad del deployment: permite al servicio arrancar y
    reportar qué versión espera incluso si el tracking server no responde.
    """
    config = load_config()
    destination = config.path("models") / CHAMPION_METADATA_FILENAME
    destination.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Metadata del campeón escrito en %s", destination)
    return destination


def write_model_card(summary: dict[str, Any], promotion: dict[str, Any]) -> Path:
    """
    Genera la model card en `models/model_card.md`.

    Documenta qué hace el modelo, con qué datos se entrenó, cómo rinde y cuáles
    son sus limitaciones conocidas: lo mínimo para que alguien que no lo
    entrenó pueda decidir si confiar en él.
    """
    config = load_config()
    test_metrics = summary.get("test_metrics", {})
    capacity = int(config.business.call_capacity_fraction * 100)

    def metric(key: str, fmt: str = ".4f") -> str:
        value = test_metrics.get(f"test_{key}")
        return format(value, fmt) if isinstance(value, (int, float)) else "n/d"

    content = f"""# Model Card — {config.mlflow.registered_model_name}

## Resumen

| Campo | Valor |
|---|---|
| Versión registrada | {promotion.get("champion_version", "n/d")} |
| Alias | `{config.mlflow.champion_alias}` |
| Familia del modelo | {summary.get("name", "n/d")} |
| Run de MLflow | `{summary.get("run_id", "n/d")}` |
| Umbral de decisión | {summary.get("optimal_threshold", config.training.decision_threshold):.2f} |
| Métrica de selección | `{config.training.primary_metric}` (PR-AUC) |

## Uso previsto

Priorizar a qué clientes llama el call center en una campaña de depósitos a
plazo fijo. La salida es una **probabilidad de suscripción**, no una decisión:
el umbral es un parámetro de negocio ajustable según la capacidad disponible.

**Fuera de alcance:** decidir condiciones del producto, precios o
elegibilidad crediticia. El modelo no fue validado para ninguno de esos usos.

## Datos de entrenamiento

- **Fuente:** UCI Bank Marketing (Moro, Cortez & Rita, 2014), variante
  `bank-additional-full` — campañas telefónicas de un banco portugués,
  mayo 2008 a noviembre 2010.
- **Volumen:** 41.188 contactos, 20 variables de entrada.
- **Clase positiva:** 11,3% (suscribió).
- **Partición:** 60/20/20 train/valid/test, estratificada, semilla
  {config.project.seed}.

### Variable excluida deliberadamente

`duration` (duración de la llamada) se **descarta**. Solo se conoce al terminar
la llamada, así que usarla produce un modelo con métricas excelentes e
inservible en producción, cuando hay que decidir *antes* de llamar. Lo advierte
la propia documentación del dataset.

## Desempeño (conjunto de test, nunca usado para ajustar)

| Métrica | Valor |
|---|---|
| PR-AUC (primaria) | {metric("pr_auc")} |
| ROC-AUC | {metric("roc_auc")} |
| Precision @ umbral | {metric("precision")} |
| Recall @ umbral | {metric("recall")} |
| F1 | {metric("f1")} |
| Brier score | {metric("brier_score")} |
| Lift en el top {capacity}% | {metric(f"top{capacity}_lift", ".2f")}x |
| Recall en el top {capacity}% | {metric(f"top{capacity}_recall", ".3f")} |

## Impacto de negocio estimado

Con los supuestos de `configs/config.yaml`
(${config.business.cost_per_call_usd:.0f} por llamada,
${config.business.profit_per_subscription_usd:.0f} por suscripción):

| Métrica | Valor |
|---|---|
| Utilidad del modelo | ${metric("profit_usd", ",.0f")} |
| Utilidad llamando a todos | ${metric("profit_call_everyone_usd", ",.0f")} |
| **Ganancia incremental** | **${metric("profit_uplift_usd", ",.0f")}** |
| Costo por adquisición | ${metric("cost_per_acquisition_usd", ",.2f")} |

## Limitaciones conocidas

1. **Deriva macroeconómica.** Cinco de las features son indicadores
   macroeconómicos (`euribor3m`, `nr.employed`, `emp.var.rate`, …) y el periodo
   de entrenamiento incluye la crisis de 2008. En otro régimen económico el
   modelo degrada rápido; de ahí el monitoreo de drift sobre esas variables en
   particular.
2. **Sesgo de un solo banco.** Los datos provienen de una institución
   portuguesa. Trasladarlo a otro mercado exige revalidación.
3. **Variables sensibles.** `age`, `marital` y `education` son predictivas y a
   la vez atributos protegidos o correlacionados con ellos. Antes de producción
   real hace falta una auditoría de equidad por subgrupo; este proyecto no la
   incluye.
4. **Probabilidades sin calibrar.** Se reporta el Brier score, pero no se aplicó
   calibración isotónica ni Platt. Los scores sirven para **ordenar**
   clientes, no como estimaciones absolutas de riesgo.
5. **Sin ground truth inmediato.** El resultado real de una llamada se conoce
   con retraso, así que el monitoreo de desempeño va por detrás del de drift.

## Reentrenamiento

Disparadores definidos en `configs/config.yaml`:

- PSI > {config.monitoring.psi_alert} en cualquier feature, o
- caída relativa de PR-AUC > {config.monitoring.performance_drop_alert:.0%} vs. la referencia, o
- cadencia programada (el deployment de Prefect corre semanalmente).

La promoción no es automática: un modelo nuevo debe superar al campeón en
`{config.training.primary_metric}` sobre test para tomar el alias
`{config.mlflow.champion_alias}` (ver `src/models/registry.py`).

---
*Generada automáticamente por el pipeline de entrenamiento.*
"""
    destination = config.path("models") / MODEL_CARD_FILENAME
    destination.write_text(content, encoding="utf-8")
    logger.info("Model card escrita en %s", destination)
    return destination
