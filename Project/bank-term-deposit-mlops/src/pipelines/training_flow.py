"""
Flow de entrenamiento orquestado con Prefect.

    ingesta -> validación -> split -> exploración -> tuning -> test
            -> registro -> promoción -> documentación

Cada paso es un `@task` con reintentos donde tiene sentido (la ingesta depende
de la red; el entrenamiento, no). El flow deja artifacts en la UI de Prefect
para que el resultado se lea sin abrir los logs.

Ejecutar:
    uv run python -m src.pipelines.training_flow
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact, create_table_artifact

from src.config import load_config
from src.data.download import load_raw_dataset
from src.data.split import DataSplits, build_splits, save_splits
from src.data.validate import validate_dataframe
from src.models.registry import (
    promote_if_better,
    register_model,
    write_champion_metadata,
    write_model_card,
)
from src.models.train import (
    TrainedCandidate,
    finalize_best_model,
    optimize_xgboost,
    run_exploration,
    setup_mlflow,
)

logging.getLogger("mlflow").setLevel(logging.WARNING)


@task(name="ingest-data", retries=3, retry_delay_seconds=10, log_prints=True)
def ingest_data_task(force_download: bool = False):
    """Descarga (o reutiliza) el dataset crudo. Reintenta: depende de la red."""
    logger = get_run_logger()
    frame = load_raw_dataset(force_download=force_download)
    logger.info("Dataset cargado: %d filas x %d columnas", len(frame), frame.shape[1])
    return frame


@task(name="validate-data", log_prints=True)
def validate_data_task(frame) -> dict[str, Any]:
    """Valida el dataset contra el contrato de datos y falla si no cumple."""
    logger = get_run_logger()
    result = validate_dataframe(frame, require_target=True)

    for warning in result.warnings:
        logger.warning("Validación: %s", warning)

    if not result.is_valid:
        for error in result.errors:
            logger.error("Validación: %s", error)
        result.raise_if_invalid()

    logger.info(
        "Validación superada: %d filas, %d advertencias", result.n_rows, len(result.warnings)
    )
    return result.summary()


@task(name="build-splits", log_prints=True)
def build_splits_task(frame) -> DataSplits:
    """Construye y persiste los splits estratificados, más la referencia de drift."""
    logger = get_run_logger()
    splits = build_splits(frame)
    paths = save_splits(splits)
    logger.info("Tamaños: %s", splits.sizes())
    logger.info("Referencia de monitoreo: %s", paths["reference"])
    return splits


@task(name="explore-models", log_prints=True)
def explore_models_task(splits: DataSplits) -> list[TrainedCandidate]:
    """Entrena baselines y modelos de árbol, todos trackeados en MLflow."""
    logger = get_run_logger()
    setup_mlflow()
    candidates = run_exploration(splits)
    logger.info(
        "Mejor candidato de la exploración: %s (%.4f)", candidates[0].name, candidates[0].score
    )
    return candidates


@task(name="tune-model", log_prints=True)
def tune_model_task(splits: DataSplits, n_trials: int | None = None) -> TrainedCandidate:
    """Optimiza hiperparámetros con Optuna, un run anidado por trial."""
    logger = get_run_logger()
    setup_mlflow()
    tuned = optimize_xgboost(splits, n_trials=n_trials)
    logger.info("Modelo optimizado: valid PR-AUC=%.4f", tuned.score)
    return tuned


@task(name="select-and-evaluate", log_prints=True)
def select_and_evaluate_task(
    candidates: list[TrainedCandidate], tuned: TrainedCandidate, splits: DataSplits
) -> dict[str, Any]:
    """
    Elige el ganador y lo evalúa en test.

    El modelo optimizado solo gana si de verdad supera al mejor de la fase
    exploratoria: 20 trials de Optuna no garantizan una mejora.
    """
    logger = get_run_logger()
    setup_mlflow()

    pool = [*candidates, tuned]
    winner = max(pool, key=lambda candidate: candidate.score)
    logger.info("Modelo seleccionado: '%s' (valid PR-AUC=%.4f)", winner.name, winner.score)

    summary = finalize_best_model(winner, splits)
    summary["candidates_evaluated"] = len(pool)
    summary["ranking"] = [
        {"name": candidate.name, "valid_pr_auc": round(candidate.score, 5)}
        for candidate in sorted(pool, key=lambda item: item.score, reverse=True)
    ]
    return summary


@task(name="register-and-promote", log_prints=True)
def register_and_promote_task(summary: dict[str, Any]) -> dict[str, Any]:
    """Registra el ganador y lo promueve a campeón solo si mejora al actual."""
    logger = get_run_logger()
    config = load_config()
    setup_mlflow()

    test_metrics = summary["test_metrics"]
    version = register_model(
        model_uri=summary["model_uri"],
        tags={
            "model_family": summary["name"],
            "test_pr_auc": round(test_metrics["test_pr_auc"], 5),
            "test_roc_auc": round(test_metrics["test_roc_auc"], 5),
            "decision_threshold": round(summary["optimal_threshold"], 3),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "dataset": "uci-bank-marketing",
        },
    )
    promotion = promote_if_better(
        candidate_version=version,
        metric_key=f"test_{config.training.primary_metric}",
    )
    promotion["registered_version"] = version

    logger.info(
        "Versión %s registrada | promovida=%s (%s)",
        version,
        promotion["promoted"],
        promotion["reason"],
    )
    return promotion


@task(name="write-documentation", log_prints=True)
def write_documentation_task(summary: dict[str, Any], promotion: dict[str, Any]) -> dict[str, str]:
    """Escribe la model card y el metadata del campeón."""
    logger = get_run_logger()
    card_path = write_model_card(summary, promotion)
    metadata_path = write_champion_metadata(
        {
            "model_name": load_config().mlflow.registered_model_name,
            "version": promotion.get("champion_version"),
            "registered_version": promotion.get("registered_version"),
            "model_family": summary["name"],
            "run_id": summary["run_id"],
            "model_uri": summary["model_uri"],
            "decision_threshold": summary["optimal_threshold"],
            "test_metrics": summary["test_metrics"],
            "promotion": promotion,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    logger.info("Documentación generada")
    return {"model_card": str(card_path), "champion_metadata": str(metadata_path)}


@task(name="publish-artifacts", log_prints=True)
def publish_artifacts_task(
    summary: dict[str, Any], promotion: dict[str, Any], validation: dict[str, Any]
) -> None:
    """Publica tabla y resumen en la UI de Prefect."""
    config = load_config()
    test_metrics = summary["test_metrics"]
    capacity = int(config.business.call_capacity_fraction * 100)

    create_table_artifact(
        key="model-ranking",
        table=[
            {"posición": position, "modelo": item["name"], "valid_pr_auc": item["valid_pr_auc"]}
            for position, item in enumerate(summary["ranking"], start=1)
        ],
        description="Candidatos ordenados por PR-AUC en validation",
    )

    create_markdown_artifact(
        key="training-summary",
        markdown=f"""# Resumen del entrenamiento

**Ejecutado:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Modelo seleccionado
- **Familia:** `{summary["name"]}`
- **Run de MLflow:** `{summary["run_id"]}`
- **Umbral de decisión calibrado:** {summary["optimal_threshold"]:.2f}
- **Candidatos evaluados:** {summary["candidates_evaluated"]}

## Desempeño en test
| Métrica | Valor |
|---|---|
| PR-AUC | {test_metrics["test_pr_auc"]:.4f} |
| ROC-AUC | {test_metrics["test_roc_auc"]:.4f} |
| Precision | {test_metrics["test_precision"]:.4f} |
| Recall | {test_metrics["test_recall"]:.4f} |
| F1 | {test_metrics["test_f1"]:.4f} |
| Lift top {capacity}% | {test_metrics[f"test_top{capacity}_lift"]:.2f}x |

## Impacto de negocio
| Concepto | Valor |
|---|---|
| Utilidad del modelo | ${test_metrics["test_profit_usd"]:,.0f} |
| Utilidad llamando a todos | ${test_metrics["test_profit_call_everyone_usd"]:,.0f} |
| **Ganancia incremental** | **${test_metrics["test_profit_uplift_usd"]:,.0f}** |

## Registry
- **Versión registrada:** {promotion.get("registered_version")}
- **Promovida a campeón:** {"sí" if promotion["promoted"] else "no"} ({promotion["reason"]})
- **Campeón vigente:** versión {promotion.get("champion_version")}

## Validación de datos
- Filas: {validation["n_rows"]:,}
- Errores: {validation["n_errors"]} | Advertencias: {validation["n_warnings"]}
""",
        description="Resumen del run de entrenamiento",
    )


@flow(
    name="training-pipeline",
    description="Pipeline end-to-end: ingesta, validación, experimentación, registro y promoción",
    log_prints=True,
)
def training_pipeline(
    force_download: bool = False,
    optuna_trials: int | None = None,
    skip_tuning: bool = False,
) -> dict[str, Any]:
    """
    Pipeline completo de entrenamiento.

    Args:
        force_download: vuelve a descargar el dataset crudo.
        optuna_trials: sobreescribe el número de trials de la configuración.
            Útil para un smoke test rápido en CI.
        skip_tuning: salta la búsqueda de hiperparámetros (solo exploración).

    Returns:
        Resumen del run: modelo elegido, métricas y resultado de la promoción.
    """
    logger = get_run_logger()
    config = load_config()
    logger.info("=" * 70)
    logger.info("PIPELINE DE ENTRENAMIENTO — %s", config.project.name)
    logger.info("=" * 70)

    raw = ingest_data_task(force_download=force_download)
    validation = validate_data_task(raw)
    splits = build_splits_task(raw)

    candidates = explore_models_task(splits)

    if skip_tuning:
        logger.info("Tuning omitido por configuración del flow")
        tuned = candidates[0]
    else:
        tuned = tune_model_task(splits, n_trials=optuna_trials)

    summary = select_and_evaluate_task(candidates, tuned, splits)
    promotion = register_and_promote_task(summary)
    documentation = write_documentation_task(summary, promotion)
    publish_artifacts_task(summary, promotion, validation)

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETADO")
    logger.info("  Modelo: %s", summary["name"])
    logger.info("  Test PR-AUC: %.4f", summary["test_metrics"]["test_pr_auc"])
    logger.info("  Versión registrada: %s", promotion.get("registered_version"))
    logger.info("  Campeón: versión %s", promotion.get("champion_version"))
    logger.info("=" * 70)

    return {
        "status": "success",
        "model": summary["name"],
        "run_id": summary["run_id"],
        "decision_threshold": summary["optimal_threshold"],
        "test_metrics": summary["test_metrics"],
        "promotion": promotion,
        "documentation": documentation,
        "validation": validation,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    result = training_pipeline()
    print("\n" + "=" * 70)
    print(f"Modelo:            {result['model']}")
    print(f"Test PR-AUC:       {result['test_metrics']['test_pr_auc']:.4f}")
    print(f"Test ROC-AUC:      {result['test_metrics']['test_roc_auc']:.4f}")
    print(f"Umbral:            {result['decision_threshold']:.2f}")
    print(f"Versión campeón:   {result['promotion'].get('champion_version')}")
    print("=" * 70)
