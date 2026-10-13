"""
Flow de monitoreo.

Compara un lote de producción contra la referencia de entrenamiento, mide
drift de features y de scores, evalúa el desempeño si hay ground truth, y
decide si hace falta reentrenar.

El flow **detecta y recomienda**; no reentrena por su cuenta. Un reentrenamiento
automático disparado por un falso positivo de drift es peor que una alerta que
alguien revisa: el `training-pipeline` queda a un comando de distancia.

Ejecutar:
    uv run python -m src.pipelines.monitoring_flow
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact, create_table_artifact

from src.config import load_config
from src.data.split import load_reference_data, load_splits
from src.models.evaluate import classification_metrics
from src.models.predict import get_prediction_service
from src.monitoring.drift import DriftReport, detect_drift, simulate_drifted_batch
from src.monitoring.report import save_drift_report

logging.getLogger("mlflow").setLevel(logging.WARNING)


@task(name="load-monitoring-inputs", log_prints=True)
def load_inputs_task(
    current_file: str | None = None,
    n_rows: int = 3000,
    drift_strength: float = 0.6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga la referencia y el lote actual a comparar.

    Args:
        current_file: Parquet con datos de producción. None = simular un lote
            con drift, para poder demostrar que el detector funciona.
        n_rows: filas del lote simulado.
        drift_strength: intensidad del drift simulado.

    Returns:
        (referencia, lote actual).
    """
    logger = get_run_logger()
    reference = load_reference_data()

    if current_file:
        current = pd.read_parquet(current_file)
        logger.info("Lote de producción cargado desde %s: %d filas", current_file, len(current))
    else:
        current = simulate_drifted_batch(reference, n_rows=n_rows, drift_strength=drift_strength)
        logger.info(
            "Lote simulado para la demostración: %d filas, drift_strength=%.2f",
            len(current),
            drift_strength,
        )

    logger.info("Referencia: %d filas", len(reference))
    return reference, current


@task(name="check-data-drift", log_prints=True)
def check_drift_task(
    reference: pd.DataFrame, current: pd.DataFrame
) -> tuple[DriftReport, dict[str, Any]]:
    """
    Mide drift de features y de la distribución de scores del modelo.

    Returns:
        (reporte de drift, info del modelo usado para puntuar).
    """
    logger = get_run_logger()
    config = load_config()
    service = get_prediction_service()
    target = config.data.target

    reference_features = (
        reference.drop(columns=[target]) if target in reference.columns else reference
    )
    reference_scores = service.predict_proba(reference_features)
    current_scores = service.predict_proba(current)

    report = detect_drift(
        reference=reference_features,
        current=current,
        reference_scores=reference_scores,
        current_scores=current_scores,
    )

    logger.info(
        "Drift: %d alertas, %d advertencias de %d features",
        len(report.alerts),
        len(report.warnings),
        len(report.features),
    )
    if report.prediction_drift:
        logger.info(
            "Drift de predicciones: PSI=%.4f (%s) | score medio %.4f -> %.4f",
            report.prediction_drift.psi,
            report.prediction_drift.severity,
            report.prediction_drift.reference_mean,
            report.prediction_drift.current_mean,
        )

    model_info = {
        "model_name": service.champion.name,
        "model_version": service.champion.version,
        "decision_threshold": round(service.threshold, 4),
        "run_id": service.champion.run_id,
    }
    return report, model_info


@task(name="check-model-performance", log_prints=True)
def check_performance_task() -> dict[str, Any] | None:
    """
    Evalúa el modelo contra ground truth y compara con la referencia.

    Usa el split de test como sustituto de las etiquetas que llegarían con
    retraso en producción. Señala la degradación comparando contra el PR-AUC
    registrado en el entrenamiento.

    Returns:
        Métricas de desempeño con el veredicto, o None si no hay ground truth.
    """
    logger = get_run_logger()
    config = load_config()

    try:
        splits = load_splits()
    except FileNotFoundError:
        logger.warning("No hay datos con ground truth: se omite el chequeo de desempeño")
        return None

    service = get_prediction_service()
    scores = service.predict_proba(splits.x_test)
    current = classification_metrics(splits.y_test, scores, threshold=service.threshold)

    baseline = service.champion.metrics.get("test_pr_auc")
    result: dict[str, Any] = {
        "pr_auc": round(current["pr_auc"], 4),
        "roc_auc": round(current["roc_auc"], 4),
        "precision": round(current["precision"], 4),
        "recall": round(current["recall"], 4),
        "f1": round(current["f1"], 4),
        "n_rows_evaluated": len(splits.x_test),
    }

    if baseline:
        relative_drop = (baseline - current["pr_auc"]) / baseline
        result["baseline_pr_auc"] = round(float(baseline), 4)
        result["relative_drop"] = round(float(relative_drop), 4)
        result["degraded"] = bool(relative_drop > config.monitoring.performance_drop_alert)
        logger.info(
            "Desempeño: PR-AUC %.4f vs. referencia %.4f (caída %.1f%%)",
            current["pr_auc"],
            baseline,
            relative_drop * 100,
        )
    else:
        result["degraded"] = False
        logger.info(
            "Desempeño: PR-AUC %.4f (sin referencia con la que comparar)", current["pr_auc"]
        )

    return result


@task(name="generate-monitoring-report", log_prints=True)
def generate_report_task(
    report: DriftReport,
    performance: dict[str, Any] | None,
    model_info: dict[str, Any],
) -> dict[str, str]:
    """Escribe el reporte HTML y el JSON de métricas."""
    logger = get_run_logger()
    paths = save_drift_report(report, performance=performance, model_info=model_info)
    logger.info("Reporte HTML: %s", paths["html"])
    return {key: str(value) for key, value in paths.items()}


@task(name="decide-retraining", log_prints=True)
def decide_retraining_task(
    report: DriftReport, performance: dict[str, Any] | None
) -> dict[str, Any]:
    """
    Decide si recomendar un reentrenamiento, y por qué.

    Dos disparadores independientes: drift de datos y degradación de desempeño.
    Cualquiera de los dos basta.
    """
    logger = get_run_logger()
    reasons: list[str] = []

    if report.alerts:
        reasons.append(
            f"drift severo en {len(report.alerts)} feature(s): "
            + ", ".join(item.feature for item in report.alerts)
        )
    if report.share_drifted > 0.33:
        reasons.append(f"{report.share_drifted:.0%} de las features con drift")
    if report.prediction_drift and report.prediction_drift.severity == "alert":
        reasons.append(f"drift severo de scores (PSI={report.prediction_drift.psi:.3f})")
    if performance and performance.get("degraded"):
        reasons.append(f"caída de PR-AUC del {performance['relative_drop']:.1%}")

    decision = {
        "retrain_recommended": bool(reasons),
        "reasons": reasons,
        "drift_summary": report.summary(),
        "performance": performance,
        "next_action": (
            "Ejecutar `uv run python -m src.pipelines.training_flow` y revisar si el "
            "candidato supera al campeón"
            if reasons
            else "Sin acción: continuar con el monitoreo programado"
        ),
    }

    if reasons:
        logger.warning("REENTRENAMIENTO RECOMENDADO: %s", "; ".join(reasons))
    else:
        logger.info("Modelo estable: no se recomienda reentrenar")
    return decision


@task(name="publish-monitoring-artifacts", log_prints=True)
def publish_artifacts_task(
    report: DriftReport,
    decision: dict[str, Any],
    performance: dict[str, Any] | None,
    model_info: dict[str, Any],
    paths: dict[str, str],
) -> None:
    """Publica tabla de drift y resumen en la UI de Prefect."""
    create_table_artifact(
        key="feature-drift",
        table=[item.as_row() for item in report.features],
        description="Drift por feature, ordenado por PSI descendente",
    )

    performance_block = "No hubo ground truth disponible en esta ejecución."
    if performance:
        performance_block = "\n".join(
            [
                "| Métrica | Valor |",
                "|---|---|",
                f"| PR-AUC | {performance['pr_auc']:.4f} |",
                f"| PR-AUC de referencia | {performance.get('baseline_pr_auc', 'n/d')} |",
                f"| Caída relativa | {performance.get('relative_drop', 0):.2%} |",
                f"| Recall | {performance['recall']:.4f} |",
                f"| Precision | {performance['precision']:.4f} |",
                f"| Degradado | {'sí' if performance.get('degraded') else 'no'} |",
            ]
        )

    reasons_block = (
        "\n".join(f"- {reason}" for reason in decision["reasons"])
        if decision["reasons"]
        else "- Ninguno: todas las features dentro de los umbrales."
    )
    top_drift = "\n".join(
        f"| `{item.feature}` | {item.psi:.4f} | {item.severity} |" for item in report.features[:8]
    )

    create_markdown_artifact(
        key="monitoring-summary",
        markdown=f"""# Reporte de monitoreo

**Ejecutado:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Veredicto
**{"REENTRENAMIENTO RECOMENDADO" if decision["retrain_recommended"] else "MODELO ESTABLE"}**

### Motivos
{reasons_block}

**Siguiente acción:** {decision["next_action"]}

## Modelo monitoreado
- **Nombre:** `{model_info["model_name"]}`
- **Versión:** {model_info["model_version"]}
- **Umbral:** {model_info["decision_threshold"]}

## Drift de datos
| Métrica | Valor |
|---|---|
| Filas de referencia | {report.n_reference:,} |
| Filas actuales | {report.n_current:,} |
| Features evaluadas | {len(report.features)} |
| Alertas (PSI alto) | {len(report.alerts)} |
| Advertencias | {len(report.warnings)} |
| Fracción con drift | {report.share_drifted:.1%} |

### Mayores PSI
| Feature | PSI | Severidad |
|---|---|---|
{top_drift}

## Desempeño
{performance_block}

## Reporte completo
`{paths["latest_html"]}`
""",
        description="Resumen del chequeo de monitoreo",
    )


@flow(
    name="monitoring-pipeline",
    description="Mide drift de datos y de predicciones, evalúa desempeño y decide si reentrenar",
    log_prints=True,
)
def monitoring_pipeline(
    current_file: str | None = None,
    n_rows: int = 3000,
    drift_strength: float = 0.6,
) -> dict[str, Any]:
    """
    Pipeline de monitoreo.

    Args:
        current_file: Parquet con datos de producción. None = simular un lote
            con drift para demostrar la detección.
        n_rows: filas del lote simulado.
        drift_strength: intensidad del drift simulado (0 = ninguno).

    Returns:
        Veredicto del monitoreo con las rutas de los reportes.
    """
    logger = get_run_logger()
    logger.info("=" * 70)
    logger.info("PIPELINE DE MONITOREO")
    logger.info("=" * 70)

    reference, current = load_inputs_task(
        current_file=current_file, n_rows=n_rows, drift_strength=drift_strength
    )
    report, model_info = check_drift_task(reference, current)
    performance = check_performance_task()
    paths = generate_report_task(report, performance, model_info)
    decision = decide_retraining_task(report, performance)
    publish_artifacts_task(report, decision, performance, model_info, paths)

    logger.info("=" * 70)
    logger.info(
        "MONITOREO COMPLETADO — reentrenar: %s",
        "sí" if decision["retrain_recommended"] else "no",
    )
    logger.info("=" * 70)

    return {
        "status": "success",
        "decision": decision,
        "reports": paths,
        "model": model_info,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }


def _summarize(result: dict[str, Any]) -> None:
    decision = result["decision"]
    summary = decision["drift_summary"]
    print("\n" + "=" * 70)
    print(f"Reentrenar:          {'SÍ' if decision['retrain_recommended'] else 'no'}")
    print(f"Features con alerta: {summary['n_alerts']}")
    print(f"PSI máximo:          {summary['max_psi']}")
    print(f"Reporte HTML:        {result['reports']['latest_html']}")
    if decision["reasons"]:
        print("Motivos:")
        for reason in decision["reasons"]:
            print(f"  - {reason}")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Dos escenarios, para mostrar que el detector distingue uno del otro.
    print("\n### Escenario 1: lote SIN drift (control) ###")
    stable = monitoring_pipeline(drift_strength=0.0, n_rows=3000)
    _summarize(stable)

    print("\n### Escenario 2: lote CON drift macroeconómico ###")
    drifted = monitoring_pipeline(drift_strength=0.8, n_rows=3000)
    _summarize(drifted)
