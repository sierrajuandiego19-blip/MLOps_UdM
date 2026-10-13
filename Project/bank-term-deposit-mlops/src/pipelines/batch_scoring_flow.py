"""
Flow de scoring batch.

Es el segundo modo de deployment (el primero es la API). Puntúa un lote de
clientes y escribe el resultado, ya ordenado por propensión, para que el call
center trabaje la lista de arriba hacia abajo.

Las predicciones se guardan con la versión del modelo que las generó: sin eso
es imposible auditar después por qué se llamó a quién.

Ejecutar:
    uv run python -m src.pipelines.batch_scoring_flow
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact

from src.config import load_config
from src.data.split import load_reference_data
from src.data.validate import validate_dataframe
from src.models.predict import get_prediction_service
from src.monitoring.drift import simulate_drifted_batch

logging.getLogger("mlflow").setLevel(logging.WARNING)

PREDICTIONS_FILENAME_TEMPLATE = "predictions_{batch_id}.parquet"


@task(name="load-scoring-batch", log_prints=True)
def load_batch_task(
    input_file: str | None = None,
    n_rows: int = 2000,
    drift_strength: float = 0.0,
) -> pd.DataFrame:
    """
    Carga el lote a puntuar.

    En producción esto leería de la base del banco. Aquí, si no se pasa un
    archivo, se sintetiza un lote remuestreando la referencia: el proyecto
    tiene que poder ejecutarse de punta a punta sin una fuente externa.

    Args:
        input_file: Parquet o CSV con los clientes a puntuar.
        n_rows: filas a generar si no hay archivo.
        drift_strength: intensidad del drift a inyectar (0 = ninguno).

    Returns:
        DataFrame con los clientes a puntuar.
    """
    logger = get_run_logger()

    if input_file:
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo de entrada: {path}")
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        logger.info("Lote cargado desde %s: %d filas", path, len(frame))
        return frame

    reference = load_reference_data()
    frame = simulate_drifted_batch(reference, n_rows=n_rows, drift_strength=drift_strength)
    logger.info("Lote sintetizado: %d filas (drift_strength=%.2f)", len(frame), drift_strength)
    return frame


@task(name="validate-scoring-batch", log_prints=True)
def validate_batch_task(frame: pd.DataFrame) -> dict[str, Any]:
    """Valida el lote sin exigir target: en inferencia es lo desconocido."""
    logger = get_run_logger()
    result = validate_dataframe(frame, require_target=False)
    for warning in result.warnings:
        logger.warning("Validación: %s", warning)
    result.raise_if_invalid()
    logger.info("Lote validado: %d filas", result.n_rows)
    return result.summary()


@task(name="score-batch", log_prints=True)
def score_batch_task(frame: pd.DataFrame, batch_id: str) -> dict[str, Any]:
    """
    Puntúa el lote con el modelo campeón y guarda el resultado.

    Args:
        frame: clientes a puntuar.
        batch_id: identificador del lote, para trazabilidad.

    Returns:
        Resumen del scoring con la ruta de salida.
    """
    logger = get_run_logger()
    config = load_config()
    service = get_prediction_service()

    scored = service.predict(frame)

    output = frame.reset_index(drop=True).copy()
    output["batch_id"] = batch_id
    output["probability"] = scored["probability"].to_numpy()
    output["will_subscribe"] = scored["will_subscribe"].to_numpy()
    output["propensity_band"] = scored["propensity_band"].to_numpy()
    output["recommended_action"] = scored["recommended_action"].to_numpy()
    output["model_name"] = service.champion.name
    output["model_version"] = service.champion.version
    output["threshold_used"] = service.threshold
    output["scored_at"] = datetime.now().isoformat(timespec="seconds")

    # Ordenado por propensión: la salida *es* la cola de llamadas.
    output = output.sort_values("probability", ascending=False).reset_index(drop=True)
    output["call_priority"] = output.index + 1

    destination = config.path("data_predictions") / PREDICTIONS_FILENAME_TEMPLATE.format(
        batch_id=batch_id
    )
    output.to_parquet(destination, index=False)

    # Copia en CSV: el equipo de campañas trabaja en Excel, no en Parquet.
    csv_destination = destination.with_suffix(".csv")
    call_list_columns = [
        "call_priority",
        "age",
        "job",
        "marital",
        "contact",
        "probability",
        "propensity_band",
        "recommended_action",
    ]
    output[[c for c in call_list_columns if c in output.columns]].to_csv(
        csv_destination, index=False
    )

    capacity = config.business.call_capacity_fraction
    n_capacity = max(1, int(len(output) * capacity))

    summary = {
        "batch_id": batch_id,
        "n_clients": len(output),
        "n_to_call": int(output["will_subscribe"].sum()),
        "call_rate": round(float(output["will_subscribe"].mean()), 4),
        "mean_probability": round(float(output["probability"].mean()), 6),
        "max_probability": round(float(output["probability"].max()), 6),
        "min_probability": round(float(output["probability"].min()), 6),
        "mean_probability_top_capacity": round(
            float(output["probability"].head(n_capacity).mean()), 6
        ),
        "band_distribution": output["propensity_band"].value_counts().to_dict(),
        "model_name": service.champion.name,
        "model_version": service.champion.version,
        "threshold_used": round(service.threshold, 4),
        "output_file": str(destination),
        "call_list_csv": str(csv_destination),
    }

    logger.info(
        "Puntuados %d clientes | %d a llamar (%.1f%%) | modelo v%s",
        summary["n_clients"],
        summary["n_to_call"],
        summary["call_rate"] * 100,
        summary["model_version"],
    )
    return summary


@task(name="publish-batch-artifact", log_prints=True)
def publish_batch_artifact_task(summary: dict[str, Any], validation: dict[str, Any]) -> None:
    """Publica el resumen del lote en la UI de Prefect."""
    config = load_config()
    bands = summary["band_distribution"]
    band_rows = "\n".join(
        f"| {band} | {count:,} | {count / summary['n_clients']:.1%} |"
        for band, count in sorted(bands.items(), key=lambda item: -item[1])
    )

    create_markdown_artifact(
        key="batch-scoring-summary",
        markdown=f"""# Scoring batch — `{summary["batch_id"]}`

**Ejecutado:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Modelo
- **Nombre:** `{summary["model_name"]}`
- **Versión:** {summary["model_version"]}
- **Umbral:** {summary["threshold_used"]}

## Resultado
| Métrica | Valor |
|---|---|
| Clientes puntuados | {summary["n_clients"]:,} |
| Recomendados para llamar | {summary["n_to_call"]:,} ({summary["call_rate"]:.1%}) |
| Probabilidad media | {summary["mean_probability"]:.4f} |
| Probabilidad media del top {int(config.business.call_capacity_fraction * 100)}% | {summary["mean_probability_top_capacity"]:.4f} |
| Rango | {summary["min_probability"]:.4f} &ndash; {summary["max_probability"]:.4f} |

## Distribución por banda de propensión
| Banda | Clientes | % |
|---|---|---|
{band_rows}

## Archivos
- Predicciones: `{summary["output_file"]}`
- Lista de llamadas (CSV): `{summary["call_list_csv"]}`

## Validación de entrada
- Filas: {validation["n_rows"]:,} | Advertencias: {validation["n_warnings"]}
""",
        description=f"Resumen del lote {summary['batch_id']}",
    )


@flow(
    name="batch-scoring-pipeline",
    description="Puntúa un lote de clientes con el modelo campeón y genera la lista de llamadas",
    log_prints=True,
)
def batch_scoring_pipeline(
    input_file: str | None = None,
    n_rows: int = 2000,
    drift_strength: float = 0.0,
    batch_id: str | None = None,
) -> dict[str, Any]:
    """
    Pipeline de scoring batch.

    Args:
        input_file: archivo de entrada. None = sintetizar un lote.
        n_rows: filas a sintetizar si no hay archivo.
        drift_strength: drift a inyectar en el lote sintético.
        batch_id: identificador del lote. None = timestamp.

    Returns:
        Resumen del lote puntuado.
    """
    logger = get_run_logger()
    resolved_batch_id = batch_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Iniciando scoring batch — batch_id=%s", resolved_batch_id)

    frame = load_batch_task(input_file=input_file, n_rows=n_rows, drift_strength=drift_strength)
    validation = validate_batch_task(frame)
    summary = score_batch_task(frame, resolved_batch_id)
    publish_batch_artifact_task(summary, validation)

    logger.info("Scoring batch completado: %s", summary["output_file"])
    return {"status": "success", **summary, "validation": validation}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    result = batch_scoring_pipeline()
    print("\n" + "=" * 70)
    print(f"Batch ID:            {result['batch_id']}")
    print(f"Clientes puntuados:  {result['n_clients']:,}")
    print(f"A llamar:            {result['n_to_call']:,} ({result['call_rate']:.1%})")
    print(f"Modelo:              {result['model_name']} v{result['model_version']}")
    print(f"Salida:              {result['output_file']}")
    print("=" * 70)
