"""
Registro de los flows de Prefect con sus schedules.

Crea tres deployments con cadencias distintas, elegidas por lo que cada uno
responde:

| Deployment        | Cadencia          | Por qué                                    |
|-------------------|-------------------|--------------------------------------------|
| `training-weekly` | Lunes 02:00       | Reentrenar más seguido no aporta: los datos macro se publican mensualmente |
| `scoring-daily`   | Diario 06:00      | La lista de llamadas tiene que estar lista antes del turno |
| `monitoring-daily`| Diario 07:00      | Después del scoring, sobre datos frescos   |

Uso:

    # 1. Levantar el servidor de Prefect (en otra terminal)
    uv run prefect server start

    # 2. Registrar los deployments
    uv run python deploy_flows.py

    # 3. Levantar un worker que los ejecute
    uv run prefect worker start --pool local-pool

Para disparar un run a mano sin esperar el schedule:

    uv run prefect deployment run 'batch-scoring-pipeline/scoring-daily'
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prefect import Flow
from prefect.client.schemas.schedules import CronSchedule

from src.pipelines.batch_scoring_flow import batch_scoring_pipeline
from src.pipelines.monitoring_flow import monitoring_pipeline
from src.pipelines.training_flow import training_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

WORK_POOL = "local-pool"
TIMEZONE = "America/Bogota"
PROJECT_ROOT = Path(__file__).resolve().parent

# (nombre, entrypoint, descripción, cron, tags, parámetros)
DEPLOYMENT_SPECS: list[tuple[str, str, str, str, list[str], dict]] = [
    (
        "training-weekly",
        "src/pipelines/training_flow.py:training_pipeline",
        "Reentrena el modelo semanalmente. Promueve el candidato a campeón "
        "solo si supera al actual en PR-AUC sobre test.",
        "0 2 * * 1",
        ["training", "weekly", "mlops"],
        {"force_download": False},
    ),
    (
        "scoring-daily",
        "src/pipelines/batch_scoring_flow.py:batch_scoring_pipeline",
        "Puntúa la base de clientes y genera la lista de llamadas priorizada "
        "antes del turno de la mañana.",
        "0 6 * * *",
        ["scoring", "daily", "mlops"],
        {"n_rows": 5000, "drift_strength": 0.0},
    ),
    (
        "monitoring-daily",
        "src/pipelines/monitoring_flow.py:monitoring_pipeline",
        "Mide drift de datos y de predicciones contra la referencia de "
        "entrenamiento y recomienda reentrenar si hace falta.",
        "0 7 * * *",
        ["monitoring", "daily", "mlops"],
        {"n_rows": 3000, "drift_strength": 0.3},
    ),
]


def deploy_all(work_pool: str = WORK_POOL) -> dict[str, str]:
    """
    Registra los tres deployments programados.

    Args:
        work_pool: work pool donde se encolan los runs.

    Returns:
        Mapa nombre -> id del deployment.
    """
    deployments: dict[str, str] = {}
    source = str(PROJECT_ROOT)

    for name, entrypoint, description, cron, tags, parameters in DEPLOYMENT_SPECS:
        # `from_source` es imprescindible: sin él el deployment queda registrado
        # pero el worker no sabe de dónde sacar el código, copia a un temporal
        # vacío y el run termina en CRASHED. Con la ruta del proyecto como
        # origen, el worker (que corre en esta misma máquina) lo encuentra.
        deployed_flow = Flow.from_source(source=source, entrypoint=entrypoint)
        deployments[name] = str(
            deployed_flow.deploy(
                name=name,
                work_pool_name=work_pool,
                description=description,
                schedule=CronSchedule(cron=cron, timezone=TIMEZONE),
                tags=tags,
                parameters=parameters,
                # El work pool es de tipo `process`: ejecuta en un subproceso
                # local, no necesita imagen de contenedor.
                build=False,
                push=False,
            )
        )
        logger.info("Registrado '%s' (%s)", name, cron)

    return deployments


def serve_all() -> None:
    """
    Sirve los tres flows en este proceso, sin necesidad de un worker aparte.

    Es el camino corto para una demostración: un solo comando levanta los
    schedules. Para producción se prefieren deployments con work pool, que
    sobreviven a la caída de este proceso.
    """
    from prefect import serve

    logger.info("Sirviendo flows con schedules (Ctrl+C para detener)")
    serve(
        training_pipeline.to_deployment(
            name="training-weekly",
            schedule=CronSchedule(cron="0 2 * * 1", timezone=TIMEZONE),
            tags=["training", "weekly"],
        ),
        batch_scoring_pipeline.to_deployment(
            name="scoring-daily",
            schedule=CronSchedule(cron="0 6 * * *", timezone=TIMEZONE),
            tags=["scoring", "daily"],
            parameters={"n_rows": 5000},
        ),
        monitoring_pipeline.to_deployment(
            name="monitoring-daily",
            schedule=CronSchedule(cron="0 7 * * *", timezone=TIMEZONE),
            tags=["monitoring", "daily"],
            parameters={"n_rows": 3000, "drift_strength": 0.3},
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Registra los flows de Prefect")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="sirve los flows en este proceso en vez de registrar deployments",
    )
    parser.add_argument(
        "--work-pool",
        default=WORK_POOL,
        help=f"work pool para los deployments (por defecto: {WORK_POOL})",
    )
    args = parser.parse_args()

    if args.serve:
        serve_all()
        return

    deployments = deploy_all(work_pool=args.work_pool)

    print("\n" + "=" * 70)
    print("DEPLOYMENTS REGISTRADOS")
    print("=" * 70)
    for name, deployment_id in deployments.items():
        print(f"  {name:20s} {deployment_id}")
    print("=" * 70)
    print("\nPara ejecutarlos hace falta un worker:")
    print(f"  uv run prefect worker start --pool {args.work_pool}")
    print("\nPara disparar un run de inmediato:")
    print("  uv run prefect deployment run 'batch-scoring-pipeline/scoring-daily'")
    print("=" * 70)


if __name__ == "__main__":
    main()
