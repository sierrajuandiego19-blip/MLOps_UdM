"""
Entrenamiento con experiment tracking en MLflow.

Estrategia de experimentación, de lo simple a lo complejo:

1. **Baselines** — `DummyClassifier` (siempre la clase mayoritaria) y una
   regresión logística. Sin un piso contra el que comparar, un PR-AUC de 0.4
   no significa nada.
2. **Modelos de árbol** — Random Forest y XGBoost con hiperparámetros por
   defecto razonables.
3. **Optimización** — Optuna sobre el mejor candidato, con cada trial como run
   anidado en MLflow para poder auditar la búsqueda.

Todo se registra con `mlflow`: parámetros, métricas (técnicas y de negocio),
el pipeline serializado, y la firma del modelo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import load_config
from src.data.split import DataSplits
from src.features.preprocess import build_model_pipeline
from src.models.evaluate import evaluate_model, find_optimal_threshold

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_ARTIFACT_NAME = "model"

# MLflow 3 serializa sklearn con skops por defecto, y skops rechaza
# transformers propios como `DomainFeatureEngineer`. Con cloudpickle el
# pipeline completo —feature engineering incluido— viaja en un solo objeto.
# Contrapartida: al cargar el modelo, el paquete `src` debe ser importable.
# Lo es tanto en la API como en el batch scoring y en la imagen de Docker.
SERIALIZATION_FORMAT = mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE


@dataclass
class TrainedCandidate:
    """Un modelo entrenado junto con su run de MLflow y sus métricas."""

    name: str
    pipeline: Pipeline
    run_id: str
    metrics: dict[str, float]
    # URI del logged model de MLflow 3 (`models:/m-...`). Es lo que se pasa al
    # registry: en MLflow 3 `log_model` ya no deja el modelo como artifact del
    # run, así que `runs:/<run_id>/model` no resuelve.
    model_uri: str = ""

    @property
    def score(self) -> float:
        """Valor de la métrica primaria sobre validation."""
        config = load_config()
        return self.metrics[f"valid_{config.training.primary_metric}"]


def setup_mlflow() -> str:
    """
    Apunta MLflow al backend configurado y asegura que el experimento exista.

    Returns:
        El `experiment_id` activo.
    """
    config = load_config()
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    experiment = mlflow.set_experiment(config.mlflow.experiment_name)
    logger.info(
        "MLflow: tracking_uri=%s experimento='%s' (id=%s)",
        config.mlflow_tracking_uri,
        config.mlflow.experiment_name,
        experiment.experiment_id,
    )
    return experiment.experiment_id


def _signature_sample(frame: pd.DataFrame, n_rows: int = 20) -> pd.DataFrame:
    """
    Muestra para inferir la firma del modelo, con los enteros como float64.

    Un entero de numpy no puede representar un faltante, así que una firma con
    columnas `integer` rechaza en inferencia cualquier petición donde pandas
    haya promovido esa columna a float por un NaN —exactamente el caso que el
    imputador del pipeline está ahí para resolver—. Declararlas como `double`
    desde el principio evita ese error de schema enforcement.

    Args:
        frame: datos de entrenamiento.
        n_rows: filas de la muestra.

    Returns:
        Muestra con las columnas enteras convertidas a float64.
    """
    sample = frame.head(n_rows).copy()
    integer_columns = sample.select_dtypes(include=["int8", "int16", "int32", "int64"]).columns
    if len(integer_columns):
        sample[integer_columns] = sample[integer_columns].astype("float64")
    return sample


def _scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    """Razón negativos/positivos, para compensar el desbalance en XGBoost."""
    y = np.asarray(y).astype(int)
    positives = y.sum()
    return float((len(y) - positives) / positives) if positives else 1.0


def candidate_estimators(y_train: pd.Series) -> dict[str, Any]:
    """
    Los candidatos de la fase exploratoria.

    Todos tratan el desbalance explícitamente (`class_weight` o
    `scale_pos_weight`): sin eso los modelos aprenden a responder siempre "no".
    """
    config = load_config()
    seed = config.project.seed

    return {
        "baseline_majority": DummyClassifier(strategy="most_frequent", random_state=seed),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=_scale_pos_weight(y_train),
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        ),
    }


def _log_dataset_params(splits: DataSplits) -> None:
    """Registra la forma de los datos para que el run sea auto-explicativo."""
    config = load_config()
    mlflow.log_params(
        {
            "n_train": len(splits.x_train),
            "n_valid": len(splits.x_valid),
            "n_test": len(splits.x_test),
            "n_features_input": splits.x_train.shape[1],
            "train_positive_rate": round(float(splits.y_train.mean()), 5),
            "dropped_columns": ",".join(config.data.drop_columns),
            "seed": config.project.seed,
        }
    )


def train_candidate(
    name: str,
    estimator: Any,
    splits: DataSplits,
    run_cross_validation: bool = True,
) -> TrainedCandidate:
    """
    Entrena un candidato en su propio run de MLflow.

    Args:
        name: nombre del candidato, usado como nombre del run.
        estimator: clasificador sklearn-compatible.
        splits: datos.
        run_cross_validation: si se corre CV estratificada sobre train. Se
            desactiva en la búsqueda de hiperparámetros, donde sería redundante.

    Returns:
        `TrainedCandidate` con el pipeline entrenado y sus métricas.
    """
    config = load_config()

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags(
            {
                "model_family": name,
                "stage": "exploration",
                "dataset": "uci-bank-marketing",
            }
        )
        _log_dataset_params(splits)
        mlflow.log_params({f"param_{key}": value for key, value in estimator.get_params().items()})

        pipeline = build_model_pipeline(estimator)
        logger.info("Entrenando '%s'...", name)
        pipeline.fit(splits.x_train, splits.y_train)

        metrics: dict[str, float] = {}
        metrics.update(evaluate_model(pipeline, splits.x_train, splits.y_train, prefix="train_"))
        metrics.update(evaluate_model(pipeline, splits.x_valid, splits.y_valid, prefix="valid_"))

        # Cross-validation sobre train: distingue un modelo bueno de uno con
        # suerte en un split concreto.
        if run_cross_validation and name != "baseline_majority":
            cv = StratifiedKFold(
                n_splits=config.training.cv_folds,
                shuffle=True,
                random_state=config.project.seed,
            )
            cv_scores = cross_val_score(
                build_model_pipeline(estimator),
                splits.x_train,
                splits.y_train,
                cv=cv,
                scoring="average_precision",
                n_jobs=1,
            )
            metrics["cv_pr_auc_mean"] = float(cv_scores.mean())
            metrics["cv_pr_auc_std"] = float(cv_scores.std())
            logger.info(
                "  CV PR-AUC: %.4f +/- %.4f (%d folds)",
                cv_scores.mean(),
                cv_scores.std(),
                config.training.cv_folds,
            )

        mlflow.log_metrics(metrics)

        sample = _signature_sample(splits.x_train)
        signature = infer_signature(sample, pipeline.predict_proba(sample))
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name=MODEL_ARTIFACT_NAME,
            signature=signature,
            input_example=sample.head(3),
            serialization_format=SERIALIZATION_FORMAT,
        )

        logger.info(
            "  '%s': valid PR-AUC=%.4f ROC-AUC=%.4f utilidad=$%s",
            name,
            metrics["valid_pr_auc"],
            metrics["valid_roc_auc"],
            f"{metrics['valid_profit_usd']:,.0f}",
        )

        return TrainedCandidate(
            name=name,
            pipeline=pipeline,
            run_id=run.info.run_id,
            metrics=metrics,
            model_uri=model_info.model_uri,
        )


def run_exploration(splits: DataSplits) -> list[TrainedCandidate]:
    """
    Entrena todos los candidatos y los devuelve ordenados por la métrica primaria.

    Args:
        splits: datos.

    Returns:
        Lista de candidatos, el mejor primero.
    """
    candidates = [
        train_candidate(name, estimator, splits)
        for name, estimator in candidate_estimators(splits.y_train).items()
    ]
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)

    logger.info("Ranking de la fase exploratoria (valid PR-AUC):")
    for position, candidate in enumerate(candidates, start=1):
        logger.info("  %d. %-22s %.4f", position, candidate.name, candidate.score)

    return candidates


def optimize_xgboost(splits: DataSplits, n_trials: int | None = None) -> TrainedCandidate:
    """
    Optimiza XGBoost con Optuna; cada trial queda como run anidado en MLflow.

    El espacio de búsqueda sesga hacia la regularización: con 26k filas de
    entrenamiento y 11% de positivos, un XGBoost profundo sobreajusta rápido.

    Args:
        splits: datos.
        n_trials: número de trials. None = el de configuración.

    Returns:
        El mejor modelo reentrenado y registrado en su propio run.
    """
    config = load_config()
    trials = n_trials if n_trials is not None else config.training.optuna_trials
    seed = config.project.seed
    pos_weight = _scale_pos_weight(splits.y_train)

    with mlflow.start_run(run_name="xgboost-optuna-search") as parent_run:
        mlflow.set_tags({"stage": "hyperparameter_search", "optimizer": "optuna"})
        mlflow.log_params({"n_trials": trials, "search_metric": "valid_pr_auc"})

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            }

            with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True):
                mlflow.log_params(params)
                estimator = XGBClassifier(
                    **params,
                    scale_pos_weight=pos_weight,
                    eval_metric="aucpr",
                    tree_method="hist",
                    n_jobs=-1,
                    random_state=seed,
                )
                pipeline = build_model_pipeline(estimator)
                pipeline.fit(splits.x_train, splits.y_train)

                valid_metrics = evaluate_model(
                    pipeline, splits.x_valid, splits.y_valid, prefix="valid_"
                )
                mlflow.log_metrics(valid_metrics)
                return valid_metrics[f"valid_{config.training.primary_metric}"]

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            study_name="xgboost-pr-auc",
        )
        logger.info("Lanzando búsqueda con Optuna (%d trials)...", trials)
        study.optimize(
            objective,
            n_trials=trials,
            timeout=config.training.optuna_timeout_seconds,
            show_progress_bar=False,
        )

        mlflow.log_params({f"best_{key}": value for key, value in study.best_params.items()})
        mlflow.log_metric("best_valid_pr_auc", study.best_value)
        logger.info(
            "Mejor trial: #%d con valid PR-AUC=%.4f", study.best_trial.number, study.best_value
        )
        logger.info("Mejores hiperparámetros: %s", study.best_params)
        parent_run_id = parent_run.info.run_id

    # Reentrena el ganador en un run limpio y de primer nivel: es el candidato
    # que irá al registry, y así no queda sepultado dentro de la búsqueda.
    best_estimator = XGBClassifier(
        **study.best_params,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )
    tuned = train_candidate("xgboost_tuned", best_estimator, splits, run_cross_validation=True)

    with mlflow.start_run(run_id=tuned.run_id):
        mlflow.set_tags({"stage": "tuned", "optuna_parent_run_id": parent_run_id})

    return tuned


def finalize_best_model(candidate: TrainedCandidate, splits: DataSplits) -> dict[str, Any]:
    """
    Evalúa el ganador en test y calibra el umbral de decisión.

    El test set se toca **una sola vez**, aquí. El umbral se busca en validation
    (nunca en test) para no contaminar la estimación de desempeño.

    Args:
        candidate: el modelo ganador.
        splits: datos.

    Returns:
        Resumen con métricas de test y el umbral óptimo.
    """
    valid_proba = candidate.pipeline.predict_proba(splits.x_valid)[:, 1]
    optimal_threshold, expected_profit = find_optimal_threshold(splits.y_valid, valid_proba)

    test_metrics = evaluate_model(
        candidate.pipeline,
        splits.x_test,
        splits.y_test,
        prefix="test_",
        threshold=optimal_threshold,
    )

    with mlflow.start_run(run_id=candidate.run_id):
        mlflow.log_metrics(test_metrics)
        mlflow.log_metrics(
            {
                "optimal_threshold": optimal_threshold,
                "valid_profit_at_optimal_threshold": expected_profit,
            }
        )
        mlflow.set_tags({"stage": "final", "evaluated_on_test": "true"})

    logger.info(
        "Desempeño en test | PR-AUC=%.4f ROC-AUC=%.4f recall=%.3f precision=%.3f",
        test_metrics["test_pr_auc"],
        test_metrics["test_roc_auc"],
        test_metrics["test_recall"],
        test_metrics["test_precision"],
    )

    return {
        "name": candidate.name,
        "run_id": candidate.run_id,
        "model_uri": candidate.model_uri,
        "optimal_threshold": optimal_threshold,
        "valid_metrics": candidate.metrics,
        "test_metrics": test_metrics,
    }
