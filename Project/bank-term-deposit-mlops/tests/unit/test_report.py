"""
Tests de la generación de reportes de monitoreo.

El renderizado es lógica pura, así que se puede probar sin modelo ni MLflow.
Vale la pena probarlo: un reporte que falla al generarse hace caer el flow de
monitoreo completo, justo cuando hay un problema que reportar.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.monitoring.drift import DriftReport, detect_drift
from src.monitoring.report import render_drift_html, save_drift_report

MODEL_INFO = {
    "model_name": "test-model",
    "model_version": "2",
    "decision_threshold": 0.4,
    "run_id": "run-de-prueba",
}


@pytest.fixture
def reports_dir(tmp_path, monkeypatch):
    """
    Redirige la escritura de reportes a un directorio temporal.

    Se parchea `load_config` **en el módulo de reportes**, no el objeto que
    devuelve el fixture `config`: `load_config` está memoizado y otros tests
    limpian esa caché, así que parchear la instancia no garantiza que
    `save_drift_report` vea el parche. Sin esto, estos tests escribirían en el
    `reports/` real del proyecto.
    """
    real_config = load_config()

    class ConfigWithTempPaths:
        """Delega todo en la configuración real, salvo las rutas de salida."""

        def __getattr__(self, name):
            return getattr(real_config, name)

        def path(self, key: str):
            return tmp_path

    monkeypatch.setattr("src.monitoring.report.load_config", ConfigWithTempPaths)
    return tmp_path


@pytest.fixture
def stable_report(synthetic_features) -> DriftReport:
    """Reporte sin drift: un remuestreo de los mismos datos."""
    return detect_drift(
        synthetic_features,
        synthetic_features.sample(300, replace=True, random_state=1),
        reference_scores=np.random.default_rng(1).beta(2, 8, 500),
        current_scores=np.random.default_rng(2).beta(2, 8, 300),
    )


@pytest.fixture
def drifted_report(synthetic_features) -> DriftReport:
    """Reporte con drift severo inyectado."""
    current = synthetic_features.sample(300, replace=True, random_state=3).copy()
    current["euribor3m"] = current["euribor3m"] + 4.0
    current["nr.employed"] = current["nr.employed"] - 200
    return detect_drift(synthetic_features, current)


def test_html_renders_for_stable_report(stable_report):
    """Un reporte estable produce HTML válido con el veredicto correcto."""
    html = render_drift_html(stable_report, model_info=MODEL_INFO)

    assert html.lstrip().startswith("<!doctype html>")
    assert "MODELO ESTABLE" in html
    assert "REENTRENAMIENTO RECOMENDADO" not in html


def test_html_renders_for_drifted_report(drifted_report):
    """Un reporte con drift severo lo dice en el veredicto."""
    html = render_drift_html(drifted_report, model_info=MODEL_INFO)

    assert "REENTRENAMIENTO RECOMENDADO" in html
    assert "euribor3m" in html


def test_html_includes_every_feature(stable_report):
    """Todas las features evaluadas aparecen en la tabla."""
    html = render_drift_html(stable_report, model_info=MODEL_INFO)
    for item in stable_report.features:
        assert item.feature in html


def test_html_includes_prediction_drift_section(stable_report):
    """Si hay drift de predicciones, se renderiza su sección."""
    html = render_drift_html(stable_report, model_info=MODEL_INFO)
    assert "Drift de predicciones" in html
    assert "PSI del score" in html


def test_html_omits_prediction_section_when_absent(drifted_report):
    """Sin scores, esa sección no se renderiza vacía."""
    html = render_drift_html(drifted_report, model_info=MODEL_INFO)
    assert "PSI del score" not in html


def test_html_includes_performance_when_given(stable_report):
    """Las métricas de desempeño se incluyen si hay ground truth."""
    html = render_drift_html(
        stable_report,
        performance={"pr_auc": 0.49, "recall": 0.68, "n_rows_evaluated": 8238},
        model_info=MODEL_INFO,
    )
    assert "Desempeño sobre ground truth" in html
    assert "0.4900" in html


def test_html_handles_empty_report():
    """
    Un reporte sin features no puede romper el renderizado.

    Pasa si la referencia y el lote actual no comparten ninguna columna, que es
    exactamente el momento en que hace falta ver un reporte.
    """
    html = render_drift_html(DriftReport(n_reference=0, n_current=0))
    assert html.lstrip().startswith("<!doctype html>")
    assert "0 feature(s)" in html or "sobre 0 evaluadas" in html


def test_html_defines_colors_outside_media_queries(stable_report):
    """El HTML trae su propio color de fondo: no depende del tema del visor."""
    html = render_drift_html(stable_report, model_info=MODEL_INFO)
    assert "background: #f6f8fa" in html
    assert "color-scheme" in html


def test_save_writes_html_and_json(stable_report, reports_dir):
    """Se escriben las cuatro rutas: con timestamp y los alias `latest`."""
    paths = save_drift_report(stable_report, model_info=MODEL_INFO, label="prueba")

    assert set(paths) == {"html", "json", "latest_html", "latest_json"}
    for path in paths.values():
        assert path.exists(), f"no se escribió {path}"
        assert path.parent == reports_dir, "se escribió fuera del directorio temporal"


def test_saved_json_is_machine_readable(stable_report, reports_dir):
    """El JSON tiene que ser parseable: es lo que consume un dashboard."""
    paths = save_drift_report(stable_report, model_info=MODEL_INFO, label="prueba")
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))

    assert payload["summary"]["n_features_checked"] == len(stable_report.features)
    assert len(payload["features"]) == len(stable_report.features)
    assert payload["model"]["model_version"] == "2"
    assert "generated_at" in payload


def test_latest_mirrors_timestamped_file(stable_report, reports_dir):
    """`latest` debe ser una copia exacta del reporte recién generado."""
    paths = save_drift_report(stable_report, model_info=MODEL_INFO, label="prueba")
    assert paths["latest_html"].read_text(encoding="utf-8") == paths["html"].read_text(
        encoding="utf-8"
    )


def test_summary_counts_match_features(drifted_report):
    """El resumen agregado tiene que cuadrar con el detalle por feature."""
    summary = drifted_report.summary()

    assert summary["n_features_checked"] == len(drifted_report.features)
    assert summary["n_alerts"] == len(drifted_report.alerts)
    assert summary["n_warnings"] == len(drifted_report.warnings)
    assert summary["max_psi"] == pytest.approx(
        round(max(item.psi for item in drifted_report.features), 4)
    )


def test_feature_row_is_serializable(drifted_report):
    """Cada fila debe poder volcarse a JSON para los artifacts de Prefect."""
    rows = [item.as_row() for item in drifted_report.features]
    assert json.loads(json.dumps(rows))
    assert pd.DataFrame(rows).shape[0] == len(drifted_report.features)
