"""
Generación de reportes de monitoreo.

Dos salidas por cada chequeo:

* **HTML** (`reports/drift_report_*.html`) — para que una persona lo lea.
* **JSON** (`reports/drift_metrics_*.json`) — para que lo lea una máquina. En
  producción este archivo sería el que raspa Prometheus para alimentar Grafana
  (ver `docs/monitoring.md`).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import load_config
from src.monitoring.drift import DriftReport

logger = logging.getLogger(__name__)

SEVERITY_COLORS = {
    "stable": "#1a7f37",
    "warning": "#9a6700",
    "alert": "#cf222e",
}
SEVERITY_LABELS = {
    "stable": "Estable",
    "warning": "Advertencia",
    "alert": "Alerta",
}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _feature_rows_html(report: DriftReport) -> str:
    """Filas de la tabla de features, ordenadas por PSI descendente."""
    rows = []
    for item in report.features:
        color = SEVERITY_COLORS[item.severity]
        label = SEVERITY_LABELS[item.severity]
        reference_mean = (
            f"{item.reference_mean:.3f}" if item.reference_mean is not None else "&mdash;"
        )
        current_mean = f"{item.current_mean:.3f}" if item.current_mean is not None else "&mdash;"
        rows.append(f"""      <tr>
        <td><code>{item.feature}</code></td>
        <td>{item.feature_type}</td>
        <td class="num"><strong>{item.psi:.4f}</strong></td>
        <td><span class="badge" style="background:{color}">{label}</span></td>
        <td>{item.test_name}</td>
        <td class="num">{item.p_value:.2e}</td>
        <td class="num">{reference_mean}</td>
        <td class="num">{current_mean}</td>
      </tr>""")
    return "\n".join(rows)


def render_drift_html(
    report: DriftReport,
    performance: dict[str, Any] | None = None,
    model_info: dict[str, Any] | None = None,
) -> str:
    """
    Construye el HTML del reporte de drift.

    Args:
        report: resultado de `detect_drift`.
        performance: métricas de desempeño si hay ground truth disponible.
        model_info: datos del modelo campeón (nombre, versión, umbral).

    Returns:
        El documento HTML como string.
    """
    config = load_config()
    summary = report.summary()

    verdict_color = "#cf222e" if report.needs_retraining else "#1a7f37"
    verdict_text = "REENTRENAMIENTO RECOMENDADO" if report.needs_retraining else "MODELO ESTABLE"

    prediction_block = ""
    if report.prediction_drift:
        drift = report.prediction_drift
        prediction_block = f"""
  <section>
    <h2>Drift de predicciones</h2>
    <p>La distribución de scores del modelo suele moverse antes de que el
    desempeño caiga de forma medible: resume en una sola serie el efecto de
    todos los cambios en las entradas.</p>
    <table>
      <tr><th>PSI del score</th><td class="num"><strong>{drift.psi:.4f}</strong></td></tr>
      <tr><th>Severidad</th><td><span class="badge" style="background:{SEVERITY_COLORS[drift.severity]}">{SEVERITY_LABELS[drift.severity]}</span></td></tr>
      <tr><th>Score medio (referencia)</th><td class="num">{drift.reference_mean:.4f}</td></tr>
      <tr><th>Score medio (actual)</th><td class="num">{drift.current_mean:.4f}</td></tr>
      <tr><th>KS p-valor</th><td class="num">{drift.p_value:.2e}</td></tr>
    </table>
  </section>"""

    performance_block = ""
    if performance:
        rows = "\n".join(
            (
                f'      <tr><th>{key}</th><td class="num">{value:.4f}</td></tr>'
                if isinstance(value, float)
                else f'      <tr><th>{key}</th><td class="num">{value}</td></tr>'
            )
            for key, value in performance.items()
        )
        performance_block = f"""
  <section>
    <h2>Desempeño sobre ground truth</h2>
    <table>
{rows}
    </table>
  </section>"""

    model_block = ""
    if model_info:
        rows = "\n".join(
            f"      <tr><th>{key}</th><td><code>{value}</code></td></tr>"
            for key, value in model_info.items()
        )
        model_block = f"""
  <section>
    <h2>Modelo monitoreado</h2>
    <table>
{rows}
    </table>
  </section>"""

    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reporte de drift &mdash; {config.project.name}</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         margin: 0; padding: 2rem 1.25rem; background: #f6f8fa; color: #1f2328; line-height: 1.55; }}
  .wrap {{ max-width: 1040px; margin: 0 auto; }}
  h1 {{ font-size: 1.6rem; margin: 0 0 .25rem; }}
  h2 {{ font-size: 1.15rem; margin: 0 0 .75rem; }}
  .meta {{ color: #59636e; font-size: .9rem; margin-bottom: 1.5rem; }}
  section {{ background: #fff; border: 1px solid #d1d9e0; border-radius: 8px;
             padding: 1.25rem; margin-bottom: 1.25rem; }}
  .verdict {{ background: {verdict_color}; color: #fff; border: 0; }}
  .verdict h2 {{ color: #fff; }}
  .verdict p {{ margin: 0; opacity: .95; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
  th, td {{ text-align: left; padding: .5rem .6rem; border-bottom: 1px solid #eaeef2; }}
  th {{ color: #59636e; font-weight: 600; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  code {{ background: #eaeef2; padding: .1rem .35rem; border-radius: 4px; font-size: .85em; }}
  .badge {{ color: #fff; padding: .1rem .5rem; border-radius: 999px; font-size: .78rem;
            white-space: nowrap; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .75rem; }}
  .card {{ background: #f6f8fa; border: 1px solid #d1d9e0; border-radius: 6px; padding: .75rem; }}
  .card .k {{ color: #59636e; font-size: .78rem; text-transform: uppercase; letter-spacing: .03em; }}
  .card .v {{ font-size: 1.4rem; font-weight: 650; font-variant-numeric: tabular-nums; }}
  .scroll {{ overflow-x: auto; }}
  footer {{ color: #59636e; font-size: .82rem; text-align: center; padding: 1rem 0; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Reporte de monitoreo de drift</h1>
  <p class="meta">{config.project.name} &middot; generado el
     {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

  <section class="verdict">
    <h2>{verdict_text}</h2>
    <p>{len(report.alerts)} feature(s) con drift severo y {len(report.warnings)} con drift
       moderado, sobre {len(report.features)} evaluadas
       ({summary["share_drifted"]:.0%} del total).</p>
  </section>

  <section>
    <h2>Resumen</h2>
    <div class="cards">
      <div class="card"><div class="k">Filas referencia</div><div class="v">{report.n_reference:,}</div></div>
      <div class="card"><div class="k">Filas actuales</div><div class="v">{report.n_current:,}</div></div>
      <div class="card"><div class="k">PSI máximo</div><div class="v">{summary["max_psi"]:.3f}</div></div>
      <div class="card"><div class="k">Alertas</div><div class="v">{len(report.alerts)}</div></div>
      <div class="card"><div class="k">Advertencias</div><div class="v">{len(report.warnings)}</div></div>
    </div>
  </section>
{model_block}{prediction_block}{performance_block}
  <section>
    <h2>Drift por feature</h2>
    <p>Umbrales de PSI: &lt;{config.monitoring.psi_warning} estable &middot;
       {config.monitoring.psi_warning}&ndash;{config.monitoring.psi_alert} advertencia &middot;
       &ge;{config.monitoring.psi_alert} alerta.</p>
    <div class="scroll">
    <table>
      <thead>
      <tr>
        <th>Feature</th><th>Tipo</th><th>PSI</th><th>Severidad</th>
        <th>Test</th><th>p-valor</th><th>Media ref.</th><th>Media actual</th>
      </tr>
      </thead>
      <tbody>
{_feature_rows_html(report)}
      </tbody>
    </table>
    </div>
  </section>

  <footer>Generado por <code>src/monitoring/report.py</code></footer>
</div>
</body>
</html>
"""


def save_drift_report(
    report: DriftReport,
    performance: dict[str, Any] | None = None,
    model_info: dict[str, Any] | None = None,
    label: str | None = None,
) -> dict[str, Path]:
    """
    Escribe el reporte de drift en HTML y JSON.

    Args:
        report: resultado de `detect_drift`.
        performance: métricas de desempeño, si hay ground truth.
        model_info: datos del modelo monitoreado.
        label: sufijo para los nombres de archivo. None = timestamp.

    Returns:
        Mapa formato -> ruta escrita.
    """
    config = load_config()
    reports_dir = config.path("reports")
    suffix = label or _timestamp()

    html_path = reports_dir / f"drift_report_{suffix}.html"
    html_path.write_text(
        render_drift_html(report, performance=performance, model_info=model_info),
        encoding="utf-8",
    )

    json_path = reports_dir / f"drift_metrics_{suffix}.json"
    payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": report.summary(),
        "features": [item.as_row() for item in report.features],
        "prediction_drift": report.prediction_drift.as_row() if report.prediction_drift else None,
        "performance": performance,
        "model": model_info,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # `latest` siempre apunta al último reporte: así el README y cualquier
    # dashboard tienen una ruta estable que leer.
    latest_html = reports_dir / "drift_report_latest.html"
    latest_html.write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_json = reports_dir / "drift_metrics_latest.json"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")

    logger.info("Reporte de drift: %s", html_path)
    return {
        "html": html_path,
        "json": json_path,
        "latest_html": latest_html,
        "latest_json": latest_json,
    }
