# Diseño de monitoreo

El proyecto implementa el monitoreo de datos y de predicciones, y documenta
cómo se completaría el de desempeño y de negocio en una operación real.

## Por qué este modelo necesita monitoreo en serio

La importancia de features (`notebooks/03_experiments.ipynb`) deja algo claro:
el modelo apoya la mayor parte de su decisión en variables **macroeconómicas**
(`nr.employed`, `emp.var.rate`, `cons.conf.idx`), no en el perfil del cliente.

Se entrenó con datos de 2008-2010 —la crisis financiera—, donde el euríbor cae
de ~5% a ~0,7% y el empleo se desploma. El modelo aprendió las correlaciones de
ese régimen. En otro contexto económico, la relación puede invertirse y el
modelo seguiría prediciendo con total confianza a partir de un mundo que ya no
existe.

El monitoreo de drift es la única defensa contra eso. Por eso vigila esas
variables en particular, y por eso el escenario de prueba es un cambio de
régimen macro.

## Las cuatro capas

| Capa | Qué detecta | Estado | Dónde |
|---|---|---|---|
| 1. Drift de datos | Las entradas cambiaron de distribución | **Implementado** | `src/monitoring/drift.py` |
| 2. Drift de predicciones | Los scores del modelo se desplazaron | **Implementado** | `src/monitoring/drift.py` |
| 3. Desempeño | El modelo acierta menos contra ground truth | **Implementado** (con test como sustituto) | `src/pipelines/monitoring_flow.py` |
| 4. Operación y negocio | Latencia, errores, conversión real | **Parcial**: la API expone `/metrics`; el resto es diseño | `src/api/main.py`, este documento |

El orden no es arbitrario: va de lo que se detecta **inmediatamente** a lo que
se detecta **con semanas de retraso**. El resultado real de una llamada se
conoce días después de hacerla, así que el monitoreo de desempeño siempre va por
detrás del de datos. De ahí que las capas 1 y 2 sean la alerta temprana.

---

## Capa 1 — Drift de datos

### Tres tests, cada uno para lo que sabe medir

**PSI (Population Stability Index)** — numéricas y categóricas. Es el estándar
de facto en banca y su gran virtud es tener **umbrales interpretables** que no
dependen del tamaño de la muestra:

| PSI | Interpretación | Acción |
|---|---|---|
| < 0,10 | Estable | Ninguna |
| 0,10 – 0,25 | Cambio moderado | Advertencia; revisar si se acumulan |
| ≥ 0,25 | Cambio severo | Alerta; evaluar reentrenamiento |

Para numéricas los bins son **cuantiles de la referencia** (robustos a
distribuciones asimétricas); para categóricas, las categorías observadas. Un
epsilon evita que un bin vacío produzca `log(0)`.

**Kolmogorov-Smirnov** — numéricas. Da un p-valor.

**Chi-cuadrado** — categóricas. También da un p-valor.

### Por qué los p-valores no deciden nada aquí

Con 3.000 filas, el test KS detecta diferencias **estadísticamente
significativas y operativamente irrelevantes**. Un desplazamiento del 0,1% en la
media de `age` sale con p < 0,001 y no cambia ninguna predicción.

Por eso **el PSI es el que decide** y los p-valores se reportan como información
complementaria. Es una decisión deliberada: es la diferencia entre un sistema de
alertas que alguien lee y uno que se silencia en la primera semana.

### La referencia

`data/reference/reference.parquet` es un snapshot del split de entrenamiento,
escrito por el propio pipeline de entrenamiento. Es la distribución **sobre la
que el modelo aprendió**, que es exactamente la línea base correcta: cuando se
reentrena, la referencia se actualiza con el modelo.

---

## Capa 2 — Drift de predicciones

Se mide el PSI sobre la distribución de scores del modelo, con los mismos
umbrales.

Suele ser **la primera señal útil**, por dos razones:

1. Resume en una sola serie el efecto combinado de todos los cambios de entrada,
   incluidos los que individualmente no superan el umbral.
2. Está disponible al instante, sin esperar ground truth.

Si el score medio pasa de 0,38 a 0,56, el modelo está marcando mucha más gente
como propensa. Puede ser un cambio real del mercado o puede ser que el modelo se
rompió; en ambos casos hay que mirar antes de que el call center llame al doble
de gente.

---

## Capa 3 — Desempeño contra ground truth

`check_performance_task` compara el PR-AUC actual contra el registrado en el
entrenamiento y marca degradación si la caída relativa supera el 10%
(`monitoring.performance_drop_alert`).

**Limitación declarada:** en este proyecto usa el split de test como sustituto
de las etiquetas que llegarían con retraso. En producción habría que:

1. Registrar cada predicción con su `batch_id` y versión de modelo (el scoring
   batch ya lo hace: ver `data/predictions/`).
2. Unir con el resultado real de la llamada cuando el CRM lo reporte.
3. Recalcular las métricas sobre esa ventana con etiquetas completas.

El retraso real sería de días a semanas, lo que refuerza por qué las capas 1 y 2
llevan el peso de la detección temprana.

---

## Capa 4 — Operación y negocio

### Implementado

`GET /metrics` expone contadores en formato texto estilo Prometheus:

| Métrica | Tipo | Para qué |
|---|---|---|
| `api_predict_requests_total` | counter | Volumen de tráfico |
| `api_predict_positive_total` | counter | Deriva en la tasa de recomendación |
| `api_batch_clients_total` | counter | Clientes puntuados en batch |
| `api_errors_total` | counter | Tasa de error |
| `api_model_reloads_total` | counter | Despliegues de modelo |
| `api_latency_ms_avg` | gauge | Latencia de inferencia |
| `api_model_loaded` | gauge | 1/0 — alimenta la alerta de servicio caído |

No es un cliente de Prometheus completo (sería una dependencia más para lo que
aquí es una demostración), pero el formato es el que un scraper espera, así que
apuntar Prometheus a esa ruta funciona tal cual.

### Diseño pendiente: métricas de negocio

Lo que de verdad habría que vigilar, y que requiere datos de la operación que el
dataset no tiene:

| Métrica | Alerta si | Por qué importa |
|---|---|---|
| Tasa de conversión real del top-k | cae > 20% vs. la esperada | Es la medida directa de si el modelo sirve |
| Costo por adquisición | sube > 30% | Traduce la degradación a dinero |
| Lift realizado vs. predicho | divergen > 25% | Detecta que el orden dejó de ser bueno |
| Cobertura de la lista | el call center ignora > 40% de lo sugerido | Detecta falta de adopción, no un problema del modelo |

La última es la que más se olvida: un modelo excelente que el equipo no usa tiene
el mismo impacto que uno malo.

---

## Disparadores de reentrenamiento

Definidos en `configs/config.yaml`, evaluados por `decide_retraining_task`.
Cualquiera de los cuatro basta:

| Disparador | Condición |
|---|---|
| Drift severo en alguna feature | PSI ≥ 0,25 |
| Drift generalizado | > 33% de las features con PSI ≥ 0,10 |
| Drift severo de scores | PSI del score ≥ 0,25 |
| Degradación de desempeño | caída relativa de PR-AUC > 10% |

### El flow no reentrena solo. A propósito.

Detecta y **recomienda**. Un reentrenamiento automático disparado por un falso
positivo de drift —por ejemplo, un lote cargado con un error en el ETL— produce
un modelo peor y lo despliega sin que nadie mire.

La segunda red de seguridad es la promoción condicional: aun cuando alguien
reentrene, el candidato **solo toma el alias `champion` si supera al campeón
actual** en PR-AUC sobre test por un margen mínimo (`src/models/registry.py`).
Si no, queda como `challenger`: registrado y disponible para comparar, pero sin
servir tráfico.

---

## Verificación: el detector se prueba a sí mismo

Un detector de drift que nunca se probó contra drift real es un adorno. El
proyecto incluye `simulate_drifted_batch`, que inyecta un escenario plausible
—cambio de régimen macro, público más joven, más insistencia del call center,
migración a celular— con intensidad controlable.

Ejecutar los dos escenarios:

```bash
uv run python -m src.pipelines.monitoring_flow
```

### Resultados reales de esa ejecución

**Escenario 1 — lote sin drift (control):**

```
Features con alerta: 0
PSI máximo:          0.0054
Reentrenar:          no
```

**Escenario 2 — cambio de régimen macroeconómico:**

```
Features con alerta: 6
PSI máximo:          11.83
Drift de predicciones: PSI=1.779, score medio 0.384 -> 0.557
Reentrenar:          SÍ
Motivos:
  - drift severo en 6 feature(s): emp.var.rate, euribor3m,
    nr.employed, cons.conf.idx, campaign, age
  - drift severo de scores (PSI=1.779)
```

Las dos propiedades que importan quedan demostradas: **no inventa drift** cuando
no lo hay, y **lo detecta** cuando lo hay. Ambas están verificadas en CI como
aserciones, no como inspección visual.

---

## Reportes generados

Cada ejecución escribe en `reports/`:

| Archivo | Para quién |
|---|---|
| `drift_report_<timestamp>.html` | Una persona: tabla por feature, severidad en color, veredicto |
| `drift_metrics_<timestamp>.json` | Una máquina: lo que raspa Prometheus o ingiere un dashboard |
| `drift_report_latest.html` | Ruta estable al último reporte |
| `drift_metrics_latest.json` | Ídem, para automatización |

Además se publican artifacts en la UI de Prefect (tabla de drift y resumen en
markdown), que es donde alguien revisando un run programado los va a buscar
primero.

---

## Cómo se vería en producción

Lo que está implementado aquí cubre la lógica de detección. El stack de
observabilidad alrededor sería:

```
API / batch scoring
    │
    ├── GET /metrics ──────────> Prometheus ──> Grafana (latencia, errores, tráfico)
    │
    └── predicciones ──────────> almacenamiento (tabla de predicciones)
                                      │
                                      ├── join con ground truth del CRM
                                      │        (retraso de días)
                                      │
                                      v
                            monitoring-pipeline (Prefect, diario)
                                      │
                                      ├── drift_metrics_latest.json ──> Grafana
                                      └── alerta (Slack / email) si retrain_recommended
                                                   │
                                                   v
                                      revisión humana ──> training-pipeline
                                                              │
                                                              v
                                                   promoción condicional
```

Decisiones de ese diseño que vale la pena señalar:

- **Grafana sobre el JSON, no sobre la base de datos.** El flow ya calculó el
  PSI; que el dashboard lo recalcule duplica lógica y abre la puerta a que
  reporten números distintos.
- **La alerta va a una persona, no al pipeline de entrenamiento.** Ver la
  sección de disparadores.
- **La referencia se actualiza al reentrenar, no por calendario.** La línea base
  tiene que ser siempre la distribución del modelo vigente.
