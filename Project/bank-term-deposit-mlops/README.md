# Predicción de suscripción a depósito a plazo fijo

**Proyecto final MLOps — Universidad de Medellín**

Sistema de ML end-to-end que predice qué clientes bancarios suscribirán un
depósito a plazo fijo, para priorizar las llamadas de un call center.

Incluye experiment tracking, orquestación, dos modos de deployment, monitoreo de
drift verificado y CI.

**API desplegada y funcionando: <https://bank-term-deposit-api.onrender.com/docs>**
*(plan gratuito: si lleva rato sin visitas, la primera carga tarda ~50 s
mientras el servicio despierta)*

---

## Resultado

Evaluado sobre **8.238 clientes** del conjunto de test, nunca usados para
entrenar ni para ajustar nada:

| Métrica | Valor |
|---|---|
| PR-AUC (primaria) | **0,4909** — 4,4x el piso de 0,113 |
| ROC-AUC | 0,8183 |
| Precision @ umbral 0,42 | 0,3469 |
| Recall @ umbral 0,42 | 0,6789 |
| Lift en el top 20% | **3,31x** |
| Suscriptores alcanzados llamando al 20% de la base | **66,3%** |

### Lo que significa en dinero

| Concepto | Llamar a todos | Con el modelo |
|---|---|---|
| Llamadas | 8.238 | **1.816** (−78%) |
| Suscripciones | 928 | 630 (−32%) |
| Utilidad | $8.336 | **$35.872** |
| Costo por adquisición | $71,02 | **$23,06** (−68%) |

El modelo **consigue menos suscripciones y cuadruplica la utilidad**. Esa es la
propuesta completa: renunciar al 32% de las suscripciones para eliminar el 78%
de las llamadas. Si el banco prefiere conservar el volumen, baja el umbral: la
curva de utilidad en [`notebooks/02_baseline.ipynb`](notebooks/02_baseline.ipynb)
muestra exactamente lo que cuesta cada punto de recall.

Detalle completo en [`docs/business_case.md`](docs/business_case.md).

---

## Arranque rápido

Requiere Python 3.11+ y [uv](https://docs.astral.sh/uv/).

```powershell
cd Project/bank-term-deposit-mlops
./run.ps1 setup        # en Linux/macOS: make setup
```

**No hace falta reentrenar.** El repositorio incluye el historial de MLflow, los
datos y el modelo entrenado, así que `setup` solo instala dependencias, reapunta
las rutas de MLflow a tu máquina y verifica. Toma ~3 minutos.

Al terminar deberías ver:

```
173 passed
campeon: bank-term-deposit-classifier v1 umbral 0.42
```

> **Por qué hay que "reapuntar".** MLflow guarda la ubicación de cada artefacto
> como ruta **absoluta**. Al clonar en otra máquina esas rutas no existen:
> la interfaz abre y muestra los runs, pero ningún modelo carga, y el fallo es
> silencioso. `scripts/relocate_mlflow.py` reescribe las 128 rutas de la base de
> datos y los 10 archivos `MLmodel`. Detalle en
> [`docs/compartir.md`](docs/compartir.md).

### Ver todo funcionando

```powershell
./run.ps1 demo-up      # MLflow, Prefect, worker y API, cada uno en su ventana
./run.ps1 demo-open    # abre las 5 pestañas + el reporte de drift
./run.ps1 demo-check   # verifica que los 3 servicios respondan
```

| Servicio | URL |
|---|---|
| API (documentación interactiva) | <http://127.0.0.1:8000/docs> |
| MLflow (experimentos y registry) | <http://127.0.0.1:5001> |
| Prefect (flows y schedules) | <http://127.0.0.1:4200> |

> MLflow va en el puerto **5001**, no en 5000: en una máquina con el repositorio
> del curso, el 5000 suele estar ocupado por otro MLflow apuntando a una base de
> datos distinta.

### Reconstruirlo todo desde cero

Si prefieres no confiar en el estado versionado y reentrenar:

```powershell
./run.ps1 clean-all    # deja el repositorio en blanco
./run.ps1 bootstrap    # reentrena y reconstruye todo (~12 min)
```

El entrenamiento es determinista (semilla fija), así que obtienes las mismas
métricas: `Test PR-AUC 0.4909`, `ROC-AUC 0.8183`.

### Probar la predicción

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":62,"job":"retired","marital":"married","education":"university.degree",
       "default":"no","housing":"no","loan":"no","contact":"cellular","month":"mar",
       "day_of_week":"tue","campaign":1,"pdays":3,"previous":2,"poutcome":"success",
       "emp.var.rate":-1.8,"cons.price.idx":92.893,"cons.conf.idx":-46.2,
       "euribor3m":1.313,"nr.employed":5099.1}'
```

```json
{
  "probability": 0.623523,
  "will_subscribe": true,
  "propensity_band": "alta",
  "recommended_action": "llamar",
  "threshold_used": 0.42,
  "model_name": "bank-term-deposit-classifier",
  "model_version": "1"
}
```

### El resto del sistema

```bash
uv run python -m src.pipelines.training_flow        # reentrena (~5 min)
uv run python -m src.pipelines.batch_scoring_flow   # lista de llamadas priorizada
uv run python -m src.pipelines.monitoring_flow      # drift -> reports/drift_report_latest.html
uv run pytest                                       # 173 tests, ~10 s
```

Todos los comandos tienen atajo: `./run.ps1 help` en Windows, `make help` en
Linux/macOS.

Guion de exposición en [`docs/presentacion.md`](docs/presentacion.md); cómo
compartirlo y entregarlo, en [`docs/compartir.md`](docs/compartir.md).

---

## El problema

Un banco vende depósitos a plazo fijo por campañas telefónicas. **Solo 11,3% de
los contactados suscribe**: casi nueve de cada diez llamadas no producen nada.

El problema no es pérdida —las suscripciones pagan de sobra la operación— sino
**capacidad desperdiciada**. La pregunta que resuelve el modelo no es "¿quién va
a suscribir?" sino **"¿a quién llamo primero?"**. Esa diferencia determina todo
el diseño: la salida es una probabilidad usada como criterio de orden, no un
veredicto.

**Dataset:** [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
(Moro, Cortez & Rita, 2014) — 41.188 contactos reales de un banco portugués,
2008-2010, 20 variables. Se descarga automáticamente.

---

## Decisiones técnicas que definen el proyecto

### 1. Se descarta `duration`, la variable más predictiva

`duration` (duración de la llamada) solo se conoce **cuando la llamada ya
terminó**, pero el modelo tiene que decidir a quién llamar *antes* de marcar.
Es data leakage puro.

Medido en [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb): usarla **infla el
PR-AUC un 41%**, y todo ese rendimiento desaparecería en producción. Está
excluida en `configs/config.yaml` y hay un test que lo verifica.

Las métricas de este proyecto son más bajas que las de otros análisis del mismo
dataset. Esa es exactamente la intención.

### 2. PR-AUC, no accuracy

Con 88,7% de negativos, responder siempre "no suscribe" acierta el **88,7%** y
tiene valor cero. El ROC-AUC es mejor pero optimista: premia acertar en los
negativos, que son abundantes y fáciles.

PR-AUC solo mejora si el modelo ordena bien la clase minoritaria. Su piso es la
tasa base, 0,113.

### 3. `pdays == 999` es un centinela, no un número

El 96,3% de los clientes tiene `pdays = 999`, que significa **"nunca
contactado"**, no "999 días". Dejarla como número metería un outlier artificial
en el escalador.

`DomainFeatureEngineer` la descompone en `was_contacted_before` y
`days_since_contact`. La diferencia que justifica el esfuerzo: **9,3% de
suscripción entre no contactados antes, 63,8% entre contactados**.

### 4. El umbral de decisión es un parámetro de negocio, calibrado empíricamente

Con $8 por llamada y $80 por suscripción, el umbral se busca maximizando
utilidad sobre validation (`find_optimal_threshold`), nunca sobre test.

Resultó **0,42**, no el 0,10 que predice `costo/beneficio`. El motivo es
informativo: `class_weight="balanced"` infla las probabilidades predichas, así
que **los scores no están calibrados**. Sirven para ordenar clientes, no como
estimaciones de riesgo. Por eso la métrica que se le reporta al negocio es el
lift sobre el top-k, que solo depende del orden.

### 5. El hallazgo incómodo: es, en buena parte, un modelo macroeconómico

La importancia de features muestra que el modelo apoya la mayor parte de su
decisión en variables **macroeconómicas** (`nr.employed` domina con 0,29, luego
`emp.var.rate`, `cons.conf.idx`), no en el perfil del cliente.

Dentro del periodo de entrenamiento —que incluye la crisis de 2008— esa es señal
real. Fuera de ese régimen, deja de valer. Es el motivo por el que el monitoreo
de drift de este proyecto no es decorativo, y por el que el escenario de prueba
es precisamente un cambio de régimen macro.

### 6. El modelo nuevo no se promueve solo porque sea nuevo

Un candidato toma el alias `champion` **únicamente si supera al campeón vigente**
en PR-AUC sobre test por un margen mínimo. Si no, queda como `challenger`:
registrado y comparable, pero sin servir tráfico.

### 7. El flow de monitoreo detecta y recomienda; no reentrena

Un reentrenamiento automático disparado por un falso positivo de drift —un ETL
roto, por ejemplo— produce un modelo peor y lo despliega sin que nadie mire.

---

## Arquitectura

```
                       configs/config.yaml
                    (único punto de verdad)
                               │
   ┌───────────────────────────┼───────────────────────────┐
   │                           │                           │
   v                           v                           v
TRAINING                   DEPLOYMENT                  MONITOREO
(semanal)                  (API + batch)                (diario)

UCI Bank Marketing         ┌─ API FastAPI               lote de producción
   │                       │    /predict                       │
   v                       │    /predict/batch                 v
validación de datos        │    /health                  drift de features
   │                       │    /model-info               (PSI + KS + chi²)
   v                       │    /reload                        │
split estratificado ──────>│    /metrics                       v
   │       │               │                             drift de scores
   │       └── referencia ─┼──────────────────────────────>    │
   v                       │                                   v
experimentos MLflow        └─ scoring batch (Prefect)     desempeño vs. baseline
 baseline · RF · XGBoost        │                               │
   │                            v                               v
   v                      lista de llamadas              ¿reentrenar?
Optuna (20 trials)         priorizada (CSV)                     │
   │                                                            │
   v                                                            │
evaluación en test                                              │
 (una sola vez)                                                 │
   │                                                            │
   v                                                            │
Model Registry <─────── promoción condicional <─────────────────┘
 alias @champion         (solo si mejora)
```

### Stack

| Capa | Herramienta | Por qué |
|---|---|---|
| Entorno | **uv** | Resolución reproducible vía `uv.lock` |
| Datos | **pandas + pyarrow** | Parquet para splits y predicciones |
| Modelado | **scikit-learn + XGBoost** | Pipeline serializable con el preprocesamiento dentro |
| Tracking y registry | **MLflow 3** | Experimentos y alias de modelo |
| Tuning | **Optuna** | TPE, con cada trial como run anidado |
| Orquestación | **Prefect 3** | Flows, reintentos, schedules, artifacts |
| API | **FastAPI + Pydantic v2** | Validación estricta y OpenAPI automático |
| Monitoreo | **Implementación propia** (PSI, KS, chi²) | Umbrales interpretables, sin dependencias frágiles |
| Testing | **pytest** | 173 tests sobre datos sintéticos |
| Calidad | **ruff + black + pre-commit** | |
| CI | **GitHub Actions** | Lint, tests, smoke del pipeline, build de imagen |

---

## Estructura del repositorio

```
bank-term-deposit-mlops/
├── configs/config.yaml          # Único punto de verdad: rutas, features, umbrales
├── src/
│   ├── config.py                # Carga de config + overrides por entorno (BTD_*)
│   ├── data/
│   │   ├── download.py          # Ingesta desde UCI, cacheada en Parquet
│   │   ├── validate.py          # Contrato de datos: errores vs. advertencias
│   │   └── split.py             # ETL + split estratificado + referencia de drift
│   ├── features/preprocess.py   # DomainFeatureEngineer + ColumnTransformer
│   ├── models/
│   │   ├── train.py             # Experimentos MLflow + búsqueda con Optuna
│   │   ├── evaluate.py          # Métricas técnicas, lift y de negocio
│   │   ├── registry.py          # Registro, promoción condicional, export local
│   │   └── predict.py           # Servicio compartido por API y batch
│   ├── api/
│   │   ├── main.py              # Endpoints
│   │   └── schemas.py           # Contratos Pydantic (validación estricta)
│   ├── monitoring/
│   │   ├── drift.py             # PSI, KS, chi² + simulador de drift
│   │   └── report.py            # Reportes HTML y JSON
│   └── pipelines/
│       ├── training_flow.py     # Flow de entrenamiento
│       ├── batch_scoring_flow.py
│       └── monitoring_flow.py
├── notebooks/
│   ├── 01_eda.ipynb             # EDA (ejecutado, con salidas)
│   ├── 02_baseline.ipynb        # Baselines y umbral de negocio
│   └── 03_experiments.ipynb     # Lectura de MLflow y modelo final
├── tests/unit/                  # 173 tests
├── docs/
│   ├── business_case.md         # Problema, supuestos, métricas, alcance
│   ├── monitoring.md            # Diseño de monitoreo en cuatro capas
│   ├── deployment.md            # Local, Docker, cloud, CI/CD, troubleshooting
│   └── api.md                   # Referencia de endpoints
├── scripts/export_model.py      # Exporta el campeón para servir sin MLflow
├── deploy_flows.py              # Registra los flows con sus schedules
├── Dockerfile                   # Build en dos etapas, usuario sin privilegios
├── docker-compose.yml           # MLflow + API
├── render.yaml                  # Blueprint de deployment en cloud
├── Makefile / run.ps1           # Atajos (Unix / Windows)
└── models/
    ├── model_card.md            # Generada por el pipeline
    └── champion/                # Copia autocontenida (~1 MB) para cloud
```

---

## Cumplimiento de las fases del proyecto

| Fase | Requisito | Estado | Dónde |
|---|---|---|---|
| **1.1** | Problema, métricas, alcance, timeline, caso de negocio | Completo | [`docs/business_case.md`](docs/business_case.md) |
| **1.2** | Git, entorno virtual (uv), dependencias | Completo | `pyproject.toml`, `uv.lock` |
| **1.3** | EDA, preprocesamiento, baseline | Completo | `notebooks/01_eda.ipynb`, `02_baseline.ipynb` |
| **2.1** | MLflow configurado, experimentos, tracking de parámetros y métricas | Completo | `src/models/train.py` |
| **2.1** | "Logging automático" | **Resuelto con logging explícito, no con `mlflow.autolog()`** | Ver la nota de abajo |
| **2.2** | RF, XGBoost, hiperparámetros, cross-validation | Completo | 5 candidatos + 20 trials + CV de 5 folds |
| **2.3** | Registry, versionado con tags, staging/production, documentación | Completo | Alias `champion`/`challenger`, `models/model_card.md` |
| **3.1** | Prefect: flows de preprocesamiento y entrenamiento, scheduling | Completo | `src/pipelines/`, `deploy_flows.py` |
| **3.2** | ETL, transformaciones reproducibles, validación, logging | Completo | `src/data/` |
| **3.3** | Entrenamiento automatizado, feature engineering, evaluación | Completo | `training_flow.py` |
| **3.3** | Retraining automático *(nice to have)* | **Parcial, a propósito** | Se detecta y recomienda; ver decisión 7 |
| **4.1** | Dockerfile | Completo, **verificado en CI** | `Dockerfile`. No se construyó en la máquina de desarrollo (sin Docker instalado), pero el job `docker` del CI lo construye en cada push y comprueba que el contenedor responda en `/health` |
| **4.1** | Imagen optimizada *(nice to have)* | Completo | Build en dos etapas, usuario sin privilegios, `.dockerignore` |
| **4.2** | API REST, endpoints, validación de inputs | Completo | `src/api/`, [`docs/api.md`](docs/api.md) |
| **4.3** | Cloud + CI/CD *(nice to have)* | Completo | Render (`render.yaml`), CI en `.github/workflows/` con 5 jobs |
| **5** | Diseño de monitoreo | **Completo e implementado**, más allá del diseño pedido | [`docs/monitoring.md`](docs/monitoring.md) |
| **6.1** | Unit tests | Completo | 173 tests |
| **6.2** | Linter, formatter, pre-commit *(nice to have)* | Completo | ruff, black, `.pre-commit-config.yaml` |
| **6.3** | README, documentación de API, guías de deployment | Completo | `docs/` |

### Sobre el "logging automático" (fase 2.1)

El enunciado pide implementar *logging automático*, que normalmente significa
`mlflow.autolog()`. **No lo usamos**, y es una decisión deliberada que conviene
explicar porque alguien con la lista de chequeo en la mano puede buscar esa
línea y no encontrarla.

`mlflow.autolog()` registra automáticamente los parámetros y métricas estándar
de scikit-learn. El problema es que en este proyecto eso **se queda corto**:

1. Las métricas que importan aquí son **de negocio** —utilidad en dólares, lift
   en el top 20%, costo por adquisición, suscripciones perdidas— y autolog no
   las conoce. Son ~20 métricas por run que hay que calcular y registrar
   explícitamente (`src/models/evaluate.py`).
2. Con un `Pipeline` de sklearn, autolog registra los parámetros del pipeline
   completo con nombres poco legibles, en vez de los del estimador final.
3. La búsqueda con Optuna necesita **runs anidados** (un run padre y 20 hijos),
   estructura que hay que crear a mano con `mlflow.start_run(nested=True)`.

Lo que sí es automático es el **proceso**: cada candidato que entra en
`train_candidate()` queda registrado por completo —parámetros, métricas
técnicas, métricas de negocio, tags, firma del modelo y el modelo
serializado— sin que nadie tenga que acordarse de loguear nada. El pipeline de
Prefect lo hace para los 5 candidatos y los 20 trials en una sola ejecución.

En resumen: el requisito se cumple con logging explícito y exhaustivo, que da
estrictamente más información que autolog. Es `src/models/train.py:168-250`.

### Sobre el deployment en cloud

Desplegado en **Render** con un Blueprint (`render.yaml` en la raíz del
repositorio). Se evaluó Hugging Face Spaces, pero desde 2026 exige plan **PRO
de pago** para cualquier SDK que ejecute código; solo los *Static Spaces*
siguen gratis y un sitio estático no puede servir el modelo.

**API en producción: <https://bank-term-deposit-api.onrender.com/docs>**

En Render: **New → Blueprint** y seleccionar el repositorio. No hay nada que
configurar a mano.

> **Si la URL tarda en abrir, no está caída.** El plan gratuito de Render
> suspende el servicio tras 15 minutos sin tráfico y el arranque en frío tarda
> entre 30 y 60 segundos. La primera petición es lenta; a partir de ahí
> responde en milisegundos.

Lo que hace que esto no sea un proyecto aparte: el modelo exportado (~1 MB)
viaja **dentro de la imagen**, así que el contenedor no necesita un tracking
server ni una base de datos. Y cabe de sobra en los 512 MB del plan gratuito:
el pico medido es de 324 MB.

Verificado localmente: la API sirviendo la copia exportada, con
`BTD_SERVING__PREFER_LOCAL_MODEL=true` y sin ningún MLflow accesible:

```
/health:  healthy | modelo bank-term-deposit-classifier v1
/predict: prob = 0.623523 | version 1
coincide con el valor del registry (0.623523): True
```

Cubierto como regresión en
`tests/unit/test_registry.py::test_local_champion_round_trip_preserves_predictions`.
Alternativas evaluadas y descartadas (AWS ECS, SageMaker, Fly.io) en
[`docs/deployment.md`](docs/deployment.md).

---

## Monitoreo: verificado, no solo diseñado

Un detector de drift que nunca se probó contra drift real es un adorno. El
proyecto incluye un simulador que inyecta un escenario plausible —cambio de
régimen macroeconómico, público más joven, más insistencia del call center— con
intensidad controlable.

```bash
uv run python -m src.pipelines.monitoring_flow
```

Corre dos escenarios. **Resultados reales de esa ejecución:**

| | Sin drift (control) | Con drift macro |
|---|---|---|
| Features en alerta | **0** de 19 | **6** de 19 |
| PSI máximo | 0,0054 | 11,83 |
| PSI del score del modelo | 0,0038 (estable) | 1,779 (**alerta**) |
| Score medio | 0,384 → 0,388 | 0,384 → **0,557** |
| Veredicto | no reentrenar | **reentrenar** |

Las dos propiedades que importan quedan demostradas: **no inventa drift** cuando
no lo hay, y **lo detecta** cuando lo hay. Ambas están verificadas en CI como
aserciones, no como inspección visual.

Diseño de las cuatro capas (datos, predicciones, desempeño, negocio) en
[`docs/monitoring.md`](docs/monitoring.md).

---

## Testing y calidad

```bash
uv run pytest                                  # 173 tests, ~10 s
uv run pytest --cov=src --cov-report=html      # con cobertura
uv run ruff check src tests && uv run black --check src tests
```

**173 tests, 52% de cobertura de líneas.**

Sobre ese 52%, con honestidad: los módulos de lógica están bien cubiertos
(`validate.py` 99%, `drift.py` 97%, `report.py` 98%, `schemas.py` 100%,
`predict.py` 91%, `preprocess.py` 90%, `api/main.py` 86%). Los que aparecen en
0% son los **flows de Prefect** y `train.py`, que no se prueban con unit tests
—requerirían un servidor de MLflow y entrenar de verdad— sino con el job
`pipeline-smoke` del CI, que los ejecuta de punta a punta en una máquina limpia.

Inflar el número con tests que solo mockean Prefect daría una cobertura más
bonita y ninguna garantía adicional.

### Lo que los tests verifican de verdad

| Test | Qué protege |
|---|---|
| `test_clean_drops_leakage_column` | Que `duration` no vuelva a colarse |
| `test_splits_are_disjoint` | Que ninguna fila esté en dos splits |
| `test_splits_are_stratified` | Que las métricas entre splits sigan siendo comparables |
| `test_predict_one_matches_batch` | Que online y batch no divergan (training/serving skew) |
| `test_column_order_irrelevant` | Que el orden de claves de un JSON no cambie la predicción |
| `test_unseen_category_does_not_crash` | Que un `job` nuevo no tumbe el servicio |
| `test_optimal_threshold_maximizes_profit` | Que la búsqueda de umbral sea correcta |
| `test_local_champion_round_trip_preserves_predictions` | Que la copia para cloud sirva el mismo modelo |
| `test_champion_version_coerced_to_string` | Regresión de un 500 real en `/health` |
| `test_detect_drift_finds_nothing_on_resample` | Que el detector no genere falsos positivos |

---

## CI/CD

`.github/workflows/ci.yml` (y su copia en la raíz del monorepo), cuatro jobs:

| Job | Verifica |
|---|---|
| `quality` | ruff + black |
| `tests` | Los 173 tests con cobertura |
| `pipeline-smoke` | Borra el estado versionado, entrena desde cero, puntúa un lote y comprueba el monitoreo |
| `shipped-state` | Verifica lo que recibe quien clona: reapunta MLflow y comprueba que el campeón cargue y prediga el valor esperado |
| `docker` | Construye la imagen y comprueba que `/health` responda |

El smoke test **afirma**, no solo ejecuta:

```python
assert result['test_metrics']['test_roc_auc'] > 0.70   # piso de calidad
assert not stable['decision']['retrain_recommended']   # no inventa drift
assert drifted['decision']['retrain_recommended']      # sí lo detecta
```

Un pipeline que "corre sin errores" puede estar produciendo basura.

> GitHub Actions solo lee workflows desde la raíz del repositorio y este
> proyecto vive en un subdirectorio, de ahí las dos copias. Explicado en
> [`docs/deployment.md`](docs/deployment.md).

---

## Limitaciones conocidas

Declaradas aquí y en [`models/model_card.md`](models/model_card.md) para que
nadie las descubra en producción:

1. **Dependencia del régimen macroeconómico.** La limitación principal, y la
   razón del monitoreo. Ver decisión 5.
2. **Probabilidades sin calibrar.** Los scores ordenan bien pero no son
   estimaciones absolutas. Falta calibración isotónica o de Platt.
3. **Sin auditoría de equidad.** `age`, `marital` y `education` son predictivas
   y a la vez atributos sensibles. Antes de producción real haría falta un
   análisis por subgrupo; hacerlo a medias sería peor que declararlo pendiente.
4. **Sesgo de un solo banco.** Datos de una institución portuguesa, 2008-2010.
5. **Supuestos económicos.** Los $8 y $80 son razonables para el ejercicio, no
   cifras auditadas. Lo que el proyecto demuestra es el método de atarlos a la
   métrica.
6. **Sin autenticación en la API.** Decisión de alcance; `/reload` en particular
   no debería ser público. Detallado en [`docs/api.md`](docs/api.md).
7. **`month` no es estacionalidad**, es el calendario de campañas del banco. Si
   la operación lo cambia, la feature se vuelve engañosa.

---

## Documentación

| Documento | Contenido |
|---|---|
| [`docs/compartir.md`](docs/compartir.md) | Cómo clonarlo en otra máquina, cómo entregarlo, y cómo trabajar en equipo sin romper `mlflow.db` |
| [`docs/mensaje-entrega.md`](docs/mensaje-entrega.md) | Texto listo para enviar al profesor y a quienes hagan la evaluación por pares |
| [`docs/business_case.md`](docs/business_case.md) | Problema, supuestos, métricas de éxito, timeline, alcance |
| [`docs/monitoring.md`](docs/monitoring.md) | Monitoreo en cuatro capas, disparadores, verificación |
| [`docs/deployment.md`](docs/deployment.md) | Local, Docker, cloud, CI/CD, schedules, troubleshooting |
| [`docs/api.md`](docs/api.md) | Endpoints, validación, códigos de estado, lo que falta |
| [`models/model_card.md`](models/model_card.md) | Uso previsto, desempeño, limitaciones (autogenerada) |
| [`docs/presentacion.md`](docs/presentacion.md) | Recorrido comentado del proyecto, pantalla por pantalla |
| `reports/drift_report_latest.html` | Último reporte de monitoreo |

---

## Para quien evalúa este proyecto

```powershell
./run.ps1 setup        # instala y verifica, ~3 min (make setup en Linux/macOS)
./run.ps1 demo-up      # levanta MLflow, Prefect y la API
./run.ps1 demo-open    # abre todo en el navegador
```

No hace falta reentrenar: el repositorio incluye el historial de MLflow
completo. Si prefieres verificarlo desde cero, `./run.ps1 clean-all` seguido de
`./run.ps1 bootstrap` reentrena todo y debe dar las mismas métricas.

Los notebooks están **ejecutados y con salidas**: se pueden leer sin ejecutar
nada.

Una cosa que se declara por adelantado, para que no haya que buscarla: el
**Dockerfile no se construyó en la máquina de desarrollo** porque no teníamos
Docker instalado. Lo construye el job `docker` del CI en cada push, sobre un
runner limpio, y además comprueba que el contenedor arranque y responda en
`/health`. El resultado se puede ver en la pestaña Actions del repositorio.

---

## Referencia del dataset

Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the
Success of Bank Telemarketing.* Decision Support Systems, 62, 22-31.
<https://doi.org/10.1016/j.dss.2014.03.001>
