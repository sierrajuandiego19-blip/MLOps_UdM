# Guía de deployment

El proyecto soporta tres modos de deployment. Los tres sirven **el mismo
modelo** a través del mismo `PredictionService`, lo que garantiza que una
predicción online y una batch sobre el mismo cliente den el mismo número.

| Modo | Para qué | Estado |
|---|---|---|
| **API REST** (FastAPI) | Decisión en tiempo real, un cliente a la vez | Verificado en local y en contenedor |
| **Scoring batch** (Prefect) | Lista de llamadas diaria para el call center | Verificado en local |
| **Cloud** (Render / HF Spaces) | API pública | Configuración lista; **no ejecutada** — ver la sección de cloud |

---

## 1. Local

### Requisitos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

### Instalación

```bash
cd Project/bank-term-deposit-mlops
uv sync --all-groups
```

### Entrenar (obligatorio antes de servir)

```bash
uv run python -m src.pipelines.training_flow
```

Descarga el dataset, entrena, registra todo en MLflow y promueve el campeón.
Toma unos 3-6 minutos según la máquina (lo que más tarda son los 20 trials de
Optuna). Para un arranque rápido:

```bash
uv run python -c "from src.pipelines.training_flow import training_pipeline; training_pipeline(skip_tuning=True)"
```

### Levantar la API

```bash
uv run uvicorn src.api.main:app --reload
```

- Servicio: <http://localhost:8000>
- Documentación interactiva: <http://localhost:8000/docs>

### Scoring batch

```bash
uv run python -m src.pipelines.batch_scoring_flow
```

Escribe en `data/predictions/`:

- `predictions_<batch_id>.parquet` — todo, para auditoría
- `predictions_<batch_id>.csv` — la lista de llamadas priorizada, que es lo que
  el equipo de campañas abre en Excel

### Monitoreo

```bash
uv run python -m src.pipelines.monitoring_flow
```

Genera `reports/drift_report_latest.html`.

---

## 2. Docker

### Build y ejecución del stack completo

```bash
docker compose up --build
```

Levanta dos servicios:

| Servicio | Puerto | Qué es |
|---|---|---|
| `mlflow` | 5000 | Tracking server y registry, sobre el `mlflow.db` del host |
| `api` | 8000 | La API, leyendo el modelo del tracking server |

### Solo la API

```bash
docker build -t bank-term-deposit-api .
docker run -p 8000:8000 -e BTD_SERVING__PREFER_LOCAL_MODEL=true bank-term-deposit-api
```

Esto funciona **sin MLflow**, porque la imagen lleva `models/champion/` dentro.

### Entrenar dentro del stack

```bash
docker compose --profile training up trainer
```

### Decisiones del Dockerfile

| Decisión | Motivo |
|---|---|
| Build en dos etapas | Las herramientas de compilación no llegan a la imagen final; queda alrededor de un tercio del tamaño |
| `uv` en vez de `pip` | Respeta el `uv.lock` exacto: el contenedor obtiene las mismas versiones que local |
| Manifiestos antes del código | Mientras `pyproject.toml`/`uv.lock` no cambien, Docker reutiliza la capa de dependencias y el rebuild es de segundos |
| Usuario `appuser` sin privilegios | Un proceso comprometido no debe correr como root |
| `HEALTHCHECK` contra `/health` | El orquestador distingue "proceso vivo" de "modelo cargado" |
| Sin notebooks, tests ni datos | No aportan en runtime y engordan la imagen |

### Estado de verificación

> El Dockerfile y el `docker-compose.yml` **no se construyeron en la máquina de
> desarrollo**: Docker no estaba instalado allí.
>
> Sí se construyen en CI. El job `docker` levanta un runner limpio de Ubuntu,
> construye la imagen desde cero y arranca el contenedor comprobando que
> `/health` responda. Corre en cada push y el resultado está en la pestaña
> Actions del repositorio.
>
> `docker-compose.yml` no lo cubre el CI: está revisado pero no ejecutado.

---

## 3. Cloud

### Qué se eligió y por qué

**Hugging Face Spaces** con SDK Docker:

| Opción | Dificultad | Veredicto |
|---|---|---|
| **Hugging Face Spaces** | Baja | **Elegida.** Gratis sin tarjeta, construye desde un Dockerfile, y solo se duerme tras **48 h** sin tráfico — decisivo si hay que mostrarla en vivo |
| Render (Blueprint) | Baja | Configurado como respaldo (`render.yaml`). Igual de simple, pero el plan gratuito apaga el servicio tras **15 min** sin tráfico y el arranque en frío tarda ~50 s |
| Fly.io | Media | Requiere tarjeta y CLI |
| AWS ECS + ECR | Alta | VPC, IAM, task definitions, load balancer. Desproporcionado para una API de un endpoint |
| AWS SageMaker | Alta | Formato de modelo propio, endpoints gestionados, costo por hora |

Lo que hace viable la opción simple es una decisión de diseño concreta: el
modelo exportado pesa ~1 MB y viaja **dentro de la imagen**. Sin eso habría que
desplegar también un tracking server y una base de datos, y el deployment en
cloud dejaría de ser un ejercicio de tres comandos.

### Pasos

```bash
# 1. Exportar el campeón a un directorio autocontenido
uv run python scripts/export_model.py

# 2. Ensamblar el repositorio del Space en build/huggingface-space/
uv run python scripts/deploy_huggingface.py

# 3. Subirlo (requiere el Space ya creado en huggingface.co/new-space)
uv run python scripts/deploy_huggingface.py --space TU-USUARIO/bank-term-deposit-api --push
```

Al crear el Space: **SDK Docker**, template **Blank**, hardware **CPU basic**
(gratis). Para el push, el usuario es el nombre de HF y la contraseña es un
*Access Token* con permiso de escritura
(<https://huggingface.co/settings/tokens>).

### Qué se sube y qué no

`scripts/deploy_huggingface.py` copia solo lo que el servicio necesita en
runtime — `src/`, `configs/`, `models/champion/`, `pyproject.toml`, `uv.lock`,
más el `README.md` con el frontmatter del Space y el `Dockerfile` de
`deploy/huggingface/`. Quedan fuera notebooks, tests, datos y `mlruns/`: 2,6 MB
en total.

El Dockerfile de Spaces difiere del principal en tres cosas, todas impuestas
por la plataforma: puerto **7860**, usuario con **UID 1000** (es el que Spaces
usa, y sin él el proceso no puede escribir), y
`BTD_SERVING__PREFER_LOCAL_MODEL=true` fijado en la imagen.

### Alternativa: Render

`render.yaml` está listo. En Render: **New → Blueprint**, apuntar al
repositorio. No hay nada que configurar en la interfaz.

### Estado de verificación

> **El push al Space lo hace quien tenga la cuenta**; el build de la imagen
> ocurre en la infraestructura de Hugging Face.
>
> Lo que **sí** está verificado localmente es exactamente el contenido que se
> sube. Se ejecutó la API usando `build/huggingface-space/` como raíz, con
> `BTD_SERVING__PREFER_LOCAL_MODEL=true` y sin ningún MLflow accesible:
>
> ```
> /health:  healthy | modelo bank-term-deposit-classifier v1
> /predict: prob = 0.623523 | version 1
> coincide con el valor del registry (0.623523): True
> ```
>
> Es decir: el código, la configuración y el modelo que viajan al Space sirven
> predicciones **idénticas** a las del registry, sin tracking server. Cubierto
> además como regresión en
> `tests/unit/test_registry.py::test_local_champion_round_trip_preserves_predictions`.

---

## 4. CI/CD

### Los dos archivos de workflow

GitHub Actions solo lee workflows desde `.github/workflows/` **en la raíz del
repositorio**, y este proyecto vive en un subdirectorio del monorepo del curso.
De ahí que haya dos copias:

| Archivo | Cuándo aplica |
|---|---|
| `<raíz del monorepo>/.github/workflows/project-mlops-ci.yml` | **La que se ejecuta hoy.** Los mismos jobs con `working-directory` apuntando al proyecto |
| `Project/bank-term-deposit-mlops/.github/workflows/ci.yml` | La copia canónica, para cuando el proyecto se extraiga a su propio repositorio |

### Los cuatro jobs

| Job | Qué verifica |
|---|---|
| `quality` | `ruff check` y `black --check` sobre `src/` y `tests/` |
| `tests` | Los 140 tests con cobertura. Corren sobre datos sintéticos: segundos, sin red |
| `pipeline-smoke` | Entrena de verdad desde cero (descargando el dataset), puntúa un lote y verifica el monitoreo. Es exactamente lo que hará quien evalúe el proyecto en una máquina limpia |
| `docker` | Construye la imagen y comprueba que `/health` responda |

### Lo que el smoke test afirma, no solo ejecuta

```python
assert result['test_metrics']['test_roc_auc'] > 0.70   # piso de calidad
assert not stable['decision']['retrain_recommended']   # no inventa drift
assert drifted['decision']['retrain_recommended']      # sí lo detecta
```

Un pipeline que "corre sin errores" puede estar produciendo basura. Estas tres
aserciones convierten el CI en una verificación de que el sistema **funciona**,
no solo de que no se cae.

---

## 5. Orquestación y schedules

### Registrar los deployments

```bash
# Terminal 1 — servidor de Prefect
uv run prefect server start

# Terminal 2 — registrar
uv run python deploy_flows.py

# Terminal 3 — worker que los ejecuta
uv run prefect worker start --pool local-pool
```

### Las tres cadencias

| Deployment | Cron | Por qué esa frecuencia |
|---|---|---|
| `training-pipeline/training-weekly` | Lunes 02:00 | Reentrenar más seguido no aporta: los indicadores macro se publican mensualmente |
| `batch-scoring-pipeline/scoring-daily` | Diario 06:00 | La lista de llamadas tiene que estar lista antes del turno |
| `monitoring-pipeline/monitoring-daily` | Diario 07:00 | Después del scoring, sobre datos frescos |

Zona horaria: `America/Bogota`.

### Disparar un run a mano

```bash
uv run prefect deployment run 'batch-scoring-pipeline/scoring-daily'
```

### Alternativa de un solo comando

```bash
uv run python deploy_flows.py --serve
```

Sirve los tres flows con sus schedules en un único proceso. Es el camino corto
para una demostración; para producción se prefieren los deployments con work
pool, que sobreviven a la caída del proceso.

---

## 6. Desplegar un modelo nuevo sin reiniciar nada

El punto del Model Registry con alias:

```bash
# 1. Reentrenar. Promueve a `champion` solo si supera al actual.
uv run python -m src.pipelines.training_flow

# 2. La API recarga el campeón en caliente.
curl -X POST http://localhost:8000/reload
```

Ni la API ni el batch saben qué número de versión está en producción: piden
`models:/bank-term-deposit-classifier@champion`. Mover el alias cambia el modelo
servido sin tocar una línea de código.

Para un rollback, mover el alias a la versión anterior y volver a llamar a
`/reload`.

---

## 7. Configuración por entorno

Todo se configura desde `configs/config.yaml`, y cualquier valor anidado se
puede sobreescribir con variables de entorno con prefijo `BTD_` y `__` como
separador. Así Docker, CI y cloud cambian configuración sin editar archivos:

```bash
BTD_MLFLOW__TRACKING_URI=http://mlflow:5000      # tracking remoto
BTD_SERVING__PREFER_LOCAL_MODEL=true             # servir la copia local
BTD_TRAINING__OPTUNA_TRIALS=3                    # búsqueda corta en CI
BTD_TRAINING__DECISION_THRESHOLD=0.25            # umbral más agresivo
BTD_BUSINESS__COST_PER_CALL_USD=12               # recalcular el caso de negocio
```

---

## 8. Resolución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| `/health` responde `degraded` | No hay modelo entrenado | `uv run python -m src.pipelines.training_flow` |
| `RuntimeError: No hay un modelo con alias 'champion'` | Ídem | Igual que arriba |
| El notebook 03 dice "No hay experimentos" | Se ejecutó antes de entrenar | Entrenar primero |
| MLflow aparece vacío al ejecutar desde otra carpeta | Ruta relativa de SQLite | Ya resuelto: `config.mlflow_tracking_uri` la ancla a la raíz del proyecto |
| `uv sync` falla con "not a member of the workspace" | El monorepo del curso declara un workspace | Ya resuelto: este `pyproject.toml` declara su propio `[tool.uv.workspace]` |
| El contenedor arranca pero no predice | Falta `models/champion/` en la imagen | `uv run python scripts/export_model.py` y reconstruir |
| Un clon del repo no trae `configs/config.yaml` ni los workflows | El `.gitignore` del monorepo del curso ignora `*.yaml`, `*.yml`, `*.json`, `*.txt`, `*.pkl` y `.vscode/` de forma global | Ya resuelto: el `.gitignore` del proyecto y `.github/.gitignore` los re-incluyen explícitamente. Verificar con `git check-ignore -q configs/config.yaml` (debe salir con código 1) |
