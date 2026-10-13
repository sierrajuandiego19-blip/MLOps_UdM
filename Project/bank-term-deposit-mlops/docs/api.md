# API de predicción

Base URL en local: `http://localhost:8000`
Documentación interactiva (Swagger): `http://localhost:8000/docs`

## Endpoints

| Método | Ruta | Para qué |
|---|---|---|
| `GET` | `/` | Metadatos del servicio y rutas disponibles |
| `GET` | `/health` | Health check (liveness / readiness) |
| `GET` | `/model-info` | Versión, umbral y métricas del modelo servido |
| `GET` | `/metrics` | Contadores en texto plano, formato Prometheus |
| `POST` | `/predict` | Puntúa un cliente |
| `POST` | `/predict/batch` | Puntúa hasta 1000 clientes |
| `POST` | `/reload` | Recarga el campeón del registry sin reiniciar |

---

## `GET /health`

Responde **200** en ambos casos, con el campo `status` como discriminante:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "bank-term-deposit-classifier",
  "model_version": "1",
  "api_version": "1.0.0",
  "detail": null
}
```

### Por qué `degraded` devuelve 200 y no 503

Si el modelo no se pudo cargar, el servicio **arranca igual** y responde:

```json
{
  "status": "degraded",
  "model_loaded": false,
  "api_version": "1.0.0",
  "detail": "No hay un modelo con alias 'champion' en 'bank-term-deposit-classifier'..."
}
```

Un contenedor que se cae al arrancar entra en bucle de reinicios y lo único que
el operador ve en los logs es el reinicio. Así, en cambio, el proceso vive, el
health check reporta el problema **con su causa**, y el balanceador lo saca de
rotación sin matarlo. Las predicciones sí devuelven 503 en ese estado.

---

## `POST /predict`

### Petición

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 62,
    "job": "retired",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "no",
    "loan": "no",
    "contact": "cellular",
    "month": "mar",
    "day_of_week": "tue",
    "campaign": 1,
    "pdays": 3,
    "previous": 2,
    "poutcome": "success",
    "emp.var.rate": -1.8,
    "cons.price.idx": 92.893,
    "cons.conf.idx": -46.2,
    "euribor3m": 1.313,
    "nr.employed": 5099.1
  }'
```

### Respuesta

```json
{
  "probability": 0.623523,
  "will_subscribe": true,
  "propensity_band": "alta",
  "recommended_action": "llamar",
  "threshold_used": 0.42,
  "model_name": "bank-term-deposit-classifier",
  "model_version": "1",
  "predicted_at": "2026-10-13T01:31:59+00:00"
}
```

### Cómo interpretar la respuesta

| Campo | Qué es y qué no es |
|---|---|
| `probability` | Score para **ordenar** clientes. **No** es una probabilidad calibrada: `class_weight="balanced"` infla los valores. Un 0,62 no significa "62% de probabilidad" |
| `will_subscribe` | Decisión al umbral vigente. Es una **recomendación**, no una predicción del futuro |
| `propensity_band` | `muy_baja` / `baja` / `media` / `alta`. Para segmentar sin exponer el score |
| `recommended_action` | `llamar` / `no_llamar` |
| `threshold_used` | El umbral aplicado. Viene del modelo, no de la API: se calibró durante el entrenamiento y viaja con él |
| `model_version` | Qué versión produjo este score. Imprescindible para auditar después |

---

## `POST /predict/batch`

Hasta 1000 clientes por petición. Bastante más eficiente que N llamadas a
`/predict`: el pipeline vectoriza el preprocesamiento y la inferencia sobre todo
el lote.

### Petición

```json
{
  "clients": [
    { "age": 41, "job": "technician", "...": "..." },
    { "age": 62, "job": "retired",    "...": "..." }
  ]
}
```

### Respuesta

```json
{
  "predictions": [
    {"index": 0, "probability": 0.345718, "will_subscribe": false, "propensity_band": "media"},
    {"index": 1, "probability": 0.623523, "will_subscribe": true,  "propensity_band": "alta"}
  ],
  "n_clients": 2,
  "n_to_call": 1,
  "mean_probability": 0.48462,
  "threshold_used": 0.42,
  "model_name": "bank-term-deposit-classifier",
  "model_version": "1",
  "predicted_at": "2026-10-13T01:31:59+00:00"
}
```

El campo `index` preserva el orden de entrada, así que el cliente puede unir los
resultados con su propia lista sin depender del orden de la respuesta.

> Para volúmenes mayores a 1000 el camino correcto es el **scoring batch**
> (`src/pipelines/batch_scoring_flow.py`), no iterar sobre este endpoint.

---

## `GET /model-info`

```json
{
  "name": "bank-term-deposit-classifier",
  "version": "1",
  "alias": "champion",
  "run_id": "ab111936b5194b719b3e7f344eb246e4",
  "decision_threshold": 0.42,
  "metrics": {
    "test_pr_auc": 0.4909,
    "test_roc_auc": 0.8183,
    "test_precision": 0.3469,
    "test_recall": 0.6789
  },
  "loaded_at": "2026-10-13T01:30:12+00:00",
  "feature_columns": ["age", "campaign", "..."]
}
```

Útil para que quien consume la API pueda verificar **qué** modelo le está
respondiendo y con qué desempeño medido, sin preguntarle a nadie.

---

## `POST /reload`

Recarga el campeón desde el registry sin reiniciar el proceso. Es el mecanismo
de despliegue: el pipeline de entrenamiento promueve una versión al alias
`champion` y luego se llama a este endpoint.

```bash
curl -X POST http://localhost:8000/reload
```

```json
{
  "status": "reloaded",
  "model_name": "bank-term-deposit-classifier",
  "model_version": "1",
  "decision_threshold": 0.42
}
```

> En un despliegue real este endpoint iría protegido por autenticación. Este
> proyecto no implementa auth: ver la sección de seguridad al final.

---

## Validación de entradas

El contrato es **estricto a propósito** (`src/api/schemas.py`).

| Regla | Ejemplo que se rechaza | Por qué |
|---|---|---|
| Categorías cerradas (`Literal`) | `"job": "astronauta"` | El modelo nunca la vio; un one-hot de ceros produciría un score sin sentido |
| Rangos numéricos | `"age": 200`, `"euribor3m": 50` | No es un escenario, es un error de datos |
| `extra="forbid"` | `"saldo": 1000` | Casi siempre es un nombre mal escrito. Fallar fuerte es mejor que puntuar ignorando una feature en silencio |
| `pdays`: 0-40 o exactamente 999 | `"pdays": 500` | 999 es el centinela "nunca contactado"; los intermedios no existen en el dataset |
| Alias con punto obligatorios | `"emp_var_rate"` en vez de `"emp.var.rate"` | El pipeline selecciona columnas por nombre: el alias equivocado perdería la feature |

Todas devuelven **422** con el campo señalado:

```json
{
  "detail": [
    {
      "type": "literal_error",
      "loc": ["body", "job"],
      "msg": "Input should be 'admin.', 'blue-collar', ...",
      "input": "astronauta"
    }
  ]
}
```

---

## Códigos de estado

| Código | Cuándo | Qué hacer |
|---|---|---|
| 200 | Todo bien | — |
| 422 | Entrada inválida | Corregir el payload; `detail` dice qué campo |
| 500 | Error inesperado al predecir | Revisar logs; incrementa `api_errors_total` |
| 503 | No hay modelo cargado | Entrenar y llamar a `/reload` |

---

## `GET /metrics`

Texto plano en formato Prometheus:

```
# HELP api_predict_requests_total Peticiones a /predict
# TYPE api_predict_requests_total counter
api_predict_requests_total 2
# HELP api_latency_ms_avg Latencia media de inferencia en ms
# TYPE api_latency_ms_avg gauge
api_latency_ms_avg 102.285
# HELP api_model_loaded 1 si hay un modelo cargado
# TYPE api_model_loaded gauge
api_model_loaded 1
```

No es un cliente de Prometheus completo —sería una dependencia más para lo que
aquí es una demostración— pero el formato es el que un scraper espera, así que
apuntar Prometheus a esta ruta funciona tal cual. Detalle en
[`monitoring.md`](monitoring.md).

**Limitación:** los contadores viven en memoria del proceso. Con varias réplicas
cada una reporta los suyos; en producción irían a un backend compartido.

---

## Lo que esta API no tiene

Declarado explícitamente para que nadie lo descubra en producción:

| Ausente | Qué haría falta |
|---|---|
| **Autenticación** | API keys o OAuth2. `/reload` en particular no debería ser público |
| **Rate limiting** | `slowapi` o el límite del API gateway |
| **CORS** | Configurar `CORSMiddleware` si la consume un navegador |
| **HTTPS** | Lo resuelve el proxy inverso (Render y HF Spaces lo dan hecho) |
| **Registro de predicciones** | La API no persiste lo que predice; el scoring batch sí. Para monitorear tráfico online habría que añadirlo |
| **Trazas distribuidas** | OpenTelemetry si formara parte de un sistema mayor |

Son decisiones de alcance, no omisiones: añadir auth a medias en un proyecto
académico da una falsa sensación de seguridad.
