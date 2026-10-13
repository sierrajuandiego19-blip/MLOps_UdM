---
title: Bank Term Deposit API
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Predice la propension de un cliente a suscribir un deposito a plazo fijo
---

# Bank Term Deposit Prediction API

API REST que predice la probabilidad de que un cliente bancario suscriba un
depósito a plazo fijo, para priorizar las llamadas de un call center.

Proyecto final de MLOps — Universidad de Medellín.

## Documentación interactiva

👉 **[/docs](/docs)** — Swagger UI, se puede probar la predicción desde el navegador.

## Endpoints

| Método | Ruta | Para qué |
|---|---|---|
| `GET` | [`/health`](/health) | Estado del servicio y versión del modelo |
| `GET` | [`/model-info`](/model-info) | Métricas de test del modelo servido |
| `GET` | [`/metrics`](/metrics) | Contadores formato Prometheus |
| `POST` | `/predict` | Puntúa un cliente |
| `POST` | `/predict/batch` | Puntúa hasta 1000 clientes |

## Ejemplo

```bash
curl -X POST https://<este-space>.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 62, "job": "retired", "marital": "married",
    "education": "university.degree", "default": "no", "housing": "no",
    "loan": "no", "contact": "cellular", "month": "mar", "day_of_week": "tue",
    "campaign": 1, "pdays": 3, "previous": 2, "poutcome": "success",
    "emp.var.rate": -1.8, "cons.price.idx": 92.893, "cons.conf.idx": -46.2,
    "euribor3m": 1.313, "nr.employed": 5099.1
  }'
```

Respuesta:

```json
{
  "probability": 0.623523,
  "will_subscribe": true,
  "propensity_band": "alta",
  "recommended_action": "llamar",
  "threshold_used": 0.42,
  "model_version": "1"
}
```

## Modelo

| | |
|---|---|
| Algoritmo | XGBoost con hiperparámetros optimizados (Optuna) |
| Dataset | [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) — 41.188 contactos |
| PR-AUC (test) | 0,4909 — 4,4x la tasa base de 0,113 |
| ROC-AUC (test) | 0,8183 |
| Lift en el top 20% | 3,31x |

El modelo se sirve desde una copia autocontenida exportada del Model Registry
de MLflow, por lo que el contenedor no necesita acceso a un tracking server.

## Cómo interpretar la salida

`probability` es un **score para ordenar clientes**, no una probabilidad
calibrada: el entrenamiento usa `class_weight="balanced"`, que infla los
valores. Un 0,62 no significa "62% de probabilidad de suscribir".

`will_subscribe` es una **recomendación** al umbral vigente (0,42), calibrado
maximizando la utilidad del negocio, no una predicción del futuro.

## Limitaciones

El modelo apoya la mayor parte de su decisión en variables macroeconómicas
(`nr.employed`, `emp.var.rate`) y se entrenó con datos de 2008-2010. Fuera de
ese régimen económico su desempeño degrada. No tiene autenticación ni rate
limiting: es una demostración académica.

---

Código y documentación completa: repositorio del proyecto.
