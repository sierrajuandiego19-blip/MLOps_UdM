# Model Card — bank-term-deposit-classifier

## Resumen

| Campo | Valor |
|---|---|
| Versión registrada | 1 |
| Alias | `champion` |
| Familia del modelo | xgboost_tuned |
| Run de MLflow | `0770a64d966747feba24b8842bff2276` |
| Umbral de decisión | 0.42 |
| Métrica de selección | `pr_auc` (PR-AUC) |

## Uso previsto

Priorizar a qué clientes llama el call center en una campaña de depósitos a
plazo fijo. La salida es una **probabilidad de suscripción**, no una decisión:
el umbral es un parámetro de negocio ajustable según la capacidad disponible.

**Fuera de alcance:** decidir condiciones del producto, precios o
elegibilidad crediticia. El modelo no fue validado para ninguno de esos usos.

## Datos de entrenamiento

- **Fuente:** UCI Bank Marketing (Moro, Cortez & Rita, 2014), variante
  `bank-additional-full` — campañas telefónicas de un banco portugués,
  mayo 2008 a noviembre 2010.
- **Volumen:** 41.188 contactos, 20 variables de entrada.
- **Clase positiva:** 11,3% (suscribió).
- **Partición:** 60/20/20 train/valid/test, estratificada, semilla
  42.

### Variable excluida deliberadamente

`duration` (duración de la llamada) se **descarta**. Solo se conoce al terminar
la llamada, así que usarla produce un modelo con métricas excelentes e
inservible en producción, cuando hay que decidir *antes* de llamar. Lo advierte
la propia documentación del dataset.

## Desempeño (conjunto de test, nunca usado para ajustar)

| Métrica | Valor |
|---|---|
| PR-AUC (primaria) | 0.4909 |
| ROC-AUC | 0.8183 |
| Precision @ umbral | 0.3469 |
| Recall @ umbral | 0.6789 |
| F1 | 0.4592 |
| Brier score | 0.1521 |
| Lift en el top 20% | 3.31x |
| Recall en el top 20% | 0.663 |

## Impacto de negocio estimado

Con los supuestos de `configs/config.yaml`
($8 por llamada,
$80 por suscripción):

| Métrica | Valor |
|---|---|
| Utilidad del modelo | $35,872 |
| Utilidad llamando a todos | $8,336 |
| **Ganancia incremental** | **$27,536** |
| Costo por adquisición | $23.06 |

## Limitaciones conocidas

1. **Deriva macroeconómica.** Cinco de las features son indicadores
   macroeconómicos (`euribor3m`, `nr.employed`, `emp.var.rate`, …) y el periodo
   de entrenamiento incluye la crisis de 2008. En otro régimen económico el
   modelo degrada rápido; de ahí el monitoreo de drift sobre esas variables en
   particular.
2. **Sesgo de un solo banco.** Los datos provienen de una institución
   portuguesa. Trasladarlo a otro mercado exige revalidación.
3. **Variables sensibles.** `age`, `marital` y `education` son predictivas y a
   la vez atributos protegidos o correlacionados con ellos. Antes de producción
   real hace falta una auditoría de equidad por subgrupo; este proyecto no la
   incluye.
4. **Probabilidades sin calibrar.** Se reporta el Brier score, pero no se aplicó
   calibración isotónica ni Platt. Los scores sirven para **ordenar**
   clientes, no como estimaciones absolutas de riesgo.
5. **Sin ground truth inmediato.** El resultado real de una llamada se conoce
   con retraso, así que el monitoreo de desempeño va por detrás del de drift.

## Reentrenamiento

Disparadores definidos en `configs/config.yaml`:

- PSI > 0.25 en cualquier feature, o
- caída relativa de PR-AUC > 10% vs. la referencia, o
- cadencia programada (el deployment de Prefect corre semanalmente).

La promoción no es automática: un modelo nuevo debe superar al campeón en
`pr_auc` sobre test para tomar el alias
`champion` (ver `src/models/registry.py`).

---
*Generada automáticamente por el pipeline de entrenamiento.*
