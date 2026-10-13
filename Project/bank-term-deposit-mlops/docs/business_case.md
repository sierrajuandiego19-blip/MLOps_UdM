# Caso de negocio

## El problema

Un banco vende depósitos a plazo fijo por campañas telefónicas. El call center
llama a la base de clientes y ofrece el producto. Con los datos históricos
disponibles, **11,3% de los contactados suscribe**: casi nueve de cada diez
llamadas no producen nada.

Eso no es un problema de pérdida —las suscripciones pagan de sobra el costo de
la operación— sino de **capacidad desperdiciada**. El equipo tiene un número
fijo de horas-agente al día y las está gastando mayoritariamente en clientes que
iban a decir no.

## La pregunta que resuelve el modelo

> Con la capacidad de llamadas que tengo hoy, ¿a quién llamo primero?

No es "¿quién va a suscribir?" (nadie lo sabe) sino **"cómo ordeno la cola"**.
Esa diferencia determina todo el diseño: la salida del modelo es una
probabilidad usada como criterio de orden, no un veredicto.

## Supuestos económicos

Declarados en `configs/config.yaml` para que cualquiera los pueda cambiar y
recalcular sin tocar código:

| Parámetro | Valor | Comentario |
|---|---|---|
| Costo por llamada | $8 | Tiempo de agente, telefonía, overhead |
| Beneficio por suscripción | $80 | Margen esperado sobre el depósito |
| Capacidad del call center | 20% de la base | Fracción que se alcanza a contactar |

La asimetría 1:10 es la que gobierna el umbral de decisión: conviene aceptar
muchas llamadas fallidas antes que perder una suscripción.

> **Nota de honestidad.** Estos tres números son supuestos razonables para el
> ejercicio, no cifras auditadas de un banco real. El dataset de UCI no incluye
> información económica de la operación. Lo que el proyecto demuestra es el
> **método** —cómo atar la métrica técnica a la decisión de negocio— y esa
> estructura se mantiene al sustituir los supuestos por datos reales.

## Métricas de éxito

### Métrica primaria del modelo: PR-AUC

Con 11,3% de positivos, el accuracy es inservible: responder siempre "no
suscribe" acierta el 88,7% de las veces y tiene valor cero. El ROC-AUC es mejor
pero sigue siendo optimista, porque premia acertar en los negativos, que son
abundantes y fáciles.

**PR-AUC** (*average precision*) solo mejora si el modelo ordena bien la clase
minoritaria. Su piso es la tasa base: 0,113. Cualquier modelo tiene que superar
ese número para aportar algo.

### Métrica que se le reporta al negocio: lift sobre el top-k

El PR-AUC no significa nada para un gerente de call center. El **lift en el top
20%** sí: cuántas veces mejor que llamar al azar es llamar a los 20% mejor
puntuados.

Tiene además una ventaja técnica: depende solo del **orden** de los scores, no
de su valor absoluto, así que es inmune al problema de calibración descrito
abajo.

### Criterios de éxito definidos antes de entrenar

| Criterio | Umbral de aceptación | Resultado obtenido |
|---|---|---|
| PR-AUC en test supera la tasa base | > 0,113 | **0,491** (4,4x el piso) |
| ROC-AUC en test | > 0,75 | **0,818** |
| Lift en el top 20% | > 2,0x | **3,31x** |
| Ganancia incremental sobre llamar a todos | > 0 | **+$27.536** |
| El pipeline corre de punta a punta en una máquina limpia | sí/no | sí (verificado en CI) |

## Resultado sobre el conjunto de test

8.238 clientes nunca vistos durante el entrenamiento ni el ajuste:

| Concepto | Llamar a todos | Con el modelo | Diferencia |
|---|---|---|---|
| Llamadas realizadas | 8.238 | 1.816 | **−78%** |
| Suscripciones conseguidas | 928 | 630 | −32% |
| Costo de la operación | $65.904 | $14.528 | −$51.376 |
| Utilidad | $8.336 | **$35.872** | **+$27.536** |
| Costo por adquisición | $71,02 | **$23,06** | −68% |

### Cómo leer esta tabla

El modelo **consigue menos suscripciones** (630 contra 928) y aun así **cuadruplica
la utilidad**. Esa es la propuesta de valor completa: renunciar al 32% de las
suscripciones para eliminar el 78% de las llamadas.

Si el banco quisiera conservar las 928 suscripciones, la lectura correcta no es
"el modelo es peor", es "baja el umbral": la curva de utilidad de
`notebooks/02_baseline.ipynb` muestra exactamente cuánto cuesta cada punto de
recall adicional. El umbral es una palanca de negocio, y el proyecto la deja
expuesta y documentada en vez de enterrarla.

### El otro resultado: capacidad liberada

Con una capacidad del 20% de la base, el modelo alcanza al **66,3%** de todos
los suscriptores. Llamar al azar al mismo 20% alcanzaría al 20%. El call center
obtiene dos tercios del resultado total usando un quinto de su capacidad.

## Limitación importante: los scores no son probabilidades

El umbral óptimo resultó **0,42**, no el 0,10 que predice la economía del
problema (`costo / beneficio = 8/80`).

La causa es `class_weight="balanced"` / `scale_pos_weight`: esos parámetros
reponderan la clase minoritaria para que el modelo la aprenda y, como efecto
secundario, **inflan las probabilidades predichas** por encima de la frecuencia
real. Un score de 0,45 no significa "45% de probabilidad".

Consecuencias asumidas:

1. El umbral se busca **empíricamente** sobre validation
   (`find_optimal_threshold`), nunca por fórmula.
2. Los scores se usan para ordenar, no como estimaciones de riesgo.
3. Queda registrado como limitación en `models/model_card.md`. Una calibración
   isotónica o de Platt sería el siguiente paso si alguien necesitara leer los
   scores como probabilidades absolutas.

## Riesgo principal: el modelo depende del contexto macroeconómico

La importancia de features (`notebooks/03_experiments.ipynb`) muestra que el
modelo apoya la mayor parte de su decisión en variables **macroeconómicas**
—`nr.employed`, `emp.var.rate`, `cons.conf.idx`— y no en el perfil del cliente.

Dentro del periodo de entrenamiento (2008-2010, la crisis financiera) esa es
señal real. Fuera de ese régimen, deja de valer. Es el motivo por el que el
monitoreo de drift de este proyecto no es un adorno: vigila específicamente esas
variables, y el escenario de drift simulado es un cambio de régimen macro.

## Timeline y responsables

Proyecto académico individual; los "responsables" son los roles que cada fase
representaría en un equipo real.

| Fase | Entregable | Esfuerzo | Rol |
|---|---|---|---|
| 1. Planificación y EDA | `notebooks/01_eda.ipynb`, decisión sobre `duration` | 1 día | Data Scientist |
| 2. Experiment tracking | MLflow + Optuna + registry con alias | 1,5 días | ML Engineer |
| 3. Pipeline de entrenamiento | `training_flow` orquestado con Prefect | 1 día | ML Engineer |
| 4. Deployment | API FastAPI, Docker, scoring batch | 1,5 días | MLOps Engineer |
| 5. Monitoreo | Drift PSI/KS/chi², reportes, disparadores | 1 día | MLOps Engineer |
| 6. Testing y calidad | 135 tests, ruff/black, pre-commit, CI | 1 día | Todos |
| **Total** | | **~7 días** | |

## Alcance: qué entra y qué no

### MVP entregado

- Modelo entrenado con tracking completo y registry con promoción condicional
- Pipeline reproducible y orquestado, con schedules registrados
- Dos modos de deployment: API REST y scoring batch
- Monitoreo de drift de datos y de predicciones, con reporte y disparadores
- 135 tests, linting, CI

### Deliberadamente fuera de alcance

| Omitido | Por qué |
|---|---|
| Deployment en cloud ejecutado | Requiere cuenta y tarjeta; la configuración queda lista y documentada en `docs/deployment.md` |
| Calibración de probabilidades | No hace falta para ordenar; queda anotado como limitación |
| Auditoría de equidad por subgrupo | `age`, `marital` y `education` son predictivas y sensibles; una auditoría seria excede el alcance y hacerla a medias sería peor que declararla pendiente |
| Feature store | Sobre-ingeniería para un modelo con 19 features y un consumidor |
| A/B testing en producción | No hay producción |
| Reentrenamiento automático | El flow detecta y recomienda; la promoción exige superar al campeón. Reentrenar en automático por un falso positivo de drift es peor que una alerta que alguien revisa |

## Siguientes pasos si esto fuera a producción

1. **Auditoría de equidad** por edad, estado civil y nivel educativo, antes de
   cualquier despliegue real.
2. **Calibrar las probabilidades** si el negocio necesita leerlas como tales.
3. **Sustituir los supuestos económicos** por cifras reales de la operación y
   recalibrar el umbral.
4. **Conectar el ground truth** con el retraso real (el resultado de una llamada
   se conoce días después), para cerrar el ciclo de monitoreo de desempeño.
5. **Revisar la dependencia macro**: evaluar un modelo que use solo variables de
   cliente y comparar su degradación ante cambios de régimen.
