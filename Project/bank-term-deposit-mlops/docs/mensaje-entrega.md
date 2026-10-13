# Entrega — Proyecto Final MLOps

> **Antes de enviar:** reemplazar `[NOMBRES]` por los nombres de los autores.
>
> Para enviarlo por **correo**, usar [`correo-entrega.txt`](correo-entrega.txt):
> es el mismo contenido en texto plano, porque Gmail y Outlook no renderizan
> las tablas ni los bloques de código de Markdown.

---

**Asunto:** Entrega proyecto final MLOps — Predicción de suscripción a depósito a plazo fijo

Profesor,

Le comparto la entrega de nuestro proyecto final. Lo hicimos [NOMBRES].

| | |
|---|---|
| **Repositorio** | <https://github.com/sierrajuandiego19-blip/MLOps_UdM/tree/main/Project/bank-term-deposit-mlops> |
| **API desplegada** | <https://bank-term-deposit-api.onrender.com/docs> |

Una nota sobre la API: está en el plan gratuito de Render, que suspende el
servicio tras 15 minutos sin tráfico. Si la primera carga tarda unos 50
segundos, es el arranque en frío y no un error. A partir de ahí responde al
instante.

---

## Qué construimos

Un sistema de machine learning end-to-end sobre el dataset
[UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing),
que recoge 41.188 contactos telefónicos reales de un banco portugués. El modelo
predice qué clientes van a suscribir un depósito a plazo fijo, con el fin de
priorizar las llamadas de un call center.

Sobre el conjunto de test —8.238 clientes que nunca se usaron para entrenar ni
para ajustar nada— estos son los resultados:

| Métrica | Valor |
|---|---|
| PR-AUC (primaria) | **0,4909** frente a un piso de 0,113 |
| ROC-AUC | 0,8183 |
| Lift en el top 20% | **3,31x** |
| Suscriptores alcanzados llamando al 20% de la base | 66,3% |

Traducido al negocio:

| Concepto | Llamar a todos | Con el modelo |
|---|---|---|
| Llamadas | 8.238 | **1.816** (−78%) |
| Suscripciones | 928 | 630 (−32%) |
| Utilidad | $8.336 | **$35.872** |
| Costo por adquisición | $71,02 | **$23,06** |

El modelo consigue **menos** suscripciones y **cuadruplica** la utilidad:
renuncia al 32% de las suscripciones para eliminar el 78% de las llamadas.

---

## Cómo ejecutarlo

Requiere Python 3.11 o superior y [uv](https://docs.astral.sh/uv/).

```
git config --global core.longpaths true
git clone https://github.com/sierrajuandiego19-blip/MLOps_UdM.git
cd MLOps_UdM/Project/bank-term-deposit-mlops
./run.ps1 setup
```

En Linux o macOS, el último comando es `make setup`.

**No hace falta reentrenar nada.** El repositorio incluye el historial completo
de MLflow, los datos y el modelo ya entrenado, así que `setup` solo instala las
dependencias y reapunta las rutas de MLflow a la máquina donde se clone. Toma
unos tres minutos y debe terminar imprimiendo:

```
173 passed
campeon: bank-term-deposit-classifier v1 umbral 0.42
```

Para ver las interfaces gráficas, `./run.ps1 demo-up` y luego
`./run.ps1 demo-open` abren MLflow con los experimentos y el registry, Prefect
con los flows programados, la documentación interactiva de la API y el reporte
de monitoreo.

Si prefiere verificarlo desde cero en vez de confiar en el estado que viene en
el repositorio, `./run.ps1 clean-all` seguido de `./run.ps1 bootstrap`
reentrena todo. El entrenamiento es determinista, así que debe arrojar las
mismas métricas.

---

## Dónde está cada criterio del enunciado

El README incluye una tabla llamada **"Cumplimiento de las fases del
proyecto"** que mapea cada punto de las fases 1 a 6 con el archivo donde está
implementado. Es el camino más rápido para recorrer la rúbrica.

Los documentos de apoyo:

| Archivo | Contenido |
|---|---|
| `docs/business_case.md` | Problema de negocio, supuestos, métricas de éxito, timeline y alcance (fase 1.1) |
| `docs/monitoring.md` | Diseño del monitoreo en cuatro capas (fase 5) |
| `docs/deployment.md` | Guías de deployment local, Docker y cloud (fase 4) |
| `docs/api.md` | Referencia de los endpoints (fase 6.3) |
| `models/model_card.md` | Uso previsto, desempeño y limitaciones (fase 2.3) |

Los tres notebooks vienen **ejecutados y con sus salidas guardadas**, de modo
que se pueden leer sin ejecutar nada.

---

## Tres decisiones que vale la pena mirar

**1. Descartamos `duration`, la variable más predictiva del dataset.** Solo se
conoce cuando la llamada ya terminó, pero el modelo tiene que decidir a quién
llamar antes de marcar. Medimos que usarla infla el PR-AUC un 41% con cero
valor en producción. Por eso nuestras métricas son más bajas que las de otros
análisis del mismo dataset: es intencional.

**2. No optimizamos accuracy.** Con 11,3% de positivos, un modelo que responda
siempre "no suscribe" acierta el 88,7% y no sirve para nada. Usamos PR-AUC como
métrica primaria, y al negocio le reportamos el lift sobre el top-k.

**3. El umbral de decisión se calibra maximizando la utilidad**, no se fija en
0,5. Resultó 0,42 y no el 0,10 que predice la relación costo/beneficio, porque
`class_weight="balanced"` infla las probabilidades predichas. Lo dejamos
documentado como limitación: los scores sirven para ordenar clientes, no como
probabilidades calibradas.

---

## Lo que no alcanzamos a hacer

Preferimos declararlo nosotros:

- **El Dockerfile no lo construimos en nuestra máquina** porque no teníamos
  Docker instalado. Lo construye el job `docker` del CI en cada push, sobre un
  runner limpio, y comprueba que el contenedor arranque y responda en
  `/health`. Se puede ver en la pestaña Actions del repositorio.
- **No hicimos auditoría de equidad.** Las variables `age`, `marital` y
  `education` son predictivas y a la vez atributos sensibles. Hacer el análisis
  a medias nos pareció peor que declararlo pendiente.
- **Las probabilidades no están calibradas.** Sirven para ordenar clientes, no
  como estimaciones absolutas de riesgo.
- **Los supuestos económicos** ($8 por llamada, $80 por suscripción) son
  razonables para el ejercicio pero no son cifras auditadas: el dataset no trae
  información económica de la operación. Están en un archivo de configuración,
  así que cambiarlos recalcula todo el caso de negocio.

Quedamos atentos a cualquier comentario.

Un saludo,
[NOMBRES]

---

## Versión corta

Para la plataforma del curso, un chat, o si el correo debe ser breve.

> Entrega del proyecto final MLOps — [NOMBRES]
>
> **Repositorio:** <https://github.com/sierrajuandiego19-blip/MLOps_UdM/tree/main/Project/bank-term-deposit-mlops>
> **API desplegada:** <https://bank-term-deposit-api.onrender.com/docs>
> *(plan gratuito: la primera carga puede tardar ~50 s mientras el servicio despierta)*
>
> Predicción de suscripción a depósito a plazo fijo sobre el dataset UCI Bank
> Marketing. En test: PR-AUC 0,4909 frente a un piso de 0,113, lift de 3,31x en
> el top 20%, y un 78% menos de llamadas que contactar a toda la base.
>
> Para ejecutarlo (Python 3.11+ y uv):
>
> ```
> git config --global core.longpaths true
> git clone https://github.com/sierrajuandiego19-blip/MLOps_UdM.git
> cd MLOps_UdM/Project/bank-term-deposit-mlops
> ./run.ps1 setup
> ```
>
> No hace falta reentrenar: el repositorio incluye el historial de MLflow. El
> comando debe terminar con `173 passed` y `campeon: ... v1 umbral 0.42`.
>
> El README tiene una tabla que mapea cada fase del enunciado con el archivo
> donde está implementada. Los notebooks vienen ejecutados y con salidas.
>
> Nota: el Dockerfile no lo construimos localmente porque no teníamos Docker;
> lo construye y verifica el CI en cada push. Tampoco hicimos auditoría de
> equidad, y está declarado como limitación en la model card.
