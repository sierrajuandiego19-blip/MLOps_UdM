# Guía de exposición — 2 personas, 25 minutos

Guion repartido, con lo que hay que levantar antes y qué decir en cada pantalla.

---

## 1. Checklist antes de empezar

### Levantar los servicios (4 terminales, en este orden)

Las cuatro arrancan desde la carpeta del proyecto:

```powershell
cd "C:\Users\gjaim\OneDrive - The University of Melbourne\Documents\MLOPS 2026\Project\bank-term-deposit-mlops"
```

```powershell
# Terminal 1 — MLflow del PROYECTO (ojo: puerto 5001, no 5000)
# La ruta del backend tiene que ser ABSOLUTA: con una relativa, MLflow abre
# una base de datos distinta segun desde donde se lance.
uv run mlflow server --backend-store-uri "sqlite:///$((Get-Location).Path -replace '\\','/')/mlflow.db" --host 127.0.0.1 --port 5001

# Terminal 2 — Servidor de Prefect
uv run prefect server start

# Terminal 3 — Worker de Prefect (ejecuta los flows programados)
uv run prefect worker start --pool local-pool

# Terminal 4 — La API
uv run uvicorn src.api.main:app --port 8000
```

O con el atajo, que ya trae la ruta resuelta:

```powershell
./run.ps1 demo-up     # explica que abrir en cada terminal
```

> **Por qué MLflow va en 5001.** En esta máquina el puerto 5000 ya lo ocupa un
> MLflow apuntando al `mlflow.db` del repositorio del curso, donde no están
> estos experimentos. Si abres 5000 por costumbre, vas a ver un MLflow vacío en
> plena exposición.

### Verificar que todo responde

```powershell
curl http://127.0.0.1:5001/health      # MLflow
curl http://127.0.0.1:4200/api/health  # Prefect
curl http://127.0.0.1:8000/health      # API -> debe decir "healthy"
```

### Pestañas del navegador, en orden de uso

| # | URL | Cuándo se usa |
|---|---|---|
| 1 | <http://127.0.0.1:5001> | Bloque 3 (experimentos) |
| 2 | <http://127.0.0.1:5001/#/models> | Bloque 3 (registry) |
| 3 | <http://127.0.0.1:4200/runs> | Bloque 4 (orquestación) |
| 4 | <http://127.0.0.1:4200/deployments> | Bloque 4 (schedules) |
| 5 | <http://127.0.0.1:8000/docs> | Bloque 5 (API en vivo) |
| 6 | `reports/drift_report_latest.html` | Bloque 6 (monitoreo) |
| 7 | `notebooks/01_eda.ipynb` en VSCode | Bloque 2 (EDA) |
| 8 | La API en Render | Bloque 5 (cloud) |

### Reparto

| | Persona A | Persona B |
|---|---|---|
| **Rol** | El problema y la ciencia de datos | La ingeniería y la operación |
| **Bloques** | 1, 2, 3 (12 min) | 4, 5, 6, 7 (13 min) |

---

## 2. Guion

### Bloque 1 — El problema (A, 3 min)

**Sin pantalla, o con la tabla del README.**

> "Un banco vende depósitos a plazo fijo llamando por teléfono. De cada 100
> llamadas, **solo 11 terminan en una suscripción**. El problema no es que
> pierdan dinero —las suscripciones pagan la operación— es que el call center
> está gastando casi toda su capacidad en gente que iba a decir no.
>
> Así que la pregunta que resolvimos **no es** '¿quién va a suscribir?'. Nadie
> lo sabe. Es **'¿a quién llamo primero?'**. Eso cambia todo el diseño: lo que
> entregamos es una lista ordenada, no un veredicto."

**Mostrar la tabla de resultados:**

| | Llamar a todos | Con el modelo |
|---|---|---|
| Llamadas | 8.238 | **1.816** (−78%) |
| Suscripciones | 928 | 630 (−32%) |
| Utilidad | $8.336 | **$35.872** |

> "Fíjense en algo contraintuitivo: el modelo consigue **menos**
> suscripciones y **cuadruplica** la utilidad. Renuncia al 32% de las
> suscripciones para eliminar el 78% de las llamadas. Y si el banco quiere
> recuperar ese volumen, no hay que reentrenar nada: se baja el umbral, y
> tenemos la curva que dice exactamente cuánto cuesta cada punto de recall."

---

### Bloque 2 — Decisiones del EDA (A, 4 min)

**Pantalla: `notebooks/01_eda.ipynb`, ya ejecutado.**

Tres decisiones, en este orden. Son el corazón técnico de la presentación.

**a) Tiramos a la basura la variable más predictiva.**

> "`duration` es la duración de la llamada, y es con diferencia la variable más
> predictiva del dataset. La eliminamos.
>
> El motivo: solo se conoce **cuando la llamada ya terminó**. Pero el modelo
> tiene que decidir a quién llamar **antes** de marcar. Usarla es hacer trampa
> con el futuro.
>
> Lo medimos: usarla **infla el PR-AUC un 41%**. Son 41 puntos de rendimiento
> que se verían en la diapositiva y se evaporarían en producción."

*(Mostrar la celda con la comparación CON/SIN duration.)*

**b) No usamos accuracy.**

> "Con 89% de negativos, un modelo que responda siempre 'no suscribe' acierta
> el **88,7%**. Y no sirve para nada: no encuentra un solo cliente.
>
> Usamos **PR-AUC**, que solo mejora si el modelo ordena bien a la minoría. Su
> piso es 0,113. Nuestro modelo da 0,49: cuatro veces el piso."

**c) `pdays = 999` no son 999 días.**

> "El 96% de los clientes tiene `pdays = 999`. No significa '999 días', es un
> centinela que significa **'nunca lo contactamos'**. Dejarlo como número mete
> un outlier falso en el escalado.
>
> Lo partimos en dos variables honestas. Y valió la pena: la tasa de
> suscripción pasa de **9,3% entre los nunca contactados a 63,8% entre los que
> sí**. Es la señal individual más fuerte que encontramos."

---

### Bloque 3 — Experimentos y registry (A, 5 min)

**Pantalla: MLflow en <http://127.0.0.1:5001>**

> "Todo lo que entrenamos está acá. No hay un notebook con el modelo bueno
> perdido en una carpeta."

1. **Lista de runs** → mostrar los 5 candidatos ordenados por `valid_pr_auc`.

   > "Fuimos de lo tonto a lo complejo: un clasificador que siempre dice 'no',
   > uno aleatorio, regresión logística, Random Forest y XGBoost. Sin ese piso,
   > un 0,49 no significa nada."

2. **Un run cualquiera** → pestaña de parámetros y métricas.

   > "Cada run guarda hiperparámetros, métricas técnicas, métricas de negocio
   > en dólares, y el modelo serializado con su firma."

3. **El run `xgboost-optuna-search`** → mostrar los runs anidados.

   > "20 trials de Optuna, cada uno como run anidado. Se puede auditar la
   > búsqueda completa, no solo el ganador."

4. **Confesión honesta** (importante, da credibilidad):

   > "Y acá algo que no esperábamos: XGBoost le gana a la regresión logística
   > por **un dígito bajo**, no por órdenes de magnitud. El tuning aportó poco.
   > La mayor parte del valor vino de las tres decisiones del bloque anterior.
   > Quien salte directo a XGBoost sin hacer ese trabajo obtiene un modelo peor."

5. **Pestaña Models** → <http://127.0.0.1:5001/#/models>

   > "Acá está el registry. Vean: hay **dos versiones**. La v1 tiene el alias
   > `champion`, la v2 tiene `challenger`.
   >
   > Eso pasó solo. Reentrenamos, la v2 empató con la v1, y **el sistema se
   > negó a promoverla**: nuestra regla es que un candidato solo toma el alias
   > `champion` si supera al campeón por un margen mínimo. Si no, queda
   > registrado y disponible para comparar, pero no sirve tráfico.
   >
   > Un modelo nuevo no se promueve por ser nuevo."

---

### Bloque 4 — Orquestación (B, 4 min)

**Pantalla: Prefect en <http://127.0.0.1:4200/deployments>**

> "Nada de esto se ejecuta a mano. Hay tres pipelines programados."

| Deployment | Cuándo | Por qué esa frecuencia |
|---|---|---|
| `training-weekly` | Lunes 02:00 | Reentrenar más seguido no aporta: los datos macro se publican mensualmente |
| `scoring-daily` | Diario 06:00 | La lista de llamadas tiene que estar lista antes del turno |
| `monitoring-daily` | Diario 07:00 | Después del scoring, sobre datos frescos |

**Luego ir a <http://127.0.0.1:4200/runs>** y mostrar el historial de runs
completados.

**Demo en vivo** (toma ~40 segundos, vale la pena):

```powershell
uv run prefect deployment run 'batch-scoring-pipeline/scoring-daily'
```

> "Lo disparo ahora y lo vemos pasar por los pasos."

Abrir el run en la UI mientras corre → mostrar el grafo de tasks y, al
terminar, la pestaña **Artifacts**.

> "El flow publica un resumen en la propia UI: cuántos clientes puntuó, cuántos
> recomienda llamar, con qué versión del modelo. Quien revise un run programado
> a las 6 de la mañana no tiene que leer logs."

Mostrar el CSV generado en `data/predictions/`:

> "Y esto es lo que recibe el equipo de campañas: un CSV ordenado por
> prioridad. La columna `call_priority` es la cola de llamadas."

---

### Bloque 5 — Deployment (B, 5 min)

**a) API en vivo — <http://127.0.0.1:8000/docs>**

> "El modelo se sirve de dos formas. Batch, que acabamos de ver. Y una API REST
> para decisión en tiempo real."

1. `GET /health` → *Try it out*

   > "Reporta el estado y **qué versión** está sirviendo."

2. `POST /predict` → usar el ejemplo que ya viene cargado, cambiar `poutcome` a
   `"success"` y `age` a `62`.

   > "Score 0,62, banda alta, recomendación: llamar."

3. **Mostrar la validación** → cambiar `job` a `"astronauta"` y ejecutar.

   > "422, y dice exactamente qué campo está mal. El contrato es estricto a
   > propósito: las categorías son cerradas, los rangos están acotados, y un
   > campo de más también es error. Casi siempre es un nombre mal escrito, y
   > preferimos fallar fuerte que puntuar ignorando una variable en silencio."

4. `GET /model-info`

   > "Quien consume la API puede verificar con qué modelo le están respondiendo
   > y con qué métricas medidas, sin preguntarle a nadie."

**b) Docker**

```powershell
docker compose up --build
```

> "Build en dos etapas para que las herramientas de compilación no lleguen a la
> imagen final, usuario sin privilegios, y un health check que distingue
> 'proceso vivo' de 'modelo cargado'. El compose levanta MLflow y la API
> juntos."

**c) Cloud — la API en Render**

> "Y está desplegado. Esta es la misma API corriendo en Render."

*(Abrir la URL de Render y hacer una predicción desde `/docs` ahí.)*

> "Lo que hizo esto fácil fue una decisión de diseño: exportamos el campeón a
> una copia autocontenida de **1 MB** que viaja dentro de la imagen. El
> contenedor no necesita un servidor de MLflow ni una base de datos.
>
> Y verificamos lo que de verdad importa: que la copia predice **exactamente**
> lo mismo que el registry. Hay un test que lo comprueba, porque si esa copia
> se desincroniza, producción sirve un modelo distinto del que medimos."

---

### Bloque 6 — Monitoreo (B, 4 min)

**Pantalla: `reports/drift_report_latest.html`**

> "El enunciado pedía *proponer ideas* de monitoreo. Lo implementamos, porque
> en este modelo no es opcional."

**Primero, el motivo:**

> "Cuando miramos la importancia de features encontramos algo incómodo: el
> modelo apoya la mayor parte de su decisión en variables **macroeconómicas**
> —nivel de empleo, variación del empleo, confianza del consumidor— no en el
> perfil del cliente.
>
> Y se entrenó con datos de 2008 a 2010, la crisis financiera. Aprendió las
> correlaciones de ese régimen. En otro contexto económico, esas relaciones
> pueden invertirse, y el modelo seguiría prediciendo con total confianza a
> partir de un mundo que ya no existe.
>
> El monitoreo es la única defensa contra eso."

**Luego, cómo se mide:**

> "Usamos **PSI** como métrica que decide: es el estándar en banca y tiene
> umbrales interpretables —bajo 0,1 estable, sobre 0,25 alerta. Reportamos
> también Kolmogorov-Smirnov y chi-cuadrado, pero **los p-valores no deciden**:
> con 3.000 filas detectan diferencias significativas e irrelevantes. Esa es la
> diferencia entre un sistema de alertas que alguien lee y uno que se silencia
> en la primera semana."

**Demo en vivo — los dos escenarios:**

```powershell
uv run python -m src.pipelines.monitoring_flow
```

| | Sin drift (control) | Con drift macro |
|---|---|---|
| Features en alerta | **0** de 19 | **6** de 19 |
| PSI máximo | 0,005 | 11,83 |
| PSI del score del modelo | 0,004 | **1,78** |
| Score medio | 0,384 → 0,388 | 0,384 → **0,557** |
| Veredicto | no reentrenar | **reentrenar** |

> "Un detector de drift que nunca se probó contra drift real es un adorno. Así
> que construimos un simulador que inyecta un cambio de régimen
> macroeconómico, y corremos **dos** escenarios: uno sin drift y uno con drift.
>
> Las dos propiedades que importan: **no inventa drift** cuando no lo hay, y
> **lo detecta** cuando lo hay. Y están verificadas en CI como aserciones."

**Cierre del bloque — la decisión de diseño:**

> "Una última cosa: el flow **detecta y recomienda, no reentrena solo**. Un
> reentrenamiento automático disparado por un falso positivo —un ETL roto, por
> ejemplo— produce un modelo peor y lo despliega sin que nadie mire. Preferimos
> una alerta que alguien revisa."

---

### Bloque 7 — Calidad y cierre (B, 2 min)

```powershell
uv run pytest
```

> "173 tests, corren en 10 segundos porque usan datos sintéticos: no descargan
> nada ni necesitan un modelo entrenado."

**Mencionar 3 tests concretos, no el número:**

> "Y no son tests de relleno. Por ejemplo: uno verifica que una predicción por
> la API y la misma por batch den el **mismo** número —ese es el bug clásico de
> training/serving skew. Otro, que el orden de las claves de un JSON no cambie
> la predicción. Otro, que una categoría nueva que el modelo nunca vio no tumbe
> el servicio."

**Cobertura, con honestidad:**

> "La cobertura es 52%. Los módulos de lógica están entre 86 y 100%. Lo que
> está en 0% son los flows de Prefect, y es deliberado: probarlos con unit
> tests exigiría mockear medio Prefect y no garantizaría nada. Los cubre el CI,
> que los ejecuta de verdad en una máquina limpia."

**CI:**

> "Cuatro jobs: lint, tests, build de la imagen, y un smoke test que entrena el
> modelo desde cero. Y ese smoke test **afirma**, no solo ejecuta: falla si el
> ROC-AUC baja de 0,70, si el monitoreo inventa drift, o si no lo detecta. Un
> pipeline que 'corre sin errores' puede estar produciendo basura."

**Cierre:**

> "En resumen: un modelo que cuadruplica la utilidad del call center, con
> experiment tracking, orquestación programada, dos modos de deployment, uno de
> ellos en cloud, monitoreo verificado y CI. Y las limitaciones documentadas,
> empezando por la dependencia macroeconómica."

---

## 3. Los cinco momentos que hay que vender

Si hay que recortar, esto es lo que **no** se saca:

1. **Tiramos `duration`** y medimos que infla el PR-AUC un 41%. Demuestra que
   entendemos leakage.
2. **El modelo consigue menos suscripciones y más utilidad.** Demuestra que
   entendemos el negocio, no solo la métrica.
3. **El registry se negó a promover la v2.** Demuestra gobernanza, y pasó solo.
4. **El monitoreo corre dos escenarios**, control y drift. Demuestra que lo
   verificamos, no solo lo escribimos.
5. **El modelo depende de variables macro**, y lo decimos nosotros antes de que
   lo pregunten. Demuestra honestidad técnica.

---

## 4. Preguntas probables

**"¿Por qué PR-AUC de 0,49 y no más alto?"**
> Porque quitamos `duration`. Con ella subiría un 41% y el modelo sería
> inservible en producción. El piso real de este problema es 0,113.

**"¿Por qué el umbral es 0,42 y no 0,5?"**
> Se calibró maximizando utilidad en dólares, no por defecto. Y hay un detalle
> interesante: la economía del problema (8 dólares por llamada, 80 por
> suscripción) predice un umbral de 0,10, pero el óptimo empírico es 0,42. La
> causa es que `class_weight="balanced"` infla las probabilidades predichas, así
> que **los scores no están calibrados**. Por eso el umbral se busca
> empíricamente y la métrica que le reportamos al negocio es el lift, que solo
> depende del orden.

**"¿Por qué no usaron Evidently para el drift?"**
> Por una dependencia menos y control de los umbrales. PSI, KS y chi-cuadrado
> son 130 líneas y las entendemos completas. Evidently sería la elección
> correcta en un equipo grande con varios modelos.

**"¿Reentrena automáticamente?"**
> Detecta y recomienda. La promoción tiene además un segundo filtro: el
> candidato solo toma el alias `champion` si le gana al actual. Lo vieron en
> vivo con la v2.

**"¿Y el sesgo? Usan edad y estado civil."**
> Son predictivas y a la vez atributos sensibles. **No** hicimos auditoría de
> equidad y está declarado como limitación en la model card. Hacerla a medias
> sería peor que decir que falta. Es el primer paso antes de producción real.

**"¿Los $8 y $80 de dónde salen?"**
> Son supuestos razonables para el ejercicio, no cifras auditadas: el dataset
> no trae información económica de la operación. Están en un archivo de
> configuración, así que cambiarlos recalcula todo. Lo que demuestra el proyecto
> es el método de atar la métrica técnica a la decisión de negocio.

**"¿Por qué XGBoost si apenas le gana a la logística?"**
> Buena pregunta, y es parte de nuestra conclusión: aportó poco. Lo elegimos
> porque rinde algo mejor y entrena rápido. Pero el hallazgo es que el valor
> estuvo en el feature engineering y el umbral, no en el algoritmo.

---

## 5. Plan B

| Si falla | Qué hacer |
|---|---|
| MLflow no abre | Verificar que es el puerto **5001**, no 5000 |
| El MLflow de 5001 sale vacío | Está apuntando al `mlflow.db` equivocado; relanzar con la ruta absoluta del proyecto |
| La API dice `degraded` | No hay modelo: `uv run python -m src.pipelines.training_flow` |
| El run de Prefect queda en `Pending` | No hay worker: `uv run prefect worker start --pool local-pool` |
| El run queda en `CRASHED` | Re-registrar: `uv run python deploy_flows.py` |
| Render está dormido o caído | Abrir la URL 2 min antes (el plan gratuito duerme a los 15 min). Si falla, mostrar la API local en `:8000/docs`: mismo código, misma predicción |
| No hay internet | Todo funciona local: el dataset está en caché en `data/raw/` |
| Docker no arranca | Mostrar el Dockerfile y el job `docker` del CI. Decirlo, no disimularlo |

### Lo que conviene decir sin que lo pregunten

Dos cosas, porque suenan mucho mejor dichas por ustedes:

- Si Docker no está instalado en la máquina de la presentación: *"el build de la
  imagen lo verifica el CI, no en nuestro portátil"*.
- La cobertura del 52% y por qué los flows están en 0%.
