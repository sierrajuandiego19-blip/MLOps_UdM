# Mensaje de entrega

Texto listo para copiar y enviar al profesor (y reutilizable para los
compañeros que hagan la evaluación por pares).

Antes de enviarlo, reemplazar:

- `[NOMBRES]` — tus nombres
- `[LINK DEL SPACE]` — la URL del Space, o borrar esa línea si no lo desplegaron

---

## Versión completa (correo al profesor)

> **Asunto:** Entrega proyecto final MLOps — Predicción de suscripción a depósito a plazo fijo
>
> Profesor,
>
> Adjunto la entrega del proyecto final. Lo hicimos [NOMBRES].
>
> **Repositorio:**
> https://github.com/sierrajuandiego19-blip/MLOps_UdM/tree/main/Project/bank-term-deposit-mlops
>
> **API desplegada:** [LINK DEL SPACE]
>
> ---
>
> **Qué construimos**
>
> Un sistema de ML end-to-end sobre el dataset UCI Bank Marketing (41.188
> contactos telefónicos reales de un banco portugués). El modelo predice qué
> clientes van a suscribir un depósito a plazo fijo, para priorizar las llamadas
> de un call center.
>
> Sobre el conjunto de test (8.238 clientes nunca usados para entrenar ni
> ajustar): PR-AUC de 0,4909 contra un piso de 0,113, y un lift de 3,31x en el
> top 20%. En términos del negocio, el modelo hace **78% menos llamadas** y
> cuadruplica la utilidad.
>
> ---
>
> **Cómo ejecutarlo**
>
> Requiere Python 3.11+ y uv. Tres comandos:
>
> ```
> git config --global core.longpaths true
> git clone https://github.com/sierrajuandiego19-blip/MLOps_UdM.git
> cd MLOps_UdM/Project/bank-term-deposit-mlops
> ./run.ps1 setup
> ```
>
> (En Linux o macOS: `make setup`.)
>
> **No hace falta reentrenar.** El repositorio incluye el historial completo de
> MLflow, los datos y el modelo, así que `setup` solo instala dependencias y
> reapunta las rutas de MLflow a la máquina donde se clone. Toma unos 3 minutos
> y al final debe imprimir:
>
> ```
> 153 passed
> campeon: bank-term-deposit-classifier v1 umbral 0.42
> ```
>
> Para levantar las interfaces: `./run.ps1 demo-up` y luego `./run.ps1 demo-open`,
> que abre MLflow (experimentos y registry), Prefect (los flows programados), la
> documentación interactiva de la API y el reporte de monitoreo.
>
> Si prefiere verificarlo desde cero en vez de confiar en el estado que viene en
> el repositorio, `./run.ps1 clean-all` y después `./run.ps1 bootstrap` reentrena
> todo. El entrenamiento es determinista, así que debe dar las mismas métricas.
>
> ---
>
> **Dónde está cada criterio del enunciado**
>
> El README tiene una tabla llamada **"Cumplimiento de las fases del proyecto"**
> que mapea cada punto de las fases 1 a 6 con el archivo donde está
> implementado. Es el camino más rápido para la rúbrica.
>
> Los documentos de apoyo:
>
> | Archivo | Contenido |
> |---|---|
> | `README.md` | Resultados, decisiones técnicas y la tabla de cumplimiento |
> | `docs/business_case.md` | Problema de negocio, supuestos, métricas de éxito, timeline y alcance (fase 1.1) |
> | `docs/monitoring.md` | Diseño del monitoreo en cuatro capas (fase 5) |
> | `docs/deployment.md` | Guías de deployment local, Docker y cloud (fases 4 y 6.3) |
> | `docs/api.md` | Referencia de los endpoints (fase 6.3) |
> | `models/model_card.md` | Uso previsto, desempeño y limitaciones del modelo (fase 2.3) |
> | `notebooks/` | EDA, baseline y análisis de experimentos, **ya ejecutados y con salidas**, se pueden leer sin correr nada |
>
> ---
>
> **Tres decisiones que vale la pena mirar**
>
> Son las que más nos costaron y las que explican los resultados:
>
> 1. **Descartamos `duration`, la variable más predictiva del dataset.** Solo se
>    conoce cuando la llamada ya terminó, pero el modelo tiene que decidir a
>    quién llamar antes de marcar. Medimos que usarla infla el PR-AUC un 41% con
>    cero valor en producción. Por eso nuestras métricas son más bajas que las de
>    otros análisis del mismo dataset: es intencional.
>
> 2. **No optimizamos accuracy.** Con 11,3% de positivos, responder siempre "no
>    suscribe" acierta el 88,7% y no sirve para nada. La métrica primaria es
>    PR-AUC, y al negocio le reportamos el lift sobre el top-k.
>
> 3. **El umbral de decisión se calibra maximizando utilidad**, no se fija en
>    0,5. Resultó 0,42, y no el 0,10 que predice la relación costo/beneficio,
>    porque `class_weight="balanced"` infla las probabilidades predichas. Está
>    documentado como limitación: los scores sirven para ordenar clientes, no
>    como probabilidades calibradas.
>
> ---
>
> **Lo que no alcanzamos a hacer**
>
> Preferimos declararlo nosotros:
>
> - **El Dockerfile no lo construimos en nuestra máquina** porque no teníamos
>   Docker instalado. Lo construye el job `docker` del CI en cada push, sobre un
>   runner limpio, y además comprueba que el contenedor arranque y responda en
>   `/health`. Se puede ver en la pestaña Actions del repositorio.
> - **No hicimos auditoría de equidad.** `age`, `marital` y `education` son
>   predictivas y a la vez atributos sensibles. Hacer el análisis a medias nos
>   pareció peor que declararlo pendiente. Está en la model card.
> - **Las probabilidades no están calibradas** (sin isotónica ni Platt). Sirven
>   para ordenar, no como estimaciones absolutas de riesgo.
> - **Los supuestos económicos** ($8 por llamada, $80 por suscripción) son
>   razonables para el ejercicio pero no son cifras auditadas: el dataset no trae
>   información económica de la operación. Están en un archivo de configuración,
>   así que cambiarlos recalcula todo el caso de negocio.
>
> Quedamos atentos a comentarios.
>
> [NOMBRES]

---

## Versión corta (plataforma del curso o chat)

> Entrega del proyecto final — [NOMBRES]
>
> **Repo:** https://github.com/sierrajuandiego19-blip/MLOps_UdM/tree/main/Project/bank-term-deposit-mlops
> **API desplegada:** [LINK DEL SPACE]
>
> Predicción de suscripción a depósito a plazo fijo sobre UCI Bank Marketing.
> Test: PR-AUC 0,4909 (piso 0,113), lift 3,31x en el top 20%, 78% menos llamadas
> que contactar a toda la base.
>
> **Para ejecutarlo** (Python 3.11+ y uv):
> ```
> git config --global core.longpaths true
> git clone https://github.com/sierrajuandiego19-blip/MLOps_UdM.git
> cd MLOps_UdM/Project/bank-term-deposit-mlops
> ./run.ps1 setup          # make setup en Linux/macOS
> ./run.ps1 demo-up        # levanta MLflow, Prefect y la API
> ./run.ps1 demo-open      # abre todo en el navegador
> ```
>
> No hace falta reentrenar: el repo incluye el historial de MLflow. `setup` debe
> terminar con `153 passed` y `campeon: ... v1 umbral 0.42`.
>
> El README tiene una tabla que mapea cada fase del enunciado con el archivo
> donde está implementada. Los notebooks vienen ejecutados y con salidas.
>
> Nota: el Dockerfile no lo construimos localmente (no teníamos Docker); lo
> construye y verifica el CI en cada push, visible en la pestaña Actions.
> Tampoco hicimos auditoría de equidad, y está declarado como limitación en la
> model card.

---

## Checklist antes de enviar

```powershell
./run.ps1 check        # lint + los 153 tests
git status             # que no quede nada sin subir
git push
```

Y abrir el repositorio en GitHub desde el navegador para confirmar que el README
se ve bien y que `configs/config.yaml` está ahí (el `.gitignore` del curso
ignora `*.yaml` de forma global y hubo que re-incluirlo explícitamente).
