# Cómo compartir y entregar el proyecto

Dos destinatarios distintos, con necesidades distintas:

| Para | Qué necesita | Cómo |
|---|---|---|
| **Tu compañero** | Reproducir el proyecto **tal cual** y poder exponerlo | Clonar el repositorio + `./run.ps1 setup` |
| **El profesor / pares** | Leer el README y ejecutar el código | Link al repositorio de GitHub |

---

## 1. Compartirlo con tu compañero

### Por qué el repositorio trae el estado completo

Normalmente `mlruns/`, `mlflow.db` y `data/` **no** se versionan: son artefactos
generados. Aquí sí se versionan, a propósito.

El motivo es la exposición. Si tu compañero clona un repositorio solo con
código, tiene que reentrenar 12 minutos antes de poder abrir MLflow y ver algo.
Y los IDs de run serían distintos de los tuyos, así que practicarían sobre
estados diferentes. Versionando el estado (~30 MB), los dos ven **exactamente**
lo mismo: los mismos 9 runs, el mismo `champion→v1` / `challenger→v2`, el mismo
reporte de drift.

El coste es tener binarios generados en git. Para un proyecto que se evalúa por
pares y se expone en vivo, vale la pena.

### El detalle que rompe todo si se ignora

**MLflow guarda rutas absolutas.** En `mlflow.db` y en los archivos `MLmodel`
queda escrito algo como:

```
file:///C:/Users/juan/.../bank-term-deposit-mlops/mlruns/1/models/m-abc/artifacts
```

En la máquina de tu compañero esa ruta no existe. MLflow abre la interfaz, se ven
los runs y las métricas... y **ningún artefacto carga**. El registry queda roto y
el fallo es silencioso hasta que alguien intenta usar un modelo.

Por eso hay un script que reapunta todas esas rutas. Son 128 filas en la base de
datos y 10 archivos `MLmodel`.

### Pasos para tu compañero

```powershell
# 0. Una sola vez: permitir rutas largas (mlruns/ anida bastante)
git config --global core.longpaths true

# 1. Clonar. Conviene una carpeta poco profunda, no dentro de OneDrive.
cd C:\dev
git clone https://github.com/sierrajuandiego19-blip/MLOps_UdM.git
cd MLOps_UdM\Project\bank-term-deposit-mlops

# 2. Setup: instala, reapunta MLflow y verifica
./run.ps1 setup
```

En Linux o macOS: `make setup`.

Eso hace cuatro cosas: `uv sync`, ejecuta `scripts/relocate_mlflow.py`, registra
los flows de Prefect y verifica con los 173 tests + la carga del campeón.

**Qué tiene que ver al final:**

```
173 passed
campeon: bank-term-deposit-classifier v1 umbral 0.42
```

Si ve eso, tiene el proyecto idéntico al tuyo.

### Levantar la demo

```powershell
./run.ps1 demo-up      # MLflow, Prefect, worker y API, cada uno en su ventana
./run.ps1 demo-open    # abre las 5 pestañas + el reporte de drift
./run.ps1 demo-check   # comprueba que los 3 servicios respondan
```

Guion de la exposición: [`presentacion.md`](presentacion.md).

### Requisitos previos en su máquina

| Necesita | Por qué | Dónde |
|---|---|---|
| **Python 3.11+** | — | <https://www.python.org/downloads/> |
| **uv** | Gestiona el entorno y respeta `uv.lock` | `winget install astral-sh.uv` o <https://docs.astral.sh/uv/> |
| **Git** | Clonar | <https://git-scm.com/> |


No necesita cuenta de MLflow, de Prefect ni de nada: todo corre local. Tampoco
necesita internet después del `uv sync`, porque el dataset viene en caché en
`data/raw/`.

### Si algo le falla

| Síntoma | Causa | Solución |
|---|---|---|
| `git clone` falla con **"Filename too long"** | Windows limita las rutas a 260 caracteres, y `mlruns/1/models/m-<hash>/artifacts/...` añade 87 | `git config --global core.longpaths true` y volver a clonar, o clonar en una carpeta menos profunda (`C:\dev\`, por ejemplo) |
| MLflow muestra los runs pero los modelos no cargan | No ejecutó la relocalización | `uv run python scripts/relocate_mlflow.py` |
| `RuntimeError: No hay un modelo con alias 'champion'` | Ídem | Ídem |
| MLflow aparece vacío | Abrió el puerto 5000 en vez del **5001** | Ver la nota de `presentacion.md` |
| `uv sync` falla con "not a member of the workspace" | — | Ya resuelto: el proyecto declara su propio workspace |
| Quiere empezar de cero | — | `./run.ps1 clean-all` y luego `./run.ps1 bootstrap` (reentrena, ~12 min) |

### Alternativa: si prefieren no usar git

Comprimir la carpeta **excluyendo `.venv`** (son 936 MB de dependencias que se
reinstalan solas):

```powershell
cd "...\Project"
Compress-Archive -Path bank-term-deposit-mlops -DestinationPath proyecto-mlops.zip `
  -Exclude "*.venv*"
```

> `Compress-Archive` no excluye carpetas de forma fiable. Más seguro: copiar la
> carpeta, borrar `.venv` y `build` de la copia, y comprimir esa copia. El ZIP
> queda en ~35 MB.

Tras descomprimir, los pasos son los mismos: `./run.ps1 setup`.

---

## 2. Entregarlo al profesor

El enunciado dice que la evaluación es por pares y que el proceso es *"revisar
el README"* y *"ejecutar el código siguiendo las instrucciones"*. Es decir: lo
que hay que entregar es **el link al repositorio**.

### Antes de entregar, comprobar esto

```powershell
./run.ps1 check        # lint + los 173 tests
./run.ps1 demo-check   # los 3 servicios responden
```

Y revisar que estos cinco archivos estén en GitHub, porque son los que un
revisor abre primero:

- `README.md` — resultados, decisiones técnicas, cumplimiento de las fases
- `docs/business_case.md` — problema, supuestos y métricas de éxito
- `docs/monitoring.md` — diseño del monitoreo
- `docs/deployment.md` — guía de deployment
- `models/model_card.md` — uso previsto y limitaciones

### Qué entregar

**1. El link al repositorio**, apuntando directo a la carpeta del proyecto:

```
https://github.com/sierrajuandiego19-blip/MLOps_UdM/tree/main/Project/bank-term-deposit-mlops
```

**2. El link de la API desplegada** en Render, una vez creada:

```
https://bank-term-deposit-api.onrender.com/docs
```

**3. Una nota de tres líneas** sobre cómo ejecutarlo, porque es lo primero que
un revisor busca:

> Proyecto en `Project/bank-term-deposit-mlops/`.
> Para ejecutarlo: `uv sync --all-groups` y luego `./run.ps1 setup`
> (o `make setup`). El repositorio incluye el historial de MLflow, así que
> no hace falta reentrenar. API desplegada en: `<link de Render>`.

### Lo que conviene declarar sin que lo pregunten

El README ya lo dice, pero vale repetirlo en la entrega: el deployment en cloud
está hecho, y el **Dockerfile no se construyó en la máquina de desarrollo**
(Docker no estaba instalado; lo verifica el job `docker` del CI). Decirlo
directamente da más credibilidad que esperar a que lo descubran.

### Si pide un archivo en vez de un link

Algunos profesores piden un ZIP por la plataforma del curso. Mismo
procedimiento que la alternativa de arriba: copiar la carpeta, borrar `.venv` y
`build`, comprimir. Quedan ~35 MB, y conviene avisar en el correo que el
historial de MLflow viene incluido y que hay que ejecutar
`scripts/relocate_mlflow.py` después de descomprimir.

---

## 3. Mantener los dos repositorios sincronizados

Si los dos van a tocar código antes de la exposición:

```powershell
# Antes de empezar a trabajar
git pull

# Al terminar
git add Project/bank-term-deposit-mlops
git commit -m "describe el cambio"
git push
```

**Cuidado con un detalle:** `mlflow.db` es un binario y está versionado. Si los
dos entrenan y hacen push, git no sabe fusionarlo y van a tener un conflicto
imposible de resolver a mano.

Acuerdos que lo evitan:

1. **Solo una persona reentrena** y hace push del estado de MLflow.
2. La otra, si necesita reentrenar para probar algo, **no commitea** `mlflow.db`
   ni `mlruns/`.
3. Si ya hay conflicto en `mlflow.db`, la salida rápida es quedarse con una de
   las dos versiones y relocalizar:

   ```powershell
   git checkout --theirs mlflow.db   # o --ours
   git checkout --theirs mlruns
   uv run python scripts/relocate_mlflow.py
   ```
