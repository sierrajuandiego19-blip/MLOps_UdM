<#
.SYNOPSIS
    Equivalente del Makefile para Windows, donde `make` no viene instalado.

.DESCRIPTION
    Atajos de los comandos del proyecto. Cada uno es un `uv run` que también se
    puede escribir a mano; están todos en el README.

.EXAMPLE
    ./run.ps1 help
    ./run.ps1 train
    ./run.ps1 check
#>

param(
    [Parameter(Position = 0)]
    [string]$Target = "help"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$targets = [ordered]@{
    "install"      = "Instala dependencias y los hooks de pre-commit"
    "train"        = "Entrena el modelo (ingesta, experimentos, registro, promocion)"
    "train-fast"   = "Entrena sin busqueda de hiperparametros"
    "score"        = "Puntua un lote y genera la lista de llamadas"
    "monitor"      = "Mide drift y genera el reporte HTML"
    "all"          = "Ejecuta train + score + monitor"
    "api"          = "Levanta la API en modo desarrollo"
    "mlflow"       = "Abre la UI de MLflow del proyecto (puerto 5001)"
    "prefect"      = "Levanta el servidor de Prefect"
    "deploy-flows" = "Registra los flows con sus schedules"
    "export-model" = "Exporta el campeon para deployment sin MLflow"
    "test"         = "Corre los tests"
    "test-cov"     = "Corre los tests con reporte de cobertura"
    "lint"         = "Revisa lint y formato sin modificar nada"
    "format"       = "Aplica formato y arregla lo que ruff pueda"
    "check"        = "Lo que verifica el CI (lint + test)"
    "notebooks"    = "Ejecuta los tres notebooks y guarda sus salidas"
    "clean"        = "Borra caches y artefactos temporales"
    "clean-all"    = "Borra ademas datos, modelos y experimentos (deja el repo en blanco)"
    "setup"        = "TRAS CLONAR: instala, reapunta MLflow a esta maquina y verifica (~3 min)"
    "bootstrap"    = "ALTERNATIVA: reproduce el estado desde cero reentrenando (~12 min)"
    "demo-up"      = "EXPOSICION: arranca MLflow, Prefect, worker y API en ventanas aparte"
    "demo-open"    = "EXPOSICION: abre todas las pestanas del navegador"
    "demo-check"   = "EXPOSICION: verifica que los 3 servicios respondan"
    "demo-down"    = "EXPOSICION: detiene los servicios de la demo"
}

# Puertos de la demo. MLflow va en 5001 y no en 5000 a proposito: en esta
# maquina el 5000 lo ocupa el MLflow del repositorio del curso, que apunta a
# otra base de datos y se veria vacio en plena exposicion.
$script:MlflowPort  = 5001
$script:PrefectPort = 4200
$script:ApiPort     = 8000

function Start-DemoService {
    param([string]$Title, [string]$Command)
    Write-Host "  arrancando $Title..." -ForegroundColor DarkGray
    $full = "`$Host.UI.RawUI.WindowTitle = '$Title'; Set-Location '$PSScriptRoot'; $Command"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $full
}

function Test-DemoServices {
    $checks = @(
        @{ Name = "MLflow ";  Url = "http://127.0.0.1:$script:MlflowPort/health" },
        @{ Name = "Prefect";  Url = "http://127.0.0.1:$script:PrefectPort/api/health" },
        @{ Name = "API    ";  Url = "http://127.0.0.1:$script:ApiPort/health" }
    )
    $allOk = $true
    foreach ($c in $checks) {
        try {
            $r = Invoke-WebRequest -Uri $c.Url -TimeoutSec 4 -UseBasicParsing
            if ($r.StatusCode -eq 200) {
                Write-Host "  [OK]   $($c.Name)  $($c.Url)" -ForegroundColor Green
            } else {
                Write-Host "  [??]   $($c.Name)  HTTP $($r.StatusCode)" -ForegroundColor Yellow
                $allOk = $false
            }
        } catch {
            Write-Host "  [NO]   $($c.Name)  no responde" -ForegroundColor Red
            $allOk = $false
        }
    }
    return $allOk
}

function Show-Help {
    Write-Host ""
    Write-Host "Proyecto final MLOps - Bank Term Deposit" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Uso: ./run.ps1 <target>"
    Write-Host ""
    foreach ($key in $targets.Keys) {
        Write-Host ("  {0,-14} {1}" -f $key, $targets[$key])
    }
    Write-Host ""
}

function Invoke-Step {
    param([string[]]$CommandArgs)
    Write-Host "> uv run $($CommandArgs -join ' ')" -ForegroundColor DarkGray
    & uv run @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo: uv run $($CommandArgs -join ' ') (exit $LASTEXITCODE)"
    }
}

switch ($Target) {
    "help" { Show-Help }

    "install" {
        & uv sync --all-groups
        if ($LASTEXITCODE -ne 0) { throw "uv sync fallo" }
        Invoke-Step @("pre-commit", "install")
    }

    "train"        { Invoke-Step @("python", "-m", "src.pipelines.training_flow") }
    "train-fast"   { Invoke-Step @("python", "-c", "from src.pipelines.training_flow import training_pipeline; training_pipeline(skip_tuning=True)") }
    "score"        { Invoke-Step @("python", "-m", "src.pipelines.batch_scoring_flow") }
    "monitor"      { Invoke-Step @("python", "-m", "src.pipelines.monitoring_flow") }

    "all" {
        Invoke-Step @("python", "-m", "src.pipelines.training_flow")
        Invoke-Step @("python", "-m", "src.pipelines.batch_scoring_flow")
        Invoke-Step @("python", "-m", "src.pipelines.monitoring_flow")
    }

    "api"          { Invoke-Step @("uvicorn", "src.api.main:app", "--reload") }
    "mlflow" {
        # Ruta ABSOLUTA a proposito: con una relativa, MLflow abre una base de
        # datos distinta segun el directorio desde donde se lance. Y puerto
        # 5001, no 5000, que suele estar ocupado por el MLflow del repo del curso.
        $dbUri = "sqlite:///" + ($PSScriptRoot -replace '\\', '/') + "/mlflow.db"
        Invoke-Step @("mlflow", "server", "--backend-store-uri", $dbUri, "--host", "127.0.0.1", "--port", "$script:MlflowPort")
    }
    "prefect"      { Invoke-Step @("prefect", "server", "start") }
    "deploy-flows" { Invoke-Step @("python", "deploy_flows.py") }
    "export-model" { Invoke-Step @("python", "scripts/export_model.py") }

    "test"     { Invoke-Step @("pytest") }
    "test-cov" { Invoke-Step @("pytest", "--cov=src", "--cov-report=term-missing", "--cov-report=html") }

    "lint" {
        Invoke-Step @("ruff", "check", "src", "tests")
        Invoke-Step @("black", "--check", "src", "tests")
    }

    "format" {
        Invoke-Step @("ruff", "check", "--fix", "src", "tests")
        Invoke-Step @("black", "src", "tests")
    }

    "check" {
        Invoke-Step @("ruff", "check", "src", "tests")
        Invoke-Step @("black", "--check", "src", "tests")
        Invoke-Step @("pytest")
    }

    "notebooks" {
        Invoke-Step @("jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=1800", "notebooks/01_eda.ipynb",
            "notebooks/02_baseline.ipynb", "notebooks/03_experiments.ipynb")
    }

    "clean" {
        foreach ($path in @(".pytest_cache", ".ruff_cache", "htmlcov", ".coverage", "coverage.xml")) {
            if (Test-Path $path) { Remove-Item -Recurse -Force $path }
        }
        Get-ChildItem -Recurse -Directory -Filter "__pycache__" |
            ForEach-Object { Remove-Item -Recurse -Force $_.FullName }
        Write-Host "Limpieza completada." -ForegroundColor Green
    }

    "clean-all" {
        Write-Host ""
        Write-Host "Esto borra el historial de MLflow, los datos y el modelo." -ForegroundColor Yellow
        Write-Host "Para recuperarlo: ./run.ps1 bootstrap (reentrena, ~12 min)" -ForegroundColor Yellow
        $answer = Read-Host "Continuar? (escribe 'si')"
        if ($answer -ne "si") { Write-Host "Cancelado." -ForegroundColor DarkGray; break }

        & $PSCommandPath clean

        $targets = @(
            "mlruns", "mlflow.db", "mlflow.db.bak", "build",
            "models\champion", "models\champion_metadata.json", "models\model_card.md"
        )
        foreach ($t in $targets) {
            if (Test-Path $t) { Remove-Item -Recurse -Force $t }
        }
        # Vaciar los directorios de datos conservando los .gitkeep
        foreach ($d in @("data\raw", "data\processed", "data\reference", "data\predictions", "reports", "logs")) {
            if (Test-Path $d) {
                Get-ChildItem -Path $d -Exclude ".gitkeep" | Remove-Item -Recurse -Force
            }
        }
        Write-Host ""
        Write-Host "Repositorio en blanco. Reconstruir con: ./run.ps1 bootstrap" -ForegroundColor Green
        Write-Host ""
    }

    # --- Primera vez en una maquina nueva ----------------------------------

    "setup" {
        # El camino normal despues de clonar: el repositorio ya trae el
        # historial de MLflow, los datos y el modelo, asi que NO hace falta
        # reentrenar. Solo hay que reapuntar las rutas absolutas de MLflow a
        # esta maquina.
        Write-Host ""
        Write-Host "===============================================================" -ForegroundColor Cyan
        Write-Host " SETUP - preparar el proyecto en esta maquina (~3 min)" -ForegroundColor Cyan
        Write-Host "===============================================================" -ForegroundColor Cyan
        Write-Host ""

        Write-Host "[1/4] Instalando dependencias..." -ForegroundColor Yellow
        & uv sync --all-groups
        if ($LASTEXITCODE -ne 0) { throw "uv sync fallo" }

        Write-Host ""
        Write-Host "[2/4] Reapuntando el historial de MLflow a esta ruta..." -ForegroundColor Yellow
        Write-Host "      (MLflow guarda rutas absolutas; sin esto los" -ForegroundColor DarkGray
        Write-Host "       artefactos no se encuentran)" -ForegroundColor DarkGray
        Invoke-Step @("python", "scripts/relocate_mlflow.py")

        Write-Host ""
        Write-Host "[3/4] Registrando los flows de Prefect..." -ForegroundColor Yellow
        try { & uv run prefect work-pool create local-pool --type process 2>$null } catch {}
        Invoke-Step @("python", "deploy_flows.py")

        Write-Host ""
        Write-Host "[4/4] Verificando..." -ForegroundColor Yellow
        Invoke-Step @("pytest", "-q")
        Invoke-Step @("python", "-c", "from src.models.registry import load_champion; c = load_champion(allow_local_fallback=False); print(f'campeon: {c.name} v{c.version} umbral {c.threshold:.2f}')")

        Write-Host ""
        Write-Host "===============================================================" -ForegroundColor Green
        Write-Host " LISTO. Deberias haber visto:" -ForegroundColor Green
        Write-Host "   173 tests pasando" -ForegroundColor Green
        Write-Host "   campeon: bank-term-deposit-classifier v1 umbral 0.42" -ForegroundColor Green
        Write-Host ""
        Write-Host " Siguiente paso:  ./run.ps1 demo-up  y luego  demo-open" -ForegroundColor Green
        Write-Host " Guion:           docs/presentacion.md" -ForegroundColor Green
        Write-Host "===============================================================" -ForegroundColor Green
        Write-Host ""
    }

    "bootstrap" {
        # Reproduce el estado exacto que se usa en la exposicion. El
        # entrenamiento es determinista (semilla fija en configs/config.yaml),
        # asi que estos pasos dan las mismas metricas en cualquier maquina.
        Write-Host ""
        Write-Host "===============================================================" -ForegroundColor Cyan
        Write-Host " BOOTSTRAP - reproduce el estado completo de la demo (~12 min)" -ForegroundColor Cyan
        Write-Host "===============================================================" -ForegroundColor Cyan
        Write-Host ""

        Write-Host "[1/7] Instalando dependencias..." -ForegroundColor Yellow
        & uv sync --all-groups
        if ($LASTEXITCODE -ne 0) { throw "uv sync fallo" }

        Write-Host ""
        Write-Host "[2/7] Tests (deben pasar 173)..." -ForegroundColor Yellow
        Invoke-Step @("pytest", "-q")

        Write-Host ""
        Write-Host "[3/7] Entrenamiento -> registra la v1 como champion (~5 min)..." -ForegroundColor Yellow
        Invoke-Step @("python", "-m", "src.pipelines.training_flow")

        Write-Host ""
        Write-Host "[4/7] Segundo entrenamiento -> la v2 empata y queda como" -ForegroundColor Yellow
        Write-Host "      challenger. Es el momento de gobernanza que se muestra" -ForegroundColor Yellow
        Write-Host "      en la exposicion (~5 min)..." -ForegroundColor Yellow
        Invoke-Step @("python", "-m", "src.pipelines.training_flow")

        Write-Host ""
        Write-Host "[5/7] Scoring batch -> genera la lista de llamadas..." -ForegroundColor Yellow
        Invoke-Step @("python", "-m", "src.pipelines.batch_scoring_flow")

        Write-Host ""
        Write-Host "[6/7] Monitoreo -> genera el reporte de drift..." -ForegroundColor Yellow
        Invoke-Step @("python", "-m", "src.pipelines.monitoring_flow")

        Write-Host ""
        Write-Host "[7/7] Exportando el campeon y registrando los flows..." -ForegroundColor Yellow
        Invoke-Step @("python", "scripts/export_model.py")
        # El work pool tiene que existir antes de registrar los deployments.
        # Si ya existe, el comando falla y da igual: se ignora.
        try { & uv run prefect work-pool create local-pool --type process 2>$null } catch {}
        Invoke-Step @("python", "deploy_flows.py")

        Write-Host ""
        Write-Host "===============================================================" -ForegroundColor Green
        Write-Host " LISTO. Deberias ver:" -ForegroundColor Green
        Write-Host "   Test PR-AUC   0.4909      ROC-AUC  0.8183" -ForegroundColor Green
        Write-Host "   Registry      champion->v1   challenger->v2" -ForegroundColor Green
        Write-Host ""
        Write-Host " Siguiente paso:  ./run.ps1 demo-up" -ForegroundColor Green
        Write-Host " Guion:           docs/presentacion.md" -ForegroundColor Green
        Write-Host "===============================================================" -ForegroundColor Green
        Write-Host ""
    }

    # --- Atajos para la exposicion -----------------------------------------

    "demo-up" {
        $dbUri = "sqlite:///" + ($PSScriptRoot -replace '\\', '/') + "/mlflow.db"

        Write-Host ""
        Write-Host "Arrancando los servicios de la demo..." -ForegroundColor Cyan
        Start-DemoService "MLflow (proyecto)" "uv run mlflow server --backend-store-uri '$dbUri' --host 127.0.0.1 --port $script:MlflowPort"
        Start-Sleep -Seconds 3
        Start-DemoService "Prefect server" "uv run prefect server start"
        Start-Sleep -Seconds 12
        Start-DemoService "Prefect worker" "uv run prefect worker start --pool local-pool"
        Start-DemoService "API prediccion" "uv run uvicorn src.api.main:app --host 127.0.0.1 --port $script:ApiPort"

        Write-Host ""
        Write-Host "Esperando a que respondan (hasta 90s)..." -ForegroundColor Cyan
        for ($i = 0; $i -lt 30; $i++) {
            Start-Sleep -Seconds 3
            $ok = Test-DemoServices
            if ($ok) { break }
            if ($i -lt 29) { Clear-Host; Write-Host "Esperando..." -ForegroundColor Cyan }
        }
        Write-Host ""
        Write-Host "Listo. Ahora: ./run.ps1 demo-open" -ForegroundColor Green
        Write-Host ""
    }

    "demo-open" {
        $urls = @(
            "http://127.0.0.1:$script:MlflowPort",
            "http://127.0.0.1:$script:MlflowPort/#/models",
            "http://127.0.0.1:$script:PrefectPort/runs",
            "http://127.0.0.1:$script:PrefectPort/deployments",
            "http://127.0.0.1:$script:ApiPort/docs"
        )
        foreach ($u in $urls) {
            Write-Host "  abriendo $u" -ForegroundColor DarkGray
            Start-Process $u
            Start-Sleep -Milliseconds 1200
        }
        $report = Join-Path $PSScriptRoot "reports\drift_report_latest.html"
        if (Test-Path $report) {
            Write-Host "  abriendo reporte de drift" -ForegroundColor DarkGray
            Start-Process $report
        } else {
            Write-Host "  (no hay reporte de drift: ./run.ps1 monitor)" -ForegroundColor Yellow
        }
        Write-Host ""
        Write-Host "Pestanas abiertas. Guion: docs/presentacion.md" -ForegroundColor Green
    }

    "demo-check" {
        Write-Host ""
        if (Test-DemoServices) {
            Write-Host ""
            Write-Host "Todo arriba." -ForegroundColor Green
        } else {
            Write-Host ""
            Write-Host "Falta algo. Arranca con: ./run.ps1 demo-up" -ForegroundColor Yellow
        }
        Write-Host ""
    }

    "demo-down" {
        # Solo se matan los procesos que escuchan en los puertos de la demo,
        # para no tumbar otros servidores que el usuario tenga abiertos.
        foreach ($port in @($script:MlflowPort, $script:ApiPort)) {
            $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
            foreach ($c in $conns) {
                try {
                    Stop-Process -Id $c.OwningProcess -Force -ErrorAction Stop
                    Write-Host "  detenido el proceso del puerto $port (PID $($c.OwningProcess))" -ForegroundColor DarkGray
                } catch {
                    Write-Host "  no se pudo detener el PID $($c.OwningProcess)" -ForegroundColor Yellow
                }
            }
        }
        Write-Host ""
        Write-Host "El servidor de Prefect y el worker siguen en sus ventanas:" -ForegroundColor Yellow
        Write-Host "cierralas con Ctrl+C para no matar otros procesos por error." -ForegroundColor Yellow
        Write-Host ""
    }

    default {
        Write-Host "Target desconocido: '$Target'" -ForegroundColor Red
        Show-Help
        exit 1
    }
}
