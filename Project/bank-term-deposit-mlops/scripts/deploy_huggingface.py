"""
Arma el repositorio del Hugging Face Space, listo para subir.

Hugging Face Spaces despliega desde un repositorio git propio. Este script
construye ese repositorio en `build/huggingface-space/` copiando únicamente lo
que el servicio necesita en runtime:

    README.md          <- deploy/huggingface/README.md (con el frontmatter del Space)
    Dockerfile         <- deploy/huggingface/Dockerfile (puerto 7860, usuario 1000)
    src/               código de la aplicación
    configs/           configuración
    models/champion/   el modelo exportado (~1 MB)
    pyproject.toml, uv.lock

No se copian notebooks, tests, datos ni `mlruns/`: no aportan en producción y
harían el push mucho más lento.

Antes de ejecutarlo hace falta el modelo exportado:

    uv run python scripts/export_model.py

Uso:
    uv run python scripts/deploy_huggingface.py
    uv run python scripts/deploy_huggingface.py --space usuario/mi-space --push
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_SOURCE = PROJECT_ROOT / "deploy" / "huggingface"
DEFAULT_BUILD_DIR = PROJECT_ROOT / "build" / "huggingface-space"

# (origen relativo a la raíz del proyecto, destino en el Space)
FILES_TO_COPY = [
    ("pyproject.toml", "pyproject.toml"),
    ("uv.lock", "uv.lock"),
]
DIRS_TO_COPY = [
    ("src", "src"),
    ("configs", "configs"),
    ("models/champion", "models/champion"),
]
IGNORED_IN_DIRS = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints", ".DS_Store")


def build_space(build_dir: Path) -> Path:
    """
    Construye el directorio del Space.

    Args:
        build_dir: destino. Se borra y recrea en cada ejecución.

    Returns:
        Ruta del directorio construido.

    Raises:
        FileNotFoundError: si falta el modelo exportado.
    """
    config = load_config()
    champion_dir = PROJECT_ROOT / config.serving.local_model_dir

    if not (champion_dir / "metadata.json").exists():
        raise FileNotFoundError(
            f"No hay un modelo exportado en {champion_dir}.\n"
            "Ejecuta primero: uv run python scripts/export_model.py"
        )

    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    # El README con frontmatter y el Dockerfile de Spaces van en la raíz del
    # Space, no en deploy/huggingface/.
    shutil.copy2(HF_SOURCE / "README.md", build_dir / "README.md")
    shutil.copy2(HF_SOURCE / "Dockerfile", build_dir / "Dockerfile")

    for source_name, target_name in FILES_TO_COPY:
        shutil.copy2(PROJECT_ROOT / source_name, build_dir / target_name)

    for source_name, target_name in DIRS_TO_COPY:
        shutil.copytree(
            PROJECT_ROOT / source_name,
            build_dir / target_name,
            ignore=IGNORED_IN_DIRS,
        )

    # `models/.gitkeep` para que el COPY del Dockerfile no falle si alguien
    # borra el campeón del Space.
    (build_dir / "models" / ".gitkeep").touch()

    total_mb = sum(f.stat().st_size for f in build_dir.rglob("*") if f.is_file()) / 1024**2
    n_files = sum(1 for f in build_dir.rglob("*") if f.is_file())
    logger.info("Space construido en %s (%d archivos, %.1f MB)", build_dir, n_files, total_mb)
    return build_dir


def push_to_space(build_dir: Path, space_id: str) -> None:
    """
    Inicializa el repo y lo sube al Space.

    Args:
        build_dir: directorio construido.
        space_id: identificador `usuario/nombre-del-space`.
    """
    remote = f"https://huggingface.co/spaces/{space_id}"

    def run(*args: str) -> None:
        logger.info("$ %s", " ".join(args))
        subprocess.run(args, cwd=build_dir, check=True)

    run("git", "init", "-b", "main")
    run("git", "add", ".")
    run("git", "commit", "-m", "Deploy API de prediccion de deposito a plazo fijo")
    run("git", "remote", "add", "origin", remote)
    # `--force`: el Space nace con un commit inicial propio que hay que reemplazar.
    run("git", "push", "--force", "origin", "main")

    logger.info("Desplegado en %s", remote)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--space",
        default=None,
        help="identificador del Space, por ejemplo 'tu-usuario/bank-term-deposit-api'",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help="directorio donde construir el Space",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="además de construir, sube el repositorio al Space (requiere --space)",
    )
    args = parser.parse_args()

    try:
        build_dir = build_space(args.build_dir)
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if args.push:
        if not args.space:
            print("ERROR: --push requiere --space usuario/nombre", file=sys.stderr)
            return 1
        try:
            push_to_space(build_dir, args.space)
        except subprocess.CalledProcessError as error:
            print(f"ERROR: el push falló ({error})", file=sys.stderr)
            return 1
        return 0

    space = args.space or "TU-USUARIO/bank-term-deposit-api"
    print()
    print("=" * 70)
    print("SPACE LISTO PARA SUBIR")
    print("=" * 70)
    print(f"  Directorio: {build_dir}")
    print()
    print("Pasos:")
    print("  1. Crear el Space en https://huggingface.co/new-space")
    print("     - SDK: Docker   - Template: Blank   - Hardware: CPU basic (gratis)")
    print()
    print("  2. Subirlo:")
    print(f"     cd {build_dir}")
    print("     git init -b main")
    print("     git add .")
    print('     git commit -m "Deploy API"')
    print(f"     git remote add origin https://huggingface.co/spaces/{space}")
    print("     git push --force origin main")
    print()
    print("     (Usuario: tu nombre de HF. Contrasena: un Access Token con")
    print("      permiso de escritura, desde huggingface.co/settings/tokens)")
    print()
    print("  O en un solo comando:")
    print(f"     uv run python scripts/deploy_huggingface.py --space {space} --push")
    print()
    print("  3. El build tarda 3-5 min. Luego:")
    print(f"     https://huggingface.co/spaces/{space}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
