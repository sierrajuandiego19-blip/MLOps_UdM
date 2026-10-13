"""
Exporta el modelo campeón a un directorio autocontenido.

Para qué: servir el modelo donde no hay un tracking server de MLflow. Un
contenedor en Render, Hugging Face Spaces o Cloud Run no tiene acceso a
`mlflow.db`, así que el modelo tiene que viajar dentro de la imagen.

El directorio resultante (`models/champion/` por defecto) lleva el modelo en
formato MLflow más un `metadata.json` con la versión, el umbral de decisión y
las métricas. La API lo carga automáticamente si el registry no responde, o
siempre si se define `BTD_SERVING__PREFER_LOCAL_MODEL=true`.

Uso:
    uv run python scripts/export_model.py
    uv run python scripts/export_model.py --destination ./build/model
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Permite ejecutar el script directamente, sin instalar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.registry import export_champion, load_local_champion  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help="directorio destino (por defecto: serving.local_model_dir de la configuración)",
    )
    args = parser.parse_args()

    try:
        destination = export_champion(args.destination)
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    # Se recarga lo escrito: un export que no se puede volver a cargar no sirve
    # de nada, y es mejor enterarse aquí que en el arranque del contenedor.
    champion = load_local_champion(destination)
    size_mb = sum(f.stat().st_size for f in destination.rglob("*") if f.is_file()) / 1024**2

    print()
    print("=" * 66)
    print("MODELO EXPORTADO Y VERIFICADO")
    print("=" * 66)
    print(f"  Destino:  {destination}")
    print(f"  Modelo:   {champion.name} v{champion.version}")
    print(f"  Umbral:   {champion.threshold:.2f}")
    print(f"  Tamaño:   {size_mb:.1f} MB")
    print("=" * 66)
    print()
    print("Para servir esta copia sin MLflow:")
    print("  BTD_SERVING__PREFER_LOCAL_MODEL=true uv run uvicorn src.api.main:app")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
