"""
Script jetable — Genere le catalogue visuel pour les 4 nouvelles erreurs
(fog, zoom_blur, contrast, pixelate) a 3 niveaux de severite.

Usage:
    python yolo/camera_sensor_errors/generate_new_errors_catalog.py
    python yolo/camera_sensor_errors/generate_new_errors_catalog.py --n-samples 5
"""

import argparse
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_sensor_errors import ERROR_REGISTRY

NEW_ERRORS = ["fog", "zoom_blur", "contrast", "pixelate"]

SEVERITIES = {
    "low": 0.2,
    "moderate": 0.5,
    "extreme": 0.9,
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "data_camera_sensor_errors"
DEFAULT_SOURCE = PROJECT_ROOT / "runs" / "LFPG_09L" / "footage"


def main():
    parser = argparse.ArgumentParser(description="Catalogue des 4 nouvelles erreurs")
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE))
    parser.add_argument("--n-samples", type=int, default=3)
    args = parser.parse_args()

    src = Path(args.source)
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    images = sorted(f for f in src.iterdir() if f.suffix.lower() in extensions)

    if not images:
        print(f"Aucune image dans {src}")
        return

    n = min(args.n_samples, len(images))
    indices = np.linspace(0, len(images) - 1, n, dtype=int)
    samples = [images[i] for i in indices]
    print(f"{n} images selectionnees sur {len(images)}")

    total = n * len(NEW_ERRORS) * len(SEVERITIES)
    count = 0
    for error_name in NEW_ERRORS:
        error_func = ERROR_REGISTRY[error_name]
        error_dir = OUTPUT_DIR / error_name
        error_dir.mkdir(parents=True, exist_ok=True)
        for sev_name, sev_val in SEVERITIES.items():
            for img_path in samples:
                img = cv2.imread(str(img_path))
                degraded = error_func(img, sev_val)
                out_name = f"{img_path.stem}_{sev_name}.jpg"
                cv2.imwrite(str(error_dir / out_name), degraded)
                count += 1
        print(f"  {error_name} done ({count}/{total})")

    print(f"\n{count} images generees dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
