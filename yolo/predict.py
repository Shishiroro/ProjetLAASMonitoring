"""
Detection YOLOv8 sur images d'approche.
Genere les labels (.txt avec bbox) et les images annotees.
"""

from pathlib import Path
from ultralytics import YOLO

# --- Chemins ---
YOLO_DIR = Path(__file__).resolve().parent
MODEL_PATH = YOLO_DIR / "yolov8n.pt"
IMAGES_DIR = YOLO_DIR / "test_images" / "test"
OUTPUT_DIR = YOLO_DIR / "output"


def _next_exp_dir() -> Path:
    """Trouve le prochain dossier exp<N> disponible."""
    i = 1
    while (OUTPUT_DIR / f"exp{i}").exists():
        i += 1
    exp_dir = OUTPUT_DIR / f"exp{i}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def predict(start: int = 0, n_images: int | None = None, conf: float = 0.25, imgsz: int = 512, images_dir: Path | None = None):
    """Lance la prediction YOLOv8 sur les images d'approche.

    Args:
        start: Index de la premiere image (defaut: 0).
        n_images: Nombre d'images a traiter (None = toutes a partir de start).
        conf: Seuil de confiance.
        imgsz: Taille d'image pour l'inference.
        images_dir: Dossier des images (defaut: test_images/test/).
    """
    src = images_dir or IMAGES_DIR
    images = sorted(list(src.glob("*.jpeg")) + list(src.glob("*.jpg")) + list(src.glob("*.png")))
    end = start + n_images if n_images is not None else None
    images = images[start:end]

    if not images:
        print(f"Aucune image trouvee dans {src}")
        return

    exp_dir = _next_exp_dir()
    print(f"Prediction sur {len(images)} images avec {MODEL_PATH.name}")
    print(f"Sortie dans : {exp_dir}")

    model = YOLO(str(MODEL_PATH))

    # Passer le dossier si toutes les images y sont, sinon une liste
    # (une liste fait perdre les noms originaux -> image0, image1...)
    all_in_dir = (start == 0 and n_images is None)
    source = str(src) if all_in_dir else [str(img) for img in images]

    model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        save_txt=True,
        save_conf=True,
        save=True,
        project=str(exp_dir),
        name=".",
        exist_ok=True,
    )

    print(f"Labels dans : {exp_dir / 'labels'}")
    print(f"Images annotees dans : {exp_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 prediction sur images d'approche")
    parser.add_argument("-s", "--start", type=int, default=0, help="Index de depart (defaut: 0)")
    parser.add_argument("-n", "--n-images", type=int, default=None, help="Nombre d'images (defaut: toutes)")
    parser.add_argument("--conf", type=float, default=0.25, help="Seuil de confiance (defaut: 0.25)")
    parser.add_argument("--imgsz", type=int, default=512, help="Taille image (defaut: 512)")
    parser.add_argument("--images", type=str, default=None, help="Dossier images (defaut: test_images/test/)")
    args = parser.parse_args()

    predict(start=args.start, n_images=args.n_images, conf=args.conf, imgsz=args.imgsz, images_dir=Path(args.images) if args.images else None)
