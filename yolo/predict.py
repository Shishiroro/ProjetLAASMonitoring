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
LABELS_DIR = OUTPUT_DIR / "labels"
IMAGES_OUT_DIR = OUTPUT_DIR / "images"


def predict(n_images: int | None = None, conf: float = 0.25, imgsz: int = 512):
    """Lance la prediction YOLOv8 sur les images d'approche.

    Args:
        n_images: Nombre d'images a traiter (None = toutes).
        conf: Seuil de confiance.
        imgsz: Taille d'image pour l'inference.
    """
    images = sorted(IMAGES_DIR.glob("*.jpeg"))
    if n_images is not None:
        images = images[:n_images]

    if not images:
        print(f"Aucune image trouvee dans {IMAGES_DIR}")
        return

    print(f"Prediction sur {len(images)} images avec {MODEL_PATH.name}")

    model = YOLO(str(MODEL_PATH))
    model.predict(
        source=[str(img) for img in images],
        imgsz=imgsz,
        conf=conf,
        save_txt=True,
        save_conf=True,
        save=True,
        project=str(OUTPUT_DIR),
        name=".",
        exist_ok=True,
    )

    print(f"Labels dans : {LABELS_DIR}")
    print(f"Images annotees dans : {IMAGES_OUT_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 prediction sur images d'approche")
    parser.add_argument("-n", "--n-images", type=int, default=None, help="Nombre d'images (defaut: toutes)")
    parser.add_argument("--conf", type=float, default=0.25, help="Seuil de confiance (defaut: 0.25)")
    parser.add_argument("--imgsz", type=int, default=512, help="Taille image (defaut: 512)")
    args = parser.parse_args()

    predict(n_images=args.n_images, conf=args.conf, imgsz=args.imgsz)
