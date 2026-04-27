"""
Detection YOLOv8 sur images d'approche.
Genere les predictions (CSV avec bbox) et les images annotees.
"""

import csv
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


def _txt_to_csv(labels_dir: Path, csv_path: Path) -> int:
    """Consolide les .txt YOLO en un seul predictions.csv et supprime les .txt.

    Returns:
        Nombre de detections ecrites.
    """
    txt_files = sorted(labels_dir.glob("*.txt"))
    n_rows = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["image", "class", "cx", "cy", "w", "h", "confidence"])
        for txt in txt_files:
            image_name = txt.stem
            for line in txt.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.strip().split()
                writer.writerow([image_name, int(parts[0]),
                                 parts[1], parts[2], parts[3], parts[4], parts[5]])
                n_rows += 1

    # Supprimer les .txt maintenant consolides
    for txt in txt_files:
        txt.unlink()

    return n_rows


def predict(start: int = 0, n_images: int | None = None, conf: float = 0.25, imgsz: int = 512,
            images_dir: Path | None = None, output_dir: Path | None = None):
    """Lance la prediction YOLOv8 sur les images d'approche.

    Args:
        start: Index de la premiere image (defaut: 0).
        n_images: Nombre d'images a traiter (None = toutes a partir de start).
        conf: Seuil de confiance.
        imgsz: Taille d'image pour l'inference.
        images_dir: Dossier des images (defaut: test_images/test/).
        output_dir: Dossier de sortie (defaut: yolo/output/expN/).
            Si specifie, predictions.csv va dans output_dir/
            et les images annotees dans output_dir/annotated/.
    Returns:
        (predictions_csv, annotated_dir) — chemin du CSV et dossier images annotees.
    """
    src = images_dir or IMAGES_DIR
    images = sorted(list(src.glob("*.jpeg")) + list(src.glob("*.jpg")) + list(src.glob("*.png")))
    end = start + n_images if n_images is not None else None
    images = images[start:end]

    if not images:
        print(f"Aucune image trouvee dans {src}")
        return None, None

    # Determiner les dossiers de sortie
    if output_dir is not None:
        out = Path(output_dir)
        annotated_dir = out / "annotated"
    else:
        out = _next_exp_dir()
        annotated_dir = out

    annotated_dir.mkdir(parents=True, exist_ok=True)

    print(f"Prediction sur {len(images)} images avec {MODEL_PATH.name}")

    model = YOLO(str(MODEL_PATH))

    # Passer le dossier si toutes les images y sont, sinon une liste
    # (une liste fait perdre les noms originaux -> image0, image1...)
    all_in_dir = (start == 0 and n_images is None)
    source = str(src) if all_in_dir else [str(img) for img in images]

    # YOLO ecrit les labels dans <project>/<name>/labels/
    # et les images annotees dans <project>/<name>/
    abs_annotated = annotated_dir.resolve()
    model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        save_txt=True,
        save_conf=True,
        save=True,
        project=str(abs_annotated.parent),
        name=abs_annotated.name,
        exist_ok=True,
    )

    # Consolider les .txt en predictions.csv
    yolo_labels = abs_annotated / "labels"
    predictions_csv = out / "predictions.csv"

    if yolo_labels.exists():
        n_dets = _txt_to_csv(yolo_labels, predictions_csv)
        # Supprimer le dossier labels/ vide
        try:
            yolo_labels.rmdir()
        except OSError:
            pass
        print(f"Predictions : {n_dets} detections dans {predictions_csv.name}")
    else:
        # Aucune detection — CSV vide avec header
        _txt_to_csv(abs_annotated, predictions_csv)
        print(f"Predictions : 0 detections")

    print(f"Images annotees dans : {annotated_dir}")
    return predictions_csv, annotated_dir


def predict_run(run_dir, conf: float = 0.25, imgsz: int = 512):
    """Lance YOLO sur un run.

    Utilise run_dir/degraded/ si present (fautes capteur deja appliquees),
    sinon run_dir/footage/. Sortie : run_dir/predictions.csv + run_dir/annotated/.

    :return: Path du predictions.csv genere (ou None si pas d'images)
    """
    run_dir = Path(run_dir)
    degraded = run_dir / "degraded"
    footage = run_dir / "footage"

    has_degraded = degraded.exists() and bool(
        list(degraded.glob("*.jpeg")) + list(degraded.glob("*.jpg")) + list(degraded.glob("*.png"))
    )
    images_dir = degraded if has_degraded else footage

    n_images = len(
        list(images_dir.glob("*.jpeg"))
        + list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.png"))
    )
    print(f"\n  [YOLO] Prediction sur {n_images} images depuis {images_dir.name}/ ({run_dir.name})...")

    predictions_csv, _ = predict(
        images_dir=images_dir,
        conf=conf,
        imgsz=imgsz,
        output_dir=run_dir,
    )

    if predictions_csv and predictions_csv.exists():
        print(f"  [YOLO] Predictions dans {predictions_csv.name}")
    else:
        print(f"  [YOLO] ATTENTION : pas de predictions generees")

    return predictions_csv


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 prediction sur images d'approche")
    parser.add_argument("-s", "--start", type=int, default=0, help="Index de depart (defaut: 0)")
    parser.add_argument("-n", "--n-images", type=int, default=None, help="Nombre d'images (defaut: toutes)")
    parser.add_argument("--conf", type=float, default=0.25, help="Seuil de confiance (defaut: 0.25)")
    parser.add_argument("--imgsz", type=int, default=512, help="Taille image (defaut: 512)")
    parser.add_argument("--images", type=str, default=None, help="Dossier images (defaut: test_images/test/)")
    parser.add_argument("--output", type=str, default=None, help="Dossier de sortie (defaut: yolo/output/expN/)")
    args = parser.parse_args()

    predict(
        start=args.start, n_images=args.n_images, conf=args.conf, imgsz=args.imgsz,
        images_dir=Path(args.images) if args.images else None,
        output_dir=Path(args.output) if args.output else None,
    )
