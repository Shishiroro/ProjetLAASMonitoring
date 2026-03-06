"""
Evaluation IoU : compare les predictions YOLO (.txt) aux ground truths LARD (.csv).
Calcule AP, F1, Precision, Recall via le module yolo/eval/.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from eval.box import box_extract
from eval.metrics_utils import bbox_convert
from eval.metrics import compute_metrics


# --- Colonnes GT LARD (NEW_CORNERS_NAMES = TR, TL, BL, BR) ---
CORNER_X_COLS = ["x_TR", "x_TL", "x_BL", "x_BR"]
CORNER_Y_COLS = ["y_TR", "y_TL", "y_BL", "y_BR"]


def load_predictions(labels_dir: Path) -> torch.Tensor:
    """Charge les .txt YOLO et retourne un tenseur (N, 7) [img_id, cls, x1, y1, x2, y2, conf].

    Les .txt sont en format cxcywh normalise, on convertit en xyxy normalise.
    """
    rows = []
    txt_files = sorted(labels_dir.glob("*.txt"))

    for img_id, txt_path in enumerate(txt_files):
        lines = txt_path.read_text().strip().split("\n")
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split()
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            conf = float(parts[5])
            rows.append([img_id, cls_id, cx, cy, w, h, conf])

    if not rows:
        return torch.zeros((0, 7))

    preds = torch.tensor(rows, dtype=torch.float32)
    # Convertir bbox cxcywh -> xyxy (colonnes 2:6)
    preds[:, 2:6] = bbox_convert(preds[:, 2:6], "cxcywh", "xyxy")
    return preds


def load_ground_truths(csv_path: Path, image_names: list[str], runway: str | None = None) -> torch.Tensor:
    """Charge le .csv LARD et retourne un tenseur (M, 6) [img_id, cls, x1, y1, x2, y2].

    Les 4 coins piste (pixels) sont convertis en bbox englobante xyxy normalise.
    Le mapping image_names[i] -> img_id=i assure la correspondance avec les predictions.
    Si runway est specifie, ne garde que les GT de cette piste.
    """
    df = pd.read_csv(csv_path, sep=";")
    if runway is not None:
        df = df[df["runway"].astype(str) == str(runway)]
    rows = []

    # Creer un mapping nom_image -> img_id
    name_to_id = {name: i for i, name in enumerate(image_names)}

    for _, row in df.iterrows():
        # Extraire le nom du fichier image depuis la colonne 'image'
        img_name = Path(str(row["image"])).stem
        if img_name not in name_to_id:
            continue

        img_id = name_to_id[img_name]
        img_w = float(row["width"])
        img_h = float(row["height"])

        # 4 coins piste en pixels -> bbox englobante xyxy
        xs = np.array([row[c] for c in CORNER_X_COLS], dtype=np.float64)
        ys = np.array([row[c] for c in CORNER_Y_COLS], dtype=np.float64)
        bbox_px = box_extract(xs, ys)  # [x_min, y_min, x_max, y_max] pixels

        # TODO: les labels LARD sont incorrects pour les scenarios GES
        # (bug dans pointcam_to_pix). En attente de fix equipe LARD.
        # Pour l'instant normalisation standard x/width, y/height.
        bbox_norm = np.clip(np.array([
            bbox_px[0] / img_w,
            bbox_px[1] / img_h,
            bbox_px[2] / img_w,
            bbox_px[3] / img_h,
        ]), 0.0, 1.0)

        rows.append([img_id, 0, *bbox_norm])  # cls_id=0 (piste)

    if not rows:
        return torch.zeros((0, 6))

    return torch.tensor(rows, dtype=torch.float32)


def evaluate(labels_dir: Path, csv_path: Path, iou_thresh: float = 0.5, iou_method: str = "CIOU", runway: str | None = None):
    """Lance l'evaluation IoU et affiche les metriques."""

    # Noms des images (stems des .txt) pour le mapping img_id
    txt_files = sorted(labels_dir.glob("*.txt"))
    image_names = [f.stem for f in txt_files]

    if not image_names:
        print(f"Aucun fichier .txt trouve dans {labels_dir}")
        return

    # Charger donnees
    preds = load_predictions(labels_dir)
    gts = load_ground_truths(csv_path, image_names, runway=runway)

    rwy_info = f" (runway {runway})" if runway else " (toutes pistes)"
    print(f"Predictions : {preds.shape[0]} detections sur {len(image_names)} images")
    print(f"Ground truths : {gts.shape[0]} bbox depuis {csv_path.name}{rwy_info}")

    if gts.shape[0] == 0:
        print("Aucun ground truth trouve (verifier correspondance noms images)")
        return

    # Evaluer
    metrics = compute_metrics(
        y_pred=preds,
        y_true=gts,
        iou_thresh=iou_thresh,
        iou_method=iou_method,
        box_format="xyxy",
    )

    # Afficher
    print(f"\n--- Resultats (IoU seuil={iou_thresh}, methode={iou_method}) ---")
    print(f"  AP       : {metrics['ap']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['p']:.4f}")
    print(f"  Recall   : {metrics['r']:.4f}")
    print(f"  Confiance: {metrics['c']:.4f}")
    print(f"  TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}")

    # Sauvegarder dans le dossier exp (parent de labels/)
    exp_dir = labels_dir.parent
    results_file = exp_dir / "eval_results.json"
    results = {
        "iou_thresh": iou_thresh,
        "iou_method": iou_method,
        "runway_filter": runway,
        "gt_csv": csv_path.name,
        "n_images": len(image_names),
        "n_preds": int(preds.shape[0]),
        "n_gts": int(gts.shape[0]),
        "ap": round(metrics["ap"], 4),
        "f1": round(metrics["f1"], 4),
        "precision": round(metrics["p"], 4),
        "recall": round(metrics["r"], 4),
        "confidence": round(metrics["c"], 4),
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
    }
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultats sauvegardes dans : {results_file}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluation IoU predictions YOLO vs GT LARD")
    parser.add_argument("--labels", type=str, required=True, help="Dossier des .txt predictions YOLO")
    parser.add_argument("--csv", type=str, required=True, help="Fichier .csv ground truth LARD")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="Seuil IoU (defaut: 0.5)")
    parser.add_argument("--iou-method", type=str, default="CIOU", choices=["IOU", "GIOU", "DIOU", "CIOU"], help="Methode IoU (defaut: CIOU)")
    parser.add_argument("--runway", type=str, default=None, help="Filtrer GT sur une piste (ex: 24). Sans filtre = toutes les pistes.")
    args = parser.parse_args()

    evaluate(
        labels_dir=Path(args.labels),
        csv_path=Path(args.csv),
        iou_thresh=args.iou_thresh,
        iou_method=args.iou_method,
        runway=args.runway,
    )
