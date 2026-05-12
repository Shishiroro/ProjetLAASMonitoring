"""
Detection_Evaluation.py — Module Phase 3 : Detection YOLO + IoU
================================================================
Centralise les fonctions de detection et d'evaluation IoU appelees par
l'orchestrateur run_pipeline.py. Pas de CLI standalone : ce module est
purement une bibliotheque.

Phase 3 consomme les sorties de Phase 2 (Export.render_run) :
  - footage/ ou degraded/  : images
  - <name>_labels.csv      : GT LARD (genere par step_ground_truth en Phase 2)

API publique :
    step_predict(run_dir, conf, imgsz)        -> Path predictions.csv | None
    step_iou(run_dir, runway, ...)            -> dict metriques | None
    evaluate_run(run_dir, runway, conf, ...)  -> dict resume run | None
"""

import sys
from pathlib import Path

# Centralise les sys.path via project/_paths.py (sibling de ce fichier).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import _paths  # noqa: F401


# ---------------------------------------------------------------------------
# Sous-etapes : prediction YOLO et calcul IoU
# ---------------------------------------------------------------------------

def step_predict(run_dir, conf=0.25, imgsz=512):
    """Lance YOLO sur les images du run.

    :return: Path du predictions.csv genere, ou None si echec/pas d'images
    """
    from predict import predict_run

    try:
        return predict_run(run_dir, conf=conf, imgsz=imgsz)
    except Exception as e:
        print(f"  [Eval] YOLO ERREUR : {e}")
        return None


def step_iou(run_dir, runway=None, iou_thresh=0.5, iou_method="CIOU"):
    """Calcule les metriques IoU predictions vs GT.

    :return: dict de metriques (ou None si erreur/pas de donnees)
    """
    from evaluate import evaluate_run as _evaluate_run

    try:
        return _evaluate_run(
            run_dir, runway=runway,
            iou_thresh=iou_thresh, iou_method=iou_method,
        )
    except Exception as e:
        print(f"  [Eval] IoU ERREUR : {e}")
        return None


# ---------------------------------------------------------------------------
# Orchestrateur : Detection + Evaluation pour un run
# ---------------------------------------------------------------------------

def evaluate_run(run_dir, runway=None, conf=0.25, imgsz=512,
                 iou_thresh=0.5, iou_method="CIOU"):
    """Phase 3 : Detection YOLO + evaluation IoU sur un run.

    Suppose que la donnee brute (images + GT LARD) existe deja, produite par
    Phase 2 (Export.render_run). Aucune generation ici.

    :return: dict de metriques arrondies (run, runway, ap, f1, p, r, c, tp, fp, fn)
             ou None si echec
    """
    from runway import runway_from_run_name

    run_dir = Path(run_dir)
    print(f"\n  [Eval] Detection + IoU pour {run_dir.name}")

    if not list(run_dir.glob("*_labels.csv")):
        print(f"  [Eval] Pas de GT pour {run_dir.name}. Lancer 'render' d'abord.")
        return None

    predictions_csv = step_predict(run_dir, conf=conf, imgsz=imgsz)
    if predictions_csv is None or not predictions_csv.exists():
        print(f"  [Eval] Pas de predictions pour {run_dir.name}")
        return None

    metrics = step_iou(
        run_dir, runway=runway,
        iou_thresh=iou_thresh, iou_method=iou_method,
    )
    if not metrics:
        return None

    rwy = runway or runway_from_run_name(run_dir.name)
    return {
        "run": run_dir.name,
        "runway": rwy,
        **{k: round(v, 4) if isinstance(v, float) else v
           for k, v in metrics.items()
           if k in ("ap", "f1", "p", "r", "c", "tp", "fp", "fn")},
    }
