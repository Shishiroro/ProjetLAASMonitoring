"""
run_pipeline.py — Orchestrateur bout-en-bout du pipeline LARD-LAAS-TAF
=======================================================================
Tout est centralise dans runs/<ICAO_RWY>/. Le sequencement des phases
(rendu X-Plane -> fautes capteur -> GT LARD -> YOLO -> evaluation IoU)
vit ici ; chaque phase est implementee dans son module metier.

Modes :
    python run_pipeline.py generate -n 5              # Genere .yaml dans runs/
    python run_pipeline.py evaluate LFPO_24            # GT + YOLO + IoU sur un run
    python run_pipeline.py evaluate --all              # Sur tous les runs
    python run_pipeline.py full -n 100 --xplane-dir "C:/X-Plane 12"
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
YOLO_DIR = ROOT / "yolo"

# sys.path pour imports cross-modules (lazy aussi cote fonctions metier)
for _p in (ROOT / "project", ROOT / "project" / "export", YOLO_DIR, YOLO_DIR / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from runs_layout import (
    TAF_OUTPUT_DIR, find_runs, has_images,
    create_runs_from_taf_output, aggregate_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_weather(xplane_dir):
    """Clear meteo X-Plane (no-op si xplane_dir vide ou plugin absent)."""
    if not xplane_dir:
        return
    try:
        from xplane_weather import reset_weather, set_exchange_dir
        set_exchange_dir(xplane_dir)
        reset_weather()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Etape 1 : Generation TAF (.yaml) -> runs/<ICAO_RWY>/
# ---------------------------------------------------------------------------

def step_generate(nb_scenarios=None, quiet=False):
    """Lance TAF puis reorganise les .yaml dans runs/. Retourne list[Path]."""
    print("=" * 60)
    print(" ETAPE 1 : Generation TAF")
    print("=" * 60)

    if TAF_OUTPUT_DIR.exists():
        shutil.rmtree(TAF_OUTPUT_DIR)

    from Generate import run
    run(nb_test_cases=nb_scenarios, verbose=not quiet)

    return create_runs_from_taf_output()


# ---------------------------------------------------------------------------
# Sequencement par run : render -> faults -> GT -> annotate -> predict -> eval
# ---------------------------------------------------------------------------

def _process_run(run_dir, runway=None, conf=0.25, imgsz=512,
                 iou_thresh=0.5, iou_method="CIOU", xplane_dir=None):
    """Execute la sequence complete d'evaluation sur un run.

    :return: dict de metriques pour le rapport, ou None si run skip/echoue
    """
    from xplane_bridge import render_run
    from sensor_faults import apply_faults
    from lard_bridge import generate_gt, annotate_gt
    from predict import predict_run
    from evaluate import evaluate_run
    from runway import runway_from_run_name

    print(f"\n{'-' * 50}")
    print(f" Run : {run_dir.name}")
    print(f"{'-' * 50}")

    # Rendu X-Plane si pas encore d'images
    if not has_images(run_dir):
        if not render_run(run_dir, xplane_dir or ""):
            print(f"  [SKIP] Echec rendu X-Plane pour {run_dir.name}")
            return None
    if not has_images(run_dir):
        print(f"  [SKIP] Pas d'images dans footage/ pour {run_dir.name}")
        return None

    # GT LARD
    try:
        generate_gt(run_dir)
    except Exception as e:
        print(f"  [GT] ERREUR : {e}")
        if not list(run_dir.glob("*_labels.csv")):
            print(f"  [SKIP] Pas de CSV GT pour {run_dir.name}")
            return None
        print(f"  [GT] Utilisation du CSV existant")

    # Annotations visuelles GT (echantillon)
    try:
        annotate_gt(run_dir, runway=runway)
    except Exception as e:
        print(f"  [GT-VIS] ERREUR : {e}")

    # Fautes capteur (si fault_profile.json present)
    try:
        apply_faults(run_dir)
    except Exception as e:
        print(f"  [FAULTS] ERREUR : {e}")

    # YOLO
    try:
        predictions_csv = predict_run(run_dir, conf=conf, imgsz=imgsz)
    except Exception as e:
        print(f"  [YOLO] ERREUR : {e}")
        return None
    if predictions_csv is None or not predictions_csv.exists():
        print(f"  [SKIP] Pas de predictions pour {run_dir.name}")
        return None

    # Eval IoU
    try:
        metrics = evaluate_run(run_dir, runway=runway,
                               iou_thresh=iou_thresh, iou_method=iou_method)
    except Exception as e:
        print(f"  [EVAL] ERREUR : {e}")
        return None
    if not metrics:
        return None

    rwy = runway or runway_from_run_name(run_dir.name)
    return {
        "run": run_dir.name,
        "runway": rwy,
        **{k: round(v, 4) if isinstance(v, float) else v
           for k, v in metrics.items() if k in ("ap", "f1", "p", "r", "c", "tp", "fp", "fn")},
    }


# ---------------------------------------------------------------------------
# Modes : evaluate / full
# ---------------------------------------------------------------------------

def run_evaluate(run_name=None, all_runs=False, runway=None,
                 conf=0.25, imgsz=512, iou_thresh=0.5, iou_method="CIOU",
                 xplane_dir=None):
    """Enchaine rendu -> GT -> YOLO -> evaluation sur les runs specifies."""
    print("=" * 60)
    print(" PIPELINE : Render + GT + YOLO + Evaluation IoU")
    print("=" * 60)

    runs = find_runs(run_name, all_runs)
    if not runs:
        print("[Pipeline] Aucun run valide trouve.")
        print(f"  Verifier que runs/<nom>/ contient un .yaml et poses_cam_export.json ou footage/")
        return []

    print(f"\n[Pipeline] {len(runs)} run(s) a evaluer")

    all_results = []
    for run_dir in runs:
        result = _process_run(run_dir, runway=runway, conf=conf, imgsz=imgsz,
                              iou_thresh=iou_thresh, iou_method=iou_method,
                              xplane_dir=xplane_dir)
        if result:
            all_results.append(result)

    _reset_weather(xplane_dir)
    aggregate_report(all_results)
    return all_results


def run_full(nb_scenarios=None, quiet=False, conf=0.25, imgsz=512,
             iou_thresh=0.5, iou_method="CIOU", xplane_dir=None):
    """Pipeline complet : generate + (rendu + GT + YOLO + eval) par scenario."""
    created_runs = step_generate(nb_scenarios=nb_scenarios, quiet=quiet)
    if not created_runs:
        print("[Pipeline] Aucun scenario genere, arret.")
        return []

    print(f"\n{'=' * 60}")
    print(" ETAPE 2 : Rendu X-Plane 12 + Evaluation")
    print(f"{'=' * 60}")

    all_results = []
    for run_dir in created_runs:
        result = _process_run(run_dir, conf=conf, imgsz=imgsz,
                              iou_thresh=iou_thresh, iou_method=iou_method,
                              xplane_dir=xplane_dir)
        if result:
            all_results.append(result)

    _reset_weather(xplane_dir)
    aggregate_report(all_results)
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline bout-en-bout LARD-LAAS-TAF (X-Plane 12)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Structure runs/ :
  runs/
    LFPO_24/
      LFPO_24.yaml             <- genere par 'generate'
      poses_cam_export.json    <- poses camera
      params_trace.xml         <- parametres TAF du scenario
      footage/                 <- images rendu X-Plane
      LFPO_24_labels.csv       <- GT LARD auto
      predictions.csv          <- predictions YOLO auto
      annotated/               <- images annotees auto
      eval_results.json        <- metriques IoU auto
      xplane_config.json       <- config rendu X-Plane
      fault_profile.json       <- profil fautes capteur (si actif)
      weather_profile.json     <- profil meteo X-Plane (si actif)
    pipeline_report.json    <- rapport agrege

Exemples :
  python run_pipeline.py generate -n 5
  python run_pipeline.py evaluate LFPO_24
  python run_pipeline.py evaluate --all
  python run_pipeline.py full -n 100 --xplane-dir "C:/X-Plane 12"
        """,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    xplane_args = argparse.ArgumentParser(add_help=False)
    _default_xp = "C:/X-Plane 12" if os.name == "nt" else os.path.expanduser("~/X-Plane 12")
    xplane_args.add_argument("--xplane-dir", type=str, default=_default_xp,
                             help=f"Repertoire X-Plane 12 (defaut: {_default_xp})")

    p_gen = sub.add_parser("generate", parents=[xplane_args],
                           help="Genere les scenarios TAF (.yaml) dans runs/")
    p_gen.add_argument("-n", "--nb-scenarios", type=int, default=None,
                       help="Nombre de scenarios (surcharge settings.xml)")
    p_gen.add_argument("-q", "--quiet", action="store_true")

    p_eval = sub.add_parser("evaluate", parents=[xplane_args],
                            help="GT + YOLO + IoU sur un ou tous les runs")
    p_eval.add_argument("run", nargs="?", default=None, help="Nom du run (ex: LFPO_24)")
    p_eval.add_argument("--all", action="store_true", dest="all_runs",
                        help="Evaluer tous les runs dans runs/")
    p_eval.add_argument("--runway", type=str, default=None, help="Filtrer GT sur une piste")
    p_eval.add_argument("--conf", type=float, default=0.25, help="Seuil confiance YOLO")
    p_eval.add_argument("--imgsz", type=int, default=512, help="Taille image YOLO")
    p_eval.add_argument("--iou-thresh", type=float, default=0.5, help="Seuil IoU")
    p_eval.add_argument("--iou-method", type=str, default="CIOU",
                        choices=["IOU", "GIOU", "DIOU", "CIOU"])

    p_full = sub.add_parser("full", parents=[xplane_args],
                            help="Pipeline complet (generate + rendu X-Plane + evaluate)")
    p_full.add_argument("-n", "--nb-scenarios", type=int, default=None)
    p_full.add_argument("-q", "--quiet", action="store_true")
    p_full.add_argument("--conf", type=float, default=0.25)
    p_full.add_argument("--imgsz", type=int, default=512)
    p_full.add_argument("--iou-thresh", type=float, default=0.5)
    p_full.add_argument("--iou-method", type=str, default="CIOU",
                        choices=["IOU", "GIOU", "DIOU", "CIOU"])

    args = parser.parse_args()

    if args.mode == "generate":
        step_generate(nb_scenarios=args.nb_scenarios, quiet=args.quiet)
        print(f"  Prochaine etape : lancer le rendu X-Plane (run_pipeline.py full)")

    elif args.mode == "evaluate":
        if not args.run and not args.all_runs:
            print("Specifier un run ou --all. Ex: python run_pipeline.py evaluate LFPO_24")
            return
        run_evaluate(
            run_name=args.run, all_runs=args.all_runs, runway=args.runway,
            conf=args.conf, imgsz=args.imgsz,
            iou_thresh=args.iou_thresh, iou_method=args.iou_method,
            xplane_dir=args.xplane_dir,
        )

    elif args.mode == "full":
        run_full(
            nb_scenarios=args.nb_scenarios, quiet=args.quiet,
            conf=args.conf, imgsz=args.imgsz,
            iou_thresh=args.iou_thresh, iou_method=args.iou_method,
            xplane_dir=args.xplane_dir,
        )


if __name__ == "__main__":
    main()
