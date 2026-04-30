"""
run_pipeline.py — CLI orchestrateur LARD-LAAS-TAF
==================================================
Entry point CLI minimal. Toute la logique vit dans :
  - Phase 1            : project/Generate.generate_runs
  - Phase 2            : project/export/Export.generate_images_and_GT
  - Phase 3            : Detection_Evaluation.evaluate_run
  - Modes batch        : project/runs.evaluate_runs / full_pipeline
                         (boucles + agregation + cleanup meteo)

Modes :
    python run_pipeline.py generate -n 5
    python run_pipeline.py evaluate LFPO_24
    python run_pipeline.py evaluate --all
    python run_pipeline.py full -n 100 --xplane-dir "C:/X-Plane 12"
"""

import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

for _p in (ROOT, ROOT / "project", ROOT / "project" / "export"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Generate import generate_runs
from runs import evaluate_runs, full_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline bout-en-bout LARD-LAAS-TAF (X-Plane 12)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Structure runs/ :
  runs/
    LFPO_24/
      LFPO_24.yaml             <- genere par 'generate' (Phase 1)
      poses_cam_export.json    <- poses camera
      params_trace.xml         <- parametres TAF du scenario
      fault_profile.json       <- profil fautes capteur (si actif)
      weather_profile.json     <- profil meteo X-Plane (si actif)
      footage/                 <- images rendu X-Plane (Phase 2)
      degraded/                <- images avec fautes capteur (Phase 2)
      LFPO_24_labels.csv       <- GT LARD (Phase 2)
      annotated_lard/          <- GT visualisee (Phase 2)
      predictions.csv          <- predictions YOLO (Phase 3)
      annotated/               <- images annotees YOLO (Phase 3)
      eval_results.json        <- metriques IoU (Phase 3)
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
                           help="Phase 1 : genere les scenarios TAF (.yaml) dans runs/")
    p_gen.add_argument("-n", "--nb-scenarios", type=int, default=None,
                       help="Nombre de scenarios (surcharge settings.xml)")
    p_gen.add_argument("-q", "--quiet", action="store_true")

    p_eval = sub.add_parser("evaluate", parents=[xplane_args],
                            help="Phases 2+3 : Images + GT + Detection + IoU")
    p_eval.add_argument("run", nargs="?", default=None,
                        help="Nom du run (ex: LFPO_24)")
    p_eval.add_argument("--all", action="store_true", dest="all_runs",
                        help="Evaluer tous les runs dans runs/")
    p_eval.add_argument("--runway", type=str, default=None,
                        help="Filtrer GT sur une piste")
    p_eval.add_argument("--conf", type=float, default=0.25,
                        help="Seuil confiance YOLO")
    p_eval.add_argument("--imgsz", type=int, default=512,
                        help="Taille image YOLO")
    p_eval.add_argument("--iou-thresh", type=float, default=0.5,
                        help="Seuil IoU")
    p_eval.add_argument("--iou-method", type=str, default="CIOU",
                        choices=["IOU", "GIOU", "DIOU", "CIOU"])

    p_full = sub.add_parser("full", parents=[xplane_args],
                            help="Pipeline complet (Phase 1 + 2 + 3)")
    p_full.add_argument("-n", "--nb-scenarios", type=int, default=None)
    p_full.add_argument("-q", "--quiet", action="store_true")
    p_full.add_argument("--conf", type=float, default=0.25)
    p_full.add_argument("--imgsz", type=int, default=512)
    p_full.add_argument("--iou-thresh", type=float, default=0.5)
    p_full.add_argument("--iou-method", type=str, default="CIOU",
                        choices=["IOU", "GIOU", "DIOU", "CIOU"])

    args = parser.parse_args()

    if args.mode == "generate":
        generate_runs(nb_scenarios=args.nb_scenarios, quiet=args.quiet)
        print(f"  Prochaine etape : lancer le rendu X-Plane (run_pipeline.py full)")

    elif args.mode == "evaluate":
        if not args.run and not args.all_runs:
            print("Specifier un run ou --all. Ex: python run_pipeline.py evaluate LFPO_24")
            return
        evaluate_runs(
            run_name=args.run, all_runs=args.all_runs, runway=args.runway,
            conf=args.conf, imgsz=args.imgsz,
            iou_thresh=args.iou_thresh, iou_method=args.iou_method,
            xplane_dir=args.xplane_dir,
        )

    elif args.mode == "full":
        full_pipeline(
            nb_scenarios=args.nb_scenarios, quiet=args.quiet,
            conf=args.conf, imgsz=args.imgsz,
            iou_thresh=args.iou_thresh, iou_method=args.iou_method,
            xplane_dir=args.xplane_dir,
        )


if __name__ == "__main__":
    main()
