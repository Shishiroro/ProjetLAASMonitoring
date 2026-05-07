"""
run_pipeline.py — CLI orchestrateur LARD-LAAS-TAF
==================================================
Entry point CLI minimal. Toute la logique vit dans :
  - Phase 1            : project/Generate.generate_runs
  - Phase 2            : project/export/Export.render_run
  - Phase 3            : Detection_Evaluation.evaluate_run
  - Modes batch        : project/runs.render_runs / evaluate_runs / full_pipeline
                         (boucles + agregation + cleanup meteo)

Modes :
    python run_pipeline.py generate -n 5
    python run_pipeline.py render LFPO_24 --xplane-dir "C:/X-Plane 12"
    python run_pipeline.py render --all --xplane-dir "C:/X-Plane 12"
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
from runs import render_runs, evaluate_runs, full_pipeline


def _add_generate_args(parser):
    """Ajoute les args de generation TAF partages par 'generate' et 'full'."""
    parser.add_argument("-n", "--nb-scenarios", type=int, default=None,
                        help="Nombre de scenarios (surcharge settings.xml)")
    parser.add_argument("-q", "--quiet", action="store_true")


def _add_yolo_args(parser):
    """Ajoute les args YOLO/IoU partages par les modes 'evaluate' et 'full'."""
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Seuil confiance YOLO")
    parser.add_argument("--imgsz", type=int, default=512,
                        help="Taille image YOLO")
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                        help="Seuil IoU")
    parser.add_argument("--iou-method", type=str, default="CIOU",
                        choices=["IOU", "GIOU", "DIOU", "CIOU"])


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline bout-en-bout LARD-LAAS-TAF (X-Plane 12)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Structure runs/ :
  runs/
    LFPO_24/
      LFPO_24.yaml             <- genere par 'generate' (Phase 1)
      poses_cam_export.json    <- poses camera + trajectory_config TAF
      fault_profile.json       <- profil fautes capteur (si actif)
      weather_profile.json     <- profil meteo X-Plane (si actif)
      footage/                 <- images rendu X-Plane (Phase 2)
      degraded/                <- images avec fautes capteur (Phase 2)
      LFPO_24_labels.csv       <- GT LARD (Phase 2)
      predictions.csv          <- predictions YOLO (Phase 3)
      predictions_txt/         <- labels YOLO (.txt, Phase 3)
    pipeline_report.json       <- rapport agrege (metriques IoU par run)

  Visualisations on-demand via notebook.ipynb : yolo_box/, lard_box/,
  xplane_config.json, params_trace.xml.
        """,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    xplane_args = argparse.ArgumentParser(add_help=False)
    _default_xp = "C:/X-Plane 12" if os.name == "nt" else os.path.expanduser("~/X-Plane 12")
    xplane_args.add_argument("--xplane-dir", type=str, default=_default_xp,
                             help=f"Repertoire X-Plane 12 (defaut: {_default_xp})")

    p_gen = sub.add_parser("generate",
                           help="Phase 1 : genere les scenarios TAF (.yaml) dans runs/")
    _add_generate_args(p_gen)

    p_render = sub.add_parser("render", parents=[xplane_args],
                              help="Phase 2 : rendu X-Plane + fautes capteur")
    p_render.add_argument("run", nargs="?", default=None,
                          help="Nom du run (ex: LFPO_24)")
    p_render.add_argument("--all", action="store_true", dest="all_runs",
                          help="Rendre tous les runs dans runs/")

    p_eval = sub.add_parser("evaluate",
                            help="Phase 3 : GT LARD + Detection YOLO + IoU")
    p_eval.add_argument("run", nargs="?", default=None,
                        help="Nom du run (ex: LFPO_24)")
    p_eval.add_argument("--all", action="store_true", dest="all_runs",
                        help="Evaluer tous les runs dans runs/")
    p_eval.add_argument("--runway", type=str, default=None,
                        help="Filtrer GT sur une piste")
    _add_yolo_args(p_eval)

    p_full = sub.add_parser("full", parents=[xplane_args],
                            help="Pipeline complet (Phase 1 + 2 + 3)")
    _add_generate_args(p_full)
    _add_yolo_args(p_full)

    args = parser.parse_args()

    if args.mode == "generate":
        generate_runs(nb_scenarios=args.nb_scenarios, quiet=args.quiet)
        print(f"  Prochaine etape : run_pipeline.py render --all")

    elif args.mode == "render":
        if not args.run and not args.all_runs:
            print("Specifier un run ou --all. Ex: python run_pipeline.py render LFPO_24")
            return
        render_runs(
            run_name=args.run, all_runs=args.all_runs,
            xplane_dir=args.xplane_dir,
        )

    elif args.mode == "evaluate":
        if not args.run and not args.all_runs:
            print("Specifier un run ou --all. Ex: python run_pipeline.py evaluate LFPO_24")
            return
        evaluate_runs(
            run_name=args.run, all_runs=args.all_runs, runway=args.runway,
            conf=args.conf, imgsz=args.imgsz,
            iou_thresh=args.iou_thresh, iou_method=args.iou_method,
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
