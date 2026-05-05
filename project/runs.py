"""
runs.py — Structure et orchestration des dossiers runs/<ICAO_RWY>/
====================================================================
Centralise tout ce qui touche au cycle de vie des runs :

Layout :
  - ou se trouvent les runs (RUNS_DIR)
  - a quoi ressemble un dossier de run (yaml, poses, profils, footage, etc.)
  - comment creer ces dossiers a partir de la sortie TAF (Phase 1)
  - comment enumerer les runs evaluables
  - comment agreger un rapport multi-run

Orchestration :
  - process_run        : Phase 2 + Phase 3 sur un seul run
  - evaluate_runs      : mode "evaluate" (Phases 2+3 sur N runs filtres)
  - full_pipeline      : mode "full"     (Phase 1 + Phases 2+3 sur N runs)
"""

import json
import shutil
from pathlib import Path

# --- Chemins ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
TAF_OUTPUT_DIR = PROJECT_ROOT / "output"
REPORT_FILE = RUNS_DIR / "pipeline_report.json"

IMAGE_EXTS = (".jpeg", ".jpg", ".png")


def list_images(directory):
    """Liste triee des images (jpeg/jpg/png) d'un dossier (vide si absent)."""
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def has_images(run_dir):
    """Un run a-t-il des images dans footage/ ?"""
    return bool(list_images(Path(run_dir) / "footage"))


def pick_image_source(run_dir):
    """Retourne le dossier d'images a utiliser : degraded/ si non vide, sinon footage/."""
    run = Path(run_dir)
    if list_images(run / "degraded"):
        return run / "degraded"
    return run / "footage"


def find_runs(run_name=None, all_runs=False):
    """Trouve les runs evaluables (yaml + poses ou images). Retourne list[Path]."""
    if run_name:
        candidates = [RUNS_DIR / run_name]
        if not candidates[0].exists():
            candidates = [Path(run_name).resolve()]
    elif all_runs and RUNS_DIR.exists():
        candidates = sorted(d for d in RUNS_DIR.iterdir() if d.is_dir())
    else:
        candidates = []

    valid = []
    for run_dir in candidates:
        if not run_dir.exists():
            print(f"[ERREUR] Run introuvable : {run_dir}")
            continue
        if not list(run_dir.glob("*.yaml")):
            print(f"[SKIP] {run_dir.name} : pas de .yaml")
            continue
        has_poses = (run_dir / "poses_cam_export.json").exists()
        if not has_poses and not has_images(run_dir):
            print(f"[SKIP] {run_dir.name} : ni poses_cam_export.json ni images dans footage/")
            continue
        valid.append(run_dir)
    return valid


def create_runs_from_taf_output():
    """Reorganise la sortie TAF (output/) vers runs/<ICAO_RWY>/.

    Pour chaque .yaml trouve dans output/ : copie yaml + poses_cam_export.json
    (avec scenario_name mis a jour si suffixe) + fault_profile.json +
    weather_profile.json. Genere un suffixe _NNN si une piste est generee
    plusieurs fois.

    :return: list[Path] des runs crees
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    yaml_files = list(TAF_OUTPUT_DIR.rglob("*.yaml"))

    created_runs = []
    for yf in yaml_files:
        name = yf.stem  # ex: LFPO_24

        # Suffixe auto si le dossier existe deja (meme piste generee 2x)
        run_dir = RUNS_DIR / name
        if run_dir.exists():
            idx = 2
            while (RUNS_DIR / f"{name}_{idx:03d}").exists():
                idx += 1
            run_dir = RUNS_DIR / f"{name}_{idx:03d}"

        run_dir.mkdir(parents=True, exist_ok=True)
        run_name = run_dir.name

        shutil.copy2(yf, run_dir / f"{run_name}.yaml")

        # poses_cam_export.json : maj scenario_name si suffixe _NNN
        poses_json = yf.parent / "poses_cam_export.json"
        if poses_json.exists():
            poses_dst = run_dir / "poses_cam_export.json"
            shutil.copy2(poses_json, poses_dst)
            if run_name != name:
                poses_data = json.load(open(poses_dst))
                poses_data["scenario_name"] = run_name
                with open(poses_dst, "w") as pf:
                    json.dump(poses_data, pf, indent=2)

        extras = ""
        for extra in ("fault_profile.json", "weather_profile.json"):
            src = yf.parent / extra
            if src.exists():
                shutil.copy2(src, run_dir / extra)
                extras += f" + {extra}"

        created_runs.append(run_dir)
        print(f"  [RUNS] {run_dir.name}/ <- .yaml + poses_cam_export.json{extras}")

    print(f"\n[Pipeline] {len(created_runs)} run(s) cree(s) dans runs/")
    return created_runs


def aggregate_report(results):
    """Affiche le rapport agrege et sauve REPORT_FILE."""
    if not results:
        return
    print(f"\n{'=' * 60}")
    print(f" RAPPORT FINAL ({len(results)} run(s))")
    print(f"{'=' * 60}")
    print(f"{'Run':<20} {'AP':>6} {'F1':>6} {'P':>6} {'R':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    print(f"{'-' * 60}")
    for r in results:
        print(f"{r['run']:<20} {r.get('ap', 0):>6.3f} {r.get('f1', 0):>6.3f} "
              f"{r.get('p', 0):>6.3f} {r.get('r', 0):>6.3f} "
              f"{r.get('tp', 0):>5} {r.get('fp', 0):>5} {r.get('fn', 0):>5}")

    with open(REPORT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRapport sauvegarde : {REPORT_FILE.relative_to(PROJECT_ROOT)}")


# ===========================================================================
# Orchestration : Phases 2 + 3 par run, et modes batch
# ---------------------------------------------------------------------------
# process_run    : 1 run = generate_images_and_GT + evaluate_run
# evaluate_runs  : N runs filtres (mode "evaluate" du CLI)
# full_pipeline  : Phase 1 (generate_runs) puis N runs (mode "full" du CLI)
# ===========================================================================

def process_run(run_dir, runway=None, conf=0.25, imgsz=512,
                iou_thresh=0.5, iou_method="CIOU", xplane_dir=None):
    """Phase 2 + Phase 3 sur un seul run.

    Enchaine Export.generate_images_and_GT (rendu X-Plane + fautes + GT LARD)
    puis Detection_Evaluation.evaluate_run (YOLO + IoU).

    :return: dict de metriques pour le rapport, ou None si echec
    """
    import sys
    _root = Path(__file__).resolve().parent.parent
    for _p in (_root, _root / "project" / "export"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

    from Export import generate_images_and_GT
    from Detection_Evaluation import evaluate_run

    print(f"\n{'-' * 50}")
    print(f" Run : {run_dir.name}")
    print(f"{'-' * 50}")

    if not generate_images_and_GT(run_dir, xplane_dir, runway=runway):
        return None

    return evaluate_run(
        run_dir, runway=runway, conf=conf, imgsz=imgsz,
        iou_thresh=iou_thresh, iou_method=iou_method,
    )


def evaluate_runs(run_name=None, all_runs=False, runway=None,
                  conf=0.25, imgsz=512, iou_thresh=0.5, iou_method="CIOU",
                  xplane_dir=None):
    """Mode "evaluate" : Phases 2+3 sur les runs filtres."""
    from xplane_weather import reset_if_active

    print("=" * 60)
    print(" PHASES 2+3 : Images + GT (Export) -> Detection + IoU")
    print("=" * 60)

    runs = find_runs(run_name, all_runs)
    if not runs:
        print("[Pipeline] Aucun run valide trouve.")
        print(f"  Verifier que runs/<nom>/ contient un .yaml et "
              f"poses_cam_export.json ou footage/")
        return []

    print(f"\n[Pipeline] {len(runs)} run(s) a evaluer")

    all_results = []
    for run_dir in runs:
        result = process_run(run_dir, runway=runway, conf=conf, imgsz=imgsz,
                             iou_thresh=iou_thresh, iou_method=iou_method,
                             xplane_dir=xplane_dir)
        if result:
            all_results.append(result)

    reset_if_active(xplane_dir)
    aggregate_report(all_results)
    return all_results


def full_pipeline(nb_scenarios=None, quiet=False, conf=0.25, imgsz=512,
                  iou_thresh=0.5, iou_method="CIOU", xplane_dir=None):
    """Mode "full" : Phase 1 (generate_runs) puis Phases 2+3 par scenario."""
    from Generate import generate_runs
    from xplane_weather import reset_if_active

    created_runs = generate_runs(nb_scenarios=nb_scenarios, quiet=quiet)
    if not created_runs:
        print("[Pipeline] Aucun scenario genere, arret.")
        return []

    print(f"\n{'=' * 60}")
    print(" PHASES 2+3 : Images + GT (Export) -> Detection + IoU")
    print(f"{'=' * 60}")

    all_results = []
    for run_dir in created_runs:
        result = process_run(run_dir, conf=conf, imgsz=imgsz,
                             iou_thresh=iou_thresh, iou_method=iou_method,
                             xplane_dir=xplane_dir)
        if result:
            all_results.append(result)

    reset_if_active(xplane_dir)
    aggregate_report(all_results)
    return all_results
