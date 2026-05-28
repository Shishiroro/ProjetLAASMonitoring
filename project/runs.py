"""
runs.py — Structure et orchestration des dossiers runs/<generation>/<ICAO_RWY>/
==============================================================================
Centralise tout ce qui touche au cycle de vie des runs :

Layout :
  - runs/<generation>/<ICAO_RWY>/    (regroupement par batch)
  - <generation> = `generation_NN` (defaut) ou `<nom>_NN` (si --name fourni)
  - resolution des dossiers, enumeration des runs evaluables
  - creation des dossiers a partir de la sortie TAF (Phase 1)
  - agregation d'un rapport multi-run (dans la generation)

Orchestration (3 phases independantes) :
  - render_runs           : mode "render"        (Phase 2 : X-Plane + fautes)
  - evaluate_runs         : mode "evaluate"      (Phase 3 : YOLO + IoU)
  - full_pipeline         : mode "full"          (Phase 1 + 2 enchainees)
  - full_evaluate_pipeline: mode "full_evaluate" (Phase 1 + 2 + 3 enchainees)
"""

import json
import re
import shutil
from pathlib import Path

# --- Chemins ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
TAF_OUTPUT_DIR = PROJECT_ROOT / "output"

DEFAULT_GENERATION_PREFIX = "generation"
IMAGE_EXTS = (".jpeg", ".jpg", ".png")


# ===========================================================================
# Generations : <prefix>_NN sous runs/
# ===========================================================================

def _generation_prefix(name=None):
    """Prefixe a utiliser : 'generation' si pas de nom, sinon le nom donne."""
    return DEFAULT_GENERATION_PREFIX if not name else name


def next_generation_dir(name=None):
    """Retourne le prochain dossier de generation disponible (non cree).

    Sans nom : runs/generation_01, generation_02, ...
    Avec nom : runs/<nom>_01, <nom>_02, ...
    """
    prefix = _generation_prefix(name)
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")

    used = set()
    if RUNS_DIR.exists():
        for d in RUNS_DIR.iterdir():
            if not d.is_dir():
                continue
            m = pattern.match(d.name)
            if m:
                used.add(int(m.group(1)))

    idx = 1
    while idx in used:
        idx += 1
    return RUNS_DIR / f"{prefix}_{idx:02d}"


def resolve_generation_dir(generation):
    """Retourne le Path d'une generation existante, ou None si introuvable."""
    if not generation:
        return None
    d = RUNS_DIR / generation
    return d if d.is_dir() else None


def clean_runs_dir():
    """Supprime tout le contenu de runs/ (utilise par --clean)."""
    if not RUNS_DIR.exists():
        return
    for child in RUNS_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    print(f"[Pipeline] runs/ vide.")


# ===========================================================================
# Helpers images / lookup
# ===========================================================================

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


def _scan_all_run_dirs():
    """Scan complet de runs/ : descend d'un niveau dans chaque generation.

    Layout attendu : runs/<generation>/<run>/. Un sous-dossier n'est retenu
    que s'il contient un .yaml, ce qui filtre les dossiers post-traitement
    (dataset/, dataset_regroup/, ...).
    """
    if not RUNS_DIR.exists():
        return []
    found = []
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        for sub in sorted(d.iterdir()):
            if sub.is_dir() and list(sub.glob("*.yaml")):
                found.append(sub)
    return found


def find_runs(run_name=None, all_runs=False, generation=None):
    """Trouve les runs evaluables (yaml + poses ou images).

    Resolution du `run_name` :
      - "gen/run"                : chemin compose, split sur / ou \\
      - run_name + generation    : cible runs/<generation>/<run_name>/
      - chemin absolu existant   : utilise tel quel
      - nom seul sans generation : cherche dans toutes les generations ; si
                                   trouve dans une seule, on prend ; si multiple,
                                   erreur (demander --generation)

    Modes `all_runs` :
      - + generation             : scanne runs/<generation>/
      - sans generation          : scanne tout runs/ (toutes les generations)
                                   — surtout utile en mode notebook

    Le CLI applique une regle supplementaire (`--all requiert --generation`)
    AVANT d'appeler cette fonction (cf. run_pipeline.py), pour ne pas mixer
    plusieurs batchs par inadvertance.

    :return: list[Path]
    """
    if run_name:
        # Chemin compose "gen_01/LFPO_24" ?
        if "/" in run_name or "\\" in run_name:
            rel = Path(run_name)
            candidates = [RUNS_DIR / rel]
        elif generation:
            candidates = [RUNS_DIR / generation / run_name]
        else:
            # Chemin absolu eventuellement
            p = Path(run_name)
            if p.is_absolute() and p.exists():
                candidates = [p]
            else:
                # Recherche par nom dans toutes les generations
                matches = [d for d in _scan_all_run_dirs() if d.name == run_name]
                if len(matches) > 1:
                    print(f"[ERREUR] '{run_name}' trouve dans plusieurs emplacements :")
                    for m in matches:
                        print(f"  - {m.relative_to(PROJECT_ROOT)}")
                    print("  Specifier --generation ou utiliser le chemin compose '<gen>/<run>'.")
                    return []
                candidates = matches
    elif all_runs:
        if generation:
            gen_dir = resolve_generation_dir(generation)
            if not gen_dir:
                print(f"[ERREUR] Generation introuvable : runs/{generation}/")
                return []
            candidates = sorted(d for d in gen_dir.iterdir() if d.is_dir())
        else:
            # Pas de generation : scan global (notebook-friendly)
            candidates = _scan_all_run_dirs()
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


def create_runs_from_taf_output(generation_dir):
    """Reorganise la sortie TAF (output/) vers <generation_dir>/<ICAO_RWY>/.

    TAF ecrit dans output/test_case_*/test_artifact_*/, un dossier "plat" non
    indexable par nom de piste. On regroupe le .yaml + ses JSON compagnons
    sous <generation_dir>/<ICAO_RWY>/ (le pipeline travaille a ce niveau-la).

    Pour chaque .yaml trouve dans output/, on copie :
      - <name>.yaml                (renomme depuis le stem)
      - poses_cam_export.json      (scenario_name patche si suffixe applique)
      - fault_profile.json         (optionnel)
      - weather_profile.json       (optionnel)

    Si une meme piste est generee plusieurs fois (collisions de seed ou
    re-generation), on suffixe _002, _003, ... pour eviter d'ecraser.

    :param generation_dir: dossier cible (ex: runs/generation_01/)
    :return: list[Path] des runs crees
    """
    generation_dir = Path(generation_dir)
    generation_dir.mkdir(parents=True, exist_ok=True)
    yaml_files = list(TAF_OUTPUT_DIR.rglob("*.yaml"))

    created_runs = []
    for yf in yaml_files:
        name = yf.stem  # ex: LFPO_24

        # Suffixe auto si le dossier existe deja (meme piste generee 2x)
        run_dir = generation_dir / name
        if run_dir.exists():
            idx = 2
            while (generation_dir / f"{name}_{idx:03d}").exists():
                idx += 1
            run_dir = generation_dir / f"{name}_{idx:03d}"

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
        rel = run_dir.relative_to(PROJECT_ROOT)
        print(f"  [RUNS] {rel}/ <- .yaml + poses_cam_export.json{extras}")

    gen_rel = generation_dir.relative_to(PROJECT_ROOT)
    print(f"\n[Pipeline] {len(created_runs)} run(s) cree(s) dans {gen_rel}/")
    return created_runs


def aggregate_report(results, generation_dir=None):
    """Affiche le rapport agrege et sauve dans <generation_dir>/pipeline_report.json.

    Si generation_dir est None, deduit depuis le parent du premier run_dir.
    """
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

    if generation_dir is None:
        first_run = Path(results[0].get("run_dir", "")) if results[0].get("run_dir") else None
        if first_run and first_run.parent.parent == RUNS_DIR:
            generation_dir = first_run.parent
        else:
            generation_dir = RUNS_DIR
    generation_dir = Path(generation_dir)
    generation_dir.mkdir(parents=True, exist_ok=True)
    report_file = generation_dir / "pipeline_report.json"

    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRapport sauvegarde : {report_file.relative_to(PROJECT_ROOT)}")


# ===========================================================================
# Orchestration : 3 phases independantes
# ---------------------------------------------------------------------------
# render_runs            : Phase 2 (X-Plane + fautes) sur N runs filtres
# evaluate_runs          : Phase 3 (YOLO + IoU)       sur N runs filtres
# full_pipeline          : Phase 1 + 2     enchainees (mode "full" du CLI)
# full_evaluate_pipeline : Phase 1 + 2 + 3 enchainees (mode "full_evaluate")
# ===========================================================================

def _render_loop(runs, xplane_dir):
    """Boucle Phase 2 partagee par render_runs et les pipelines full.

    Itere sur une liste de runs deja resolus, appelle Export.render_run pour
    chacun, et reset la meteo a la fin. Retourne la sous-liste des runs
    effectivement rendus.
    """
    from Export import render_run
    from xplane_weather import reset_if_active

    rendered = []
    for run_dir in runs:
        print(f"\n{'-' * 50}")
        print(f" Run : {run_dir.parent.name}/{run_dir.name}")
        print(f"{'-' * 50}")
        if render_run(run_dir, xplane_dir):
            rendered.append(run_dir)
    reset_if_active(xplane_dir)
    return rendered


def _evaluate_loop(runs, runway=None, conf=0.25, imgsz=512,
                   iou_thresh=0.5, iou_method="CIOU"):
    """Boucle Phase 3 partagee par evaluate_runs et full_evaluate_pipeline."""
    from Detection_Evaluation import evaluate_run

    all_results = []
    for run_dir in runs:
        print(f"\n{'-' * 50}")
        print(f" Run : {run_dir.parent.name}/{run_dir.name}")
        print(f"{'-' * 50}")
        result = evaluate_run(run_dir, runway=runway, conf=conf, imgsz=imgsz,
                              iou_thresh=iou_thresh, iou_method=iou_method)
        if result:
            result.setdefault("run_dir", str(run_dir))
            all_results.append(result)
    return all_results


def render_runs(run_name=None, all_runs=False, generation=None, xplane_dir=None):
    """Mode "render" : Phase 2 (X-Plane + fautes) sur les runs filtres.

    :return: list[Path] des runs rendus avec succes
    """
    print("=" * 60)
    print(" PHASE 2 : Rendu X-Plane + fautes capteur")
    print("=" * 60)

    runs = find_runs(run_name, all_runs, generation=generation)
    if not runs:
        print("[Pipeline] Aucun run valide trouve.")
        return []

    print(f"\n[Pipeline] {len(runs)} run(s) a rendre")
    return _render_loop(runs, xplane_dir)


def evaluate_runs(run_name=None, all_runs=False, generation=None, runway=None,
                  conf=0.25, imgsz=512, iou_thresh=0.5, iou_method="CIOU",
                  xplane_dir=None):
    """Mode "evaluate" : Phase 3 (YOLO + IoU) sur les runs filtres.

    `xplane_dir` est conserve pour compat de signature mais inutilise
    (Phase 3 ne touche pas a X-Plane).
    """
    print("=" * 60)
    print(" PHASE 3 : Detection YOLO + IoU")
    print("=" * 60)

    runs = find_runs(run_name, all_runs, generation=generation)
    if not runs:
        print("[Pipeline] Aucun run valide trouve.")
        print(f"  Verifier que runs/<generation>/<nom>/ contient un .yaml et "
              f"poses_cam_export.json ou footage/")
        return []

    print(f"\n[Pipeline] {len(runs)} run(s) a evaluer")
    all_results = _evaluate_loop(runs, runway=runway, conf=conf, imgsz=imgsz,
                                 iou_thresh=iou_thresh, iou_method=iou_method)
    aggregate_report(all_results, generation_dir=runs[0].parent)
    return all_results


def full_pipeline(nb_scenarios=None, name=None, clean=False,
                  xplane_dir=None):
    """Mode "full" : Phase 1 + Phase 2 enchainees (sans evaluation YOLO)."""
    from Generate import generate_runs

    created_runs = generate_runs(nb_scenarios=nb_scenarios,
                                 name=name, clean=clean)
    if not created_runs:
        print("[Pipeline] Aucun scenario genere, arret.")
        return []

    print(f"\n{'=' * 60}")
    print(" PHASE 2 : Rendu X-Plane + fautes capteur")
    print(f"{'=' * 60}")
    return _render_loop(created_runs, xplane_dir)


def full_evaluate_pipeline(nb_scenarios=None, name=None, clean=False,
                           conf=0.25, imgsz=512, iou_thresh=0.5, iou_method="CIOU",
                           xplane_dir=None):
    """Mode "full_evaluate" : Phase 1 + 2 + 3 enchainees sur les runs crees."""
    from Generate import generate_runs

    created_runs = generate_runs(nb_scenarios=nb_scenarios,
                                 name=name, clean=clean)
    if not created_runs:
        print("[Pipeline] Aucun scenario genere, arret.")
        return []

    generation_dir = created_runs[0].parent

    print(f"\n{'=' * 60}")
    print(" PHASE 2 : Rendu X-Plane + fautes capteur")
    print(f"{'=' * 60}")
    rendered = _render_loop(created_runs, xplane_dir)

    if not rendered:
        print("[Pipeline] Aucun rendu reussi, arret avant Phase 3.")
        return []

    print(f"\n{'=' * 60}")
    print(" PHASE 3 : Detection YOLO + IoU")
    print(f"{'=' * 60}")
    all_results = _evaluate_loop(rendered, conf=conf, imgsz=imgsz,
                                 iou_thresh=iou_thresh, iou_method=iou_method)
    aggregate_report(all_results, generation_dir=generation_dir)
    return all_results
