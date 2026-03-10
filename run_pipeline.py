"""
run_pipeline.py — Orchestrateur bout-en-bout du pipeline LARD-LAAS-TAF
=======================================================================
Tout est centralise dans runs/<ICAO_RWY>/ :
  - .esp/.yaml generes par TAF
  - .zip GES (tu le poses la)
  - footage/ images/ predictions/ annotated/ eval_results.json (auto)

Modes :
    python run_pipeline.py generate -n 5          # Etape 1 : genere .esp/.yaml dans runs/
    python run_pipeline.py evaluate LFPO_24        # Etape 2 : GT + YOLO + IoU sur un run
    python run_pipeline.py evaluate --all          # Etape 2 : sur tous les runs
    python run_pipeline.py full -n 5               # Tout : generate + pause GES + evaluate
"""

import sys
import os
import json
import shutil
import zipfile
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
YOLO_DIR = ROOT / "yolo"
TAF_OUTPUT_DIR = ROOT / "output"  # TAF ecrit ici, on reorganise ensuite


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _find_runs(run_name=None, all_runs=False):
    """Trouve les runs evaluables (ont un .yaml + des images ou un .zip)."""
    runs = []

    if run_name:
        run_dir = RUNS_DIR / run_name
        if not run_dir.exists():
            # Tenter chemin absolu/relatif
            run_dir = Path(run_name).resolve()
        if run_dir.exists():
            runs = [run_dir]
        else:
            print(f"[ERREUR] Run introuvable : {run_name}")
            return []
    elif all_runs:
        if RUNS_DIR.exists():
            runs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()])

    valid = []
    for run_dir in runs:
        yamls = list(run_dir.glob("*.yaml"))
        has_zip = bool(list(run_dir.glob("*.zip")))
        footage = run_dir / "footage"
        has_images = footage.exists() and bool(
            list(footage.glob("*.jpeg")) + list(footage.glob("*.jpg")) + list(footage.glob("*.png"))
        )

        if not yamls:
            print(f"[SKIP] {run_dir.name} : pas de .yaml")
            continue
        if not has_zip and not has_images:
            print(f"[SKIP] {run_dir.name} : ni .zip ni images dans footage/")
            continue

        valid.append({
            "dir": run_dir,
            "name": run_dir.name,
            "yaml": yamls[0],
            "has_zip": has_zip,
            "has_images": has_images,
        })

    return valid


def _unzip_ges(run_dir):
    """Dezippe le .zip GES dans le run, extrait les images vers footage/."""
    zips = sorted(run_dir.glob("*.zip"))
    if not zips:
        return False

    footage_dir = run_dir / "footage"
    if footage_dir.exists() and list(footage_dir.glob("*.jpeg")):
        print(f"  [ZIP] footage/ existe deja avec des images, skip dezip")
        return True

    for zp in zips:
        print(f"  [ZIP] Extraction de {zp.name}...")
        with zipfile.ZipFile(zp, 'r') as zf:
            # Lister le contenu pour trouver les images
            image_files = [
                f for f in zf.namelist()
                if f.lower().endswith(('.jpeg', '.jpg', '.png'))
                and not f.startswith('__MACOSX')
            ]

            if not image_files:
                print(f"  [ZIP] Aucune image trouvee dans {zp.name}")
                continue

            footage_dir.mkdir(parents=True, exist_ok=True)

            for img_path in image_files:
                # Extraire juste le nom du fichier (ignorer les sous-dossiers du zip)
                img_name = Path(img_path).name
                target = footage_dir / img_name
                with zf.open(img_path) as src, open(target, 'wb') as dst:
                    dst.write(src.read())

            print(f"  [ZIP] {len(image_files)} images extraites dans footage/")

    # Nettoyer exported_images/ (cree par LARD export_labels, doublon de footage/)
    exported = run_dir / "exported_images"
    if exported.exists():
        shutil.rmtree(exported)

    return True


# ---------------------------------------------------------------------------
# Etape 1 : Generation TAF (.esp/.yaml) → runs/<ICAO_RWY>/
# ---------------------------------------------------------------------------

def step_generate(nb_scenarios=None, quiet=False):
    """Lance TAF puis reorganise les .esp/.yaml dans runs/."""
    print("=" * 60)
    print(" ETAPE 1 : Generation TAF")
    print("=" * 60)

    project_dir = ROOT / "project"
    sys.path.insert(0, str(project_dir))

    # Nettoyer output/ avant generation (sinon les anciens .esp s'accumulent)
    if TAF_OUTPUT_DIR.exists():
        shutil.rmtree(TAF_OUTPUT_DIR)

    from Generate import run
    run(nb_test_cases=nb_scenarios, verbose=not quiet)

    # Reorganiser : copier les .esp/.yaml de output/ vers runs/<ICAO_RWY>/
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    esp_files = list(TAF_OUTPUT_DIR.rglob("*.esp"))
    yaml_files = list(TAF_OUTPUT_DIR.rglob("*.yaml"))

    created_runs = []
    for esp in esp_files:
        name = esp.stem  # ex: LFPO_24

        # Suffixe auto si le dossier existe deja (meme piste generee 2x)
        run_dir = RUNS_DIR / name
        if run_dir.exists():
            idx = 2
            while (RUNS_DIR / f"{name}_{idx:03d}").exists():
                idx += 1
            run_dir = RUNS_DIR / f"{name}_{idx:03d}"

        run_dir.mkdir(parents=True, exist_ok=True)

        # Copier .esp
        shutil.copy2(esp, run_dir / esp.name)

        # Trouver le .yaml correspondant
        matching_yaml = [y for y in yaml_files if y.stem == name]
        if matching_yaml:
            shutil.copy2(matching_yaml[0], run_dir / matching_yaml[0].name)

        # Copier le .xml des parametres du scenario (ex: scenario_0.xml)
        scenario_xml = esp.parent.parent / f"{esp.parent.parent.name}.xml"
        if scenario_xml.exists():
            shutil.copy2(scenario_xml, run_dir / "params.xml")

        created_runs.append(run_dir)
        print(f"  [RUNS] {run_dir.name}/ <- .esp + .yaml")

    print(f"\n[Pipeline] {len(created_runs)} run(s) cree(s) dans runs/")
    print(f"  Prochaine etape : importer les .esp dans GES, poser les .zip dans runs/<nom>/")

    return created_runs


# ---------------------------------------------------------------------------
# Etape 2 : Ground truth LARD (.csv)
# ---------------------------------------------------------------------------

def step_generate_gt(run_info):
    """Genere le CSV ground truth LARD pour un run."""
    run_dir = run_info["dir"]
    yaml_path = run_info["yaml"]

    print(f"\n  [GT] Generation CSV pour {run_info['name']}...")

    # Import LARD + monkey-patch
    lard_path = str(ROOT / "LARD")
    export_path = str(ROOT / "project" / "export")
    if lard_path not in sys.path:
        sys.path.insert(0, lard_path)
    if export_path not in sys.path:
        sys.path.insert(0, export_path)

    import src.labeling.label_export as _le
    _le.runway_is_facing_us = lambda *args, **kwargs: True

    from lard_bridge import generate_labels_csv

    csv_file = generate_labels_csv(
        yaml_path=str(yaml_path),
        dataset_dir=str(run_dir),
    )
    return Path(csv_file)


# ---------------------------------------------------------------------------
# Etape 3 : YOLO prediction
# ---------------------------------------------------------------------------

def step_predict(run_info, conf=0.25, imgsz=512):
    """Lance YOLO sur les images d'un run, sortie dans le meme dossier."""
    run_dir = run_info["dir"]
    footage_dir = run_dir / "footage"

    n_images = len(
        list(footage_dir.glob("*.jpeg")) + list(footage_dir.glob("*.jpg")) + list(footage_dir.glob("*.png"))
    )
    print(f"\n  [YOLO] Prediction sur {n_images} images ({run_info['name']})...")

    yolo_path = str(YOLO_DIR)
    if yolo_path not in sys.path:
        sys.path.insert(0, yolo_path)

    from predict import predict

    predictions_dir, annotated_dir = predict(
        images_dir=footage_dir,
        conf=conf,
        imgsz=imgsz,
        output_dir=run_dir,
    )

    if predictions_dir and predictions_dir.exists():
        n_labels = len(list(predictions_dir.glob("*.txt")))
        print(f"  [YOLO] {n_labels} labels generes dans predictions/")
    else:
        print(f"  [YOLO] ATTENTION : pas de labels generes")

    return predictions_dir


# ---------------------------------------------------------------------------
# Etape 4 : Evaluation IoU
# ---------------------------------------------------------------------------

def step_evaluate(run_dir, predictions_dir, csv_path, runway=None, iou_thresh=0.5, iou_method="CIOU"):
    """Evalue IoU predictions vs GT."""
    print(f"\n  [EVAL] IoU evaluation...")

    eval_path = str(YOLO_DIR / "eval")
    yolo_path = str(YOLO_DIR)
    if eval_path not in sys.path:
        sys.path.insert(0, eval_path)
    if yolo_path not in sys.path:
        sys.path.insert(0, yolo_path)

    from evaluate import evaluate

    metrics = evaluate(
        labels_dir=predictions_dir,
        csv_path=csv_path,
        iou_thresh=iou_thresh,
        iou_method=iou_method,
        runway=runway,
        results_dir=run_dir,
    )
    return metrics


# ---------------------------------------------------------------------------
# Pipeline evaluate : dezip + GT + YOLO + IoU
# ---------------------------------------------------------------------------

def run_evaluate(run_name=None, all_runs=False, runway=None,
                 conf=0.25, imgsz=512, iou_thresh=0.5, iou_method="CIOU"):
    """Enchaine dezip → GT → YOLO → evaluation sur les runs specifies."""
    print("=" * 60)
    print(" PIPELINE : Dezip + GT + YOLO + Evaluation IoU")
    print("=" * 60)

    runs = _find_runs(run_name, all_runs)
    if not runs:
        print("[Pipeline] Aucun run valide trouve.")
        print(f"  Verifier que runs/<nom>/ contient un .yaml et un .zip ou footage/")
        return []

    print(f"\n[Pipeline] {len(runs)} run(s) a evaluer :")
    for r in runs:
        status = "images" if r["has_images"] else "zip"
        print(f"  {r['name']} ({status})")

    all_results = []

    for run_info in runs:
        run_dir = run_info["dir"]
        print(f"\n{'-' * 50}")
        print(f" Run : {run_info['name']}")
        print(f"{'-' * 50}")

        # Dezip si necessaire
        if run_info["has_zip"] and not run_info["has_images"]:
            if not _unzip_ges(run_dir):
                print(f"  [SKIP] Echec dezip pour {run_info['name']}")
                continue
            # Mettre a jour has_images
            run_info["has_images"] = True

        footage_dir = run_dir / "footage"
        if not footage_dir.exists() or not list(footage_dir.glob("*.jpeg")):
            print(f"  [SKIP] Pas d'images dans footage/ pour {run_info['name']}")
            continue

        # Extraire le runway du nom du run (ex: LFPO_24 → 24)
        rwy = runway
        if rwy is None:
            parts = run_info["name"].split("_")
            if len(parts) >= 2:
                rwy = parts[-1]

        # GT
        try:
            csv_path = step_generate_gt(run_info)
        except Exception as e:
            print(f"  [GT] ERREUR : {e}")
            csv_candidates = list(run_dir.glob("*_labels.csv"))
            if csv_candidates:
                csv_path = csv_candidates[0]
                print(f"  [GT] Utilisation du CSV existant : {csv_path.name}")
            else:
                print(f"  [SKIP] Pas de CSV GT pour {run_info['name']}")
                continue

        # YOLO
        try:
            predictions_dir = step_predict(run_info, conf=conf, imgsz=imgsz)
        except Exception as e:
            print(f"  [YOLO] ERREUR : {e}")
            continue

        if predictions_dir is None or not predictions_dir.exists():
            print(f"  [SKIP] Pas de predictions pour {run_info['name']}")
            continue

        # Eval
        try:
            metrics = step_evaluate(
                run_dir=run_dir,
                predictions_dir=predictions_dir,
                csv_path=csv_path,
                runway=rwy,
                iou_thresh=iou_thresh,
                iou_method=iou_method,
            )
            if metrics:
                all_results.append({
                    "run": run_info["name"],
                    "runway": rwy,
                    **{k: (round(v, 4) if isinstance(v, float) else int(v) if isinstance(v, (int, float)) else v)
                       for k, v in metrics.items() if k in ("ap", "f1", "p", "r", "c", "tp", "fp", "fn")},
                })
        except Exception as e:
            print(f"  [EVAL] ERREUR : {e}")
            continue

    # Rapport final
    if all_results:
        print(f"\n{'=' * 60}")
        print(f" RAPPORT FINAL ({len(all_results)} run(s))")
        print(f"{'=' * 60}")
        print(f"{'Run':<20} {'AP':>6} {'F1':>6} {'P':>6} {'R':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
        print(f"{'-' * 60}")
        for r in all_results:
            print(f"{r['run']:<20} {r.get('ap',0):>6.3f} {r.get('f1',0):>6.3f} "
                  f"{r.get('p',0):>6.3f} {r.get('r',0):>6.3f} "
                  f"{r.get('tp',0):>5} {r.get('fp',0):>5} {r.get('fn',0):>5}")

        report_path = RUNS_DIR / "pipeline_report.json"
        with open(report_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nRapport sauvegarde : {report_path.relative_to(ROOT)}")

    return all_results


# ---------------------------------------------------------------------------
# Mode full : generate + pause GES + evaluate
# ---------------------------------------------------------------------------

def run_full(nb_scenarios=None, quiet=False, conf=0.25, imgsz=512,
             iou_thresh=0.5, iou_method="CIOU"):
    """Pipeline complet avec pause pour le rendu GES."""

    # Etape 1 : generation
    created_runs = step_generate(nb_scenarios=nb_scenarios, quiet=quiet)

    if not created_runs:
        print("[Pipeline] Aucun .esp genere, arret.")
        return

    # Pause GES
    print(f"\n{'=' * 60}")
    print(" ETAPE MANUELLE : Google Earth Studio")
    print(f"{'=' * 60}")
    print(f"\n1. Importer les .esp depuis runs/<nom>/ dans Google Earth Studio")
    print(f"2. Rendre les images")
    print(f"3. Poser le .zip telecharge dans runs/<nom>/")
    print()

    for run_dir in created_runs:
        esp_files = list(run_dir.glob("*.esp"))
        if esp_files:
            print(f"   {run_dir.name}/ ← poser le .zip ici")

    print()
    try:
        input("Appuyer sur Entree une fois les .zip GES en place...")
    except (EOFError, KeyboardInterrupt):
        print("\n[Pipeline] Arret. Relancer avec 'evaluate --all'.")
        return

    # Etape 2+ : evaluate all
    return run_evaluate(all_runs=True, conf=conf, imgsz=imgsz,
                        iou_thresh=iou_thresh, iou_method=iou_method)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline bout-en-bout LARD-LAAS-TAF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Structure runs/ :
  runs/
    LFPO_24/
      LFPO_24.esp           <- genere par 'generate'
      LFPO_24.yaml          <- genere par 'generate'
      LFPO_24.zip           <- tu poses le zip GES ici
      footage/              <- dezip auto
      gt_labels.csv         <- GT auto
      predictions/          <- YOLO auto
      annotated/            <- images annotees auto
      eval_results.json     <- metriques auto
    pipeline_report.json    <- rapport agrege

Exemples :
  python run_pipeline.py generate -n 5
  python run_pipeline.py evaluate LFPO_24
  python run_pipeline.py evaluate --all
  python run_pipeline.py full -n 3
        """,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- generate ---
    p_gen = sub.add_parser("generate", help="Genere les scenarios TAF (.esp/.yaml) dans runs/")
    p_gen.add_argument("-n", "--nb-scenarios", type=int, default=None,
                       help="Nombre de scenarios (surcharge settings.xml)")
    p_gen.add_argument("-q", "--quiet", action="store_true")

    # --- evaluate ---
    p_eval = sub.add_parser("evaluate", help="Dezip + GT + YOLO + IoU sur un ou tous les runs")
    p_eval.add_argument("run", nargs="?", default=None, help="Nom du run (ex: LFPO_24)")
    p_eval.add_argument("--all", action="store_true", dest="all_runs",
                        help="Evaluer tous les runs dans runs/")
    p_eval.add_argument("--runway", type=str, default=None, help="Filtrer GT sur une piste")
    p_eval.add_argument("--conf", type=float, default=0.25, help="Seuil confiance YOLO")
    p_eval.add_argument("--imgsz", type=int, default=512, help="Taille image YOLO")
    p_eval.add_argument("--iou-thresh", type=float, default=0.5, help="Seuil IoU")
    p_eval.add_argument("--iou-method", type=str, default="CIOU",
                        choices=["IOU", "GIOU", "DIOU", "CIOU"])

    # --- full ---
    p_full = sub.add_parser("full", help="Pipeline complet (generate + GES + evaluate)")
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

    elif args.mode == "evaluate":
        if not args.run and not args.all_runs:
            print("Specifier un run ou --all. Ex: python run_pipeline.py evaluate LFPO_24")
            return
        run_evaluate(
            run_name=args.run,
            all_runs=args.all_runs,
            runway=args.runway,
            conf=args.conf,
            imgsz=args.imgsz,
            iou_thresh=args.iou_thresh,
            iou_method=args.iou_method,
        )

    elif args.mode == "full":
        run_full(
            nb_scenarios=args.nb_scenarios,
            quiet=args.quiet,
            conf=args.conf,
            imgsz=args.imgsz,
            iou_thresh=args.iou_thresh,
            iou_method=args.iou_method,
        )


if __name__ == "__main__":
    main()
