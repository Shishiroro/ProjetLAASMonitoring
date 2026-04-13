"""
run_pipeline.py — Orchestrateur bout-en-bout du pipeline LARD-LAAS-TAF
=======================================================================
Tout est centralise dans runs/<ICAO_RWY>/ :
  - .esp/.yaml/poses_cam_export.json generes par TAF
  - .zip GES (renderer=ges) OU images X-Plane (renderer=xplane)
  - footage/ images/ predictions.csv/ annotated/ eval_results.json (auto)

Renderers :
    --renderer ges     : Google Earth Studio (defaut, etape manuelle)
    --renderer xplane  : X-Plane 12 (rendu automatise, necessite --xplane-dir)

Modes :
    python run_pipeline.py generate -n 5              # Genere .esp/.yaml dans runs/
    python run_pipeline.py evaluate LFPO_24            # GT + YOLO + IoU sur un run
    python run_pipeline.py evaluate --all              # Sur tous les runs
    python run_pipeline.py full -n 5                   # Generate + GES + evaluate
    python run_pipeline.py full -n 100 --renderer xplane --xplane-dir "C:/X-Plane 12"
"""

import os
import sys
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

def reciprocal_runway(rwy):
    """Retourne le nom reciproque d'une piste (ex: 28L -> 10R, 09R -> 27L)."""
    import re
    m = re.match(r"^(\d{1,2})([LRC]?)$", rwy)
    if not m:
        return rwy
    num = int(m.group(1))
    suffix = m.group(2)
    recip_num = (num + 18) % 36
    if recip_num == 0:
        recip_num = 36
    recip_suffix = {"L": "R", "R": "L", "C": "C", "": ""}.get(suffix, suffix)
    return f"{recip_num:02d}{recip_suffix}"


def _find_runs(run_name=None, all_runs=False, renderer="ges"):
    """Trouve les runs evaluables (ont un .yaml + des images ou un .zip ou poses_cam_export.json)."""
    runs = []

    if run_name:
        run_dir = RUNS_DIR / run_name
        if not run_dir.exists():
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
        has_poses = (run_dir / "poses_cam_export.json").exists()
        footage = run_dir / "footage"
        has_images = footage.exists() and bool(
            list(footage.glob("*.jpeg")) + list(footage.glob("*.jpg")) + list(footage.glob("*.png"))
        )

        if not yamls:
            print(f"[SKIP] {run_dir.name} : pas de .yaml")
            continue

        # Pour xplane, on accepte les runs avec poses_cam_export.json (pas encore rendus)
        if renderer == "xplane":
            if not has_poses and not has_images:
                print(f"[SKIP] {run_dir.name} : ni poses_cam_export.json ni images dans footage/")
                continue
        else:
            if not has_zip and not has_images:
                print(f"[SKIP] {run_dir.name} : ni .zip ni images dans footage/")
                continue

        valid.append({
            "dir": run_dir,
            "name": run_dir.name,
            "yaml": yamls[0],
            "has_zip": has_zip,
            "has_poses": has_poses,
            "has_images": has_images,
        })

    return valid


def _unzip_ges(run_dir):
    """Dezippe le .zip GES dans le run, extrait les images vers footage/.
    Renomme les images si leur prefixe ne correspond pas au nom du dossier
    (cas d'un suffixe _002, _003... quand la meme piste est generee 2x)."""
    zips = sorted(run_dir.glob("*.zip"))
    if not zips:
        return False

    footage_dir = run_dir / "footage"
    if footage_dir.exists() and list(footage_dir.glob("*.jpeg")):
        print(f"  [ZIP] footage/ existe deja avec des images, skip dezip")
        return True

    run_name = run_dir.name  # ex: LFPG_09L_002

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

            # Detecter le prefixe des images dans le zip (ex: LFPG_09L)
            first_img = Path(image_files[0]).name
            # Prefixe = tout avant le dernier _NNN.ext
            import re
            m = re.match(r"^(.+)_\d+\.\w+$", first_img)
            zip_prefix = m.group(1) if m else None

            needs_rename = zip_prefix and zip_prefix != run_name

            for img_path in image_files:
                img_name = Path(img_path).name
                if needs_rename:
                    img_name = img_name.replace(zip_prefix, run_name, 1)
                target = footage_dir / img_name
                with zf.open(img_path) as src, open(target, 'wb') as dst:
                    dst.write(src.read())

            if needs_rename:
                print(f"  [ZIP] Images renommees : {zip_prefix}_* -> {run_name}_*")
            print(f"  [ZIP] {len(image_files)} images extraites dans footage/")

    # Nettoyer exported_images/ (cree par LARD export_labels, doublon de footage/)
    exported = run_dir / "exported_images"
    if exported.exists():
        shutil.rmtree(exported)

    return True


# ---------------------------------------------------------------------------
# Etape 1b : Rendu X-Plane (poses_cam_export.json → footage/)
# ---------------------------------------------------------------------------

def step_render_xplane(run_info, xplane_dir):
    """Rend les images d'un run via X-Plane 12.

    Lit poses_cam_export.json, injecte les poses dans X-Plane, capture les screenshots.
    Les images sont sauvees dans footage/.

    :param run_info: dict avec dir, name, etc.
    :param xplane_dir: chemin du repertoire X-Plane 12
    :return: True si les images ont ete rendues, False sinon
    """
    run_dir = run_info["dir"]
    poses_file = run_dir / "poses_cam_export.json"
    footage_dir = run_dir / "footage"

    if not poses_file.exists():
        print(f"  [XPLANE] Pas de poses_cam_export.json pour {run_info['name']}")
        return False

    # Skip si footage/ existe deja avec des images
    if footage_dir.exists():
        imgs = (
            list(footage_dir.glob("*.png"))
            + list(footage_dir.glob("*.jpeg"))
            + list(footage_dir.glob("*.jpg"))
        )
        if imgs:
            print(f"  [XPLANE] footage/ existe deja ({len(imgs)} images), skip rendu")
            return True

    print(f"\n  [XPLANE] Rendu de {run_info['name']}...")

    export_path = str(ROOT / "project" / "export")
    if export_path not in sys.path:
        sys.path.insert(0, export_path)

    from xplane_bridge import render_scenario, XPlaneConfig

    config = XPlaneConfig(xplane_dir=xplane_dir)

    # Passer le profil meteo si present
    weather_file = run_dir / "weather_profile.json"
    weather_path = str(weather_file) if weather_file.exists() else None

    render_scenario(str(poses_file), str(footage_dir), config,
                    weather_profile_path=weather_path)
    return True


# ---------------------------------------------------------------------------
# Etape 1 : Generation TAF (.esp/.yaml) → runs/<ICAO_RWY>/
# ---------------------------------------------------------------------------

def step_generate(nb_scenarios=None, quiet=False, renderer="ges"):
    """Lance TAF puis reorganise les .esp/.yaml dans runs/."""
    print("=" * 60)
    print(f" ETAPE 1 : Generation TAF (renderer={renderer})")
    print("=" * 60)

    project_dir = ROOT / "project"
    sys.path.insert(0, str(project_dir))

    # Passer le renderer a Export.py via variable d'environnement
    os.environ["LARD_RENDERER"] = renderer

    # Nettoyer output/ avant generation (sinon les anciens .esp s'accumulent)
    if TAF_OUTPUT_DIR.exists():
        shutil.rmtree(TAF_OUTPUT_DIR)

    from Generate import run
    run(nb_test_cases=nb_scenarios, verbose=not quiet)

    # Reorganiser : copier les fichiers de output/ vers runs/<ICAO_RWY>/
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    yaml_files = list(TAF_OUTPUT_DIR.rglob("*.yaml"))
    esp_files = list(TAF_OUTPUT_DIR.rglob("*.esp"))

    # Iterer sur les .yaml (present avec tout renderer) pour trouver les scenarios
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
        run_name = run_dir.name  # ex: LFPG_09L ou LFPG_09L_002

        # Copier .yaml (renomme si suffixe)
        shutil.copy2(yf, run_dir / f"{run_name}.yaml")

        # Copier .esp si present (GES seulement)
        matching_esp = [e for e in esp_files if e.stem == name]
        if matching_esp:
            shutil.copy2(matching_esp[0], run_dir / f"{run_name}.esp")

        # Copier le .xml des parametres du scenario (ex: scenario_0.xml)
        scenario_xml = yf.parent.parent / f"{yf.parent.parent.name}.xml"
        if scenario_xml.exists():
            shutil.copy2(scenario_xml, run_dir / "params_trace.xml")

        # Copier poses_cam_export.json (format universel pour renderers)
        # Mettre a jour scenario_name si le dossier a un suffixe (_002, _003...)
        poses_json = yf.parent / "poses_cam_export.json"
        if poses_json.exists():
            poses_dst = run_dir / "poses_cam_export.json"
            shutil.copy2(poses_json, poses_dst)
            if run_name != name:
                import json as _json
                poses_data = _json.load(open(poses_dst))
                poses_data["scenario_name"] = run_name
                with open(poses_dst, "w") as pf:
                    _json.dump(poses_data, pf, indent=2)

        # Copier fault_profile.json (si fautes capteur actives)
        fault_json = yf.parent / "fault_profile.json"
        if fault_json.exists():
            shutil.copy2(fault_json, run_dir / "fault_profile.json")

        # Copier weather_profile.json (si effets meteo actifs)
        weather_json = yf.parent / "weather_profile.json"
        if weather_json.exists():
            shutil.copy2(weather_json, run_dir / "weather_profile.json")

        # Sauver le renderer utilise dans un fichier de config
        run_config = {"renderer": renderer}
        with open(run_dir / "renderer_choice.json", "w") as f:
            json.dump(run_config, f, indent=2)

        created_runs.append(run_dir)
        extras = ""
        if fault_json.exists():
            extras += " + fault_profile.json"
        if weather_json.exists():
            extras += " + weather_profile.json"
        if renderer == "ges":
            print(f"  [RUNS] {run_dir.name}/ <- .esp + .yaml + poses_cam_export.json{extras}")
        else:
            print(f"  [RUNS] {run_dir.name}/ <- .yaml + poses_cam_export.json{extras}")

    print(f"\n[Pipeline] {len(created_runs)} run(s) cree(s) dans runs/")
    if renderer == "ges":
        print(f"  Prochaine etape : importer les .esp dans GES, poser les .zip dans runs/<nom>/")
    else:
        print(f"  Prochaine etape : lancer le rendu X-Plane (run_pipeline.py full --renderer xplane)")

    return created_runs


# ---------------------------------------------------------------------------
# Etape 2 : Ground truth LARD (.csv)
# ---------------------------------------------------------------------------

def step_generate_gt(run_info, renderer="ges"):
    """Genere le CSV ground truth pour un run."""
    run_dir = run_info["dir"]
    yaml_path = run_info["yaml"]

    print(f"\n  [GT] Generation CSV pour {run_info['name']}...")

    # Pipeline LARD pour GES et X-Plane
    lard_path = str(ROOT / "LARD")
    export_path = str(ROOT / "project" / "export")
    if lard_path not in sys.path:
        sys.path.insert(0, lard_path)
    if export_path not in sys.path:
        sys.path.insert(0, export_path)

    # Fix convention yaw : +180 pour passer du cap de vol au sens de regard
    # (fix envisage par LARD, ligne 132 commentee dans label_export.py)
    import src.labeling.label_export as _le
    _original_facing = _le.runway_is_facing_us
    def _fixed_facing(heading, runway):
        return _original_facing((heading + 180) % 360, runway)
    _le.runway_is_facing_us = _fixed_facing

    from lard_bridge import generate_labels_csv

    csv_file = generate_labels_csv(
        yaml_path=str(yaml_path),
        dataset_dir=str(run_dir),
        renderer=renderer,
    )

    # Restaurer
    _le.runway_is_facing_us = _original_facing

    return Path(csv_file)


# ---------------------------------------------------------------------------
# Etape 2b : Annotations visuelles GT LARD (echantillon)
# ---------------------------------------------------------------------------

def step_annotate_lard(run_info, csv_path, max_images=0, target_runway=None):
    """Dessine les bbox GT LARD sur les images dans annotated_lard/ (0 = toutes)."""
    import csv as csvmod
    from PIL import Image, ImageDraw, ImageFont

    run_dir = run_info["dir"]
    footage_dir = run_dir / "footage"
    out_dir = run_dir / "annotated_lard"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Police lisible (taille adaptee a l'image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    # Charger le CSV
    entries = {}
    with open(csv_path, "r") as f:
        reader = csvmod.DictReader(f, delimiter=";")
        for row in reader:
            filename = Path(row["image"]).name
            corners = (
                (int(float(row["x_TR"])), int(float(row["y_TR"]))),
                (int(float(row["x_TL"])), int(float(row["y_TL"]))),
                (int(float(row["x_BL"])), int(float(row["y_BL"]))),
                (int(float(row["x_BR"])), int(float(row["y_BR"]))),
            )
            runway = row.get("runway", "?")
            # Filtrer sur la piste cible si specifie
            # Comparer sans zero-padding (03 == 3, 09R == 9R)
            # Aussi accepter le reciprocal (10L == 28R pour le meme strip)
            if target_runway:
                norm_target = target_runway.lstrip("0") or "0"
                norm_runway = str(runway).lstrip("0") or "0"
                # Accepter la cible OU son reciprocal (meme bande physique)
                recip_target = reciprocal_runway(target_runway).lstrip("0") or "0"
                if norm_runway != norm_target and norm_runway != recip_target:
                    continue
            if filename not in entries:
                entries[filename] = []
            entries[filename].append({"corners": corners, "runway": runway})

    COLORS = ["cyan", "red", "yellow", "lime", "magenta", "orange"]
    runway_colors = {}
    color_idx = 0
    processed = 0

    # Nom a afficher : la piste d'approche (pas le reciprocal LARD)
    run_name = run_info.get("name", "")
    # Extraire le runway du nom de run (ex: KPDX_21 -> 21, KPDX_28L -> 28L)
    display_runway = run_name.split("_", 1)[1] if "_" in run_name else None

    for filename in sorted(entries.keys()):
        if max_images > 0 and processed >= max_images:
            break
        img_path = footage_dir / filename
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for entry in entries[filename]:
            rwy = display_runway or entry["runway"]
            if rwy not in runway_colors:
                runway_colors[rwy] = COLORS[color_idx % len(COLORS)]
                color_idx += 1
            color = runway_colors[rwy]
            c = entry["corners"]
            for i in range(4):
                draw.line([c[i], c[(i + 1) % 4]], fill=color, width=1)
            # Label avec fond noir pour lisibilite
            cx = sum(p[0] for p in c) // 4
            cy = min(p[1] for p in c) - 25  # au-dessus de la bbox
            bbox = draw.textbbox((cx, cy), rwy, font=font)
            draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill="black")
            draw.text((cx, cy), rwy, fill=color, font=font)

        img.save(out_dir / f"gt_{filename}")
        processed += 1

    print(f"  [GT-VIS] {processed} images annotees dans annotated_lard/")


# ---------------------------------------------------------------------------
# Etape 2c : Application des fautes capteur
# ---------------------------------------------------------------------------

def step_apply_faults(run_info):
    """Applique les fautes capteur aux images si fault_profile.json existe.
    Les images degradees vont dans degraded/, les originales restent dans footage/.
    Retourne le dossier a utiliser pour YOLO (degraded/ ou footage/)."""
    run_dir = run_info["dir"]
    fault_json = run_dir / "fault_profile.json"
    footage_dir = run_dir / "footage"
    degraded_dir = run_dir / "degraded"

    if not fault_json.exists():
        return footage_dir

    # Si degraded/ existe deja avec des images, skip
    if degraded_dir.exists() and (
        list(degraded_dir.glob("*.jpeg"))
        + list(degraded_dir.glob("*.jpg"))
        + list(degraded_dir.glob("*.png"))
    ):
        print(f"  [FAULTS] degraded/ existe deja, skip application")
        return degraded_dir

    print(f"\n  [FAULTS] Application des fautes capteur ({run_info['name']})...")

    export_path = str(ROOT / "project" / "export")
    if export_path not in sys.path:
        sys.path.insert(0, export_path)

    from sensor_faults import load_fault_profile, apply_faults_to_directory

    faults, n_frames = load_fault_profile(fault_json)
    if not faults:
        print(f"  [FAULTS] Aucune faute dans le profil, skip")
        return footage_dir

    fault_str = ", ".join(f"{f.fault_type}({f.severity:.2f})" for f in faults)
    print(f"  [FAULTS] Fautes : {fault_str}")

    apply_faults_to_directory(footage_dir, degraded_dir, faults, n_frames)
    return degraded_dir


# ---------------------------------------------------------------------------
# Etape 3 : YOLO prediction
# ---------------------------------------------------------------------------

def step_predict(run_info, conf=0.25, imgsz=512, images_dir=None):
    """Lance YOLO sur les images d'un run, sortie dans le meme dossier.
    :param images_dir: dossier d'images (defaut: footage/). Peut etre degraded/."""
    run_dir = run_info["dir"]
    footage_dir = images_dir or (run_dir / "footage")

    n_images = len(
        list(footage_dir.glob("*.jpeg")) + list(footage_dir.glob("*.jpg")) + list(footage_dir.glob("*.png"))
    )
    src_label = footage_dir.name
    print(f"\n  [YOLO] Prediction sur {n_images} images depuis {src_label}/ ({run_info['name']})...")

    yolo_path = str(YOLO_DIR)
    if yolo_path not in sys.path:
        sys.path.insert(0, yolo_path)

    from predict import predict

    predictions_csv, annotated_dir = predict(
        images_dir=footage_dir,
        conf=conf,
        imgsz=imgsz,
        output_dir=run_dir,
    )

    if predictions_csv and predictions_csv.exists():
        print(f"  [YOLO] Predictions dans {predictions_csv.name}")
    else:
        print(f"  [YOLO] ATTENTION : pas de predictions generees")

    return predictions_csv


# ---------------------------------------------------------------------------
# Etape 4 : Evaluation IoU
# ---------------------------------------------------------------------------

def step_evaluate(run_dir, predictions_csv, csv_path, runway=None, iou_thresh=0.5, iou_method="CIOU"):
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
        predictions_csv=predictions_csv,
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
                 conf=0.25, imgsz=512, iou_thresh=0.5, iou_method="CIOU",
                 renderer="ges", xplane_dir=None):
    """Enchaine [rendu] → dezip → GT → YOLO → evaluation sur les runs specifies."""
    print("=" * 60)
    print(f" PIPELINE : {'Render + ' if renderer == 'xplane' else ''}GT + YOLO + Evaluation IoU")
    print("=" * 60)

    runs = _find_runs(run_name, all_runs, renderer=renderer)
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

        # Obtenir les images selon le renderer
        if renderer == "xplane":
            # Rendu X-Plane si pas encore d'images
            if not run_info["has_images"]:
                if not step_render_xplane(run_info, xplane_dir or ""):
                    print(f"  [SKIP] Echec rendu X-Plane pour {run_info['name']}")
                    continue
                run_info["has_images"] = True
        else:
            # Dezip GES si necessaire
            if run_info["has_zip"] and not run_info["has_images"]:
                if not _unzip_ges(run_dir):
                    print(f"  [SKIP] Echec dezip pour {run_info['name']}")
                    continue
                run_info["has_images"] = True

        footage_dir = run_dir / "footage"
        if not footage_dir.exists() or not (
            list(footage_dir.glob("*.jpeg"))
            + list(footage_dir.glob("*.jpg"))
            + list(footage_dir.glob("*.png"))
        ):
            print(f"  [SKIP] Pas d'images dans footage/ pour {run_info['name']}")
            continue

        # Extraire le runway du nom du run (ex: LFPO_24 → 24, LFPG_09L_002 → 09L)
        rwy = runway
        if rwy is None:
            import re
            m = re.match(r"^[A-Z]{4}_(.+?)(?:_\d{3})?$", run_info["name"])
            if m:
                rwy = m.group(1)

        # Detecter le renderer du run (renderer_choice.json) ou utiliser celui passe
        run_renderer = renderer
        run_config_file = run_dir / "renderer_choice.json"
        if run_config_file.exists():
            with open(run_config_file) as f:
                run_cfg = json.load(f)
                run_renderer = run_cfg.get("renderer", renderer)

        # GT
        try:
            csv_path = step_generate_gt(run_info, renderer=run_renderer)
        except Exception as e:
            print(f"  [GT] ERREUR : {e}")
            csv_candidates = list(run_dir.glob("*_labels.csv"))
            if csv_candidates:
                csv_path = csv_candidates[0]
                print(f"  [GT] Utilisation du CSV existant : {csv_path.name}")
            else:
                print(f"  [SKIP] Pas de CSV GT pour {run_info['name']}")
                continue

        # Annotations visuelles GT LARD (echantillon)
        try:
            step_annotate_lard(run_info, csv_path, target_runway=rwy)
        except Exception as e:
            print(f"  [GT-VIS] ERREUR : {e}")

        # Fautes capteur (si fault_profile.json present)
        try:
            yolo_images_dir = step_apply_faults(run_info)
        except Exception as e:
            print(f"  [FAULTS] ERREUR : {e}")
            yolo_images_dir = run_dir / "footage"

        # YOLO
        try:
            predictions_csv = step_predict(run_info, conf=conf, imgsz=imgsz,
                                           images_dir=yolo_images_dir)
        except Exception as e:
            print(f"  [YOLO] ERREUR : {e}")
            continue

        if predictions_csv is None or not predictions_csv.exists():
            print(f"  [SKIP] Pas de predictions pour {run_info['name']}")
            continue

        # Eval
        try:
            metrics = step_evaluate(
                run_dir=run_dir,
                predictions_csv=predictions_csv,
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

    # Clear meteo final (une seule fois apres tous les scenarios)
    if renderer == "xplane":
        try:
            export_path = str(ROOT / "project" / "export")
            if export_path not in sys.path:
                sys.path.insert(0, export_path)
            from xplane_weather import reset_weather, set_exchange_dir
            if xplane_dir:
                set_exchange_dir(xplane_dir)
                reset_weather()
        except Exception:
            pass

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
             iou_thresh=0.5, iou_method="CIOU",
             renderer="ges", xplane_dir=None):
    """Pipeline complet : generate + rendu + evaluate.

    Avec GES : pause manuelle pour le rendu (import .esp, download .zip).
    Avec X-Plane : rendu automatise (poses_cam_export.json → screenshots → footage/).
    """

    # Etape 1 : generation
    created_runs = step_generate(nb_scenarios=nb_scenarios, quiet=quiet,
                                 renderer=renderer)

    if not created_runs:
        print("[Pipeline] Aucun scenario genere, arret.")
        return

    if renderer == "xplane":
        # Etape 2 : Rendu automatise X-Plane
        print(f"\n{'=' * 60}")
        print(" ETAPE 2 : Rendu X-Plane 12")
        print(f"{'=' * 60}")

        for run_dir in created_runs:
            run_info = {
                "dir": run_dir,
                "name": run_dir.name,
            }
            step_render_xplane(run_info, xplane_dir or "")

        # Clear meteo final apres tous les rendus
        try:
            export_path = str(ROOT / "project" / "export")
            if export_path not in sys.path:
                sys.path.insert(0, export_path)
            from xplane_weather import reset_weather, set_exchange_dir
            if xplane_dir:
                set_exchange_dir(xplane_dir)
                reset_weather()
        except Exception:
            pass

    else:
        # Etape 2 : Pause manuelle GES
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
                print(f"   {run_dir.name}/ <- poser le .zip ici")

        print()
        try:
            input("Appuyer sur Entree une fois les .zip GES en place...")
        except (EOFError, KeyboardInterrupt):
            print("\n[Pipeline] Arret. Relancer avec 'evaluate --all'.")
            return

    # Etape 3+ : evaluate all
    return run_evaluate(all_runs=True, conf=conf, imgsz=imgsz,
                        iou_thresh=iou_thresh, iou_method=iou_method,
                        renderer=renderer, xplane_dir=xplane_dir)


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
      LFPO_24.esp              <- genere par 'generate' (GES seulement)
      LFPO_24.yaml             <- genere par 'generate'
      poses_cam_export.json    <- poses camera (format universel)
      renderer_choice.json     <- renderer utilise (ges/xplane)
      params_trace.xml         <- parametres TAF du scenario
      footage/                 <- images (dezip GES ou rendu X-Plane)
      LFPO_24_labels.csv       <- GT LARD auto
      predictions.csv          <- predictions YOLO auto
      annotated/               <- images annotees auto
      eval_results.json        <- metriques IoU auto
      xplane_config.json       <- config rendu X-Plane (X-Plane only)
      fault_profile.json       <- profil fautes capteur (si actif)
      weather_profile.json     <- profil meteo X-Plane (si actif)
    pipeline_report.json    <- rapport agrege

Exemples :
  python run_pipeline.py generate -n 5
  python run_pipeline.py generate -n 100 --renderer xplane
  python run_pipeline.py evaluate LFPO_24
  python run_pipeline.py evaluate --all
  python run_pipeline.py full -n 3
  python run_pipeline.py full -n 100 --renderer xplane --xplane-dir "C:/X-Plane 12"
        """,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- Arguments communs renderer ---
    renderer_args = argparse.ArgumentParser(add_help=False)
    renderer_args.add_argument("--renderer", type=str, default="xplane",
                               choices=["ges", "xplane"],
                               help="Renderer d'images (defaut: xplane)")
    _default_xp = "C:/X-Plane 12" if os.name == "nt" else os.path.expanduser("~/X-Plane 12")
    renderer_args.add_argument("--xplane-dir", type=str, default=_default_xp,
                               help=f"Repertoire X-Plane 12 (defaut: {_default_xp})")

    # --- generate ---
    p_gen = sub.add_parser("generate", parents=[renderer_args],
                           help="Genere les scenarios TAF (.esp/.yaml) dans runs/")
    p_gen.add_argument("-n", "--nb-scenarios", type=int, default=None,
                       help="Nombre de scenarios (surcharge settings.xml)")
    p_gen.add_argument("-q", "--quiet", action="store_true")

    # --- evaluate ---
    p_eval = sub.add_parser("evaluate", parents=[renderer_args],
                            help="Dezip + GT + YOLO + IoU sur un ou tous les runs")
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
    p_full = sub.add_parser("full", parents=[renderer_args],
                            help="Pipeline complet (generate + rendu + evaluate)")
    p_full.add_argument("-n", "--nb-scenarios", type=int, default=None)
    p_full.add_argument("-q", "--quiet", action="store_true")
    p_full.add_argument("--conf", type=float, default=0.25)
    p_full.add_argument("--imgsz", type=int, default=512)
    p_full.add_argument("--iou-thresh", type=float, default=0.5)
    p_full.add_argument("--iou-method", type=str, default="CIOU",
                        choices=["IOU", "GIOU", "DIOU", "CIOU"])

    args = parser.parse_args()

    if args.mode == "generate":
        step_generate(nb_scenarios=args.nb_scenarios, quiet=args.quiet,
                      renderer=args.renderer)

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
            renderer=args.renderer,
            xplane_dir=args.xplane_dir,
        )

    elif args.mode == "full":
        run_full(
            nb_scenarios=args.nb_scenarios,
            quiet=args.quiet,
            conf=args.conf,
            imgsz=args.imgsz,
            iou_thresh=args.iou_thresh,
            iou_method=args.iou_method,
            renderer=args.renderer,
            xplane_dir=args.xplane_dir,
        )


if __name__ == "__main__":
    main()
