"""
Generate.py — Lance TAF pour generer des scenarios d'approche
=============================================================
Utilisation :
    python project/Generate.py          (depuis la racine)
    python Generate.py                  (depuis project/)

Ce script :
    1. Se place dans project/ (CWD requis par TAF pour settings.xml)
    2. Insere project/export/ en tete de sys.path (notre Export.py)
    3. Insere taf/src/ dans sys.path (modules TAF)
    4. Lance TAF : parse template → z3 solve → export()
"""

import sys
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def _precreate_output_dirs(project_dir):
    """
    Pre-cree l'arborescence de sortie TAF.

    TAF utilise `mkdir -p` (Unix) qui n'existe pas sur Windows.
    Sur Linux, TAF gere tout seul, mais on pre-cree quand meme
    pour uniformiser le comportement cross-platform.
    """
    tree = ET.parse(project_dir / "settings.xml")
    params = {p.attrib["name"]: p.attrib["value"] for p in tree.getroot()}

    nb_cases = int(params.get("nb_test_cases", 3))
    nb_artifacts = int(params.get("nb_test_artifacts", 1))
    experiment = Path(params["experiment_path"]) / params["experiment_folder_name"]

    for i in range(nb_cases):
        for j in range(nb_artifacts):
            d = experiment / f"{params['test_case_folder_name']}_{i}" / f"{params['test_artifact_folder_name']}_{j}"
            d.mkdir(parents=True, exist_ok=True)


def run(nb_test_cases=None, verbose=True):
    """
    Point d'entree pour lancer la generation TAF.

    :param nb_test_cases: surcharge le nb de scenarios (sinon, utilise settings.xml)
    :param verbose: afficher les details TAF
    """
    project_dir = Path(__file__).resolve().parent
    root_dir = project_dir.parent

    # CWD = project/ (TAF lit ./settings.xml depuis le CWD)
    os.chdir(project_dir)

    # Copier notre Export.py dans taf/src/ (point d'extension prevu par TAF)
    src_export = project_dir / "export" / "Export.py"
    taf_export = root_dir / "taf" / "src" / "Export.py"
    shutil.copy2(src_export, taf_export)

    # sys.path : TAF + nos modules
    export_dir = str(project_dir / "export")
    taf_src = str(root_dir / "taf" / "src")
    if export_dir not in sys.path:
        sys.path.insert(0, export_dir)
    if taf_src not in sys.path:
        sys.path.insert(1, taf_src)

    # Surcharger nb_test_cases dans le XML avant import TAF (SETTINGS lit au module-level)
    if nb_test_cases is not None:
        settings_file = project_dir / "settings.xml"
        tree = ET.parse(settings_file)
        for p in tree.getroot():
            if p.attrib["name"] == "nb_test_cases":
                p.set("value", str(nb_test_cases))
        tree.write(settings_file)

    # Pre-creer les dossiers : mkdir -p n'existe pas sur Windows,
    # et sur Linux ca evite le prompt interactif de TAF.
    # Path.mkdir() est cross-platform, donc on appelle toujours.
    _precreate_output_dirs(project_dir)

    import Taf

    taf = Taf.CLI()
    taf.verbose = verbose
    taf.auto = True
    taf.do_overwrite(None)

    print(f"[Generate] Template : {Taf.SETTINGS.get('template_path')}"
          f"{Taf.SETTINGS.get('template_file_name')}")
    print(f"[Generate] Sortie   : {Taf.SETTINGS.get('experiment_path')}"
          f"{Taf.SETTINGS.get('experiment_folder_name')}/")
    print(f"[Generate] Scenarios: {Taf.SETTINGS.get('nb_test_cases')}")
    print()

    taf.do_parse_template()
    taf.do_generate()

    # Nettoyage : TAF cree parfois un stub Export.py dans le CWD
    stub = project_dir / "Export.py"
    if stub.exists() and stub.stat().st_size <= 50:
        stub.unlink()

    # Restaurer le placeholder dans taf/src/ (garder taf/ propre)
    taf_export.write_text("def export(root_node, path):\n\tpass\n")

    print("\n[Generate] Termine.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generation de scenarios TAF")
    parser.add_argument("-n", "--nb-scenarios", type=int, default=None,
                        help="Nombre de scenarios (surcharge settings.xml)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Mode silencieux")
    args = parser.parse_args()

    run(nb_test_cases=args.nb_scenarios, verbose=not args.quiet)
