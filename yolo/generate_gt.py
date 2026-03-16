"""
Genere le .csv ground truth LARD apres recuperation des images Google Earth Studio.
A lancer une fois les images placees dans le dossier footage/ du dataset.
"""

import sys
from pathlib import Path

# Ajouter project/ au path pour importer lard_bridge
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "project" / "export"))
sys.path.insert(0, str(PROJECT_ROOT / "LARD"))

# Fix convention yaw : LARD retourne un azimut camera (LTP→FPAP) qui, apres % 360,
# donne le cap de vol (~oppose au heading piste). runway_is_facing_us attend le sens
# de regard. Le +180 corrige l'inversion (fix envisage par LARD, ligne 132 commentee).
import src.labeling.label_export as _le
_original_facing = _le.runway_is_facing_us
def _fixed_facing(heading, runway):
    return _original_facing((heading + 180) % 360, runway)
_le.runway_is_facing_us = _fixed_facing

from lard_bridge import generate_labels_csv


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generer le CSV ground truth LARD (post-GES)")
    parser.add_argument("--yaml", type=str, required=True, help="Chemin vers le .yaml du scenario")
    parser.add_argument("--dataset", type=str, required=True, help="Dossier du dataset (contient footage/)")
    parser.add_argument("--csv-name", type=str, default=None, help="Nom du CSV de sortie (defaut: <yaml_stem>_labels.csv)")
    args = parser.parse_args()

    generate_labels_csv(
        yaml_path=args.yaml,
        dataset_dir=args.dataset,
        csv_name=args.csv_name,
    )
