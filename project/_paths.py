"""
_paths.py — Bootstrap sys.path centralise pour LARD-LAAS-TAF
=============================================================
Import simple : `import _paths` ajoute tous les dossiers necessaires au
sys.path (idempotent).

Modules entry-points (run_pipeline, Generate, predict, evaluate, Export
copie par TAF) ont juste a inserer ce fichier en tete de leurs imports.
Les modules feuilles n'ont alors rien a faire eux-memes.
"""

import sys
from pathlib import Path

# project/_paths.py -> project/ -> racine
PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR.parent
EXPORT_DIR = PROJECT_DIR / "export"
YOLO_DIR = ROOT / "yolo"
LARD_ROOT = ROOT / "LARD"
TAF_SRC = ROOT / "taf" / "src"

_PATHS = (ROOT, PROJECT_DIR, EXPORT_DIR, YOLO_DIR, YOLO_DIR / "eval", LARD_ROOT)

for _p in _PATHS:
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)
