"""
main.py — Point d'entree CLI du projet LARD-LAAS-TAF
=====================================================
Usage :
    python main.py                    # genere 3 scenarios (defaut settings.xml)
    python main.py -n 5 -q            # 5 scenarios, mode silencieux
"""

import sys
from pathlib import Path

# Ajouter project/ au sys.path
project_dir = Path(__file__).resolve().parent / "project"
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

from Generate import run

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LARD-LAAS-TAF : Generation de trajectoires d'approche realistes"
    )
    parser.add_argument("-n", "--nb-scenarios", type=int, default=None,
                        help="Nombre de scenarios a generer (surcharge settings.xml)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Mode silencieux")
    args = parser.parse_args()

    run(nb_test_cases=args.nb_scenarios, verbose=not args.quiet)
