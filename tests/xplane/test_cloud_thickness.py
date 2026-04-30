"""
test_cloud_thickness.py — Test visuel epaisseur nuages par type
===============================================================
Injecte toutes les combinaisons cloud_type x thickness pour observer
l'effet visuel dans X-Plane 12.

Workflow :
  1. Injection nuage → regarder
  2. Entree → reset meteo
  3. Entree → injection suivante

Prerequis :
  - X-Plane 12 lance, avion place quelque part
  - Plugin PI_weather.py actif (XPPython3)

Usage :
  python tests/xplane/test_cloud_thickness.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "project", "export"))

from xplane_weather import (
    WeatherConfig, set_exchange_dir, inject_weather, reset_weather, check_plugin
)

RESET_DELAY_S = 6

CLOUD_TYPES = {
    0: "Cirrus",
    1: "Stratus",
    2: "Cumulus",
    3: "Cumulonimbus",
}

THICKNESSES = [0, 200, 1000, 3000, 8000]

# KPDX (Portland) — coordonnees pour conversion heure locale → UTC
KPDX_LAT = 45.5887
KPDX_LON = -122.6


def main():
    xplane_dir = "C:/X-Plane 12" if os.name == "nt" else os.path.expanduser("~/X-Plane 12")
    if len(sys.argv) > 1:
        xplane_dir = sys.argv[1]

    set_exchange_dir(xplane_dir)

    print("Verification plugin...")
    if not check_plugin():
        print("ERREUR : plugin PI_weather.py ne repond pas.")
        sys.exit(1)
    print("Plugin OK\n")

    combos = []
    for thickness in THICKNESSES:
        for ct, ct_name in CLOUD_TYPES.items():
            combos.append((ct, ct_name, thickness))

    total = len(combos)
    print(f"{'='*60}")
    print(f"  TEST CLOUD THICKNESS — {total} combinaisons")
    print(f"  Types : {list(CLOUD_TYPES.values())}")
    print(f"  Epaisseurs : {THICKNESSES}m")
    print(f"{'='*60}\n")

    for i, (ct, ct_name, thickness) in enumerate(combos):
        config = WeatherConfig(
            cloud_type=float(ct),
            cloud_coverage=1.0,
            cloud_margin_m=200.0,
            cloud_thickness_m=float(thickness),
            time_of_day_h=12.0,
        )

        print(f"--- [{i+1}/{total}] {ct_name} (type={ct}) — thickness={thickness}m ---")
        print(f"    cloud_base = alt_avion + 200m")
        print(f"    cloud_top  = cloud_base + {thickness}m")
        print(f"    Appuie Entree pour injecter...")
        input()

        inject_weather(config, latitude=KPDX_LAT, longitude=KPDX_LON)

        print(f"    -> Injecte. Regarde le ciel.")
        print(f"    Appuie Entree pour reset + passer au suivant\n")
        input()

        print(f"  Reset meteo...")
        reset_weather()
        time.sleep(RESET_DELAY_S)

    print(f"\n{'='*60}")
    print(f"  TERMINE — {total} combinaisons testees")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
