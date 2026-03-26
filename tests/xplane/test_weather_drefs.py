"""
test_weather_drefs.py — Test quels datarefs meteo X-Plane 12 sont ecrivables
===============================================================================
Lance ce script avec X-Plane 12 ouvert. Il va :
  1. Lire la valeur actuelle de chaque dataref
  2. Ecrire une valeur test
  3. Relire pour voir si ca a change
  4. Restaurer la valeur d'origine

Usage : python tests/xplane/test_weather_drefs.py
"""

import sys
import time
from pathlib import Path

# Ajouter project/export au path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "project" / "export"))

from xplane_bridge import XPlaneConnection, XPlaneConfig

# Datarefs meteo a tester (nom, valeur test, description)
DREFS_TO_TEST = [
    # --- Precipitation ---
    ("sim/weather/rain_percent", 80.0, "pluie (ancien)"),
    ("sim/weather/snow_percent", 80.0, "neige (ancien)"),
    ("sim/weather/precipitation_on_aircraft_ratio", 1.0, "precipitation sur avion"),

    # --- Visibilite ---
    ("sim/weather/visibility_reported_m", 2000.0, "visibilite reported"),
    ("sim/weather/visibility_effective_m", 2000.0, "visibilite effective"),

    # --- Nuages ---
    ("sim/weather/cloud_type[0]", 2.0, "type nuage layer 0"),
    ("sim/weather/cloud_base_msl_m[0]", 100.0, "base nuage layer 0"),
    ("sim/weather/cloud_tops_msl_m[0]", 2000.0, "sommet nuage layer 0"),
    ("sim/weather/cloud_coverage[0]", 6.0, "couverture nuage layer 0"),

    # --- Piste ---
    ("sim/weather/runway_wetness[0]", 15.0, "piste mouillee [0]"),
    ("sim/weather/runway_friction[0]", 0.3, "friction piste [0]"),

    # --- Vent / turbulence ---
    ("sim/weather/wind_speed_kt[0]", 30.0, "vent vitesse"),
    ("sim/weather/turbulence[0]", 0.8, "turbulence"),
    ("sim/weather/thunderstorm_percent", 80.0, "orage"),

    # --- Mode meteo ---
    ("sim/weather/use_real_weather_bool", 0.0, "desactiver meteo reelle"),
    ("sim/weather/weather_preset", 5.0, "preset meteo (0-7?)"),

    # --- XP12 nouveaux datarefs possibles ---
    ("sim/weather/aircraft/rain_ratio", 1.0, "rain ratio sur avion"),
    ("sim/weather/region/rain_ratio", 1.0, "rain ratio region"),
    ("sim/weather/region/visibility_reported_m", 2000.0, "visibilite region"),
    ("sim/weather/region/change_mode", 1.0, "mode changement region"),

    # --- Temperature ---
    ("sim/weather/temperature_sealevel_c", -5.0, "temp niveau mer"),

    # --- Commandes alternatives ---
    ("sim/private/controls/rain/set_rain", 0.8, "set_rain (private)"),
]


def main():
    conn = XPlaneConnection(XPlaneConfig())

    if not conn.check_connection():
        print("ERREUR : X-Plane non joignable sur localhost:49000")
        return

    print("Connexion X-Plane OK\n")
    print(f"{'Dataref':<55} {'Avant':>8} {'Ecrit':>8} {'Apres':>8} {'OK?':>5}")
    print("-" * 90)

    # D'abord, desactiver la meteo reelle
    conn.send_dref("sim/weather/use_real_weather_bool", 0.0)
    time.sleep(0.5)

    idx = 30  # index RREF de depart

    results_ok = []
    results_fail = []

    for dref, test_val, desc in DREFS_TO_TEST:
        idx += 1

        # Lire avant
        before = conn.read_dref(dref, idx)
        time.sleep(0.05)

        # Ecrire la valeur test
        conn.send_dref(dref, test_val)
        time.sleep(0.3)

        # Relire
        idx += 1
        after = conn.read_dref(dref, idx)
        time.sleep(0.05)

        # Verifier si ca a change
        before_s = f"{before:.2f}" if before is not None else "N/A"
        after_s = f"{after:.2f}" if after is not None else "N/A"

        changed = (after is not None and before is not None
                   and abs(after - before) > 0.01)
        wrote_val = (after is not None and abs(after - test_val) < 1.0)

        ok = changed or wrote_val
        marker = "  OK" if ok else "FAIL"

        print(f"{dref:<55} {before_s:>8} {test_val:>8.1f} {after_s:>8} {marker:>5}  ({desc})")

        if ok:
            results_ok.append((dref, desc))
        else:
            results_fail.append((dref, desc))

        # Restaurer (best effort)
        if before is not None:
            conn.send_dref(dref, before)
            time.sleep(0.1)

    print(f"\n{'=' * 60}")
    print(f"ECRIVABLES ({len(results_ok)}) :")
    for dref, desc in results_ok:
        print(f"  + {dref} ({desc})")

    print(f"\nNON-ECRIVABLES / TIMEOUT ({len(results_fail)}) :")
    for dref, desc in results_fail:
        print(f"  - {dref} ({desc})")

    print(f"\n{'=' * 60}")
    print("TIP : regarde aussi dans X-Plane si visuellement quelque chose a change")
    print("      pendant le test (pluie, brouillard, nuages, etc.)")

    conn.close()


if __name__ == "__main__":
    main()
