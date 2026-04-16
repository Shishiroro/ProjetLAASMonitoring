"""
test_precip_altitude.py — Test apparition de la pluie en fonction de l'altitude
================================================================================
Pour chaque precip_rate, monte l'avion par paliers depuis la piste pour
observer a quelle altitude la pluie devient visible.

Workflow par precip_rate :
  1. Injection meteo au sol
  2. Monte par paliers (+50m, +100m, +200m, +500m, +1000m)
  3. Entree pour passer au palier suivant
  4. Reset et precip_rate suivant

Prerequis :
  - X-Plane 12 lance, avion place sur une piste
  - Plugin PI_lard_weather.py actif (XPPython3)
  - Sim en pause ou override planepath (le script gere)

Usage :
  python tests/xplane/test_precip_altitude.py
  python tests/xplane/test_precip_altitude.py --altitudes 0,20,50,100,200,500
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "project", "export"))

from xplane_weather import (
    WeatherConfig, set_exchange_dir, inject_weather, reset_weather, check_plugin
)
from xplane_bridge import XPlaneConnection, XPlaneConfig

KPDX_LON = -122.6
RESET_DELAY_S = 6

PRECIP_RATES = [ 0.3, 1.0]
DEFAULT_ALTITUDES = [0, 50, 100, 200, 500]


def main():
    parser = argparse.ArgumentParser(
        description="Test apparition pluie vs altitude XP12")
    parser.add_argument("--xplane-dir", type=str,
                        default="C:/X-Plane 12" if os.name == "nt"
                        else os.path.expanduser("~/X-Plane 12"))
    parser.add_argument("--altitudes", type=str, default=None,
                        help="Altitudes au-dessus de la piste (virgules). "
                             "Defaut: 0,50,100,200,500,1000")
    parser.add_argument("--precip", type=str, default=None,
                        help="Precip rates a tester (virgules). "
                             "Defaut: 0.1,0.3,0.5,0.7,1.0")
    args = parser.parse_args()

    altitudes = DEFAULT_ALTITUDES
    if args.altitudes:
        altitudes = [int(a) for a in args.altitudes.split(",")]

    precip_rates = PRECIP_RATES
    if args.precip:
        precip_rates = [float(p) for p in args.precip.split(",")]

    # --- Connexions ---
    set_exchange_dir(args.xplane_dir)

    print("Verification plugin meteo...")
    if not check_plugin():
        print("ERREUR : plugin PI_lard_weather.py ne repond pas.")
        sys.exit(1)
    print("Plugin OK")

    conn = XPlaneConnection(XPlaneConfig())
    print("Connexion UDP X-Plane...")
    if not conn.check_connection():
        print("ERREUR : X-Plane ne repond pas sur UDP.")
        sys.exit(1)
    print("UDP OK")

    # Lire la position actuelle (on suppose avion sur la piste)
    pose = conn.read_actual_pose()
    base_lat = pose["lat"]
    base_lon = pose["lon"]
    base_alt = pose["alt_m"]
    base_hdg = pose["heading"]
    print(f"\nPosition de base : lat={base_lat:.6f} lon={base_lon:.6f} "
          f"alt={base_alt:.1f}m hdg={base_hdg:.0f}")

    # Override planepath pour pouvoir repositionner
    conn.send_dref("sim/operation/override/override_planepath[0]", 1.0)
    conn.send_dref("sim/operation/override/override_flight_control", 1.0)
    time.sleep(0.2)

    total_combos = len(precip_rates) * len(altitudes)
    combo_i = 0

    print(f"\n{'='*60}")
    print(f"  TEST PRECIP x ALTITUDE — {total_combos} combinaisons")
    print(f"  Precip rates : {precip_rates}")
    print(f"  Altitudes AGL : {altitudes}m")
    print(f"{'='*60}")

    try:
        for precip in precip_rates:
            print(f"\n{'='*60}")
            print(f"  PRECIP_RATE = {precip}")
            print(f"{'='*60}")

            # Reset + injection meteo
            print(f"  Reset meteo...")
            reset_weather()
            time.sleep(RESET_DELAY_S)

            # Cloud base tres haute pour ne pas etre dans les nuages
            config = WeatherConfig(
                precip_rate=precip,
                cloud_type=3.0,           # Cb pour la pluie
                cloud_coverage=1.0,
                cloud_margin_m=2000.0,    # nuages loin au-dessus
                cloud_thickness_m=3000.0,
                visibility_m=50000.0,     # pas de brouillard (isoler l'effet pluie)
                time_of_day_h=12.0,
            )

            max_test_alt = base_alt + max(altitudes)
            inject_weather(config, aircraft_max_alt_m=max_test_alt, longitude=KPDX_LON)

            for alt_agl in altitudes:
                combo_i += 1
                target_alt = base_alt + alt_agl

                # Positionner l'avion
                conn.send_posi(base_lat, base_lon, target_alt,
                               base_hdg, 0.0, 0.0)
                # Zero velocites
                conn.send_dref("sim/flightmodel/position/local_vx", 0.0)
                conn.send_dref("sim/flightmodel/position/local_vy", 0.0)
                conn.send_dref("sim/flightmodel/position/local_vz", 0.0)
                time.sleep(0.5)

                print(f"\n  [{combo_i}/{total_combos}] precip={precip} | "
                      f"alt={alt_agl}m AGL ({target_alt:.0f}m MSL)")
                print(f"    Pluie visible ? Gouttes ? Effets sol ?")
                print(f"    Entree pour monter...")
                input()

        # Remettre au sol
        print(f"\n  Retour au sol...")
        conn.send_posi(base_lat, base_lon, base_alt,
                       base_hdg, 0.0, 0.0)
        reset_weather()

    finally:
        # Relacher les overrides
        conn.send_dref("sim/operation/override/override_planepath[0]", 0.0)
        conn.send_dref("sim/operation/override/override_flight_control", 0.0)

    print(f"\n{'='*60}")
    print(f"  TERMINE — {total_combos} combinaisons testees")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
