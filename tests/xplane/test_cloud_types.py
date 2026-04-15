"""
test_cloud_types.py — Test visuel des 4 types de nuages + pluie dans X-Plane 12
=================================================================================
Pour chaque cloud_type (0-3), reset complet + inject avec precip_rate + cloud_coverage.
Reset entre chaque type pour isoler le comportement.

Prerequis :
  - X-Plane 12 lance, avion place sur une piste
  - Plugin PI_lard_weather.py actif (XPPython3)

Usage :
  python tests/xplane/test_cloud_types.py
  python tests/xplane/test_cloud_types.py --pause 12
  python tests/xplane/test_cloud_types.py --xplane-dir "D:/X-Plane 12"
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "project", "export"))

from xplane_weather import (
    WeatherConfig, set_exchange_dir, inject_weather, reset_weather, check_plugin
)

CLOUD_NAMES = {
    0: "Cirrus       (fins, haute altitude, transparents)",
    1: "Stratus      (couche uniforme, gris, bas)",
    2: "Cumulus      (moutonnes, beau temps)",
    3: "Cumulonimbus (epais, sombres, orage/pluie)",
}

RESET_DELAY_S = 6


def main():
    parser = argparse.ArgumentParser(description="Test cloud_type + pluie X-Plane 12")
    parser.add_argument("--xplane-dir", type=str,
                        default="C:/X-Plane 12" if os.name == "nt"
                        else os.path.expanduser("~/X-Plane 12"))
    parser.add_argument("--pause", type=int, default=10,
                        help="Pause en secondes entre chaque palier (defaut: 10)")
    parser.add_argument("--precip", type=float, default=0.7,
                        help="precip_rate pour chaque test (defaut: 0.7)")
    args = parser.parse_args()

    set_exchange_dir(args.xplane_dir)

    print("Verification plugin...")
    if not check_plugin():
        print("ERREUR : plugin PI_lard_weather.py ne repond pas.")
        print("Verifier que X-Plane est lance et le plugin actif.")
        sys.exit(1)
    print("Plugin OK\n")

    cloud_types = [0, 1, 2, 3]
    total = len(cloud_types) + 1  # +1 pour le bonus -1

    print(f"Test : {total} cloud_types avec precip_rate={args.precip}")
    print(f"Reset complet entre chaque type, pause {args.pause}s\n")

    for i, ct in enumerate(cloud_types):
        # Reset avant chaque injection
        print(f"  Reset meteo...")
        reset_weather()
        print(f"  Attente {RESET_DELAY_S}s apres reset...")
        time.sleep(RESET_DELAY_S)

        print(f"--- [{i+1}/{total}] cloud_type={ct} : {CLOUD_NAMES[ct]} ---")

        config = WeatherConfig(
            precip_rate=args.precip,
            cloud_type=ct,
            cloud_coverage=0.8,
            visibility_m=3000,
            time_of_day_h=14.0,
        )
        inject_weather(config)

        print(f"  Observe... Appuie Entree pour passer au suivant\n")
        input()

    # --- Test final : cloud_type=-1 (auto Cb) pour comparer ---
    print(f"  Reset meteo...")
    reset_weather()
    print(f"  Attente {RESET_DELAY_S}s apres reset...")
    time.sleep(RESET_DELAY_S)

    print(f"--- [{total}/{total}] cloud_type=-1 (auto, Cb forces par precip) ---")
    config = WeatherConfig(
        precip_rate=args.precip,
        cloud_type=-1,
        visibility_m=3000,
        time_of_day_h=14.0,
    )
    inject_weather(config)

    print(f"\n  Observe... Appuie Entree pour reset et quitter")
    input()

    print("Reset meteo...")
    reset_weather()
    print("Done.")


if __name__ == "__main__":
    main()
