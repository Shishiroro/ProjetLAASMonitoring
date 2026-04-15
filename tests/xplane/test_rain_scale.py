"""
test_rain_scale.py — Test visuel rain_scale dans X-Plane 12
============================================================
Injecte de la pluie (precip_rate=1.0) puis fait varier rain_scale
de 0.1 a 5.0 par paliers, avec une pause entre chaque pour observer.

Prerequis :
  - X-Plane 12 lance, avion place sur une piste
  - Plugin PI_lard_weather.py actif (XPPython3)

Usage :
  python tests/xplane/test_rain_scale.py
  python tests/xplane/test_rain_scale.py --xplane-dir "D:/X-Plane 12"
  python tests/xplane/test_rain_scale.py --pause 10
"""

import sys
import os
import time
import argparse

# Ajouter project/export/ au path pour importer xplane_weather
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "project", "export"))

from xplane_weather import (
    WeatherConfig, set_exchange_dir, inject_weather, reset_weather, check_plugin
)


def main():
    parser = argparse.ArgumentParser(description="Test rain_scale X-Plane 12")
    parser.add_argument("--xplane-dir", type=str,
                        default="C:/X-Plane 12" if os.name == "nt"
                        else os.path.expanduser("~/X-Plane 12"))
    parser.add_argument("--pause", type=int, default=8,
                        help="Pause en secondes entre chaque palier (defaut: 8)")
    args = parser.parse_args()

    set_exchange_dir(args.xplane_dir)

    # Verifier que le plugin repond
    print("Verification plugin...")
    if not check_plugin():
        print("ERREUR : plugin PI_lard_weather.py ne repond pas.")
        print("Verifier que X-Plane est lance et le plugin actif.")
        sys.exit(1)
    print("Plugin OK\n")

    # Paliers de rain_scale a tester
    scales = [ 5 ]

    print(f"Test rain_scale : {len(scales)} paliers, pause {args.pause}s entre chaque")
    print(f"precip_rate=1.0 (pluie max), cloud_type=-1 (Cb auto)\n")

    for i, scale in enumerate(scales):
        print(f"--- [{i+1}/{len(scales)}] rain_scale = {scale} ---")

        config = WeatherConfig(
            precip_rate=1.0,
            cloud_type=-1,      # Cb forces auto
            visibility_m=3000,
            rain_scale=scale,
          
            time_of_day_h=20.0,  # Plein jour pour mieux voir les gouttes   
        )
        inject_weather(config)

        if i < len(scales) - 1:
            print(f"  Observe pendant {args.pause}s...\n")
            time.sleep(args.pause)

    print(f"\n--- Dernier palier (rain_scale={scales[-1]}), appuie Entree pour clear ---")
    input()

    print("Reset meteo...")
    reset_weather()
    print("Termine.")


if __name__ == "__main__":
    main()
