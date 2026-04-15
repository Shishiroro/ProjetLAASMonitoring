"""
test_precip_thresholds.py — Trouver les seuils visuels de pluie + effets piste XP12
=====================================================================================
Deux series de tests :
  1) precip_rate de 0.1 a 1.0 par pas de 0.1 (cloud_coverage fixe a 1.0)
  2) cloud_coverage de 0.1 a 1.0 par pas de 0.1 (precip_rate fixe a 1.0)

Observer pour chaque palier :
  - Particules de pluie visibles ? (gouttes dans l'air)
  - Sol mouille / flaques sur la piste ?
  - Neige au sol ? (si temperature_c < 0)
  - Eclairage / reflets changes ?

Prerequis :
  - X-Plane 12 lance, avion place SUR une piste (pour voir les effets sol)
  - Plugin PI_lard_weather.py actif (XPPython3)

Usage :
  python tests/xplane/test_precip_thresholds.py
  python tests/xplane/test_precip_thresholds.py --mode coverage
  python tests/xplane/test_precip_thresholds.py --mode snow
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "project", "export"))

from xplane_weather import (
    WeatherConfig, set_exchange_dir, inject_weather, reset_weather, check_plugin
)

RESET_DELAY_S = 6


def run_series(name, configs):
    """Execute une serie de tests avec reset + Entree entre chaque."""
    total = len(configs)
    print(f"\n{'='*60}")
    print(f"  SERIE : {name} ({total} paliers)")
    print(f"{'='*60}\n")

    for i, (label, config) in enumerate(configs):
        print(f"  Reset meteo...")
        reset_weather()
        time.sleep(RESET_DELAY_S)

        print(f"--- [{i+1}/{total}] {label} ---")
        print(f"    precip_rate={config.precip_rate:.1f}  "
              f"cloud_coverage={config.cloud_coverage:.1f}  "
              f"temp={config.temperature_c:.0f}C  "
              f"vis={config.visibility_m:.0f}m")

        inject_weather(config)

        print(f"    Observer : gouttes? sol mouille? flaques? reflets?")
        print(f"    Appuie Entree pour passer au suivant\n")
        input()


def main():
    parser = argparse.ArgumentParser(description="Test seuils precip + effets piste XP12")
    parser.add_argument("--xplane-dir", type=str,
                        default="C:/X-Plane 12" if os.name == "nt"
                        else os.path.expanduser("~/X-Plane 12"))
    parser.add_argument("--mode", type=str, default="all",
                        choices=["precip", "coverage", "snow", "all"],
                        help="Serie a executer (defaut: all)")
    args = parser.parse_args()

    set_exchange_dir(args.xplane_dir)

    print("Verification plugin...")
    if not check_plugin():
        print("ERREUR : plugin PI_lard_weather.py ne repond pas.")
        sys.exit(1)
    print("Plugin OK")

    # --- Serie 1 : precip_rate 0.1 → 1.0, cloud_coverage=1.0 fixe ---
    if args.mode in ("precip", "all"):
        configs = []
        for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            config = WeatherConfig(
                precip_rate=p,
                cloud_type=-1,        # Cb auto
                cloud_coverage=1.0,   # fixe
                visibility_m=3000,
                time_of_day_h=14.0,
            )
            configs.append((f"precip_rate={p:.1f} (coverage=1.0)", config))
        run_series("PRECIP_RATE 0.1→1.0 (coverage fixe 1.0)", configs)

    # --- Serie 2 : cloud_coverage 0.1 → 1.0, precip_rate=1.0 fixe ---
    if args.mode in ("coverage", "all"):
        configs = []
        for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            config = WeatherConfig(
                precip_rate=1.0,      # fixe
                cloud_type=1,         # Stratus (confirme fonctionnel)
                cloud_coverage=c,
                visibility_m=3000,
                time_of_day_h=14.0,
            )
            configs.append((f"cloud_coverage={c:.1f} (precip=1.0, Stratus)", config))
        run_series("CLOUD_COVERAGE 0.1→1.0 (precip fixe 1.0, Stratus)", configs)

    # --- Serie 3 : neige — precip_rate 0.3→1.0, temperature_c=-10 ---
    if args.mode in ("snow", "all"):
        configs = []
        for p in [0.3, 0.5, 0.7, 1.0]:
            config = WeatherConfig(
                precip_rate=p,
                cloud_type=-1,        # Cb auto
                visibility_m=3000,
                temperature_c=-10.0,  # neige
                time_of_day_h=14.0,
            )
            configs.append((f"NEIGE precip={p:.1f} temp=-10C", config))
        run_series("NEIGE (temp=-10C, precip 0.3→1.0)", configs)

    # Reset final
    print("\nReset meteo final...")
    reset_weather()
    print("Done.")


if __name__ == "__main__":
    main()
