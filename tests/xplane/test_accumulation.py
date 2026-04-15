"""
test_accumulation.py — Test accumulation effets sol (flaques, neige) via sim_speed
===================================================================================
Injecte de la pluie forte puis accelere la sim pour observer l'accumulation
des flaques sur la piste. Teste aussi la neige (temp < 0).

Placer l'avion SUR la piste avant de lancer.

Usage :
  python tests/xplane/test_accumulation.py
  python tests/xplane/test_accumulation.py --mode snow
  python tests/xplane/test_accumulation.py --boost 16 --duration 8
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "project", "export"))

from xplane_weather import (
    WeatherConfig, set_exchange_dir, inject_weather, reset_weather,
    check_plugin, set_sim_speed
)


def run_test(label, config, boost, duration, args):
    """Injecte meteo, accelere, puis laisse observer."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    print(f"\n  Reset meteo...")
    reset_weather()
    time.sleep(6)

    print(f"  Injection meteo (precip={config.precip_rate}, temp={config.temperature_c}C)...")
    inject_weather(config)

    # L'injection fait deja l'acceleration si precip > 0,
    # mais on ajoute des paliers manuels pour observer en direct

    print(f"\n  --- Etat AVANT acceleration ---")
    print(f"  Observe la piste. Appuie Entree pour accelerer a {boost}x...")
    input()

    print(f"  Acceleration {boost}x pendant {duration}s reel "
          f"({boost * duration}s simule)...")
    set_sim_speed(boost)
    time.sleep(duration)
    set_sim_speed(1)

    print(f"  Retour 1x.")
    print(f"  --- Etat APRES {boost * duration}s simulees ---")
    print(f"  Observe la piste. Appuie Entree pour continuer...")
    input()

    # Deuxieme round d'acceleration
    print(f"  Encore {boost}x pendant {duration}s ({boost * duration}s de plus)...")
    set_sim_speed(boost)
    time.sleep(duration)
    set_sim_speed(1)

    print(f"  --- Etat APRES {2 * boost * duration}s simulees au total ---")
    print(f"  Observe. Appuie Entree pour passer au test suivant...")
    input()


def main():
    parser = argparse.ArgumentParser(description="Test accumulation sol XP12")
    parser.add_argument("--xplane-dir", type=str,
                        default="C:/X-Plane 12" if os.name == "nt"
                        else os.path.expanduser("~/X-Plane 12"))
    parser.add_argument("--mode", type=str, default="all",
                        choices=["rain", "snow", "all"])
    parser.add_argument("--boost", type=int, default=8,
                        help="Multiplicateur sim_speed (defaut: 8)")
    parser.add_argument("--duration", type=int, default=5,
                        help="Duree reel par palier en secondes (defaut: 5)")
    args = parser.parse_args()

    set_exchange_dir(args.xplane_dir)

    print("Verification plugin...")
    if not check_plugin():
        print("ERREUR : plugin ne repond pas.")
        sys.exit(1)
    print("Plugin OK")

    if args.mode in ("rain", "all"):
        config = WeatherConfig(
            precip_rate=1.0,
            cloud_type=-1,
            visibility_m=3000,
            time_of_day_h=14.0,
        )
        run_test("PLUIE FORTE — flaques sur piste", config, args.boost, args.duration, args)

    if args.mode in ("snow", "all"):
        config = WeatherConfig(
            precip_rate=1.0,
            cloud_type=-1,
            visibility_m=3000,
            temperature_c=-10.0,
            time_of_day_h=14.0,
        )
        run_test("NEIGE — accumulation sur piste", config, args.boost, args.duration, args)

    print("\nReset meteo final...")
    reset_weather()
    print("Done.")


if __name__ == "__main__":
    main()
