"""
test_rain_position.py — Chercher ou la pluie tombe autour de la piste
=====================================================================
Injecte une meteo pluvieuse puis teleporte l'avion en grille autour
de la piste pour trouver ou les particules de pluie apparaissent.

Prerequis :
  - X-Plane 12 lance, avion sur la piste
  - Plugin PI_weather.py actif

Usage :
  python tests/xplane/test_rain_position.py
  python tests/xplane/test_rain_position.py --precip 1.0 --step 500 --range 2000
"""

import sys
import os
import time
import math
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "project", "export"))

from xplane_weather import (
    WeatherConfig, set_exchange_dir, inject_weather, reset_weather, check_plugin
)
from xplane_bridge import XPlaneConnection, XPlaneConfig

KPDX_LAT = 45.5887
KPDX_LON = -122.6


def offset_latlon(lat, lon, north_m, east_m):
    """Decale lat/lon de north_m metres au nord et east_m metres a l'est."""
    d_lat = north_m / 111320.0
    d_lon = east_m / (111320.0 * math.cos(math.radians(lat)))
    return lat + d_lat, lon + d_lon


def main():
    parser = argparse.ArgumentParser(
        description="Chercher ou tombe la pluie autour de la piste")
    parser.add_argument("--xplane-dir", type=str,
                        default="C:/X-Plane 12" if os.name == "nt"
                        else os.path.expanduser("~/X-Plane 12"))
    parser.add_argument("--precip", type=float, default=1.0,
                        help="Precip rate (defaut: 1.0)")
    parser.add_argument("--step", type=int, default=500,
                        help="Pas de la grille en metres (defaut: 500)")
    parser.add_argument("--range", type=int, default=2000,
                        help="Distance max depuis la piste en metres (defaut: 2000)")
    parser.add_argument("--alt-agl", type=float, default=200.0,
                        help="Altitude au-dessus de la piste en metres (defaut: 200)")
    args = parser.parse_args()

    step = args.step
    rng = args.range

    # --- Connexions ---
    set_exchange_dir(args.xplane_dir)

    print("Verification plugin meteo...")
    if not check_plugin():
        print("ERREUR : plugin PI_weather.py ne repond pas.")
        sys.exit(1)
    print("Plugin OK")

    conn = XPlaneConnection(XPlaneConfig())
    print("Connexion UDP X-Plane...")
    if not conn.check_connection():
        print("ERREUR : X-Plane ne repond pas sur UDP.")
        sys.exit(1)
    print("UDP OK")

    # Position de base
    pose = conn.read_actual_pose()
    base_lat = pose["lat"]
    base_lon = pose["lon"]
    base_alt = pose["alt_m"]
    base_hdg = pose["heading"]
    fly_alt = base_alt + args.alt_agl

    print(f"\nPosition piste : lat={base_lat:.6f} lon={base_lon:.6f} "
          f"alt={base_alt:.1f}m")
    print(f"Altitude test : {fly_alt:.0f}m MSL ({args.alt_agl:.0f}m AGL)")

    # Override
    conn.send_dref("sim/operation/override/override_planepath[0]", 1.0)
    conn.send_dref("sim/operation/override/override_flight_control", 1.0)
    time.sleep(0.2)

    # Injection meteo
    print(f"\nReset meteo...")
    reset_weather()
    time.sleep(8)

    config = WeatherConfig(
        precip_rate=args.precip,
        cloud_type=3.0,
        cloud_coverage=1.0,
        cloud_margin_m=500.0,
        cloud_thickness_m=3000.0,
        visibility_m=50000.0,
        time_of_day_h=12.0,
    )
    inject_weather(config, aircraft_max_alt_m=fly_alt, latitude=KPDX_LAT, longitude=KPDX_LON)

    # Grille de positions
    offsets_1d = list(range(-rng, rng + 1, step))
    positions = []
    for north in offsets_1d:
        for east in offsets_1d:
            positions.append((north, east))

    total = len(positions)
    print(f"\n{'='*60}")
    print(f"  GRILLE {len(offsets_1d)}x{len(offsets_1d)} = {total} positions")
    print(f"  Pas: {step}m | Range: +/-{rng}m | precip={args.precip}")
    print(f"  (0,0) = position piste actuelle")
    print(f"{'='*60}")
    print(f"\n  Entree pour commencer...")
    input()

    try:
        for i, (north, east) in enumerate(positions):
            lat, lon = offset_latlon(base_lat, base_lon, north, east)

            conn.send_posi(lat, lon, fly_alt, base_hdg, 0.0, 0.0)
            conn.send_dref("sim/flightmodel/position/local_vx", 0.0)
            conn.send_dref("sim/flightmodel/position/local_vy", 0.0)
            conn.send_dref("sim/flightmodel/position/local_vz", 0.0)
            time.sleep(0.5)

            direction = ""
            if north > 0:
                direction += f"N{north}m"
            elif north < 0:
                direction += f"S{-north}m"
            if east > 0:
                direction += f" E{east}m"
            elif east < 0:
                direction += f" W{-east}m"
            if not direction:
                direction = "PISTE (centre)"

            print(f"  [{i+1}/{total}] {direction:20s} | "
                  f"Pluie ? Entree pour suivant...")
            input()

        # Retour
        print(f"\n  Retour a la piste...")
        conn.send_posi(base_lat, base_lon, base_alt, base_hdg, 0.0, 0.0)
        reset_weather()

    finally:
        conn.send_dref("sim/operation/override/override_planepath[0]", 0.0)
        conn.send_dref("sim/operation/override/override_flight_control", 0.0)

    print(f"\n{'='*60}")
    print(f"  TERMINE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
