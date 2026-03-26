"""
test_weather_live.py — Test meteo en temps reel dans X-Plane 12
================================================================
Met la pluie a fond et attend pour voir si ca prend visuellement.
Teste aussi differentes methodes pour forcer la meteo manuelle.

Usage : python tests/xplane/test_weather_live.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "project" / "export"))

from xplane_bridge import XPlaneConnection, XPlaneConfig


def main():
    conn = XPlaneConnection(XPlaneConfig())

    if not conn.check_connection():
        print("ERREUR : X-Plane non joignable")
        return

    print("Connexion OK\n")

    # ---- Etape 1 : Lire l'etat actuel ----
    print("=== ETAT ACTUEL ===")
    for dref, label in [
        ("sim/weather/use_real_weather_bool", "real weather"),
        ("sim/weather/region/change_mode", "change_mode"),
        ("sim/weather/rain_percent", "rain%"),
        ("sim/weather/cloud_base_msl_m[0]", "cloud base"),
        ("sim/weather/cloud_tops_msl_m[0]", "cloud tops"),
        ("sim/weather/temperature_sealevel_c", "temp °C"),
        ("sim/weather/visibility_reported_m", "visibility"),
        ("sim/weather/weather_preset", "preset"),
    ]:
        val = conn.read_dref(dref, hash(dref) % 100 + 50)
        print(f"  {label:20s} = {val}")
        time.sleep(0.05)

    # ---- Etape 2 : Forcer meteo manuelle ----
    print("\n=== FORCER METEO MANUELLE ===")

    # Methode 1 : use_real_weather_bool = 0
    conn.send_dref("sim/weather/use_real_weather_bool", 0.0)
    print("  use_real_weather_bool = 0")
    time.sleep(0.3)

    # Methode 2 : region/change_mode = 0 (manual)
    conn.send_dref("sim/weather/region/change_mode", 0.0)
    print("  region/change_mode = 0")
    time.sleep(0.3)

    # Methode 3 : commande menu "clear weather"
    # puis on remet nos valeurs
    conn.send_command("sim/operation/clear_all_weather")
    print("  commande clear_all_weather")
    time.sleep(1.0)

    # ---- Etape 3 : Mettre pluie a fond ----
    print("\n=== SET PLUIE 100% ===")
    conn.send_dref("sim/weather/rain_percent", 100.0)
    time.sleep(0.2)

    # Relire pour verifier
    val = conn.read_dref("sim/weather/rain_percent", 80)
    print(f"  rain_percent relu = {val}")

    # ---- Etape 4 : Nuages tres bas ----
    print("\n=== SET NUAGES BAS (100m base, 600m top) ===")
    conn.send_dref("sim/weather/cloud_base_msl_m[0]", 100.0)
    conn.send_dref("sim/weather/cloud_tops_msl_m[0]", 600.0)
    time.sleep(0.2)

    val_base = conn.read_dref("sim/weather/cloud_base_msl_m[0]", 81)
    val_top = conn.read_dref("sim/weather/cloud_tops_msl_m[0]", 82)
    print(f"  cloud base relu = {val_base}")
    print(f"  cloud tops relu = {val_top}")

    # ---- Etape 5 : Temperature basse ----
    print("\n=== SET TEMPERATURE -10°C ===")
    conn.send_dref("sim/weather/temperature_sealevel_c", -10.0)
    time.sleep(0.2)

    val = conn.read_dref("sim/weather/temperature_sealevel_c", 83)
    print(f"  temperature relu = {val}")

    # ---- Etape 6 : Attendre et relire ----
    print("\n=== ATTENTE 5s — REGARDE X-PLANE ===")
    print("  (la pluie, les nuages bas et le froid devraient etre visibles)")
    time.sleep(5.0)

    # Relire apres 5s pour voir si XP12 a ecrase
    print("\n=== RELECTURE APRES 5s ===")
    for dref, label in [
        ("sim/weather/rain_percent", "rain%"),
        ("sim/weather/cloud_base_msl_m[0]", "cloud base"),
        ("sim/weather/cloud_tops_msl_m[0]", "cloud tops"),
        ("sim/weather/temperature_sealevel_c", "temp °C"),
        ("sim/weather/region/change_mode", "change_mode"),
    ]:
        val = conn.read_dref(dref, hash(dref) % 100 + 90)
        print(f"  {label:20s} = {val}")
        time.sleep(0.05)

    # ---- Etape 7 : Spam continu (forcer) ----
    print("\n=== SPAM PLUIE 100% pendant 10s (1 envoi/100ms) ===")
    print("  Si la pluie apparait maintenant mais pas avant,")
    print("  c'est que XP12 ecrase la valeur et il faut spammer.")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 10.0:
        conn.send_dref("sim/weather/rain_percent", 100.0)
        conn.send_dref("sim/weather/cloud_base_msl_m[0]", 100.0)
        conn.send_dref("sim/weather/cloud_tops_msl_m[0]", 600.0)
        conn.send_dref("sim/weather/temperature_sealevel_c", -10.0)
        time.sleep(0.1)

    print("\n=== FIN — reset meteo ===")
    conn.send_dref("sim/weather/rain_percent", 0.0)
    conn.send_dref("sim/weather/cloud_base_msl_m[0]", 3000.0)
    conn.send_dref("sim/weather/cloud_tops_msl_m[0]", 3500.0)
    conn.send_dref("sim/weather/temperature_sealevel_c", 15.0)

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
