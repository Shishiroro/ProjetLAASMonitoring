"""
test_weather_lua.py — Test injection meteo via FlyWithLua
=========================================================
Injecte de la pluie a 100% pendant 20 secondes.
Verifie visuellement dans X-Plane que la pluie est visible.

Prerequis : X-Plane lance, lard_bridge.lua charge, graphiques >= Medium.

Usage : python tests/xplane/test_weather_lua.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "project" / "export"))

from lua_bridge import LuaBridgeConnection
from xplane_bridge import XPlaneConfig

XPLANE_DIR = Path("C:/X-Plane 12")
EXCHANGE_DIR = XPLANE_DIR / "Resources" / "plugins" / "FlyWithLua" / "Scripts" / "lard_exchange"

config = XPlaneConfig(xplane_dir=str(XPLANE_DIR))
conn = LuaBridgeConnection(config, EXCHANGE_DIR)

# Verifier connection
print("Connexion FlyWithLua...", end=" ")
if not conn.check_connection():
    print("TIMEOUT - FlyWithLua ne repond pas")
    sys.exit(1)
print("OK")

# Lire position actuelle pour verifier
status = conn._send_and_wait("read_pose")
pose = status.get("actual_pose", {})
print(f"Position : lat={pose.get('lat', '?'):.4f}, lon={pose.get('lon', '?'):.4f}, alt={pose.get('alt_m', '?'):.0f}m")

# Altitude terrain approximative (ajuster selon l'aeroport)
terrain_elev = pose.get("alt_m", 0.0)
print(f"Altitude terrain estimee : {terrain_elev:.0f}m MSL")

# Liste de datarefs meteo a tester
WEATHER_TESTS = [
    {
        "name": "PLUIE 100% + NUAGES BAS (200m AGL)",
        "drefs": {
            "sim/weather/use_real_weather_bool": 0,
            "sim/weather/rain_percent": 100.0,
            "sim/weather/cloud_base_msl_m[0]": terrain_elev + 200.0,
            "sim/weather/cloud_tops_msl_m[0]": terrain_elev + 700.0,
        },
    },
    {
        "name": "NUAGES SEULS (100m AGL, sans pluie)",
        "drefs": {
            "sim/weather/use_real_weather_bool": 0,
            "sim/weather/rain_percent": 0.0,
            "sim/weather/cloud_base_msl_m[0]": terrain_elev + 100.0,
            "sim/weather/cloud_tops_msl_m[0]": terrain_elev + 600.0,
        },
    },
]

for test in WEATHER_TESTS:
    print(f"\n=== {test['name']} ===")
    print(f"  Datarefs : {test['drefs']}")
    print(f"  Injection pendant 20 secondes...")

    t0 = time.time()
    while time.time() - t0 < 20:
        # Envoyer weather drefs via set_pose (sans deplacer l'avion)
        conn._send_and_wait("set_pose", weather=test["drefs"])
        # Re-lire pour verifier que les datarefs ont ete ecrits
        time.sleep(1.0)

    # Verifier la valeur lue
    rain = conn._send_and_wait("read_pose")
    print(f"  Termine. Vois-tu la pluie dans X-Plane ?")

    # Pause pour observer
    input("  Appuie sur Entree pour passer au test suivant...")

# Reset
print("\nReset meteo...")
conn._send_and_wait("set_pose", weather={
    "sim/weather/rain_percent": 0.0,
    "sim/weather/use_real_weather_bool": 1,
})
print("Meteo remise a zero.")

conn.close()
print("\n=== FIN DES TESTS ===")
