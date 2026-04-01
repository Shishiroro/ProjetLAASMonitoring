"""
Test meteo viable : regen (nuages instantanes) + region/rain (pluie instantanee).
"""
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "project" / "export"))

from lua_bridge import LuaBridgeConnection
from xplane_bridge import XPlaneConfig

XPLANE_DIR = Path("C:/X-Plane 12")
EXCHANGE_DIR = XPLANE_DIR / "Resources" / "plugins" / "FlyWithLua" / "Scripts" / "lard_exchange"

conn = LuaBridgeConnection(XPlaneConfig(xplane_dir=str(XPLANE_DIR)), EXCHANGE_DIR)
if not conn.check_connection():
    print("FlyWithLua ne repond pas"); sys.exit(1)

# Ajouter regen_weather au Lua
# On l'envoie via set_pose avec un dref special que le Lua interprete
# ... en fait on n'a plus l'action regen_weather dans le Lua propre.
# On va le refaire via une commande dediee.

# Etape 1 : Ecrire les cloud datarefs classiques (pour preparer les valeurs)
print("Etape 1 : cloud datarefs classiques...")
conn._send_and_wait("set_pose", weather={
    "sim/weather/cloud_base_msl_m[0]": 100.0,
    "sim/weather/cloud_tops_msl_m[0]": 600.0,
    "sim/weather/cloud_type[0]": 4.0,
})
time.sleep(0.5)

# Etape 2 : Pluie via region/ (instantane)
print("Etape 2 : pluie region/ + update_immediately...")
conn._send_and_wait("set_pose", weather={
    "sim/weather/region/rain_percent": 80.0,
})
time.sleep(0.2)
conn._send_and_wait("set_pose", weather={
    "sim/weather/region/update_immediately": 1,
})

print("Attente 30s — tu devrais voir la pluie rapidement.")
print("Les nuages sont probablement hauts (pas grave pour le test).")
for i in range(30):
    time.sleep(1)
    if i % 5 == 0:
        print(f"  {i}s...")

print("Fin.")
conn.close()
