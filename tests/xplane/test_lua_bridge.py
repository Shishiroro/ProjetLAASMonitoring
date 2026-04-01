"""
test_lua_bridge.py — Test rapide du bridge FlyWithLua
=====================================================
Verifie que lard_bridge.lua repond aux commandes JSON.

Usage : python tests/xplane/test_lua_bridge.py
"""

import sys
import json
import time
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "project" / "export"))

from lua_bridge import LuaBridgeConnection
from xplane_bridge import XPlaneConfig

# Trouver le dossier d'echange
XPLANE_DIR = Path("C:/X-Plane 12")
EXCHANGE_DIR = XPLANE_DIR / "Resources" / "plugins" / "FlyWithLua" / "Scripts" / "lard_exchange"

if not EXCHANGE_DIR.exists():
    print(f"ERREUR : dossier d'echange introuvable : {EXCHANGE_DIR}")
    print("Verifier que FlyWithLua est installe et lard_bridge.lua charge")
    sys.exit(1)

print(f"Exchange dir : {EXCHANGE_DIR}")
print()

config = XPlaneConfig(xplane_dir=str(XPLANE_DIR))
conn = LuaBridgeConnection(config, EXCHANGE_DIR)

# --- Test 1 : noop (heartbeat) ---
print("=== TEST 1 : noop ===")
try:
    ok = conn.check_connection()
    print(f"  Reponse : {'OK' if ok else 'TIMEOUT'}")
except Exception as e:
    print(f"  Erreur : {e}")
    sys.exit(1)

if not ok:
    print("\nFlyWithLua ne repond pas. Verifier :")
    print("  - X-Plane est lance")
    print("  - lard_bridge.lua est dans FlyWithLua/Scripts/ (pas en quarantaine)")
    print("  - Plugins > FlyWithLua > Reload all scripts")
    sys.exit(1)

# --- Test 2 : read_pose ---
print("\n=== TEST 2 : read_pose ===")
try:
    status = conn._send_and_wait("read_pose")
    pose = status.get("actual_pose", {})
    ref = status.get("ref_point", {})
    print(f"  Position : lat={pose.get('lat', '?')}, lon={pose.get('lon', '?')}")
    print(f"  Altitude : {pose.get('alt_m', '?')} m")
    print(f"  Heading  : {pose.get('heading', '?')}°")
    print(f"  Pitch    : {pose.get('pitch', '?')}°")
    print(f"  Roll     : {pose.get('roll', '?')}°")
    print(f"  Ref local: x={ref.get('local_x', '?')}, y={ref.get('local_y', '?')}, z={ref.get('local_z', '?')}")
except Exception as e:
    print(f"  Erreur : {e}")

# --- Test 3 : setup (pause + overrides) ---
print("\n=== TEST 3 : setup ===")
try:
    status = conn._send_and_wait("setup", timeout=10.0)
    pe = status.get("pilot_eye", {})
    fov = status.get("fov_deg", "?")
    print(f"  OK : {status.get('ok')}")
    print(f"  Pilot eye : x={pe.get('x', '?')}, y={pe.get('y', '?')}, z={pe.get('z', '?')}")
    print(f"  FOV : {fov}°")
    print(f"  (X-Plane devrait etre en pause maintenant)")
except Exception as e:
    print(f"  Erreur : {e}")

# --- Test 4 : release (unpause) ---
print("\n=== TEST 4 : release ===")
try:
    status = conn._send_and_wait("release")
    print(f"  OK : {status.get('ok')}")
    print(f"  (X-Plane devrait reprendre)")
except Exception as e:
    print(f"  Erreur : {e}")

# Nettoyage
conn.close()
print("\n=== TOUS LES TESTS OK ===")
