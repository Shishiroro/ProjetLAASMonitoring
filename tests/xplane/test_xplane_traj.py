"""
Test visuel — rejoue une vraie trajectoire dans X-Plane
========================================================
Lit poses.json, convertit lat/lon/alt en coordonnees locales OpenGL,
et deplace la camera frame par frame (sans screenshot).
"""
import struct
import socket
import time
import json
import math

HOST = "127.0.0.1"
PORT = 49000
POSES_PATH = "runs/KPDX_28R_005/poses.json"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(5.0)
addr = (HOST, PORT)


def send_dref(dref, value):
    msg = b"DREF\x00"
    msg += struct.pack('<f', value)
    msg += dref.encode('utf-8').ljust(500, b'\x00')
    sock.sendto(msg, addr)


def send_cmnd(command):
    msg = b"CMND\x00"
    msg += command.encode('utf-8') + b'\x00'
    sock.sendto(msg, addr)


def read_dref(dref, idx=0):
    msg = b"RREF\x00"
    msg += struct.pack('<ii', 1, idx)
    msg += dref.encode('utf-8').ljust(400, b'\x00')
    sock.sendto(msg, addr)
    try:
        data, _ = sock.recvfrom(1024)
        val = struct.unpack('<f', data[9:13])[0] if len(data) >= 13 else None
    except socket.timeout:
        val = None
    msg2 = b"RREF\x00"
    msg2 += struct.pack('<ii', 0, idx)
    msg2 += dref.encode('utf-8').ljust(400, b'\x00')
    sock.sendto(msg2, addr)
    return val


def set_pose_local(lx, ly, lz, heading, pitch, roll):
    send_dref("sim/flightmodel/position/local_x", lx)
    send_dref("sim/flightmodel/position/local_y", ly)
    send_dref("sim/flightmodel/position/local_z", lz)
    send_dref("sim/flightmodel/position/theta", pitch)
    send_dref("sim/flightmodel/position/phi", roll)
    send_dref("sim/flightmodel/position/psi", heading)
    send_dref("sim/flightmodel/position/local_vx", 0.0)
    send_dref("sim/flightmodel/position/local_vy", 0.0)
    send_dref("sim/flightmodel/position/local_vz", 0.0)


def latlon_to_local(lat, lon, alt_m, ref_lat, ref_lon, ref_elev, ref_lx, ref_ly, ref_lz):
    """Convertit lat/lon/alt en local_x/y/z par rapport a un point de reference.

    Coordonnees OpenGL X-Plane :
      local_x = est (metres)
      local_y = up (altitude)
      local_z = sud (metres) — nord = negatif
    """
    dlat = lat - ref_lat
    dlon = lon - ref_lon

    # Metres par degre a cette latitude
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(ref_lat))

    # Offset en metres : nord = -z, est = +x
    dn = dlat * m_per_deg_lat   # metres vers le nord
    de = dlon * m_per_deg_lon   # metres vers l'est

    lx = ref_lx + de            # est = +x
    lz = ref_lz - dn            # nord = -z, donc +nord = -z
    ly = ref_ly + (alt_m - ref_elev)

    return lx, ly, lz


# ===== Connexion =====
print("=== Connexion X-Plane ===")
try:
    msg = b"RREF\x00"
    msg += struct.pack('<ii', 1, 0)
    msg += "sim/time/total_running_time_sec".encode('utf-8').ljust(400, b'\x00')
    sock.sendto(msg, addr)
    sock.recvfrom(1024)
    msg2 = b"RREF\x00"
    msg2 += struct.pack('<ii', 0, 0)
    msg2 += "sim/time/total_running_time_sec".encode('utf-8').ljust(400, b'\x00')
    sock.sendto(msg2, addr)
    print("  OK")
except socket.timeout:
    print("  ECHEC")
    exit(1)

# ===== Setup =====
send_dref("sim/time/paused", 1.0)
time.sleep(0.3)
send_dref("sim/operation/override/override_planepath[0]", 1.0)
send_dref("sim/operation/override/override_flight_control", 1.0)
time.sleep(0.2)
send_cmnd("sim/view/forward_with_nothing")
time.sleep(0.5)

# ===== Point de reference : position actuelle =====
ref_lat = read_dref("sim/flightmodel/position/latitude", 1)
ref_lon = read_dref("sim/flightmodel/position/longitude", 2)
ref_elev = read_dref("sim/flightmodel/position/elevation", 3)
ref_lx = read_dref("sim/flightmodel/position/local_x", 4)
ref_ly = read_dref("sim/flightmodel/position/local_y", 5)
ref_lz = read_dref("sim/flightmodel/position/local_z", 6)
print(f"\nReference : lat={ref_lat}, lon={ref_lon}, elev={ref_elev}m")
print(f"  local: x={ref_lx:.0f}, y={ref_ly:.0f}, z={ref_lz:.0f}")

# ===== Charger poses =====
with open(POSES_PATH) as f:
    data = json.load(f)

poses = data["poses"]
n = len(poses)
print(f"\n=== Trajectoire {data['scenario_name']} — {n} frames ===")

# On affiche toutes les 10 frames pour aller plus vite (pas 314 x 3sec)
# 0.3s par frame pour le test visuel
STEP = 1       # toutes les frames
DELAY = 0.15   # secondes entre frames (rapide mais visible)

print(f"  Replay : {n} frames, {DELAY}s entre chaque\n")

input("Appuie Entree pour lancer le replay...")

for i in range(0, n, STEP):
    pose = poses[i]
    lat = pose["lat"]
    lon = pose["lon"]
    alt_m = pose["alt_m"]
    heading = pose["heading"]
    pitch = pose["pitch_ges"] - 90.0  # GES → X-Plane
    roll = pose["roll"]

    lx, ly, lz = latlon_to_local(
        lat, lon, alt_m,
        ref_lat, ref_lon, ref_elev,
        ref_lx, ref_ly, ref_lz,
    )

    set_pose_local(lx, ly, lz, heading, pitch, roll)
    time.sleep(DELAY)

    if (i + 1) % 50 == 0 or i == 0 or i == n - 1:
        print(f"  Frame {i+1}/{n} : alt={alt_m:.0f}m, hdg={heading:.0f}°")

print(f"\n  Replay termine !")

# ===== Cleanup =====
input("\nAppuie Entree pour nettoyer...")
send_dref("sim/time/paused", 0.0)
send_dref("sim/operation/override/override_planepath[0]", 0.0)
send_dref("sim/operation/override/override_flight_control", 0.0)
print("Termine.")
sock.close()
