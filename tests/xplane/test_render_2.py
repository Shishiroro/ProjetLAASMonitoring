"""
Test rendu 2 scenarios KPDX avec screenshots.
Utilise la methode validee : pause + override + local_x/y/z.

Usage : python tests/xplane/test_render_2.py
  (X-Plane doit etre lance avec un vol sur KPDX)
"""
import struct
import socket
import time
import json
import math
from pathlib import Path
import shutil

HOST = "127.0.0.1"
PORT = 49000

SCENARIOS = [
    "runs/KPDX_28R_005",
    "runs/KPDX_10R_003",
]

# Config X-Plane — MODIFIE CE CHEMIN si besoin
XPLANE_DIR = Path("C:/X-Plane 12")
SCREENSHOT_DIR = XPLANE_DIR / "Output" / "screenshots"

SETTLE_TIME = 0.1       # attente apres pose (sec)
SCREENSHOT_DELAY = 0.3  # attente apres screenshot (sec)

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


def latlon_to_local(lat, lon, alt_m, ref_lat, ref_lon, ref_elev, ref_lx, ref_ly, ref_lz):
    """Convertit lat/lon/alt en local_x/y/z X-Plane (OpenGL).
    local_x = est, local_y = up, local_z = sud (negatif = nord).
    """
    dlat = lat - ref_lat
    dlon = lon - ref_lon
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(ref_lat))
    dn = dlat * m_per_deg_lat
    de = dlon * m_per_deg_lon
    lx = ref_lx + de
    lz = ref_lz - dn
    ly = ref_ly + (alt_m - ref_elev)
    return lx, ly, lz


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


# ===== Connexion =====
print("=== Connexion X-Plane ===")
msg = b"RREF\x00"
msg += struct.pack('<ii', 1, 0)
msg += "sim/time/total_running_time_sec".encode('utf-8').ljust(400, b'\x00')
sock.sendto(msg, addr)
try:
    sock.recvfrom(1024)
    msg2 = b"RREF\x00"
    msg2 += struct.pack('<ii', 0, 0)
    msg2 += "sim/time/total_running_time_sec".encode('utf-8').ljust(400, b'\x00')
    sock.sendto(msg2, addr)
    print("  OK\n")
except socket.timeout:
    print("  ECHEC — X-Plane n'est pas lance ?")
    exit(1)

# ===== Setup =====
send_dref("sim/time/paused", 1.0)
time.sleep(0.3)
send_dref("sim/operation/override/override_planepath[0]", 1.0)
send_dref("sim/operation/override/override_flight_control", 1.0)
send_dref("sim/operation/override/override_throttles", 1.0)
time.sleep(0.2)
send_cmnd("sim/view/forward_with_nothing")
time.sleep(0.5)

# ===== Reference (position actuelle dans X-Plane) =====
ref_lat = read_dref("sim/flightmodel/position/latitude", 1)
ref_lon = read_dref("sim/flightmodel/position/longitude", 2)
ref_elev = read_dref("sim/flightmodel/position/elevation", 3)
ref_lx = read_dref("sim/flightmodel/position/local_x", 4)
ref_ly = read_dref("sim/flightmodel/position/local_y", 5)
ref_lz = read_dref("sim/flightmodel/position/local_z", 6)
print(f"Reference : lat={ref_lat:.4f}, lon={ref_lon:.4f}, elev={ref_elev:.1f}m")
print(f"  local: x={ref_lx:.0f}, y={ref_ly:.0f}, z={ref_lz:.0f}\n")

# ===== Snapshot screenshots existants =====
existing_screenshots = set()
if SCREENSHOT_DIR.exists():
    existing_screenshots = set(SCREENSHOT_DIR.glob("*.png"))

# ===== Boucle sur les scenarios =====
for scenario_path in SCENARIOS:
    run_dir = Path(scenario_path)
    poses_file = run_dir / "poses.json"

    if not poses_file.exists():
        print(f"SKIP {scenario_path} — pas de poses.json")
        continue

    with open(poses_file) as f:
        data = json.load(f)

    name = data["scenario_name"]
    poses = data["poses"]
    n = len(poses)

    footage_dir = run_dir / "footage"
    footage_dir.mkdir(exist_ok=True)

    print(f"=== {name} — {n} frames ===")

    for i, pose in enumerate(poses):
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
        time.sleep(SETTLE_TIME)

        # Screenshot
        send_cmnd("sim/operation/screenshot")
        time.sleep(SCREENSHOT_DELAY)

        # Recuperer le screenshot
        dst = footage_dir / f"{name}_{i + 1:04d}.png"
        if SCREENSHOT_DIR.exists():
            current = set(SCREENSHOT_DIR.glob("*.png"))
            new_files = current - existing_screenshots
            if new_files:
                src = max(new_files, key=lambda f: f.stat().st_mtime)
                for attempt in range(5):
                    try:
                        shutil.copy2(str(src), str(dst))
                        break
                    except PermissionError:
                        time.sleep(0.3)
                existing_screenshots = current

        if (i + 1) % 50 == 0 or i == 0 or (i + 1) == n:
            print(f"  Frame {i+1}/{n} — alt={alt_m:.0f}m hdg={heading:.0f}°")

    n_rendered = len(list(footage_dir.glob("*.png")))
    print(f"  -> {n_rendered}/{n} images dans {footage_dir}\n")

# ===== Cleanup =====
send_dref("sim/time/paused", 0.0)
send_dref("sim/operation/override/override_planepath[0]", 0.0)
send_dref("sim/operation/override/override_flight_control", 0.0)
send_dref("sim/operation/override/override_throttles", 0.0)
print("Termine !")
sock.close()
