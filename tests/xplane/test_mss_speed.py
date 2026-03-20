"""
Test comparatif : screenshot X-Plane vs capture mss.
Lance avec X-Plane ouvert (vol KPDX, sim en pause pas grave).

Mesure le temps de 10 captures avec chaque methode.
"""
import struct
import socket
import time
import mss
import mss.tools
from pathlib import Path

HOST = "127.0.0.1"
PORT = 49000
XPLANE_DIR = Path("C:/X-Plane 12")
SCREENSHOT_DIR = XPLANE_DIR / "Output" / "screenshots"
OUT_DIR = Path("tests/xplane/speed_test")
OUT_DIR.mkdir(exist_ok=True)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(5.0)
addr = (HOST, PORT)

N = 10  # nombre de captures


def send_cmnd(command):
    msg = b"CMND\x00"
    msg += command.encode('utf-8') + b'\x00'
    sock.sendto(msg, addr)


# ===== Trouver la fenetre X-Plane =====
# mss capture par region d'ecran — on prend le moniteur principal
# (si X-Plane est en plein ecran c'est tout l'ecran)
sct = mss.mss()
# Moniteur 1 (principal)
monitor = sct.monitors[1]
print(f"Moniteur : {monitor['width']}x{monitor['height']}")
print(f"Captures : {N}\n")

# ===== Methode 1 : X-Plane screenshot command =====
print("--- Methode 1 : sim/operation/screenshot ---")
existing = set(SCREENSHOT_DIR.glob("*.png")) if SCREENSHOT_DIR.exists() else set()

t0 = time.perf_counter()
for i in range(N):
    send_cmnd("sim/operation/screenshot")
    time.sleep(0.3)  # attente ecriture disque
t1 = time.perf_counter()

dt_xplane = (t1 - t0) / N
print(f"  Temps moyen : {dt_xplane*1000:.0f} ms/frame")
print(f"  Total {N} frames : {t1-t0:.2f}s")

# ===== Methode 2 : mss screen capture =====
print("\n--- Methode 2 : mss screen capture ---")

t0 = time.perf_counter()
for i in range(N):
    img = sct.grab(monitor)
    # Sauver en PNG
    path = str(OUT_DIR / f"mss_{i:04d}.png")
    mss.tools.to_png(img.rgb, img.size, output=path)
t1 = time.perf_counter()

dt_mss_png = (t1 - t0) / N
print(f"  Temps moyen (avec save PNG) : {dt_mss_png*1000:.0f} ms/frame")
print(f"  Total {N} frames : {t1-t0:.2f}s")

# ===== Methode 3 : mss grab seulement (sans save) =====
print("\n--- Methode 3 : mss grab seul (sans ecriture disque) ---")

t0 = time.perf_counter()
frames = []
for i in range(N):
    img = sct.grab(monitor)
    frames.append(img)
t1 = time.perf_counter()

dt_mss_grab = (t1 - t0) / N
print(f"  Temps moyen (grab seul) : {dt_mss_grab*1000:.0f} ms/frame")
print(f"  Total {N} frames : {t1-t0:.2f}s")

# Sauver apres coup
print(f"\n  Sauvegarde des {N} frames...")
t0 = time.perf_counter()
for i, img in enumerate(frames):
    path = str(OUT_DIR / f"mss_batch_{i:04d}.png")
    mss.tools.to_png(img.rgb, img.size, output=path)
t1 = time.perf_counter()
print(f"  Sauvegarde : {(t1-t0)/N*1000:.0f} ms/frame")

# ===== Resume =====
print(f"\n=== Resume ===")
print(f"  X-Plane screenshot : {dt_xplane*1000:.0f} ms/frame")
print(f"  mss + save PNG     : {dt_mss_png*1000:.0f} ms/frame")
print(f"  mss grab seul      : {dt_mss_grab*1000:.0f} ms/frame")
speedup = dt_xplane / dt_mss_png if dt_mss_png > 0 else 0
print(f"  Speedup mss/xplane : {speedup:.1f}x")

sct.close()
sock.close()
