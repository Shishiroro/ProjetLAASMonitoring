"""
Test capture X-Plane : resize fenetre 1024x1024 puis capture zone client.
Gere le DPI scaling Windows.
"""
import time
import ctypes
import mss
from pathlib import Path
from PIL import Image

try:
    import win32gui
except ImportError:
    print("pip install pywin32")
    exit(1)

# FIX DPI scaling — DOIT etre appele avant toute operation fenetre
ctypes.windll.user32.SetProcessDPIAware()

OUT_DIR = Path("tests/xplane/speed_test")
OUT_DIR.mkdir(exist_ok=True)

N = 5
TARGET = 1024


def find_xplane_window():
    result = {}
    def callback(hwnd, _):
        title = win32gui.GetWindowText(hwnd)
        if "X-Plane" in title and win32gui.IsWindowVisible(hwnd):
            result['hwnd'] = hwnd
            result['title'] = title
        return True
    win32gui.EnumWindows(callback, None)
    return result if result else None


def get_client_rect(hwnd):
    """Zone client en coordonnees ecran reelles (DPI-aware)."""
    client = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (client[0], client[1]))
    right, bottom = win32gui.ClientToScreen(hwnd, (client[2], client[3]))
    return left, top, right, bottom


# ===== Trouver la fenetre =====
info = find_xplane_window()
if not info:
    print("Fenetre X-Plane introuvable.")
    exit(1)

hwnd = info['hwnd']
print(f"Fenetre : {info['title']}")

# ===== Mesurer bordures =====
full = win32gui.GetWindowRect(hwnd)
client = get_client_rect(hwnd)
border_w = (full[2] - full[0]) - (client[2] - client[0])
border_h = (full[3] - full[1]) - (client[3] - client[1])

print(f"  Fenetre totale : {full[2]-full[0]}x{full[3]-full[1]}")
print(f"  Zone client    : {client[2]-client[0]}x{client[3]-client[1]}")
print(f"  Bordures       : {border_w}x{border_h}px")

# ===== Resize pour client = 1024x1024 =====
print(f"\n  Resize fenetre pour client {TARGET}x{TARGET}...")
left, top = full[0], full[1]
win32gui.MoveWindow(hwnd, left, top, TARGET + border_w, TARGET + border_h, True)
time.sleep(0.5)

# Verifier apres resize
full = win32gui.GetWindowRect(hwnd)
client = get_client_rect(hwnd)
cw = client[2] - client[0]
ch = client[3] - client[1]
print(f"  Zone client apres resize : {cw}x{ch}")

# ===== Capture =====
region = {
    "left": client[0],
    "top": client[1],
    "width": cw,
    "height": ch,
}

sct = mss.mss()

print(f"\n--- Capture {N} frames ({cw}x{ch}) ---")
t0 = time.perf_counter()
for i in range(N):
    img = sct.grab(region)
    pil_img = Image.frombytes("RGB", img.size, bytes(img.rgb))
    pil_img.save(str(OUT_DIR / f"capture_{i:04d}.jpg"), quality=90)
t1 = time.perf_counter()

print(f"  Taille image : {img.size[0]}x{img.size[1]}")
print(f"  {(t1-t0)/N*1000:.0f} ms/frame")
print(f"  Images dans {OUT_DIR}/capture_*.jpg")
print(f"\nVerifie que c'est propre (1024x1024, pas de barre, pas de zoom).")

sct.close()
