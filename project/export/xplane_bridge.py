"""
xplane_bridge.py — Interface avec X-Plane 12 pour le rendu automatise
=====================================================================
Communication UDP avec X-Plane pour injecter des poses camera
frame par frame et capturer des screenshots automatiquement.

Protocole UDP X-Plane :
  - DREF : set un dataref (float)
  - CMND : execute une commande X-Plane
  - RREF : request dataref (lecture)

Positionnement valide :
  - Pause sim + override planepath
  - Coordonnees locales OpenGL (local_x/y/z) pour la precision
  - Point de reference lu au setup pour conversion lat/lon → local

Capture :
  - mss (screen grab) cible sur la fenetre X-Plane
  - JPEG pour la vitesse (~25ms/frame vs 300ms avec screenshot X-Plane)

Conventions de poses :
  - Notre flight_data (GES) : (lon, lat, alt, yaw, pitch_ges, roll)
    pitch_ges : 90 = level (convention Google Earth Studio)
  - X-Plane : pitch 0 = level, negatif = nez en bas
  - Conversion : xplane_pitch = pitch_ges - 90

Usage :
    from xplane_bridge import render_scenario, XPlaneConfig
    config = XPlaneConfig(xplane_dir="C:/X-Plane 12")
    render_scenario("runs/LFPO_24/poses.json", "runs/LFPO_24/footage", config)
"""

import struct
import socket
import time
import json
import math
import ctypes
from pathlib import Path
from dataclasses import dataclass

import mss
import mss.tools
from PIL import Image

try:
    import win32gui
    HAS_WIN32 = True
    # Fix DPI scaling — coordonnees reelles au lieu de coordonnees virtuelles
    ctypes.windll.user32.SetProcessDPIAware()
except ImportError:
    HAS_WIN32 = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class XPlaneConfig:
    """Configuration de la connexion X-Plane."""
    host: str = "127.0.0.1"
    port: int = 49000               # Port UDP X-Plane (reception)
    xplane_dir: str = ""            # Repertoire d'installation X-Plane
    settle_time: float = 0.1        # Attente apres changement de pose (sec)
    window_width: int = 1280        # Largeur zone client X-Plane (min X-Plane)
    window_height: int = 1024       # Hauteur zone client X-Plane


# ---------------------------------------------------------------------------
# Protocole UDP X-Plane
# ---------------------------------------------------------------------------

def _pack_dref(dref: str, value: float) -> bytes:
    """Empaquete un message DREF (set dataref)."""
    msg = b"DREF\x00"
    msg += struct.pack('<f', value)
    msg += dref.encode('utf-8').ljust(500, b'\x00')
    return msg


def _pack_cmnd(command: str) -> bytes:
    """Empaquete un message CMND (execute commande)."""
    msg = b"CMND\x00"
    msg += command.encode('utf-8') + b'\x00'
    return msg


# ---------------------------------------------------------------------------
# Fenetre X-Plane (Windows)
# ---------------------------------------------------------------------------

def find_xplane_window():
    """Trouve la fenetre X-Plane et retourne son HWND + rect.

    :return: dict {hwnd, rect, title} ou None si introuvable
    """
    if not HAS_WIN32:
        return None
    result = {}

    def callback(hwnd, _):
        title = win32gui.GetWindowText(hwnd)
        if "X-Plane" in title and win32gui.IsWindowVisible(hwnd):
            result['hwnd'] = hwnd
            result['rect'] = win32gui.GetWindowRect(hwnd)
            result['title'] = title
        return True

    win32gui.EnumWindows(callback, None)
    return result if result else None


def _get_client_rect(hwnd):
    """Zone client (rendu pur, sans bordures ni barre titre) en coords ecran."""
    client = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (client[0], client[1]))
    right, bottom = win32gui.ClientToScreen(hwnd, (client[2], client[3]))
    return left, top, right, bottom


def resize_xplane_window(width=1280, height=1024):
    """Redimensionne la fenetre X-Plane pour que la zone client fasse width x height.

    Calcule les bordures dynamiquement (DPI-aware).
    """
    info = find_xplane_window()
    if not info:
        print("  [XPLANE] Fenetre introuvable, resize impossible")
        return None
    hwnd = info['hwnd']
    full = win32gui.GetWindowRect(hwnd)
    client = _get_client_rect(hwnd)
    border_w = (full[2] - full[0]) - (client[2] - client[0])
    border_h = (full[3] - full[1]) - (client[3] - client[1])
    win32gui.MoveWindow(
        hwnd, full[0], full[1],
        width + border_w, height + border_h, True,
    )
    time.sleep(0.5)
    info['rect'] = win32gui.GetWindowRect(hwnd)
    return info


def get_xplane_capture_region():
    """Retourne la region mss pour capturer la zone client X-Plane.

    Utilise GetClientRect (DPI-aware) pour exclure bordures et barre titre.
    """
    info = find_xplane_window()
    if not info:
        return None
    client = _get_client_rect(info['hwnd'])
    return {
        "left": client[0],
        "top": client[1],
        "width": client[2] - client[0],
        "height": client[3] - client[1],
    }


# ---------------------------------------------------------------------------
# Connexion X-Plane
# ---------------------------------------------------------------------------

class XPlaneConnection:
    """Connexion UDP vers X-Plane 12."""

    def __init__(self, config: XPlaneConfig = None):
        self.config = config or XPlaneConfig()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(5.0)
        self.addr = (self.config.host, self.config.port)
        # Point de reference pour conversion lat/lon → local
        self.ref_lat = None
        self.ref_lon = None
        self.ref_elev = None
        self.ref_lx = None
        self.ref_ly = None
        self.ref_lz = None
        # Capture ecran
        self.sct = None
        self.capture_region = None

    def send_dref(self, dref: str, value: float):
        """Set un dataref X-Plane."""
        self.sock.sendto(_pack_dref(dref, value), self.addr)

    def send_command(self, command: str):
        """Execute une commande X-Plane."""
        self.sock.sendto(_pack_cmnd(command), self.addr)

    def read_dref(self, dref: str, idx: int = 0):
        """Lit un dataref X-Plane via RREF."""
        msg = b"RREF\x00"
        msg += struct.pack('<ii', 1, idx)
        msg += dref.encode('utf-8').ljust(400, b'\x00')
        self.sock.sendto(msg, self.addr)
        try:
            data, _ = self.sock.recvfrom(1024)
            val = struct.unpack('<f', data[9:13])[0] if len(data) >= 13 else None
        except socket.timeout:
            val = None
        # Stop stream
        msg2 = b"RREF\x00"
        msg2 += struct.pack('<ii', 0, idx)
        msg2 += dref.encode('utf-8').ljust(400, b'\x00')
        self.sock.sendto(msg2, self.addr)
        return val

    def check_connection(self) -> bool:
        """Verifie que X-Plane est joignable via RREF."""
        freq, idx = 1, 0
        dref = "sim/time/total_running_time_sec"
        msg = b"RREF\x00"
        msg += struct.pack('<ii', freq, idx)
        msg += dref.encode('utf-8').ljust(400, b'\x00')
        try:
            self.sock.sendto(msg, self.addr)
            self.sock.recvfrom(1024)
            msg_stop = b"RREF\x00"
            msg_stop += struct.pack('<ii', 0, idx)
            msg_stop += dref.encode('utf-8').ljust(400, b'\x00')
            self.sock.sendto(msg_stop, self.addr)
            return True
        except socket.timeout:
            return False

    def _read_reference_point(self):
        """Lit la position actuelle comme point de reference.

        Necessaire pour convertir lat/lon/alt en coordonnees locales OpenGL
        avec une precision metrique (les DREF lat/lon sont float32, insuffisant).
        """
        self.ref_lat = self.read_dref("sim/flightmodel/position/latitude", 1)
        self.ref_lon = self.read_dref("sim/flightmodel/position/longitude", 2)
        self.ref_elev = self.read_dref("sim/flightmodel/position/elevation", 3)
        self.ref_lx = self.read_dref("sim/flightmodel/position/local_x", 4)
        self.ref_ly = self.read_dref("sim/flightmodel/position/local_y", 5)
        self.ref_lz = self.read_dref("sim/flightmodel/position/local_z", 6)
        print(f"  [XPLANE] Reference : lat={self.ref_lat:.4f}, lon={self.ref_lon:.4f}, "
              f"elev={self.ref_elev:.1f}m")
        print(f"  [XPLANE]   local: x={self.ref_lx:.0f}, y={self.ref_ly:.0f}, "
              f"z={self.ref_lz:.0f}")

    def _latlon_to_local(self, lat, lon, alt_m):
        """Convertit lat/lon/alt en local_x/y/z OpenGL X-Plane.

        Coordonnees OpenGL : x=est, y=up, z=sud (nord=negatif).
        Utilise le point de reference lu au setup.
        """
        dlat = lat - self.ref_lat
        dlon = lon - self.ref_lon
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(math.radians(self.ref_lat))
        dn = dlat * m_per_deg_lat   # metres vers le nord
        de = dlon * m_per_deg_lon   # metres vers l'est
        lx = self.ref_lx + de       # est = +x
        lz = self.ref_lz - dn       # nord = -z
        ly = self.ref_ly + (alt_m - self.ref_elev)
        return lx, ly, lz

    def setup_view(self):
        """Configure X-Plane pour le rendu automatise.

        - Pause le simulateur
        - Override du chemin de vol
        - Zero velocites et accelerations
        - Vue forward with nothing (pas de cockpit)
        - Lit le point de reference pour la conversion locale
        - Resize et cible la fenetre X-Plane pour capture mss
        """
        # Pause le sim
        self.send_dref("sim/time/paused", 1.0)
        time.sleep(0.3)

        # Override position avion
        self.send_dref("sim/operation/override/override_planepath[0]", 1.0)
        time.sleep(0.1)
        self.send_dref("sim/operation/override/override_flight_control", 1.0)
        self.send_dref("sim/operation/override/override_throttles", 1.0)
        time.sleep(0.1)

        # Zero velocites + accelerations + rotation rates
        for dref in [
            "sim/flightmodel/position/local_vx",
            "sim/flightmodel/position/local_vy",
            "sim/flightmodel/position/local_vz",
            "sim/flightmodel/position/local_ax",
            "sim/flightmodel/position/local_ay",
            "sim/flightmodel/position/local_az",
            "sim/flightmodel/position/P",
            "sim/flightmodel/position/Q",
            "sim/flightmodel/position/R",
        ]:
            self.send_dref(dref, 0.0)
        time.sleep(0.1)

        # Vue clean sans cockpit
        self.send_command("sim/view/forward_with_nothing")
        time.sleep(0.3)

        # Desactiver le mouse yoke (reticule au centre de l'ecran)
        self.send_dref("sim/operation/override/override_joystick", 1.0)
        self.send_dref("sim/joystick/mouse_is_yoke", 0.0)
        time.sleep(0.1)

        # Deplacer la souris hors de la fenetre X-Plane
        if HAS_WIN32:
            ctypes.windll.user32.SetCursorPos(0, 0)
            time.sleep(0.1)

        # Point de reference
        self._read_reference_point()

        # Resize + capture region
        if HAS_WIN32:
            resize_xplane_window(self.config.window_width, self.config.window_height)
            self.capture_region = get_xplane_capture_region()
            if self.capture_region:
                print(f"  [XPLANE] Capture : {self.capture_region['width']}x"
                      f"{self.capture_region['height']}")
            self.sct = mss.mss()

    def set_camera_pose(self, lat, lon, alt_m, heading, pitch, roll):
        """Positionne la camera via coordonnees locales OpenGL.

        Convertit lat/lon/alt en local_x/y/z pour une precision metrique.
        """
        lx, ly, lz = self._latlon_to_local(lat, lon, alt_m)
        self.send_dref("sim/flightmodel/position/local_x", lx)
        self.send_dref("sim/flightmodel/position/local_y", ly)
        self.send_dref("sim/flightmodel/position/local_z", lz)
        self.send_dref("sim/flightmodel/position/theta", pitch)
        self.send_dref("sim/flightmodel/position/phi", roll)
        self.send_dref("sim/flightmodel/position/psi", heading)
        self.send_dref("sim/flightmodel/position/local_vx", 0.0)
        self.send_dref("sim/flightmodel/position/local_vy", 0.0)
        self.send_dref("sim/flightmodel/position/local_vz", 0.0)
        time.sleep(self.config.settle_time)

    def capture_frame(self, output_path):
        """Capture la fenetre X-Plane et sauve en JPEG.

        :param output_path: chemin du fichier de sortie (.jpg)
        :return: True si capture reussie
        """
        if not self.sct or not self.capture_region:
            return False
        img = self.sct.grab(self.capture_region)
        pil_img = Image.frombytes("RGB", img.size, bytes(img.rgb))
        pil_img.save(str(output_path), quality=90)
        return True

    def release(self):
        """Rend le controle de l'avion a X-Plane."""
        self.send_dref("sim/time/paused", 0.0)
        self.send_dref("sim/operation/override/override_planepath[0]", 0.0)
        self.send_dref("sim/operation/override/override_flight_control", 0.0)
        self.send_dref("sim/operation/override/override_throttles", 0.0)

    def close(self):
        """Libere les ressources."""
        try:
            self.release()
        except Exception:
            pass
        if self.sct:
            self.sct.close()
        self.sock.close()


# ---------------------------------------------------------------------------
# Conversion de poses GES → X-Plane
# ---------------------------------------------------------------------------

def convert_pose_ges_to_xplane(lon, lat, alt, yaw, pitch_ges, roll):
    """Convertit une pose du format GES vers le format X-Plane.

    GES : pitch 90 = regard horizontal (level)
    X-Plane : pitch 0 = level, negatif = nez en bas

    :return: dict {lat, lon, alt_m, heading, pitch, roll}
    """
    return {
        "lat": float(lat),
        "lon": float(lon),
        "alt_m": float(alt),
        "heading": float(yaw),
        "pitch": float(pitch_ges) - 90.0,
        "roll": float(roll),
    }


# ---------------------------------------------------------------------------
# Fichier poses.json (format universel, independant du renderer)
# ---------------------------------------------------------------------------

def save_poses_json(flight_data, fps, scenario_name, output_path):
    """Sauvegarde les poses dans un fichier JSON universel.

    :param flight_data: list de tuples (lon, lat, alt, yaw, pitch_ges, roll)
    :param fps: frames par seconde
    :param scenario_name: nom du scenario (ex: LFPO_24)
    :param output_path: chemin du fichier JSON de sortie
    :return: chemin du fichier cree
    """
    output_path = Path(output_path)

    poses = []
    for lon, lat, alt, yaw, pitch, roll in flight_data:
        poses.append({
            "lon": float(lon),
            "lat": float(lat),
            "alt_m": float(alt),
            "heading": float(yaw),
            "pitch_ges": float(pitch),
            "roll": float(roll),
        })

    data = {
        "scenario_name": scenario_name,
        "n_frames": len(poses),
        "fps": fps,
        "poses": poses,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  .json poses -> {output_path}")
    return str(output_path)


def load_poses_json(path):
    """Charge un fichier poses.json.

    :return: dict {scenario_name, n_frames, fps, poses: [{lon,lat,alt_m,heading,pitch_ges,roll}]}
    """
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Rendu d'un scenario complet
# ---------------------------------------------------------------------------

def render_scenario(poses_path, output_dir, config=None):
    """Rend un scenario complet via X-Plane.

    Lit le fichier poses.json, injecte chaque pose dans X-Plane,
    capture la fenetre via mss, et sauve les images en JPEG.

    :param poses_path: chemin vers le fichier poses.json
    :param output_dir: dossier de sortie pour les images (footage/)
    :param config: XPlaneConfig (optionnel)
    :return: Path du dossier de sortie
    """
    config = config or XPlaneConfig()
    data = load_poses_json(poses_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_name = data["scenario_name"]
    n_frames = data["n_frames"]

    print(f"  [XPLANE] Rendu de {scenario_name} ({n_frames} frames)...")

    conn = XPlaneConnection(config)

    try:
        if not conn.check_connection():
            raise ConnectionError(
                "Impossible de joindre X-Plane. "
                "Verifier que X-Plane 12 est lance et ecoute sur "
                f"{config.host}:{config.port}"
            )

        conn.setup_view()

        t_start = time.perf_counter()

        # Padding identique a LARD : img_digits = len(str(n_frames - 1))
        img_digits = len(str(n_frames - 1))

        for i, pose in enumerate(data["poses"]):
            xp = convert_pose_ges_to_xplane(
                pose["lon"], pose["lat"], pose["alt_m"],
                pose["heading"], pose["pitch_ges"], pose["roll"],
            )

            conn.set_camera_pose(
                xp["lat"], xp["lon"], xp["alt_m"],
                xp["heading"], xp["pitch"], xp["roll"],
            )

            # Nommage 0-based, padding LARD : {name}_{i:0Nd}.jpg
            dst = output_dir / f"{scenario_name}_{str(i).zfill(img_digits)}.jpg"
            conn.capture_frame(dst)

            if (i + 1) % 50 == 0 or (i + 1) == n_frames:
                elapsed = time.perf_counter() - t_start
                fps = (i + 1) / elapsed
                print(f"  [XPLANE] {i + 1}/{n_frames} frames "
                      f"({fps:.1f} fps, {elapsed:.0f}s)")

        n_rendered = len(list(output_dir.glob("*.jpg")))
        total = time.perf_counter() - t_start
        print(f"  [XPLANE] {n_rendered} images dans {output_dir} ({total:.0f}s)")

        if n_rendered < n_frames:
            print(f"  [XPLANE] ATTENTION : {n_frames - n_rendered} frames manquantes")

    finally:
        conn.close()

    return output_dir
