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
    window_width: int = 1024        # Largeur zone client X-Plane (carre)
    window_height: int = 1024       # Hauteur zone client X-Plane (carre)
    fov_h: float = 65.0             # FOV horizontal (reglages X-Plane)
    fov_v: float = 65.0             # FOV vertical (= horizontal car fenetre carree)
    exchange_dir: str = ""          # Dossier echange FlyWithLua (auto si vide)
    lua_timeout: float = 5.0        # Timeout attente reponse Lua (sec)


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
    """Trouve la fenetre X-Plane principale (la plus grande) et retourne son HWND + rect.

    :return: dict {hwnd, rect, title} ou None si introuvable
    """
    if not HAS_WIN32:
        return None
    candidates = []

    def callback(hwnd, _):
        title = win32gui.GetWindowText(hwnd)
        if "X-Plane" in title and win32gui.IsWindowVisible(hwnd):
            rect = win32gui.GetWindowRect(hwnd)
            area = (rect[2] - rect[0]) * (rect[3] - rect[1])
            candidates.append({'hwnd': hwnd, 'rect': rect, 'title': title, 'area': area})
        return True

    win32gui.EnumWindows(callback, None)
    if not candidates:
        return None
    best = max(candidates, key=lambda c: c['area'])
    return {'hwnd': best['hwnd'], 'rect': best['rect'], 'title': best['title']}


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
        # Offset yeux pilote (cockpit → reference avion)
        self.pilot_eye_x = 0.0   # lateral (m)
        self.pilot_eye_y = 0.0   # vertical (m)
        self.pilot_eye_z = 0.0   # longitudinal (m)
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

    def move_reference_to(self, lat, lon):
        """Deplace l'avion a lat/lon et relit le point de reference.

        La conversion _latlon_to_local est precise pres du point de reference
        mais diverge avec la distance (~22m a 3km). En placant le ref au seuil
        de piste, toutes les positions de la trajectoire (qui convergent vers
        le seuil) auront une conversion precise la ou ca compte.
        """
        # Utiliser la conversion actuelle (approximative) pour deplacer
        lx, ly, lz = self._latlon_to_local(lat, lon, self.ref_elev)
        self.send_dref("sim/flightmodel/position/local_x", lx)
        self.send_dref("sim/flightmodel/position/local_y", ly)
        self.send_dref("sim/flightmodel/position/local_z", lz)
        time.sleep(0.15)
        # Relire : maintenant le ref est pres de la cible
        self._read_reference_point()
        print(f"  [XPLANE] Reference deplacee vers lat={lat:.6f}, lon={lon:.6f}")

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

        # Resize fenetre a la taille cible
        if HAS_WIN32:
            resize_xplane_window(self.config.window_width, self.config.window_height)
            print(f"  [XPLANE] Fenetre resizee -> {self.config.window_width}x{self.config.window_height}")

        # Fermer tous les panneaux/popups d'instruments (GPS, radio, etc.)
        popup_close_commands = [
            # Garmin G1000
            "sim/GPS/g1000n1_popup_close",
            "sim/GPS/g1000n3_popup_close",
            # Garmin 430/530
            "sim/instruments/G430n1_popup_close",
            "sim/instruments/G430n2_popup_close",
            "sim/instruments/G530n1_popup_close",
            "sim/instruments/G530n2_popup_close",
            # Map / FMS
            "sim/instruments/map_close",
            # Radios / transponder
            "sim/instruments/com1_standy_flip_close",
            "sim/instruments/com2_standy_flip_close",
            # Generic instrument popups
            "sim/instruments/popup_1_close",
            "sim/instruments/popup_2_close",
            "sim/instruments/popup_3_close",
            "sim/instruments/popup_4_close",
            "sim/instruments/popup_5_close",
            "sim/instruments/popup_6_close",
            "sim/instruments/popup_7_close",
            "sim/instruments/popup_8_close",
            "sim/instruments/popup_9_close",
            "sim/instruments/popup_10_close",
            "sim/instruments/popup_11_close",
            "sim/instruments/popup_12_close",
        ]
        for cmd in popup_close_commands:
            self.send_command(cmd)
        time.sleep(0.3)

        # Masquer tous les panneaux via le dataref show_popup
        for i in range(20):
            self.send_dref(f"sim/cockpit2/radios/indicators/show_popup[{i}]", 0.0)
        time.sleep(0.2)

        # Desactiver le mouse yoke (reticule au centre de l'ecran)
        self.send_dref("sim/operation/override/override_joystick", 1.0)
        self.send_dref("sim/joystick/mouse_is_yoke", 0.0)
        time.sleep(0.1)

        # Cacher le curseur dans la fenetre X-Plane sans deplacer la souris
        # On desactive juste le mouse yoke (ci-dessus) — pas de SetCursorPos
        # pour ne pas perturber l'utilisateur.

        # Lire l'offset yeux pilote du modele avion charge
        # acf_peX = lateral, acf_peY = vertical, acf_peZ = longitudinal (metres)
        self.pilot_eye_x = self.read_dref("sim/aircraft/view/acf_peX", 11) or 0.0
        self.pilot_eye_y = self.read_dref("sim/aircraft/view/acf_peY", 12) or 0.0
        self.pilot_eye_z = self.read_dref("sim/aircraft/view/acf_peZ", 13) or 0.0
        print(f"  [XPLANE] Pilot eye offset: x={self.pilot_eye_x:.2f} "
              f"y={self.pilot_eye_y:.2f} z={self.pilot_eye_z:.2f} m")

        # Point de reference
        self._read_reference_point()

        # Capture region (pas de resize — on garde la fenetre telle quelle)
        if HAS_WIN32:
            self.capture_region = get_xplane_capture_region()
            if self.capture_region:
                print(f"  [XPLANE] Capture : {self.capture_region['width']}x"
                      f"{self.capture_region['height']}")
            self.sct = mss.mss()

        # --- FOV camera ---
        # On ne touche PAS au FOV — on utilise les reglages X-Plane (65° H, 42.3° V).
        # Lire le dataref pour verifier la coherence.
        readback = self.read_dref("sim/graphics/view/field_of_view_deg", 10)
        print(f"  [XPLANE] FOV config: H={self.config.fov_h}° V={self.config.fov_v}°"
              f" (dataref={readback:.1f}°)" if readback else
              f"  [XPLANE] FOV config: H={self.config.fov_h}° V={self.config.fov_v}°"
              f" (dataref timeout)")

    def set_camera_pose(self, lat, lon, alt_m, heading, pitch, roll):
        """Positionne la camera via coordonnees locales OpenGL.

        Convertit lat/lon/alt en local_x/y/z pour une precision metrique.
        NOTE: pas de compensation pilot eye pour l'instant — a investiguer
        si forward_with_nothing applique ou non l'offset acf_peX/Y/Z.
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

    def read_actual_pose(self):
        """Relit la position/orientation reelle depuis X-Plane apres positionnement.

        Retourne la position avion telle que lue par X-Plane.
        NOTE: pas de compensation pilot eye pour l'instant.

        :return: dict {lat, lon, alt_m, heading, pitch, roll}
        """
        lat = self.read_dref("sim/flightmodel/position/latitude", 20)
        lon = self.read_dref("sim/flightmodel/position/longitude", 21)
        elev = self.read_dref("sim/flightmodel/position/elevation", 22)
        theta = self.read_dref("sim/flightmodel/position/theta", 23)
        phi = self.read_dref("sim/flightmodel/position/phi", 24)
        psi = self.read_dref("sim/flightmodel/position/psi", 25)
        return {
            "lat": lat, "lon": lon, "alt_m": elev,
            "heading": psi, "pitch": theta, "roll": phi,
        }

    def read_terrain_elevation(self, lat, lon):
        """Lit l'altitude terrain reelle a une position lat/lon.

        Deplace temporairement l'avion au sol a cette position,
        lit l'elevation, puis restaure.

        :return: altitude terrain MSL en metres
        """
        # Sauver la position actuelle
        old_lx = self.read_dref("sim/flightmodel/position/local_x", 20)
        old_ly = self.read_dref("sim/flightmodel/position/local_y", 21)
        old_lz = self.read_dref("sim/flightmodel/position/local_z", 22)

        # Deplacer au point cible, altitude basse pour toucher le terrain
        lx, ly, lz = self._latlon_to_local(lat, lon, 0.0)
        self.send_dref("sim/flightmodel/position/local_x", lx)
        self.send_dref("sim/flightmodel/position/local_y", 0.0)  # y=up, poser au sol
        self.send_dref("sim/flightmodel/position/local_z", lz)
        time.sleep(0.3)

        # Lire l'elevation terrain (AGL = 0 → elevation = altitude terrain)
        elev = self.read_dref("sim/flightmodel/position/elevation", 23)

        # Restaurer
        if old_lx is not None:
            self.send_dref("sim/flightmodel/position/local_x", old_lx)
            self.send_dref("sim/flightmodel/position/local_y", old_ly)
            self.send_dref("sim/flightmodel/position/local_z", old_lz)
            time.sleep(0.2)

        return elev or 0.0

    def capture_frame(self, output_path):
        """Capture la fenetre X-Plane telle quelle, sauve en JPEG.

        Pas de crop ni resize — on garde la resolution native de la fenetre.
        Le FOV est lu depuis les parametres X-Plane (H=28°, V=22.56° par defaut).

        :param output_path: chemin du fichier de sortie (.jpg)
        :return: True si capture reussie
        """
        if not self.sct or not self.capture_region:
            return False
        img = self.sct.grab(self.capture_region)
        pil_img = Image.frombytes("RGB", img.size, bytes(img.rgb))
        pil_img.save(str(output_path), quality=90)
        return True

    def apply_weather(self, active_weather, ltp_elevation_msl=0.0):
        """Applique les effets meteo actifs pour la frame courante.

        :param active_weather: liste de (weather_type, severity) actifs
                               ou liste vide si aucun effet actif
        :param ltp_elevation_msl: altitude MSL du seuil de piste (pour cloud_low)
        """
        from sensor_fault_profile import (
            KNOWN_WEATHER_TYPES, weather_severity_to_dref,
        )

        active_types = {wt for wt, _ in active_weather}

        for weather_type in KNOWN_WEATHER_TYPES:
            if weather_type in active_types:
                sev = next(s for wt, s in active_weather if wt == weather_type)
                value = weather_severity_to_dref(weather_type, sev, ltp_elevation_msl)
            else:
                value = {"base": ltp_elevation_msl + 3000.0, "top": ltp_elevation_msl + 3500.0} \
                    if weather_type == "cloud_low" else 0.0

            if weather_type == "cloud_low":
                self.send_dref("sim/weather/cloud_base_msl_m[0]", value["base"])
                self.send_dref("sim/weather/cloud_tops_msl_m[0]", value["top"])
            else:
                self.send_dref(KNOWN_WEATHER_TYPES[weather_type], value)

    def setup_weather(self):
        """Configure X-Plane pour accepter la meteo manuelle.

        Desactive la meteo reelle (METAR) pour que nos DREF ne soient
        pas ecrases par les donnees meteorologiques en temps reel.
        """
        # Forcer meteo manuelle (pas de METAR/real weather)
        self.send_dref("sim/weather/use_real_weather_bool", 0.0)
        time.sleep(0.1)

    def reset_weather(self, ltp_elevation_msl=0.0):
        """Remet la meteo par defaut (clair) a la fin du rendu."""
        self.send_dref("sim/weather/rain_percent", 0.0)
        self.send_dref("sim/weather/cloud_base_msl_m[0]", ltp_elevation_msl + 3000.0)
        self.send_dref("sim/weather/cloud_tops_msl_m[0]", ltp_elevation_msl + 3500.0)

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

# ---------------------------------------------------------------------------
# Factory : choix du backend (FlyWithLua ou UDP)
# ---------------------------------------------------------------------------

def create_connection(config=None):
    """Cree la meilleure connexion X-Plane disponible.

    1. Si FlyWithLua est detecte (lard_bridge.lua present + repond au noop) → LuaBridgeConnection
    2. Sinon → XPlaneConnection (UDP, mode degrade)

    :param config: XPlaneConfig (optionnel)
    :return: LuaBridgeConnection ou XPlaneConnection
    """
    config = config or XPlaneConfig()

    # Tenter FlyWithLua si xplane_dir est configure
    if config.xplane_dir:
        xp = Path(config.xplane_dir)
        lua_script = xp / "Resources" / "plugins" / "FlyWithLua" / "Scripts" / "lard_bridge.lua"

        if lua_script.exists():
            # Determiner le dossier d'echange
            if config.exchange_dir:
                exchange_dir = Path(config.exchange_dir)
            else:
                exchange_dir = lua_script.parent / "lard_exchange"

            try:
                from lua_bridge import LuaBridgeConnection
                conn = LuaBridgeConnection(config, exchange_dir)
                if conn.check_connection():
                    print(f"  [XPLANE] FlyWithLua detecte et actif -> mode Lua")
                    return conn
                else:
                    print(f"  [XPLANE] FlyWithLua detecte mais pas de reponse -> fallback UDP")
            except Exception as e:
                print(f"  [XPLANE] Erreur FlyWithLua: {e} -> fallback UDP")

    print(f"  [XPLANE] Mode UDP (pas de FlyWithLua)")
    return XPlaneConnection(config)


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

def save_poses_json(flight_data, fps, scenario_name, output_path, ltp_alt=None):
    """Sauvegarde les poses dans un fichier JSON universel.

    :param flight_data: list de tuples (lon, lat, alt, yaw, pitch_ges, roll)
    :param fps: frames par seconde
    :param scenario_name: nom du scenario (ex: LFPO_24)
    :param output_path: chemin du fichier JSON de sortie
    :param ltp_alt: altitude MSL du LTP (seuil de piste) en metres, pour la meteo
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
        "ltp_alt": float(ltp_alt) if ltp_alt is not None else 0.0,
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

def render_scenario(poses_path, output_dir, config=None, weather_profile_path=None):
    """Rend un scenario complet via X-Plane.

    Lit le fichier poses.json, injecte chaque pose dans X-Plane,
    capture la fenetre via mss, et sauve les images en JPEG.

    :param poses_path: chemin vers le fichier poses.json
    :param output_dir: dossier de sortie pour les images (footage/)
    :param config: XPlaneConfig (optionnel)
    :param weather_profile_path: chemin vers weather_profile.json (optionnel)
    :return: Path du dossier de sortie
    """
    config = config or XPlaneConfig()
    data = load_poses_json(poses_path)
    ltp_alt = data.get("ltp_alt", 0.0)

    # Charger le profil meteo si present
    weather_per_frame = None
    if weather_profile_path and Path(weather_profile_path).exists():
        from sensor_fault_profile import load_weather_profile, compute_frame_weather
        weather_list, _ = load_weather_profile(weather_profile_path)
        if weather_list:
            weather_per_frame = compute_frame_weather(weather_list, data["n_frames"])
            weather_str = ", ".join(
                f"{w.weather_type}({w.severity:.2f})[{w.from_pct:.0f}-{w.to_pct:.0f}%]"
                for w in weather_list
            )
            print(f"  [XPLANE] Meteo active : {weather_str}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_name = data["scenario_name"]
    n_frames = data["n_frames"]

    print(f"  [XPLANE] Rendu de {scenario_name} ({n_frames} frames)...")

    conn = create_connection(config)

    try:
        if not conn.check_connection():
            raise ConnectionError(
                "Impossible de joindre X-Plane. "
                "Verifier que X-Plane 12 est lance et ecoute sur "
                f"{config.host}:{config.port}"
            )

        conn.setup_view()

        # Configurer la meteo si un profil est actif
        if weather_per_frame:
            conn.setup_weather()

        # Deplacer le point de reference au seuil de piste (derniere pose).
        # La conversion _latlon_to_local est precise pres du ref mais diverge
        # avec la distance (~22m a 3km). En prenant le ref pres du seuil,
        # les frames proches (ou la precision bbox compte le plus) seront exactes.
        last_pose = data["poses"][-1]
        conn.move_reference_to(last_pose["lat"], last_pose["lon"])

        # Sauver la config de rendu (FOV + resolution + pilot eye) pour le labeling GT
        render_cfg = {
            "width": conn.capture_region["width"] if conn.capture_region else config.window_width,
            "height": conn.capture_region["height"] if conn.capture_region else config.window_height,
            "fov_h": config.fov_h,
            "fov_v": config.fov_v,
            "pilot_eye_x": conn.pilot_eye_x,  # lateral (m, + = droite)
            "pilot_eye_y": conn.pilot_eye_y,  # vertical (m, + = haut)
            "pilot_eye_z": conn.pilot_eye_z,  # longitudinal (m, + = avant/nez)
        }
        render_cfg_file = output_dir.parent / "render_config.json"
        with open(render_cfg_file, "w") as rf:
            json.dump(render_cfg, rf, indent=2)
        print(f"  [XPLANE] Render config : {render_cfg['width']}x{render_cfg['height']}"
              f" FOV {render_cfg['fov_h']}x{render_cfg['fov_v']}°")

        t_start = time.perf_counter()

        # Padding identique a LARD : img_digits = len(str(n_frames - 1))
        img_digits = len(str(n_frames - 1))

        actual_poses = []  # Poses reelles lues depuis X-Plane

        for i, pose in enumerate(data["poses"]):
            xp = convert_pose_ges_to_xplane(
                pose["lon"], pose["lat"], pose["alt_m"],
                pose["heading"], pose["pitch_ges"], pose["roll"],
            )

            # Appliquer la meteo AVANT la pose (les drefs sont envoyes avec set_camera_pose)
            if weather_per_frame:
                active_weather = weather_per_frame[i] if i < len(weather_per_frame) else []
                conn.apply_weather(active_weather, ltp_elevation_msl=ltp_alt)

            conn.set_camera_pose(
                xp["lat"], xp["lon"], xp["alt_m"],
                xp["heading"], xp["pitch"], xp["roll"],
            )

            # Nommage 0-based, padding LARD : {name}_{i:0Nd}.jpg
            dst = output_dir / f"{scenario_name}_{str(i).zfill(img_digits)}.jpg"
            conn.capture_frame(dst)

            # Lire la position reelle pour le YAML corrige
            real = conn.read_actual_pose()
            if real["lat"] is not None:
                # Convertir X-Plane → GES (inverse de convert_pose_ges_to_xplane)
                actual_poses.append({
                    "lon": real["lon"],
                    "lat": real["lat"],
                    "alt_m": real["alt_m"],
                    "heading": real["heading"],
                    "pitch_ges": (real["pitch"] or 0.0) + 90.0,
                    "roll": real["roll"],
                })
            else:
                # Fallback sur la pose commandee
                actual_poses.append(pose)

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

        # Sauver les poses reelles pour regeneration GT
        actual_path = output_dir.parent / "poses_actual.json"
        with open(actual_path, "w") as f:
            json.dump(actual_poses, f, indent=2)
        print(f"  [XPLANE] Poses reelles -> {actual_path}")

    finally:
        # Remettre la meteo par defaut avant de liberer
        if weather_per_frame:
            conn.reset_weather(ltp_elevation_msl=ltp_alt)
        conn.close()

    return output_dir
