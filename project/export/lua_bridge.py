"""
lua_bridge.py — Connection X-Plane via FlyWithLua (fichiers JSON)
=================================================================
Alternative a XPlaneConnection (UDP) qui communique avec X-Plane
via un script FlyWithLua (lard_bridge.lua) tournant dans le render loop.

Avantages vs UDP :
  - Datarefs ecrits dans le render loop (meteo XP12 prise en compte)
  - Lecture lat/lon en double precision (vs float32 en UDP RREF)
  - Tous les datarefs d'une frame appliques atomiquement (1 tick = 1 frame)

Protocole :
  Python ecrit command.json {seq, action, drefs, weather}
  Lua lit a chaque frame, applique datarefs, ecrit status.json {ack_seq, ok, actual_pose, ...}
  Python poll status.json jusqu'a ack_seq == seq

Usage :
    from lua_bridge import LuaBridgeConnection
    from xplane_bridge import XPlaneConfig
    conn = LuaBridgeConnection(XPlaneConfig(), exchange_dir)
    conn.setup_view()
    conn.set_camera_pose(lat, lon, alt_m, heading, pitch, roll)
    conn.close()
"""

import json
import math
import os
import time
from pathlib import Path
from dataclasses import dataclass

import mss
import mss.tools
from PIL import Image

try:
    import win32gui
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False


class LuaBridgeConnection:
    """Connection X-Plane via FlyWithLua JSON file exchange.

    Meme API publique que XPlaneConnection pour etre interchangeable.
    """

    def __init__(self, config, exchange_dir):
        """
        :param config: XPlaneConfig
        :param exchange_dir: Path du dossier d'echange (contient command.json / status.json)
        """
        self.config = config
        self.exchange_dir = Path(exchange_dir)
        self.exchange_dir.mkdir(parents=True, exist_ok=True)

        self.cmd_file = self.exchange_dir / "command.json"
        self.cmd_tmp = self.exchange_dir / "command.tmp"
        self.sts_file = self.exchange_dir / "status.json"

        self._seq = 0

        # Point de reference pour conversion lat/lon → local
        self.ref_lat = None
        self.ref_lon = None
        self.ref_elev = None
        self.ref_lx = None
        self.ref_ly = None
        self.ref_lz = None

        # Offset yeux pilote
        self.pilot_eye_x = 0.0
        self.pilot_eye_y = 0.0
        self.pilot_eye_z = 0.0

        # Capture ecran
        self.sct = None
        self.capture_region = None

        # Derniere pose reelle lue
        self._last_actual_pose = None

        # Weather drefs a envoyer avec le prochain set_pose
        self._pending_weather = {}

        # Connexion UDP pour le setup/release lourd
        # (FlyWithLua crashe si trop d'appels dans un seul callback)
        from xplane_bridge import XPlaneConnection
        self._udp = XPlaneConnection(config)

    # ------------------------------------------------------------------
    # Protocole JSON
    # ------------------------------------------------------------------

    def _write_command(self, action, drefs=None, weather=None):
        """Ecrit une commande JSON et retourne le seq.

        Ecriture atomique : .tmp puis rename.
        """
        self._seq += 1
        cmd = {"seq": self._seq, "action": action}
        if drefs:
            cmd["drefs"] = drefs
        if weather:
            cmd["weather"] = weather

        content = json.dumps(cmd, indent=2)

        # Ecriture atomique
        with open(self.cmd_tmp, "w") as f:
            f.write(content)

        # Windows : remove avant rename si fichier existe
        try:
            os.replace(str(self.cmd_tmp), str(self.cmd_file))
        except OSError:
            if self.cmd_file.exists():
                self.cmd_file.unlink()
            self.cmd_tmp.rename(self.cmd_file)

        return self._seq

    def _read_status(self):
        """Lit status.json. Retourne dict ou None si absent/invalide."""
        try:
            with open(self.sts_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    def _wait_ack(self, seq, timeout=None):
        """Poll status.json jusqu'a ack_seq == seq.

        :param seq: sequence number attendu
        :param timeout: timeout en secondes (defaut: config.lua_timeout)
        :return: dict status ou None si timeout
        """
        if timeout is None:
            timeout = getattr(self.config, 'lua_timeout', 5.0)

        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            status = self._read_status()
            if status and status.get("ack_seq") == seq:
                return status
            time.sleep(0.01)  # 10ms poll

        return None

    def _send_and_wait(self, action, drefs=None, weather=None, timeout=None):
        """Ecrit une commande et attend l'ack.

        :return: dict status
        :raises TimeoutError: si pas de reponse
        """
        seq = self._write_command(action, drefs=drefs, weather=weather)
        status = self._wait_ack(seq, timeout=timeout)
        if status is None:
            raise TimeoutError(
                f"FlyWithLua ne repond pas (action={action}, seq={seq}). "
                f"Verifier que X-Plane est lance et lard_bridge.lua est charge."
            )
        if not status.get("ok", False):
            err = status.get("error", "unknown error")
            raise RuntimeError(f"FlyWithLua erreur: {err}")

        # Sauver la pose reelle si presente
        if "actual_pose" in status:
            self._last_actual_pose = status["actual_pose"]

        return status

    # ------------------------------------------------------------------
    # API publique (meme interface que XPlaneConnection)
    # ------------------------------------------------------------------

    def check_connection(self):
        """Verifie que FlyWithLua est joignable via noop."""
        try:
            self._send_and_wait("noop", timeout=5.0)
            return True
        except (TimeoutError, RuntimeError):
            return False

    def setup_view(self):
        """Configure X-Plane pour le rendu automatise.

        Approche hybride :
        1. UDP : setup lourd (pause, overrides, view, popups) — trop d'appels
           pour un seul callback FlyWithLua qui a un watchdog temps-reel.
        2. Lua : lecture reference point, pilot eye, FOV — precision double.
        """
        # Setup lourd via UDP (identique a XPlaneConnection.setup_view)
        self._udp.setup_view()

        # Lire les donnees precises via Lua (double precision lat/lon)
        status = self._send_and_wait("setup", timeout=10.0)

        # Extraire le point de reference
        ref = status.get("ref_point", {})
        self.ref_lat = ref.get("lat", 0.0)
        self.ref_lon = ref.get("lon", 0.0)
        self.ref_elev = ref.get("elev", 0.0)
        self.ref_lx = ref.get("local_x", 0.0)
        self.ref_ly = ref.get("local_y", 0.0)
        self.ref_lz = ref.get("local_z", 0.0)
        print(f"  [LUA] Reference : lat={self.ref_lat:.6f}, lon={self.ref_lon:.6f}, "
              f"elev={self.ref_elev:.1f}m")
        print(f"  [LUA]   local: x={self.ref_lx:.0f}, y={self.ref_ly:.0f}, "
              f"z={self.ref_lz:.0f}")

        # Pilot eye offset
        pe = status.get("pilot_eye", {})
        self.pilot_eye_x = pe.get("x", 0.0)
        self.pilot_eye_y = pe.get("y", 0.0)
        self.pilot_eye_z = pe.get("z", 0.0)
        print(f"  [LUA] Pilot eye offset: x={self.pilot_eye_x:.2f} "
              f"y={self.pilot_eye_y:.2f} z={self.pilot_eye_z:.2f} m")

        # FOV
        fov = status.get("fov_deg", self.config.fov_h)
        print(f"  [LUA] FOV config: H={self.config.fov_h}° V={self.config.fov_v}°"
              f" (dataref={fov:.1f}°)")

        # Recuperer capture region depuis UDP setup (deja fait: resize + mss)
        self.capture_region = self._udp.capture_region
        self.sct = self._udp.sct

    def move_reference_to(self, lat, lon):
        """Deplace l'avion a lat/lon et relit le point de reference.

        Meme logique que XPlaneConnection.move_reference_to() :
        on deplace via la conversion approximative, puis on relit le ref
        pour que les positions proches soient precises.
        """
        # Deplacer via la conversion actuelle (approximative)
        lx, ly, lz = self._latlon_to_local(lat, lon, self.ref_elev)
        drefs = {
            "sim/flightmodel/position/local_x": lx,
            "sim/flightmodel/position/local_y": ly,
            "sim/flightmodel/position/local_z": lz,
        }
        self._send_and_wait("set_pose", drefs=drefs)
        time.sleep(0.15)

        # Relire le point de reference (read_pose retourne ref_point)
        status = self._send_and_wait("read_pose")
        ref = status.get("ref_point", {})
        self.ref_lat = ref.get("lat", self.ref_lat)
        self.ref_lon = ref.get("lon", self.ref_lon)
        self.ref_elev = ref.get("elev", self.ref_elev)
        self.ref_lx = ref.get("local_x", self.ref_lx)
        self.ref_ly = ref.get("local_y", self.ref_ly)
        self.ref_lz = ref.get("local_z", self.ref_lz)
        print(f"  [LUA] Reference deplacee vers lat={lat:.6f}, lon={lon:.6f}")

    def _latlon_to_local(self, lat, lon, alt_m):
        """Convertit lat/lon/alt en local_x/y/z OpenGL X-Plane.

        Identique a XPlaneConnection._latlon_to_local().
        """
        dlat = lat - self.ref_lat
        dlon = lon - self.ref_lon
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(math.radians(self.ref_lat))
        dn = dlat * m_per_deg_lat
        de = dlon * m_per_deg_lon
        lx = self.ref_lx + de
        lz = self.ref_lz - dn
        ly = self.ref_ly + (alt_m - self.ref_elev)
        return lx, ly, lz

    def set_camera_pose(self, lat, lon, alt_m, heading, pitch, roll):
        """Positionne la camera via coordonnees locales OpenGL.

        Convertit lat/lon/alt en local_x/y/z, envoie tous les datarefs
        en une seule commande (appliques atomiquement en 1 frame par Lua).
        """
        lx, ly, lz = self._latlon_to_local(lat, lon, alt_m)

        drefs = {
            "sim/flightmodel/position/local_x": lx,
            "sim/flightmodel/position/local_y": ly,
            "sim/flightmodel/position/local_z": lz,
            "sim/flightmodel/position/theta": pitch,
            "sim/flightmodel/position/phi": roll,
            "sim/flightmodel/position/psi": heading,
            "sim/flightmodel/position/local_vx": 0.0,
            "sim/flightmodel/position/local_vy": 0.0,
            "sim/flightmodel/position/local_vz": 0.0,
        }

        # Inclure les weather drefs en attente
        weather = self._pending_weather if self._pending_weather else None
        self._pending_weather = {}

        self._send_and_wait("set_pose", drefs=drefs, weather=weather)
        time.sleep(self.config.settle_time)

    def read_actual_pose(self):
        """Retourne la pose reelle lue par Lua dans le dernier status.

        Contrairement a XPlaneConnection qui fait 6 RREF round-trips,
        la pose est deja dans le status.json (lecture gratuite).

        :return: dict {lat, lon, alt_m, heading, pitch, roll}
        """
        if self._last_actual_pose:
            return self._last_actual_pose
        # Fallback : demander explicitement
        status = self._send_and_wait("read_pose")
        return status.get("actual_pose", {})

    def read_terrain_elevation(self, lat, lon):
        """Lit l'altitude terrain reelle a une position lat/lon.

        Deplace temporairement l'avion au sol, lit l'elevation, restaure.
        """
        # Sauver la position actuelle
        status = self._send_and_wait("read_pose")
        old_pose = status.get("actual_pose", {})
        old_ref = status.get("ref_point", {})

        # Deplacer au point cible, altitude basse
        lx, ly, lz = self._latlon_to_local(lat, lon, 0.0)
        drefs = {
            "sim/flightmodel/position/local_x": lx,
            "sim/flightmodel/position/local_y": 0.0,
            "sim/flightmodel/position/local_z": lz,
        }
        self._send_and_wait("set_pose", drefs=drefs)
        time.sleep(0.3)

        # Lire l'elevation terrain
        status = self._send_and_wait("read_pose")
        elev = status.get("actual_pose", {}).get("alt_m", 0.0)

        # Restaurer
        if old_ref.get("local_x") is not None:
            drefs_restore = {
                "sim/flightmodel/position/local_x": old_ref["local_x"],
                "sim/flightmodel/position/local_y": old_ref["local_y"],
                "sim/flightmodel/position/local_z": old_ref["local_z"],
            }
            self._send_and_wait("set_pose", drefs=drefs_restore)
            time.sleep(0.2)

        return elev or 0.0

    def capture_frame(self, output_path):
        """Capture la fenetre X-Plane via mss, sauve en JPEG.

        Identique a XPlaneConnection.capture_frame().
        """
        if not self.sct or not self.capture_region:
            return False
        img = self.sct.grab(self.capture_region)
        pil_img = Image.frombytes("RGB", img.size, bytes(img.rgb))
        pil_img.save(str(output_path), quality=90)
        return True

    def apply_weather(self, active_weather, ltp_elevation_msl=0.0):
        """Prepare les datarefs meteo pour la prochaine frame.

        Les datarefs sont envoyes avec le prochain set_camera_pose
        (dans le meme command.json, appliques atomiquement par Lua).

        :param active_weather: liste de (weather_type, severity)
        """
        from sensor_fault_profile import (
            KNOWN_WEATHER_TYPES, weather_severity_to_dref,
        )

        weather_drefs = {}
        active_types = {wt for wt, _ in active_weather}

        for weather_type in KNOWN_WEATHER_TYPES:
            if weather_type in active_types:
                sev = next(s for wt, s in active_weather if wt == weather_type)
                value = weather_severity_to_dref(weather_type, sev)
            else:
                if weather_type == "cloud_low":
                    value = {"base": 3000.0, "top": 3500.0}
                elif weather_type == "temperature":
                    value = 15.0
                else:
                    value = 0.0

            if weather_type == "cloud_low":
                weather_drefs["sim/weather/cloud_base_msl_m[0]"] = value["base"]
                weather_drefs["sim/weather/cloud_tops_msl_m[0]"] = value["top"]
                weather_drefs["sim/weather/cloud_type[0]"] = 4.0  # overcast
            else:
                weather_drefs[KNOWN_WEATHER_TYPES[weather_type]] = value

        # Toujours desactiver METAR pour que nos datarefs aient la priorite
        weather_drefs["sim/weather/use_real_weather_bool"] = 0
        self._pending_weather = weather_drefs

    def setup_weather(self):
        """Desactive la meteo reelle (METAR) via Lua."""
        self._send_and_wait("set_pose", weather={
            "sim/weather/use_real_weather_bool": 0,
        })

    def reset_weather(self, ltp_elevation_msl=0.0):
        """Remet la meteo par defaut (ciel clair, METAR reactif)."""
        self._send_and_wait("set_pose", weather={
            "sim/weather/use_real_weather_bool": 1,
            "sim/weather/rain_percent": 0.0,
            "sim/weather/cloud_base_msl_m[0]": 5000.0,
            "sim/weather/cloud_tops_msl_m[0]": 5500.0,
            "sim/weather/cloud_type[0]": 0.0,
        })

    def release(self):
        """Rend le controle a X-Plane via UDP (unpause, remove overrides, reset meteo)."""
        self._udp.release()

    def close(self):
        """Libere les ressources et nettoie les fichiers d'echange."""
        try:
            self.release()
        except Exception:
            pass
        # Ne pas fermer sct ici — il appartient a _udp
        self._udp.close()
        # Nettoyer les fichiers d'echange
        for f in [self.cmd_file, self.cmd_tmp, self.sts_file]:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass
