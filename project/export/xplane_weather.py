"""
xplane_weather.py — Effets meteo X-Plane 12 via XPPython3
==========================================================
Gere la configuration, validation et injection des effets meteo
dans X-Plane 12 via le plugin PI_lard_weather.py (XPPython3).

Communication par fichiers JSON :
  Python ecrit weather_command.json → plugin lit, applique, ecrit weather_status.json

Effets disponibles et testes sur XP12 :
  - rain       : pluie (severity → precip_rate, necessite visibility <= 10km)
  - cloud      : nuages (severity → couverture, type overcast)
  - visibility : brouillard (severity → 500m a 50km)

Chaque effet a severity + from_pct + to_pct (meme pattern que sensor_faults).
"""

import os
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path


# Effets meteo XP12 disponibles (testes fonctionnels via XPPython3)
KNOWN_WEATHER_TYPES = {"rain", "cloud", "visibility"}

# Visibilite max pour que la pluie soit visible dans XP12
MAX_VISIBILITY_FOR_RAIN_M = 10000.0


@dataclass
class WeatherConfig:
    """Configuration d'un effet meteo X-Plane (genere par TAF)."""
    weather_type: str
    severity: float
    from_pct: float
    to_pct: float


def validate_weather(weather_list):
    """Valide les contraintes sur les effets meteo."""
    for i, w in enumerate(weather_list):
        prefix = f"[Weather {i}] ({w.weather_type})"
        if w.weather_type not in KNOWN_WEATHER_TYPES:
            raise ValueError(
                f"{prefix} Type inconnu '{w.weather_type}'. "
                f"Types disponibles : {sorted(KNOWN_WEATHER_TYPES)}"
            )
        if not 0.0 <= w.severity <= 1.0:
            raise ValueError(f"{prefix} severity={w.severity} hors [0.0, 1.0]")
        if not 0.0 <= w.from_pct <= 100.0:
            raise ValueError(f"{prefix} from_pct={w.from_pct} hors [0, 100]")
        if not 0.0 <= w.to_pct <= 100.0:
            raise ValueError(f"{prefix} to_pct={w.to_pct} hors [0, 100]")
        if w.from_pct >= w.to_pct:
            raise ValueError(
                f"{prefix} from_pct={w.from_pct} >= to_pct={w.to_pct}"
            )


def compute_frame_weather(weather_list, n_frames):
    """Calcule les effets meteo actifs par frame.

    frame 0 = 0% (debut approche, loin), frame N-1 = 100% (pres de la piste).
    """
    if not weather_list or n_frames <= 0:
        return [[] for _ in range(max(n_frames, 0))]
    per_frame = []
    for i in range(n_frames):
        pct = (i / max(n_frames - 1, 1)) * 100.0
        active = [(w.weather_type, w.severity) for w in weather_list if w.from_pct <= pct <= w.to_pct]
        per_frame.append(active)
    return per_frame


def severity_to_xp12(weather_type, severity):
    """Convertit severity [0,1] en parametres XPPython3 pour PI_lard_weather.py.

    Retourne un dict de parametres a merger dans la commande weather JSON.
    """
    if weather_type == "rain":
        # precip_rate 0-1, visibility forcee <= 10km pour rendu visuel
        vis = MAX_VISIBILITY_FOR_RAIN_M * (1.0 - 0.5 * severity)  # 10km → 5km
        return {
            "precip_rate": severity,
            "visibility_m": vis,
        }
    elif weather_type == "cloud":
        # coverage = severity, nuages bas 100-600m MSL
        # cloud_type: 1=few, 2=scattered, 3=broken, 4=overcast
        if severity < 0.25:
            cloud_type = 1.0
        elif severity < 0.5:
            cloud_type = 2.0
        elif severity < 0.75:
            cloud_type = 3.0
        else:
            cloud_type = 4.0
        return {
            "cloud_base_msl": 100.0,
            "cloud_top_msl": 600.0,
            "cloud_type": cloud_type,
            "cloud_coverage": severity,
        }
    elif weather_type == "visibility":
        # 50km (severity=0) → 500m (severity=1), echelle log
        vis = 50000.0 * (0.01 ** severity)  # 50000 → 500
        return {"visibility_m": vis}
    return {}


def build_weather_command(active_weather):
    """Construit les parametres weather JSON a partir des effets actifs.

    Combine tous les effets actifs en une seule commande pour le plugin.
    Si rain et visibility sont actifs ensemble, la visibility de rain
    est overridee par celle de visibility (plus restrictive).

    :param active_weather: liste de (weather_type, severity)
    :return: dict de parametres weather pour PI_lard_weather.py
    """
    if not active_weather:
        return None

    params = {}
    has_rain = False
    has_visibility = False

    for weather_type, severity in active_weather:
        xp_params = severity_to_xp12(weather_type, severity)
        if weather_type == "rain":
            has_rain = True
        elif weather_type == "visibility":
            has_visibility = True
        params.update(xp_params)

    # Si rain actif sans visibility explicite, forcer vis <= 10km
    if has_rain and not has_visibility:
        if params.get("visibility_m", 50000) > MAX_VISIBILITY_FOR_RAIN_M:
            params["visibility_m"] = MAX_VISIBILITY_FOR_RAIN_M

    return params


def save_weather_profile(weather_list, n_frames, output_path):
    """Sauvegarde le profil meteo dans un fichier JSON."""
    output_path = Path(output_path)
    per_frame = compute_frame_weather(weather_list, n_frames)
    per_frame_summary = {}
    for i, active in enumerate(per_frame):
        if active:
            per_frame_summary[str(i)] = [
                {"type": wt, "severity": sev} for wt, sev in active
            ]
    profile = {
        "weather": [asdict(w) for w in weather_list],
        "n_frames": n_frames,
        "per_frame_summary": per_frame_summary,
        "n_frames_affected": len(per_frame_summary),
    }
    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"  .json weather -> {output_path}")
    return str(output_path)


def load_weather_profile(profile_path):
    """Charge un profil meteo depuis un fichier JSON."""
    with open(profile_path, "r") as f:
        profile = json.load(f)
    weather_list = [WeatherConfig(**wc) for wc in profile["weather"]]
    return weather_list, profile["n_frames"]


# ---------------------------------------------------------------------------
# Communication avec le plugin XPPython3 (PI_lard_weather.py)
# ---------------------------------------------------------------------------

# Dossier d'echange par defaut
DEFAULT_EXCHANGE_DIR = None  # configure par set_exchange_dir()

_seq_counter = 0


def set_exchange_dir(xplane_dir):
    """Configure le dossier d'echange pour la communication avec le plugin."""
    global DEFAULT_EXCHANGE_DIR
    DEFAULT_EXCHANGE_DIR = os.path.join(
        xplane_dir, "Resources", "plugins", "FlyWithLua", "Scripts", "lard_exchange"
    )
    os.makedirs(DEFAULT_EXCHANGE_DIR, exist_ok=True)


def _send_weather_command(action, weather=None, timeout=5.0):
    """Envoie une commande au plugin PI_lard_weather.py et attend l'ack."""
    global _seq_counter

    if DEFAULT_EXCHANGE_DIR is None:
        raise RuntimeError("Exchange dir not set. Call set_exchange_dir(xplane_dir) first.")

    _seq_counter += 1
    seq = _seq_counter

    cmd_file = os.path.join(DEFAULT_EXCHANGE_DIR, "weather_command.json")
    sts_file = os.path.join(DEFAULT_EXCHANGE_DIR, "weather_status.json")

    cmd = {"seq": seq, "action": action}
    if weather:
        cmd["weather"] = weather

    # Ecriture atomique
    tmp = cmd_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cmd, f)
    for _ in range(20):
        try:
            if os.path.exists(cmd_file):
                os.remove(cmd_file)
            os.rename(tmp, cmd_file)
            break
        except PermissionError:
            time.sleep(0.05)

    # Attente ack
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with open(sts_file, "r") as f:
                status = json.load(f)
            if status.get("ack_seq") == seq:
                if not status.get("ok"):
                    print(f"  [WEATHER] Plugin error: {status.get('error')}")
                return status
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        time.sleep(0.05)

    print(f"  [WEATHER] Timeout — plugin ne repond pas (seq={seq})")
    return None


def check_plugin():
    """Verifie que le plugin XPPython3 est actif."""
    result = _send_weather_command("noop", timeout=3.0)
    return result is not None and result.get("ok", False)


def apply_weather(active_weather):
    """Applique les effets meteo actifs via le plugin XPPython3.

    :param active_weather: liste de (weather_type, severity) ou liste vide
    """
    params = build_weather_command(active_weather)
    if params:
        _send_weather_command("set_weather", weather=params)
    else:
        _send_weather_command("clear_weather")


def setup_weather():
    """Initialise la meteo (desactive METAR via le plugin)."""
    # Le plugin desactive use_real_weather_bool automatiquement au set_weather
    pass


def reset_weather():
    """Remet la meteo par defaut (ciel clair)."""
    _send_weather_command("clear_weather")
