"""
xplane_weather.py — Effets meteo X-Plane 12 via XPPython3 (v2, per-scenario)
=============================================================================
Gere la configuration et injection des effets meteo dans X-Plane 12
via le plugin PI_lard_weather.py (XPPython3 / XPLMWeather API).

La meteo est injectee UNE FOIS avant le rendu d'un scenario (pas per-frame).
Le pipeline attend ~60s pour la stabilisation GPU des nuages.

Parametres directs XP12 (plus de severity/from_pct/to_pct) :
  - precip_rate     : taux precipitation [0, 1]
  - cloud_type      : enum XPLMWeather (0=Cirrus, 1=Stratus, 2=Cumulus, 3=Cumulonimbus, -1=off)
  - cloud_coverage  : couverture nuageuse [0, 1]
  - visibility_m    : visibilite en metres [500, 50000]
  - temperature_c   : temperature Celsius [-30, 45] — < 0 = neige
  - time_of_day_h   : heure locale [0, 24]

Communication par fichiers JSON :
  Python ecrit weather_command.json → plugin lit, applique, ecrit weather_status.json
"""

import os
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path


# Visibilite max pour que la pluie soit visible dans XP12
MAX_VISIBILITY_FOR_RAIN_M = 10000.0

# Temps de stabilisation nuages GPU (secondes)
# Teste a ~5-10s avec updateImmediately=True sur XP12
CLOUD_STABILIZATION_DELAY_S = 15


@dataclass
class WeatherConfig:
    """Configuration meteo X-Plane per-scenario (genere par TAF)."""
    rain_intensity: float = 0.0      # 0 = utilise params individuels, >0 = override
    precip_rate: float = 0.0
    cloud_type: float = -1.0         # -1 = pas de nuages
    cloud_coverage: float = 0.0
    cloud_margin_m: float = 200.0    # marge au-dessus de l'avion pour base nuages
    visibility_m: float = 50000.0
    temperature_c: float = 15.0
    time_of_day_h: float = 12.0
    rain_scale: float = 1.0          # taille visuelle des gouttes (sim/private/controls/rain/scale)


def expand_rain_intensity(config):
    """Expanse rain_intensity en params individuels si > 0.

    Mapping rain_intensity → params XP12 (interpolation lineaire entre paliers) :
      0.0  Sec            precip=0.0  vis=50000  cloud_type=-1  cov=0.0
      0.1  Bruine         precip=0.2  vis=12000  Stratus(1)     cov=0.5
      0.3  Pluie legere   precip=0.4  vis=7000   Stratus(1)     cov=0.7
      0.5  Pluie moderee  precip=0.6  vis=4000   Cumulus(2)     cov=0.85
      0.7  Forte pluie    precip=0.8  vis=2000   Cb(3)          cov=1.0
      1.0  Deluge/orage   precip=1.0  vis=800    Cb(3)          cov=1.0

    Modifie config in-place et retourne config.
    """
    ri = config.rain_intensity
    if ri <= 0:
        return config

    # Paliers : (rain_intensity, precip_rate, visibility_m, cloud_type, cloud_coverage)
    PALIERS = [
        (0.0, 0.0, 50000, -1, 0.0),
        (0.1, 0.2, 12000,  1, 0.5),
        (0.3, 0.4,  7000,  1, 0.7),
        (0.5, 0.6,  4000,  2, 0.85),
        (0.7, 0.8,  2000,  3, 1.0),
        (1.0, 1.0,   800,  3, 1.0),
    ]

    # Trouver les 2 paliers encadrants
    for i in range(len(PALIERS) - 1):
        if PALIERS[i][0] <= ri <= PALIERS[i + 1][0]:
            lo, hi = PALIERS[i], PALIERS[i + 1]
            t = (ri - lo[0]) / (hi[0] - lo[0]) if hi[0] != lo[0] else 0.0
            config.precip_rate = lo[1] + t * (hi[1] - lo[1])
            config.visibility_m = lo[2] + t * (hi[2] - lo[2])
            # cloud_type : pas d'interpolation, on prend celui du palier superieur
            config.cloud_type = hi[3] if t > 0 else lo[3]
            config.cloud_coverage = lo[4] + t * (hi[4] - lo[4])
            break

    return config


def has_weather(config):
    """Retourne True si la config modifie la meteo par rapport au defaut."""
    return (config.rain_intensity > 0
            or config.precip_rate > 0
            or config.cloud_type >= 0
            or config.visibility_m < 50000
            or config.temperature_c < 0
            or config.time_of_day_h != 12.0)


def validate_weather(config):
    """Valide les parametres meteo."""
    if not 0.0 <= config.rain_intensity <= 1.0:
        raise ValueError(f"rain_intensity={config.rain_intensity} hors [0, 1]")
    if not 0.0 <= config.precip_rate <= 1.0:
        raise ValueError(f"precip_rate={config.precip_rate} hors [0, 1]")
    if config.cloud_type not in (-1, 0, 1, 2, 3) and not (-1 <= config.cloud_type <= 3):
        raise ValueError(f"cloud_type={config.cloud_type} hors [-1, 3]")
    if not 0.0 <= config.cloud_coverage <= 1.0:
        raise ValueError(f"cloud_coverage={config.cloud_coverage} hors [0, 1]")
    if not 500 <= config.visibility_m <= 50000:
        raise ValueError(f"visibility_m={config.visibility_m} hors [500, 50000]")
    if not -30 <= config.temperature_c <= 45:
        raise ValueError(f"temperature_c={config.temperature_c} hors [-30, 45]")
    if not 0 <= config.time_of_day_h <= 24:
        raise ValueError(f"time_of_day_h={config.time_of_day_h} hors [0, 24]")
    if not 0.1 <= config.rain_scale <= 5.0:
        raise ValueError(f"rain_scale={config.rain_scale} hors [0.1, 5.0]")

    # Note : l'ancienne contrainte "pluie necessite vis <= 10km" a ete retiree.
    # Elle venait de tests avec le code v1 buggé (mauvais enum cloud_type).
    # A re-evaluer apres tests v2.


def local_hour_to_zulu(local_hour, longitude):
    """Convertit une heure locale solaire en heure UTC.

    Approximation par longitude : UTC_offset = longitude / 15.
    Coherent avec le rendu solaire XP12 (erreur max ~1h vs fuseau politique).

    :param local_hour: heure locale [0, 24]
    :param longitude: longitude de l'aeroport (degres, negatif = ouest)
    :return: heure UTC [0, 24)
    """
    utc_offset = longitude / 15.0
    zulu = local_hour - utc_offset
    return zulu % 24.0


def build_plugin_command(config, aircraft_max_alt_m=200.0, longitude=0.0):
    """Construit les parametres JSON pour PI_lard_weather.py.

    Traduit WeatherConfig en params attendus par le plugin.

    :param config: WeatherConfig
    :param aircraft_max_alt_m: altitude max de l'avion dans le scenario (MSL, metres).
        Les nuages sont places au-dessus avec une marge de config.cloud_margin_m.
    :param longitude: longitude de l'aeroport (pour conversion heure locale → UTC)
    """
    # Cap visibilite si pluie active : au-dela de 10km la pluie est quasi invisible
    visibility = config.visibility_m
    if config.precip_rate > 0:
        visibility = min(visibility, MAX_VISIBILITY_FOR_RAIN_M)

    params = {
        "precip_rate": config.precip_rate,
        "visibility_m": visibility,
        "radius_nm": 50.0,
        "max_alt_ft": 30000.0,
    }

    # Temperature
    if config.temperature_c != 15.0:
        params["temperature_c"] = config.temperature_c

    # Heure du jour — conversion locale → UTC via longitude
    zulu_h = local_hour_to_zulu(config.time_of_day_h, longitude)
    params["time_of_day_h"] = zulu_h

    # Nuages — base dynamique : au-dessus de l'avion avec marge
    cloud_base = aircraft_max_alt_m + config.cloud_margin_m
    if config.cloud_type >= 0:
        # Nuages manuels demandes par l'utilisateur
        params["cloud_type"] = config.cloud_type
        params["cloud_coverage"] = config.cloud_coverage
        params["cloud_base_msl"] = cloud_base
        params["cloud_top_msl"] = cloud_base + 2000.0
    elif config.precip_rate > 0:
        # Pluie sans nuages manuels : forcer Cumulonimbus
        # XP12 ne genere pas ses propres nuages via l'API setWeatherAtLocation,
        # contrairement a l'interface utilisateur
        params["cloud_type"] = 3.0   # Cumulonimbus
        params["cloud_coverage"] = 1.0
        params["cloud_base_msl"] = cloud_base
        params["cloud_top_msl"] = cloud_base + 2000.0

    # Taille des gouttes (dataref prive, ignore si non supporte)
    if config.rain_scale != 1.0:
        params["rain_scale"] = config.rain_scale

    return params


def save_weather_profile(config, output_path):
    """Sauvegarde le profil meteo dans un fichier JSON."""
    output_path = Path(output_path)
    profile = {
        "version": 2,
        "mode": "per_scenario",
        "weather": asdict(config),
    }
    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"  .json weather -> {output_path}")
    return str(output_path)


def load_weather_profile(profile_path):
    """Charge un profil meteo depuis un fichier JSON."""
    with open(profile_path, "r") as f:
        profile = json.load(f)

    # Support v1 (ancien format per-frame) : ignorer
    if profile.get("version", 1) < 2:
        print(f"  [WEATHER] Ancien format v1 ignore : {profile_path}")
        return None

    return WeatherConfig(**profile["weather"])


# ---------------------------------------------------------------------------
# Communication avec le plugin XPPython3 (PI_lard_weather.py)
# ---------------------------------------------------------------------------

DEFAULT_EXCHANGE_DIR = None
_seq_counter = 0


def set_exchange_dir(xplane_dir):
    """Configure le dossier d'echange pour la communication avec le plugin."""
    global DEFAULT_EXCHANGE_DIR
    DEFAULT_EXCHANGE_DIR = os.path.join(
        xplane_dir, "Resources", "plugins", "PythonPlugins", "lard_exchange"
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

    # Ecriture atomique (os.replace = atomique sur POSIX, remplace sur Windows)
    tmp = cmd_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cmd, f)
    for _ in range(20):
        try:
            os.replace(tmp, cmd_file)
            break
        except OSError:
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


def set_time_of_day(hour, xplane_dir=None):
    """Change l'heure du jour dans X-Plane via dataref UDP.

    :param hour: heure locale [0-24]
    :param xplane_dir: chemin X-Plane (pour trouver le port UDP) — non utilise
    """
    # L'heure est geree par dataref sim/time/zulu_time_sec via UDP
    # Pour l'instant on l'envoie comme parametre weather au plugin
    # qui le gere via xp.setDataf()
    pass


def inject_weather(config, aircraft_max_alt_m=200.0, longitude=0.0):
    """Injecte la meteo dans X-Plane et attend la stabilisation.

    Appeler UNE FOIS avant le rendu du scenario.

    :param config: WeatherConfig
    :param aircraft_max_alt_m: altitude max de l'avion (MSL, metres)
    :param longitude: longitude de l'aeroport (pour conversion heure locale → UTC)
    :return: True si l'injection a reussi
    """
    if not has_weather(config):
        print(f"  [WEATHER] Pas de meteo active, skip")
        return True

    params = build_plugin_command(config, aircraft_max_alt_m=aircraft_max_alt_m, longitude=longitude)
    result = _send_weather_command("set_weather", weather=params)

    if not result or not result.get("ok"):
        print(f"  [WEATHER] ECHEC injection meteo")
        return False

    # Log
    parts = []
    if config.precip_rate > 0:
        parts.append(f"precip={config.precip_rate:.2f}")
    if config.cloud_type >= 0:
        cloud_names = {0: "Cirrus", 1: "Stratus", 2: "Cumulus", 3: "Cumulonimbus"}
        parts.append(f"clouds={cloud_names.get(int(config.cloud_type), '?')}"
                     f"({config.cloud_coverage:.0%})")
    if config.visibility_m < 50000:
        parts.append(f"vis={config.visibility_m:.0f}m")
    if config.temperature_c < 0:
        parts.append(f"temp={config.temperature_c:.0f}C (neige)")
    if config.time_of_day_h != 12.0:
        zulu_h = local_hour_to_zulu(config.time_of_day_h, longitude)
        parts.append(f"heure={config.time_of_day_h:.0f}h local -> {zulu_h:.1f}h UTC")
    parts.append(f"avion_max={aircraft_max_alt_m:.0f}m")
    if config.cloud_type >= 0:
        cloud_base = aircraft_max_alt_m + config.cloud_margin_m
        parts.append(f"nuages={cloud_base:.0f}-{cloud_base + 2000:.0f}m MSL")

    print(f"  [WEATHER] Injecte : {', '.join(parts)}")

    # Attente stabilisation nuages GPU
    needs_cloud_wait = (config.cloud_type >= 0 and config.cloud_coverage > 0) or config.precip_rate > 0
    if needs_cloud_wait:
        print(f"  [WEATHER] Attente {CLOUD_STABILIZATION_DELAY_S}s stabilisation nuages...")
        time.sleep(CLOUD_STABILIZATION_DELAY_S)
    else:
        # Visibilite/brouillard seul = quasi-instantane
        time.sleep(3)

    print(f"  [WEATHER] Stabilisation terminee")
    return True


def reset_weather():
    """Remet la meteo par defaut (ciel clair)."""
    result = _send_weather_command("clear_weather")
    if result and result.get("ok"):
        print(f"  [WEATHER] Clear OK")
    else:
        print(f"  [WEATHER] Clear ECHEC")
