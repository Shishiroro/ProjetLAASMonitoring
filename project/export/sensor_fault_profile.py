"""
sensor_fault_profile.py — Profil de fautes capteur pour trajectoire
===================================================================
Gere la configuration, validation et application des fautes capteur
camera definies dans le XML utilisateur via TAF.

Chaque faute a :
  - fault_type : type de degradation (fog, dead_pixels, motion_blur, etc.)
  - severity   : degre de severite [0.0, 1.0]
  - from_pct   : debut d'application en % de la trajectoire [0, 100]
  - to_pct     : fin d'application en % de la trajectoire [0, 100]

Usage :
    faults = [FaultConfig("fog", 0.6, 20.0, 80.0)]
    validate_faults(faults)
    per_frame = compute_frame_faults(faults, n_frames=150)
    save_fault_profile(faults, 150, "runs/LFPO_24/fault_profile.json")
"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path


# Types de fautes connus (cles de ERROR_REGISTRY dans camera_sensor_errors.py)
KNOWN_FAULT_TYPES = {
    "gaussian_noise", "shot_noise", "salt_pepper", "dead_pixels",
    "motion_blur", "defocus_blur", "rolling_shutter",
    "overexposure", "underexposure",
    "vignetting", "chromatic_aberration", "radial_distortion", "lens_flare",
    "banding", "jpeg_artifacts", "color_shift",
    "condensation", "dirt_on_lens",
    "fog", "zoom_blur", "contrast", "pixelate",
}

# Types d'effets meteo X-Plane 12 (datarefs controlables via UDP DREF)
# Testes ecrivables via test_weather_drefs.py (2026-03-26).
# Chaque type a severity + from_pct + to_pct, meme pattern que les fautes capteur.
# cloud_low utilise 2 datarefs (base + top), gere dans xplane_bridge.py.
KNOWN_WEATHER_TYPES = {
    "rain":        "sim/weather/rain_percent",          # 0-100% — ecrivable XP12
    "cloud_low":   "sim/weather/cloud_base_msl_m[0]",   # nuages bas (base + top) — ecrivable XP12
    "temperature": "sim/weather/temperature_sealevel_c", # temperature °C — ecrivable XP12
}


@dataclass
class FaultConfig:
    """Configuration d'une faute capteur (generee par TAF)."""
    fault_type: str
    severity: float
    from_pct: float
    to_pct: float


@dataclass
class WeatherConfig:
    """Configuration d'un effet meteo X-Plane (genere par TAF).

    Meme structure que FaultConfig :
      - weather_type : type d'effet (rain, cloud_low, temperature)
      - severity     : degre de severite [0.0, 1.0]
      - from_pct     : debut d'application en % de la trajectoire [0, 100]
      - to_pct       : fin d'application en % de la trajectoire [0, 100]

    La severity est mappee vers des valeurs dataref X-Plane :
      - rain        : severity * 100 → rain_percent (0-100)
      - cloud_low   : severity → base nuage entre 500m et 50m AGL
                       severity=0 → 500m (haut), severity=1 → 50m (ras du sol, brouillard)
                       top = base + 500m (epaisseur fixe)
      - temperature : severity → temperature entre 15°C et -15°C
                       severity=0 → 15°C (doux), severity=1 → -15°C (froid, neige possible)
    """
    weather_type: str
    severity: float
    from_pct: float
    to_pct: float


def validate_faults(faults):
    """
    Valide les contraintes sur les fautes capteur.
    Leve ValueError avec message clair si une contrainte est violee.

    Backup de z3 (qui valide deja les ranges min/max des parametres) :
      - type connu
      - severity in [0, 1]
      - from_pct, to_pct in [0, 100]
      - from_pct < to_pct
    """
    for i, f in enumerate(faults):
        prefix = f"[Fault {i}] ({f.fault_type})"

        if f.fault_type not in KNOWN_FAULT_TYPES:
            raise ValueError(
                f"{prefix} Type inconnu '{f.fault_type}'. "
                f"Types disponibles : {sorted(KNOWN_FAULT_TYPES)}"
            )

        if not 0.0 <= f.severity <= 1.0:
            raise ValueError(
                f"{prefix} severity={f.severity} hors [0.0, 1.0]"
            )

        if not 0.0 <= f.from_pct <= 100.0:
            raise ValueError(
                f"{prefix} from_pct={f.from_pct} hors [0, 100]"
            )

        if not 0.0 <= f.to_pct <= 100.0:
            raise ValueError(
                f"{prefix} to_pct={f.to_pct} hors [0, 100]"
            )

        if f.from_pct >= f.to_pct:
            raise ValueError(
                f"{prefix} from_pct={f.from_pct} >= to_pct={f.to_pct} "
                f"(from_pct doit etre strictement inferieur a to_pct)"
            )


def compute_frame_faults(faults, n_frames):
    """
    Calcule les fautes actives par frame.

    Pourcentage de progression : frame 0 = 0% (debut approche, loin),
    frame N-1 = 100% (fin approche, pres de la piste).

    :param faults: liste de FaultConfig
    :param n_frames: nombre total de frames
    :return: list de n_frames elements, chaque element est une liste
             de (fault_type, severity) actifs pour cette frame
    """
    if not faults or n_frames <= 0:
        return [[] for _ in range(max(n_frames, 0))]

    per_frame = []
    for i in range(n_frames):
        pct = (i / max(n_frames - 1, 1)) * 100.0
        active = []
        for f in faults:
            if f.from_pct <= pct <= f.to_pct:
                active.append((f.fault_type, f.severity))
        per_frame.append(active)

    return per_frame


def save_fault_profile(faults, n_frames, output_path):
    """
    Sauvegarde le profil de fautes dans un fichier JSON.

    Contient :
      - faults : liste des configs de fautes
      - n_frames : nombre de frames
      - per_frame_summary : frames avec fautes actives (compact)
      - n_frames_affected : nombre de frames affectees
    """
    output_path = Path(output_path)

    per_frame = compute_frame_faults(faults, n_frames)

    # Resume compact : seulement les frames avec des fautes
    per_frame_summary = {}
    for i, active in enumerate(per_frame):
        if active:
            per_frame_summary[str(i)] = [
                {"type": ft, "severity": sev} for ft, sev in active
            ]

    profile = {
        "faults": [asdict(f) for f in faults],
        "n_frames": n_frames,
        "per_frame_summary": per_frame_summary,
        "n_frames_affected": len(per_frame_summary),
    }

    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"  .json faults -> {output_path}")
    return str(output_path)


def load_fault_profile(profile_path):
    """
    Charge un profil de fautes depuis un fichier JSON.

    :return: (list de FaultConfig, n_frames)
    """
    with open(profile_path, "r") as f:
        profile = json.load(f)

    faults = [
        FaultConfig(**fc) for fc in profile["faults"]
    ]
    return faults, profile["n_frames"]


def apply_faults_to_directory(input_dir, output_dir, faults, n_frames):
    """
    Applique les fautes capteur aux images d'un dossier.

    Les images sont triees par nom (correspondance frame index).
    Chaque image est degradee selon les fautes actives a son pourcentage
    de progression dans la trajectoire.

    :param input_dir: dossier source (footage/)
    :param output_dir: dossier destination (degraded/)
    :param faults: liste de FaultConfig
    :param n_frames: nombre total de frames attendu
    :return: nombre d'images traitees
    """
    import cv2
    import sys

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Trouver le module camera_sensor_errors
    project_root = Path(__file__).resolve().parent.parent.parent
    cse_dir = str(project_root / "yolo" / "camera_sensor_errors")
    if cse_dir not in sys.path:
        sys.path.insert(0, cse_dir)

    from camera_sensor_errors import apply_errors

    # Lister les images, triees par nom
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    images = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in extensions
    )

    if not images:
        print(f"  [FAULTS] Aucune image dans {input_dir}")
        return 0

    per_frame = compute_frame_faults(faults, len(images))

    count = 0
    count_affected = 0
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        active = per_frame[i] if i < len(per_frame) else []

        if active:
            error_names = [ft for ft, _ in active]
            severities = {ft: sev for ft, sev in active}
            img = apply_errors(img, error_names, severities=severities)
            count_affected += 1

        cv2.imwrite(str(output_dir / img_path.name), img)
        count += 1

    print(f"  [FAULTS] {count} images traitees, {count_affected} degradees -> {output_dir}")
    return count


# ---------------------------------------------------------------------------
# Effets meteo X-Plane 12
# ---------------------------------------------------------------------------

def validate_weather(weather_list):
    """
    Valide les contraintes sur les effets meteo.
    Meme logique que validate_faults mais pour WeatherConfig.
    """
    for i, w in enumerate(weather_list):
        prefix = f"[Weather {i}] ({w.weather_type})"

        if w.weather_type not in KNOWN_WEATHER_TYPES:
            raise ValueError(
                f"{prefix} Type inconnu '{w.weather_type}'. "
                f"Types disponibles : {sorted(KNOWN_WEATHER_TYPES.keys())}"
            )

        if not 0.0 <= w.severity <= 1.0:
            raise ValueError(
                f"{prefix} severity={w.severity} hors [0.0, 1.0]"
            )

        if not 0.0 <= w.from_pct <= 100.0:
            raise ValueError(
                f"{prefix} from_pct={w.from_pct} hors [0, 100]"
            )

        if not 0.0 <= w.to_pct <= 100.0:
            raise ValueError(
                f"{prefix} to_pct={w.to_pct} hors [0, 100]"
            )

        if w.from_pct >= w.to_pct:
            raise ValueError(
                f"{prefix} from_pct={w.from_pct} >= to_pct={w.to_pct} "
                f"(from_pct doit etre strictement inferieur a to_pct)"
            )


def compute_frame_weather(weather_list, n_frames):
    """
    Calcule les effets meteo actifs par frame.

    Meme logique que compute_frame_faults :
    frame 0 = 0% (debut approche, loin), frame N-1 = 100% (pres de la piste).

    :param weather_list: liste de WeatherConfig
    :param n_frames: nombre total de frames
    :return: list de n_frames elements, chaque element est une liste
             de (weather_type, severity) actifs pour cette frame
    """
    if not weather_list or n_frames <= 0:
        return [[] for _ in range(max(n_frames, 0))]

    per_frame = []
    for i in range(n_frames):
        pct = (i / max(n_frames - 1, 1)) * 100.0
        active = []
        for w in weather_list:
            if w.from_pct <= pct <= w.to_pct:
                active.append((w.weather_type, w.severity))
        per_frame.append(active)

    return per_frame


def weather_severity_to_dref(weather_type, severity):
    """
    Convertit la severity normalisee [0, 1] en valeur(s) dataref X-Plane.

    Mapping :
      - rain        : severity * 100 → 0-100%
      - cloud_low   : base = 500 - severity * 450 → 500m (haut) a 50m (ras du sol)
                       Retourne un dict {base: val, top: val}
      - temperature : 15 - severity * 30 → 15°C (doux) a -15°C (froid)

    :return: float ou dict pour cloud_low (2 datarefs)
    """
    if weather_type == "rain":
        return severity * 100.0
    elif weather_type == "cloud_low":
        base = 500.0 - severity * 450.0  # 500m → 50m AGL
        top = base + 500.0               # epaisseur fixe 500m
        return {"base": base, "top": top}
    elif weather_type == "temperature":
        return 15.0 - severity * 30.0    # 15°C → -15°C
    else:
        return severity


def save_weather_profile(weather_list, n_frames, output_path):
    """
    Sauvegarde le profil meteo dans un fichier JSON.

    Contient :
      - weather : liste des configs meteo
      - n_frames : nombre de frames
      - per_frame_summary : frames avec effets actifs (compact)
      - n_frames_affected : nombre de frames affectees
    """
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
    """
    Charge un profil meteo depuis un fichier JSON.

    :return: (list de WeatherConfig, n_frames)
    """
    with open(profile_path, "r") as f:
        profile = json.load(f)

    weather_list = [
        WeatherConfig(**wc) for wc in profile["weather"]
    ]
    return weather_list, profile["n_frames"]
