"""
lard_bridge.py — Interface avec LARD (import uniquement, rien modifie)
======================================================================
Responsabilites :
    1. Obtenir la geometrie piste via compute_aiming_point
    2. Convertir flight_data en .esp via GEODataset.create_scenario
    3. Ecrire les fichiers .esp et .yaml de sortie

Note : create_scenario lit 'data/template.json' relativement au CWD,
       donc on change temporairement vers LARD_ROOT.
"""

import os
import sys
import csv
import json
import yaml
import random
from pathlib import Path
from contextlib import contextmanager
from dataclasses import asdict

# --- Chemins LARD ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # LARD-LAAS-TAF/
LARD_ROOT = PROJECT_ROOT / "LARD"
RUNWAY_DB = str(LARD_ROOT / "data" / "filtered_runways_database_Final.json")

# Ajouter LARD/ au sys.path pour ses imports internes
if str(LARD_ROOT) not in sys.path:
    sys.path.insert(0, str(LARD_ROOT))

from src.geo.geo_dataset import compute_aiming_point, GEODataset
from src.geo.geo_utils import ecef2llh


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

@contextmanager
def _lard_cwd():
    """Change temporairement le CWD vers LARD_ROOT (pour template.json)."""
    prev = os.getcwd()
    os.chdir(LARD_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


def get_runway_geometry(airport, runway, dist_ap_m=300.0):
    """
    Recupere la geometrie d'une piste depuis la DB LARD.

    :return: dict avec ltp_lat, ltp_lon, ltp_alt,
             runway_heading_deg, runway_back_azimuth_deg
    """
    _, _, rwy_psi, ltp, _ = compute_aiming_point(
        RUNWAY_DB, airport, runway, dist_ap_m
    )
    ltp_lat, ltp_lon, ltp_alt = ecef2llh(ltp[0], ltp[1], ltp[2])

    return {
        "ltp_lat": ltp_lat,
        "ltp_lon": ltp_lon,
        "ltp_alt": ltp_alt,
        "runway_heading_deg": rwy_psi[0],      # forward azimuth
        "runway_back_azimuth_deg": rwy_psi[1],  # back azimuth (direction approche)
    }


# ---------------------------------------------------------------------------
# Generation de timestamps pour chaque frame
# ---------------------------------------------------------------------------

def generate_frame_times(n_frames, fps):
    """
    Genere un timestamp par frame (date/heure aleatoire, increment 1/fps).

    LARD/GES attend un dict {year, month, day, hour, minute, second} par frame.
    On choisit une date/heure de base aleatoire puis on incremente.
    """
    base_year = random.randint(2020, 2025)
    base_month = random.randint(1, 12)
    base_day = random.randint(1, 28)
    base_hour = random.randint(6, 18)  # heures de jour
    base_minute = random.randint(0, 59)
    base_second = random.randint(0, 59)

    dt_frame = 1.0 / fps
    times = []
    for i in range(n_frames):
        elapsed = i * dt_frame
        sec_total = base_second + elapsed
        minute_total = base_minute + int(sec_total // 60)
        second = int(sec_total % 60)
        hour_total = base_hour + int(minute_total // 60)
        minute = int(minute_total % 60)
        hour = int(hour_total % 24)

        times.append({
            "year": base_year,
            "month": base_month,
            "day": base_day,
            "hour": hour,
            "minute": minute,
            "second": second,
        })

    return times


# ---------------------------------------------------------------------------
# Export .esp + .yaml
# ---------------------------------------------------------------------------

def export_scenario(flight_data, cfg, ou_params, airport, runway,
                    output_dir, scenario_name="scenario",
                    distances_m=None, ltp_alt=None):
    """
    Exporte un scenario complet : .esp + .yaml + .csv.

    :param flight_data: list de tuples (lon, lat, alt, yaw, pitch, roll)
    :param cfg: TrajectoryConfig (parametres utilisateur)
    :param ou_params: OUParams (hyperparametres)
    :param airport: code ICAO
    :param runway: identifiant piste
    :param output_dir: dossier de sortie
    :param scenario_name: nom de base des fichiers
    :param distances_m: array des distances par frame (pour CSV)
    :param ltp_alt: altitude LTP (pour CSV)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_frames = len(flight_data)
    times = generate_frame_times(n_frames, cfg.fps)

    # --- .esp via LARD ---
    dataset = GEODataset(str(output_path), scenario_name)

    with _lard_cwd():
        scenario, _ = dataset.create_scenario(
            flight_data,
            fov_vertical=30,
            width=3840,
            height=2160,
            nb_frames=n_frames,
            fps=cfg.fps,
            times=times,
        )

    esp_file = output_path / f"{scenario_name}.esp"
    with open(esp_file, "w") as f:
        json.dump(scenario, f, indent=2)

    # --- .yaml metadata ---
    def _to_python(val):
        """Convertit numpy scalaires en types Python natifs pour YAML."""
        if hasattr(val, 'item'):
            return val.item()
        return val

    first = [_to_python(v) for v in flight_data[0]]
    last = [_to_python(v) for v in flight_data[-1]]

    metadata = {
        "airport": airport,
        "runway": runway,
        "n_frames": n_frames,
        "trajectory_config": asdict(cfg),
        "ou_params": asdict(ou_params),
        "flight_data_sample": {
            "first": first,
            "last": last,
        },
    }

    yaml_file = output_path / f"{scenario_name}.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(metadata, f, sort_keys=False, default_flow_style=False)

    # --- .csv flight data ---
    csv_file = output_path / f"{scenario_name}.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "lon", "lat", "alt", "yaw", "pitch", "roll",
            "distance_m", "alt_above_ltp",
        ])
        for i, row in enumerate(flight_data):
            lon, lat, alt, yaw, pitch, roll = [
                v.item() if hasattr(v, 'item') else v for v in row
            ]
            dist = distances_m[i] if distances_m is not None else ""
            alt_above = alt - ltp_alt if ltp_alt is not None else ""
            writer.writerow([i, lon, lat, alt, yaw, pitch, roll, dist, alt_above])

    print(f"  .esp -> {esp_file}")
    print(f"  .yaml -> {yaml_file}")
    print(f"  .csv  -> {csv_file}")

    return str(esp_file), str(yaml_file)
