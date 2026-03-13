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

#Fais le lien entre les parametres TAF et notre pipeline d'export vers LARD/GES.

import os
import sys
import json
import yaml
import uuid
import random
import numpy as np
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
from src.labeling.label_export import export_labels
from src.labeling.export_config import DatasetTypes


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
    #  interroge la base de données de pistes LARD (filtered_runways_database_Final.json), 
    # retourne lat/lon/alt du LTP + cap piste (heading et back azimuth) pour l'aéroport/piste donnés.
    _, _, rwy_psi, ltp, _ = compute_aiming_point(
        RUNWAY_DB, airport, runway, dist_ap_m
    )
    ltp_lat, ltp_lon, ltp_alt = ecef2llh(ltp[0], ltp[1], ltp[2])

    return {
        "ltp_lat": ltp_lat,
        "ltp_lon": ltp_lon,
        "ltp_alt": ltp_alt,
        "runway_heading_deg": rwy_psi[0],      # azimut LTP→FPAP (cap camera)
        "runway_back_azimuth_deg": rwy_psi[1],  # azimut FPAP→LTP (positionnement)
    }


# ---------------------------------------------------------------------------
# Generation de timestamps pour chaque frame (au format attendu par LARD/GES)
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
    base_hour = random.randint(9, 16)  # heures de plein jour (evite crepuscule)
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
                    output_dir, scenario_name="scenario"):
    """
    Exporte un scenario complet : .esp + .yaml.
    Le .yaml est fidele au format LARD (poses, image, trajectory)
    pour etre compatible avec export_labels() de LARD.

    :param flight_data: list de tuples (lon, lat, alt, yaw, pitch, roll)
    :param cfg: TrajectoryConfig (parametres utilisateur)
    :param ou_params: OUParams (hyperparametres)
    :param airport: code ICAO
    :param runway: identifiant piste
    :param output_dir: dossier de sortie
    :param scenario_name: nom de base des fichiers
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_frames = len(flight_data)
    times = generate_frame_times(n_frames, cfg.fps)

    # --- Params image ---
    img_width = 512
    img_height = 512
    fov_x = 30.0
    watermark_height = 0
    # fov_y calcule comme dans LARD (write_scenario.py)
    f_focal = img_height / 2.0 / np.tan(np.deg2rad(fov_x / 2.0))
    fov_y = round(float(2 * np.rad2deg(np.arctan2(img_width / 2.0, f_focal))), 6)

    # --- .esp via LARD ---
    dataset = GEODataset(str(output_path), scenario_name)

    with _lard_cwd():
        scenario, _ = dataset.create_scenario(
            flight_data,
            fov_vertical=fov_x,
            width=img_width,
            height=img_height,
            nb_frames=n_frames,
            fps=cfg.fps,
            times=times,
        )

    esp_file = output_path / f"{scenario_name}.esp"
    with open(esp_file, "w") as f:
        json.dump(scenario, f, indent=2)

    # --- .yaml au format LARD (compatible export_labels) ---
    def _to_python(val):
        """Convertit numpy scalaires en types Python natifs pour YAML."""
        if hasattr(val, 'item'):
            return val.item()
        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.floating):
            return float(val)
        return val

    # Construire la liste poses au format LARD exact :
    # {uuid, airport, runway, pose: [lon, lat, alt, yaw, pitch, roll], time: {...}}
    poses = []
    for i, fd in enumerate(flight_data):
        poses.append({
            'uuid': str(uuid.uuid4()),
            'airport': airport,
            'runway': runway,
            'pose': [_to_python(v) for v in fd],
            'time': times[i],
        })

    # Structure yaml LARD (asdict(ScenarioContent) dans write_scenario.py)
    yaml_content = {
        'airports_runways': {airport: [runway]},
        'image': {
            'height': img_height,
            'width': img_width,
            'fov_x': fov_x,
            'fov_y': fov_y,
            'watermark_height': watermark_height,
        },
        'poses': poses,
        'runways_database': 'data/filtered_runways_database_Final.json',
        'trajectory': {
            'sample_number': n_frames,
            'dist_ap_m': ou_params.dist_ap_m,
            'max_distance_m': cfg.segment_start_m,
            'min_distance_m': cfg.segment_end_m,
            'alpha_h_deg': ou_params.alpha_h_deg,
            'std_alpha_h_deg': ou_params.std_alpha_h_deg,
            'alpha_h_distrib': 'normal',
            'alpha_v_deg': ou_params.alpha_v_deg,
            'std_alpha_v_deg': ou_params.std_alpha_v_deg,
            'alpha_v_distrib': 'normal',
            'yaw_deg': ou_params.yaw_deg,
            'std_yaw_deg': ou_params.std_yaw_deg,
            'yaw_distrib': 'normal',
            'pitch_deg': ou_params.pitch_deg,
            'std_pitch_deg': ou_params.std_pitch_deg,
            'pitch_distrib': 'normal',
            'roll_deg': ou_params.roll_deg,
            'std_roll_deg': ou_params.std_roll_deg,
            'roll_distrib': 'normal',
            'use_ODD': False,
        },
        'time': {
            'second_min': 0, 'second_max': 59,
            'minute_min': 0, 'minute_max': 59,
            'hour_min': 9, 'hour_max': 16,
            'day_min': 1, 'day_max': 28,
            'month_min': 1, 'month_max': 12,
            'year_min': 2020, 'year_max': 2025,
        },
    }

    yaml_file = output_path / f"{scenario_name}.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)

    print(f"  .esp -> {esp_file}")
    print(f"  .yaml -> {yaml_file}")

    return str(esp_file), str(yaml_file)


def generate_labels_csv(yaml_path, dataset_dir, csv_name=None):
    """
    Genere le .csv ground truth LARD a partir du .yaml et des images dans footage/.
    A appeler APRES avoir recupere les images de Google Earth Studio.

    :param yaml_path: chemin vers le .yaml du scenario
    :param dataset_dir: dossier contenant footage/ avec les images GES
    :param csv_name: nom du fichier CSV (defaut: <yaml_stem>_labels.csv)
    """
    yaml_path = Path(yaml_path).resolve()
    dataset_dir = Path(dataset_dir).resolve()

    if csv_name is None:
        csv_name = f"{yaml_path.stem}_labels.csv"
    csv_file = dataset_dir / csv_name

    with _lard_cwd():
        export_labels(
            dataset_type=DatasetTypes.EARTH_STUDIO,
            yaml_scenario_path=yaml_path,
            export_dir=dataset_dir,
            out_labels_file=csv_file,
        )

    print(f"  .csv GT -> {csv_file}")
    return str(csv_file)
