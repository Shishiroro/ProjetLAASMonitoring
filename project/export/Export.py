"""
Export.py — Point d'entree TAF + Production images & GT
=========================================================
Deux fonctions independantes hebergees dans le meme fichier :

  1. export(root_node, path)
     Callback TAF appele pendant la generation z3. Ecrit .yaml + JSON
     configs dans output/test_artifact_X (temporaire).

  2. generate_images_and_GT(run_dir, xplane_dir, runway=None)
     Orchestrateur post-TAF appele par run_pipeline.py sur chaque run_dir.
     Enchaine rendu X-Plane -> fautes capteur -> GT LARD.
     TAF n'invoque jamais cette fonction.

Chaine TAF : root_node -> TrajectoryConfig -> build_trajectory -> .yaml
Chaine pipeline : run_dir -> render_run -> apply_faults -> generate_gt
"""

import sys
import math
from pathlib import Path

# Ajouter project/export/ au sys.path pour nos imports
_export_dir = Path(__file__).resolve().parent
if str(_export_dir) not in sys.path:
    sys.path.insert(0, str(_export_dir))

from trajectory_builder import TrajectoryConfig, OUParams, build_trajectory
from lard_bridge import get_runway_geometry, export_scenario
from xplane_bridge import save_poses_json
from sensor_faults import (
    FaultConfig, KNOWN_FAULT_TYPES, validate_faults, save_fault_profile,
)
from xplane_weather import (
    WeatherConfig, validate_weather, has_weather, save_weather_profile,
)

 
def _read_param(node, name):
    """Lit un parametre TAF depuis un node (instance 0)."""
    return node.get_parameter_n(name).values[0]


def export(root_node, path):
    """
    Point d'entree TAF. Appele pour chaque test_artifact genere.

    :param root_node: Node racine de l'arbre TAF (apres generation z3)
    :param path: chemin du dossier test_artifact courant
    """
    scenario_node = root_node.get_child_n("scenario")
    trajectory_node = scenario_node.get_child_n("trajectory")
    weather_node = scenario_node.get_child_n("weather")
    faults_node = scenario_node.get_child_n("faults")

    # --- Trajectoire ---
    fps = int(_read_param(trajectory_node, "fps"))
    along_track_distance_start = float(_read_param(trajectory_node, "along_track_distance_start"))
    along_track_distance_end = float(_read_param(trajectory_node, "along_track_distance_end"))
    ground_speed_kts = float(_read_param(trajectory_node, "ground_speed_kts"))
    turbulence_intensity = float(_read_param(trajectory_node, "turbulence_intensity"))
    wind_speed_kts = float(_read_param(trajectory_node, "wind_speed_kts"))
    wind_direction_deg = float(_read_param(trajectory_node, "wind_direction_deg"))
    stabilization_distance_m = float(_read_param(trajectory_node, "stabilization_distance_m"))

    # airport_runway : format "ICAO_RWY" (ex: "LFPO_06")
    airport_runway = str(_read_param(trajectory_node, "airport_runway"))
    if "_" not in airport_runway:
        raise ValueError(f"Format airport_runway invalide : '{airport_runway}' (attendu: ICAO_RWY)")
    airport, runway = airport_runway.split("_", 1)

    # --- Fautes capteur (severity > 0 = actif) ---
    # XML : start_pct + duration_pct (configurables par faute)
    # Interne : FaultConfig garde from_pct/to_pct ; to_pct = start + duration
    faults = []
    for fault_type in sorted(KNOWN_FAULT_TYPES):
        fault_node = faults_node.get_child_n(fault_type)
        severity = float(_read_param(fault_node, "severity"))
        if severity > 0:
            start_pct = float(_read_param(fault_node, "start_pct"))
            duration_pct = float(_read_param(fault_node, "duration_pct"))
            from_pct = start_pct
            to_pct = min(start_pct + duration_pct, 100.0)
            faults.append(FaultConfig(fault_type, severity, from_pct, to_pct))

    if faults:
        validate_faults(faults)
        fault_str = ", ".join(f"{f.fault_type}({f.severity:.2f})[{f.from_pct:.0f}-{f.to_pct:.0f}%]"
                              for f in faults)
        print(f"[Export] Fautes capteur : {fault_str}")

    # --- Meteo X-Plane (per-scenario) ---
    weather_cfg = WeatherConfig(
        precip_rate=float(_read_param(weather_node, "precip_rate")),
        cloud_type=float(_read_param(weather_node, "cloud_type")),
        cloud_coverage=float(_read_param(weather_node, "cloud_coverage")),
        cloud_margin_m=float(_read_param(weather_node, "cloud_margin_m")),
        cloud_thickness_m=float(_read_param(weather_node, "cloud_thickness_m")),
        visibility_m=float(_read_param(weather_node, "visibility_m")),
        temperature_c=float(_read_param(weather_node, "temperature_c")),
        time_of_day_h=float(_read_param(weather_node, "time_of_day_h")),
        rain_scale=float(_read_param(weather_node, "rain_scale")),
        settle_s=float(_read_param(weather_node, "xplane_weather_settle_s")),
    )

    if has_weather(weather_cfg):
        validate_weather(weather_cfg)
        parts = []
        if weather_cfg.precip_rate > 0:
            parts.append(f"precip={weather_cfg.precip_rate:.2f}")
        if weather_cfg.cloud_type >= 0:
            parts.append(f"cloud_type={weather_cfg.cloud_type:.0f} cov={weather_cfg.cloud_coverage:.1f}")
        if weather_cfg.visibility_m < 50000:
            parts.append(f"vis={weather_cfg.visibility_m:.0f}m")
        if weather_cfg.temperature_c < 0:
            parts.append(f"temp={weather_cfg.temperature_c:.0f}C")
        if weather_cfg.time_of_day_h != 12.0:
            parts.append(f"heure={weather_cfg.time_of_day_h:.1f}h")
        print(f"[Export] Meteo X-Plane : {', '.join(parts)}")

    # --- Calcul auto de tau  ---
    # A nos altitudes (~300m max), , donc tau ≈ h / V
    h_m = along_track_distance_start * math.tan(math.radians(3.0))
    speed_ms = ground_speed_kts * 0.514444
    correlation_time_s = h_m / speed_ms

    # --- Construire les configs ---
    cfg = TrajectoryConfig(
        fps=fps,
        along_track_distance_start=along_track_distance_start,
        along_track_distance_end=along_track_distance_end,
        ground_speed_kts=ground_speed_kts,
        correlation_time_s=correlation_time_s,
        turbulence_intensity=turbulence_intensity,
        wind_speed_kts=wind_speed_kts,
        wind_direction_deg=wind_direction_deg,
        stabilization_distance_m=stabilization_distance_m,
    )
    ou = OUParams()

    # --- Geometrie piste ---
    rwy = get_runway_geometry(airport, runway, dist_ap_m=ou.dist_ap_m)

    # --- Generer la trajectoire ---
    print(f"[Export] {airport}/{runway} | fps={fps} | "
          f"dist=[{along_track_distance_start:.0f}-{along_track_distance_end:.0f}m] | "
          f"wind={wind_speed_kts:.0f}kts@{wind_direction_deg:.0f}deg")

    flight_data, _, _ = build_trajectory(
        cfg, ou,
        ltp_lat=rwy["ltp_lat"],
        ltp_lon=rwy["ltp_lon"],
        ltp_alt=rwy["ltp_alt"],
        runway_heading_deg=rwy["runway_heading_deg"],
        runway_back_azimuth_deg=rwy["runway_back_azimuth_deg"],
    )

    # --- Exporter .yaml (format LARD) ---
    weather_arg = weather_cfg if has_weather(weather_cfg) else None
    scenario_name = f"{airport}_{runway}"
    export_scenario(
        flight_data, cfg, ou,
        airport, runway,
        output_dir=path,
        scenario_name=scenario_name,
        faults=faults,
        weather=weather_arg,
    )

    # --- Sauver poses_cam_export.json (format X-Plane, coords locales) ---
    save_poses_json(
        flight_data, cfg.fps, scenario_name,
        output_path=Path(path) / "poses_cam_export.json",
        ltp_alt=rwy["ltp_alt"],
        trajectory_config={
            "along_track_distance_start": along_track_distance_start,
            "along_track_distance_end": along_track_distance_end,
            "ground_speed_kts": ground_speed_kts,
            "turbulence_intensity": turbulence_intensity,
            "wind_speed_kts": wind_speed_kts,
            "wind_direction_deg": wind_direction_deg,
            "stabilization_distance_m": stabilization_distance_m,
            "airport_runway": f"{airport}_{runway}",
        },
    )

    # --- Sauver le profil de fautes (si actif) ---
    if faults:
        save_fault_profile(
            faults,
            n_frames=len(flight_data),
            output_path=Path(path) / "fault_profile.json",
        )

    # --- Sauver le profil meteo (si actif) ---
    if has_weather(weather_cfg):
        save_weather_profile(
            weather_cfg,
            output_path=Path(path) / "weather_profile.json",
        )


# ===========================================================================
# Phase 2 : Production des images + GT LARD (post-TAF, par run_dir)
# ---------------------------------------------------------------------------
# Les fonctions ci-dessous sont appelees par run_pipeline.py apres que TAF
# a fini sa generation et que les runs ont ete deplaces dans runs/<name>/.
# TAF n'utilise QUE export() ci-dessus, jamais ce qui suit.
# ===========================================================================

_IMAGE_EXTS = (".jpeg", ".jpg", ".png")


def _has_images(run_dir):
    """Retourne True si run_dir/footage/ contient au moins une image."""
    footage = Path(run_dir) / "footage"
    if not footage.exists():
        return False
    return any(f.suffix.lower() in _IMAGE_EXTS
               for f in footage.iterdir() if f.is_file())


def step_render(run_dir, xplane_dir):
    """Rendu X-Plane 12 (meteo injectee a l'interieur si profil present).

    :return: True si images presentes apres rendu, False sinon
    """
    from xplane_bridge import render_run

    run_dir = Path(run_dir)

    if _has_images(run_dir):
        # render_run gere lui-meme le skip, on log juste pour la trace
        pass

    if not render_run(run_dir, xplane_dir or ""):
        print(f"  [Image] Echec rendu X-Plane pour {run_dir.name}")
        return False

    if not _has_images(run_dir):
        print(f"  [Image] Pas d'images dans footage/ pour {run_dir.name}")
        return False

    return True


def step_faults(run_dir):
    """Applique les fautes capteur si fault_profile.json present (no-op sinon).

    Les exceptions sont logees mais ne bloquent pas le pipeline : la presence
    de fautes est optionnelle.
    """
    from sensor_faults import apply_faults

    try:
        apply_faults(run_dir)
    except Exception as e:
        print(f"  [Image] FAULTS ERREUR : {e}")


def step_ground_truth(run_dir, runway=None):
    """Genere le CSV GT LARD + les images annotees (annotated_lard/).

    :return: True si CSV present apres l'etape (genere ou existant), False sinon
    """
    from lard_bridge import generate_gt, annotate_gt

    run_dir = Path(run_dir)

    try:
        generate_gt(run_dir)
    except Exception as e:
        print(f"  [Image] GT ERREUR : {e}")
        if not list(run_dir.glob("*_labels.csv")):
            return False
        print(f"  [Image] Utilisation du CSV existant")

    try:
        annotate_gt(run_dir, runway=runway)
    except Exception as e:
        print(f"  [Image] GT-VIS ERREUR : {e}")

    return True


def generate_images_and_GT(run_dir, xplane_dir, runway=None):
    """Production des images + GT pour un run.

    Enchaine sur run_dir (deja existant dans runs/, avec ses configs JSON) :
      1. step_render          -> footage/*.jpeg (et meteo si profil)
      2. step_faults          -> degraded/*.jpeg (si fault_profile.json)
      3. step_ground_truth    -> <name>_labels.csv + annotated_lard/

    :param run_dir: dossier du run dans runs/
    :param xplane_dir: chemin vers X-Plane 12
    :param runway: filtre piste pour annotate_gt (defaut: extrait du nom)
    :return: True si tout est passe, False sinon
    """
    run_dir = Path(run_dir)
    print(f"\n  [Image] Production images + GT pour {run_dir.name}")

    if not step_render(run_dir, xplane_dir):
        return False

    step_faults(run_dir)

    if not step_ground_truth(run_dir, runway=runway):
        return False

    return True
