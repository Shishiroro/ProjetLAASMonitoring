"""
Export.py — Point d'entree TAF (PONT TAF -> LE CODE)
===============================
TAF appelle export(root_node, path) apres chaque generation de test case.
Ce module fait le lien entre les parametres TAF et notre pipeline.

Chaine : TAF root_node → TrajectoryConfig → build_trajectory → export .esp/.yaml
"""

import os
import sys
import math
from pathlib import Path

# Ajouter project/export/ au sys.path pour nos imports
_export_dir = Path(__file__).resolve().parent
if str(_export_dir) not in sys.path:
    sys.path.insert(0, str(_export_dir))

from trajectory_builder import TrajectoryConfig, OUParams, build_trajectory
from lard_bridge import get_runway_geometry, export_scenario
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

    # --- Lire les parametres TAF ---
    fps = int(_read_param(scenario_node, "fps"))
    segment_start_m = float(_read_param(scenario_node, "segment_start_m"))
    segment_end_m = float(_read_param(scenario_node, "segment_end_m"))
    ground_speed_kts = float(_read_param(scenario_node, "ground_speed_kts"))
    turbulence_intensity = float(_read_param(scenario_node, "turbulence_intensity"))
    wind_speed_kts = float(_read_param(scenario_node, "wind_speed_kts"))
    wind_direction_deg = float(_read_param(scenario_node, "wind_direction_deg"))
    stabilization_distance_m = float(_read_param(scenario_node, "stabilization_distance_m"))

    # airport_runway : format "ICAO_RWY" (ex: "LFPO_06")
    airport_runway = str(_read_param(scenario_node, "airport_runway"))
    if "_" not in airport_runway:
        raise ValueError(f"Format airport_runway invalide : '{airport_runway}' (attendu: ICAO_RWY)")
    airport, runway = airport_runway.split("_", 1)

    # --- Lire les fautes capteur (severity > 0 = actif) ---
    faults = []
    for fault_type in sorted(KNOWN_FAULT_TYPES):
        severity = float(_read_param(scenario_node, f"{fault_type}_severity"))
        if severity > 0:
            from_pct = float(_read_param(scenario_node, f"{fault_type}_from_pct"))
            to_pct = float(_read_param(scenario_node, f"{fault_type}_to_pct"))
            faults.append(FaultConfig(fault_type, severity, from_pct, to_pct))

    if faults:
        validate_faults(faults)
        fault_str = ", ".join(f"{f.fault_type}({f.severity:.2f})[{f.from_pct:.0f}-{f.to_pct:.0f}%]"
                              for f in faults)
        print(f"[Export] Fautes capteur : {fault_str}")

    # --- Lire les effets meteo X-Plane (per-scenario) ---
    weather_cfg = WeatherConfig(
        precip_rate=float(_read_param(scenario_node, "precip_rate")),
        cloud_type=float(_read_param(scenario_node, "cloud_type")),
        cloud_coverage=float(_read_param(scenario_node, "cloud_coverage")),
        visibility_m=float(_read_param(scenario_node, "visibility_m")),
        temperature_c=float(_read_param(scenario_node, "temperature_c")),
        time_of_day_h=float(_read_param(scenario_node, "time_of_day_h")),
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
    h_m = segment_start_m * math.tan(math.radians(3.0))
    speed_ms = ground_speed_kts * 0.514444
    correlation_time_s = h_m / speed_ms

    # --- Construire les configs ---
    cfg = TrajectoryConfig(
        fps=fps,
        segment_start_m=segment_start_m,
        segment_end_m=segment_end_m,
        ground_speed_kts=ground_speed_kts,
        correlation_time_s=correlation_time_s,
        turbulence_intensity=turbulence_intensity,
        wind_speed_kts=wind_speed_kts,
        wind_direction_deg=wind_direction_deg,
        stabilization_distance_m=stabilization_distance_m,
    )
    ou = OUParams()

    # --- Renderer (ges ou xplane, via variable d'environnement) ---
    renderer = os.environ.get("LARD_RENDERER", "ges").lower()

    # --- Geometrie piste ---
    rwy = get_runway_geometry(airport, runway, dist_ap_m=ou.dist_ap_m, renderer=renderer)

    # --- Generer la trajectoire ---
    print(f"[Export] {airport}/{runway} | fps={fps} | "
          f"dist=[{segment_start_m:.0f}-{segment_end_m:.0f}m] | "
          f"wind={wind_speed_kts:.0f}kts@{wind_direction_deg:.0f}deg")

    flight_data, _, _ = build_trajectory(
        cfg, ou,
        ltp_lat=rwy["ltp_lat"],
        ltp_lon=rwy["ltp_lon"],
        ltp_alt=rwy["ltp_alt"],
        runway_heading_deg=rwy["runway_heading_deg"],
        runway_back_azimuth_deg=rwy["runway_back_azimuth_deg"],
    )

    # --- Renderer (ges ou xplane, via variable d'environnement) ---
    renderer = os.environ.get("LARD_RENDERER", "ges").lower()

    # --- Exporter .esp + .yaml + poses_cam_export.json ---
    weather_arg = weather_cfg if has_weather(weather_cfg) else None
    export_scenario(
        flight_data, cfg, ou,
        airport, runway,
        output_dir=path,
        scenario_name=f"{airport}_{runway}",
        faults=faults,
        weather=weather_arg,
        renderer=renderer,
        ltp_alt=rwy["ltp_alt"],
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
