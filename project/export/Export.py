"""
Export.py — Point d'entree TAF (PONT TAF -> LE CODE)
===============================
TAF appelle export(root_node, path) apres chaque generation de test case.
Ce module fait le lien entre les parametres TAF et notre pipeline.

Chaine : TAF root_node → TrajectoryConfig → build_trajectory → export .esp/.yaml
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

    # --- Geometrie piste ---
    rwy = get_runway_geometry(airport, runway, dist_ap_m=ou.dist_ap_m)

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

    # --- Exporter .esp + .yaml ---
    export_scenario(
        flight_data, cfg, ou,
        airport, runway,
        output_dir=path,
        scenario_name=f"{airport}_{runway}",
    )
