"""
gt_xplane.py — Ground truth bounding boxes pour X-Plane (sans LARD)
====================================================================
Projection ENU pinhole des 4 coins piste sur l'image, a partir de :
  - poses_actual.json (poses reelles lues depuis X-Plane)
  - render_config.json (resolution + FOV)
  - apt.dat (coordonnees piste X-Plane)
  - terrain_elevation.json (altitude terrain reelle)

Genere un CSV au format LARD (compatible evaluate.py).
"""

import math
import json
import csv
import uuid
from pathlib import Path

import pyproj


# ---------------------------------------------------------------------------
# Projection ENU pinhole (non-carree, FOV H/V separees)
# ---------------------------------------------------------------------------

def project_point(corner_latlon, cam_lat, cam_lon, cam_alt,
                  heading, pitch_ges, roll, fov_h, fov_v, width, height,
                  pp_offset_x=0.0, pp_offset_y=0.0):
    """Projette un point 3D (lat, lon, alt) en pixel (x, y).

    :param corner_latlon: (lat, lon, alt) du point a projeter
    :param cam_lat/lon/alt: position camera
    :param heading: cap compas (deg, CW depuis nord)
    :param pitch_ges: pitch convention GES (90 = level)
    :param roll: roulis (deg, positif = aile droite basse)
    :param fov_h: FOV horizontal (deg)
    :param fov_v: FOV vertical (deg)
    :param width: largeur image (px)
    :param height: hauteur image (px)
    :return: (px_x, px_y) ou None si derriere la camera
    """
    geod = pyproj.Geod(ellps="WGS84")
    c_lat, c_lon, c_alt = corner_latlon

    # ENU : distance + azimut geodesique
    azimuth, _, distance = geod.inv(cam_lon, cam_lat, c_lon, c_lat)
    az_rad = math.radians(azimuth)
    de = distance * math.sin(az_rad)   # est
    dn = distance * math.cos(az_rad)   # nord
    du = c_alt - cam_alt               # up

    # Rotation camera : ENU → (forward, right, down)
    pitch_xp = pitch_ges - 90.0
    yaw_rad = math.radians(heading)
    pitch_rad = math.radians(pitch_xp)
    roll_rad = math.radians(roll)

    # Yaw (rotation autour de Up)
    fwd = dn * math.cos(yaw_rad) + de * math.sin(yaw_rad)
    right = -dn * math.sin(yaw_rad) + de * math.cos(yaw_rad)
    down = -du

    # Pitch (rotation autour de Right)
    fwd2 = fwd * math.cos(pitch_rad) - down * math.sin(pitch_rad)
    down2 = fwd * math.sin(pitch_rad) + down * math.cos(pitch_rad)

    # Roll (rotation autour de Forward)
    right3 = right * math.cos(roll_rad) + down2 * math.sin(roll_rad)
    down3 = -right * math.sin(roll_rad) + down2 * math.cos(roll_rad)

    if fwd2 <= 0:
        return None

    # Focale unique depuis le FOV horizontal (pixels carres)
    # Le FOV vertical se deduit de l'aspect ratio, pas d'un reglage independant
    f = (width / 2.0) / math.tan(math.radians(fov_h / 2.0))

    px_x = width / 2.0 + pp_offset_x + (right3 / fwd2) * f
    px_y = height / 2.0 + pp_offset_y + (down3 / fwd2) * f

    return (px_x, px_y)


# ---------------------------------------------------------------------------
# Generation CSV GT
# ---------------------------------------------------------------------------

# Colonnes CSV identiques a LARD (compatibles evaluate.py)
CSV_COLUMNS = [
    "uuid", "image", "height", "width", "type", "original_dataset",
    "scenario", "airport", "runway", "time_to_landing", "weather", "night",
    "time", "lat_cam", "lon_cam", "alt_cam", "yaw", "pitch", "roll",
    "slant_distance", "along_track_distance", "height_above_runway",
    "lateral_path_angle", "vertical_path_angle", "watermark_height",
    "runway_in_cone",
    "x_TR", "y_TR", "x_TL", "y_TL", "x_BL", "y_BL", "x_BR", "y_BR",
]


def generate_gt_csv(run_dir, airport, runway):
    """Genere le CSV ground truth pour un run X-Plane.

    Lit automatiquement :
      - poses_actual.json (poses reelles)
      - render_config.json (resolution + FOV)
      - terrain_elevation.json (altitude piste)
      - runways_db_V2_XPlane_aptdat.json (coins piste)

    :param run_dir: Path du dossier du run (runs/KPDX_28L/)
    :param airport: code ICAO (ex: "KPDX")
    :param runway: nom piste (ex: "28L")
    :return: Path du fichier CSV genere
    """
    run_dir = Path(run_dir)
    project_root = Path(__file__).resolve().parent.parent.parent

    # --- Charger les donnees ---
    poses_path = run_dir / "poses_actual.json"
    if not poses_path.exists():
        raise FileNotFoundError(f"poses_actual.json introuvable dans {run_dir}")
    poses = json.load(open(poses_path))

    render_cfg_path = run_dir / "render_config.json"
    if not render_cfg_path.exists():
        raise FileNotFoundError(f"render_config.json introuvable dans {run_dir}")
    render_cfg = json.load(open(render_cfg_path))
    img_w = render_cfg["width"]
    img_h = render_cfg["height"]
    fov_h = render_cfg["fov_h"]
    fov_v = render_cfg["fov_v"]
    # Pilot eye offset (body frame: x=lateral, y=up, z=forward)
    pe_x = render_cfg.get("pilot_eye_x", 0.0)
    pe_y = render_cfg.get("pilot_eye_y", 0.0)
    pe_z = render_cfg.get("pilot_eye_z", 0.0)
    # Principal point offset (pixels) — laisser a 0, le pilot eye gere le shift
    pp_x = render_cfg.get("pp_offset_x", 0.0)
    pp_y = render_cfg.get("pp_offset_y", 0.0)

    # Coins piste depuis apt.dat
    db_path = project_root / "project" / "data" / "runways_db_V2_XPlane_aptdat.json"
    if not db_path.exists():
        raise FileNotFoundError(f"DB apt.dat introuvable : {db_path}")
    runway_db = json.load(open(db_path))

    if airport not in runway_db or runway not in runway_db[airport]:
        raise ValueError(f"Piste {airport}/{runway} introuvable dans la DB apt.dat")

    rwy = runway_db[airport][runway]

    # Altitude terrain reelle (corrige l'elevation apt.dat)
    terrain_path = run_dir / "terrain_elevation.json"
    rwy_alt = 0.0
    if terrain_path.exists():
        terrain = json.load(open(terrain_path))
        rwy_alt = terrain["elevation_m"]
    else:
        # Fallback : altitude de la DB
        rwy_alt = rwy["A"]["coordinate"].get("altitude", 0.0)

    # 4 coins : A=TR(far right), B=TL(far left), C=BL(near left), D=BR(near right)
    corners_geo = {}
    for label in ["A", "B", "C", "D"]:
        c = rwy[label]["coordinate"]
        corners_geo[label] = (c["latitude"], c["longitude"], rwy_alt)

    # --- Nommage images ---
    scenario_name = run_dir.name
    n_frames = len(poses)
    img_digits = len(str(n_frames - 1))

    # --- Projection + ecriture CSV ---
    csv_path = run_dir / f"{scenario_name}_labels.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, delimiter=";")
        writer.writeheader()

        for i, pose in enumerate(poses):
            heading = pose["heading"]
            pitch_ges = pose["pitch_ges"]
            roll = pose["roll"]

            # Position camera = position avion + pilot eye offset (body→ENU)
            # Body frame: z=forward(nose), x=right, y=up
            # ENU: east, north, up
            h_rad = math.radians(heading)
            # Projection body → ENU (horizontal plane)
            eye_east = pe_z * math.sin(h_rad) + pe_x * math.cos(h_rad)
            eye_north = pe_z * math.cos(h_rad) - pe_x * math.sin(h_rad)
            eye_up = pe_y

            m_per_deg_lat = 111320.0
            m_per_deg_lon = 111320.0 * math.cos(math.radians(pose["lat"]))
            cam_lat = pose["lat"] + eye_north / m_per_deg_lat
            cam_lon = pose["lon"] + eye_east / m_per_deg_lon
            cam_alt = pose["alt_m"] + eye_up

            img_name = f"{scenario_name}_{str(i).zfill(img_digits)}.jpg"

            # Projeter les 4 coins
            corners_px = {}
            for label in ["A", "B", "C", "D"]:
                pt = project_point(
                    corners_geo[label],
                    cam_lat, cam_lon, cam_alt,
                    heading, pitch_ges, roll,
                    fov_h, fov_v, img_w, img_h,
                    pp_offset_x=pp_x, pp_offset_y=pp_y,
                )
                corners_px[label] = pt

            # Mapping coins → colonnes CSV LARD
            # A=TR, B=TL, C=BL, D=BR
            def _px(label, axis):
                pt = corners_px[label]
                if pt is None:
                    return -1
                return pt[0] if axis == "x" else pt[1]

            # Distance au seuil (C/D midpoint)
            geod = pyproj.Geod(ellps="WGS84")
            threshold_lat = (corners_geo["C"][0] + corners_geo["D"][0]) / 2
            threshold_lon = (corners_geo["C"][1] + corners_geo["D"][1]) / 2
            _, _, slant = geod.inv(cam_lon, cam_lat, threshold_lon, threshold_lat)
            h_above = cam_alt - rwy_alt

            row = {
                "uuid": str(uuid.uuid4()),
                "image": img_name,
                "height": img_h,
                "width": img_w,
                "type": "xplane",
                "original_dataset": "",
                "scenario": scenario_name,
                "airport": airport,
                "runway": runway,
                "time_to_landing": "",
                "weather": "",
                "night": "",
                "time": "",
                "lat_cam": cam_lat,
                "lon_cam": cam_lon,
                "alt_cam": cam_alt,
                "yaw": heading,
                "pitch": pitch_ges,
                "roll": roll,
                "slant_distance": round(slant, 1),
                "along_track_distance": round(slant, 1),
                "height_above_runway": round(h_above, 1),
                "lateral_path_angle": "",
                "vertical_path_angle": "",
                "watermark_height": 0,
                "runway_in_cone": 1,
                "x_TR": round(_px("A", "x"), 1),
                "y_TR": round(_px("A", "y"), 1),
                "x_TL": round(_px("B", "x"), 1),
                "y_TL": round(_px("B", "y"), 1),
                "x_BL": round(_px("C", "x"), 1),
                "y_BL": round(_px("C", "y"), 1),
                "x_BR": round(_px("D", "x"), 1),
                "y_BR": round(_px("D", "y"), 1),
            }
            writer.writerow(row)

    print(f"  [GT-XPLANE] {n_frames} frames -> {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GT X-Plane (ENU pinhole)")
    parser.add_argument("--run", required=True, help="Nom du run (ex: KPDX_28L)")
    parser.add_argument("--airport", required=True, help="Code ICAO")
    parser.add_argument("--runway", required=True, help="Nom piste")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    run_dir = project_root / "runs" / args.run

    csv_path = generate_gt_csv(run_dir, args.airport, args.runway)
    print(f"Done: {csv_path}")
