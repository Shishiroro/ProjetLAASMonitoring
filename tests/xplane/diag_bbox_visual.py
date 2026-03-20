"""
diag_bbox_visual.py — Diagnostic visuel des bbox GT LARD sur images X-Plane
===========================================================================
Reproduit la chaine computeLabels() de LARD pas a pas, affiche les valeurs
intermediaires, et dessine le trapeze GT sur les images pour verification.

Usage :
    python tests/xplane/diag_bbox_visual.py --run KPDX_28L
    python tests/xplane/diag_bbox_visual.py --run KPDX_28L --frames 0 5 10
    python tests/xplane/diag_bbox_visual.py --run KPDX_28L --all
"""

import sys
import json
import yaml
import math
import argparse
import numpy as np
import pyproj
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LARD_ROOT = PROJECT_ROOT / "LARD"

sys.path.insert(0, str(LARD_ROOT))
from src.labeling.export_config import DatasetTypes, database_name, NEW_CORNERS_NAMES


# ===========================================================================
# Reproduction EXACTE des fonctions LARD (label_export.py) — pour comparaison
# ===========================================================================

def geodetic_to_cartesian(lat, lon, alt, ref_lat, ref_lon, ref_alt, runw_az):
    """Identique a label_export.geodetic_to_cartesian."""
    geod = pyproj.Geod(ellps="WGS84")
    azimuth, _, distance = geod.inv(ref_lon, ref_lat, lon, lat)
    x = distance * np.cos(np.radians(azimuth - runw_az))
    y = -distance * np.sin(np.radians(azimuth - runw_az))
    z = alt - ref_alt
    return np.array([x, y, z])


def get_forward_azimuth(P1, P2):
    """Identique a label_export.get_forward_azimuth."""
    geod = pyproj.Geod(ellps="WGS84")
    azimuth, _, _ = geod.inv(P1[1], P1[0], P2[1], P2[0])
    return azimuth


def rotation_matrix(axis, angle):
    """Identique a label_export.rotation_matrix."""
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    Q = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    I = np.eye(3)
    P = np.array([[x*x, x*y, x*z], [y*x, y*y, y*z], [z*x, z*y, z*z]])
    return P + cos_theta * (I - P) + sin_theta * Q


def apply_rotations_int(yaw, pitch, roll):
    """Identique a label_export.apply_rotations_int (angles en radians)."""
    X_init = np.array([1, 0, 0])
    Y_init = np.array([0, 1, 0])
    Z_init = np.array([0, 0, 1])

    Ryaw = rotation_matrix(Z_init, yaw)
    Xprim = Ryaw @ X_init
    Yprim = Ryaw @ Y_init

    Rpitch = rotation_matrix(Yprim, pitch)
    Xter = Rpitch @ Xprim
    Zter = Rpitch @ Z_init

    Rroll = rotation_matrix(Xter, roll)
    Yquart = Rroll @ Yprim
    Zquart = Rroll @ Zter

    return Xter, Yquart, Zquart


def point_in_new_repere(point, XNew, YNew, ZNew):
    """Identique a label_export.point_in_new_repere."""
    return np.array([np.dot(point, XNew), np.dot(point, YNew), np.dot(point, ZNew)])


def pointcam_to_pix(P, fov_h, fov_v, img_h, img_v):
    """Identique a label_export.pointcam_to_pix. Retourne (x_pixel, y_pixel) ou None."""
    if P[0] <= 0:
        return None, None

    fov_h_rad = np.radians(fov_h)
    fov_v_rad = np.radians(fov_v)

    world_width = 2 * P[0] * np.tan(fov_h_rad * 0.5)
    world_height = 2 * P[0] * np.tan(fov_v_rad * 0.5)

    x_norm = (1 - ((P[1] / world_width) + 0.5))
    y_norm = (1 - ((P[2] / world_height) + 0.5))

    x_pixel = int(x_norm * img_h)
    y_pixel = int(y_norm * img_v)

    return x_pixel, y_pixel


# ===========================================================================
# Diagnostic principal
# ===========================================================================

def diagnose_frame(pose, runway_db, airport, target_runway,
                   fov_x, fov_y, img_width, img_height, verbose=True):
    """
    Reproduit computeLabels pas a pas pour une seule pose et une piste.
    Retourne les coins en pixels + les valeurs intermediaires.
    """
    lon, lat, alt, yaw_cam, pitch_cam, roll_cam = pose

    runway_points = runway_db[airport][target_runway]
    # Meme mapping que label_export.convert_label
    pts = [
        list(runway_points['A']['coordinate'].values()),  # TR : [lat, lon, alt]
        list(runway_points['B']['coordinate'].values()),  # TL
        list(runway_points['C']['coordinate'].values()),  # BL
        list(runway_points['D']['coordinate'].values()),  # BR
    ]
    A, B, C, D = [np.array(p) for p in pts]

    # Azimut piste oppose (LARD calcule depuis milieu(D,C) vers milieu(A,B))
    opp_runway_azimuth = get_forward_azimuth((D + C) / 2, (A + B) / 2)
    if opp_runway_azimuth < 0:
        opp_runway_azimuth += 360

    # Origine = milieu(C,D) = LTP
    O = (C + D) / 2  # LTP
    P_mid = (A + B) / 2  # FPAP

    # Orientation camera (LARD convention)
    cam_orientation = np.radians([opp_runway_azimuth - yaw_cam, 90 - pitch_cam, roll_cam])

    # Camera en cartesien
    cam_pos = np.array([lat, lon, alt])
    cam_pos_cart = geodetic_to_cartesian(*cam_pos, *O, opp_runway_azimuth)

    # Repere camera
    x_cam, y_cam, z_cam = apply_rotations_int(*cam_orientation)

    if verbose:
        print(f"\n{'='*70}")
        print(f"DIAGNOSTIC FRAME — Piste {target_runway}")
        print(f"{'='*70}")
        print(f"\n--- Pose camera (YAML) ---")
        print(f"  lon={lon:.8f}  lat={lat:.8f}  alt={alt:.2f}m")
        print(f"  yaw={yaw_cam:.2f}  pitch={pitch_cam:.2f}  roll={roll_cam:.2f}")
        print(f"\n--- DB piste {target_runway} (4 coins lat/lon/alt) ---")
        for name, pt in zip(["A(TR)", "B(TL)", "C(BL)", "D(BR)"], [A, B, C, D]):
            print(f"  {name}: lat={pt[0]:.8f}  lon={pt[1]:.8f}  alt={pt[2]:.2f}m")
        print(f"  LTP  (milieu CD): lat={O[0]:.8f}  lon={O[1]:.8f}  alt={O[2]:.2f}m")
        print(f"  FPAP (milieu AB): lat={P_mid[0]:.8f}  lon={P_mid[1]:.8f}  alt={P_mid[2]:.2f}m")
        print(f"\n--- Geometrie ---")
        print(f"  opp_runway_azimuth = {opp_runway_azimuth:.2f} deg")
        print(f"  cam_orientation (rad) = yaw:{cam_orientation[0]:.4f}  "
              f"pitch:{cam_orientation[1]:.4f}  roll:{cam_orientation[2]:.4f}")
        print(f"  cam_orientation (deg) = yaw:{np.degrees(cam_orientation[0]):.2f}  "
              f"pitch:{np.degrees(cam_orientation[1]):.2f}  roll:{np.degrees(cam_orientation[2]):.2f}")
        print(f"  cam_pos_cart (local) = x:{cam_pos_cart[0]:.2f}  y:{cam_pos_cart[1]:.2f}  z:{cam_pos_cart[2]:.2f}")
        print(f"\n--- Repere camera ---")
        print(f"  x_cam = {x_cam}")
        print(f"  y_cam = {y_cam}")
        print(f"  z_cam = {z_cam}")

    # Projeter les 4 coins
    corners_px = []
    for i, (name, pt) in enumerate(zip(["A(TR)", "B(TL)", "C(BL)", "D(BR)"], pts)):
        pt_arr = np.array(pt)
        pt_cart = geodetic_to_cartesian(*pt_arr, *O, opp_runway_azimuth)
        pt_trans = pt_cart - cam_pos_cart
        pt_cam = point_in_new_repere(pt_trans, x_cam, y_cam, z_cam)
        px_x, px_y = pointcam_to_pix(pt_cam, fov_x, fov_y, img_height, img_width)

        if verbose:
            print(f"\n--- Coin {name} ---")
            print(f"  cartesien (local)  = x:{pt_cart[0]:.2f}  y:{pt_cart[1]:.2f}  z:{pt_cart[2]:.2f}")
            print(f"  translate (vs cam) = x:{pt_trans[0]:.2f}  y:{pt_trans[1]:.2f}  z:{pt_trans[2]:.2f}")
            print(f"  repere camera      = x:{pt_cam[0]:.2f}  y:{pt_cam[1]:.2f}  z:{pt_cam[2]:.2f}")
            if px_x is not None:
                print(f"  pixel              = ({px_x}, {px_y})")
            else:
                print(f"  pixel              = DERRIERE CAMERA")

        corners_px.append((px_x, px_y) if px_x is not None else None)

    return corners_px


def draw_diagnostic(img_path, corners_px, output_path, frame_idx, runway):
    """Dessine le trapeze GT + labels sur l'image."""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        font_small = font

    valid = [c for c in corners_px if c is not None]
    if len(valid) < 4:
        draw.text((10, 10), f"Frame {frame_idx} — {runway} — COINS DERRIERE CAMERA",
                  fill="red", font=font)
        img.save(output_path)
        return

    # Trapeze GT (cyan, epaisseur 3)
    for i in range(4):
        draw.line([corners_px[i], corners_px[(i + 1) % 4]], fill="cyan", width=3)

    # Labels aux coins
    corner_names = ["A(TR)", "B(TL)", "C(BL)", "D(BR)"]
    for name, (px, py) in zip(corner_names, corners_px):
        # Point
        r = 4
        draw.ellipse([px - r, py - r, px + r, py + r], fill="yellow")
        # Texte
        txt = f"{name} ({px},{py})"
        bbox = draw.textbbox((px + 8, py - 8), txt, font=font_small)
        draw.rectangle([bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1], fill="black")
        draw.text((px + 8, py - 8), txt, fill="yellow", font=font_small)

    # Centre du trapeze
    cx = sum(c[0] for c in corners_px) // 4
    cy = sum(c[1] for c in corners_px) // 4
    draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill="red")

    # Titre
    title = f"Frame {frame_idx} | {runway} | GT LARD diag"
    bbox = draw.textbbox((10, 10), title, font=font)
    draw.rectangle([8, 8, bbox[2] + 4, bbox[3] + 4], fill="black")
    draw.text((10, 10), title, fill="white", font=font)

    # Croix au centre image
    iw, ih = img.size
    draw.line([(iw // 2 - 10, ih // 2), (iw // 2 + 10, ih // 2)], fill="red", width=1)
    draw.line([(iw // 2, ih // 2 - 10), (iw // 2, ih // 2 + 10)], fill="red", width=1)

    img.save(output_path, quality=95)


def sweep_fov(pose, runway_db, airport, runway, img_w, img_h,
              img_path, out_dir, frame_idx, img_digits,
              fov_range=(26.0, 34.0), fov_step=0.5):
    """Teste plusieurs FOV et genere une image par FOV pour comparaison visuelle."""
    import os
    sweep_dir = out_dir / "fov_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    fov = fov_range[0]
    results = []
    while fov <= fov_range[1] + 0.01:
        corners = diagnose_frame(pose, runway_db, airport, runway,
                                 fov, fov, img_w, img_h, verbose=False)
        fname = f"fov_{fov:.1f}_{frame_idx}.jpg"
        out_path = sweep_dir / fname

        # Dessiner avec le FOV en titre
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 22)
        except OSError:
            font = ImageFont.load_default()

        valid = [c for c in corners if c is not None]
        if len(valid) == 4:
            for i in range(4):
                draw.line([corners[i], corners[(i + 1) % 4]], fill="cyan", width=3)
            # Mesurer la largeur base (C-D) en pixels
            base_w = abs(corners[3][0] - corners[2][0])
            results.append((fov, base_w))
        else:
            results.append((fov, None))

        title = f"FOV={fov:.1f} | {runway}"
        bbox = draw.textbbox((10, 10), title, font=font)
        draw.rectangle([8, 8, bbox[2] + 4, bbox[3] + 4], fill="black")
        draw.text((10, 10), title, fill="white", font=font)

        # Croix au centre image
        draw.line([(img_w // 2 - 10, img_h // 2), (img_w // 2 + 10, img_h // 2)], fill="red", width=1)
        draw.line([(img_w // 2, img_h // 2 - 10), (img_w // 2, img_h // 2 + 10)], fill="red", width=1)

        img.save(out_path, quality=95)
        fov += fov_step

    print(f"\n{'='*50}")
    print(f"FOV SWEEP — Frame {frame_idx}, piste {runway}")
    print(f"{'='*50}")
    print(f"{'FOV':>6}  {'Base C-D (px)':>14}")
    for fov_val, base_w in results:
        w_str = f"{base_w}" if base_w is not None else "N/A"
        print(f"{fov_val:6.1f}  {w_str:>14}")
    print(f"\nImages dans {sweep_dir}")
    print(f"Compare visuellement quel FOV fait coller le trapeze sur la piste.")


def sweep_offset(pose, runway_db, airport, runway, fov_x, fov_y, img_w, img_h,
                 img_path, out_dir, frame_idx):
    """Teste avec/sans cockpit offset et differentes altitudes."""
    sweep_dir = out_dir / "offset_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    lon, lat, alt, yaw, pitch, roll = pose

    # Variations a tester
    variants = [
        ("baseline",        pose),
        ("alt+1m",          [lon, lat, alt + 1.0, yaw, pitch, roll]),
        ("alt-1m",          [lon, lat, alt - 1.0, yaw, pitch, roll]),
        ("alt+2m",          [lon, lat, alt + 2.0, yaw, pitch, roll]),
        ("alt-2m",          [lon, lat, alt - 2.0, yaw, pitch, roll]),
        ("lat+0.00001",     [lon, lat + 0.00001, alt, yaw, pitch, roll]),
        ("lat-0.00001",     [lon, lat - 0.00001, alt, yaw, pitch, roll]),
        ("lon+0.00001",     [lon + 0.00001, lat, alt, yaw, pitch, roll]),
        ("lon-0.00001",     [lon - 0.00001, lat, alt, yaw, pitch, roll]),
        ("pitch+0.5",       [lon, lat, alt, yaw, pitch + 0.5, roll]),
        ("pitch-0.5",       [lon, lat, alt, yaw, pitch - 0.5, roll]),
    ]

    print(f"\n{'='*50}")
    print(f"OFFSET SWEEP — Frame {frame_idx}, piste {runway}")
    print(f"{'='*50}")

    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    for name, var_pose in variants:
        corners = diagnose_frame(var_pose, runway_db, airport, runway,
                                 fov_x, fov_y, img_w, img_h, verbose=False)
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        valid = [c for c in corners if c is not None]
        if len(valid) == 4:
            for i in range(4):
                draw.line([corners[i], corners[(i + 1) % 4]], fill="cyan", width=3)

        title = f"{name} | {runway}"
        bbox = draw.textbbox((10, 10), title, font=font)
        draw.rectangle([8, 8, bbox[2] + 4, bbox[3] + 4], fill="black")
        draw.text((10, 10), title, fill="white", font=font)

        draw.line([(img_w // 2 - 10, img_h // 2), (img_w // 2 + 10, img_h // 2)], fill="red", width=1)
        draw.line([(img_w // 2, img_h // 2 - 10), (img_w // 2, img_h // 2 + 10)], fill="red", width=1)

        out_path = sweep_dir / f"{name}_{frame_idx}.jpg"
        img.save(out_path, quality=95)
        print(f"  {name:20s} -> {out_path.name}")

    print(f"\nImages dans {sweep_dir}")


def main():
    parser = argparse.ArgumentParser(description="Diagnostic visuel bbox GT LARD")
    parser.add_argument("--run", required=True, help="Nom du run (ex: KPDX_28L)")
    parser.add_argument("--frames", nargs="*", type=int, default=None,
                        help="Indices de frames a diagnostiquer (defaut: 0, mid, last)")
    parser.add_argument("--all", action="store_true", help="Toutes les frames")
    parser.add_argument("--db", default="xplane", choices=["xplane", "ges"],
                        help="Base de pistes a utiliser (defaut: xplane)")
    parser.add_argument("--sweep-fov", action="store_true",
                        help="Tester differentes valeurs de FOV")
    parser.add_argument("--sweep-offset", action="store_true",
                        help="Tester differentes corrections de position")
    args = parser.parse_args()

    runs_dir = PROJECT_ROOT / "runs"
    run_dir = runs_dir / args.run
    yaml_path = run_dir / f"{args.run}.yaml"
    footage_dir = run_dir / "footage"
    out_dir = run_dir / "diag_bbox"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not yaml_path.exists():
        print(f"ERREUR: {yaml_path} introuvable")
        sys.exit(1)

    # Charger le YAML
    with open(yaml_path) as f:
        scenario = yaml.safe_load(f)

    img_info = scenario["image"]
    img_w, img_h = img_info["width"], img_info["height"]
    fov_x, fov_y = img_info["fov_x"], img_info["fov_y"]
    poses = scenario["poses"]

    print(f"Image: {img_w}x{img_h}, FOV: {fov_x}x{fov_y}")
    print(f"Nombre de poses: {len(poses)}")

    # Charger la DB pistes
    if args.db == "xplane":
        ds_type = DatasetTypes.XPLANE
    else:
        ds_type = DatasetTypes.EARTH_STUDIO
    # Utiliser la DB corrigee si disponible (apt.dat)
    _fixed_db = PROJECT_ROOT / "project" / "data" / "runways_db_V2_XPlane_fixed.json"
    if ds_type == DatasetTypes.XPLANE and _fixed_db.exists():
        db_path = _fixed_db
    else:
        db_rel_path = database_name(ds_type)
        db_path = LARD_ROOT / db_rel_path
    print(f"DB pistes: {db_path}")

    with open(db_path) as f:
        runway_db = json.load(f)

    # Determiner les frames a traiter
    n = len(poses)
    if args.all:
        frame_indices = list(range(n))
    elif args.frames is not None:
        frame_indices = [i for i in args.frames if 0 <= i < n]
    else:
        # Par defaut : premiere, milieu, derniere
        frame_indices = sorted(set([0, n // 2, n - 1]))

    print(f"Frames a diagnostiquer: {frame_indices}")
    print(f"Sortie: {out_dir}")

    # Determiner le padding pour les noms de fichiers
    img_digits = len(str(n - 1))

    for idx in frame_indices:
        pose_data = poses[idx]
        airport = pose_data["airport"]
        runway = pose_data["runway"]
        pose = pose_data["pose"]  # [lon, lat, alt, yaw, pitch, roll]

        # Trouver l'image correspondante
        img_name = f"{args.run}_{str(idx).zfill(img_digits)}.jpg"
        img_path = footage_dir / img_name
        if not img_path.exists():
            # Essayer .jpeg, .png
            for ext in ["jpeg", "png"]:
                candidate = footage_dir / f"{args.run}_{str(idx).zfill(img_digits)}.{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

        if not img_path.exists():
            print(f"\n[SKIP] Frame {idx}: image {img_name} introuvable")
            continue

        print(f"\n{'#'*70}")
        print(f"# FRAME {idx} — {img_path.name}")
        print(f"{'#'*70}")

        # Diagnostic verbose pour les premieres frames, bref sinon
        verbose = (idx in frame_indices[:3]) or len(frame_indices) <= 5

        corners = diagnose_frame(
            pose, runway_db, airport, runway,
            fov_x, fov_y, img_w, img_h,
            verbose=verbose,
        )

        # Dessiner
        out_path = out_dir / f"diag_{args.run}_{str(idx).zfill(img_digits)}.jpg"
        draw_diagnostic(img_path, corners, out_path, idx, runway)
        print(f"  -> {out_path}")

        # Sweeps optionnels
        if args.sweep_fov:
            sweep_fov(pose, runway_db, airport, runway, img_w, img_h,
                      img_path, out_dir, idx, img_digits)
        if args.sweep_offset:
            sweep_offset(pose, runway_db, airport, runway,
                         fov_x, fov_y, img_w, img_h,
                         img_path, out_dir, idx)

    # Resume
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC TERMINE — {len(frame_indices)} frames dans {out_dir}")
    print(f"{'='*70}")
    print(f"\nOuvre les images dans {out_dir} pour verifier visuellement")
    print(f"si le trapeze cyan colle sur la piste dans l'image.")
    print(f"  - Les points jaunes sont les 4 coins A(TR), B(TL), C(BL), D(BR)")
    print(f"  - La croix rouge au centre = centre image (512, 512)")
    print(f"  - Le point rouge = centre du trapeze GT")


if __name__ == "__main__":
    main()
