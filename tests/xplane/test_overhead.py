"""
test_overhead.py — Diagnostic complet bbox : overhead + pilot eye + terrain
==========================================================================
3 tests en 1 :
  1) Camera au-dessus du seuil → projection bbox → decalage visible ?
  2) Pilot eye offset : delta entre position avion (local_x/y/z)
     et position camera (view_x/y/z)
  3) Altitude terrain des 4 coins piste vs altitude DB

Usage :
    python tests/xplane/test_overhead.py
    python tests/xplane/test_overhead.py --airport KPDX --runway 28L --alt 200
"""

import sys
import json
import math
import time
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "project" / "export"))

from xplane_bridge import XPlaneConnection, XPlaneConfig


def load_runway_db():
    db_path = PROJECT_ROOT / "project" / "data" / "runways_db_V2_XPlane_aptdat.json"
    with open(db_path) as f:
        return json.load(f)


def project_corner(corner_latlon, cam_lat, cam_lon, cam_alt, cam_yaw,
                   cam_pitch_xp, cam_roll, fov_h, fov_v, img_w, img_h):
    """Projection pinhole ENU → pixel (meme logique que gt_xplane.py)."""
    import pyproj
    geod = pyproj.Geod(ellps="WGS84")

    c_lat, c_lon, c_alt = corner_latlon

    azimuth, _, distance = geod.inv(cam_lon, cam_lat, c_lon, c_lat)

    # ENU
    dx_east = distance * math.sin(math.radians(azimuth))
    dx_north = distance * math.cos(math.radians(azimuth))
    dx_up = c_alt - cam_alt

    yaw_rad = math.radians(cam_yaw)
    pitch_rad = math.radians(cam_pitch_xp)
    roll_rad = math.radians(cam_roll)

    # ENU -> (forward, right, down)
    fwd = dx_north * math.cos(yaw_rad) + dx_east * math.sin(yaw_rad)
    right = -dx_north * math.sin(yaw_rad) + dx_east * math.cos(yaw_rad)
    down = -dx_up

    # Pitch
    fwd2 = fwd * math.cos(pitch_rad) - down * math.sin(pitch_rad)
    down2 = fwd * math.sin(pitch_rad) + down * math.cos(pitch_rad)
    right2 = right

    # Roll
    right3 = right2 * math.cos(roll_rad) + down2 * math.sin(roll_rad)
    down3 = -right2 * math.sin(roll_rad) + down2 * math.cos(roll_rad)

    if fwd2 <= 0:
        return None

    # Pinhole avec FOV separes H/V
    half_fov_h = math.radians(fov_h / 2)
    half_fov_v = math.radians(fov_v / 2)
    px_x = img_w / 2 + (right3 / fwd2) * (img_w / 2) / math.tan(half_fov_h)
    px_y = img_h / 2 + (down3 / fwd2) * (img_h / 2) / math.tan(half_fov_v)

    return (int(px_x), int(px_y))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alt", type=float, default=150, help="Altitude AGL (m)")
    parser.add_argument("--point", default="mid", choices=["ltp", "mid", "fpap"])
    parser.add_argument("--airport", default="KPDX")
    parser.add_argument("--runway", default="28L")
    parser.add_argument("--pitch", type=float, default=-90, help="Pitch X-Plane (deg)")
    args = parser.parse_args()

    db = load_runway_db()
    rwy = db[args.airport][args.runway]

    A = rwy["A"]["coordinate"]
    B = rwy["B"]["coordinate"]
    C = rwy["C"]["coordinate"]
    D = rwy["D"]["coordinate"]

    corners_db = {
        "A": (A["latitude"], A["longitude"], A["altitude"]),
        "B": (B["latitude"], B["longitude"], B["altitude"]),
        "C": (C["latitude"], C["longitude"], C["altitude"]),
        "D": (D["latitude"], D["longitude"], D["altitude"]),
    }

    ltp_lat = (C["latitude"] + D["latitude"]) / 2
    ltp_lon = (C["longitude"] + D["longitude"]) / 2
    fpap_lat = (A["latitude"] + B["latitude"]) / 2
    fpap_lon = (A["longitude"] + B["longitude"]) / 2
    mid_lat = (ltp_lat + fpap_lat) / 2
    mid_lon = (ltp_lon + fpap_lon) / 2

    if args.point == "ltp":
        cam_lat, cam_lon = ltp_lat, ltp_lon
    elif args.point == "mid":
        cam_lat, cam_lon = mid_lat, mid_lon
    else:
        cam_lat, cam_lon = fpap_lat, fpap_lon

    # Heading piste
    import pyproj
    geod = pyproj.Geod(ellps="WGS84")
    az, _, _ = geod.inv(ltp_lon, ltp_lat, fpap_lon, fpap_lat)
    rwy_heading = az % 360
    cam_heading = rwy_heading

    config = XPlaneConfig()
    conn = XPlaneConnection(config)

    if not conn.check_connection():
        print("ERREUR: X-Plane non joignable")
        return

    conn.setup_view()

    # === TEST 2 : Pilot eye offset ===
    print(f"\n{'='*60}")
    print(f"TEST 2 : PILOT EYE OFFSET")
    print(f"{'='*60}")
    print(f"  acf_peX (lateral)      = {conn.pilot_eye_x:.3f} m")
    print(f"  acf_peY (vertical)     = {conn.pilot_eye_y:.3f} m")
    print(f"  acf_peZ (longitudinal) = {conn.pilot_eye_z:.3f} m")

    # Positionner pour mesurer le delta view vs avion
    conn.move_reference_to(cam_lat, cam_lon)
    terrain_elev = conn.ref_elev
    cam_alt = terrain_elev + args.alt
    conn.set_camera_pose(cam_lat, cam_lon, cam_alt, cam_heading, args.pitch, 0.0)
    time.sleep(0.5)

    # Lire position avion
    ax = conn.read_dref("sim/flightmodel/position/local_x", 56)
    ay = conn.read_dref("sim/flightmodel/position/local_y", 57)
    az_pos = conn.read_dref("sim/flightmodel/position/local_z", 58)
    # Lire position camera
    vx = conn.read_dref("sim/graphics/view/view_x", 50)
    vy = conn.read_dref("sim/graphics/view/view_y", 51)
    vz = conn.read_dref("sim/graphics/view/view_z", 52)

    if ax is not None and vx is not None:
        dx = vx - ax
        dy = vy - ay if vy else 0
        dz = vz - az_pos if vz else 0
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        print(f"\n  Position avion  (local): x={ax:.3f} y={ay:.3f} z={az_pos:.3f}")
        print(f"  Position camera (view) : x={vx:.3f} y={vy:.3f} z={vz:.3f}")
        print(f"  Delta (cam - avion)    : dx={dx:.3f} dy={dy:.3f} dz={dz:.3f} ({dist:.3f}m)")
        print(f"\n  → C'est l'offset reel camera/avion. Si >1m lateral, c'est le pilot eye.")

    # === TEST 3 : Altitude terrain des 4 coins ===
    print(f"\n{'='*60}")
    print(f"TEST 3 : ALTITUDE TERRAIN DES COINS")
    print(f"{'='*60}")
    print(f"  Alt DB (aeroport) : {A['altitude']:.1f}m")

    corner_elevs = {}
    for label in ["A", "B", "C", "D"]:
        c = corners_db[label]
        elev = conn.read_terrain_elevation(c[0], c[1])
        corner_elevs[label] = elev
        delta = elev - c[2] if elev else None
        print(f"  Coin {label}: DB={c[2]:.1f}m  terrain={elev:.1f}m  delta={delta:+.1f}m"
              if elev else f"  Coin {label}: DB={c[2]:.1f}m  terrain=ERREUR")

    # Repositionner la camera apres les mesures terrain
    conn.move_reference_to(cam_lat, cam_lon)
    terrain_elev = conn.ref_elev
    cam_alt = terrain_elev + args.alt
    conn.set_camera_pose(cam_lat, cam_lon, cam_alt, cam_heading, args.pitch, 0.0)
    time.sleep(0.5)

    # === TEST 1 : Overhead + projection bbox ===
    print(f"\n{'='*60}")
    print(f"TEST 1 : OVERHEAD — PROJECTION BBOX")
    print(f"{'='*60}")
    print(f"  Piste     : {args.airport}/{args.runway}")
    print(f"  Position  : {args.point}, alt AGL={args.alt}m, pitch={args.pitch}")
    print(f"  Terrain   : {terrain_elev:.1f}m")
    print(f"  FOV       : H={config.fov_h}° V={config.fov_v}°")
    print(f"  Resolution: {config.window_width}x{config.window_height}")

    real = conn.read_actual_pose()
    print(f"\n  Pose reelle:")
    print(f"    lat={real['lat']:.8f} lon={real['lon']:.8f} alt={real['alt_m']:.2f}m")
    print(f"    heading={real['heading']:.2f} pitch={real['pitch']:.2f} roll={real['roll']:.2f}")

    # Capture
    out_dir = PROJECT_ROOT / "tests" / "xplane" / "overhead_diag"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"overhead_{args.airport}_{args.runway}_{args.point}.jpg"
    conn.capture_frame(img_path)
    print(f"\n  Image: {img_path}")

    # Projeter avec terrain reel pour les coins
    corners_corrected = {}
    for label in ["A", "B", "C", "D"]:
        c = corners_db[label]
        alt = corner_elevs.get(label, c[2])
        corners_corrected[label] = (c[0], c[1], alt)

    img_w = config.window_width
    img_h = config.window_height

    print(f"\n  Projection (pose reelle + terrain reel):")
    px_real = {}
    for label in ["A", "B", "C", "D"]:
        px = project_corner(
            corners_corrected[label],
            real["lat"], real["lon"], real["alt_m"],
            real["heading"], real["pitch"], real["roll"],
            config.fov_h, config.fov_v, img_w, img_h,
        )
        px_real[label] = px
        print(f"    {label}: {px}")

    # Aussi projeter avec alt DB (pour comparer)
    print(f"\n  Projection (pose reelle + alt DB):")
    px_db = {}
    for label in ["A", "B", "C", "D"]:
        px = project_corner(
            corners_db[label],
            real["lat"], real["lon"], real["alt_m"],
            real["heading"], real["pitch"], real["roll"],
            config.fov_h, config.fov_v, img_w, img_h,
        )
        px_db[label] = px
        print(f"    {label}: {px}")

    # Dessiner sur l'image
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        font_sm = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    corner_order = ["A", "B", "C", "D"]

    # Trapeze terrain reel = cyan
    valid = [px_real[l] for l in corner_order if px_real[l] is not None]
    if len(valid) == 4:
        pts = [px_real[l] for l in corner_order]
        for i in range(4):
            draw.line([pts[i], pts[(i + 1) % 4]], fill="cyan", width=3)
        for l in corner_order:
            px = px_real[l]
            draw.ellipse([px[0]-5, px[1]-5, px[0]+5, px[1]+5], fill="cyan")
            draw.text((px[0]+10, px[1]-10), f"{l} terrain", fill="cyan", font=font_sm)

    # Trapeze alt DB = jaune
    valid_db = [px_db[l] for l in corner_order if px_db[l] is not None]
    if len(valid_db) == 4:
        pts = [px_db[l] for l in corner_order]
        for i in range(4):
            draw.line([pts[i], pts[(i + 1) % 4]], fill="yellow", width=2)

    # Croix centre
    cx, cy = img_w // 2, img_h // 2
    draw.line([(cx-20, cy), (cx+20, cy)], fill="red", width=1)
    draw.line([(cx, cy-20), (cx, cy+20)], fill="red", width=1)

    # Legende
    legend = (f"{args.airport}/{args.runway} | {args.point} +{args.alt}m | "
              f"cyan=terrain reel, jaune=alt DB")
    bbox = draw.textbbox((10, 10), legend, font=font)
    draw.rectangle([8, 8, bbox[2]+4, bbox[3]+4], fill="black")
    draw.text((10, 10), legend, fill="white", font=font)

    out_ann = out_dir / f"overhead_{args.airport}_{args.runway}_{args.point}_annotated.jpg"
    img.save(out_ann, quality=95)
    print(f"\n  Image annotee: {out_ann}")

    # Resume
    print(f"\n{'='*60}")
    print(f"RESUME")
    print(f"{'='*60}")
    if ax is not None and vx is not None:
        print(f"  Pilot eye delta: dx={dx:.3f} dy={dy:.3f} dz={dz:.3f}m")
    for label in ["A", "B", "C", "D"]:
        if corner_elevs.get(label) is not None:
            d = corner_elevs[label] - corners_db[label][2]
            print(f"  Coin {label} altitude delta: {d:+.1f}m")
    print(f"\n  → Ouvre {out_ann.name} et verifie si le trapeze cyan colle a la piste.")
    print(f"  → Si decale: probleme de projection ou de coordonnees apt.dat.")
    print(f"  → Si OK: le bug est dans le positionnement en approche.")

    conn.close()


if __name__ == "__main__":
    main()
