"""
test_fov_calibration.py — Test FOV X-Plane pour calibrer les bbox GT
=====================================================================
Pre-requis :
    1. X-Plane 12 lance, charge sur KPDX (Portland)
    2. Mode fenetre (pas plein ecran)
    3. Etre pret a voler (pas dans les menus)

Usage :
    python tests/xplane/test_fov_calibration.py

Ce script :
    1. Se connecte a X-Plane (UDP 49000)
    2. Resize la fenetre a 1280x1024
    3. Pause + override + vue sans cockpit
    4. Positionne la camera a une pose connue (KPDX_21, frame 69)
    5. Teste 3 valeurs de FOV : 30, 37 (actuel), et le FOV par defaut X-Plane
    6. Pour chaque FOV : capture l'image + dessine le trapeze GT
    7. Sauve les images dans tests/xplane/fov_calibration/

Tu compares visuellement les 3 images pour trouver quel FOV colle.
"""

import sys
import time
import math
import json
import struct
import socket
import numpy as np
import pyproj
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- Pose de test : KPDX_21 frame 69 (proche, facile a juger) ---
TEST_POSE = {
    "airport": "KPDX",
    "runway": "21",
    "lon": -122.59613217,
    "lat": 45.59695311,
    "alt": 45.65,
    "yaw": -135.08,       # heading LARD (convention aviation)
    "pitch": 86.03,       # LARD convention : 90 = level
    "roll": 0.00,
}

# Conversion LARD → X-Plane
XPLANE_HEADING = TEST_POSE["yaw"] % 360       # X-Plane veut 0-360
XPLANE_PITCH = TEST_POSE["pitch"] - 90        # LARD 90=level → XP 0=level
XPLANE_ROLL = TEST_POSE["roll"]

IMG_W, IMG_H = 1024, 1024
FOV_YAML = 30.0


# =====================================================================
# UDP X-Plane (copie minimale de xplane_bridge)
# =====================================================================

def pack_dref(dref, value):
    msg = b"DREF\x00"
    msg += struct.pack('<f', value)
    msg += dref.encode('utf-8').ljust(500, b'\x00')
    return msg

def pack_cmnd(command):
    msg = b"CMND\x00"
    msg += command.encode('utf-8') + b'\x00'
    return msg

def read_dref(sock, addr, dref, idx=0):
    msg = b"RREF\x00"
    msg += struct.pack('<ii', 1, idx)
    msg += dref.encode('utf-8').ljust(400, b'\x00')
    sock.sendto(msg, addr)
    try:
        data, _ = sock.recvfrom(1024)
        val = struct.unpack('<f', data[9:13])[0] if len(data) >= 13 else None
    except socket.timeout:
        val = None
    # Stop stream
    msg2 = b"RREF\x00"
    msg2 += struct.pack('<ii', 0, idx)
    msg2 += dref.encode('utf-8').ljust(400, b'\x00')
    sock.sendto(msg2, addr)
    return val


# =====================================================================
# Projection GT (reproduction de computeLabels)
# =====================================================================

def geodetic_to_cartesian(lat, lon, alt, ref_lat, ref_lon, ref_alt, runw_az):
    geod = pyproj.Geod(ellps="WGS84")
    azimuth, _, distance = geod.inv(ref_lon, ref_lat, lon, lat)
    x = distance * np.cos(np.radians(azimuth - runw_az))
    y = -distance * np.sin(np.radians(azimuth - runw_az))
    z = alt - ref_alt
    return np.array([x, y, z])

def get_forward_azimuth(P1, P2):
    geod = pyproj.Geod(ellps="WGS84")
    azimuth, _, _ = geod.inv(P1[1], P1[0], P2[1], P2[0])
    return azimuth

def rotation_matrix(axis, angle):
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cos_t, sin_t = np.cos(angle), np.sin(angle)
    Q = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    I = np.eye(3)
    P = np.array([[x*x, x*y, x*z], [y*x, y*y, y*z], [z*x, z*y, z*z]])
    return P + cos_t * (I - P) + sin_t * Q

def apply_rotations_int(yaw, pitch, roll):
    X, Y, Z = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
    Ryaw = rotation_matrix(Z, yaw)
    Xp, Yp = Ryaw @ X, Ryaw @ Y
    Rpitch = rotation_matrix(Yp, pitch)
    Xt, Zt = Rpitch @ Xp, Rpitch @ Z
    Rroll = rotation_matrix(Xt, roll)
    return Xt, Rroll @ Yp, Rroll @ Zt

def pointcam_to_pix(P, fov_h, fov_v, img_h, img_v):
    if P[0] <= 0:
        return None, None
    ww = 2 * P[0] * np.tan(np.radians(fov_h) * 0.5)
    wh = 2 * P[0] * np.tan(np.radians(fov_v) * 0.5)
    xn = 1 - (P[1] / ww + 0.5)
    yn = 1 - (P[2] / wh + 0.5)
    return int(xn * img_h), int(yn * img_v)

def compute_gt_corners(pose, runway_db, airport, runway, fov_x, fov_y, img_w, img_h):
    """Retourne les 4 coins en pixels."""
    lon, lat, alt, yaw_cam, pitch_cam, roll_cam = pose
    rp = runway_db[airport][runway]
    pts = [list(rp[k]['coordinate'].values()) for k in ['A', 'B', 'C', 'D']]
    A, B, C, D = [np.array(p) for p in pts]

    opp_az = get_forward_azimuth((D + C) / 2, (A + B) / 2)
    if opp_az < 0:
        opp_az += 360
    O = (C + D) / 2

    cam_orient = np.radians([opp_az - yaw_cam, 90 - pitch_cam, roll_cam])
    cam_cart = geodetic_to_cartesian(lat, lon, alt, *O, opp_az)
    x_cam, y_cam, z_cam = apply_rotations_int(*cam_orient)

    corners = []
    for pt in pts:
        pt_cart = geodetic_to_cartesian(*np.array(pt), *O, opp_az)
        pt_cam = np.array([np.dot(pt_cart - cam_cart, ax) for ax in [x_cam, y_cam, z_cam]])
        px = pointcam_to_pix(pt_cam, fov_x, fov_y, img_h, img_w)
        corners.append(px)
    return corners


# =====================================================================
# Main
# =====================================================================

def main():
    from PIL import Image, ImageDraw, ImageFont
    try:
        import mss
        import ctypes
        import win32gui
    except ImportError:
        print("ERREUR: pip install mss pillow pywin32")
        sys.exit(1)

    out_dir = PROJECT_ROOT / "tests" / "xplane" / "fov_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Charger DB pistes ---
    db_path = PROJECT_ROOT / "LARD" / "data" / "runways_db_V2_XPlane.json"
    with open(db_path) as f:
        runway_db = json.load(f)

    pose = [TEST_POSE["lon"], TEST_POSE["lat"], TEST_POSE["alt"],
            TEST_POSE["yaw"], TEST_POSE["pitch"], TEST_POSE["roll"]]

    # --- Connexion X-Plane ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(5.0)
    addr = ("127.0.0.1", 49000)

    print("Connexion a X-Plane...")
    sock.sendto(pack_dref("sim/time/paused", 1.0), addr)
    time.sleep(0.3)

    # Test connexion
    t = read_dref(sock, addr, "sim/time/total_running_time_sec", 0)
    if t is None:
        print("ERREUR: X-Plane ne repond pas sur UDP 49000")
        print("Verifie que X-Plane est lance et pas dans un menu.")
        sys.exit(1)
    print(f"  Connecte (uptime={t:.0f}s)")

    # --- DPI aware ---
    ctypes.windll.user32.SetProcessDPIAware()

    # --- Resize fenetre ---
    print("Resize fenetre 1280x1024...")
    sys.path.insert(0, str(PROJECT_ROOT / "project" / "export"))
    from xplane_bridge import resize_xplane_window, get_xplane_capture_region
    resize_xplane_window(1280, 1024)
    time.sleep(0.5)

    # --- Setup vue ---
    print("Setup vue...")
    sock.sendto(pack_dref("sim/time/paused", 1.0), addr)
    sock.sendto(pack_dref("sim/operation/override/override_planepath[0]", 1.0), addr)
    time.sleep(0.1)
    sock.sendto(pack_dref("sim/operation/override/override_flight_control", 1.0), addr)
    sock.sendto(pack_dref("sim/operation/override/override_throttles", 1.0), addr)
    time.sleep(0.1)

    # Zero velocites
    for d in ["local_vx", "local_vy", "local_vz", "P", "Q", "R"]:
        sock.sendto(pack_dref(f"sim/flightmodel/position/{d}", 0.0), addr)
    time.sleep(0.1)

    # Vue sans cockpit
    sock.sendto(pack_cmnd("sim/view/forward_with_nothing"), addr)
    time.sleep(0.3)

    # Masquer curseur
    sock.sendto(pack_dref("sim/operation/override/override_joystick", 1.0), addr)
    sock.sendto(pack_dref("sim/joystick/mouse_is_yoke", 0.0), addr)
    ctypes.windll.user32.SetCursorPos(0, 0)
    time.sleep(0.2)

    # --- Lire pilot eye offset ---
    pe_x = read_dref(sock, addr, "sim/aircraft/view/acf_peX", 11) or 0.0
    pe_y = read_dref(sock, addr, "sim/aircraft/view/acf_peY", 12) or 0.0
    pe_z = read_dref(sock, addr, "sim/aircraft/view/acf_peZ", 13) or 0.0
    print(f"  Pilot eye offset: x={pe_x:.2f} y={pe_y:.2f} z={pe_z:.2f}")

    # --- Point de reference ---
    ref_lat = read_dref(sock, addr, "sim/flightmodel/position/latitude", 1)
    ref_lon = read_dref(sock, addr, "sim/flightmodel/position/longitude", 2)
    ref_elev = read_dref(sock, addr, "sim/flightmodel/position/elevation", 3)
    ref_lx = read_dref(sock, addr, "sim/flightmodel/position/local_x", 4)
    ref_ly = read_dref(sock, addr, "sim/flightmodel/position/local_y", 5)
    ref_lz = read_dref(sock, addr, "sim/flightmodel/position/local_z", 6)
    print(f"  Ref: lat={ref_lat:.4f} lon={ref_lon:.4f} elev={ref_elev:.1f}")

    # --- Positionner la camera ---
    lat, lon, alt = TEST_POSE["lat"], TEST_POSE["lon"], TEST_POSE["alt"]
    heading, pitch_xp, roll = XPLANE_HEADING, XPLANE_PITCH, XPLANE_ROLL

    # lat/lon → local
    dlat = lat - ref_lat
    dlon = lon - ref_lon
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(ref_lat))
    dn = dlat * m_per_deg_lat
    de = dlon * m_per_deg_lon
    lx = ref_lx + de
    lz = ref_lz - dn
    ly = ref_ly + (alt - ref_elev)

    # Cockpit offset compensation (heading only)
    hdg_rad = math.radians(heading)
    sin_h, cos_h = math.sin(hdg_rad), math.cos(hdg_rad)
    lx -= pe_z * sin_h + pe_x * cos_h
    ly -= pe_y
    lz -= -pe_z * cos_h + pe_x * sin_h

    def set_pose():
        sock.sendto(pack_dref("sim/flightmodel/position/local_x", lx), addr)
        sock.sendto(pack_dref("sim/flightmodel/position/local_y", ly), addr)
        sock.sendto(pack_dref("sim/flightmodel/position/local_z", lz), addr)
        sock.sendto(pack_dref("sim/flightmodel/position/theta", pitch_xp), addr)
        sock.sendto(pack_dref("sim/flightmodel/position/phi", roll), addr)
        sock.sendto(pack_dref("sim/flightmodel/position/psi", heading), addr)
        sock.sendto(pack_dref("sim/flightmodel/position/local_vx", 0.0), addr)
        sock.sendto(pack_dref("sim/flightmodel/position/local_vy", 0.0), addr)
        sock.sendto(pack_dref("sim/flightmodel/position/local_vz", 0.0), addr)

    print(f"\nCamera: lat={lat:.6f} lon={lon:.6f} alt={alt:.1f}")
    print(f"  heading={heading:.1f} pitch_xp={pitch_xp:.1f} roll={roll:.1f}")
    print(f"  local: x={lx:.1f} y={ly:.1f} z={lz:.1f}")

    # --- Capture avec differents FOV ---
    sct = mss.mss()

    # Lire FOV actuel AVANT de changer quoi que ce soit
    fov_default = read_dref(sock, addr, "sim/graphics/view/field_of_view_deg", 20)
    print(f"\nFOV par defaut X-Plane: {fov_default}")

    # FOV a tester :
    # - "current_37": ce qu'on envoyait avant (37°, en pensant que c'est horizontal)
    # - "direct_30": 30° directement (hypothese: le dataref est le vertical FOV)
    # - "default": le FOV par defaut X-Plane (pour reference)
    fov_tests = [
        ("fov_37_horizontal", 37.03, "Actuel : on envoie 37 pensant horizontal"),
        ("fov_30_direct",     30.0,  "Hypothese : dataref = vertical, on envoie 30"),
        ("fov_33",            33.0,  "Intermediaire 33"),
        ("fov_35",            35.0,  "Intermediaire 35"),
    ]
    if fov_default and abs(fov_default - 37.03) > 1.0 and abs(fov_default - 30.0) > 1.0:
        fov_tests.append(("fov_default", fov_default, f"Defaut X-Plane ({fov_default:.1f})"))

    try:
        font = ImageFont.truetype("arial.ttf", 24)
        font_sm = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    results = []

    for name, fov_val, desc in fov_tests:
        print(f"\n--- Test {name} (FOV={fov_val:.1f}°) ---")
        print(f"    {desc}")

        # Set FOV
        sock.sendto(pack_dref("sim/graphics/view/field_of_view_deg", fov_val), addr)
        time.sleep(0.5)

        # Repositionner
        set_pose()
        time.sleep(0.3)

        # Readback FOV
        rb = read_dref(sock, addr, "sim/graphics/view/field_of_view_deg", 21)
        print(f"    Readback FOV: {rb}")

        # Capture
        region = get_xplane_capture_region()
        if not region:
            print("    ERREUR: fenetre introuvable")
            continue

        grab = sct.grab(region)
        from PIL import Image as PILImage
        pil_img = PILImage.frombytes("RGB", (grab.width, grab.height),
                                      grab.rgb, "raw", "RGB")
        # Crop centre 1024x1024
        w, h = pil_img.size
        if w > h:
            margin = (w - h) // 2
            pil_img = pil_img.crop((margin, 0, w - margin, h))

        # --- Calculer GT avec FOV=30 (ce que LARD fait) ---
        corners_30 = compute_gt_corners(pose, runway_db,
                                        TEST_POSE["airport"], TEST_POSE["runway"],
                                        30.0, 30.0, IMG_W, IMG_H)

        # --- Aussi calculer avec le FOV reel pour comparer ---
        # Si le dataref est le vFOV, le FOV apres crop carre = fov_val directement
        # Si le dataref est le hFOV, le vFOV = 2*atan(h/w * tan(fov_val/2))
        #   pour 1280x1024 : vfov = 2*atan(0.8*tan(fov_val/2))
        #   apres crop carre : fov_crop = vfov
        vfov_if_horizontal = 2 * math.degrees(math.atan(
            1024/1280 * math.tan(math.radians(fov_val/2))))
        corners_vfov = compute_gt_corners(pose, runway_db,
                                          TEST_POSE["airport"], TEST_POSE["runway"],
                                          vfov_if_horizontal, vfov_if_horizontal,
                                          IMG_W, IMG_H)
        corners_direct = compute_gt_corners(pose, runway_db,
                                            TEST_POSE["airport"], TEST_POSE["runway"],
                                            fov_val, fov_val, IMG_W, IMG_H)

        # --- Dessiner ---
        draw = ImageDraw.Draw(pil_img)

        # Trapeze CYAN = GT a FOV=30 (ce que le pipeline fait)
        c30 = corners_30
        if all(c[0] is not None for c in c30):
            for i in range(4):
                draw.line([c30[i], c30[(i+1)%4]], fill="cyan", width=3)

        # Trapeze JAUNE = GT recalcule SI dataref = horizontal FOV
        if all(c[0] is not None for c in corners_vfov):
            for i in range(4):
                draw.line([corners_vfov[i], corners_vfov[(i+1)%4]], fill="yellow", width=2)

        # Trapeze ROUGE = GT recalcule SI dataref = vertical FOV (direct)
        if all(c[0] is not None for c in corners_direct):
            for i in range(4):
                draw.line([corners_direct[i], corners_direct[(i+1)%4]], fill="red", width=2)

        # Legende
        y_leg = 10
        for color, label in [("cyan", f"CYAN = GT actuel (FOV=30)"),
                              ("yellow", f"JAUNE = GT si dataref=hFOV (vfov={vfov_if_horizontal:.1f})"),
                              ("red", f"ROUGE = GT si dataref=vFOV (fov={fov_val:.1f})")]:
            bb = draw.textbbox((10, y_leg), label, font=font_sm)
            draw.rectangle([8, y_leg-2, bb[2]+4, bb[3]+4], fill="black")
            draw.text((10, y_leg), label, fill=color, font=font_sm)
            y_leg += 22

        title = f"FOV dataref={fov_val:.1f} | {TEST_POSE['runway']}"
        bb = draw.textbbox((10, y_leg+5), title, font=font)
        draw.rectangle([8, y_leg+3, bb[2]+4, bb[3]+6], fill="black")
        draw.text((10, y_leg+5), title, fill="white", font=font)

        # Centre image
        draw.line([(IMG_W//2-15, IMG_H//2), (IMG_W//2+15, IMG_H//2)], fill="green", width=1)
        draw.line([(IMG_W//2, IMG_H//2-15), (IMG_W//2, IMG_H//2+15)], fill="green", width=1)

        out_path = out_dir / f"{name}.jpg"
        pil_img.save(str(out_path), quality=95)
        print(f"    -> {out_path}")

        # Mesurer largeur base C-D pour chaque trapeze
        def base_w(corners):
            if corners[2][0] is not None and corners[3][0] is not None:
                return abs(corners[3][0] - corners[2][0])
            return None

        results.append({
            "name": name,
            "fov_sent": fov_val,
            "fov_readback": rb,
            "base_cd_fov30": base_w(c30),
            "base_cd_vfov": base_w(corners_vfov),
            "base_cd_direct": base_w(corners_direct),
        })

    # --- Resume ---
    print(f"\n{'='*70}")
    print("RESUME CALIBRATION FOV")
    print(f"{'='*70}")
    print(f"{'Test':<25} {'Sent':>6} {'Read':>6} {'Base30':>7} {'BaseH':>7} {'BaseV':>7}")
    for r in results:
        rb_str = f"{r['fov_readback']:.1f}" if r['fov_readback'] else "N/A"
        print(f"{r['name']:<25} {r['fov_sent']:6.1f} {rb_str:>6} "
              f"{r['base_cd_fov30'] or 'N/A':>7} "
              f"{r['base_cd_vfov'] or 'N/A':>7} "
              f"{r['base_cd_direct'] or 'N/A':>7}")

    print(f"\nImages dans {out_dir}")
    print(f"\nComment lire les images :")
    print(f"  CYAN  = trapeze GT actuel (FOV=30, ce que le pipeline produit)")
    print(f"  JAUNE = trapeze si le dataref est le FOV horizontal")
    print(f"  ROUGE = trapeze si le dataref est le FOV vertical")
    print(f"\nLe trapeze qui COLLE PARFAITEMENT sur la piste = la bonne interpretation.")
    print(f"Si c'est le ROUGE → il faut changer le code pour envoyer 30 direct.")
    print(f"Si c'est le JAUNE → l'interpretation actuelle est correcte, probleme ailleurs.")
    print(f"Si c'est le CYAN  → le FOV actuel est bon, probleme dans la DB.")

    # Cleanup
    sock.close()
    sct.close()


if __name__ == "__main__":
    main()
