"""
test_camera_position.py — Verifie que la camera X-Plane est bien positionnee
============================================================================
Place la camera aux coordonnees LARD du seuil de piste (LTP), a quelques
metres du sol, regard le long de la piste. Si on voit le seuil de piste
sous nos pieds, la position est correcte. Sinon, le positionnement est faux.

Teste aussi des positions connues (milieu piste, FPAP) pour verifier.

Usage :
    python tests/xplane/test_camera_position.py
    python tests/xplane/test_camera_position.py --alt 5
"""

import sys
import json
import math
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LARD_ROOT = PROJECT_ROOT / "LARD"
sys.path.insert(0, str(LARD_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "project" / "export"))

from src.geo.geo_dataset import compute_aiming_point
from src.geo.geo_utils import ecef2llh
from xplane_bridge import XPlaneConnection, XPlaneConfig

RUNWAY_DB = str(LARD_ROOT / "data" / "filtered_runways_database_Final.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--airport", default="KPDX")
    parser.add_argument("--runway", default="28L")
    parser.add_argument("--alt", type=float, default=5, help="Altitude au-dessus du sol (m)")
    args = parser.parse_args()

    # Recuperer la geometrie piste via LARD
    _, _, rwy_psi, ltp_ecef, fpap_ecef = compute_aiming_point(
        RUNWAY_DB, args.airport, args.runway, 300.0
    )
    ltp_lat, ltp_lon, ltp_alt = ecef2llh(ltp_ecef[0], ltp_ecef[1], ltp_ecef[2])
    fpap_lat, fpap_lon, fpap_alt = ecef2llh(fpap_ecef[0], fpap_ecef[1], fpap_ecef[2])

    heading = rwy_psi[0]  # heading de la piste
    mid_lat = (ltp_lat + fpap_lat) / 2
    mid_lon = (ltp_lon + fpap_lon) / 2

    print(f"=== VERIFICATION POSITION CAMERA ===")
    print(f"Piste: {args.airport}/{args.runway}")
    print(f"LTP  (seuil) : lat={ltp_lat:.8f} lon={ltp_lon:.8f} alt={ltp_alt:.1f}m")
    print(f"FPAP (oppose): lat={fpap_lat:.8f} lon={fpap_lon:.8f} alt={fpap_alt:.1f}m")
    print(f"Heading piste: {heading:.1f} deg")
    print()

    # Positions a tester
    positions = [
        ("LTP_seuil",   ltp_lat,  ltp_lon,  heading, -10),
        ("MID_milieu",  mid_lat,  mid_lon,  heading, -10),
        ("FPAP_oppose", fpap_lat, fpap_lon, (heading + 180) % 360, -10),
        ("LTP_approach", ltp_lat, ltp_lon,  (heading + 180) % 360, -3),
    ]

    config = XPlaneConfig()
    conn = XPlaneConnection(config)

    if not conn.check_connection():
        print("ERREUR: X-Plane non joignable")
        return

    conn.setup_view()

    out_dir = PROJECT_ROOT / "runs" / f"{args.airport}_{args.runway}" / "diag_bbox"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, lat, lon, hdg, pitch in positions:
        print(f"\n--- {name} ---")
        print(f"  Commande: lat={lat:.8f} lon={lon:.8f}")

        # Deplacer le ref au point cible pour precision
        conn.move_reference_to(lat, lon)
        terrain_elev = conn.ref_elev
        cam_alt = terrain_elev + args.alt

        print(f"  Terrain: {terrain_elev:.1f}m, Camera: {cam_alt:.1f}m")

        conn.set_camera_pose(lat, lon, cam_alt, hdg, pitch, 0.0)
        time.sleep(0.3)

        # Readback
        real = conn.read_actual_pose()
        print(f"  Readback: lat={real['lat']:.8f} lon={real['lon']:.8f} alt={real['alt_m']:.2f}m")

        dlat = (real['lat'] - lat) * 111320
        dlon = (real['lon'] - lon) * 111320 * math.cos(math.radians(lat))
        dalt = real['alt_m'] - cam_alt
        print(f"  Ecart: lat={dlat:.2f}m lon={dlon:.2f}m alt={dalt:.2f}m")

        # Capture
        img_path = out_dir / f"pos_{name}.jpg"
        conn.capture_frame(img_path)
        print(f"  Image: {img_path}")

    conn.close()
    print(f"\n=== IMAGES DANS {out_dir} ===")
    print("Verifie visuellement :")
    print("  - LTP_seuil : on devrait etre AU DESSUS du seuil de piste")
    print("  - MID_milieu : au milieu de la piste")
    print("  - FPAP_oppose : a l'autre bout, regardant vers nous")
    print("  - LTP_approach : au seuil, regardant en arriere (vue d'approche)")


if __name__ == "__main__":
    main()
