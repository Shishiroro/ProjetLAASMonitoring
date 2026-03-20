"""
fix_runway_db.py — Genere une DB pistes corrigee depuis apt.dat pour X-Plane
=============================================================================
Les coords dans runways_db_V2_XPlane.json sont decalees par rapport au rendu
X-Plane reel (jusqu'a 395m pour certaines pistes KPDX).

Ce script :
1. Lit les centres de seuils depuis apt.dat (coords exactes X-Plane)
2. Calcule les 4 coins (A, B, C, D) a partir de centre + largeur + heading
3. Convertit en ECEF (format DB LARD)
4. Sauve dans project/data/runways_db_V2_XPlane_fixed.json

Usage :
    python tests/xplane/fix_runway_db.py
    python tests/xplane/fix_runway_db.py --apt-dat "C:/X-Plane 12/Global Scenery/Global Airports/Earth nav data/apt.dat"
"""

import json
import math
import argparse
import numpy as np
import pyproj
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LARD_ROOT = PROJECT_ROOT / "LARD"
ORIGINAL_DB = LARD_ROOT / "data" / "runways_db_V2_XPlane.json"
OUTPUT_DB = PROJECT_ROOT / "project" / "data" / "runways_db_V2_XPlane_fixed.json"


# =====================================================================
# Donnees KPDX depuis apt.dat (extraites dans ProblemeXPLANE.txt)
# Format : (lat_seuil, lon_seuil) pour chaque bout de piste
# =====================================================================

KPDX_APT_DATA = {
    # Paire 10L/28R : largeur 46m
    "10L": {"lat": 45.5965044, "lon": -122.6000210, "width_m": 46.0},
    "28R": {"lat": 45.5834197, "lon": -122.5664451, "width_m": 46.0},
    # Paire 10R/28L : largeur 45m
    "10R": {"lat": 45.5951448, "lon": -122.6214766, "width_m": 45.0},
    "28L": {"lat": 45.5805023, "lon": -122.5838805, "width_m": 45.0},
    # Paire 3/21 : largeur 45m
    "3":   {"lat": 45.5824057, "lon": -122.6168213, "width_m": 45.0},
    "21":  {"lat": 45.5940460, "lon": -122.6002578, "width_m": 45.0},
}

# Paires de pistes (chaque piste pointe vers l'extremite opposee)
RUNWAY_PAIRS = {
    "10L": "28R", "28R": "10L",
    "10R": "28L", "28L": "10R",
    "3": "21", "21": "3",
}


def llh_to_ecef(lat, lon, alt):
    """Convertit lat/lon/alt (WGS84) en ECEF (x, y, z)."""
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=False)
    x, y, z = transformer.transform(lat, lon, alt)
    return x, y, z


def compute_runway_corners(airport_icao, runway_id, apt_data, pairs):
    """
    Calcule les 4 coins A, B, C, D d'une piste.

    Convention LARD (get_runway_points) :
        LTP  = milieu(C, D) = seuil de piste (cote approche)
        FPAP = milieu(A, B) = extremite opposee

    Pour la piste "runway_id" :
        - C, D = seuil (LTP) = les 2 coins cote approche
        - A, B = extremite (FPAP) = les 2 coins cote oppose
        - D est a droite en regardant depuis l'avion, C a gauche

    Vue du dessus (avion arrive par le bas) :
        A (TR) -------- B (TL)
        |    extremite   |
        |                |
        |     piste      |
        |                |
        |    seuil       |
        D (BR) -------- C (BL)
    """
    this_rwy = apt_data[runway_id]
    opp_id = pairs[runway_id]
    opp_rwy = apt_data[opp_id]

    # Heading de la piste : azimut du seuil vers l'extremite opposee
    geod = pyproj.Geod(ellps="WGS84")
    heading, _, rwy_length = geod.inv(
        this_rwy["lon"], this_rwy["lat"],
        opp_rwy["lon"], opp_rwy["lat"]
    )

    width = this_rwy["width_m"]
    half_w = width / 2.0

    # Perpendiculaire a la piste (vers la droite quand on regarde depuis l'avion)
    perp_right = heading + 90.0  # azimut vers la droite
    perp_left = heading - 90.0   # azimut vers la gauche

    # Altitude du seuil : lire depuis la DB originale si disponible, sinon 0
    alt = 0.0

    # --- Coins du seuil (C, D) ---
    # D (BR) = seuil + half_w vers la droite
    d_lon, d_lat, _ = geod.fwd(this_rwy["lon"], this_rwy["lat"], perp_right, half_w)
    # C (BL) = seuil + half_w vers la gauche
    c_lon, c_lat, _ = geod.fwd(this_rwy["lon"], this_rwy["lat"], perp_left, half_w)

    # --- Coins de l'extremite (A, B) ---
    # A (TR) = extremite + half_w vers la droite
    a_lon, a_lat, _ = geod.fwd(opp_rwy["lon"], opp_rwy["lat"], perp_right, half_w)
    # B (TL) = extremite + half_w vers la gauche
    b_lon, b_lat, _ = geod.fwd(opp_rwy["lon"], opp_rwy["lat"], perp_left, half_w)

    return {
        "A": {"lat": a_lat, "lon": a_lon},
        "B": {"lat": b_lat, "lon": b_lon},
        "C": {"lat": c_lat, "lon": c_lon},
        "D": {"lat": d_lat, "lon": d_lon},
        "heading": heading,
        "length_m": rwy_length,
        "width_m": width,
    }


def build_db_entry(corners, alt):
    """Construit l'entree DB au format LARD (position ECEF + coordinate lat/lon/alt)."""
    entry = {}
    for key in ["A", "B", "C", "D"]:
        lat, lon = corners[key]["lat"], corners[key]["lon"]
        x, y, z = llh_to_ecef(lat, lon, alt)
        entry[key] = {
            "position": {"x": x, "y": y, "z": z},
            "coordinate": {"latitude": lat, "longitude": lon, "altitude": alt},
        }
    return entry


def main():
    parser = argparse.ArgumentParser(description="Corrige la DB pistes X-Plane depuis apt.dat")
    parser.add_argument("--apt-dat", type=str, default=None,
                        help="Chemin vers apt.dat (optionnel, utilise les coords codees en dur)")
    parser.add_argument("--compare", action="store_true",
                        help="Comparer DB originale vs corrigee")
    args = parser.parse_args()

    # Charger DB originale
    with open(ORIGINAL_DB) as f:
        original_db = json.load(f)

    # Copier la DB originale
    fixed_db = json.loads(json.dumps(original_db))

    # Corriger KPDX
    airport = "KPDX"
    print(f"Correction des pistes {airport}")
    print(f"{'='*60}")

    # Lire les altitudes depuis la DB originale
    for runway_id in KPDX_APT_DATA:
        if runway_id not in original_db.get(airport, {}):
            print(f"  [WARN] {runway_id} absent de la DB originale, skip")
            continue

        # Altitude : moyenne des coins C et D de la DB originale
        orig = original_db[airport][runway_id]
        alt_c = orig["C"]["coordinate"]["altitude"]
        alt_d = orig["D"]["coordinate"]["altitude"]
        alt = (alt_c + alt_d) / 2.0

        corners = compute_runway_corners(airport, runway_id, KPDX_APT_DATA, RUNWAY_PAIRS)

        print(f"\n  Piste {runway_id}:")
        print(f"    heading={corners['heading']:.1f}°  "
              f"length={corners['length_m']:.0f}m  width={corners['width_m']:.0f}m  alt={alt:.1f}m")

        if args.compare:
            # Comparer les centres de seuils
            geod = pyproj.Geod(ellps="WGS84")
            orig_c = orig["C"]["coordinate"]
            orig_d = orig["D"]["coordinate"]
            orig_ltp_lat = (orig_c["latitude"] + orig_d["latitude"]) / 2
            orig_ltp_lon = (orig_c["longitude"] + orig_d["longitude"]) / 2
            new_ltp_lat = (corners["C"]["lat"] + corners["D"]["lat"]) / 2
            new_ltp_lon = (corners["C"]["lon"] + corners["D"]["lon"]) / 2
            _, _, dist = geod.inv(orig_ltp_lon, orig_ltp_lat, new_ltp_lon, new_ltp_lat)
            print(f"    Ecart LTP (seuil) : {dist:.1f}m")

            # Comparer les largeurs
            orig_c_lat = orig["C"]["coordinate"]["latitude"]
            orig_d_lat = orig["D"]["coordinate"]["latitude"]
            orig_c_lon = orig["C"]["coordinate"]["longitude"]
            orig_d_lon = orig["D"]["coordinate"]["longitude"]
            _, _, orig_w = geod.inv(orig_c_lon, orig_c_lat, orig_d_lon, orig_d_lat)
            print(f"    Largeur DB originale : {orig_w:.1f}m  ->  apt.dat : {corners['width_m']:.0f}m")

        # Mettre a jour la DB
        fixed_db[airport][runway_id] = build_db_entry(corners, alt)

    # Sauver
    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DB, "w") as f:
        json.dump(fixed_db, f, indent=4)

    print(f"\n{'='*60}")
    print(f"DB corrigee sauvee dans : {OUTPUT_DB}")
    print(f"Seules les pistes KPDX ont ete corrigees.")
    print(f"Les autres aeroports sont copies tels quels de la DB originale.")


if __name__ == "__main__":
    main()
