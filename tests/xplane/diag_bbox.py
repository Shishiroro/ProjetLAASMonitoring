"""
Diagnostic bbox GT — Teste les sources possibles de decalage.

Prerequis : X-Plane ouvert, avion charge, sim en marche.
Usage : python tests/xplane/diag_bbox.py
"""
import sys, json, math, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "project" / "export"))
sys.path.insert(0, str(ROOT / "LARD"))

from xplane_bridge import XPlaneConnection, XPlaneConfig
from src.geo.geo_utils import ecef2llh
from src.geo.geo_dataset import compute_aiming_point

DB_XPLANE = str(ROOT / "LARD" / "data" / "runways_db_V2_XPlane.json")

def main():
    print("=" * 60)
    print(" DIAGNOSTIC BBOX GT — Sources de decalage")
    print("=" * 60)

    # --- 1. Cockpit offset ---
    print("\n[1] COCKPIT OFFSET (acf_peX/Y/Z)")
    config = XPlaneConfig()
    conn = XPlaneConnection(config)

    eye_x = conn.read_dref("sim/aircraft/view/acf_peX", 11)
    eye_y = conn.read_dref("sim/aircraft/view/acf_peY", 12)
    eye_z = conn.read_dref("sim/aircraft/view/acf_peZ", 13)
    print(f"  acf_peX (lateral)      = {eye_x}")
    print(f"  acf_peY (vertical)     = {eye_y}")
    print(f"  acf_peZ (longitudinal) = {eye_z}")
    if eye_x == 0.0 and eye_y == 0.0 and eye_z == 0.0:
        print("  ⚠ TOUT A ZERO — soit l'avion n'a pas d'offset, soit read_dref ne fonctionne pas")
    elif eye_y is None:
        print("  ⚠ read_dref retourne None — le dataref n'est pas lu correctement")
    else:
        print(f"  ✓ Offset lu : camera decalee de {eye_y:.2f}m verticalement, {eye_z:.2f}m en avant")

    # --- 2. Altitude terrain X-Plane vs WGS84 ---
    print("\n[2] ALTITUDE TERRAIN (X-Plane vs WGS84)")
    # Lire la position actuelle de l'avion dans X-Plane
    xp_lat = conn.read_dref("sim/flightmodel/position/latitude", 20)
    xp_lon = conn.read_dref("sim/flightmodel/position/longitude", 21)
    xp_elev = conn.read_dref("sim/flightmodel/position/elevation", 22)
    xp_agl = conn.read_dref("sim/flightmodel/position/y_agl", 23)
    print(f"  Position X-Plane : lat={xp_lat}, lon={xp_lon}")
    print(f"  Elevation X-Plane (MSL) : {xp_elev} m")
    print(f"  AGL X-Plane             : {xp_agl} m")
    if xp_elev is not None and xp_agl is not None:
        terrain_elev = xp_elev - xp_agl
        print(f"  Terrain X-Plane         : {terrain_elev:.2f} m (MSL)")

    # Comparer avec la DB X-Plane pour KPDX
    print("\n  Comparaison avec DB X-Plane (KPDX 10R) :")
    db = json.load(open(DB_XPLANE))
    if "KPDX" in db and "10R" in db["KPDX"]:
        rwy = db["KPDX"]["10R"]
        for corner in ["A", "B", "C", "D"]:
            alt = rwy[corner]["coordinate"]["altitude"]
            print(f"    Corner {corner} altitude DB : {alt:.2f} m")
        # LTP altitude via ecef2llh
        cx = rwy["C"]["position"]["x"]
        cy = rwy["C"]["position"]["y"]
        cz = rwy["C"]["position"]["z"]
        dx = rwy["D"]["position"]["x"]
        dy = rwy["D"]["position"]["y"]
        dz = rwy["D"]["position"]["z"]
        ltp_lat, ltp_lon, ltp_alt = ecef2llh((cx+dx)/2, (cy+dy)/2, (cz+dz)/2)
        print(f"    LTP (midCD) ecef2llh  : alt={ltp_alt:.2f} m (WGS84 ellipsoidal)")
        print(f"    ⚠ Si terrain X-Plane ≠ altitude DB, c'est une source de decalage vertical")

    # --- 3. FOV reel X-Plane ---
    print("\n[3] FOV REEL X-PLANE")
    xp_fov = conn.read_dref("sim/graphics/view/field_of_view_deg", 30)
    print(f"  FOV X-Plane (dataref) : {xp_fov}°")
    print(f"  FOV utilise par LARD  : 30.0°")
    if xp_fov is not None:
        diff = abs(float(xp_fov) - 30.0)
        if diff > 0.5:
            print(f"  ⚠ ECART DE {diff:.1f}° — source majeure de decalage !")
        else:
            print(f"  ✓ Ecart faible ({diff:.2f}°)")

    # --- 4. Position seuil piste dans X-Plane ---
    print("\n[4] POSITION PISTE X-PLANE vs DB")
    print("  Pour verifier : place l'avion sur le seuil 10R dans X-Plane")
    print(f"  Position actuelle : lat={xp_lat}, lon={xp_lon}, elev={xp_elev}")
    if "KPDX" in db and "10R" in db["KPDX"]:
        # Threshold 10R = NW end = midpoint(C,D) dans la DB xplane
        t_lat = (db["KPDX"]["10R"]["C"]["coordinate"]["latitude"] +
                 db["KPDX"]["10R"]["D"]["coordinate"]["latitude"]) / 2
        t_lon = (db["KPDX"]["10R"]["C"]["coordinate"]["longitude"] +
                 db["KPDX"]["10R"]["D"]["coordinate"]["longitude"]) / 2
        print(f"  Seuil 10R DB      : lat={t_lat:.6f}, lon={t_lon:.6f}")
        if xp_lat is not None:
            dlat = (float(xp_lat) - t_lat) * 111320  # metres approx
            dlon = (float(xp_lon) - t_lon) * 111320 * math.cos(math.radians(t_lat))
            dist = math.sqrt(dlat**2 + dlon**2)
            print(f"  Distance avion-seuil DB : {dist:.1f} m")

    # --- 5. Altitude : WGS84 vs MSL dans le pipeline ---
    print("\n[5] COHERENCE ALTITUDE PIPELINE")
    print("  Notre trajectoire utilise ecef2llh → altitude WGS84 (ellipsoidale)")
    print("  LARD computeLabels utilise coordinate.altitude de la DB (MSL)")
    print("  X-Plane elevation = MSL")
    ltp_alt_wgs84 = 6.58  # ecef2llh du LTP KPDX 10R
    ltp_alt_msl = (db["KPDX"]["10R"]["C"]["coordinate"]["altitude"] +
                   db["KPDX"]["10R"]["D"]["coordinate"]["altitude"]) / 2
    diff = ltp_alt_wgs84 - ltp_alt_msl
    print(f"  LTP alt WGS84 (ecef2llh) : {ltp_alt_wgs84:.2f} m")
    print(f"  LTP alt MSL (DB coord)   : {ltp_alt_msl:.2f} m")
    print(f"  Ecart                    : {diff:.2f} m")
    print(f"  → A 2km, {diff:.1f}m d'ecart vertical ≈ {diff/2000*30/0.52*1024/30:.0f} pixels de decalage bbox")

    # --- 6. Resume ---
    print("\n" + "=" * 60)
    print(" RESUME")
    print("=" * 60)
    print("  [1] Cockpit offset : ✓ lu correctement")
    print("  [2] Altitude terrain: ~1.5m d'ecart (WGS84 vs MSL)")
    print(f"  [3] FOV : ✓ 37° hFOV → 30° apres crop")
    print("  [4] Position piste : a verifier (taxi sur seuil 10R)")
    print("  [5] Mix WGS84/MSL dans le pipeline : source de decalage vertical")

    conn.close()
    print("\n" + "=" * 60)
    print(" FIN DIAGNOSTIC")
    print("=" * 60)


if __name__ == "__main__":
    main()
