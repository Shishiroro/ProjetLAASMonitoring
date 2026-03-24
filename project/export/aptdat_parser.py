"""
Parser apt.dat X-Plane pour extraire les coordonnees exactes des pistes.

Lit le fichier apt.dat de X-Plane 12 et retourne les 4 coins de chaque piste
au format compatible avec les DB LARD (A/TR, B/TL, C/BL, D/BR).

Format apt.dat ligne 100 (runway) :
  100 width surface ... rwy1 lat1 lon1 disp1 ... rwy2 lat2 lon2 disp2 ...

Reference : https://developer.x-plane.com/article/airport-data-apt-dat-file-format-specification/
"""

import math
from pathlib import Path

# Chemin par defaut vers l'apt.dat global de X-Plane 12
DEFAULT_XPLANE_PATH = Path("C:/X-Plane 12")
DEFAULT_APT_DAT = (
    DEFAULT_XPLANE_PATH
    / "Global Scenery"
    / "Global Airports"
    / "Earth nav data"
    / "apt.dat"
)


def parse_airport_runways(airport_icao, apt_dat_path=None):
    """Parse apt.dat et extrait les pistes d'un aeroport.

    :param airport_icao: code ICAO (ex: "KPDX")
    :param apt_dat_path: chemin vers apt.dat (defaut: X-Plane 12 global)
    :return: dict {runway_name: {lat, lon, width, disp_threshold, other_end: ...}}
    """
    apt_dat_path = Path(apt_dat_path) if apt_dat_path else DEFAULT_APT_DAT
    if not apt_dat_path.exists():
        raise FileNotFoundError(f"apt.dat introuvable : {apt_dat_path}")

    found_airport = False
    runways = {}

    with open(apt_dat_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if not parts:
                continue

            code = parts[0]

            # Debut d'un aeroport (code 1 = land, 16 = seaplane, 17 = heliport)
            if code in ("1", "16", "17"):
                if found_airport:
                    # On a depasse notre aeroport, stop
                    break
                if len(parts) >= 5 and parts[4] == airport_icao:
                    found_airport = True
                continue

            # Ligne runway (code 100)
            if code == "100" and found_airport:
                # Format: 100 width surface shoulder smoothness centerline
                #         edge signs rwy1 lat1 lon1 disp1 blast1 marking1 app1 tdz1
                #         rwy2 lat2 lon2 disp2 blast2 marking2 app2 tdz2
                if len(parts) < 22:
                    continue
                width = float(parts[1])

                rwy1_name = parts[8].strip()
                rwy1_lat = float(parts[9])
                rwy1_lon = float(parts[10])
                rwy1_disp = float(parts[11])  # displaced threshold (m)

                rwy2_name = parts[17].strip()
                rwy2_lat = float(parts[18])
                rwy2_lon = float(parts[19])
                rwy2_disp = float(parts[20])

                runways[rwy1_name] = {
                    "lat": rwy1_lat,
                    "lon": rwy1_lon,
                    "width": width,
                    "disp_threshold_m": rwy1_disp,
                    "other_end": rwy2_name,
                    "other_lat": rwy2_lat,
                    "other_lon": rwy2_lon,
                }
                runways[rwy2_name] = {
                    "lat": rwy2_lat,
                    "lon": rwy2_lon,
                    "width": width,
                    "disp_threshold_m": rwy2_disp,
                    "other_end": rwy1_name,
                    "other_lat": rwy1_lat,
                    "other_lon": rwy1_lon,
                }

    if not found_airport:
        raise ValueError(f"Aeroport {airport_icao} introuvable dans {apt_dat_path}")

    return runways


def runway_corners(rwy_data):
    """Calcule les 4 coins d'une piste a partir du centre seuil + largeur.

    Convention LARD :
      - A(TR) = far end, droite   (= other end + width/2 a droite)
      - B(TL) = far end, gauche   (= other end + width/2 a gauche)
      - C(BL) = seuil, gauche     (= threshold + width/2 a gauche)
      - D(BR) = seuil, droite     (= threshold + width/2 a droite)

    "Droite" et "gauche" vus depuis le seuil regardant vers le far end.

    :param rwy_data: dict depuis parse_airport_runways (une piste)
    :return: dict {A: {lat, lon}, B: {lat, lon}, C: {lat, lon}, D: {lat, lon},
                   heading, width, threshold: {lat, lon}, far_end: {lat, lon}}
    """
    lat1 = rwy_data["lat"]       # debut physique piste
    lon1 = rwy_data["lon"]
    lat2 = rwy_data["other_lat"]  # far end physique
    lon2 = rwy_data["other_lon"]
    width = rwy_data["width"]
    disp = rwy_data.get("disp_threshold_m", 0.0)  # seuil deplace (m)

    # Heading du seuil vers le far end
    dlat = lat2 - lat1
    dlon = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    heading = math.degrees(math.atan2(dlon, dlat)) % 360

    # Avancer le seuil de disp metres si displaced threshold
    if disp > 0:
        h_rad = math.radians(heading)
        lat1 = lat1 + disp * math.cos(h_rad) / 111320.0
        lon1 = lon1 + disp * math.sin(h_rad) / (
            111320.0 * math.cos(math.radians(lat1))
        )

    # Vecteur perpendiculaire (vers la droite vu depuis le seuil)
    perp_heading = (heading + 90) % 360
    perp_rad = math.radians(perp_heading)

    half_w = width / 2.0
    # Offset en degres pour half_w metres
    dlat_perp = half_w * math.cos(perp_rad) / 111320.0
    dlon_perp = half_w * math.sin(perp_rad) / (
        111320.0 * math.cos(math.radians((lat1 + lat2) / 2))
    )

    # 4 coins
    # C(BL) = seuil - perp (gauche)
    c_lat = lat1 - dlat_perp
    c_lon = lon1 - dlon_perp
    # D(BR) = seuil + perp (droite)
    d_lat = lat1 + dlat_perp
    d_lon = lon1 + dlon_perp
    # B(TL) = far end - perp (gauche)
    b_lat = lat2 - dlat_perp
    b_lon = lon2 - dlon_perp
    # A(TR) = far end + perp (droite)
    a_lat = lat2 + dlat_perp
    a_lon = lon2 + dlon_perp

    return {
        "A": {"lat": a_lat, "lon": a_lon},  # TR = far right
        "B": {"lat": b_lat, "lon": b_lon},  # TL = far left
        "C": {"lat": c_lat, "lon": c_lon},  # BL = near left
        "D": {"lat": d_lat, "lon": d_lon},  # BR = near right
        "heading": heading,
        "width": width,
        "threshold": {"lat": lat1, "lon": lon1},
        "far_end": {"lat": lat2, "lon": lon2},
    }


def get_xplane_runway_geometry(airport, runway, apt_dat_path=None):
    """Retourne la geometrie de piste depuis apt.dat, au format attendu par
    le pipeline (ltp_lat, ltp_lon, ltp_alt, heading, back_azimuth).

    :param airport: code ICAO (ex: "KPDX")
    :param runway: nom de piste (ex: "28L", "21")
    :return: dict compatible avec get_runway_geometry()
    """
    runways = parse_airport_runways(airport, apt_dat_path)

    # Normaliser le nom (apt.dat peut avoir "3" ou "03")
    rwy_data = None
    for name, data in runways.items():
        if name.lstrip("0") == runway.lstrip("0") or name == runway:
            rwy_data = data
            break

    if rwy_data is None:
        available = ", ".join(sorted(runways.keys()))
        raise ValueError(
            f"Piste {runway} introuvable pour {airport}. "
            f"Disponibles : {available}"
        )

    corners = runway_corners(rwy_data)

    heading = corners["heading"]
    back_azimuth = (heading + 180) % 360

    # Altitude : on ne l'a pas dans apt.dat (elevation aeroport seulement).
    # On utilise l'elevation de l'aeroport comme approximation.
    # Pour une precision parfaite, on pourrait lire l'elevation terrain X-Plane.
    # L'elevation de l'aeroport est dans la ligne "1" : parts[1]
    apt_dat_path = Path(apt_dat_path) if apt_dat_path else DEFAULT_APT_DAT
    elev_ft = _read_airport_elevation(airport, apt_dat_path)
    ltp_alt = elev_ft * 0.3048  # feet -> metres

    return {
        "ltp_lat": rwy_data["lat"],
        "ltp_lon": rwy_data["lon"],
        "ltp_alt": ltp_alt,
        "runway_heading_deg": heading,
        "runway_back_azimuth_deg": back_azimuth,
        "corners": corners,
    }


def _read_airport_elevation(airport_icao, apt_dat_path):
    """Lit l'elevation (ft) d'un aeroport depuis apt.dat."""
    with open(apt_dat_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.split()
            if (
                parts
                and parts[0] == "1"
                and len(parts) >= 5
                and parts[4] == airport_icao
            ):
                return float(parts[1])
    return 0.0


def build_lard_db_from_aptdat(airport_icao, apt_dat_path=None):
    """Construit un dict au format DB LARD pour un aeroport depuis apt.dat.

    Peut servir de remplacement direct pour runways_db_V2_XPlane.json.

    :return: dict {airport: {runway: {A: {coordinate: ...}, B: ..., C: ..., D: ...}}}
    """
    runways = parse_airport_runways(airport_icao, apt_dat_path)
    apt_dat_path = Path(apt_dat_path) if apt_dat_path else DEFAULT_APT_DAT
    elev_ft = _read_airport_elevation(airport_icao, apt_dat_path)
    elev_m = elev_ft * 0.3048

    db = {airport_icao: {}}

    for name, data in runways.items():
        corners = runway_corners(data)
        entry = {}
        for corner_name, corner_key in [("A", "A"), ("B", "B"), ("C", "C"), ("D", "D")]:
            entry[corner_name] = {
                "coordinate": {
                    "latitude": corners[corner_key]["lat"],
                    "longitude": corners[corner_key]["lon"],
                    "altitude": elev_m,
                },
            }
        db[airport_icao][name] = entry

    return db


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    icao = sys.argv[1] if len(sys.argv) > 1 else "KPDX"
    runways = parse_airport_runways(icao)
    print(f"=== {icao} : {len(runways)} pistes ===")
    for name, data in sorted(runways.items()):
        corners = runway_corners(data)
        print(f"  {name:4s}: seuil=({data['lat']:.7f}, {data['lon']:.7f}) "
              f"width={data['width']:.0f}m heading={corners['heading']:.1f}")

    # Comparer avec la DB LARD
    db_path = Path(__file__).resolve().parent.parent.parent / "LARD" / "data" / "runways_db_V2_XPlane.json"
    if db_path.exists():
        db = json.load(open(db_path))
        if icao in db:
            print(f"\n=== Comparaison avec DB LARD ===")
            import math
            for name in sorted(runways.keys()):
                if name not in db[icao]:
                    print(f"  {name}: PAS DANS LA DB LARD")
                    continue
                rwy_db = db[icao][name]
                apt = runways[name]
                # LTP DB = midpoint(C,D)
                ltp_lat = (rwy_db['C']['coordinate']['latitude'] + rwy_db['D']['coordinate']['latitude']) / 2
                ltp_lon = (rwy_db['C']['coordinate']['longitude'] + rwy_db['D']['coordinate']['longitude']) / 2
                dlat = (ltp_lat - apt['lat']) * 111320
                dlon = (ltp_lon - apt['lon']) * 111320 * math.cos(math.radians(apt['lat']))
                dist = math.sqrt(dlat**2 + dlon**2)
                print(f"  {name}: ecart seuil = {dist:.1f}m")
