"""
Test X-Plane 12 UDP — v2 (diagnostic altitude)
================================================
Le probleme : lat/lon bougent via DREF, mais l'altitude reste au sol.
Ce script teste des approches specifiques pour resoudre le probleme d'altitude.

Lancer avec X-Plane ouvert (vol sur KPDX).
"""
import struct
import socket
import time

HOST = "127.0.0.1"
PORT = 49000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(3.0)
addr = (HOST, PORT)


def send_dref(dref, value):
    msg = b"DREF\x00"
    msg += struct.pack('<f', value)
    msg += dref.encode('utf-8').ljust(500, b'\x00')
    sock.sendto(msg, addr)


def send_cmnd(command):
    msg = b"CMND\x00"
    msg += command.encode('utf-8') + b'\x00'
    sock.sendto(msg, addr)


def send_data(index, values):
    msg = b"DATA\x00"
    msg += struct.pack('<i', index)
    padded = list(values) + [-999.0] * (8 - len(values))
    for v in padded[:8]:
        msg += struct.pack('<f', v)
    sock.sendto(msg, addr)


def read_dref(dref, idx=0):
    """Lit un dataref via RREF (subscribe 1Hz, lire, unsubscribe)."""
    msg = b"RREF\x00"
    msg += struct.pack('<ii', 1, idx)
    msg += dref.encode('utf-8').ljust(400, b'\x00')
    sock.sendto(msg, addr)
    try:
        data, _ = sock.recvfrom(1024)
        # Reponse : RREF\0 + (int idx + float val) par dataref
        if len(data) >= 13:
            val = struct.unpack('<f', data[9:13])[0]
        else:
            val = None
    except socket.timeout:
        val = None
    # Unsubscribe
    msg2 = b"RREF\x00"
    msg2 += struct.pack('<ii', 0, idx)
    msg2 += dref.encode('utf-8').ljust(400, b'\x00')
    sock.sendto(msg2, addr)
    return val


# ===== Test 0 : Connexion =====
print("=== Test 0 : Connexion UDP ===")
try:
    msg = b"RREF\x00"
    msg += struct.pack('<ii', 1, 0)
    msg += "sim/time/total_running_time_sec".encode('utf-8').ljust(400, b'\x00')
    sock.sendto(msg, addr)
    data, _ = sock.recvfrom(1024)
    print(f"  OK — X-Plane repond (recu {len(data)} bytes)")
    msg2 = b"RREF\x00"
    msg2 += struct.pack('<ii', 0, 0)
    msg2 += "sim/time/total_running_time_sec".encode('utf-8').ljust(400, b'\x00')
    sock.sendto(msg2, addr)
except socket.timeout:
    print("  ECHEC — X-Plane ne repond pas.")
    exit(1)

# Lire position actuelle
print("\n--- Position actuelle ---")
lat = read_dref("sim/flightmodel/position/latitude", 1)
lon = read_dref("sim/flightmodel/position/longitude", 2)
elev = read_dref("sim/flightmodel/position/elevation", 3)
local_y = read_dref("sim/flightmodel/position/local_y", 4)
y_agl = read_dref("sim/flightmodel/position/y_agl", 5)
print(f"  lat={lat}, lon={lon}")
print(f"  elevation={elev} m, local_y={local_y} m, y_agl={y_agl} m")

# Setup : vue propre
send_cmnd("sim/view/forward_with_nothing")
time.sleep(0.5)

input("\nAppuie Entree pour Test 1 (PAUSE + override + local_y)...")

# ===== Test 1 : Pause sim + override + local_y =====
print("\n=== Test 1 : Pause + override + local_y ===")
print("  Idee : pauser le sim pour empecher la physique de clamper l'altitude")

# Pause
send_dref("sim/time/paused", 1.0)
time.sleep(0.3)

# Override planepath
send_dref("sim/operation/override/override_planepath[0]", 1.0)
time.sleep(0.2)

# Position lat/lon via DREF (on sait que ca marche)
send_dref("sim/flightmodel/position/latitude", 45.58)
send_dref("sim/flightmodel/position/longitude", -122.59)

# Altitude via local_y (coordonnee OpenGL Y = altitude MSL en metres)
send_dref("sim/flightmodel/position/local_y", 500.0)
time.sleep(0.2)

# Zero velocites et accelerations
send_dref("sim/flightmodel/position/local_vx", 0.0)
send_dref("sim/flightmodel/position/local_vy", 0.0)
send_dref("sim/flightmodel/position/local_vz", 0.0)
send_dref("sim/flightmodel/position/local_ax", 0.0)
send_dref("sim/flightmodel/position/local_ay", 0.0)
send_dref("sim/flightmodel/position/local_az", 0.0)

# Attitude
send_dref("sim/flightmodel/position/theta", -3.0)  # pitch
send_dref("sim/flightmodel/position/phi", 0.0)      # roll
send_dref("sim/flightmodel/position/psi", 210.0)    # heading

time.sleep(1)
print("  Verif position :")
elev2 = read_dref("sim/flightmodel/position/elevation", 3)
ly2 = read_dref("sim/flightmodel/position/local_y", 4)
yagl2 = read_dref("sim/flightmodel/position/y_agl", 5)
print(f"  elevation={elev2} m, local_y={ly2} m, y_agl={yagl2} m")
print("  L'avion est-il en l'air ? (sim en pause)")

input("\nAppuie Entree pour Test 2 (POSI format officiel X-Plane 12)...")

# ===== Test 2 : POSI format officiel =====
print("\n=== Test 2 : POSI officiel (int32 + 3 doubles + 4 floats) ===")
print("  Format : POSI\\0 + aircraft(int32) + lat(d) + lon(d) + alt_msl_m(d)")
print("           + heading(f) + pitch(f) + roll(f) + gear(f)")

# S'assurer que le sim est en pause et override actif
send_dref("sim/time/paused", 1.0)
send_dref("sim/operation/override/override_planepath[0]", 1.0)
time.sleep(0.2)

msg = b"POSI\x00"
msg += struct.pack('<i', 0)          # aircraft 0 (int32, 4 bytes)
msg += struct.pack('<3d', 45.57, -122.61, 450.0)  # lat, lon, alt MSL (doubles)
msg += struct.pack('<4f', 210.0, -3.0, 0.0, 0.0)  # heading, pitch, roll, gear (floats)
print(f"  Taille paquet : {len(msg)} bytes (attendu: 49)")
sock.sendto(msg, addr)

time.sleep(1)
elev3 = read_dref("sim/flightmodel/position/elevation", 3)
ly3 = read_dref("sim/flightmodel/position/local_y", 4)
print(f"  elevation={elev3} m, local_y={ly3} m")
print("  L'avion est-il en l'air ?")

input("\nAppuie Entree pour Test 3 (POSI format XPlaneConnect)...")

# ===== Test 3 : POSI format XPlaneConnect (byte + 7 doubles) =====
print("\n=== Test 3 : POSI XPlaneConnect (byte + 7 doubles) ===")

send_dref("sim/time/paused", 1.0)
send_dref("sim/operation/override/override_planepath[0]", 1.0)
time.sleep(0.2)

msg = b"POSI\x00"
msg += struct.pack('<b', 0)  # aircraft 0 (1 byte)
# lat, lon, alt_msl_m, pitch, roll, heading, gear — all doubles
msg += struct.pack('<7d', 45.56, -122.62, 400.0, -3.0, 0.0, 210.0, 0.0)
print(f"  Taille paquet : {len(msg)} bytes (attendu: 62)")
sock.sendto(msg, addr)

time.sleep(1)
elev4 = read_dref("sim/flightmodel/position/elevation", 3)
ly4 = read_dref("sim/flightmodel/position/local_y", 4)
print(f"  elevation={elev4} m, local_y={ly4} m")
print("  L'avion est-il en l'air ?")

input("\nAppuie Entree pour Test 4 (boucle continue local_y)...")

# ===== Test 4 : Forcer local_y en boucle (30 Hz pendant 5 sec) =====
print("\n=== Test 4 : Forcer local_y en boucle (5 sec a 30 Hz) ===")
print("  Idee : renvoyer local_y en continu pour vaincre le ground clamping")

send_dref("sim/time/paused", 0.0)  # Depause pour voir si ca tient
send_dref("sim/operation/override/override_planepath[0]", 1.0)
time.sleep(0.2)

target_y = 500.0  # metres MSL
for i in range(150):  # 5 sec @ 30 Hz
    send_dref("sim/flightmodel/position/local_y", target_y)
    send_dref("sim/flightmodel/position/local_vy", 0.0)
    send_dref("sim/flightmodel/position/latitude", 45.58)
    send_dref("sim/flightmodel/position/longitude", -122.59)
    time.sleep(1/30)

elev5 = read_dref("sim/flightmodel/position/elevation", 3)
ly5 = read_dref("sim/flightmodel/position/local_y", 4)
print(f"  elevation={elev5} m, local_y={ly5} m")
print("  L'avion est-il en l'air (sim depause) ?")

input("\nAppuie Entree pour Test 5 (VEHL + local coords)...")

# ===== Test 5 : VEHL (Vehicle Location, X-Plane 12+) =====
print("\n=== Test 5 : VEHL packet (X-Plane 12 nouveau format) ===")
print("  Format XP12 : VEHL\\0 + int32(aircraft) + double(lat) + double(lon)")
print("                + double(alt_msl_m) + float(psi) + float(the) + float(phi)")

send_dref("sim/time/paused", 1.0)
send_dref("sim/operation/override/override_planepath[0]", 1.0)
time.sleep(0.2)

# VEHL est le nouveau format XP12 pour positionner un vehicule
msg = b"VEHL\x00"
msg += struct.pack('<i', 0)  # aircraft index (int32)
msg += struct.pack('<3d', 45.59, -122.60, 500.0)  # lat, lon, alt MSL (doubles)
msg += struct.pack('<3f', 210.0, -3.0, 0.0)  # heading, pitch, roll (floats)
print(f"  Taille paquet : {len(msg)} bytes")
sock.sendto(msg, addr)

time.sleep(1)
elev6 = read_dref("sim/flightmodel/position/elevation", 3)
ly6 = read_dref("sim/flightmodel/position/local_y", 4)
print(f"  elevation={elev6} m, local_y={ly6} m")
print("  L'avion est-il en l'air ?")

input("\nAppuie Entree pour Test 6 (override_gearbrake + POSI)...")

# ===== Test 6 : Desactiver freins/roues + POSI =====
print("\n=== Test 6 : Override gear + POSI ===")
print("  Idee : X-Plane garde l'avion au sol a cause du train d'atterrissage")

send_dref("sim/time/paused", 1.0)
send_dref("sim/operation/override/override_planepath[0]", 1.0)
send_dref("sim/operation/override/override_gearbrake", 1.0)
# Rentrer le train
send_dref("sim/cockpit2/controls/gear_handle_down", 0.0)
time.sleep(0.3)

# POSI officiel
msg = b"POSI\x00"
msg += struct.pack('<i', 0)
msg += struct.pack('<3d', 45.58, -122.59, 500.0)
msg += struct.pack('<4f', 210.0, -3.0, 0.0, 0.0)  # heading, pitch, roll, gear=0 (rentre)
sock.sendto(msg, addr)

time.sleep(1)
elev7 = read_dref("sim/flightmodel/position/elevation", 3)
ly7 = read_dref("sim/flightmodel/position/local_y", 4)
print(f"  elevation={elev7} m, local_y={ly7} m")
print("  L'avion est-il en l'air ?")

# ===== Cleanup =====
input("\nAppuie Entree pour nettoyer et quitter...")
send_dref("sim/time/paused", 0.0)
send_dref("sim/operation/override/override_planepath[0]", 0.0)
send_dref("sim/operation/override/override_flight_control", 0.0)
send_dref("sim/operation/override/override_gearbrake", 0.0)
print("Overrides desactives. Termine.")
sock.close()
