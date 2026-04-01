"""
Test XPPython3 weather plugin (PI_lard_weather.py).
Envoie une commande meteo via JSON et verifie la reponse.

Usage:
  1. Lancer X-Plane 12 (avec XPPython3 + PI_lard_weather.py installe)
  2. python tests/xplane/test_weather_xppython3.py
"""
import os
import json
import time
import sys

EXCHANGE_DIR = "C:/X-Plane 12/Resources/plugins/FlyWithLua/Scripts/lard_exchange"
CMD_FILE = os.path.join(EXCHANGE_DIR, "weather_command.json")
STS_FILE = os.path.join(EXCHANGE_DIR, "weather_status.json")


def send_command(seq, action, weather=None):
    """Ecrit une commande JSON et attend l'ack."""
    cmd = {"seq": seq, "action": action}
    if weather:
        cmd["weather"] = weather

    os.makedirs(EXCHANGE_DIR, exist_ok=True)

    # Ecriture atomique
    tmp = CMD_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cmd, f)
    if os.path.exists(CMD_FILE):
        os.remove(CMD_FILE)
    os.rename(tmp, CMD_FILE)

    print(f"  Commande envoyee: seq={seq}, action={action}")

    # Attendre ack
    deadline = time.time() + 10.0
    while time.time() < deadline:
        try:
            with open(STS_FILE, "r") as f:
                status = json.load(f)
            if status.get("ack_seq") == seq:
                print(f"  Reponse: {status}")
                return status
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.05)

    print("  TIMEOUT — pas de reponse du plugin (XPPython3 charge?)")
    return None


def main():
    print("=" * 60)
    print("Test XPPython3 Weather Plugin")
    print("=" * 60)

    # Nettoyage
    for f in [CMD_FILE, STS_FILE]:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

    # Test 1 : heartbeat (noop)
    print("\n[1] Heartbeat (noop)...")
    result = send_command(1, "noop")
    if not result:
        print("ECHEC: le plugin ne repond pas.")
        print("Verifier que XPPython3 est installe et PI_lard_weather.py dans PythonPlugins/")
        sys.exit(1)
    print("OK — plugin actif!\n")

    # Test 2 : meteo lourde (nuages bas + pluie)
    print("[2] Injection meteo: nuages bas 100-600m + pluie forte...")
    result = send_command(2, "set_weather", weather={
        "cloud_base_msl": 100.0,   # base nuages 100m MSL
        "cloud_top_msl": 600.0,    # sommet 600m MSL
        "cloud_type": 4,           # overcast
        "cloud_coverage": 1.0,     # couverture totale
        "precip_rate": 0.8,        # pluie forte
        "visibility_m": 3000.0,    # visibilite 3km
    })
    if result and result.get("ok"):
        print("OK — meteo injectee!")
    else:
        print(f"ECHEC: {result}")

    print("\nRegarde X-Plane — tu devrais voir nuages + pluie IMMEDIATEMENT.")
    print("Attente 30s pour observer...")
    for i in range(30):
        time.sleep(1)
        if i % 10 == 0:
            print(f"  {i}s...")

    # Test 3 : clear weather
    print("\n[3] Reset meteo (ciel clair)...")
    result = send_command(3, "clear_weather")
    if result and result.get("ok"):
        print("OK — ciel clair!")
    else:
        print(f"ECHEC: {result}")

    print("\nLe ciel devrait redevenir clair.")
    print("Attente 10s...")
    time.sleep(10)

    print("\n" + "=" * 60)
    print("Test termine.")
    print("=" * 60)


if __name__ == "__main__":
    main()
