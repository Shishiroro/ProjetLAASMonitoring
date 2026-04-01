"""
Diagnostic pluie XPPython3 : teste les combinaisons pluie/visibilite/nuages.
"""
import os
import json
import time
import sys

EXCHANGE_DIR = "C:/X-Plane 12/Resources/plugins/FlyWithLua/Scripts/lard_exchange"
CMD_FILE = os.path.join(EXCHANGE_DIR, "weather_command.json")
STS_FILE = os.path.join(EXCHANGE_DIR, "weather_status.json")

seq_counter = 0


def send(action, weather=None, wait=8):
    global seq_counter
    seq_counter += 1
    cmd = {"seq": seq_counter, "action": action}
    if weather:
        cmd["weather"] = weather
    os.makedirs(EXCHANGE_DIR, exist_ok=True)
    tmp = CMD_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cmd, f)
    for _ in range(20):
        try:
            if os.path.exists(CMD_FILE):
                os.remove(CMD_FILE)
            os.rename(tmp, CMD_FILE)
            break
        except PermissionError:
            time.sleep(0.05)

    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            with open(STS_FILE, "r") as f:
                status = json.load(f)
            if status.get("ack_seq") == seq_counter:
                if not status.get("ok"):
                    print(f"    ERREUR: {status.get('error')}")
                    return False
                break
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.05)
    else:
        print("    TIMEOUT")
        return False

    print(f"    Observe {wait}s...")
    for i in range(wait):
        time.sleep(1)
    return True


def clear():
    print("  [CLEAR]")
    send("clear_weather", wait=3)


BASE = {
    "cloud_base_msl": 5000.0,
    "cloud_top_msl": 5500.0,
    "cloud_type": 0.0,
    "cloud_coverage": 0.0,
    "precip_rate": 0.0,
    "visibility_m": 50000.0,
}

TESTS = [
    # --- PLUIE SANS NUAGES, visibilite variable ---
    ("Pluie 1.0 + vis 50km + PAS de nuages", {
        **BASE,
        "precip_rate": 1.0,
        "visibility_m": 50000.0,
    }),
    ("Pluie 1.0 + vis 10km + PAS de nuages", {
        **BASE,
        "precip_rate": 1.0,
        "visibility_m": 10000.0,
    }),
    ("Pluie 1.0 + vis 5km + PAS de nuages", {
        **BASE,
        "precip_rate": 1.0,
        "visibility_m": 5000.0,
    }),
    ("Pluie 1.0 + vis 3km + PAS de nuages", {
        **BASE,
        "precip_rate": 1.0,
        "visibility_m": 3000.0,
    }),
    ("Pluie 1.0 + vis 2km + PAS de nuages", {
        **BASE,
        "precip_rate": 1.0,
        "visibility_m": 2000.0,
    }),
    ("Pluie 1.0 + vis 1km + PAS de nuages", {
        **BASE,
        "precip_rate": 1.0,
        "visibility_m": 1000.0,
    }),

    # --- PLUIE AVEC NUAGES, vis 2km (marche deja) ---
    ("Pluie 1.0 + vis 2km + nuages overcast 100-600m", {
        **BASE,
        "cloud_base_msl": 100.0, "cloud_top_msl": 600.0,
        "cloud_type": 4.0, "cloud_coverage": 1.0,
        "precip_rate": 1.0,
        "visibility_m": 2000.0,
    }),

    # --- PLUIE AVEC NUAGES, vis plus haute ---
    ("Pluie 1.0 + vis 5km + nuages overcast 100-600m", {
        **BASE,
        "cloud_base_msl": 100.0, "cloud_top_msl": 600.0,
        "cloud_type": 4.0, "cloud_coverage": 1.0,
        "precip_rate": 1.0,
        "visibility_m": 5000.0,
    }),
    ("Pluie 1.0 + vis 10km + nuages overcast 100-600m", {
        **BASE,
        "cloud_base_msl": 100.0, "cloud_top_msl": 600.0,
        "cloud_type": 4.0, "cloud_coverage": 1.0,
        "precip_rate": 1.0,
        "visibility_m": 10000.0,
    }),

    # --- PLUIE legere avec vis basse ---
    ("Pluie 0.3 + vis 2km + nuages overcast", {
        **BASE,
        "cloud_base_msl": 100.0, "cloud_top_msl": 600.0,
        "cloud_type": 4.0, "cloud_coverage": 1.0,
        "precip_rate": 0.3,
        "visibility_m": 2000.0,
    }),
    ("Pluie 0.5 + vis 3km + nuages overcast", {
        **BASE,
        "cloud_base_msl": 100.0, "cloud_top_msl": 600.0,
        "cloud_type": 4.0, "cloud_coverage": 1.0,
        "precip_rate": 0.5,
        "visibility_m": 3000.0,
    }),

    # --- SEUIL : vis sans pluie pour comparer ---
    ("PAS de pluie + vis 2km + nuages overcast", {
        **BASE,
        "cloud_base_msl": 100.0, "cloud_top_msl": 600.0,
        "cloud_type": 4.0, "cloud_coverage": 1.0,
        "precip_rate": 0.0,
        "visibility_m": 2000.0,
    }),
]


def main():
    print("=" * 60)
    print("DIAGNOSTIC PLUIE XPPython3")
    print("=" * 60)
    print(f"\n{len(TESTS)} tests. Pour chaque, note: PLUIE? OUI/NON\n")

    print("[0] Heartbeat...")
    if not send("noop", wait=1):
        sys.exit(1)
    print("OK\n")
    clear()

    for i, (name, weather) in enumerate(TESTS, 1):
        print(f"\n[{i}/{len(TESTS)}] {name}")
        send("set_weather", weather, wait=10)
        clear()

    print("\n" + "=" * 60)
    print("Termine. Resume tes observations!")
    print("=" * 60)


if __name__ == "__main__":
    main()
