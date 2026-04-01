"""
Diagnostic meteo XPPython3 : teste chaque effet un par un.
Lance X-Plane + plugin PI_lard_weather, puis ce script.

Pour chaque effet, regarde X-Plane et note ce qui change visuellement.
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
    # Retry rename in case plugin has file locked
    for _ in range(20):
        try:
            if os.path.exists(CMD_FILE):
                os.remove(CMD_FILE)
            os.rename(tmp, CMD_FILE)
            break
        except PermissionError:
            time.sleep(0.05)

    # Wait for ack
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            with open(STS_FILE, "r") as f:
                status = json.load(f)
            if status.get("ack_seq") == seq_counter:
                if not status.get("ok"):
                    print(f"    ERREUR plugin: {status.get('error')}")
                    return False
                break
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.05)
    else:
        print("    TIMEOUT — plugin ne repond pas")
        return False

    print(f"    Observe X-Plane pendant {wait}s...")
    for i in range(wait):
        time.sleep(1)
    return True


def clear():
    print("  [CLEAR] Reset meteo...")
    send("clear_weather", wait=3)


# Base weather template (clear sky)
BASE = {
    "cloud_base_msl": 5000.0,
    "cloud_top_msl": 5500.0,
    "cloud_type": 0.0,
    "cloud_coverage": 0.0,
    "precip_rate": 0.0,
    "visibility_m": 50000.0,
}


TESTS = [
    # --- NUAGES ---
    ("Nuages OVERCAST bas (100-600m)", {
        **BASE,
        "cloud_base_msl": 100.0, "cloud_top_msl": 600.0,
        "cloud_type": 4.0, "cloud_coverage": 1.0,
    }),
    ("Nuages BROKEN moyen (500-1500m)", {
        **BASE,
        "cloud_base_msl": 500.0, "cloud_top_msl": 1500.0,
        "cloud_type": 3.0, "cloud_coverage": 0.7,
    }),
    ("Nuages SCATTERED haut (2000-4000m)", {
        **BASE,
        "cloud_base_msl": 2000.0, "cloud_top_msl": 4000.0,
        "cloud_type": 2.0, "cloud_coverage": 0.4,
    }),
    ("Nuages FEW tres haut (5000-8000m)", {
        **BASE,
        "cloud_base_msl": 5000.0, "cloud_top_msl": 8000.0,
        "cloud_type": 1.0, "cloud_coverage": 0.2,
    }),

    # --- PRECIPITATIONS ---
    ("Pluie legere (0.2) + overcast bas", {
        **BASE,
        "cloud_base_msl": 100.0, "cloud_top_msl": 600.0,
        "cloud_type": 4.0, "cloud_coverage": 1.0,
        "precip_rate": 0.2,
    }),
    ("Pluie forte (0.8) + overcast bas", {
        **BASE,
        "cloud_base_msl": 100.0, "cloud_top_msl": 600.0,
        "cloud_type": 4.0, "cloud_coverage": 1.0,
        "precip_rate": 0.8,
    }),
    ("Pluie MAX (1.0) + overcast bas", {
        **BASE,
        "cloud_base_msl": 100.0, "cloud_top_msl": 600.0,
        "cloud_type": 4.0, "cloud_coverage": 1.0,
        "precip_rate": 1.0,
    }),

    # --- VISIBILITE ---
    ("Visibilite 500m (brouillard) + ciel clair", {
        **BASE,
        "visibility_m": 500.0,
    }),
    ("Visibilite 1000m (brume)", {
        **BASE,
        "visibility_m": 1000.0,
    }),
    ("Visibilite 3000m", {
        **BASE,
        "visibility_m": 3000.0,
    }),
    ("Visibilite 500m + nuages bas", {
        **BASE,
        "cloud_base_msl": 50.0, "cloud_top_msl": 300.0,
        "cloud_type": 4.0, "cloud_coverage": 1.0,
        "visibility_m": 500.0,
    }),

    # --- COMBO ---
    ("COMBO: overcast + pluie + vis 2km", {
        **BASE,
        "cloud_base_msl": 100.0, "cloud_top_msl": 800.0,
        "cloud_type": 4.0, "cloud_coverage": 1.0,
        "precip_rate": 1.0,
        "visibility_m": 2000.0,
    }),
]


def main():
    print("=" * 60)
    print("DIAGNOSTIC METEO XPPython3")
    print("=" * 60)
    print(f"\n{len(TESTS)} tests a passer.")
    print("Pour chaque test, note ce que tu vois dans X-Plane.\n")

    # Heartbeat
    print("[0] Heartbeat...")
    if not send("noop", wait=1):
        print("Plugin ne repond pas. Quitte.")
        sys.exit(1)
    print("Plugin OK.\n")

    clear()

    for i, (name, weather) in enumerate(TESTS, 1):
        print(f"\n[{i}/{len(TESTS)}] {name}")
        send("set_weather", weather, wait=10)
        clear()

    print("\n" + "=" * 60)
    print("Diagnostic termine.")
    print("Note quels effets etaient visibles!")
    print("=" * 60)


if __name__ == "__main__":
    main()
