"""
test_weather_v2.py — Test meteo XP12 avec corrections API
==========================================================
Teste l'injection meteo per-scenario via PI_lard_weather.py (v2).

Corrections appliquees :
  - cloud_type : nouveau enum (0=Cirrus, 1=Stratus, 2=Cumulus, 3=Cumulonimbus)
  - use_real_weather_bool → change_mode dataref
  - isIncremental=0 pour clear propre
  - radius_nm et max_altitude_msl_ft toujours fixes (jamais 0)
  - Pas de hack datarefs rain_percent
  - Attente 60s stabilisation nuages avant validation visuelle

Usage:
  1. Lancer X-Plane 12 (avec XPPython3 + PI_lard_weather.py v2)
  2. Se positionner sur un aeroport (menu Developer > Set Position)
  3. python tests/xplane/test_weather_v2.py
"""
import os
import json
import time
import sys

EXCHANGE_DIR = "C:/X-Plane 12/Resources/plugins/FlyWithLua/Scripts/lard_exchange"
CMD_FILE = os.path.join(EXCHANGE_DIR, "weather_command.json")
STS_FILE = os.path.join(EXCHANGE_DIR, "weather_status.json")

_seq = 0


def send(action, weather=None, timeout=10.0):
    """Envoie une commande au plugin et attend l'ack."""
    global _seq
    _seq += 1

    cmd = {"seq": _seq, "action": action}
    if weather:
        cmd["weather"] = weather

    os.makedirs(EXCHANGE_DIR, exist_ok=True)

    tmp = CMD_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cmd, f, indent=2)
    for _ in range(20):
        try:
            if os.path.exists(CMD_FILE):
                os.remove(CMD_FILE)
            os.rename(tmp, CMD_FILE)
            break
        except PermissionError:
            time.sleep(0.05)

    print(f"  -> seq={_seq} action={action}", end="")
    if weather:
        keys = ", ".join(f"{k}={v}" for k, v in weather.items())
        print(f" [{keys}]", end="")
    print()

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with open(STS_FILE, "r") as f:
                status = json.load(f)
            if status.get("ack_seq") == _seq:
                ok = status.get("ok", False)
                tag = "OK" if ok else "ERREUR"
                err = f" — {status.get('error')}" if not ok else ""
                print(f"  <- {tag}{err}")
                return status
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        time.sleep(0.1)

    print("  <- TIMEOUT (plugin ne repond pas)")
    return None


def wait(seconds, msg=""):
    """Attente avec compteur."""
    if msg:
        print(f"\n  {msg}")
    for i in range(seconds):
        remaining = seconds - i
        if remaining % 10 == 0 or remaining <= 5:
            print(f"  ... {remaining}s")
        time.sleep(1)


def cleanup():
    """Nettoie les fichiers d'echange."""
    for f in [CMD_FILE, STS_FILE]:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------
# Scenarios de test
# ---------------------------------------------------------------

def test_heartbeat():
    print("\n" + "=" * 50)
    print("TEST 0 : Heartbeat")
    print("=" * 50)
    result = send("noop")
    if not result or not result.get("ok"):
        print("\nECHEC : plugin ne repond pas.")
        print("Verifier :")
        print("  1. XPPython3 installe dans Resources/plugins/XPPython3/")
        print("  2. PI_lard_weather.py dans Resources/plugins/PythonPlugins/")
        print("  3. Menu Plugins > XPPython3 > Reload Scripts")
        return False
    print("Plugin actif.")
    return True


def test_clear():
    """Test clear propre — baseline."""
    print("\n" + "=" * 50)
    print("TEST 1 : Clear weather (baseline)")
    print("=" * 50)
    send("clear_weather")
    wait(5, "Le ciel devrait etre clair (ou meteo par defaut).")
    input("  [Entree] pour continuer...")


def test_rain():
    """Test pluie — severity 0.8, cumulonimbus, vis 5km."""
    print("\n" + "=" * 50)
    print("TEST 2 : Pluie forte (per-scenario)")
    print("=" * 50)
    print("  Injection pluie severity=0.8...")
    send("set_weather", weather={
        "precip_rate": 0.8,
        "visibility_m": 5000.0,
        # cloud_type 3 = Cumulonimbus (nouveau enum XPLMWeather)
        "cloud_type": 3.0,
        "cloud_coverage": 1.0,
        "cloud_base_msl": 800.0,
        "cloud_top_msl": 3000.0,
        "radius_nm": 50.0,
        "max_alt_ft": 30000.0,
    })
    wait(65, "Attente 65s stabilisation (nuages GPU ~60s)...")
    print("  Tu devrais voir : pluie + nuages sombres + visibilite reduite.")
    input("  [Entree] pour continuer...")


def test_visibility():
    """Test brouillard — vis 1km."""
    print("\n" + "=" * 50)
    print("TEST 3 : Brouillard dense (vis=1km)")
    print("=" * 50)
    # D'abord clear
    send("clear_weather")
    wait(5)
    # Puis brouillard
    send("set_weather", weather={
        "visibility_m": 1000.0,
        "precip_rate": 0.0,
        "cloud_type": 0.0,
        "cloud_coverage": 0.0,
        "radius_nm": 50.0,
        "max_alt_ft": 30000.0,
    })
    wait(10, "Le brouillard devrait etre quasi-instantane (pas de nuages).")
    print("  Tu devrais voir : horizon bouche, sol visible de pres.")
    input("  [Entree] pour continuer...")


def test_overcast():
    """Test nuages stratus — couverture totale."""
    print("\n" + "=" * 50)
    print("TEST 4 : Nuages stratus (couvert)")
    print("=" * 50)
    send("clear_weather")
    wait(5)
    send("set_weather", weather={
        "visibility_m": 30000.0,
        "precip_rate": 0.0,
        # cloud_type 1 = Stratus (nouveau enum)
        "cloud_type": 1.0,
        "cloud_coverage": 1.0,
        "cloud_base_msl": 1500.0,
        "cloud_top_msl": 3000.0,
        "radius_nm": 50.0,
        "max_alt_ft": 30000.0,
    })
    wait(65, "Attente 65s stabilisation nuages...")
    print("  Tu devrais voir : couche de stratus uniforme au-dessus.")
    input("  [Entree] pour continuer...")


def test_snow():
    """Test neige — temperature negative + precipitation."""
    print("\n" + "=" * 50)
    print("TEST 5 : Neige (temp < 0 + precip)")
    print("=" * 50)
    send("clear_weather")
    wait(5)
    send("set_weather", weather={
        "precip_rate": 0.6,
        "visibility_m": 5000.0,
        "temperature_c": -5.0,
        "cloud_type": 1.0,
        "cloud_coverage": 1.0,
        "cloud_base_msl": 1000.0,
        "cloud_top_msl": 3000.0,
        "radius_nm": 50.0,
        "max_alt_ft": 30000.0,
    })
    wait(65, "Attente 65s stabilisation...")
    print("  Si temp < 0, XP12 devrait rendre de la neige au lieu de la pluie.")
    print("  (Depend de l'altitude et de la temperature.)")
    input("  [Entree] pour continuer...")


def test_clear_after_rain():
    """Test critique : clear apres pluie — le bug actuel."""
    print("\n" + "=" * 50)
    print("TEST 6 : Clear apres pluie (test du bug)")
    print("=" * 50)
    print("  Ce test verifie que la pluie disparait bien apres clear.")

    # Pluie
    send("set_weather", weather={
        "precip_rate": 1.0,
        "visibility_m": 3000.0,
        "cloud_type": 3.0,
        "cloud_coverage": 1.0,
        "cloud_base_msl": 800.0,
        "cloud_top_msl": 3000.0,
        "radius_nm": 50.0,
        "max_alt_ft": 30000.0,
    })
    wait(65, "Attente 65s — pluie forte...")
    print("  Pluie visible ?")
    input("  [Entree] pour lancer le clear...")

    # Clear
    send("clear_weather")
    wait(65, "Attente 65s apres clear...")
    print("  La pluie et les nuages ont disparu ?")
    answer = input("  [o/n] > ").strip().lower()
    if answer == "o":
        print("  SUCCES — le clear fonctionne!")
    else:
        print("  ECHEC — le bug persiste, investiguer les logs XP.")

    # Rejouer la pluie pour verifier reproductibilite
    print("\n  Re-injection pluie (test reproductibilite)...")
    send("set_weather", weather={
        "precip_rate": 0.8,
        "visibility_m": 5000.0,
        "cloud_type": 3.0,
        "cloud_coverage": 1.0,
        "cloud_base_msl": 800.0,
        "cloud_top_msl": 3000.0,
        "radius_nm": 50.0,
        "max_alt_ft": 30000.0,
    })
    wait(65, "Attente 65s — pluie devrait revenir...")
    print("  Pluie de retour ?")
    input("  [Entree] pour finir...")


def test_chain():
    """Enchaine 3 scenarios differents sans relancer X-Plane."""
    print("\n" + "=" * 50)
    print("TEST 7 : Chaine de 3 scenarios (simule pipeline)")
    print("=" * 50)

    scenarios = [
        ("Scenario A : pluie moderee", {
            "precip_rate": 0.5, "visibility_m": 8000.0,
            "cloud_type": 2.0, "cloud_coverage": 0.7,
            "cloud_base_msl": 1000.0, "cloud_top_msl": 2500.0,
            "radius_nm": 50.0, "max_alt_ft": 30000.0,
        }),
        ("Scenario B : brouillard pur", {
            "precip_rate": 0.0, "visibility_m": 800.0,
            "cloud_type": 0.0, "cloud_coverage": 0.0,
            "radius_nm": 50.0, "max_alt_ft": 30000.0,
        }),
        ("Scenario C : ciel clair", None),
    ]

    for name, weather in scenarios:
        print(f"\n  --- {name} ---")
        if weather:
            send("clear_weather")
            wait(5)
            send("set_weather", weather=weather)
            wait(65, "Stabilisation 65s...")
        else:
            send("clear_weather")
            wait(65, "Clear + stabilisation 65s...")
        print(f"  Resultat visuel correct pour '{name}' ?")
        input("  [Entree] pour passer au suivant...")

    print("\n  Chaine terminee.")


def test_cloud_timing():
    """Mesure le delai reel d'apparition des nuages avec updateImmediately=True.

    Capture un screenshot toutes les 5s pendant 90s pour observer
    quand les nuages apparaissent visuellement.
    """
    print("\n" + "=" * 50)
    print("TEST 8 : Timing apparition nuages")
    print("=" * 50)

    output_dir = os.path.join(os.path.dirname(__file__), "cloud_timing")
    os.makedirs(output_dir, exist_ok=True)

    # Capturer un screenshot via mss
    try:
        import mss
        sct = mss.mss()
    except ImportError:
        print("  ERREUR: pip install mss requis pour ce test")
        return

    def capture(tag):
        """Capture l'ecran et sauve avec un tag."""
        img = sct.grab(sct.monitors[1])
        path = os.path.join(output_dir, f"{tag}.png")
        mss.tools.to_png(img.rgb, img.size, output=path)
        return path

    # Baseline : ciel clair
    print("  Clear baseline...")
    send("clear_weather")
    time.sleep(10)
    capture("00_baseline")
    print("  Screenshot baseline sauve")

    # Injection nuages denses
    print("\n  Injection nuages Stratus couverture 100%...")
    t0 = time.time()
    send("set_weather", weather={
        "precip_rate": 0.0,
        "visibility_m": 30000.0,
        "cloud_type": 1.0,       # Stratus
        "cloud_coverage": 1.0,
        "cloud_base_msl": 500.0,
        "cloud_top_msl": 2000.0,
        "radius_nm": 50.0,
        "max_alt_ft": 30000.0,
    })

    # Capturer toutes les 5s pendant 90s
    print(f"\n  Capture toutes les 5s pendant 90s...")
    print(f"  Screenshots dans : {output_dir}")
    for i in range(18):  # 18 x 5s = 90s
        elapsed = time.time() - t0
        tag = f"{int(elapsed):02d}s"
        path = capture(tag)
        print(f"  {tag} — capture sauvee")
        time.sleep(5)

    elapsed_total = time.time() - t0
    print(f"\n  Total : {elapsed_total:.0f}s de captures")
    print(f"  Regarde les screenshots dans {output_dir}")
    print("  Compare baseline vs les captures pour voir QUAND les nuages apparaissent.")
    input("  [Entree] pour continuer...")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    print("=" * 50)
    print("Test Meteo XP12 v2 — Corrections API")
    print("=" * 50)
    print("Ce script teste le plugin PI_lard_weather.py corrige.")
    print()

    cleanup()

    if not test_heartbeat():
        sys.exit(1)

    tests = {
        "1": ("Clear baseline", test_clear),
        "2": ("Pluie forte", test_rain),
        "3": ("Brouillard", test_visibility),
        "4": ("Nuages stratus", test_overcast),
        "5": ("Neige", test_snow),
        "6": ("Clear apres pluie (bug)", test_clear_after_rain),
        "7": ("Chaine 3 scenarios", test_chain),
        "8": ("Timing nuages (screenshots 5s)", test_cloud_timing),
        "a": ("Tout lancer", None),
    }

    print("\nTests disponibles :")
    for k, (name, _) in tests.items():
        print(f"  {k} — {name}")

    choice = input("\nChoix (numero ou 'a' pour tout) > ").strip().lower()

    if choice == "a":
        for k, (_, fn) in tests.items():
            if fn:
                fn()
    elif choice in tests and tests[choice][1]:
        tests[choice][1]()
    else:
        print(f"Choix invalide: {choice}")
        sys.exit(1)

    # Cleanup final
    print("\n\nNettoyage final...")
    send("clear_weather")
    wait(5)
    print("\nTermine.")


if __name__ == "__main__":
    main()
