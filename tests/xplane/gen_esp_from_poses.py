"""
Genere un .esp (Google Earth Studio) a partir d'un poses.json existant.
Permet de comparer le rendu X-Plane vs GES sur le meme scenario.

Usage : python tests/xplane/gen_esp_from_poses.py runs/KPDX_28R_005
"""
import sys
import os
import json
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LARD_ROOT = PROJECT_ROOT / "LARD"

if str(LARD_ROOT) not in sys.path:
    sys.path.insert(0, str(LARD_ROOT))
if str(PROJECT_ROOT / "project" / "export") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "project" / "export"))

os.chdir(str(LARD_ROOT))

from src.geo.geo_dataset import GEODataset

# --- Args ---
if len(sys.argv) < 2:
    print("Usage : python tests/xplane/gen_esp_from_poses.py <run_dir>")
    print("Ex    : python tests/xplane/gen_esp_from_poses.py runs/KPDX_28R_005")
    sys.exit(1)

run_dir = PROJECT_ROOT / sys.argv[1]
poses_file = run_dir / "poses.json"
yaml_file = list(run_dir.glob("*.yaml"))[0]

if not poses_file.exists():
    print(f"ERREUR : {poses_file} introuvable")
    sys.exit(1)

# --- Charger poses ---
with open(poses_file) as f:
    data = json.load(f)

name = data["scenario_name"]
fps = data["fps"]
poses = data["poses"]
n = len(poses)

# Convertir en flight_data (lon, lat, alt, yaw, pitch_ges, roll)
flight_data = []
for p in poses:
    flight_data.append((
        p["lon"], p["lat"], p["alt_m"],
        p["heading"], p["pitch_ges"], p["roll"],
    ))

print(f"Scenario : {name}")
print(f"Frames   : {n}")
print(f"FPS      : {fps}")

# --- Generer .esp via LARD ---
dataset = GEODataset(str(run_dir), name)

# Times (simple, meme format que lard_bridge)
import random
base = {
    'year': 2023, 'month': 4, 'day': 12,
    'hour': 13, 'minute': 27, 'second': 21,
}
times = [dict(base) for _ in range(n)]

scenario, _ = dataset.create_scenario(
    flight_data,
    fov_vertical=30.0,
    width=1024,
    height=1024,
    nb_frames=n,
    fps=fps,
    times=times,
)

esp_file = run_dir / f"{name}.esp"
with open(esp_file, "w") as f:
    json.dump(scenario, f, indent=2)

print(f"\n.esp -> {esp_file}")
print("Importe ce fichier dans Google Earth Studio pour comparer avec X-Plane.")
