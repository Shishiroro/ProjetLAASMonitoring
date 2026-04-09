# LARD-LAAS-TAF — Claude Context

## Projet
Pipeline de generation de trajectoires d'approche realistes pour avions, decoupled de LARD.
Utilise **TAF** (Testing Automation Framework, LAAS) pour le sampling de scenarios sous contraintes,
et **LARD** (importe comme bibliotheque) pour l'export .esp/.yaml.
Deux renderers supportes : **Google Earth Studio** (GES, manuel) et **X-Plane 12** (automatise).

Developpe par : ONERA, IRT Saint Exupery, Airbus, DGA.

## Origine
Ce projet est issu de LARDv2 ou le mode sequentiel (trajectoires realistes via processus
Ornstein-Uhlenbeck) etait melange au code LARD de base + interface Streamlit.
Decision : decoupler dans un projet independant pilote par TAF + code pur (pas de Streamlit).

## Pipeline
```
[1] trajectory_template.xml    — Contraintes utilisateur (fps, distances, vitesse...)
         |
[2] TAF Generation (taf/src/) — Sample N scenarios valides (solveur z3)
         |
[3] Export.py (project/export/)— Calcul trajectoire (OU, convergence, crab angle)
         |                       + appel LARD pour generer .esp/.yaml + poses.json
         |
[4] Sorties                    — .esp (GES) ou poses.json (X-Plane) + .yaml
         |
[5] Rendu images               — GES (manuel, .esp) ou X-Plane 12 (auto, UDP)
         |
[6] Fautes capteur camera      — Alteration realiste des images (fog, dead_pixels, etc.)
         |                       Pilote par fault_profile.json (genere par TAF)
[7] YOLOv8 (yolo/predict.py)  — Detection piste sur images (embarque avion, hors ligne)
         |
[8] Labels prediction          — .txt (bbox piste) + images annotees
         |
[9] IoU evaluation             — Comparaison predictions YOLO vs ground truth LARD (.csv)
         |                       Metriques : IoU, accuracy, precision, recall
[10] Rapport                   — Resultats evaluation du modele
```

### Contexte modele embarque
Le modele YOLOv8 (.pt) est destine a etre embarque dans les avions (inference hors ligne).
L'evaluation IoU sert a valider son accuracy avant deploiement :
- **Ground truth** : bbox piste issues des .csv generes par LARD (coordonnees connues)
- **Predictions** : bbox detectees par le modele YOLOv8
- **IoU** = intersection / union des deux bbox → mesure de precision

## Stack technique
- Python, NumPy, PyProj, PyYAML, OpenCV, scikit-spatial
- TAF : z3-solver, numpy
- YOLOv8 : ultralytics
- X-Plane 12 : mss, Pillow, pywin32 (capture fenetre, DPI-aware)
- Pas de Streamlit — tout via code/CLI

## Structure du projet
```
LARD-LAAS-TAF/
├── LARD/                              # Clone vierge — NE PAS MODIFIER
├── taf/                               # Clone TAF LAAS — NE PAS MODIFIER
│   └── src/                           # Taf.py, Generator.py, etc.
├── project/                           # NOTRE CODE (generation)
│   ├── templates/
│   │   └── trajectory_template.xml    # Contraintes utilisateur (pistes, fps, distances...)
│   ├── export/
│   │   ├── Export.py                  # Point d'entree TAF (copie auto → taf/src/)
│   │   ├── trajectory_builder.py      # OU, convergence, timeline, crab angle
│   │   ├── lard_bridge.py            # Import LARD, genere .esp/.yaml + GT CSV
│   │   ├── xplane_bridge.py         # Interface X-Plane 12 (UDP + factory Lua/UDP, mss capture)
│   │   ├── xplane_weather.py         # Effets meteo X-Plane 12 (via plugin XPPython3)
│   │   ├── sensor_faults.py           # Fautes capteur hardware (22 types, post-traitement OpenCV)
│   │   ├── xplane_weather.py         # Effets meteo X-Plane 12 (via plugin XPPython3)
│   │   └── sensor_weather_faults.py  # Effets meteo capteur (TODO: pluie lentille, givre)
│   ├── settings.xml                   # Config TAF (paths, nb_test_cases)
│   └── Generate.py                    # Script lancement TAF
├── yolo/                              # Detection YOLOv8 + evaluation (tooling)
│   ├── predict.py                     # Prediction YOLO (CLI ou import)
│   ├── evaluate.py                    # Evaluation IoU preds vs GT (CLI ou import)
│   ├── generate_gt.py                 # Generation GT CSV standalone
│   ├── yolov8n.pt                     # Poids du modele
│   ├── test_images/test/              # Images d'approche (dataset test existant)
│   └── eval/                          # Module evaluation (inspire de lardv2-ml-detect)
│       ├── box.py                     # Conversions formats bbox (tlbr, tlwh, xywh)
│       ├── metrics.py                 # AP, F1, P, R, match preds↔GT
│       └── metrics_utils.py           # IoU (IOU/GIOU/DIOU/CIOU) + conversions torch
├── runs/                              # DONNEES PAR SCENARIO (auto-genere)
│   ├── LFPO_24/                       # Un dossier par scenario
│   │   ├── LFPO_24.esp               #   genere par 'generate' (GES uniquement)
│   │   ├── LFPO_24.yaml              #   genere par 'generate'
│   │   ├── poses.json                #   poses camera universel (GES + X-Plane)
│   │   ├── run_config.json           #   {"renderer": "ges"|"xplane"}
│   │   ├── params.xml                #   parametres TAF du scenario
│   │   ├── LFPO_24.zip               #   zip GES (pose manuellement, GES uniquement)
│   │   ├── fault_profile.json        #   profil fautes capteur (auto, si fautes actives)
│   │   ├── weather_profile.json      #   profil meteo X-Plane (auto, si meteo active)
│   │   ├── footage/                   #   images GES (dezip) ou X-Plane (JPEG 1280x1024)
│   │   ├── degraded/                  #   images avec fautes capteur (auto, si fautes actives)
│   │   ├── exported_images/           #   (cree par LARD, doublon de footage/, nettoye auto)
│   │   ├── predictions/               #   .txt labels YOLO (auto)
│   │   ├── annotated/                 #   images annotees YOLO (auto)
│   │   ├── *_labels.csv               #   GT LARD (auto)
│   │   └── eval_results.json          #   metriques IoU (auto)
│   └── pipeline_report.json           # Rapport agrege tous runs
├── output/                            # Temporaire TAF (nettoye avant chaque generate)
├── tests/                             # Scripts de test
│   └── xplane/                        # Tests X-Plane (diagnostic, benchmark, rendu)
├── PI_lard_weather.py                 # Plugin XPPython3 (a copier dans PythonPlugins/)
├── run_pipeline.py                    # Orchestrateur bout-en-bout
├── main.py                            # Point d'entree CLI (generation seule)
├── xplane.txt                         # Documentation integration X-Plane
├── CLAUDE.md                          # Ce fichier
└── historique.txt                     # Resume de ce qui a ete fait
```

## Pipeline CLI (run_pipeline.py)
```
# GES (defaut)
python run_pipeline.py generate -n 5      # Genere .esp/.yaml dans runs/
python run_pipeline.py evaluate LFPO_24   # Dezip + GT + YOLO + IoU sur un run
python run_pipeline.py evaluate --all     # Sur tous les runs
python run_pipeline.py full -n 5          # Generate + pause GES + evaluate

# X-Plane 12 (full auto)
python run_pipeline.py full -n 5 --renderer xplane
python run_pipeline.py generate -n 5 --renderer xplane
python run_pipeline.py evaluate --all --renderer xplane
```
Workflow :
1. `generate -n N` → cree runs/<ICAO_RWY>/ avec .esp + .yaml + params.xml
2. Importer les .esp dans GES, rendre, telecharger les .zip
3. Poser chaque .zip dans son dossier runs/<nom>/
4. `evaluate --all` → dezip + GT + fautes capteur + YOLO + IoU pour tous les runs

## Fautes capteur camera (project/export/sensor_faults.py)

### Principe
Les fautes capteur simulent des degradations realistes de la camera embarquee
(fog, dead_pixels, motion_blur, etc.) pour tester la robustesse du modele YOLOv8.
Elles sont definies dans le XML utilisateur, validees par TAF/z3, et appliquees
aux images GES avant inference YOLO.

### Configuration dans le XML
Les 22 types de fautes sont declares directement avec 3 parametres chacun :
```xml
<!-- severity min="0" max="0" = desactive -->
<parameter name="fog_severity"   type="real" min="0.3" max="0.7"/>
<parameter name="fog_from_pct"   type="real" min="20" max="40"/>
<parameter name="fog_to_pct"     type="real" min="60" max="90"/>

<constraint name="fog_pct_order"
            expressions="scenario[i]\fog_from_pct INF scenario[i]\fog_to_pct"
            quantifiers="i" ranges="[0, 0]" types="forall"/>
```
- `{type}_severity` min=max=0 desactive la faute, > 0 l'active
- `min=max` fixe une valeur, `min<max` laisse TAF sampler dans la plage
- `from_pct/to_pct` : pourcentage de progression (0%=loin, 100%=piste)
- Plusieurs fautes se cumulent si leurs plages se chevauchent
- Export.py lit les 22 types depuis KNOWN_FAULT_TYPES, filtre severity > 0

### Pipeline d'application
1. `generate` : TAF sample les params, Export.py cree `fault_profile.json`
2. `evaluate` : apres dezip GES, `step_apply_faults` lit le profil et degrade les images
3. Images degradees dans `degraded/`, originales conservees dans `footage/`
4. YOLO tourne sur `degraded/` (ou `footage/` si pas de fautes)
5. GT calculee sur les positions originales (les fautes n'affectent pas la geometrie)

### Tracabilite
- `fault_profile.json` : config fautes + resume par frame (quelles fautes, quelle severite)
- `params.xml` : parametres TAF du scenario (inclut les fault_N_* generes)
- `.yaml` : section `sensor_faults` dans les metadonnees du scenario

### 22 types de fautes disponibles
gaussian_noise, shot_noise, salt_pepper, dead_pixels, motion_blur,
defocus_blur, rolling_shutter, overexposure, underexposure, vignetting,
chromatic_aberration, radial_distortion, lens_flare, banding, jpeg_artifacts,
color_shift, condensation, dirt_on_lens, fog, zoom_blur, contrast, pixelate

## Effets meteo X-Plane 12 (project/export/xplane_weather.py + PI_lard_weather.py)

### Principe
Les effets meteo sont injectes **une fois avant le rendu** du scenario via le plugin
XPPython3 (PI_lard_weather.py v2) qui utilise l'API officielle XPLMWeather.
La meteo est per-scenario (pas per-frame) : elle ne change pas pendant les ~30s d'approche.
Contrairement aux fautes capteur (post-traitement OpenCV), la meteo modifie
la scene 3D directement. Ignores si renderer=ges.

### Architecture
- **PI_lard_weather.py** : plugin XPPython3 dans X-Plane (XPLMWeather API v2)
- **xplane_weather.py** : cote pipeline Python (config, validation, communication JSON)
- Communication par fichiers JSON (weather_command.json / weather_status.json)
- cloud_type enum XPLMWeather : 0=Cirrus, 1=Stratus, 2=Cumulus, 3=Cumulonimbus

### Comportement meteo
- **Pluie seule** (precip > 0, cloud_type = -1) : Cumulonimbus forces automatiquement
- **Nuages seuls** (cloud_type >= 0, precip = 0) : nuages manuels
- **Pluie + nuages** (precip > 0, cloud_type >= 0) : nuages manuels utilises
- **Rien** (precip = 0, cloud_type = -1) : pas de meteo injectee

Les nuages sont places dynamiquement au-dessus de l'avion (alt max + 200m de marge).

### Parametres XML (per-scenario, valeurs directes)
```xml
<parameter name="precip_rate"      type="real" min="0" max="1"/>
<parameter name="cloud_type"       type="real" min="-1" max="3"/>
<parameter name="cloud_coverage"   type="real" min="0" max="1"/>
<parameter name="visibility_m"     type="real" min="500" max="50000"/>
<parameter name="temperature_c"    type="real" min="-30" max="45"/>
<parameter name="time_of_day_h"    type="real" min="0" max="24"/>
```

### Pipeline d'application
1. `generate` : TAF sample les params, Export.py cree `weather_profile.json`
2. `evaluate` / `full` : xplane_weather.py injecte UNE FOIS avant le rendu
3. PI_lard_weather.py applique via XPLMWeather API (isIncremental=False, updateImmediately=True)
4. Stabilisation ~6s puis capture des frames
5. clear_weather apres le scenario (regen_weather + change_mode=7)

### Installation plugin
1. Installer XPPython3 dans `X-Plane 12/Resources/plugins/XPPython3/`
2. Copier `PI_lard_weather.py` dans `X-Plane 12/Resources/plugins/PythonPlugins/`
3. Recharger : menu Plugins > XPPython3 > Reload Scripts

### Cumul avec les fautes capteur
Les trois systemes sont independants et se cumulent :
- Meteo X-Plane modifie la scene 3D (pluie, nuages, brouillard dans le renderer)
- Fautes capteur degradent l'image apres capture (OpenCV sur les pixels)
- Fautes meteo capteur (TODO) : pluie sur lentille, givre (post-traitement)
- Aucun n'affecte la GT (geometrie piste inchangee)

## Ce qu'on importe de LARD (sans modifier)
- `src/geo/geo_utils.py` — conversions ECEF/LLH, geodesie
- `src/geo/geo_dataset.py` — GEODataset.create_scenario() pour .esp
- `src/scenario/scenario_config.py` — ScenarioConfig
- `src/scenario/write_scenario.py` — generation .yaml
- `src/labeling/label_export.py` — export_labels(), computeLabels() (projection 3D→2D piste)
- `data/filtered_runways_database_Final.json` — DB pistes (aussi runways_db_V2_GEarth.json pour labeling)

## Notre code (project/export/)
- Processus Ornstein-Uhlenbeck a dt variable
- Convergence en finale (stabilisation_distance)
- Calcul crab angle (vent)
- Timeline spatiale (frames uniformes en distance, dt variable)
- Vitesse constante 145 kts (Vref approche standard)

## Detection YOLOv8 (yolo/)
- Modele : yolov8n.pt (entraine sur LARD pour detection de pistes, embarque avion)
- predict.py : `--images <dir> --output <dir>` ou standalone `python yolo/predict.py -n 100`
- evaluate.py : `--labels <dir> --csv <file>` ou via run_pipeline.py
- imgsz=512 : resize interne YOLO pour inference (coherent avec l'entrainement)
- IoU calcule sur coordonnees normalisees (0-1), independant de la resolution source

## Evaluation IoU (yolo/eval/)

### Formats de donnees
- **Predictions YOLO** (.txt dans runs/<nom>/predictions/) :
  - Format par ligne : `classe cx cy w h confiance` (cxcywh normalise 0-1)
- **Ground truth LARD** (.csv dans runs/<nom>/) :
  - 4 coins piste en pixels : x_TR, y_TR, x_TL, y_TL, x_BL, y_BL, x_BR, y_BR
  - Genere par LARD export_labels() (projection 3D → 2D)
  - Note : exported_images/ est cree par LARD (doublon de footage/), nettoye auto

### Module yolo/eval/ (inspire de lardv2-ml-detect-develop)
- box.py — conversions formats bbox numpy (tlbr, tlwh, xywh) + box_extract (coins → xyxy)
- metrics_utils.py — IoU torch (IOU/GIOU/DIOU/CIOU), conversions bbox torch
- metrics.py — match_predictions, compute_ap_per_cls, compute_metrics
- Depend de torch pour les tenseurs et le calcul IoU

## Regles
1. **NE PAS modifier LARD/** ni **taf/** — sauf `taf/src/Export.py` (point d'extension prevu par TAF, copie auto par Generate.py depuis `project/export/Export.py`)
2. Tout notre code va dans **project/**, **yolo/** et **run_pipeline.py**
3. Style : **dataclasses** pour configs, **fonctions pures** pour algorithmes
4. Mettre a jour `historique.txt` quand une feature majeure est implementee
5. Pas de cles API ni credentials dans le code (safe pour git public)
6. **runs/** est auto-genere, ne pas versionner (ajouter a .gitignore)
7. **output/** est temporaire TAF, nettoye avant chaque generate

## Parametres utilisateur (template XML)
Parametres que l'utilisateur peut moduler via le XML :
- `fps` : frames par seconde (10-15)
- `segment_start_m` : distance debut approche (ex: 2000-3000m)
- `segment_end_m` : distance fin approche (min 280m)
- `ground_speed_kts` : vitesse sol constante (145 kts)
- `turbulence_intensity` : intensite turbulence [0, 1]
- `wind_speed_kts`, `wind_direction_deg` : vent
- `stabilization_distance_m` : distance de convergence finale
- `airport`, `runway` : piste cible
- `nb_test_cases` : nombre de scenarios a generer

## Hyperparametres (dans le code, pas dans le XML)
- Formule OU : x[i+1] = mean + (x[i] - mean) * exp(-dt/tau) + sigma_d * N(0,1)
- Tau auto-calcule (Dryden simplifie) : tau = h / V, h = segment_start_m * tan(3°)
- Convergence : conv(d) = 0.08 + 0.92 * clamp((d - end) / (stab - end), 0, 1)^0.5
- Std par variable OU (alpha_v, alpha_h, yaw, pitch, roll)
- Clamp altitude >= ltp_alt + 1.0

## Preferences
- Langue : francais
- Communication : concise et directe
