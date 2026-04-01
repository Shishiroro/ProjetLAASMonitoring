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
│   │   ├── lua_bridge.py            # Interface X-Plane via FlyWithLua (fichiers JSON)
│   │   └── sensor_fault_profile.py   # Profil fautes capteur (config, validation, application)
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
├── lard_bridge.lua                    # Script FlyWithLua (a copier dans X-Plane)
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

## Fautes capteur camera (project/export/sensor_fault_profile.py)

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

## Effets meteo X-Plane 12 (project/export/sensor_fault_profile.py + xplane_bridge.py)

### Principe
Les effets meteo sont appliques **pendant** le rendu X-Plane via datarefs UDP,
contrairement aux fautes capteur qui sont appliquees en post-traitement (OpenCV).
Meme pattern XML (severity + from_pct + to_pct), meme logique frame par frame.
Ignores si renderer=ges (GES ne supporte pas les datarefs).

### Configuration dans le XML
```xml
<!-- severity min="0" max="0" = desactive -->
<parameter name="rain_severity"      type="real" min="0.3" max="0.7"/>
<parameter name="rain_from_pct"      type="real" min="20" max="40"/>
<parameter name="rain_to_pct"        type="real" min="60" max="90"/>
```

### 1 type d'effet meteo disponible (teste ecrivable XP12)
- **rain** : pluie (severity * 100 → 0-100% precipitation)

### Pipeline d'application
1. `generate` : TAF sample les params, Export.py cree `weather_profile.json`
2. `evaluate` / `full` : xplane_bridge.py charge le profil, applique frame par frame
3. `setup_weather()` desactive la meteo reelle (METAR) au debut du rendu
4. `apply_weather()` envoie les datarefs meteo a chaque frame selon from_pct/to_pct
5. `reset_weather()` remet le ciel clair a la fin du rendu

### Tracabilite
- `weather_profile.json` : config meteo + resume par frame
- `params.xml` : parametres TAF du scenario (inclut les weather_* generes)
- `.yaml` : section `xplane_weather` dans les metadonnees du scenario

### Cumul avec les fautes capteur
Les deux systemes sont independants et se cumulent :
- Meteo X-Plane modifie la scene 3D (pluie dans le moteur de rendu)
- Fautes capteur degradent l'image apres capture (OpenCV sur les pixels)
- Ni l'un ni l'autre n'affecte la GT (geometrie piste inchangee)

## Bridge FlyWithLua (project/export/lua_bridge.py + lard_bridge.lua)

### Principe
Remplacement de la communication UDP DREF par un plugin FlyWithLua qui tourne
dans le render loop X-Plane. Les datarefs sont ecrits via XPLMSetDataf depuis
do_every_draw(), ce qui garantit leur prise en compte par le moteur XP12
(contrairement a l'UDP externe que le nouveau moteur meteo ignore).

### Communication Python ↔ Lua
- Fichiers JSON dans un dossier d'echange (FlyWithLua/Scripts/lard_exchange/)
- Python ecrit `command.json` {seq, action, drefs, weather}
- Lua lit a chaque frame, applique datarefs, ecrit `status.json` {ack_seq, ok, actual_pose, ...}
- Sequence number pour synchronisation, ecriture atomique (.tmp + rename)

### Actions supportees
- `setup` : pause, overrides, forward_with_nothing, disable METAR, close popups
- `set_pose` : applique datarefs position/attitude + weather (atomique, 1 frame)
- `read_pose` : lit pose reelle + reference point (double precision lat/lon)
- `release` : unpause, remove overrides, reset meteo
- `noop` : heartbeat / detection FlyWithLua

### Factory (xplane_bridge.py)
`create_connection(config)` detecte automatiquement FlyWithLua :
1. Verifie si lard_bridge.lua existe dans FlyWithLua/Scripts/
2. Envoie noop, attend ack 5s
3. Si OK → LuaBridgeConnection ; sinon → fallback XPlaneConnection (UDP)

### Installation
1. Copier `lard_bridge.lua` dans `X-Plane 12/Resources/plugins/FlyWithLua/Scripts/`
2. Lancer X-Plane (le script se charge automatiquement)
3. Configurer `xplane_dir` dans XPlaneConfig (pour la detection auto)

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
