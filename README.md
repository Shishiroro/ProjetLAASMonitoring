# LARD-LAAS-TAF

Pipeline de génération de **trajectoires d'approche réalistes** pour avions, rendu
sous **X-Plane 12**, avec dégradation capteur et évaluation d'un modèle de détection
de piste **YOLOv8**.

Le pipeline échantillonne des scénarios sous contraintes avec **TAF** (Testing
Automation Framework, LAAS-CNRS), calcule les trajectoires, génère les images via
X-Plane 12, puis évalue la détection de piste contre une vérité terrain produite
par **LARD** (ONERA / DEEL).

L'utilisateur n'a normalement qu'à éditer des fichiers **XML** pour définir ses
scénarios, puis à lancer le pipeline en ligne de commande.

---

## Aperçu du pipeline

```
  base_template.xml  /  templates/<profil>/*.xml      (contraintes utilisateur)
          │
          ▼
   generate   ──►  TAF échantillonne N scénarios valides (solveur z3)
          │
          ▼
   render     ──►  X-Plane 12 rend les images
                   + fautes capteur + vérité terrain (GT) LARD
          │
          ▼
   evaluate   ──►  YOLOv8 détecte la piste + calcul IoU vs GT
          │
          ▼
   runs/pipeline_report.json   (métriques : IoU, AP, F1, P, R)
```

---

## Prérequis

| Composant   | Détail |
|-------------|--------|
| **X-Plane 12** | Version **complète (payante)** obligatoire. La version démo ne convient pas : durée et zone limitées, plugins bridés. |
| **Python**  | 3.9 ou supérieur (testé avec 3.10). |
| **Git**     | Pour cloner les dépôts. |
| **OS**      | Windows ou Linux. |

---

## Installation

### 1. Cloner le projet

```bash
git clone https://github.com/Shishiroro/ProjetLAASMonitoring
cd ProjetLAASMonitoring
```

### 2. Récupérer LARD

LARD n'est **pas** inclus dans le dépôt. Depuis la racine du projet :

```bash
git clone https://github.com/deel-ai/LARD
```

Le dossier `LARD/` doit se trouver à la racine, à côté de `project/`.

### 3. Récupérer TAF

TAF n'est **pas** inclus dans le dépôt. Le télécharger depuis :

> https://wp.laas.fr/taf/download/

Extraire l'archive dans un dossier `taf/` à la racine du projet, de sorte que le
chemin `taf/src/` existe.

Après les étapes 1 à 3, la racine doit contenir :

```
ProjetLAASMonitoring/
├── LARD/          ← cloné à l'étape 2
├── taf/           ← téléchargé à l'étape 3 (taf/src/ doit exister)
├── project/
├── yolo/
├── XPlanePlugin/
├── run_pipeline.py
└── requirements.txt
```

### 4. Installer les dépendances Python

```bash
pip install -r requirements.txt
```

### 5. Installer le plugin météo dans X-Plane 12

Deux éléments distincts à installer :

1. **XPPython3** — le moteur de plugins Python pour X-Plane.
   Le télécharger (version pour X-Plane 12) et l'installer dans :
   `X-Plane 12/Resources/plugins/`

2. **PI_weather.py** — le plugin météo de ce projet.
   Copier `XPlanePlugin/PI_weather.py` dans :
   `X-Plane 12/Resources/plugins/PythonPlugins/`

Puis **recharger les scripts depuis le simulateur** : une fois X-Plane 12 lancé,
utiliser la barre de menu en haut de la fenêtre du simulateur :
**Plugins → XPPython3 → Reload Scripts**.

---

## Configuration de X-Plane 12

- Lancer X-Plane 12 en **mode fenêtré** (pas en plein écran) : la capture des
  images se fait par capture d'écran de la fenêtre.
- Régler la **mise à l'échelle de l'affichage (scaling) à 100 %**.
  La capture est ensuite recadrée à une résolution fixe. Si le scaling de l'OS
  n'est pas à 100 %, les pixels capturés ne correspondent plus aux coordonnées
  attendues : la **bounding box de la vérité terrain (GT LARD)** se retrouve
  décalée par rapport à la piste. Le 100 % garantit un alignement pixel-exact
  entre l'image et la GT.

---

## Configurer un scénario : les fichiers XML

C'est la seule partie à éditer pour définir ses propres scénarios.

### Choisir le profil actif — `project/settings.xml`

```xml
<parameter name="template_path"      type="path" value="templates/rain/" />
<parameter name="template_file_name" type="file" value="rain_heavy.xml" />
<parameter name="nb_test_cases"      type="integer" value="1" />
```

- `template_path` + `template_file_name` : le template XML utilisé pour la génération.
- `nb_test_cases` : nombre de scénarios à générer (peut être surchargé par `-n` en
  ligne de commande).

### Templates disponibles

- `project/templates/base_template.xml` — template de base (trajectoire + météo + 26 fautes).
- `project/templates/<profil>/*.xml` — variantes météo pré-générées, profils
  `clear`, `fog`, `clouds`, `rain`, `snow`, chacun en intensités *light / moderate / heavy*.

Pour régénérer les variantes météo après modification du template de base :

```bash
python project/templates/build_weather_templates.py
```

### Convention min / max

Chaque paramètre a un `min` et un `max` :

- `min` et `max` **identiques** → valeur fixe.
- `min` et `max` **différents** → TAF échantillonne une valeur dans la plage
  (résolution sous contraintes par le solveur z3).

### Les 3 blocs d'un template

| Bloc | Contenu |
|------|---------|
| **trajectory** | `fps`, distances de début/fin d'approche, `ground_speed_kts`, `turbulence_intensity`, vent, distance de stabilisation, `airport_runway`. |
| **weather** | Précipitations, type/couverture/épaisseur de nuages, visibilité, température. Injecté une fois avant le rendu. |
| **faults** | 26 types de fautes capteur. Chaque faute a `severity`, `start_pct`, `duration_pct`. Une faute est **active si `severity > 0`**, désactivée si `severity = 0`. |

### Piste cible

Le paramètre `airport_runway` utilise le format `ICAO_RWY` (exemple : `LFPO_24`,
`KPDX_10L`). La liste des pistes disponibles se trouve dans :
`LARD/data/runways_db_V2_XPlane.json`.

---

## Lancer le pipeline

```bash
# Phase 1 — génère les scénarios (.yaml + poses caméra) dans runs/
python run_pipeline.py generate -n 5

# Phase 2 — rendu X-Plane + fautes capteur + vérité terrain
python run_pipeline.py render --all --xplane-dir "C:/X-Plane 12"

# Phase 3 — détection YOLO + calcul IoU
python run_pipeline.py evaluate --all

# Tout enchaîner d'un coup
python run_pipeline.py full -n 5 --xplane-dir "C:/X-Plane 12"
```

Adapter `--xplane-dir` au chemin réel de l'installation X-Plane 12.

---

## Résultats

Chaque scénario produit un dossier dans `runs/` :

```
runs/
├── LFPO_24/
│   ├── LFPO_24.yaml          scénario généré (Phase 1)
│   ├── poses_cam_export.json poses caméra
│   ├── footage/              images rendues par X-Plane
│   ├── degraded/             images avec fautes capteur (si actives)
│   ├── LFPO_24_labels.csv    vérité terrain LARD
│   ├── predictions.csv       détections YOLO
│   └── predictions_txt/      labels YOLO bruts
└── pipeline_report.json      rapport agrégé : IoU, AP, F1, P, R par scénario
```
