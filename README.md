# LARD-LAAS-TAF

Pipeline de génération de **trajectoires d'approche réalistes** pour avions, rendu
sous **X-Plane 12**, avec dégradation capteur et évaluation d'un modèle de
détection de piste.

Le pipeline échantillonne des scénarios sous contraintes avec **TAF** (Testing
Automation Framework, LAAS-CNRS), calcule les trajectoires et génère les images
via X-Plane 12. Sur ces images, un **modèle de détection** (choisi par
l'utilisateur dans `settings.xml`) prédit la position de la piste. Ces prédictions
sont ensuite comparées à la **vérité terrain** — la position réelle de la piste
projetée en 2D sur l'image — produite par **LARD** (ONERA, IRT Saint Exupéry et
AIRBUS). La qualité de la détection est mesurée par l'**IoU** (intersection sur
union) entre prédiction et vérité terrain.

L'utilisateur n'a qu'à éditer des fichiers **XML** pour définir ses scénarios,
puis à lancer le pipeline en ligne de commande. Un `notebook.ipynb` est également
disponible pour ceux qui préfèrent travailler en interactif.

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
   evaluate   ──►  Le modèle détecte la piste + calcul IoU vs GT
          │
          ▼
   runs/pipeline_report.json   (métriques : IoU, AP, F1, P, R)
```

---

## Prérequis

| Composant   | Détail |
|-------------|--------|
| **X-Plane 12** | Version **complète (payante)** obligatoire. Sinon, l'accès est limité à seulement quelques aéroports/pistes. |
| **Python**  | Python 3.10.10 (version utilisée pour le développement). Python 3.9+ devrait convenir (les dépendances `scipy ≥ 1.11` et `albumentations` exigent au minimum 3.9). |
| **OS**      | Compatible Windows et Linux. |

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

TAF n'est **pas** inclus dans le dépôt. Depuis la racine du projet :

```bash
git clone https://redmine.laas.fr/laas/taf.git
```

*Plus de détails sur TAF : <https://wp.laas.fr/taf/download/>*

Après les étapes 1 à 3, la racine doit contenir :

```
ProjetLAASMonitoring/
├── LARD/          ← cloné à l'étape 2
├── taf/           ← cloné à l'étape 3
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
   Suivre la procédure officielle :

   > https://xppython3.readthedocs.io/en/latest/usage/installation_plugin.html

   Il s'installe dans : `X-Plane 12/Resources/plugins/`.

   *Note : l'API **XPLMWeather** utilisée par le plugin météo est intégrée à
   X-Plane 12 et exposée directement par XPPython3 — rien à télécharger en plus.*

2. **PI_weather.py** — le plugin météo de ce projet.
   Créer le dossier `X-Plane 12/Resources/plugins/PythonPlugins/`, puis y copier
   `XPlanePlugin/PI_weather.py`.

Puis **recharger les scripts depuis le simulateur** : une fois X-Plane 12 lancé,
utiliser la barre de menu en haut de la fenêtre du simulateur :
**Plugins → XPPython3 → Reload Scripts**.

---

## Configuration de X-Plane 12

- Lancer X-Plane 12 en **mode fenêtré** (pas en plein écran). Le réglage se fait
  dans les paramètres d'affichage du simulateur. La capture des images se fait
  par capture d'écran de la fenêtre.
- Régler la **mise à l'échelle de l'affichage (scaling) à 100 %**.
  La capture est ensuite recadrée à une résolution fixe. Si le scaling de l'OS
  n'est pas à 100 %, les pixels capturés ne correspondent plus aux coordonnées
  attendues : la **bounding box de la vérité terrain (GT LARD)** se retrouve
  décalée par rapport à la piste.

---

=

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

# Phase 3 — détection + calcul IoU
python run_pipeline.py evaluate --all

# Tout enchaîner d'un coup
python run_pipeline.py full -n 5 --xplane-dir "C:/X-Plane 12"
```

**Pour une utilisation normale, la commande `full` suffit** : elle enchaîne les
trois phases en une seule invocation. Les commandes `generate` / `render` /
`evaluate` ci-dessus restent disponibles pour relancer une phase précise.

Le chemin d'installation X-Plane 12 peut aussi être renseigné directement dans
`project/settings.xml` via le paramètre `xplane_dir` :

```xml
<parameter name="xplane_dir" type="path" value="C:/X-Plane 12" />
```

Dans ce cas, l'option `--xplane-dir` en ligne de commande n'est plus nécessaire.

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
│   ├── predictions.csv       détections du modèle
│   └── predictions_txt/      labels bruts du modèle
└── pipeline_report.json      rapport agrégé : IoU, AP, F1, P, R par scénario
```

### Aller plus loin avec le notebook

Le fichier `notebook.ipynb` à la racine du projet propose des fonctionnalités
complémentaires, à utiliser à la demande :

- création de **datasets** à partir des images générées,
- assemblage des images d'un scénario en **flux vidéo**,
- export de **fichiers optionnels** (`params_trace.xml`, `xplane_config.json`),
- **visualisations des bounding boxes** (`yolo_box/`, `lard_box/`) pour comparer
  prédictions du modèle et vérité terrain LARD.

Le notebook permet également de lancer les **trois phases du pipeline**
(`generate`, `render`, `evaluate`) directement depuis ses cellules, sans passer
par la ligne de commande. C'est une alternative complète au CLI pour les
utilisateurs qui préfèrent travailler en interactif.
