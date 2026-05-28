# Référence des commandes

Référence complète du CLI `run_pipeline.py` (toutes les sous-commandes et leurs
options) et de ses équivalents notebook.

- Pour l'**installation**, les **prérequis** et la **configuration des scénarios
  (XML)** → voir [README.md](README.md).
- Pour le **mode interactif** → voir `notebook_generation.ipynb` (les 3 phases)
  et `notebook_features.ipynb` (outils annexes : dataset, regroup, sanity,
  exports, vidéo). Chaque section est documentée en tête de cellule.

Toutes les commandes se lancent depuis la racine du projet.

---

## Vue d'ensemble

Le pipeline se découpe en **3 phases indépendantes**, plus **2 modes qui les
enchaînent**.

| Commande | Phases | Rôle | X-Plane requis |
|----------|--------|------|----------------|
| `generate`      | 1       | Échantillonne N scénarios via TAF/z3 (`.yaml` + poses + profils JSON) | Non |
| `render`        | 2       | Rend les images sous X-Plane 12, applique les fautes capteur **et génère la vérité terrain LARD** | **Oui** |
| `evaluate`      | 3       | Lance la détection YOLO + calcule l'IoU vs la vérité terrain (produite en Phase 2) | Non |
| `full`          | 1 + 2   | `generate` puis `render` (**sans** évaluation) | **Oui** |
| `full_evaluate` | 1 + 2 + 3 | Pipeline complet de bout en bout | **Oui** |

> **Attention :** `full` n'enchaîne **que** la génération et le rendu. Pour le
> pipeline complet (avec détection + IoU), utiliser **`full_evaluate`**.

Aide intégrée :

```bash
py run_pipeline.py --help            # liste des sous-commandes + structure runs/
py run_pipeline.py <commande> --help # options d'une sous-commande
```

---

## Les 3 phases

1. **`generate`** (Phase 1) — TAF résout les contraintes XML et écrit, pour
   chaque scénario, `runs/<generation>/<ICAO_RWY>/` avec `.yaml`,
   `poses_cam_export.json`, et si actifs `fault_profile.json` /
   `weather_profile.json`. Hors ligne, ne touche pas à X-Plane.
2. **`render`** (Phase 2) — X-Plane 12 rend les images dans `footage/`, les
   fautes capteur produisent `degraded/`, puis la **vérité terrain LARD** est
   générée (`*_labels.csv`, projection 3D→2D des coins de piste). Requiert
   X-Plane lancé.
3. **`evaluate`** (Phase 3) — lance YOLO (`predictions.csv`), calcule l'IoU
   contre la vérité terrain **produite en Phase 2** et agrège un
   `pipeline_report.json`. Hors ligne, ré-exécutable à volonté avec d'autres
   seuils sans re-rendre.

---

## Concepts communs

### Générations (batchs)

Chaque exécution de `generate` crée un **dossier de génération** sous `runs/` qui
regroupe tous les scénarios du batch :

```
runs/
└── generation_01/            ← une génération = un batch
    ├── LFPO_24/              ← un dossier par scénario (format ICAO_RWY)
    ├── KLAX_25R/
    └── pipeline_report.json  ← rapport agrégé (après evaluate)
```

- Sans `--name` : `generation_01`, `generation_02`, … (auto-incrément).
- Avec `--name pluie` : `pluie_01`, `pluie_02`, … (auto-incrément séparé).
- Si la **même piste** est générée 2× dans le même batch, suffixe automatique :
  `LFPO_24`, `LFPO_24_002`, `LFPO_24_003`, …

### Cibler des runs (`render` / `evaluate`)

Ces deux commandes prennent **soit** un run précis, **soit** `--all` :

| Forme | Exemple | Effet |
|-------|---------|-------|
| Chemin composé | `render generation_01/LFPO_24` | Cible exactement ce run (recommandé) |
| Nom + `--generation` | `render LFPO_24 --generation generation_01` | Idem, autre syntaxe |
| Nom seul | `render LFPO_24` | Cherche dans toutes les générations ; **erreur** si le nom existe dans plusieurs |
| `--all --generation` | `render --all --generation generation_01` | Tous les runs de cette génération |

> **`--all` exige `--generation`** en ligne de commande, pour éviter de mélanger
> plusieurs batchs par inadvertance.

---

## Référence par sous-commande

### `generate` — Phase 1

```bash
py run_pipeline.py generate [-n N] [--name NOM] [--clean]
```

Échantillonne les scénarios via TAF et crée la génération sous `runs/`.

| Option | Défaut | Description |
|--------|--------|-------------|
| `-n`, `--nb-scenarios` | `nb_test_cases` de `settings.xml` (= 3) | Nombre de scénarios à générer |
| `--name NOM` | `generation` | Préfixe du dossier de génération (`<NOM>_NN/`) |
| `--clean` | désactivé | Vide **tout** `runs/` avant de générer |

```bash
py run_pipeline.py generate -n 5
py run_pipeline.py generate -n 100 --name pluie --clean
```

> `--clean` supprime l'intégralité de `runs/` (toutes les générations). À
> utiliser en connaissance de cause.

---

### `render` — Phase 2

```bash
py run_pipeline.py render (<run> | --all --generation NOM) [--xplane-dir CHEMIN]
```

Rend les images sous X-Plane 12, applique les fautes capteur, puis génère la
vérité terrain LARD (`*_labels.csv`). Suppose la Phase 1 déjà faite. **X-Plane 12
doit être lancé** (mode fenêtré, scaling 100 %).

| Option | Défaut | Description |
|--------|--------|-------------|
| `run` (positionnel) | — | Run à rendre : `<gen>/<run>` ou nom seul |
| `--all` | désactivé | Tous les runs de la génération (requiert `--generation`) |
| `--generation NOM` | — | Génération ciblée |
| `--xplane-dir CHEMIN` | `xplane_dir` de `settings.xml` (= `C:/X-Plane 12`) | **Optionnel.** Surcharge ponctuelle du répertoire X-Plane 12. Sert uniquement à localiser le plugin météo ; sans météo, sa valeur n'a aucun effet. À renseigner de préférence dans `settings.xml`. |

```bash
py run_pipeline.py render generation_01/LFPO_24
py run_pipeline.py render --all --generation generation_01
py run_pipeline.py render --all --generation pluie_01 --xplane-dir "D:/X-Plane 12"
```

---

### `evaluate` — Phase 3

```bash
py run_pipeline.py evaluate (<run> | --all --generation NOM) \
    [--runway PISTE] [--conf C] [--imgsz S] [--iou-thresh T] [--iou-method M]
```

Lance la détection YOLO et calcule l'IoU contre la vérité terrain. Suppose la
Phase 2 faite : images (`footage/` ou `degraded/`) **et** `*_labels.csv` déjà
présents. **Pas besoin de X-Plane.** Écrit `pipeline_report.json` dans la génération.

| Option | Défaut | Description |
|--------|--------|-------------|
| `run` (positionnel) | — | Run à évaluer : `<gen>/<run>` ou nom seul |
| `--all` | désactivé | Tous les runs de la génération (requiert `--generation`) |
| `--generation NOM` | — | Génération ciblée |
| `--runway PISTE` | aucun filtre | Restreint la vérité terrain à une piste |
| `--conf` | `0.25` | Seuil de confiance YOLO |
| `--imgsz` | `512` | Taille d'image pour l'inférence YOLO |
| `--iou-thresh` | `0.5` | Seuil d'IoU pour le matching prédiction/GT |
| `--iou-method` | `CIOU` | Variante d'IoU : `IOU`, `GIOU`, `DIOU`, `CIOU` |

```bash
py run_pipeline.py evaluate generation_01/LFPO_24
py run_pipeline.py evaluate --all --generation pluie_01
py run_pipeline.py evaluate --all --generation generation_01 --conf 0.4 --iou-method GIOU
```

> Le modèle YOLO utilisé est défini par `yolo_model` dans `settings.xml`
> (défaut : `yolov8nTest.pt`).

---

### `full` — Phases 1 + 2

```bash
py run_pipeline.py full [-n N] [--name NOM] [--clean] [--xplane-dir CHEMIN]
```

Enchaîne `generate` puis `render` sur les runs créés. **Sans évaluation.**
X-Plane 12 doit être lancé.

Options : celles de [`generate`](#generate--phase-1) + `--xplane-dir`.

```bash
py run_pipeline.py full -n 5
py run_pipeline.py full -n 100 --name pluie --clean --xplane-dir "C:/X-Plane 12"
```

---

### `full_evaluate` — Phases 1 + 2 + 3

```bash
py run_pipeline.py full_evaluate [-n N] [--name NOM] [--clean] \
    [--xplane-dir CHEMIN] [--conf C] [--imgsz S] [--iou-thresh T] [--iou-method M]
```

**Pipeline complet de bout en bout** : génération + rendu + détection + IoU,
zéro intervention. C'est la commande à utiliser pour un cycle complet.
X-Plane 12 doit être lancé.

Options : celles de [`generate`](#generate--phase-1) + `--xplane-dir` + les
options YOLO/IoU de [`evaluate`](#evaluate--phase-3).

```bash
py run_pipeline.py full_evaluate -n 5
py run_pipeline.py full_evaluate -n 100 --name pluie --xplane-dir "C:/X-Plane 12"
```

---

## Tableau récapitulatif des options

| Option | `generate` | `render` | `evaluate` | `full` | `full_evaluate` |
|--------|:--:|:--:|:--:|:--:|:--:|
| `-n / --nb-scenarios` | ✓ | | | ✓ | ✓ |
| `--name`              | ✓ | | | ✓ | ✓ |
| `--clean`             | ✓ | | | ✓ | ✓ |
| `run` (positionnel)   | | ✓ | ✓ | | |
| `--all`               | | ✓ | ✓ | | |
| `--generation`        | | ✓ | ✓ | | |
| `--xplane-dir` *(opt.)* | | ✓ | | ✓ | ✓ |
| `--runway`            | | | ✓ | | |
| `--conf`              | | | ✓ | | ✓ |
| `--imgsz`             | | | ✓ | | ✓ |
| `--iou-thresh`        | | | ✓ | | ✓ |
| `--iou-method`        | | | ✓ | | ✓ |

---

## Workflows typiques

**Cycle complet en une commande** (le plus courant) :

```bash
py run_pipeline.py full_evaluate -n 5 --xplane-dir "C:/X-Plane 12"
```

**Phase par phase** (contrôle fin, ré-exécution ciblée) :

```bash
py run_pipeline.py generate -n 5 --name test
py run_pipeline.py render   --all --generation test_01
py run_pipeline.py evaluate --all --generation test_01
```

**Ré-évaluer sans re-rendre** (tester d'autres seuils YOLO/IoU) :

```bash
py run_pipeline.py evaluate --all --generation test_01 --conf 0.4 --iou-method DIOU
```

**Générer et rendre maintenant, évaluer plus tard** :

```bash
py run_pipeline.py full -n 20 --name pluie --xplane-dir "C:/X-Plane 12"
# ... plus tard, X-Plane peut être fermé ...
py run_pipeline.py evaluate --all --generation pluie_01
```

---

## Équivalents dans le notebook

Deux notebooks : `notebook_generation.ipynb` reproduit les 3 phases (Setup,
Generate, Render, Evaluate — exécutables séparément) ; `notebook_features.ipynb`
ajoute les outils à la demande. Lancer les cellules dans l'ordre après la
section **Setup**.

| Notebook (fonction)       | CLI équivalent | Rôle |
|---------------------------|----------------|------|
| `generate_runs(...)`      | `generate`     | Phase 1 |
| `render_runs(...)`        | `render`       | Phase 2 |
| `evaluate_runs(...)`      | `evaluate`     | Phase 3 |
| `build_yolo_box(...)`     | —              | Images annotées avec les bbox YOLO (`yolo_box/`) |
| `build_lard_box(...)`     | —              | Images annotées avec la vérité terrain LARD (`lard_box/`) |
| `show_sanity(...)` / `show_sanity_lard(...)` | — | Aperçu rapide (1re / milieu / dernière image) |
| `build_xplane_config(...)`| —              | (Re)génère `xplane_config.json` |
| `build_params_trace(...)` | —              | (Re)génère `params_trace.xml` |
| `build_video(...)`        | —              | Assemble les images d'un run en MP4 |

Les fonctions du notebook acceptent les mêmes formes de ciblage que le CLI :
sans argument = tous les runs ; nom seul si unique ; chemin composé
`"<gen>/<run>"` sinon.

---

## Où vont les résultats

Voir la section **Résultats** du [README.md](README.md) pour le détail de
l'arborescence `runs/<generation>/<ICAO_RWY>/`. En résumé :

- `footage/` — images brutes X-Plane ; `degraded/` — images avec fautes capteur
- `*_labels.csv` — vérité terrain LARD ; `predictions.csv` — détections YOLO
- `pipeline_report.json` — métriques agrégées (IoU, AP, F1, P, R) par batch
