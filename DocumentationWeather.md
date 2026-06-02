# DocumentationWeather.md — Paramètres météo XML

Documentation des paramètres météo des templates (`project/templates/`), de leur
effet et de leurs dépendances (quels paramètres doivent être réunis pour
qu'un effet — pluie, neige, accumulation au sol — fonctionne réellement).

> Portée : uniquement la météo X-Plane 12 (scène 3D, injectée 1× avant le
> rendu via le plugin XPPython3). Les fautes capteur (post-traitement OpenCV)
> ne sont pas couvertes ici.

Code de référence : `project/export/xplane_weather.py`
(`build_plugin_command`, `inject_weather`).

---

## Vue d'ensemble — les 3 nœuds

Un template météo se compose de trois nœuds dans le scénario :

| Nœud | Rôle | Quand il agit |
|------|------|---------------|
| `weather`  | Météo de la scène 3D (pluie, nuages, brouillard, neige) | 1× avant rendu |
| `settings` | Réglages de rendu/sim (heure, délais, rayon zone) | 1× avant rendu |
| `faults`   | Fautes capteur (hors périmètre de ce document) | par frame |

Convention TAF dans les XML : `min`/`max` identiques = valeur fixe ;
`min`/`max` différents = TAF échantillonne dans la plage (solveur z3).

---

## Tester une météo sans lancer de scénario (`injection_weather_test.py`)

Avant de lancer un batch complet (long, rend des images), prévisualisation de 
la météo du XML actif directement dans X-Plane :

```powershell
py injection_weather_test.py
py injection_weather_test.py --run KPDX_10L
py injection_weather_test.py --xplane-dir "C:/X-Plane 12" --alt-offset 200
```

Ce que fait le script :
- Lit le XML actif (`project/settings.xml` → `template_file_name`) et en extrait
  la météo (valeur `(min+max)/2` par paramètre → déterministe).
- Téléporte l'avion / caméra au-dessus de la piste (pose initiale lue dans
  `runs/<run>/poses_cam_export.json`, par défaut le run le plus récent),
  `--alt-offset` mètres plus haut (défaut 100 m).
- Injecte la météo via le plugin XPPython3, puis laisse la sim en pause :
  tu observes le rendu directement dans X-Plane, sans capturer d'image.
- Idempotent : relancer ne cumule pas l'altitude (référence = le JSON, pas
  la pose courante).

> Prérequis : un run doit déjà exister dans `runs/`. Le script ne lit pas le
> `.yaml` mais le `poses_cam_export.json` du run (pour la pose au-dessus de la
> piste) — ce fichier est produit par la Phase 1 `generate` en même temps que le
> `.yaml`. Il faut donc avoir lancé `generate` au moins une fois
> (`py run_pipeline.py generate -n 1` suffit). X-Plane doit aussi tourner et le
> plugin `PI_weather.py` être actif. Utile pour itérer rapidement sur les
> paramètres météo (pluie, nuages, brouillard) avant de générer un batch.

---

## Nœud `weather`

### `precip_rate` — `[0, 1]`
Taux de précipitation. Interrupteur principal pluie/neige.
- `0` : pas de précipitation.
- `> 0` : active la pluie si les conditions sont réunies (voir
  [Recette pluie](#recette-pluie)). Le type de précipitation (pluie vs neige)
  dépend de `temperature_c`, pas de ce paramètre.

### `cloud_type` — enum `{-1, 0, 1, 2, 3}`
Type de nuage manuel (`XPLMWeather`).

| Valeur | Type | Note |
|--------|------|------|
| `-1` | aucun nuage manuel | combiné à `precip>0` → Cumulonimbus forcé auto |
| `0`  | Cirrus | haut, fin |
| `1`  | Stratus | couche basse |
| `2`  | Cumulus | bourgeonnant |
| `3`  | Cumulonimbus | nuage d'orage |

> Confirmation : oui, le type `3` (Cumulonimbus) donne la meilleure couverture
> pour la pluie. C'est d'ailleurs le type que le code force automatiquement en
> mode « pluie seule », avec `cloud_coverage = 1.0`. Pour de la pluie fiable,
> `cloud_type = 3` est le choix recommandé.

### `cloud_coverage` — `[0, 1]`
Densité de la couverture nuageuse.
- Plus la valeur est haute, plus la zone est couverte → plus de chances que
  la précipitation tombe à l'endroit où se trouve l'avion.
- Pour de la pluie locale fiable, viser `cloud_coverage` proche de `1`.
  (En mode pluie seule, le code l'impose déjà à `1.0`.)

### `cloud_thickness_m` — épaisseur du nuage (m)
`alt_top = alt_base + cloud_thickness_m`.
- Doit être > 0 pour avoir de la pluie/neige. Un nuage d'épaisseur `0`
  (base = sommet) est dégénéré pour `XPLMWeather` et ne génère aucune
  particule (bug historique « pluie invisible »).
- Sécurité dans le code : si `<= 0`, une épaisseur par défaut est substituée
  selon le type (`THICKNESS_DEFAULT_BY_TYPE = {Cirrus:1000, Stratus:500,
  Cumulus:2000, Cb:6000}`). Toute valeur positive (même 50 m) est respectée
  telle quelle. Pour un Cumulonimbus de pluie, viser plusieurs milliers de m.

### `cloud_margin_m` — marge au-dessus de l'avion (m)
Hauteur de la base du nuage au-dessus de l'altitude max de l'avion :
`cloud_base = aircraft_max_alt + cloud_margin_m`.
- Placé dynamiquement au-dessus de l'avion (et non à une altitude absolue) pour
  éviter que le nuage passe sous l'avion sur les approches en montagne.
- Ne pas mettre trop haut : si la base est trop loin au-dessus de l'avion,
  les particules de pluie ne sont pas visibles dans le champ de la caméra.
  Garder une marge modérée pour voir la pluie pendant l'approche.

### `fog_visibility` — visibilité (m), `[500, 50000]`
Distance de visibilité (brouillard).
- `50000` (ou plus) : pas de brouillard.
- `< 50000` : brouillard ; plus la valeur est petite, plus c'est dense.
- Indépendant de la pluie : agit même sans nuage ni précipitation.

### `temperature_c` — température (°C), `[-30, 45]`
Détermine pluie vs neige quand `precip_rate > 0` :
- Température positive → pluie.
- Température négative → neige.
- Sans précipitation, n'a pas d'effet visuel direct (mais `< 0` est compté
  comme « météo active »).

### `rain_scale` — taille des gouttes, `[0.1, 5.0]`
Taille visuelle des particules de pluie. `1.0` = défaut, `0.1` = fines,
`5.0` = grosses.
- Purement esthétique, ignoré si `precip_rate = 0`.
- Dataref privé XP12, susceptible de casser dans une future version.

### `weather_effect_duration` — accumulation au sol (s), `[0, 60]`
Délai d'accumulation des effets sol : flaques de pluie / couche de neige
sur la piste et l'environnement. La sim tourne en accéléré (×8) pendant ce
délai pour simuler plusieurs minutes de météo en quelques secondes réelles.
- À augmenter si on veut des flaques d'eau / de la neige accumulée au sol.
- `0` : la pluie tombe mais la piste reste sèche (pas d'accumulation).
- Ignoré si `precip_rate = 0`.
- Commun à tous les profils (jamais remplacé par `build_weather_templates.py`).

---

## Nœud `settings`

### `time_of_day_h` — heure locale, `[0, 24]`
Heure civile locale ; convertie en UTC via le fuseau politique réel de
l'aéroport (timezonefinder + pytz, gère l'heure d'été). Affecte la luminosité
de la scène.

### `load_texture_duration` — délai chargement (s), `[0, 60]`
Délai (vitesse sim normale) pour le streaming des textures de la zone
téléportée + stabilisation GPU des nuages. Appliqué à tous les scénarios,
même sans météo. ~30 s recommandé.

### `screenshot_duration` — délai pose (s)
Délai de stabilisation de la pose après chaque téléportation de la caméra,
avant la capture. Très court (~0.07 s).

### `weather_zone_radius_nm` — rayon zone météo (nm), `[1, 10000]`
Rayon de la zone météo injectée dans `XPLMWeather` (1 nm = 1.852 km).
- Petit rayon → météo concentrée localement (effets constants). `50` recommandé
  en usage général.
- Grand rayon (plusieurs centaines) → nuages visibles au loin, rendu plus joli.

---

## Recettes (combinaisons de paramètres)

### Recette pluie
Pour de la pluie visible et fiable, réunir :
1. `precip_rate > 0` (interrupteur)
2. `cloud_type` défini — n'importe lequel, mais `3` (Cumulonimbus) donne la
   meilleure couverture *(ou laisser `-1` : le code force alors Cb auto)*
3. `cloud_coverage` élevé (proche de `1`) → la pluie tombe là où est l'avion
4. `cloud_thickness_m > 0` (sinon aucune particule) — viser plusieurs milliers de m
5. `temperature_c > 0` (positif = pluie)
6. `cloud_margin_m` modéré (nuage pas trop haut, sinon pluie hors champ)

Optionnel : `rain_scale` pour la taille des gouttes ;
`weather_effect_duration > 0` pour des flaques au sol.

### Recette neige
Identique à la pluie, mais `temperature_c < 0` (négatif = neige).
`weather_effect_duration > 0` pour accumuler une couche de neige au sol.

### Recette nuages seuls (sans précipitation)
- `precip_rate = 0`
- `cloud_type >= 0`, `cloud_coverage` selon densité voulue,
  `cloud_thickness_m > 0`, `cloud_margin_m` modéré.

### Recette brouillard
- `fog_visibility < 50000` (plus petit = plus dense). Indépendant du reste.

### Recette accumulation au sol (flaques / neige)
- Recette pluie ou neige + `weather_effect_duration > 0` (plus la valeur est
  grande, plus l'accumulation est marquée).

---

## Tableau récapitulatif des dépendances

| Effet voulu | Paramètres requis |
|-------------|-------------------|
| Pluie | `precip_rate>0` + `cloud_type` (idéal 3) + `cloud_coverage`↑ + `cloud_thickness_m>0` + `temperature_c>0` + `cloud_margin_m` modéré |
| Neige | idem pluie mais `temperature_c<0` |
| Nuages secs | `cloud_type>=0` + `cloud_thickness_m>0` + `precip_rate=0` |
| Brouillard | `fog_visibility<50000` |
| Flaques / neige au sol | recette pluie/neige + `weather_effect_duration>0` |
| Grosses/fines gouttes | `rain_scale` (avec pluie active) |
