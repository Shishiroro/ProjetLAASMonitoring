"""
weather_profiles.py — Presets meteo X-Plane pilotes par TAF
============================================================

Le XML expose 2 selecteurs (profile + intensity) en tete du node weather, et
conserve tous les params meteo de base (precip_rate, cloud_type, ...). Quand
profile != 0 ET intensity != 0, le preset retourne par lookup() ECRASE les
params lies au profil (precip_rate, cloud_type, cloud_coverage,
cloud_thickness_m, visibility_m, temperature_c, rain_scale).

Sinon, lookup() retourne None et les valeurs XML de base sont utilisees telles
quelles (rien n'est ecrase).

Les params orthogonaux (time_of_day_h, xplane_weather_settle_s) ne sont JAMAIS
ecrases par le preset — toujours lus du XML.

Profils :
  0 = none          : passthrough (pas d'override, garde les valeurs XML)
  1 = fog           : override visibility (pas de nuages/precip)
  2 = clouds        : override cloud_type/coverage/thickness (pas de precip)
  3 = rain          : override precip + Cb auto (cloud_type=-1) + rain_scale
  4 = snow_falling  : override precip + temperature_c < 0 (XP12 bascule en neige)
  5 = clear         : force ciel degage (precip=0, pas de nuages, vis=50km)

Intensites : 0 = off (pas d'override), 1 = light, 2 = moderate, 3 = heavy
  Note : pour profile=5 (clear), l'intensity n'a pas d'effet (clear est binaire).
  Il suffit que intensity != 0 pour activer l'override.

Pour ajuster un preset : editer PRESETS ci-dessous puis relancer 'generate'.
Pour un tweak one-shot sur un run deja genere : editer directement
runs/<scenario>/weather_profile.json (lu tel quel par xplane_weather).
"""

from dataclasses import dataclass


# IDs profils
PROFILE_NONE = 0           # passthrough (pas d'override)
PROFILE_FOG = 1
PROFILE_CLOUDS = 2
PROFILE_RAIN = 3
PROFILE_SNOW_FALLING = 4
PROFILE_CLEAR = 5          # force ciel degage (override actif)

PROFILE_NAMES = {
    PROFILE_NONE: "none",
    PROFILE_FOG: "fog",
    PROFILE_CLOUDS: "clouds",
    PROFILE_RAIN: "rain",
    PROFILE_SNOW_FALLING: "snow_falling",
    PROFILE_CLEAR: "clear",
}


@dataclass
class WeatherValues:
    """Valeurs meteo concretes resolues a partir du preset.

    Sert de payload neutre entre weather_profiles et Export.py. Les champs ici
    correspondent aux params XML qui peuvent etre ecrases par un preset.
    Les orthogonaux (time_of_day_h, settle_s) ne sont PAS ici car ils ne sont
    jamais touches par le profil.
    """
    precip_rate: float = 0.0
    cloud_type: float = -1.0          # -1 = pas de nuages manuels (Cb auto si precip > 0)
    cloud_coverage: float = 0.0
    cloud_thickness_m: float = 0.0    # 0 = laisse build_plugin_command appliquer MIN_THICKNESS_BY_TYPE
    visibility_m: float = 50000.0
    temperature_c: float = 15.0
    rain_scale: float = 1.0
    cloud_margin_m: float = 2000.0    # marge nuages au-dessus de l'avion pour la base nuages (m)


# Valeurs ciel degage (utilisees par PROFILE_CLEAR a toutes intensites).
# Aucun impact meteo : pas de pluie/neige, visibilite max, temperature douce.
# Nota : cloud_type=0 (Cirrus) + coverage=0 force une injection effective qui
# bascule X-Plane en mode meteo manuel et clear toutes les couches de nuages.
# Si on laissait cloud_type=-1, has_weather() retournerait False, l'injection
# serait skip, et X-Plane resterait en mode real-weather (nuages visibles).
CLEAR_VALUES = WeatherValues(
    precip_rate=0.0,
    cloud_type=0.0,           # Cirrus, mais coverage=0 -> aucun nuage visible
    cloud_coverage=0.0,
    cloud_thickness_m=10.0,
    visibility_m=50000.0,
    temperature_c=15.0,
    rain_scale=1.0,
    cloud_margin_m=2000.0,
)


# Presets : PRESETS[profile_id][intensity] -> WeatherValues
# Cas profile=0 (PROFILE_NONE) ou intensity=0 sont geres par lookup() qui
# retourne None (pas d'override -> on garde les valeurs XML de base).
PRESETS = {
    PROFILE_FOG: {
        # Brouillard : visibilite seule, pas de nuages ni precip
        1: WeatherValues(visibility_m=15000),    
        2: WeatherValues(visibility_m=7000),    
        3: WeatherValues(visibility_m=2000),     
    },

    #0 = Cirrus, 1 = Stratus, 2 = Cumulus, 3 = Cumulonimbus 
    PROFILE_CLOUDS: {
        # Cirrus 
        1: WeatherValues(cloud_type=0.0, cloud_coverage=0.3, cloud_thickness_m=1000,cloud_margin_m=2000.0),
        2: WeatherValues(cloud_type=0.0, cloud_coverage=0.7, cloud_thickness_m=2000,cloud_margin_m=2000.0),
        3: WeatherValues(cloud_type=0.0, cloud_coverage=1.0, cloud_thickness_m=6000,cloud_margin_m=2000.0),
        # Stratus
        4: WeatherValues(cloud_type=1.0, cloud_coverage=0.3, cloud_thickness_m=1000,cloud_margin_m=2000.0),
        5: WeatherValues(cloud_type=1.0, cloud_coverage=0.7, cloud_thickness_m=2000,cloud_margin_m=2000.0),
        6: WeatherValues(cloud_type=1.0, cloud_coverage=1.0, cloud_thickness_m=6000,cloud_margin_m=2000.0),
        # Cumulus
        7: WeatherValues(cloud_type=2.0, cloud_coverage=0.3, cloud_thickness_m=1000,cloud_margin_m=2000.0),
        8: WeatherValues(cloud_type=2.0, cloud_coverage=0.7, cloud_thickness_m=2000,cloud_margin_m=2000.0),
        9: WeatherValues(cloud_type=2.0, cloud_coverage=1.0, cloud_thickness_m=6000,cloud_margin_m=2000.0),
         # Cumulonimbus
        10: WeatherValues(cloud_type=3.0, cloud_coverage=0.3, cloud_thickness_m=1000,cloud_margin_m=2000.0),
        11: WeatherValues(cloud_type=3.0, cloud_coverage=0.7, cloud_thickness_m=2000,cloud_margin_m=2000.0),
        12: WeatherValues(cloud_type=3.0, cloud_coverage=1.0, cloud_thickness_m=6000,cloud_margin_m=2000.0),

    },

    #à modifier : RAJOUT blur pluie sur le capteur (pré-traitement lié au profil?)
    PROFILE_RAIN: {
        # Pluie : precip + Cb auto (cloud_type=-1 -> Cb forcé par xplane_weather)
        1: WeatherValues(precip_rate=0.3, rain_scale=0.5, cloud_thickness_m=6000,cloud_coverage=1.0,cloud_margin_m=2000.0),
        2: WeatherValues(precip_rate=0.6, rain_scale=2.0, cloud_thickness_m=6000,cloud_coverage=1.0,cloud_margin_m=1000.0),
        3: WeatherValues(precip_rate=1.0, rain_scale=5.0, cloud_thickness_m=6000,cloud_coverage=1.0,cloud_margin_m=200.0),
    },
    PROFILE_SNOW_FALLING: {
        # Neige : meme principe que pluie mais T < 0 (XP12 bascule particules en neige)
        1: WeatherValues(precip_rate=0.3, temperature_c=-3.0, rain_scale=0.5, cloud_coverage=1.0, cloud_margin_m=2000.0),
        2: WeatherValues(precip_rate=0.6, temperature_c=-8.0, rain_scale=2.0, cloud_coverage=1.0, cloud_margin_m=2000.0),
        3: WeatherValues(precip_rate=1.0, temperature_c=-15.0, rain_scale=5.0, cloud_coverage=1.0, cloud_margin_m=2000.0),
    },
    PROFILE_CLEAR: {

        1: CLEAR_VALUES
    },
    ##add les 2 derniers du pw
}


def lookup(profile_id, intensity):
    """Retourne les WeatherValues du preset, ou None si pas d'override.

    None = profile_id=PROFILE_NONE (0) OU intensity=0 -> les params XML de base
    sont utilises tels quels (rien n'est ecrase).
    """
    profile_id = int(profile_id)
    intensity = int(intensity)

    if profile_id == PROFILE_NONE or intensity == 0:
        return None

    if profile_id not in PRESETS:
        raise ValueError(
            f"weather profile inconnu : {profile_id} "
            f"(valides : {sorted(PROFILE_NAMES.keys())})"
        )

    if intensity not in PRESETS[profile_id]:
        raise ValueError(
            f"weather intensity={intensity} hors sa plage pour profile={profile_id}"
        )

    return PRESETS[profile_id][intensity]


def profile_name(profile_id):
    """Nom lisible du profil (pour logs et trace)."""
    return PROFILE_NAMES.get(int(profile_id), f"unknown({profile_id})")
