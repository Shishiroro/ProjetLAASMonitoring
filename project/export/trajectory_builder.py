"""
trajectory_builder.py — Generation de trajectoires d'approche realistes
=======================================================================
Migration depuis LARDv2/LARD/src/trajectory/sequence_generator.py
vers un module decouple, style fonctions pures + dataclasses.

Pipeline :
    1. Timeline spatiale (frames uniformes en distance, dt variable)
    2. Processus Ornstein-Uhlenbeck a dt variable (5 canaux)
    3. Convergence finale (deviations → 0 pres de la piste)
    4. Crab angle (correction vent)
    5. Assemblage des poses (lon, lat, alt, yaw, pitch, roll)
"""

import numpy as np
import pyproj
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

KTS_TO_MS = 0.514444


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryConfig:
    """Parametres utilisateur (viennent du XML via TAF)."""
    fps: int = 5
    segment_start_m: float = 3000.0
    segment_end_m: float = 280.0
    ground_speed_kts: float = 137.0
    correlation_time_s: float = 10.0
    turbulence_intensity: float = 0.3
    wind_speed_kts: float = 0.0
    wind_direction_deg: float = 0.0
    stabilization_distance_m: float = 800.0


@dataclass
class OUParams:
    """Hyperparametres OU (dans le code, pas dans le XML)."""
    alpha_v_deg: float = -3.0       # angle vertical moyen (glide slope ~3°), pente de descente standard
    alpha_h_deg: float = 0.0        # angle horizontal moyen (axe piste), l'avion viste la piste
    pitch_deg: float = -4.0         # pitch moyen
    yaw_deg: float = 0.0            # yaw moyen (deviation)
    roll_deg: float = 0.0           # roll moyen

    std_alpha_v_deg: float = 0.4    # ecart-type vertical (LARDv2: 0.4)
    std_alpha_h_deg: float = 1.0    # ecart-type horizontal (ILS localizer: ±2.5°, 2σ=2.0°)
    std_yaw_deg: float = 2.5        # ecart-type yaw (reduit pour vue frontale stable)
    std_pitch_deg: float = 2.0      # ecart-type pitch (LARDv2: 2.0)
    std_roll_deg: float = 5.0       # ecart-type roll (LARDv2: 5.0)

    dist_ap_m: float = 300.0        # distance aiming point depuis LTP (où l'avion vise)


# ---------------------------------------------------------------------------
# Timeline spatiale
# ---------------------------------------------------------------------------

def compute_spatial_timeline(cfg: TrajectoryConfig):
    """
    Frames uniformes en distance, dt variable (vitesse constante).

    Avec vitesse constante, dt est en fait constant aussi.
    On garde la structure a dt variable pour compatibilite future
    (si on reintroduit la deceleration).

    :return: (distances_m, dt_array)
        distances_m : ndarray (n_frames,), decroissant
        dt_array    : ndarray (n_frames-1,)
    """
    # crée N frames uniformément espacées en distance (décroissant de start à end), calcule le dt entre chaque frame
    speed_ms = cfg.ground_speed_kts * KTS_TO_MS
    total_dist = cfg.segment_start_m - cfg.segment_end_m
    duration = total_dist / speed_ms
    n_frames = max(int(round(duration * cfg.fps)), 2)

    distances_m = np.linspace(cfg.segment_start_m, cfg.segment_end_m, n_frames)
    spatial_step = abs(distances_m[1] - distances_m[0]) if n_frames > 1 else 0.0

    dt_array = np.full(n_frames - 1, spatial_step / speed_ms)

    return distances_m, dt_array


# ---------------------------------------------------------------------------
# Processus Ornstein-Uhlenbeck a dt variable
# ---------------------------------------------------------------------------

def generate_ou_process(n_steps, dt_array, correlation_time, std, mean=0.0,
                        conv_factors=None):
    """
    Processus OU discret avec pas de temps variables et convergence integree.

    x[i+1] = mean + (x[i] - mean) * exp(-dt/tau) + sigma_d * N(0,1)
    ou sigma_d = std_i * sqrt(1 - exp(-2*dt/tau))

    Si conv_factors est fourni, le std varie par pas (std_i = std * conv[i]).
    Le processus OU converge alors naturellement vers mean car le bruit
    injecte diminue progressivement, tout en gardant la correlation temporelle.

    :param n_steps: nombre de pas
    :param dt_array: array de dt (n_steps-1,)
    :param correlation_time: tau du processus OU (secondes)
    :param std: ecart-type stationnaire de base
    :param mean: moyenne de reversion
    :param conv_factors: ndarray (n_steps,) facteurs de convergence [0,1]
    :return: ndarray (n_steps,)
    """
    #  processus Ornstein-Uhlenbeck discret à dt variable. Formule : x[i+1] = mean + (x[i]-mean)*exp(-dt/tau) + sigma*N(0,1)
    #  Si conv_factors est passé, le bruit diminue progressivement → convergence naturelle
    if n_steps <= 0:
        return np.array([])

    tau = max(correlation_time, 1e-6)
    x = np.zeros(n_steps)

    std_0 = std * (conv_factors[0] if conv_factors is not None else 1.0)
    x[0] = mean + std_0 * np.random.randn()

    for i in range(1, n_steps):
        dt = dt_array[min(i - 1, len(dt_array) - 1)]
        decay = np.exp(-dt / tau)
        std_i = std * (conv_factors[i] if conv_factors is not None else 1.0)
        sigma_d = std_i * np.sqrt(max(1.0 - np.exp(-2.0 * dt / tau), 0.0))
        x[i] = mean + (x[i - 1] - mean) * decay + sigma_d * np.random.randn()

    return x


# ---------------------------------------------------------------------------
# Convergence finale
# ---------------------------------------------------------------------------
#facteurs [0,1] par frame. Vaut 1.0 loin de la piste, diminue vers residual vers la piste

def compute_convergence_factors(distances_m, segment_end_m,
                                stabilization_distance_m,
                                exponent=0.5, residual=0.08):
    """
    Facteurs de convergence : 1.0 loin, → residual pres de la piste.

    conv(d) = residual + (1 - residual) * clamp((d - end) / (stab - end), 0, 1) ^ exponent

    Le residual (defaut 8%) empeche les deviations de tomber a zero,
    gardant un bruit realiste meme en courte finale.

    :return: ndarray (len(distances_m),) dans [residual, 1]
    """
    factors = np.ones(len(distances_m))
    if stabilization_distance_m <= segment_end_m:
        return factors

    for i, d in enumerate(distances_m):
        if d < stabilization_distance_m:
            ratio = (d - segment_end_m) / (stabilization_distance_m - segment_end_m)
            raw = np.clip(ratio, 0.0, 1.0) ** exponent
            factors[i] = residual + (1.0 - residual) * raw

    return factors


# ---------------------------------------------------------------------------
# Crab angle
# ---------------------------------------------------------------------------
# Angle de crabe pour compenser le vent traversier, avec correction pilote.
def compute_crab_angle(wind_speed_kts, wind_direction_deg,
                       runway_heading_deg, groundspeed_kts,
                       pilot_correction=0.30):
    """
    Angle de crabe pour compenser le vent traversier.

    :param pilot_correction: fraction du crab corrigee par le pilote (0=full crab)
    :return: crab angle en degres
    """
    if wind_speed_kts <= 0 or groundspeed_kts <= 0:
        return 0.0

    relative_wind_deg = wind_direction_deg - runway_heading_deg
    crosswind_kts = wind_speed_kts * np.sin(np.deg2rad(relative_wind_deg))
    ratio = np.clip(crosswind_kts / groundspeed_kts, -1.0, 1.0)
    crab_deg = np.rad2deg(np.arcsin(ratio))
    return crab_deg * (1.0 - pilot_correction)


# ---------------------------------------------------------------------------
# Generation de trajectoire complete
# ---------------------------------------------------------------------------
# Genere une trajectoire d'approche complete, de segment_start_m a segment_end_m,
# Retourne la liste de poses (lon, lat, alt, yaw, pitch, roll) 
def build_trajectory(cfg: TrajectoryConfig, ou: OUParams,
                     ltp_lat, ltp_lon, ltp_alt,
                     runway_heading_deg, runway_back_azimuth_deg):
    """
    Genere une trajectoire d'approche complete.

    :param cfg: parametres utilisateur (du XML)
    :param ou: hyperparametres OU (du code)
    :param ltp_lat, ltp_lon, ltp_alt: position LTP de la piste
    :param runway_heading_deg: azimut LTP→FPAP (cap camera)
    :param runway_back_azimuth_deg: azimut FPAP→LTP (positionnement avion)
    :return: list de tuples (lon, lat, alt, yaw, pitch, roll)
    """
    # --- Timeline spatiale ---
    distances_m, dt_array = compute_spatial_timeline(cfg)
    n_frames = len(distances_m)

    # --- Scaling turbulence ---
    # turb=0 → scale=0.1 (quasi ILS), turb=0.5 → scale=0.55, turb=1 → scale=1.0
    turb_scale = 0.1 + 0.9 * max(0.0, cfg.turbulence_intensity)

    # --- Convergence (integree dans l'OU) ---
    conv = compute_convergence_factors(
        distances_m, cfg.segment_end_m, cfg.stabilization_distance_m
    )

    # --- OU deviations par canal (avec convergence integree) ---
    # Le std diminue progressivement via conv_factors, ce qui fait
    # converger le processus OU naturellement vers 0 sans "snap".
    def _ou(std, mean=0.0):
        return generate_ou_process(
            n_steps=n_frames,
            dt_array=dt_array,
            correlation_time=cfg.correlation_time_s,
            std=std * turb_scale,
            mean=mean,
            conv_factors=conv,
        )

    alpha_v_dev = _ou(ou.std_alpha_v_deg)
    alpha_h_dev = _ou(ou.std_alpha_h_deg)
    yaw_dev     = _ou(ou.std_yaw_deg)
    pitch_dev   = _ou(ou.std_pitch_deg)
    roll_dev    = _ou(ou.std_roll_deg)

    # --- Crab angle (constant, pas de convergence) ---
    # Le crab angle corrige le vent : l'avion le maintient jusqu'au toucher.
    # Seules les deviations OU convergent, pas la correction vent.
    crab_angle = compute_crab_angle(
        cfg.wind_speed_kts, cfg.wind_direction_deg,
        runway_heading_deg, cfg.ground_speed_kts
    )
    crab_angles = np.full(n_frames, crab_angle)

    # --- Calcul altitudes brutes puis lissage monotone ---
    raw_alts = np.zeros(n_frames)
    for i in range(n_frames):
        alpha_v = ou.alpha_v_deg + alpha_v_dev[i]
        alpha_v = np.clip(alpha_v, -6.0, -0.5)  # toujours en descente, range realiste
        raw_alts[i] = ltp_alt + (-np.tan(np.deg2rad(alpha_v))
                                  * (distances_m[i] + ou.dist_ap_m))
        raw_alts[i] = max(raw_alts[i], ltp_alt + 1.0)

    # Enforce quasi-monotone descent : on autorise de petites remontees
    # mais on plafonne le taux de montee a une valeur physique realiste.
    # ~1.5 m/s ≈ 300 ft/min : limite haute d'une rafale verticale en approche.
    # Converti en par-frame via dt reel (depend de vitesse + fps).
    max_climb_rate_ms = 1.5
    max_descent_rate_ms = 5.0   # ~1000 ft/min : descente max realiste en approche
    dt_frame = dt_array[0]  # constant (vitesse constante)
    max_climb_per_frame = max_climb_rate_ms * dt_frame
    max_descent_per_frame = max_descent_rate_ms * dt_frame
    for i in range(1, n_frames):
        if raw_alts[i] > raw_alts[i - 1] + max_climb_per_frame:
            raw_alts[i] = raw_alts[i - 1] + max_climb_per_frame
        elif raw_alts[i] < raw_alts[i - 1] - max_descent_per_frame:
            raw_alts[i] = raw_alts[i - 1] - max_descent_per_frame

    # --- Assemblage des poses ---
    g = pyproj.Geod(ellps='WGS84')
    flight_data = []

    for i in range(n_frames):
        dist = distances_m[i]
        alt = raw_alts[i]

        # Position horizontale : projeter depuis LTP le long du back azimuth
        # = derriere le seuil, loin de la piste (convention LARD : LTP = C/D)
        alpha_h = ou.alpha_h_deg + alpha_h_dev[i]
        lon, lat, _ = g.fwd(
            ltp_lon, ltp_lat,
            runway_back_azimuth_deg + alpha_h,
            dist, radians=False
        )

        # Camera regarde vers FPAP (runway_heading = LTP→FPAP)
        yaw   = runway_heading_deg + crab_angles[i] + yaw_dev[i]
        pitch = 90 + ou.pitch_deg + pitch_dev[i]
        roll  = ou.roll_deg + roll_dev[i]

        flight_data.append((lon, lat, alt, yaw, pitch, roll))

    return flight_data, distances_m, ltp_alt
