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
    alpha_v_deg: float = -3.0       # angle vertical moyen (glide slope ~3°)
    alpha_h_deg: float = 0.0        # angle horizontal moyen (axe piste)
    pitch_deg: float = -4.0         # pitch moyen
    yaw_deg: float = 0.0            # yaw moyen (deviation)
    roll_deg: float = 0.0           # roll moyen

    std_alpha_v_deg: float = 0.4    # ecart-type vertical
    std_alpha_h_deg: float = 2.0    # ecart-type horizontal
    std_yaw_deg: float = 5.0        # ecart-type yaw
    std_pitch_deg: float = 2.0      # ecart-type pitch
    std_roll_deg: float = 5.0       # ecart-type roll

    dist_ap_m: float = 300.0        # distance aiming point depuis LTP


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

def generate_ou_process(n_steps, dt_array, correlation_time, std, mean=0.0):
    """
    Processus OU discret avec pas de temps variables.

    x[i+1] = mean + (x[i] - mean) * exp(-dt/tau) + sigma_d * N(0,1)
    ou sigma_d = std * sqrt(1 - exp(-2*dt/tau))

    :param n_steps: nombre de pas
    :param dt_array: array de dt (n_steps-1,)
    :param correlation_time: tau du processus OU (secondes)
    :param std: ecart-type stationnaire
    :param mean: moyenne de reversion
    :return: ndarray (n_steps,)
    """
    if n_steps <= 0:
        return np.array([])

    tau = max(correlation_time, 1e-6)
    x = np.zeros(n_steps)
    x[0] = mean + std * np.random.randn()

    for i in range(1, n_steps):
        dt = dt_array[min(i - 1, len(dt_array) - 1)]
        decay = np.exp(-dt / tau)
        sigma_d = std * np.sqrt(max(1.0 - np.exp(-2.0 * dt / tau), 0.0))
        x[i] = mean + (x[i - 1] - mean) * decay + sigma_d * np.random.randn()

    return x


# ---------------------------------------------------------------------------
# Convergence finale
# ---------------------------------------------------------------------------

def compute_convergence_factors(distances_m, segment_end_m,
                                stabilization_distance_m, exponent=2.0):
    """
    Facteurs de convergence : 1.0 loin, → 0.0 pres de la piste.

    conv(d) = clamp((d - end) / (stab - end), 0, 1) ^ exponent

    :return: ndarray (len(distances_m),) dans [0, 1]
    """
    factors = np.ones(len(distances_m))
    if stabilization_distance_m <= segment_end_m:
        return factors

    for i, d in enumerate(distances_m):
        if d < stabilization_distance_m:
            ratio = (d - segment_end_m) / (stabilization_distance_m - segment_end_m)
            factors[i] = np.clip(ratio, 0.0, 1.0) ** exponent

    return factors


# ---------------------------------------------------------------------------
# Crab angle
# ---------------------------------------------------------------------------

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

def build_trajectory(cfg: TrajectoryConfig, ou: OUParams,
                     ltp_lat, ltp_lon, ltp_alt,
                     runway_heading_deg, runway_back_azimuth_deg):
    """
    Genere une trajectoire d'approche complete.

    :param cfg: parametres utilisateur (du XML)
    :param ou: hyperparametres OU (du code)
    :param ltp_lat, ltp_lon, ltp_alt: position LTP de la piste
    :param runway_heading_deg: azimut avant de la piste
    :param runway_back_azimuth_deg: azimut arriere (direction d'approche)
    :return: list de tuples (lon, lat, alt, yaw, pitch, roll)
    """
    # --- Timeline spatiale ---
    distances_m, dt_array = compute_spatial_timeline(cfg)
    n_frames = len(distances_m)

    # --- Scaling turbulence ---
    turb_scale = 0.2 + 0.8 * max(0.0, cfg.turbulence_intensity)

    # --- OU deviations par canal ---
    def _ou(std, mean=0.0):
        return generate_ou_process(
            n_steps=n_frames,
            dt_array=dt_array,
            correlation_time=cfg.correlation_time_s,
            std=std * turb_scale,
            mean=mean,
        )

    alpha_v_dev = _ou(ou.std_alpha_v_deg)
    alpha_h_dev = _ou(ou.std_alpha_h_deg)
    yaw_dev     = _ou(ou.std_yaw_deg)
    pitch_dev   = _ou(ou.std_pitch_deg)
    roll_dev    = _ou(ou.std_roll_deg)

    # --- Convergence ---
    conv = compute_convergence_factors(
        distances_m, cfg.segment_end_m, cfg.stabilization_distance_m
    )
    alpha_v_dev *= conv
    alpha_h_dev *= conv
    yaw_dev     *= conv
    pitch_dev   *= conv
    roll_dev    *= conv

    # --- Crab angle par frame ---
    crab_angles = np.array([
        compute_crab_angle(
            cfg.wind_speed_kts, cfg.wind_direction_deg,
            runway_heading_deg, cfg.ground_speed_kts
        ) * conv[i]
        for i in range(n_frames)
    ])

    # --- Assemblage des poses ---
    g = pyproj.Geod(ellps='WGS84')
    flight_data = []

    for i in range(n_frames):
        dist = distances_m[i]

        # Altitude via angle vertical + deviation OU
        alpha_v = ou.alpha_v_deg + alpha_v_dev[i]
        alpha_v = np.clip(alpha_v, -6.0, -0.5)
        alt = ltp_alt + (-np.tan(np.deg2rad(alpha_v)) * (dist + ou.dist_ap_m))
        alt = max(alt, ltp_alt + 1.0)

        # Position horizontale via angle horizontal + deviation OU
        alpha_h = ou.alpha_h_deg + alpha_h_dev[i]
        lon, lat, _ = g.fwd(
            ltp_lon, ltp_lat,
            runway_back_azimuth_deg + alpha_h,
            dist, radians=False
        )

        # Orientation
        yaw   = runway_heading_deg + crab_angles[i] + yaw_dev[i]
        pitch = 90 + ou.pitch_deg + pitch_dev[i]
        roll  = ou.roll_deg + roll_dev[i]

        flight_data.append((lon, lat, alt, yaw, pitch, roll))

    return flight_data, distances_m, ltp_alt
