"""
camera_sensor_errors.py — Simulation de fautes capteur camera embarquee avion.

Applique des degradations realistes a des images (GES, X-Plane, MSFS...)
avant inference YOLOv8, pour evaluer la robustesse du modele.

Usage standalone:
    python yolo/camera_sensor_errors.py --images runs/LFPO_24/footage/ --output runs/LFPO_24/degraded/ --errors gaussian_noise,motion_blur
    python yolo/camera_sensor_errors.py --images runs/LFPO_24/footage/ --output runs/LFPO_24/degraded/ --preset moderate
    python yolo/camera_sensor_errors.py --list

Usage depuis le code:
    from yolo.camera_sensor_errors import apply_errors, PRESETS
    img_degraded = apply_errors(img, ["gaussian_noise", "vignetting"], severity=0.5)
"""

import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Callable

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Fonctions de degradation individuelles
# ---------------------------------------------------------------------------
# Chaque fonction prend (image BGR uint8, severity 0-1) et retourne image BGR uint8.

def gaussian_noise(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Bruit gaussien capteur (amplifie en basse lumiere)."""
    sigma = 5 + severity * 45  # 5-50
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def shot_noise(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Shot noise (bruit photonique, plus visible dans les zones sombres)."""
    scale = max(0.05, 1.0 - severity * 0.95)  # 1.0 → 0.05
    noisy = np.random.poisson(img.astype(np.float32) * scale) / scale
    return np.clip(noisy, 0, 255).astype(np.uint8)


def salt_pepper(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Pixels morts / chauds (sel & poivre)."""
    ratio = 0.001 + severity * 0.029  # 0.1% - 3%
    out = img.copy()
    n_pixels = img.shape[0] * img.shape[1]
    n_salt = int(n_pixels * ratio / 2)
    n_pepper = int(n_pixels * ratio / 2)
    # Salt (pixels blancs)
    ys = np.random.randint(0, img.shape[0], n_salt)
    xs = np.random.randint(0, img.shape[1], n_salt)
    out[ys, xs] = 255
    # Pepper (pixels noirs)
    ys = np.random.randint(0, img.shape[0], n_pepper)
    xs = np.random.randint(0, img.shape[1], n_pepper)
    out[ys, xs] = 0
    return out


def dead_pixels(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Pixels morts persistants (clusters fixes, defaut capteur)."""
    n_clusters = int(1 + severity * 15)
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(n_clusters):
        cy, cx = np.random.randint(0, h), np.random.randint(0, w)
        r = np.random.randint(1, 4)
        y1, y2 = max(0, cy - r), min(h, cy + r + 1)
        x1, x2 = max(0, cx - r), min(w, cx + r + 1)
        # Pixel mort = noir ou bloque a une couleur
        color = 0 if np.random.random() < 0.7 else np.random.randint(200, 256)
        out[y1:y2, x1:x2] = color
    return out


def motion_blur(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Flou de bouge (vibrations structure avion)."""
    ksize = int(3 + severity * 22)  # 3 - 25 pixels
    if ksize % 2 == 0:
        ksize += 1
    # Direction aleatoire
    angle = np.random.uniform(0, 180)
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    for i in range(ksize):
        offset = i - center
        x = int(round(center + offset * cos_a))
        y = int(round(center + offset * sin_a))
        if 0 <= x < ksize and 0 <= y < ksize:
            kernel[y, x] = 1.0
    kernel /= kernel.sum() + 1e-8
    return cv2.filter2D(img, -1, kernel)


def defocus_blur(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Flou de mise au point (defocus)."""
    ksize = int(3 + severity * 18)  # 3 - 21
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def overexposure(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Surexposition (soleil face, auto-exposure lag)."""
    factor = 1.3 + severity * 1.7  # 1.3 - 3.0
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def underexposure(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Sous-exposition (ombre, transition luminosite rapide)."""
    factor = max(0.1, 0.7 - severity * 0.6)  # 0.7 → 0.1
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def vignetting(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Vignettage optique (assombrissement bords)."""
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    max_dist = np.sqrt(cx**2 + cy**2)
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2) / max_dist
    strength = 0.3 + severity * 0.6  # 0.3 - 0.9
    mask = 1.0 - strength * (dist ** 2)
    mask = np.clip(mask, 0, 1).astype(np.float32)
    return np.clip(img.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)


def chromatic_aberration(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Aberration chromatique (decalage R/B aux bords)."""
    shift = int(1 + severity * 6)  # 1 - 7 pixels
    h, w = img.shape[:2]
    b, g, r = cv2.split(img)
    # Decaler R vers l'exterieur, B vers l'interieur
    M_r = np.float32([[1, 0, shift], [0, 1, shift]])
    M_b = np.float32([[1, 0, -shift], [0, 1, -shift]])
    r_shifted = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REFLECT)
    b_shifted = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REFLECT)
    return cv2.merge([b_shifted, g, r_shifted])


def radial_distortion(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Distorsion radiale (barrel/pincushion)."""
    h, w = img.shape[:2]
    k1 = (severity - 0.5) * 0.8  # -0.4 (pincushion) a +0.4 (barrel)
    fx = fy = max(w, h)
    cx_cam, cy_cam = w / 2, h / 2
    camera_matrix = np.array([[fx, 0, cx_cam], [0, fy, cy_cam], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float32)
    return cv2.undistort(img, camera_matrix, dist_coeffs)


def lens_flare(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Lens flare simplifie (tache lumineuse)."""
    out = img.astype(np.float32)
    h, w = img.shape[:2]
    # Position aleatoire dans le tiers superieur (soleil)
    cx = np.random.randint(w // 4, 3 * w // 4)
    cy = np.random.randint(0, h // 3)
    radius = int((0.1 + severity * 0.3) * max(w, h))
    intensity = 100 + severity * 155
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float32)
    mask = np.clip(1.0 - dist / radius, 0, 1) ** 2
    out += mask[..., None] * intensity
    return np.clip(out, 0, 255).astype(np.uint8)


def banding(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Banding horizontal (interference electrique, readout noise)."""
    h, w = img.shape[:2]
    n_bands = int(5 + severity * 30)
    amplitude = 10 + severity * 40
    out = img.astype(np.float32)
    for _ in range(n_bands):
        y = np.random.randint(0, h)
        thickness = np.random.randint(1, 4)
        y2 = min(h, y + thickness)
        offset = np.random.uniform(-amplitude, amplitude)
        out[y:y2, :] += offset
    return np.clip(out, 0, 255).astype(np.uint8)


def rolling_shutter(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Effet rolling shutter (skew horizontal, vibrations haute freq)."""
    h, w = img.shape[:2]
    max_shift = int(2 + severity * 15)  # pixels de skew max
    out = np.zeros_like(img)
    # Sinusoide pour simuler vibration
    freq = np.random.uniform(1, 4)
    phase = np.random.uniform(0, 2 * np.pi)
    for y in range(h):
        shift = int(max_shift * np.sin(2 * np.pi * freq * y / h + phase))
        if shift > 0:
            out[y, shift:] = img[y, :w - shift]
            out[y, :shift] = img[y, 0:1]
        elif shift < 0:
            out[y, :w + shift] = img[y, -shift:]
            out[y, w + shift:] = img[y, -1:]
        else:
            out[y] = img[y]
    return out


def jpeg_artifacts(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Artefacts de compression JPEG (transmission video degradee)."""
    quality = int(max(5, 80 - severity * 75))  # 80 → 5
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def color_shift(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Derive de balance des blancs / temperature couleur."""
    out = img.astype(np.float32)
    shifts = np.random.uniform(-30 * severity, 30 * severity, 3)
    out += shifts[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def condensation(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Condensation / givre partiel sur objectif (flou non-uniforme + voile)."""
    h, w = img.shape[:2]
    # Voile semi-transparent
    haze = np.full_like(img, 220, dtype=np.float32)
    # Masque : plus opaque en peripherie (condensation bords)
    Y, X = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    max_dist = np.sqrt(cx**2 + cy**2)
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2) / max_dist
    alpha = severity * 0.7 * np.clip(dist * 1.5, 0, 1)
    # Flou local
    ksize = int(5 + severity * 20)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0).astype(np.float32)
    out = img.astype(np.float32)
    a = alpha[..., None]
    out = out * (1 - a) + (blurred * 0.5 + haze * 0.5) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def dirt_on_lens(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Salete / gouttes sur objectif (taches floues localisees)."""
    out = img.copy()
    h, w = img.shape[:2]
    n_spots = int(3 + severity * 15)
    for _ in range(n_spots):
        cy = np.random.randint(0, h)
        cx = np.random.randint(0, w)
        radius = int(10 + severity * 40 * np.random.uniform(0.5, 1.5))
        # Extraire ROI
        y1, y2 = max(0, cy - radius), min(h, cy + radius)
        x1, x2 = max(0, cx - radius), min(w, cx + radius)
        roi = out[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        ksize = max(3, radius // 2)
        if ksize % 2 == 0:
            ksize += 1
        blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)
        # Masque circulaire avec bords doux
        ry, rx = y2 - y1, x2 - x1
        Y_roi, X_roi = np.ogrid[:ry, :rx]
        center_y, center_x = (cy - y1), (cx - x1)
        d = np.sqrt((X_roi - center_x)**2 + (Y_roi - center_y)**2).astype(np.float32)
        mask = np.clip(1.0 - d / radius, 0, 1)[..., None]
        # Teinte brunâtre pour la salete
        tint = np.array([[[140, 160, 170]]], dtype=np.float32)
        dirty = blurred_roi.astype(np.float32) * 0.6 + tint * 0.4
        out[y1:y2, x1:x2] = np.clip(
            roi.astype(np.float32) * (1 - mask * 0.7) + dirty * mask * 0.7,
            0, 255
        ).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Registre des erreurs
# ---------------------------------------------------------------------------

ERROR_REGISTRY: dict[str, Callable] = {
    # Bruit
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "salt_pepper": salt_pepper,
    "dead_pixels": dead_pixels,
    # Flou
    "motion_blur": motion_blur,
    "defocus_blur": defocus_blur,
    "rolling_shutter": rolling_shutter,
    # Exposition
    "overexposure": overexposure,
    "underexposure": underexposure,
    # Optique
    "vignetting": vignetting,
    "chromatic_aberration": chromatic_aberration,
    "radial_distortion": radial_distortion,
    "lens_flare": lens_flare,
    # Defauts capteur
    "banding": banding,
    # Compression
    "jpeg_artifacts": jpeg_artifacts,
    # Couleur
    "color_shift": color_shift,
    # Environnement
    "condensation": condensation,
    "dirt_on_lens": dirt_on_lens,
}


# ---------------------------------------------------------------------------
# Presets (combinaisons realistes)
# ---------------------------------------------------------------------------

@dataclass
class ErrorPreset:
    """Combinaison nommee d'erreurs avec severites."""
    name: str
    description: str
    errors: dict[str, float]  # nom_erreur -> severity


PRESETS: dict[str, ErrorPreset] = {
    "light": ErrorPreset(
        name="light",
        description="Conditions normales, defauts mineurs (bruit faible, leger vignettage)",
        errors={"gaussian_noise": 0.15, "vignetting": 0.2, "chromatic_aberration": 0.1},
    ),
    "moderate": ErrorPreset(
        name="moderate",
        description="Turbulence moderee, capteur vieillissant",
        errors={
            "gaussian_noise": 0.3, "motion_blur": 0.3,
            "vignetting": 0.3, "dead_pixels": 0.2,
            "banding": 0.15,
        },
    ),
    "severe": ErrorPreset(
        name="severe",
        description="Conditions degradees (vibrations fortes, objectif sale, bruit eleve)",
        errors={
            "gaussian_noise": 0.6, "motion_blur": 0.5,
            "dirt_on_lens": 0.5, "banding": 0.4,
            "rolling_shutter": 0.4, "color_shift": 0.3,
        },
    ),
    "solar_glare": ErrorPreset(
        name="solar_glare",
        description="Soleil face camera (flare + surexposition)",
        errors={
            "lens_flare": 0.7, "overexposure": 0.4,
            "chromatic_aberration": 0.3, "vignetting": 0.2,
        },
    ),
    "low_light": ErrorPreset(
        name="low_light",
        description="Approche de nuit / faible luminosite",
        errors={
            "underexposure": 0.5, "gaussian_noise": 0.6,
            "shot_noise": 0.5, "banding": 0.3,
        },
    ),
    "icing": ErrorPreset(
        name="icing",
        description="Givre / condensation sur objectif",
        errors={
            "condensation": 0.6, "defocus_blur": 0.2,
            "gaussian_noise": 0.2,
        },
    ),
    "compression": ErrorPreset(
        name="compression",
        description="Liaison video degradee (compression forte)",
        errors={
            "jpeg_artifacts": 0.7, "color_shift": 0.2,
            "banding": 0.15,
        },
    ),
    "problematic_extreme": ErrorPreset(
        name="problematic_extreme",
        description="Toutes les erreurs simultanement a 0.3",
        errors={k: 0.3 for k in [
            "gaussian_noise", "shot_noise", "salt_pepper", "dead_pixels",
            "motion_blur", "defocus_blur", "rolling_shutter",
            "overexposure", "underexposure",
            "vignetting", "chromatic_aberration", "radial_distortion", "lens_flare",
            "banding", "jpeg_artifacts", "color_shift",
            "condensation", "dirt_on_lens",
        ]},
    ),
}


# ---------------------------------------------------------------------------
# API principale
# ---------------------------------------------------------------------------

def apply_errors(
    img: np.ndarray,
    errors: list[str],
    severity: float = 0.5,
    severities: Optional[dict[str, float]] = None,
) -> np.ndarray:
    """Applique une liste d'erreurs a une image.

    Args:
        img: Image BGR uint8.
        errors: Liste de noms d'erreurs (cles de ERROR_REGISTRY).
        severity: Severite globale par defaut (0-1).
        severities: Dict optionnel {nom_erreur: severite} pour override par erreur.

    Returns:
        Image degradee BGR uint8.
    """
    result = img.copy()
    for name in errors:
        if name not in ERROR_REGISTRY:
            print(f"[WARN] Erreur inconnue ignoree: {name}")
            continue
        s = severity
        if severities and name in severities:
            s = severities[name]
        result = ERROR_REGISTRY[name](result, s)
    return result


def apply_preset(img: np.ndarray, preset_name: str, severity_scale: float = 1.0) -> np.ndarray:
    """Applique un preset d'erreurs.

    Args:
        img: Image BGR uint8.
        preset_name: Nom du preset (cle de PRESETS).
        severity_scale: Multiplicateur de severite (0.5 = moitie, 2.0 = double).

    Returns:
        Image degradee BGR uint8.
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Preset inconnu: {preset_name}. Disponibles: {list(PRESETS.keys())}")
    preset = PRESETS[preset_name]
    scaled = {k: min(1.0, v * severity_scale) for k, v in preset.errors.items()}
    return apply_errors(img, list(scaled.keys()), severities=scaled)


def process_directory(
    input_dir: str,
    output_dir: str,
    errors: Optional[list[str]] = None,
    preset: Optional[str] = None,
    severity: float = 0.5,
    severity_scale: float = 1.0,
) -> int:
    """Traite toutes les images d'un dossier.

    Returns:
        Nombre d'images traitees.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    images = sorted(f for f in input_path.iterdir() if f.suffix.lower() in extensions)

    if not images:
        print(f"Aucune image trouvee dans {input_dir}")
        return 0

    count = 0
    for img_file in images:
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"[WARN] Impossible de lire: {img_file}")
            continue

        if preset:
            result = apply_preset(img, preset, severity_scale)
        elif errors:
            result = apply_errors(img, errors, severity)
        else:
            print("[WARN] Ni erreurs ni preset specifies, copie brute.")
            result = img

        out_file = output_path / img_file.name
        cv2.imwrite(str(out_file), result)
        count += 1

    print(f"{count}/{len(images)} images traitees → {output_dir}")
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def list_errors():
    """Affiche toutes les erreurs et presets disponibles."""
    print("=== Erreurs capteur disponibles ===\n")
    for name, func in ERROR_REGISTRY.items():
        doc = (func.__doc__ or "").strip().split('\n')[0]
        print(f"  {name:25s} {doc}")

    print("\n=== Presets ===\n")
    for name, preset in PRESETS.items():
        errors_str = ", ".join(f"{k}({v:.1f})" for k, v in preset.errors.items())
        print(f"  {name:15s} {preset.description}")
        print(f"  {'':15s} → {errors_str}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Simulation de fautes capteur camera embarquee avion."
    )
    parser.add_argument("--images", type=str, help="Dossier d'images source")
    parser.add_argument("--output", type=str, help="Dossier de sortie")
    parser.add_argument(
        "--errors", type=str, default=None,
        help="Erreurs a appliquer (virgules). Ex: gaussian_noise,motion_blur"
    )
    parser.add_argument(
        "--preset", type=str, default=None,
        help=f"Preset a appliquer. Choix: {', '.join(PRESETS.keys())}"
    )
    parser.add_argument(
        "--severity", type=float, default=0.5,
        help="Severite globale 0-1 (defaut: 0.5)"
    )
    parser.add_argument(
        "--severity-scale", type=float, default=1.0,
        help="Multiplicateur severite pour presets (defaut: 1.0)"
    )
    parser.add_argument("--list", action="store_true", help="Lister erreurs et presets")

    args = parser.parse_args()

    if args.list:
        list_errors()
        return

    if not args.images or not args.output:
        parser.error("--images et --output requis (ou --list)")

    error_list = args.errors.split(",") if args.errors else None

    if not error_list and not args.preset:
        parser.error("Specifier --errors ou --preset")

    process_directory(
        input_dir=args.images,
        output_dir=args.output,
        errors=error_list,
        preset=args.preset,
        severity=args.severity,
        severity_scale=args.severity_scale,
    )


if __name__ == "__main__":
    main()
