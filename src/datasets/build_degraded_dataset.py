"""Genera la versione degradata di un dataset di immagini ad alta risoluzione.

La pipeline di degradazione e' la stessa usata per tutti e tre i modelli:
blur gaussiano -> downsample x4 -> rumore gaussiano -> compressione JPEG.

L'ultimo passaggio di upsampling serve solo ai modelli di denoising (Uformer,
NAFNet), che lavorano a risoluzione piena: per la super-resolution (SRGAN)
l'immagine va lasciata a 1/4 della dimensione originale, quindi si usa
--keep-low-res.

Esempi:
    python -m src.datasets.build_degraded_dataset \
        --input data/DIV2K_train_HR --output data/DIV2K_train_LR_noisy --keep-low-res
    python -m src.datasets.build_degraded_dataset \
        --input data/DIV2K_train_HR --output data/DIV2K_train_degraded
"""

import argparse
import glob
import os
import random
from typing import Optional, Tuple

import cv2
import numpy as np

BLUR_KERNEL_CHOICES = [3, 5, 7]  # solo dimensioni dispari
DEFAULT_BLUR_SIGMA = (0.5, 2.5)
DEFAULT_NOISE = (5.0, 25.0)  # deviazione standard del rumore, in livelli di grigio
DEFAULT_JPEG = (60, 95)  # 95 = alta qualita', 60 = artefatti visibili


def set_seed(seed: int) -> None:
    """Fissa i seed di random e numpy per rendere riproducibile la degradazione."""
    random.seed(seed)
    np.random.seed(seed)


def apply_random_degradation(
    img: np.ndarray,
    scale_factor: int,
    blur_sigma: Tuple[float, float],
    noise_range: Tuple[float, float],
    jpeg_range: Tuple[int, int],
    keep_low_res: bool,
) -> np.ndarray:
    """Applica blur, downsample, rumore e compressione JPEG a una singola immagine.

    Se keep_low_res e' False l'immagine viene riportata alla risoluzione di
    partenza con interpolazione cubica, cosi' da avere una coppia pulita/degradata
    delle stesse dimensioni.
    """
    kernel_size = int(np.random.choice(BLUR_KERNEL_CHOICES))
    sigma = np.random.uniform(*blur_sigma)
    img_blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    h, w = img_blur.shape[:2]
    img_small = cv2.resize(
        img_blur, (w // scale_factor, h // scale_factor), interpolation=cv2.INTER_CUBIC
    )

    noise_level = np.random.uniform(*noise_range)
    noisy = img_small + np.random.normal(0, noise_level, img_small.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    # La compressione avviene in memoria: gli artefatti restano "impressi" nei
    # pixel, poi salviamo in PNG per non sovrapporne altri.
    quality = int(np.random.randint(*jpeg_range))
    _, buffer = cv2.imencode(".jpg", noisy, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    degraded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if keep_low_res:
        return degraded
    return cv2.resize(degraded, (w, h), interpolation=cv2.INTER_CUBIC)


def build_dataset(
    input_dir: str,
    output_dir: str,
    scale_factor: int = 4,
    blur_sigma: Tuple[float, float] = DEFAULT_BLUR_SIGMA,
    noise_range: Tuple[float, float] = DEFAULT_NOISE,
    jpeg_range: Tuple[int, int] = DEFAULT_JPEG,
    keep_low_res: bool = False,
    limit: Optional[int] = None,
) -> int:
    """Degrada tutte le immagini di input_dir e restituisce quante ne ha scritte."""
    os.makedirs(output_dir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    if not images:
        raise FileNotFoundError(f"Nessuna immagine PNG trovata in {input_dir}")
    if limit is not None:
        images = images[:limit]

    print(f"Trovate {len(images)} immagini, inizio elaborazione")
    written = 0
    for i, path in enumerate(images, start=1):
        img = cv2.imread(path)
        if img is None:
            print(f"Impossibile leggere {path}, la salto")
            continue

        degraded = apply_random_degradation(
            img, scale_factor, blur_sigma, noise_range, jpeg_range, keep_low_res
        )
        cv2.imwrite(os.path.join(output_dir, os.path.basename(path)), degraded)
        written += 1

        if i % 50 == 0:
            print(f"Processate {i}/{len(images)}")

    print(f"Fatto: {written} immagini scritte in {output_dir}")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="cartella con le immagini HR")
    parser.add_argument("--output", required=True, help="cartella di destinazione")
    parser.add_argument("--scale-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-low-res",
        action="store_true",
        help="non riporta l'immagine alla risoluzione originale (super-resolution)",
    )
    parser.add_argument(
        "--blur-sigma", type=float, nargs=2, default=DEFAULT_BLUR_SIGMA,
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--noise", type=float, nargs=2, default=DEFAULT_NOISE, metavar=("MIN", "MAX")
    )
    parser.add_argument(
        "--jpeg-quality", type=int, nargs=2, default=DEFAULT_JPEG, metavar=("MIN", "MAX")
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="processa solo le prime N immagini"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    build_dataset(
        input_dir=args.input,
        output_dir=args.output,
        scale_factor=args.scale_factor,
        blur_sigma=tuple(args.blur_sigma),
        noise_range=tuple(args.noise),
        jpeg_range=tuple(args.jpeg_quality),
        keep_low_res=args.keep_low_res,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
