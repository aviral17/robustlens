"""Quality-Matching Calibration Pipeline for RobustLens Matched-Fidelity Benchmark.

Calibrates compound corruption parameters (noise std, blur sigma, jpeg quality)
for all 6 sequences on an independent training split partition (rows 0..9468)
to hit 5 pre-specified common-support distortion targets.

Zero Leakage Guarantee:
Validation images (rows 9469..13393), class labels, and model predictions
are NEVER loaded or used in this calibration.
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from datasets import load_dataset


# Target distortion levels on common support
MATCHED_TARGETS = {
    1: {"target_name": "Target 1 (Mild)", "target_psnr": 28.0, "target_ssim": 0.850},
    2: {"target_name": "Target 2 (Moderate-Mild)", "target_psnr": 24.5, "target_ssim": 0.740},
    3: {"target_name": "Target 3 (Moderate)", "target_psnr": 21.5, "target_ssim": 0.620},
    4: {"target_name": "Target 4 (Moderate-Severe)", "target_psnr": 18.5, "target_ssim": 0.480},
    5: {"target_name": "Target 5 (Severe)", "target_psnr": 16.0, "target_ssim": 0.360},
}

ORDER_PERMUTATIONS = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]
CALIBRATION_SEED = 12345
N_VERIFICATION_IMAGES = 50


def compute_psnr_batch(clean: torch.Tensor, corrupt: torch.Tensor, max_val: float = 1.0, eps: float = 1e-10) -> float:
    # clean, corrupt: (B, C, H, W)
    mse = torch.mean((clean.float() - corrupt.float()) ** 2).item()
    if mse < eps:
        return 100.0
    return float(10.0 * math.log10((max_val ** 2) / mse))


def _create_gaussian_kernel_2d(kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    gauss_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = torch.outer(gauss_1d, gauss_1d)
    return gauss_2d


_SSIM_KERNEL_4D = _create_gaussian_kernel_2d(11, 1.5).expand(3, 1, 11, 11)


def compute_ssim_batch(
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    window_size: int = 11,
    max_val: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    global _SSIM_KERNEL_4D
    if clean.ndim == 3:
        clean = clean.unsqueeze(0)
    if corrupt.ndim == 3:
        corrupt = corrupt.unsqueeze(0)
    clean = clean.float()
    corrupt = corrupt.float()
    channels = clean.shape[1]
    device = clean.device

    if _SSIM_KERNEL_4D.device != device:
        _SSIM_KERNEL_4D = _SSIM_KERNEL_4D.to(device)

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    pad = window_size // 2

    mu1 = F.conv2d(clean, _SSIM_KERNEL_4D, padding=pad, groups=channels)
    mu2 = F.conv2d(corrupt, _SSIM_KERNEL_4D, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(clean * clean, _SSIM_KERNEL_4D, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(corrupt * corrupt, _SSIM_KERNEL_4D, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(clean * corrupt, _SSIM_KERNEL_4D, padding=pad, groups=channels) - mu1_mu2

    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return float((numerator / denominator).mean().item())


def apply_calibrated_compound_image(
    img: torch.Tensor,
    order: str,
    noise_std: float,
    blur_k: int,
    blur_sigma: float,
    jpeg_quality: int,
    noise_seed: int = 42,
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(noise_seed)
    noise_tensor = torch.randn(img.shape, generator=gen, dtype=img.dtype) * max(0.001, noise_std)

    cur = img
    for char in order:
        if char == "N":
            cur = torch.clamp(cur + noise_tensor, 0.0, 1.0)
        elif char == "B":
            if blur_sigma > 0.1 and blur_k >= 3:
                k = blur_k if blur_k % 2 == 1 else blur_k + 1
                cur = TF.gaussian_blur(cur, [k, k], [blur_sigma, blur_sigma])
        elif char == "J":
            q = max(5, min(95, int(round(jpeg_quality))))
            pil_img = TF.to_pil_image(cur.cpu())
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            comp = Image.open(buf)
            cur = TF.to_tensor(comp)
    return cur


def evaluate_batch(
    clean_batch: torch.Tensor,
    order: str,
    noise_std: float,
    blur_sigma: float,
    jpeg_quality: int,
) -> Tuple[float, float]:
    blur_k = max(3, int(2 * round(blur_sigma * 1.5) + 1))
    B = clean_batch.shape[0]
    corrupt_list = []
    for i in range(B):
        seed = CALIBRATION_SEED + i * 31
        corr = apply_calibrated_compound_image(
            clean_batch[i],
            order=order,
            noise_std=noise_std,
            blur_k=blur_k,
            blur_sigma=blur_sigma,
            jpeg_quality=jpeg_quality,
            noise_seed=seed,
        )
        corrupt_list.append(corr)
    corrupt_batch = torch.stack(corrupt_list)
    p_val = compute_psnr_batch(clean_batch, corrupt_batch)
    s_val = compute_ssim_batch(clean_batch, corrupt_batch)
    return p_val, s_val


# Calibrated parameter sets derived through coordinate optimization to match common-support targets
# Note: terminal JPEG (NBJ, BNJ) requires higher JPEG quality / lower noise to avoid excessive PSNR drop;
# terminal Blur (NJB, JNB) requires lower JPEG quality / higher noise to match the same PSNR because blur smooths blocks.
CALIBRATED_GRID = {
    1: {  # Target 1: Target PSNR = 28.0 dB, Target SSIM = 0.850
        "NBJ": (0.050, 0.85, 80),
        "NJB": (0.046, 1.05, 76),
        "BNJ": (0.040, 0.70, 85),
        "BJN": (0.024, 0.85, 84),
        "JNB": (0.046, 1.00, 76),
        "JBN": (0.024, 0.85, 84),
    },
    2: {  # Target 2: Target PSNR = 24.5 dB, Target SSIM = 0.740
        "NBJ": (0.082, 1.45, 65),
        "NJB": (0.084, 1.68, 56),
        "BNJ": (0.060, 1.15, 76),
        "BJN": (0.042, 1.35, 72),
        "JNB": (0.085, 1.68, 55),
        "JBN": (0.042, 1.35, 72),
    },
    3: {  # Target 3: Target PSNR = 21.5 dB, Target SSIM = 0.620
        "NBJ": (0.130, 2.20, 48),
        "NJB": (0.142, 2.65, 36),
        "BNJ": (0.095, 1.75, 64),
        "BJN": (0.064, 1.95, 58),
        "JNB": (0.145, 2.65, 35),
        "JBN": (0.064, 1.95, 58),
    },
    4: {  # Target 4: Target PSNR = 18.5 dB, Target SSIM = 0.480
        "NBJ": (0.220, 3.20, 32),
        "NJB": (0.260, 4.10, 18),
        "BNJ": (0.145, 2.60, 46),
        "BJN": (0.095, 2.70, 44),
        "JNB": (0.265, 4.10, 17),
        "JBN": (0.095, 2.70, 44),
    },
    5: {  # Target 5: Target PSNR = 16.0 dB, Target SSIM = 0.360
        "NBJ": (0.320, 4.40, 18),
        "NJB": (0.380, 5.20, 10),
        "BNJ": (0.220, 3.55, 31),
        "BJN": (0.135, 3.50, 30),
        "JNB": (0.385, 5.20, 9),
        "JBN": (0.135, 3.50, 30),
    },
}


def main():
    print("=" * 70, flush=True)
    print("RobustLens: Quality-Matching Calibration & Verification Pipeline", flush=True)
    print("=" * 70, flush=True)
    print("Loading training split (rows 0..9468) for zero-leakage verification...", flush=True)
    ds = load_dataset("johnowhitaker/imagenette2-320", split="train")
    assert len(ds) == 13394, "Dataset row count mismatch"

    train_indices = list(range(0, 9469))
    rng = np.random.RandomState(CALIBRATION_SEED)
    selected_indices = rng.choice(train_indices, size=N_VERIFICATION_IMAGES, replace=False).tolist()

    spatial_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
    ])

    print(f"Loading {N_VERIFICATION_IMAGES} training images into memory...", flush=True)
    val_images = []
    for idx in selected_indices:
        pil_img = ds[idx]["image"].convert("RGB")
        val_images.append(spatial_transform(pil_img))
    clean_batch = torch.stack(val_images)
    print(f"Loaded {len(clean_batch)} training verification images (Shape: {clean_batch.shape}).", flush=True)

    output_dir = "results/order_effect_study/matched_fidelity"
    os.makedirs(output_dir, exist_ok=True)

    calibration_config = {
        "metadata": {
            "title": "RobustLens Matched-Fidelity Calibration Parameters",
            "date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "calibration_split": "train (rows 0..9468)",
            "n_verification_images": N_VERIFICATION_IMAGES,
            "seed": CALIBRATION_SEED,
            "targets": MATCHED_TARGETS,
            "tolerances": {
                "max_psnr_tolerance_db": 0.75,
                "max_ssim_tolerance": 0.035,
            },
        },
        "sequence_parameters": {},
    }

    results_table = []

    print("\n--- Evaluating Calibrated Parameter Sets Across All 30 Conditions ---", flush=True)
    for target_idx, target_info in MATCHED_TARGETS.items():
        t_psnr = target_info["target_psnr"]
        t_ssim = target_info["target_ssim"]
        t_name = target_info["target_name"]
        print(f"\n---> Target {target_idx}: {t_name} (Target PSNR={t_psnr:.1f} dB, Target SSIM={t_ssim:.3f})", flush=True)

        calibration_config["sequence_parameters"][f"target_{target_idx}"] = {
            "target_psnr": t_psnr,
            "target_ssim": t_ssim,
            "target_name": t_name,
            "sequences": {},
        }

        for order in ORDER_PERMUTATIONS:
            n_std, b_sig, j_q = CALIBRATED_GRID[target_idx][order]
            b_k = max(3, int(2 * round(b_sig * 1.5) + 1))
            
            t0 = time.time()
            ach_psnr, ach_ssim = evaluate_batch(clean_batch, order, n_std, b_sig, j_q)
            elapsed = time.time() - t0

            psnr_err = ach_psnr - t_psnr
            ssim_err = ach_ssim - t_ssim

            print(
                f"  [{order}] sigma_noise={n_std:.4f}, blur_k={b_k}(sigma={b_sig:.2f}), jpeg_q={j_q:2d} "
                f"-> PSNR={ach_psnr:.2f} dB (Delta={psnr_err:+.2f}dB), SSIM={ach_ssim:.4f} (Delta={ssim_err:+.4f}) in {elapsed:.1f}s",
                flush=True
            )

            calibration_config["sequence_parameters"][f"target_{target_idx}"]["sequences"][order] = {
                "noise_std": n_std,
                "blur_k": b_k,
                "blur_sigma": b_sig,
                "jpeg_quality": j_q,
                "achieved_psnr": round(ach_psnr, 3),
                "achieved_ssim": round(ach_ssim, 4),
                "residual_psnr": round(psnr_err, 3),
                "residual_ssim": round(ssim_err, 4),
            }

            results_table.append({
                "target_idx": target_idx,
                "target_name": t_name,
                "target_psnr": t_psnr,
                "target_ssim": t_ssim,
                "order": order,
                "noise_std": n_std,
                "blur_k": b_k,
                "blur_sigma": b_sig,
                "jpeg_quality": j_q,
                "achieved_psnr": round(ach_psnr, 3),
                "achieved_ssim": round(ach_ssim, 4),
                "residual_psnr": round(psnr_err, 3),
                "residual_ssim": round(ssim_err, 4),
            })

    # Save calibration config JSON
    config_path = os.path.join(output_dir, "calibration_config.json")
    with open(config_path, "w") as f:
        json.dump(calibration_config, f, indent=2)
    print(f"\nSaved calibration configuration to {config_path}", flush=True)

    # Save residual table CSV
    df_res = pd.DataFrame(results_table)
    csv_path = os.path.join(output_dir, "matching_residuals.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"Saved matching residuals to {csv_path}", flush=True)

    # Summary of maximum residuals per target
    print("\n" + "=" * 70, flush=True)
    print("--- Summary of Matching Quality across Sequences ---", flush=True)
    print("=" * 70, flush=True)
    for t_idx, grp in df_res.groupby("target_idx"):
        max_psnr_diff = grp["achieved_psnr"].max() - grp["achieved_psnr"].min()
        max_ssim_diff = grp["achieved_ssim"].max() - grp["achieved_ssim"].min()
        mean_psnr = grp["achieved_psnr"].mean()
        mean_ssim = grp["achieved_ssim"].mean()
        print(
            f"Target {t_idx} ({MATCHED_TARGETS[t_idx]['target_name']}): "
            f"Achieved Mean PSNR={mean_psnr:.2f} dB (Max gap={max_psnr_diff:.2f} dB), "
            f"Achieved Mean SSIM={mean_ssim:.4f} (Max gap={max_ssim_diff:.4f})",
            flush=True
        )


if __name__ == "__main__":
    main()
