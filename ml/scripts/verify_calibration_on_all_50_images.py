"""Validate frozen 18 parameter configurations across all 50 calibration images.

1. Identifies the exact 50 calibration image IDs generated from seed 12345 on train split (rows 0..9468).
2. Distinguishes the 15 grid-search images (indices 0..14) from the 35 holdout calibration images (indices 15..49).
3. Evaluates the 18 frozen configurations on all 50 calibration images (18 x 50 = 900 transformations).
4. Computes per-sequence means, standard deviations, medians, IQR (25%, 75%), min, max, and SE of mean for PSNR and SSIM.
5. Verifies tolerances against target coordinates.
6. Zero validation split leakage (validation split rows 9469..13393 untouched).
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_JSON = BASE_DIR / "results" / "matched_fidelity_study" / "calibrated_common_support_3levels.json"
OUTPUT_JSON = BASE_DIR / "results" / "matched_fidelity_study" / "calibration_50images_verification.json"
OUTPUT_CSV = BASE_DIR / "results" / "matched_fidelity_study" / "calibration_50images_per_image_residuals.csv"

CALIBRATION_SEED = 12345
SEQUENCES = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]

# SSIM Kernel
def _create_ssim_kernel(k: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    k2d = torch.outer(g, g)
    return k2d.expand(3, 1, k, k)

_SSIM_K = _create_ssim_kernel()

def compute_psnr(clean: torch.Tensor, corrupt: torch.Tensor) -> float:
    mse = torch.mean((clean.float() - corrupt.float()) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return float(10.0 * math.log10(1.0 / mse))

def compute_ssim(clean: torch.Tensor, corrupt: torch.Tensor) -> float:
    global _SSIM_K
    if clean.ndim == 3:
        clean = clean.unsqueeze(0)
    if corrupt.ndim == 3:
        corrupt = corrupt.unsqueeze(0)
    clean = clean.float()
    corrupt = corrupt.float()

    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    mu1 = F.conv2d(clean, _SSIM_K, padding=5, groups=3)
    mu2 = F.conv2d(corrupt, _SSIM_K, padding=5, groups=3)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sig1_sq = F.conv2d(clean * clean, _SSIM_K, padding=5, groups=3) - mu1_sq
    sig2_sq = F.conv2d(corrupt * corrupt, _SSIM_K, padding=5, groups=3) - mu2_sq
    sig12 = F.conv2d(clean * corrupt, _SSIM_K, padding=5, groups=3) - mu12

    num = (2 * mu12 + c1) * (2 * sig12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sig1_sq + sig2_sq + c2)
    return float((num / den).mean().item())

def apply_compound_sequence(img: torch.Tensor, order: str, noise_std: float, blur_sigma: float, jpeg_quality: int, noise_seed: int = 42):
    gen = torch.Generator().manual_seed(noise_seed)
    noise = torch.randn(img.shape, generator=gen, dtype=img.dtype) * max(0.0001, noise_std)

    cur = img
    k = max(3, int(2 * round(blur_sigma * 1.5) + 1))
    if k % 2 == 0:
        k += 1

    for char in order:
        if char == "N":
            cur = torch.clamp(cur + noise, 0.0, 1.0)
        elif char == "B":
            if blur_sigma > 0.05:
                cur = TF.gaussian_blur(cur, [k, k], [blur_sigma, blur_sigma])
        elif char == "J":
            q = max(5, min(95, int(round(jpeg_quality))))
            pil_img = TF.to_pil_image(cur.cpu())
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            cur = TF.to_tensor(Image.open(buf))
    return cur


def main():
    print("=" * 80)
    print("VALIDATING 18 FROZEN CALIBRATION CONFIGURATIONS ON ALL 50 CALIBRATION IMAGES")
    print("=" * 80)

    # 1. Deterministic Calibration Image Selection
    rng = np.random.RandomState(CALIBRATION_SEED)
    cal_indices = rng.choice(range(0, 9469), size=50, replace=False).tolist()
    grid_indices = cal_indices[:15]
    holdout_indices = cal_indices[15:]

    print(f"Calibration Pool Size: 50 images (from train split 0..9468, seed={CALIBRATION_SEED})")
    print(f"Grid-Search Subset (N=15): {grid_indices}")
    print(f"Independent Calibration Holdout (N=35): {holdout_indices[:5]}... ({len(holdout_indices)} images)")

    # Load images
    ds_train = load_dataset("johnowhitaker/imagenette2-320", split="train")
    spatial = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    cal_tensors = [spatial(ds_train[idx]["image"].convert("RGB")) for idx in cal_indices]

    with open(CONFIG_JSON, "r") as f:
        calib_config = json.load(f)

    levels = calib_config["levels"]
    records = []

    for lvl_name, lvl_data in levels.items():
        t_psnr = lvl_data["target_psnr"]
        t_ssim = lvl_data["target_ssim"]

        for seq in SEQUENCES:
            p = lvl_data["sequences"][seq]
            n_std = p["noise_std"]
            b_sig = p["blur_sigma"]
            q_jpg = p["jpeg_quality"]

            for img_rank, (cal_id, img_t) in enumerate(zip(cal_indices, cal_tensors)):
                is_grid_img = (img_rank < 15)
                corr_t = apply_compound_sequence(
                    img_t, order=seq, noise_std=n_std, blur_sigma=b_sig, jpeg_quality=q_jpg,
                    noise_seed=CALIBRATION_SEED + img_rank * 17,
                )
                psnr_val = compute_psnr(img_t, corr_t)
                ssim_val = compute_ssim(img_t, corr_t)

                records.append({
                    "level": lvl_name,
                    "target_psnr": t_psnr,
                    "target_ssim": t_ssim,
                    "order": seq,
                    "final_op": p["final_op"],
                    "cal_image_id": cal_id,
                    "image_rank_in_pool": img_rank,
                    "subset": "grid_search_15" if is_grid_img else "calibration_holdout_35",
                    "noise_std": n_std,
                    "blur_sigma": b_sig,
                    "jpeg_quality": q_jpg,
                    "psnr": round(psnr_val, 3),
                    "ssim": round(ssim_val, 4),
                    "psnr_residual": round(psnr_val - t_psnr, 3),
                    "ssim_residual": round(ssim_val - t_ssim, 4),
                })

    df_res = pd.DataFrame(records)
    df_res.to_csv(OUTPUT_CSV, index=False)
    print(f"Evaluated {len(df_res)} image-condition pairs. Saved raw per-image residuals to {OUTPUT_CSV}")

    # Summary Statistics: All 50 images vs Grid 15 vs Holdout 35
    summary = {}
    for lvl_name, lvl_data in levels.items():
        summary[lvl_name] = {
            "target_psnr": lvl_data["target_psnr"],
            "target_ssim": lvl_data["target_ssim"],
            "sequences": {},
        }
        for seq in SEQUENCES:
            sub_all = df_res[(df_res["level"] == lvl_name) & (df_res["order"] == seq)]
            sub_15 = sub_all[sub_all["subset"] == "grid_search_15"]
            sub_35 = sub_all[sub_all["subset"] == "calibration_holdout_35"]

            p_vals_all = sub_all["psnr"].to_numpy()
            s_vals_all = sub_all["ssim"].to_numpy()

            summary[lvl_name]["sequences"][seq] = {
                "all_50_images": {
                    "psnr_mean": round(float(np.mean(p_vals_all)), 3),
                    "psnr_std": round(float(np.std(p_vals_all, ddof=1)), 3),
                    "psnr_median": round(float(np.median(p_vals_all)), 3),
                    "psnr_q25": round(float(np.percentile(p_vals_all, 25)), 3),
                    "psnr_q75": round(float(np.percentile(p_vals_all, 75)), 3),
                    "psnr_se": round(float(np.std(p_vals_all, ddof=1) / np.sqrt(50)), 3),
                    "psnr_error_vs_target": round(float(np.mean(p_vals_all) - lvl_data["target_psnr"]), 3),
                    "ssim_mean": round(float(np.mean(s_vals_all)), 4),
                    "ssim_std": round(float(np.std(s_vals_all, ddof=1)), 4),
                    "ssim_median": round(float(np.median(s_vals_all)), 4),
                    "ssim_q25": round(float(np.percentile(s_vals_all, 25)), 4),
                    "ssim_q75": round(float(np.percentile(s_vals_all, 75)), 4),
                    "ssim_se": round(float(np.std(s_vals_all, ddof=1) / np.sqrt(50)), 4),
                    "ssim_error_vs_target": round(float(np.mean(s_vals_all) - lvl_data["target_ssim"]), 4),
                    "within_prespecified_tolerances": bool(
                        abs(np.mean(p_vals_all) - lvl_data["target_psnr"]) <= 0.75 and
                        abs(np.mean(s_vals_all) - lvl_data["target_ssim"]) <= 0.035
                    ),
                },
                "grid_15_mean": {
                    "psnr": round(float(sub_15["psnr"].mean()), 3),
                    "ssim": round(float(sub_15["ssim"].mean()), 4),
                },
                "holdout_35_mean": {
                    "psnr": round(float(sub_35["psnr"].mean()), 3),
                    "ssim": round(float(sub_35["ssim"].mean()), 4),
                },
                "subsample_generalization_delta": {
                    "psnr_delta": round(float(sub_35["psnr"].mean() - sub_15["psnr"].mean()), 3),
                    "ssim_delta": round(float(sub_35["ssim"].mean() - sub_15["ssim"].mean()), 4),
                }
            }

    verification_report = {
        "metadata": {
            "title": "RobustLens Full 50-Image Calibration Validation Report",
            "seed": CALIBRATION_SEED,
            "total_calibration_images": 50,
            "grid_images_count": 15,
            "holdout_images_count": 35,
            "cal_image_ids": cal_indices,
            "grid_image_ids": grid_indices,
            "holdout_image_ids": holdout_indices,
        },
        "level_summaries": summary,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(verification_report, f, indent=2)

    print(f"\nSaved comprehensive verification JSON to {OUTPUT_JSON}")
    print("\n" + "=" * 95)
    print("ALL 50 CALIBRATION IMAGES VALIDATION TABLE (MEAN +/- SE [IQR]):")
    print("=" * 95)
    for lvl, data in summary.items():
        print(f"\n--- {lvl} (Target: {data['target_psnr']} dB, {data['target_ssim']} SSIM) ---")
        print(f"{'Order':<6} | {'Final Op':<8} | {'50-Img PSNR (dB)':<22} | {'50-Img SSIM':<20} | {'Grid 15 vs Hold 35 d':<20} | {'Within Tol?':<11}")
        print("-" * 95)
        for s, p in data["sequences"].items():
            all_s = p["all_50_images"]
            final_op = "JPEG" if s.endswith("J") else ("Blur" if s.endswith("B") else "Noise")
            p_str = f"{all_s['psnr_mean']:.2f} +/- {all_s['psnr_se']:.2f} [{all_s['psnr_q25']:.1f}-{all_s['psnr_q75']:.1f}]"
            s_str = f"{all_s['ssim_mean']:.3f} +/- {all_s['ssim_se']:.3f} [{all_s['ssim_q25']:.2f}-{all_s['ssim_q75']:.2f}]"
            d_str = f"dP: {p['subsample_generalization_delta']['psnr_delta']:+.2f}, dS: {p['subsample_generalization_delta']['ssim_delta']:+.3f}"
            print(f"{s:<6} | {final_op:<8} | {p_str:<22} | {s_str:<20} | {d_str:<20} | {'PASS' if all_s['within_prespecified_tolerances'] else 'FAIL':<11}")


if __name__ == "__main__":
    main()
