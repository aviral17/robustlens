"""Validate distortion fingerprints: Controlled diagnostic suite.

Tests demanded by scientific audit:
1. JPEG boundary statistic (B_JPEG) across:
   - Clean natural images
   - Pure Gaussian noise
   - Pure Gaussian blur
   - Known JPEG quality settings (q=95 down to q=10)
   - Phase shifts (offsets 0..7 pixels) to verify grid alignment
   - Natural step edges on-grid vs off-grid
   - Resizing and interpolation impact
2. Fourier frequency bands:
   - Mathematical specification & windowing analysis (Rectangular vs Hann)
   - Sensitivity analysis across cutoff thresholds (r_low in [0.10, 0.15, 0.20], r_high in [0.40, 0.45, 0.50])
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import numpy as np
import scipy.signal
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "results" / "matched_fidelity_study"
OUTPUT_JSON = OUTPUT_DIR / "fingerprint_validation_diagnostics.json"


def compute_jpeg_block_ratio(img_np: np.ndarray, phase_x: int = 7, phase_y: int = 7) -> float:
    """Calculates horizontal and vertical discontinuity ratio for arbitrary grid phase."""
    gray = img_np.mean(axis=2).astype(np.float32) if img_np.ndim == 3 else img_np.astype(np.float32)

    diff_h = np.abs(gray[:, 1:] - gray[:, :-1])
    mask_h = np.zeros(diff_h.shape[1], dtype=bool)
    mask_h[phase_x::8] = True
    b_h = diff_h[:, mask_h].mean() if mask_h.sum() > 0 else 0.0
    i_h = diff_h[:, ~mask_h].mean() if (~mask_h).sum() > 0 else 1.0

    diff_v = np.abs(gray[1:, :] - gray[:-1, :])
    mask_v = np.zeros(diff_v.shape[0], dtype=bool)
    mask_v[phase_y::8] = True
    b_v = diff_v[mask_v, :].mean() if mask_v.sum() > 0 else 0.0
    i_v = diff_v[~mask_v, :].mean() if (~mask_v).sum() > 0 else 1.0

    return float(((b_h / (i_h + 1e-6)) + (b_v / (i_v + 1e-6))) / 2.0)


def compress_jpeg(img_pil: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf)


def main():
    print("=" * 80)
    print("VALIDATING DISTORTION FINGERPRINTS: CONTROLLED TESTS")
    print("=" * 80)

    # Load 20 clean calibration images
    ds = load_dataset("johnowhitaker/imagenette2-320", split="train")
    spatial = T.Compose([T.Resize(256), T.CenterCrop(224)])
    clean_pils = [spatial(ds[i]["image"].convert("RGB")) for i in range(20)]

    # 1. JPEG Metric on Clean Images
    clean_ratios = [compute_jpeg_block_ratio(np.array(im) / 255.0) for im in clean_pils]
    mean_clean_ratio = float(np.mean(clean_ratios))
    std_clean_ratio = float(np.std(clean_ratios))

    # 2. Pure Gaussian Noise
    noise_ratios = []
    for im in clean_pils:
        arr = np.array(im) / 255.0
        noisy = np.clip(arr + np.random.normal(0, 0.10, arr.shape), 0.0, 1.0)
        noise_ratios.append(compute_jpeg_block_ratio(noisy))
    mean_noise_ratio = float(np.mean(noise_ratios))

    # 3. Pure Gaussian Blur
    blur_ratios = []
    for im in clean_pils:
        t = TF.to_tensor(im)
        blurred = TF.gaussian_blur(t, [7, 7], [2.5, 2.5])
        blur_ratios.append(compute_jpeg_block_ratio(blurred.permute(1, 2, 0).numpy()))
    mean_blur_ratio = float(np.mean(blur_ratios))

    # 4. Known JPEG Qualities Sweep
    quality_sweep = {}
    for q in [95, 80, 60, 40, 20, 10, 5]:
        r_list = [compute_jpeg_block_ratio(np.array(compress_jpeg(im, q)) / 255.0) for im in clean_pils]
        quality_sweep[q] = {
            "mean_block_ratio": round(float(np.mean(r_list)), 4),
            "std_block_ratio": round(float(np.std(r_list)), 4),
        }

    # 5. Grid Phase Shift Diagnostic (Phase 0..7) on JPEG q=20 images
    phase_sweep = {}
    sample_jpeg = np.array(compress_jpeg(clean_pils[0], 20)) / 255.0
    for ph in range(8):
        phase_sweep[f"phase_{ph}"] = round(compute_jpeg_block_ratio(sample_jpeg, phase_x=ph, phase_y=ph), 4)

    # 6. Natural Edges vs Grid Step Discontinuities
    # Synthetic image: smooth gradient with a single sharp natural vertical line at column 50 (off-grid) vs column 55 (on-grid)
    syn_off = np.tile(np.linspace(0, 1, 224, dtype=np.float32), (224, 1))
    syn_off[:, 50] = 0.0  # Off-grid step
    ratio_off = compute_jpeg_block_ratio(syn_off)

    syn_on = np.tile(np.linspace(0, 1, 224, dtype=np.float32), (224, 1))
    syn_on[:, 55] = 0.0  # On-grid step (col 55 = 7*8 - 1)
    ratio_on = compute_jpeg_block_ratio(syn_on)

    # 7. Resizing & Interpolation Impact on JPEG artifacts
    # Take JPEG q=20 image, downsample to 200x200 then bicubic upsample to 224x224
    interp_ratios = []
    for im in clean_pils:
        jpg = compress_jpeg(im, 20)
        resampled = jpg.resize((200, 200), Image.BICUBIC).resize((224, 224), Image.BICUBIC)
        interp_ratios.append(compute_jpeg_block_ratio(np.array(resampled) / 255.0))
    mean_interp_ratio = float(np.mean(interp_ratios))

    # 8. Fourier Residual Sensitivity Analysis
    fourier_sensitivity = {}
    # Use 10 clean vs corrupted pairs
    corr_samples = [compress_jpeg(im, 30) for im in clean_pils[:10]]
    deltas = [(np.array(c).mean(2)/255.0) - (np.array(o).mean(2)/255.0) for c, o in zip(corr_samples, clean_pils[:10])]

    for r_low_cand in [0.10, 0.15, 0.20]:
        for r_high_cand in [0.40, 0.45, 0.50]:
            e_low_list, e_med_list, e_high_list = [], [], []
            for d in deltas:
                fft2 = np.fft.fftshift(np.fft.fft2(d))
                power = np.abs(fft2) ** 2
                H, W = d.shape
                y, x = np.ogrid[-H/2:H/2, -W/2:W/2]
                r_norm = np.sqrt(x**2 + y**2) / np.sqrt((H/2)**2 + (W/2)**2)
                tot = power.sum() + 1e-12

                e_low_list.append(float(power[r_norm < r_low_cand].sum() / tot))
                e_med_list.append(float(power[(r_norm >= r_low_cand) & (r_norm < r_high_cand)].sum() / tot))
                e_high_list.append(float(power[r_norm >= r_high_cand].sum() / tot))

            fourier_sensitivity[f"low_{r_low_cand:.2f}_high_{r_high_cand:.2f}"] = {
                "mean_e_low": round(float(np.mean(e_low_list)), 4),
                "mean_e_med": round(float(np.mean(e_med_list)), 4),
                "mean_e_high": round(float(np.mean(e_high_list)), 4),
            }

    # Hann Windowing vs Rectangular Windowing
    window_comp = {}
    for win_type in ["rectangular", "hann"]:
        e_low_list = []
        for d in deltas:
            if win_type == "hann":
                w2d = np.outer(np.hanning(d.shape[0]), np.hanning(d.shape[1]))
                d_win = d * w2d
            else:
                d_win = d
            fft2 = np.fft.fftshift(np.fft.fft2(d_win))
            power = np.abs(fft2) ** 2
            tot = power.sum() + 1e-12
            H, W = d.shape
            y, x = np.ogrid[-H/2:H/2, -W/2:W/2]
            r_norm = np.sqrt(x**2 + y**2) / np.sqrt((H/2)**2 + (W/2)**2)
            e_low_list.append(float(power[r_norm < 0.15].sum() / tot))
        window_comp[win_type] = round(float(np.mean(e_low_list)), 4)

    diagnostics = {
        "jpeg_boundary_diagnostics": {
            "clean_images": {"mean": round(mean_clean_ratio, 4), "std": round(std_clean_ratio, 4), "expected": "~1.00"},
            "pure_gaussian_noise": {"mean": round(mean_noise_ratio, 4), "expected": "~1.00"},
            "pure_gaussian_blur": {"mean": round(mean_blur_ratio, 4), "expected": "~1.00"},
            "quality_sweep": quality_sweep,
            "grid_phase_sweep_jpeg_q20": phase_sweep,
            "synthetic_natural_step_offgrid": round(ratio_off, 4),
            "synthetic_step_ongrid": round(ratio_on, 4),
            "resampling_bicubic_smoothed_ratio": round(mean_interp_ratio, 4),
        },
        "fourier_features_diagnostics": {
            "windowing_comparison_e_low": window_comp,
            "boundary_sensitivity": fourier_sensitivity,
        }
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"Validation diagnostics written to {OUTPUT_JSON}")
    print("\n--- Summary of Fingerprint Diagnostics ---")
    print(f"Clean Images B_JPEG: {mean_clean_ratio:.4f} (Null baseline: 1.0000)")
    print(f"Pure Noise B_JPEG:   {mean_noise_ratio:.4f}")
    print(f"Pure Blur B_JPEG:    {mean_blur_ratio:.4f}")
    print(f"JPEG Quality 80:     {quality_sweep[80]['mean_block_ratio']:.4f}")
    print(f"JPEG Quality 20:     {quality_sweep[20]['mean_block_ratio']:.4f}")
    print(f"JPEG Quality 10:     {quality_sweep[10]['mean_block_ratio']:.4f}")
    print(f"Phase 7 (On-grid):   {phase_sweep['phase_7']:.4f} vs Off-grid average: {np.mean([phase_sweep[f'phase_{p}'] for p in range(7)]):.4f}")


if __name__ == "__main__":
    main()
