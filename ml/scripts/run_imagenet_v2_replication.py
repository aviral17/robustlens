"""Phase 2: Independent ImageNet-V2 External Replication Study.

Executes external counterfactual evaluation across all three ImageNet-V2 variants:
1. MatchedFrequency (100 images: 10 classes x 10 images)
2. Threshold0.7 (100 images: 10 classes x 10 images)
3. TopImages (100 images: 10 classes x 10 images)
Total: 300 unique images x 18 frozen conditions x 4 vision models = 21,600 evaluations.

Strict Protocol:
- All 18 corruption parameter configurations frozen from calibrated_common_support_3levels.json.
- Zero parameter retuning or recalibration on ImageNet-V2.
- True-class ImageNet-1K logit margin: M = z_true - max_{j != true} z_j.
- Exact LPIPS [-1, 1] scaling and 5 prespecified distortion fingerprints.
- Reports separate results for each ImageNet-V2 variant AND pooled external analysis.
- Evaluates Model 3 (Conventional 4-knot RCS, 27 cols) vs Model 5 (32 cols).
- Sensitivity: Reduced 2-df spline (24 cols vs 29 cols).
- Paired cluster bootstrap with B=5,000 resamples.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import timm
import lpips

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "imagenet_v2"
OUTPUT_DIR = BASE_DIR / "results" / "matched_fidelity_study" / "imagenet_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CALIB_JSON = BASE_DIR / "results" / "matched_fidelity_study" / "calibrated_common_support_3levels.json"
MANIFEST_PATH = DATA_DIR / "imagenet_v2_replication_manifest.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_CLASSES = {
    0: ("n01440764", "tench"),
    217: ("n02102040", "English springer"),
    482: ("n02979186", "cassette player"),
    491: ("n03000684", "chain saw"),
    497: ("n03028079", "church"),
    566: ("n03394916", "French horn"),
    569: ("n03417042", "garbage truck"),
    571: ("n03425413", "gas pump"),
    574: ("n03445777", "golf ball"),
    701: ("n03888257", "parachute"),
}

LEVELS = ["Level_1_Mild", "Level_2_Moderate", "Level_3_SevereCommon"]
SEQUENCES = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]
FINAL_OPS = {
    "NBJ": "JPEG", "BNJ": "JPEG",
    "NJB": "Blur", "JNB": "Blur",
    "BJN": "Noise", "JBN": "Noise",
}


def compute_psnr(clean: torch.Tensor, corrupt: torch.Tensor) -> float:
    mse = torch.mean((clean.float() - corrupt.float()) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return float(10.0 * math.log10(1.0 / mse))


def _create_ssim_kernel(k: int = 11, sigma: float = 1.5, device="cpu") -> torch.Tensor:
    coords = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    k2d = torch.outer(g, g)
    return k2d.expand(3, 1, k, k).to(device)


def compute_ssim_and_local_map(clean: torch.Tensor, corrupt: torch.Tensor, kernel: torch.Tensor):
    if clean.ndim == 3:
        clean = clean.unsqueeze(0)
    if corrupt.ndim == 3:
        corrupt = corrupt.unsqueeze(0)
    clean = clean.float().to(kernel.device)
    corrupt = corrupt.float().to(kernel.device)

    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    mu1 = F.conv2d(clean, kernel, padding=5, groups=3)
    mu2 = F.conv2d(corrupt, kernel, padding=5, groups=3)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(clean * clean, kernel, padding=5, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(corrupt * corrupt, kernel, padding=5, groups=3) - mu2_sq
    sigma12 = F.conv2d(clean * corrupt, kernel, padding=5, groups=3) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8)
    mean_ssim = float(ssim_map.mean().item())
    std_local_ssim = float(ssim_map.std().item())
    return mean_ssim, std_local_ssim


def compute_distortion_fingerprints(clean_t: torch.Tensor, corrupt_t: torch.Tensor, ssim_kernel: torch.Tensor):
    clean_np = clean_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    corrupt_np = corrupt_t.squeeze(0).permute(1, 2, 0).cpu().numpy()

    gray_clean = clean_np.mean(axis=2).astype(np.float32)
    gray_corrupt = corrupt_np.mean(axis=2).astype(np.float32)
    delta = gray_corrupt - gray_clean
    H, W = delta.shape

    # 1. 2D Fourier Power Spectrum Fractions
    fft2 = np.fft.fft2(delta)
    power = np.abs(np.fft.fftshift(fft2)) ** 2
    y, x = np.ogrid[-H/2:H/2, -W/2:W/2]
    r_norm = np.sqrt(x**2 + y**2) / np.sqrt((H/2)**2 + (W/2)**2)
    total_power = power.sum() + 1e-12

    e_low = float(power[r_norm < 0.15].sum() / total_power)
    e_med = float(power[(r_norm >= 0.15) & (r_norm < 0.45)].sum() / total_power)
    e_high = float(power[r_norm >= 0.45].sum() / total_power)

    # 2. JPEG 8x8 Block-Boundary Ratio
    diff_h = np.abs(gray_corrupt[:, 1:] - gray_corrupt[:, :-1])
    mask_h = np.zeros(diff_h.shape[1], dtype=bool)
    mask_h[7::8] = True
    b_h = diff_h[:, mask_h].mean() if mask_h.sum() > 0 else 0.0
    i_h = diff_h[:, ~mask_h].mean() if (~mask_h).sum() > 0 else 1.0

    diff_v = np.abs(gray_corrupt[1:, :] - gray_corrupt[:-1, :])
    mask_v = np.zeros(diff_v.shape[0], dtype=bool)
    mask_v[7::8] = True
    b_v = diff_v[mask_v, :].mean() if mask_v.sum() > 0 else 0.0
    i_v = diff_v[~mask_v, :].mean() if (~mask_v).sum() > 0 else 1.0

    block_ratio = float(((b_h / (i_h + 1e-6)) + (b_v / (i_v + 1e-6))) / 2.0)

    # 3. Sobel Edge Energy Difference
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    edge_clean = np.sqrt(scipy.signal.convolve2d(gray_clean, sobel_x, mode="same")**2 + scipy.signal.convolve2d(gray_clean, sobel_y, mode="same")**2).mean()
    edge_corr = np.sqrt(scipy.signal.convolve2d(gray_corrupt, sobel_x, mode="same")**2 + scipy.signal.convolve2d(gray_corrupt, sobel_y, mode="same")**2).mean()
    delta_edge = float((edge_corr - edge_clean) / (edge_clean + 1e-6))

    # 4. Local SSIM Spatial Variation
    _, std_local_ssim = compute_ssim_and_local_map(clean_t, corrupt_t, ssim_kernel)

    return {
        "fft_energy_low": round(e_low, 4),
        "fft_energy_med": round(e_med, 4),
        "fft_energy_high": round(e_high, 4),
        "jpeg_block_artifact": round(block_ratio, 4),
        "delta_edge_energy": round(delta_edge, 4),
        "std_local_ssim": round(std_local_ssim, 4),
    }


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


def conventional_rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    K = len(knots)
    k_K = knots[-1]
    k_Km1 = knots[-2]
    basis = [x]
    denom_last = max(1e-8, abs(k_K - k_Km1))
    d_last = (np.maximum(0.0, x - k_Km1)**3 - np.maximum(0.0, x - k_K)**3) / denom_last
    for j in range(K - 2):
        k_j = knots[j]
        denom_j = max(1e-8, abs(k_K - k_j))
        d_j = (np.maximum(0.0, x - k_j)**3 - np.maximum(0.0, x - k_K)**3) / denom_j
        basis.append(d_j - d_last)
    return np.column_stack(basis)


def reduced_spline_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    K = len(knots)
    k_K = knots[-1]
    k_Km1 = knots[-2]
    basis = [x]
    denom_last = max(1e-8, abs(k_K - k_Km1))
    d_last = (np.maximum(0.0, x - k_Km1)**3 - np.maximum(0.0, x - k_K)**3) / denom_last
    for j in range(1, K - 2):
        k_j = knots[j]
        denom_j = max(1e-8, abs(k_K - k_j))
        d_j = (np.maximum(0.0, x - k_j)**3 - np.maximum(0.0, x - k_K)**3) / denom_j
        basis.append(d_j - d_last)
    return np.column_stack(basis)


def build_design_matrix(df_part, spec, scaler_params, spline_knots):
    N = len(df_part)
    orders = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]
    levels = ["Level_1_Mild", "Level_2_Moderate", "Level_3_SevereCommon"]

    order_mat = np.zeros((N, len(orders) - 1))
    for i, o in enumerate(orders[1:]):
        order_mat[:, i] = (df_part["order"].to_numpy() == o).astype(float)

    level_mat = np.zeros((N, len(levels) - 1))
    for i, lvl in enumerate(levels[1:]):
        level_mat[:, i] = (df_part["level"].to_numpy() == lvl).astype(float)

    order_x_level = np.zeros((N, (len(orders) - 1) * (len(levels) - 1)))
    col_idx = 0
    for o_i in range(order_mat.shape[1]):
        for l_i in range(level_mat.shape[1]):
            order_x_level[:, col_idx] = order_mat[:, o_i] * level_mat[:, l_i]
            col_idx += 1

    p_std = (df_part["psnr"].to_numpy() - scaler_params["psnr"]["mean"]) / scaler_params["psnr"]["std"]
    s_std = (df_part["ssim"].to_numpy() - scaler_params["ssim"]["mean"]) / scaler_params["ssim"]["std"]
    l_std = (df_part["lpips"].to_numpy() - scaler_params["lpips"]["mean"]) / scaler_params["lpips"]["std"]

    intercept = np.ones((N, 1))
    base_parts = [intercept, order_mat, level_mat, order_x_level]

    spline_fn = conventional_rcs_basis if "conv" in spec else reduced_spline_basis
    p_sp = spline_fn(p_std, spline_knots["psnr"])
    s_sp = spline_fn(s_std, spline_knots["ssim"])
    l_sp = spline_fn(l_std, spline_knots["lpips"])

    if "extended" in spec:
        f_low = (df_part["fft_energy_low"].to_numpy() - scaler_params["fft_energy_low"]["mean"]) / scaler_params["fft_energy_low"]["std"]
        f_high = (df_part["fft_energy_high"].to_numpy() - scaler_params["fft_energy_high"]["mean"]) / scaler_params["fft_energy_high"]["std"]
        jpeg_b = (df_part["jpeg_block_artifact"].to_numpy() - scaler_params["jpeg_block_artifact"]["mean"]) / scaler_params["jpeg_block_artifact"]["std"]
        d_edge = (df_part["delta_edge_energy"].to_numpy() - scaler_params["delta_edge_energy"]["mean"]) / scaler_params["delta_edge_energy"]["std"]
        s_std_loc = (df_part["std_local_ssim"].to_numpy() - scaler_params["std_local_ssim"]["mean"]) / scaler_params["std_local_ssim"]["std"]
        fingerprints = np.column_stack([f_low, f_high, jpeg_b, d_edge, s_std_loc])
        return np.hstack(base_parts + [p_sp, s_sp, l_sp, fingerprints])
    else:
        return np.hstack(base_parts + [p_sp, s_sp, l_sp])


def evaluate_dataset_subset(df_sub, dataset_label, n_bootstrap=5000):
    print(f"\n--- Statistical Evaluation for: {dataset_label} (N={len(df_sub):,} records) ---")
    clusters = np.sort(df_sub["image_id"].unique())
    n_clusters = len(clusters)
    cluster_classes = df_sub.drop_duplicates("image_id").sort_values("image_id")["true_label"].to_numpy()

    rng_cv = np.random.RandomState(42)
    fold_assignments = np.zeros(n_clusters, dtype=int)
    for c in np.unique(cluster_classes):
        c_indices = np.where(cluster_classes == c)[0]
        perm = rng_cv.permutation(c_indices)
        c_splits = np.array_split(perm, 5)
        for fold_i, split_idx in enumerate(c_splits):
            fold_assignments[split_idx] = fold_i

    rng_b = np.random.RandomState(12345)
    bootstrap_cluster_choices = rng_b.randint(0, n_clusters, size=(n_bootstrap, n_clusters))
    W = np.zeros((n_bootstrap, n_clusters), dtype=float)
    for b in range(n_bootstrap):
        W[b, :] = np.bincount(bootstrap_cluster_choices[b], minlength=n_clusters)

    architectures = ["convnext_tiny", "vit_base", "swin_tiny", "resnet50"]
    cont_cols = ["psnr", "ssim", "lpips", "fft_energy_low", "fft_energy_high", "jpeg_block_artifact", "delta_edge_energy", "std_local_ssim"]

    results = []

    for arch in architectures:
        df_arch = df_sub[df_sub["model"] == arch].copy().sort_values(["image_id", "level", "order"]).reset_index(drop=True)
        y_true = df_arch["logit_margin"].to_numpy()
        N_eval = len(df_arch)
        K_per_cluster = 18

        pred_conv3 = np.zeros(N_eval)
        pred_conv5 = np.zeros(N_eval)
        pred_red3 = np.zeros(N_eval)
        pred_red5 = np.zeros(N_eval)

        for fold_i in range(5):
            test_c = clusters[fold_assignments == fold_i]
            train_c = clusters[fold_assignments != fold_i]
            test_mask = df_arch["image_id"].isin(test_c).to_numpy()
            train_mask = df_arch["image_id"].isin(train_c).to_numpy()

            tr_df = df_arch[train_mask]
            te_df = df_arch[test_mask]

            scaler_params = {}
            for col in cont_cols:
                vals = tr_df[col].to_numpy()
                scaler_params[col] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) + 1e-8}

            spline_knots = {}
            for col in ["psnr", "ssim", "lpips"]:
                std_v = (tr_df[col].to_numpy() - scaler_params[col]["mean"]) / scaler_params[col]["std"]
                spline_knots[col] = np.percentile(std_v, [5.0, 35.0, 65.0, 95.0])

            y_tr = tr_df["logit_margin"].to_numpy()

            # Conventional
            X_tr_c3 = build_design_matrix(tr_df, "conv", scaler_params, spline_knots)
            X_te_c3 = build_design_matrix(te_df, "conv", scaler_params, spline_knots)
            X_tr_c5 = build_design_matrix(tr_df, "conv_extended", scaler_params, spline_knots)
            X_te_c5 = build_design_matrix(te_df, "conv_extended", scaler_params, spline_knots)

            b_c3 = np.linalg.pinv(X_tr_c3.T @ X_tr_c3) @ (X_tr_c3.T @ y_tr)
            b_c5 = np.linalg.pinv(X_tr_c5.T @ X_tr_c5) @ (X_tr_c5.T @ y_tr)

            pred_conv3[test_mask] = X_te_c3 @ b_c3
            pred_conv5[test_mask] = X_te_c5 @ b_c5

            # Reduced
            X_tr_r3 = build_design_matrix(tr_df, "red", scaler_params, spline_knots)
            X_te_r3 = build_design_matrix(te_df, "red", scaler_params, spline_knots)
            X_tr_r5 = build_design_matrix(tr_df, "red_extended", scaler_params, spline_knots)
            X_te_r5 = build_design_matrix(te_df, "red_extended", scaler_params, spline_knots)

            b_r3 = np.linalg.pinv(X_tr_r3.T @ X_tr_r3) @ (X_tr_r3.T @ y_tr)
            b_r5 = np.linalg.pinv(X_tr_r5.T @ X_tr_r5) @ (X_tr_r5.T @ y_tr)

            pred_red3[test_mask] = X_te_r3 @ b_r3
            pred_red5[test_mask] = X_te_r5 @ b_r5

        ss_tot = np.sum((y_true - y_true.mean())**2)
        r2_c3 = float(1.0 - np.sum((y_true - pred_conv3)**2) / ss_tot)
        r2_c5 = float(1.0 - np.sum((y_true - pred_conv5)**2) / ss_tot)
        mae_c3 = float(np.mean(np.abs(y_true - pred_conv3)))
        mae_c5 = float(np.mean(np.abs(y_true - pred_conv5)))
        d_r2_c = float(r2_c5 - r2_c3)
        d_mae_c = float(mae_c3 - mae_c5)

        r2_r3 = float(1.0 - np.sum((y_true - pred_red3)**2) / ss_tot)
        r2_r5 = float(1.0 - np.sum((y_true - pred_red5)**2) / ss_tot)
        mae_r3 = float(np.mean(np.abs(y_true - pred_red3)))
        mae_r5 = float(np.mean(np.abs(y_true - pred_red5)))
        d_r2_r = float(r2_r5 - r2_r3)
        d_mae_r = float(mae_r3 - mae_r5)

        # Bootstrap for Conventional
        Y_c = y_true.reshape(n_clusters, K_per_cluster)
        P_c3 = pred_conv3.reshape(n_clusters, K_per_cluster)
        P_c5 = pred_conv5.reshape(n_clusters, K_per_cluster)

        c_s1 = Y_c.sum(axis=1)
        c_s2 = (Y_c**2).sum(axis=1)
        b_sum_y = W @ c_s1
        b_sum_y2 = W @ c_s2
        b_sst = b_sum_y2 - (b_sum_y**2) / float(N_eval)

        c_sse_c3 = ((Y_c - P_c3)**2).sum(axis=1)
        c_sse_c5 = ((Y_c - P_c5)**2).sum(axis=1)
        b_sse_c3 = W @ c_sse_c3
        b_sse_c5 = W @ c_sse_c5

        b_r2_c3 = 1.0 - b_sse_c3 / b_sst
        b_r2_c5 = 1.0 - b_sse_c5 / b_sst
        b_dr2_c = b_r2_c5 - b_r2_c3

        c_sae_c3 = (np.abs(Y_c - P_c3)).sum(axis=1)
        c_sae_c5 = (np.abs(Y_c - P_c5)).sum(axis=1)
        b_sae_c3 = W @ c_sae_c3
        b_sae_c5 = W @ c_sae_c5
        b_mae_c3 = b_sae_c3 / float(N_eval)
        b_mae_c5 = b_sae_c5 / float(N_eval)
        b_dmae_c = b_mae_c3 - b_mae_c5

        ci95_r2 = [float(np.percentile(b_dr2_c, 2.5)), float(np.percentile(b_dr2_c, 97.5))]
        sim_1s_r2 = float(np.percentile(b_dr2_c, 1.25))
        ci95_mae = [float(np.percentile(b_dmae_c, 2.5)), float(np.percentile(b_dmae_c, 97.5))]
        p_cal_r2 = float((1.0 + np.sum((b_dr2_c - d_r2_c) >= d_r2_c)) / (n_bootstrap + 1.0))

        results.append({
            "dataset_variant": dataset_label,
            "architecture": arch,
            "n_images": n_clusters,
            "total_evaluations": N_eval,
            "conventional_r2_m3": round(r2_c3, 4),
            "conventional_r2_m5": round(r2_c5, 4),
            "conventional_delta_r2": round(d_r2_c, 4),
            "ci95_delta_r2": [round(ci95_r2[0], 4), round(ci95_r2[1], 4)],
            "simultaneous_1s_lower_r2": round(sim_1s_r2, 4),
            "p_cal_delta_r2": round(p_cal_r2, 5),
            "conventional_mae_m3": round(mae_c3, 4),
            "conventional_mae_m5": round(mae_c5, 4),
            "conventional_delta_mae": round(d_mae_c, 4),
            "ci95_delta_mae": [round(ci95_mae[0], 4), round(ci95_mae[1], 4)],
            "reduced_delta_r2": round(d_r2_r, 4),
            "reduced_delta_mae": round(d_mae_r, 4),
        })
        print(f"  {arch:15s} | M3 R2: {r2_c3:.4f}, M5 R2: {r2_c5:.4f}, Delta R2: {d_r2_c:+.4f} (95% CI: [{ci95_r2[0]:+.4f}, {ci95_r2[1]:+.4f}]) | MAE Delta: {d_mae_c:+.4f} (p={p_cal_r2:.4f})")

    return results


def main():
    print("=" * 80)
    print("PHASE 2: IMAGENET-V2 INDEPENDENT REPLICATION EVALUATION")
    print("=" * 80)

    # 1. Load Calibration Parameters
    with open(CALIB_JSON, "r") as f:
        calib_data = json.load(f)

    # 2. Setup Models and Transforms
    print("\nLoading Pretrained Vision Models & LPIPS...")
    models = {
        "resnet50": timm.create_model("resnet50.a1_in1k", pretrained=True).eval().to(DEVICE),
        "convnext_tiny": timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=True).eval().to(DEVICE),
        "vit_base": timm.create_model("vit_base_patch16_224.augreg_in21k_ft_in1k", pretrained=True).eval().to(DEVICE),
        "swin_tiny": timm.create_model("swin_tiny_patch4_window7_224.ms_in1k", pretrained=True).eval().to(DEVICE),
    }

    model_norms = {
        "resnet50": T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        "convnext_tiny": T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        "vit_base": T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        "swin_tiny": T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    }

    spatial_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    ssim_kernel = _create_ssim_kernel(device=DEVICE)
    lpips_fn = lpips.LPIPS(net="alex").eval().to(DEVICE)

    # 3. Load Images across the 3 Variants
    with open(MANIFEST_PATH, "r") as f:
        manifest_data = json.load(f)

    records = []
    global_image_id = 0

    for variant_name in ["MatchedFrequency", "Threshold0.7", "TopImages"]:
        var_images = manifest_data[variant_name]
        print(f"\nEvaluating ImageNet-V2 Variant: {variant_name} ({len(var_images)} images)...")

        for img_idx, item in enumerate(var_images):
            img_path = DATA_DIR / item["relative_path"]
            true_lbl = item["class_idx"]
            wnid = item["wordnet_id"]
            cname = item["class_name"]

            with Image.open(img_path) as pil_im:
                clean_img = spatial_transform(pil_im.convert("RGB"))

            # Clean forward passes
            clean_margins = {}
            with torch.no_grad():
                for m_name, m_net in models.items():
                    clean_norm = model_norms[m_name](clean_img).unsqueeze(0).to(DEVICE)
                    logits = m_net(clean_norm).squeeze(0)
                    z_t = logits[true_lbl].item()
                    other_mask = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
                    other_mask[true_lbl] = False
                    z_o = logits[other_mask].max().item()
                    clean_margins[m_name] = z_t - z_o

            # Corrupted forward passes across 18 conditions
            for lvl_idx, lvl in enumerate(LEVELS):
                for seq_idx, seq in enumerate(SEQUENCES):
                    cfg = calib_data["levels"][lvl]["sequences"][seq]
                    seed = 42 + global_image_id * 100 + lvl_idx * 10 + seq_idx
                    corr_img = apply_compound_sequence(
                        clean_img,
                        order=seq,
                        noise_std=cfg["noise_std"],
                        blur_sigma=cfg["blur_sigma"],
                        jpeg_quality=cfg["jpeg_quality"],
                        noise_seed=seed,
                    )

                    # Metrics
                    p_val = compute_psnr(clean_img, corr_img)
                    s_val, _ = compute_ssim_and_local_map(clean_img, corr_img, ssim_kernel)
                    with torch.no_grad():
                        lp_clean = (clean_img.unsqueeze(0) * 2.0 - 1.0).to(DEVICE)
                        lp_corr = (corr_img.unsqueeze(0) * 2.0 - 1.0).to(DEVICE)
                        lpips_val = float(lpips_fn(lp_clean, lp_corr).item())

                    fingerprints = compute_distortion_fingerprints(clean_img, corr_img, ssim_kernel)

                    # Model passes
                    with torch.no_grad():
                        for m_name, m_net in models.items():
                            corr_norm = model_norms[m_name](corr_img).unsqueeze(0).to(DEVICE)
                            logits = m_net(corr_norm).squeeze(0)
                            probs = torch.softmax(logits, dim=0)

                            z_t = logits[true_lbl].item()
                            other_mask = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
                            other_mask[true_lbl] = False
                            z_o = logits[other_mask].max().item()
                            margin = z_t - z_o
                            delta_margin = margin - clean_margins[m_name]

                            pred_cls = int(torch.argmax(logits).item())
                            correct = bool(pred_cls == true_lbl)
                            conf = float(probs[pred_cls].item())

                            records.append({
                                "variant": variant_name,
                                "variant_image_id": img_idx,
                                "image_id": global_image_id,
                                "image_filename": item["filename"],
                                "true_label": true_lbl,
                                "wordnet_id": wnid,
                                "class_name": cname,
                                "level": lvl,
                                "order": seq,
                                "final_op": FINAL_OPS[seq],
                                "model": m_name,
                                "psnr": round(p_val, 3),
                                "ssim": round(s_val, 4),
                                "lpips": round(lpips_val, 4),
                                **fingerprints,
                                "logit_margin": round(margin, 4),
                                "delta_logit_margin": round(delta_margin, 4),
                                "clean_logit_margin": round(clean_margins[m_name], 4),
                                "top1_correct": correct,
                                "confidence": round(conf, 4),
                            })

            global_image_id += 1
            if (img_idx + 1) % 25 == 0:
                print(f"  Processed {img_idx + 1}/100 images for {variant_name}...")

    # Save evaluations CSV
    df_all = pd.DataFrame(records)
    csv_out = OUTPUT_DIR / "imagenet_v2_all_evaluations.csv.gz"
    df_all.to_csv(csv_out, index=False, compression="gzip")
    print(f"\nSaved all {len(df_all):,} evaluations to {csv_out}")

    # 4. Statistical Analysis across Variants + Pooled Analysis
    all_variant_stats = []
    for var in ["MatchedFrequency", "Threshold0.7", "TopImages"]:
        df_v = df_all[df_all["variant"] == var]
        v_stats = evaluate_dataset_subset(df_v, var, n_bootstrap=5000)
        all_variant_stats.extend(v_stats)

    # Pooled Analysis
    pooled_stats = evaluate_dataset_subset(df_all, "Pooled_External_All_Variants", n_bootstrap=5000)
    all_variant_stats.extend(pooled_stats)

    df_stats = pd.DataFrame(all_variant_stats)
    stats_csv = OUTPUT_DIR / "imagenet_v2_statistical_comparison.csv"
    df_stats.to_csv(stats_csv, index=False)
    print(f"\nSaved Statistical Comparison Table to {stats_csv}")

    audit_json = OUTPUT_DIR / "imagenet_v2_replication_audit.json"
    with open(audit_json, "w") as fp:
        json.dump(all_variant_stats, fp, indent=2)
    print(f"Saved Replication Audit JSON to {audit_json}")

    print("\n" + "=" * 130)
    print("IMAGENET-V2 REPLICATION SUMMARY TABLE:")
    print("=" * 130)
    cols = ["dataset_variant", "architecture", "conventional_r2_m3", "conventional_r2_m5", "conventional_delta_r2", "ci95_delta_r2", "conventional_delta_mae", "p_cal_delta_r2"]
    print(df_stats[cols].to_string(index=False))
    return df_all, df_stats


if __name__ == "__main__":
    main()
