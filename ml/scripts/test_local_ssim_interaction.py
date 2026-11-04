"""Formal Paired Image-Cluster Bootstrap Test of Architecture x Local-SSIM Differences.

Tests whether the standardized regression coefficient for std_local_ssim
in Model 5 for ViT-Base is statistically different from ResNet-50, ConvNeXt-Tiny, and Swin-Tiny.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONF_DIR = BASE_DIR / "results" / "matched_fidelity_study" / "confirmatory_run"
CSV_GZ_PATH = CONF_DIR / "full_distortion_fingerprints_and_margins.csv.gz"


def conventional_rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    K = len(knots)
    k_K = knots[-1]
    k_Km1 = knots[-2]
    basis = [x]
    denom_last = max(1e-8, abs(k_K - k_Km1))
    d_last = (np.maximum(0.0, x - k_Km1) ** 3 - np.maximum(0.0, x - k_K) ** 3) / denom_last
    for j in range(K - 2):
        k_j = knots[j]
        denom_j = max(1e-8, abs(k_K - k_j))
        d_j = (np.maximum(0.0, x - k_j) ** 3 - np.maximum(0.0, x - k_K) ** 3) / denom_j
        basis.append(d_j - d_last)
    return np.column_stack(basis)


def main():
    print("=" * 80, flush=True)
    print("FORMAL PAIRED BOOTSTRAP TEST: LOCAL-SSIM COEFFICIENT ACROSS ARCHITECTURES", flush=True)
    print("=" * 80, flush=True)

    df = pd.read_csv(CSV_GZ_PATH)
    df = df[df["level"] != "Clean"].copy()
    clusters = np.sort(df["image_id"].unique())
    n_clusters = len(clusters)

    cont_cols = ["psnr", "ssim", "lpips", "fft_energy_low", "fft_energy_high", "jpeg_block_artifact", "delta_edge_energy", "std_local_ssim"]
    scaler_params = {c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=1))} for c in cont_cols}
    spline_knots = {c: np.percentile((df[c] - scaler_params[c]["mean"]) / scaler_params[c]["std"], [5.0, 35.0, 65.0, 95.0]) for c in ["psnr", "ssim", "lpips"]}

    orders = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]
    levels = ["Level_1_Mild", "Level_2_Moderate", "Level_3_SevereCommon"]

    def get_design_matrix(df_arch):
        N = len(df_arch)
        intercept = np.ones((N, 1))
        order_mat = np.column_stack([(df_arch["order"].to_numpy() == o).astype(float) for o in orders[1:]])
        lvl_mat = np.column_stack([(df_arch["level"].to_numpy() == l).astype(float) for l in levels[1:]])
        oxl = np.zeros((N, (len(orders) - 1) * (len(levels) - 1)))
        c_idx = 0
        for oi in range(order_mat.shape[1]):
            for li in range(lvl_mat.shape[1]):
                oxl[:, c_idx] = order_mat[:, oi] * lvl_mat[:, li]
                c_idx += 1
        p_sp = conventional_rcs_basis((df_arch["psnr"].to_numpy() - scaler_params["psnr"]["mean"]) / scaler_params["psnr"]["std"], spline_knots["psnr"])
        s_sp = conventional_rcs_basis((df_arch["ssim"].to_numpy() - scaler_params["ssim"]["mean"]) / scaler_params["ssim"]["std"], spline_knots["ssim"])
        l_sp = conventional_rcs_basis((df_arch["lpips"].to_numpy() - scaler_params["lpips"]["mean"]) / scaler_params["lpips"]["std"], spline_knots["lpips"])

        fps = []
        for c in ["fft_energy_low", "fft_energy_high", "jpeg_block_artifact", "delta_edge_energy", "std_local_ssim"]:
            fps.append(((df_arch[c].to_numpy() - scaler_params[c]["mean"]) / scaler_params[c]["std"])[:, None])
        return np.hstack([intercept, order_mat, lvl_mat, oxl, p_sp, s_sp, l_sp] + fps)

    archs = ["vit_base", "resnet50", "convnext_tiny", "swin_tiny"]
    X_dict = {}
    y_dict = {}
    for a in archs:
        df_a = df[df["model"] == a].sort_values(["image_id", "level", "order"]).reset_index(drop=True)
        X_dict[a] = get_design_matrix(df_a)
        y_dict[a] = df_a["logit_margin"].to_numpy()

    point_betas = {}
    for a in archs:
        b = np.linalg.pinv(X_dict[a].T @ X_dict[a]) @ (X_dict[a].T @ y_dict[a])
        point_betas[a] = float(b[-1])
        print(f"  {a:15s}: beta(std_local_ssim) = {b[-1]:+.4f}", flush=True)

    B = 10000
    rng = np.random.RandomState(42)
    b_choices = rng.randint(0, n_clusters, size=(B, n_clusters))
    W = np.zeros((B, n_clusters), dtype=float)
    for b in range(B):
        W[b, :] = np.bincount(b_choices[b], minlength=n_clusters)

    K_c = 18
    p = X_dict["vit_base"].shape[1]

    boot_betas = {}
    for a in archs:
        print(f"Computing batched linear systems for {a}...", flush=True)
        X_c = X_dict[a].reshape(n_clusters, K_c, p)
        y_c = y_dict[a].reshape(n_clusters, K_c)

        # Cluster XtX: (n_clusters, p*p)
        flat_XtX = np.einsum("cki,ckj->cij", X_c, X_c).reshape(n_clusters, p * p)
        flat_Xty = np.einsum("cki,ck->ci", X_c, y_c)  # (n_clusters, p)

        # Batch over B resamples via BLAS matrix product
        XtX_all = (W @ flat_XtX).reshape(B, p, p)
        Xty_all = (W @ flat_Xty).reshape(B, p, 1)

        # Batched solve
        betas_all = np.linalg.solve(XtX_all, Xty_all).squeeze(-1)  # (B, p)
        boot_betas[a] = betas_all[:, -1]
        print(f"  Done {a}: Mean boot beta = {np.mean(boot_betas[a]):+.4f}, SE = {np.std(boot_betas[a], ddof=1):.4f}", flush=True)

    print("\nPairwise Differences vs ViT-Base (beta_vit - beta_other):", flush=True)
    results = []
    for a in ["resnet50", "convnext_tiny", "swin_tiny"]:
        diff_pt = point_betas["vit_base"] - point_betas[a]
        b_diff = boot_betas["vit_base"] - boot_betas[a]
        ci_95 = [float(np.percentile(b_diff, 2.5)), float(np.percentile(b_diff, 97.5))]
        # Two-sided null calibrated p-value:
        p_val = float((1.0 + np.sum(np.abs(b_diff - diff_pt) >= np.abs(diff_pt))) / (B + 1.0))
        results.append({
            "comparison": f"vit_base_minus_{a}",
            "arch_a": "vit_base",
            "arch_b": a,
            "beta_vit": point_betas["vit_base"],
            "beta_other": point_betas[a],
            "diff_point": round(diff_pt, 4),
            "ci95_diff": [round(ci_95[0], 4), round(ci_95[1], 4)],
            "p_value": round(p_val, 5),
            "se_diff": round(float(np.std(b_diff, ddof=1)), 4),
        })
        print(f"  ViT-Base vs {a:15s}: Diff = {diff_pt:+.4f} (SE: {np.std(b_diff, ddof=1):.4f}), 95% CI: [{ci_95[0]:+.4f}, {ci_95[1]:+.4f}], p = {p_val:.5f}", flush=True)

    out_json = CONF_DIR / "confirmatory_local_ssim_interaction_test.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved formal interaction test to: {out_json}", flush=True)


if __name__ == "__main__":
    main()
