"""Phase 3: Grouped Cross-Validated Feature Ablation & Contribution Analysis.

Evaluates on the completed confirmatory dataset (N=3,825 images, 275,400 evaluations):
1. Global fidelity baseline only (Model 3: Conventional 4-knot RCS, 27 cols).
2. Global fidelity + Fourier features (E_low, E_high) (29 cols).
3. Global fidelity + JPEG boundary feature (B_jpeg) (28 cols).
4. Global fidelity + Edge-energy feature (delta_edge) (28 cols).
5. Global fidelity + Local SSIM variability (std_local_ssim) (28 cols).
6. Global fidelity + All structural fingerprints (Model 5: 32 cols).

Also Drop-One Ablations from Model 5:
- Model 5 minus Fourier (30 cols)
- Model 5 minus JPEG (31 cols)
- Model 5 minus Edge (31 cols)
- Model 5 minus Local SSIM (31 cols)

All evaluated via 5-Fold Grouped Cross-Validation (identical physical image clusters)
and B=10,000 paired cluster bootstrap resamples.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "results" / "matched_fidelity_study"
CONF_DIR = OUTPUT_DIR / "confirmatory_run"
CSV_GZ_PATH = CONF_DIR / "full_distortion_fingerprints_and_margins.csv.gz"

BOOTSTRAP_REPLICATES = 10000
CV_SEED = 42
BOOTSTRAP_SEED = 12345


def conventional_rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Conventional 4-knot Restricted Cubic Spline (3 df: 1 linear + 2 nonlinear)."""
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


def build_ablation_design_matrix(df_part, feature_set, scaler_params, spline_knots):
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

    p_sp = conventional_rcs_basis(p_std, spline_knots["psnr"])
    s_sp = conventional_rcs_basis(s_std, spline_knots["ssim"])
    l_sp = conventional_rcs_basis(l_std, spline_knots["lpips"])

    extra_cols = []
    if "fourier" in feature_set:
        f_low = (df_part["fft_energy_low"].to_numpy() - scaler_params["fft_energy_low"]["mean"]) / scaler_params["fft_energy_low"]["std"]
        f_high = (df_part["fft_energy_high"].to_numpy() - scaler_params["fft_energy_high"]["mean"]) / scaler_params["fft_energy_high"]["std"]
        extra_cols.extend([f_low, f_high])
    if "jpeg" in feature_set:
        jpeg_b = (df_part["jpeg_block_artifact"].to_numpy() - scaler_params["jpeg_block_artifact"]["mean"]) / scaler_params["jpeg_block_artifact"]["std"]
        extra_cols.append(jpeg_b)
    if "edge" in feature_set:
        d_edge = (df_part["delta_edge_energy"].to_numpy() - scaler_params["delta_edge_energy"]["mean"]) / scaler_params["delta_edge_energy"]["std"]
        extra_cols.append(d_edge)
    if "local_ssim" in feature_set:
        s_std_loc = (df_part["std_local_ssim"].to_numpy() - scaler_params["std_local_ssim"]["mean"]) / scaler_params["std_local_ssim"]["std"]
        extra_cols.append(s_std_loc)

    if extra_cols:
        extra_mat = np.column_stack(extra_cols)
        return np.hstack(base_parts + [p_sp, s_sp, l_sp, extra_mat])
    else:
        return np.hstack(base_parts + [p_sp, s_sp, l_sp])


def run_feature_ablations():
    print("=" * 80)
    print("PHASE 3: GROUPED CV FEATURE ABLATIONS & BOOTSTRAP UNCERTAINTY")
    print("=" * 80)

    df = pd.read_csv(CSV_GZ_PATH)
    clusters = np.sort(df["image_id"].unique())
    n_clusters = len(clusters)
    cluster_classes = df.drop_duplicates("image_id").sort_values("image_id")["true_label"].to_numpy()

    # Identical 5-fold partition
    rng_cv = np.random.RandomState(CV_SEED)
    fold_assignments = np.zeros(n_clusters, dtype=int)
    for c in range(10):
        c_indices = np.where(cluster_classes == c)[0]
        perm = rng_cv.permutation(c_indices)
        c_splits = np.array_split(perm, 5)
        for fold_i, split_idx in enumerate(c_splits):
            fold_assignments[split_idx] = fold_i

    # Pre-generate cluster bootstrap weights
    rng_b = np.random.RandomState(BOOTSTRAP_SEED)
    bootstrap_cluster_choices = rng_b.randint(0, n_clusters, size=(BOOTSTRAP_REPLICATES, n_clusters))
    W = np.zeros((BOOTSTRAP_REPLICATES, n_clusters), dtype=float)
    for b in range(BOOTSTRAP_REPLICATES):
        W[b, :] = np.bincount(bootstrap_cluster_choices[b], minlength=n_clusters)

    architectures = ["convnext_tiny", "vit_base", "swin_tiny", "resnet50"]
    cont_cols = ["psnr", "ssim", "lpips", "fft_energy_low", "fft_energy_high", "jpeg_block_artifact", "delta_edge_energy", "std_local_ssim"]

    # Prespecified ablation conditions
    ablation_configs = {
        "1_Baseline_GlobalFidelity": [],
        "2_Plus_Fourier": ["fourier"],
        "3_Plus_JPEG_Boundary": ["jpeg"],
        "4_Plus_Edge_Energy": ["edge"],
        "5_Plus_Local_SSIM_Var": ["local_ssim"],
        "6_Combined_All_Fingerprints": ["fourier", "jpeg", "edge", "local_ssim"],
        # Drop-one ablations
        "Drop_Fourier": ["jpeg", "edge", "local_ssim"],
        "Drop_JPEG": ["fourier", "edge", "local_ssim"],
        "Drop_Edge": ["fourier", "jpeg", "local_ssim"],
        "Drop_Local_SSIM": ["fourier", "jpeg", "edge"],
    }

    ablation_results = []

    for arch in architectures:
        print(f"\nEvaluating Ablations for {arch}...")
        df_arch = df[df["model"] == arch].copy().sort_values(["image_id", "level", "order"]).reset_index(drop=True)
        y_true = df_arch["logit_margin"].to_numpy()
        N_eval = len(df_arch)
        K_per_cluster = 18

        # Precompute cluster reductions for y
        Y_c = y_true.reshape(n_clusters, K_per_cluster)
        c_s1 = Y_c.sum(axis=1)
        c_s2 = (Y_c**2).sum(axis=1)
        b_sum_y = W @ c_s1
        b_sum_y2 = W @ c_s2
        b_sst = b_sum_y2 - (b_sum_y**2) / float(N_eval)
        ss_tot = np.sum((y_true - y_true.mean())**2)

        # First compute baseline Model 3 predictions
        pred_dict = {}
        for cond_name, feat_set in ablation_configs.items():
            pred_arr = np.zeros(N_eval)
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
                X_tr = build_ablation_design_matrix(tr_df, feat_set, scaler_params, spline_knots)
                X_te = build_ablation_design_matrix(te_df, feat_set, scaler_params, spline_knots)

                b_coef = np.linalg.pinv(X_tr.T @ X_tr) @ (X_tr.T @ y_tr)
                pred_arr[test_mask] = X_te @ b_coef

            pred_dict[cond_name] = pred_arr

        # Now compute paired metrics against Baseline (1_Baseline_GlobalFidelity)
        p_base = pred_dict["1_Baseline_GlobalFidelity"]
        P_base_c = p_base.reshape(n_clusters, K_per_cluster)
        c_sse_base = ((Y_c - P_base_c)**2).sum(axis=1)
        c_sae_base = (np.abs(Y_c - P_base_c)).sum(axis=1)
        b_sse_base = W @ c_sse_base
        b_sae_base = W @ c_sae_base
        b_r2_base = 1.0 - b_sse_base / b_sst
        b_mae_base = b_sae_base / float(N_eval)

        r2_base = float(1.0 - np.sum((y_true - p_base)**2) / ss_tot)
        mae_base = float(np.mean(np.abs(y_true - p_base)))

        for cond_name, p_m in pred_dict.items():
            P_m_c = p_m.reshape(n_clusters, K_per_cluster)
            c_sse_m = ((Y_c - P_m_c)**2).sum(axis=1)
            c_sae_m = (np.abs(Y_c - P_m_c)).sum(axis=1)

            b_sse_m = W @ c_sse_m
            b_sae_m = W @ c_sae_m
            b_r2_m = 1.0 - b_sse_m / b_sst
            b_mae_m = b_sae_m / float(N_eval)

            # Paired differences vs Baseline
            b_dr2 = b_r2_m - b_r2_base
            b_dmae = b_mae_base - b_mae_m  # positive means error reduction

            r2_m = float(1.0 - np.sum((y_true - p_m)**2) / ss_tot)
            mae_m = float(np.mean(np.abs(y_true - p_m)))
            d_r2 = float(r2_m - r2_base)
            d_mae = float(mae_base - mae_m)

            # 95% Bootstrap CI
            ci_dr2 = [float(np.percentile(b_dr2, 2.5)), float(np.percentile(b_dr2, 97.5))]
            ci_dmae = [float(np.percentile(b_dmae, 2.5)), float(np.percentile(b_dmae, 97.5))]
            p_val_r2 = float((1.0 + np.sum((b_dr2 - d_r2) >= d_r2)) / (BOOTSTRAP_REPLICATES + 1.0)) if d_r2 > 0 else 1.0

            res_entry = {
                "architecture": arch,
                "ablation_condition": cond_name,
                "num_columns": 27 + (len(ablation_configs[cond_name]) if "fourier" not in ablation_configs[cond_name] else len(ablation_configs[cond_name]) + 1),
                "r2_oos": round(r2_m, 4),
                "delta_r2_vs_baseline": round(d_r2, 4),
                "ci95_delta_r2": [round(ci_dr2[0], 4), round(ci_dr2[1], 4)],
                "mae_oos": round(mae_m, 4),
                "delta_mae_vs_baseline": round(d_mae, 4),
                "ci95_delta_mae": [round(ci_dmae[0], 4), round(ci_dmae[1], 4)],
                "p_val_delta_r2": round(p_val_r2, 5),
            }
            ablation_results.append(res_entry)
            print(f"  {cond_name:30s} | R2: {r2_m:.4f} (Delta: {d_r2:+.4f}, 95% CI: [{ci_dr2[0]:+.4f}, {ci_dr2[1]:+.4f}]) | MAE: {mae_m:.4f} (Delta: {d_mae:+.4f})")

    df_abl = pd.DataFrame(ablation_results)
    out_csv = CONF_DIR / "confirmatory_feature_ablations.csv"
    df_abl.to_csv(out_csv, index=False)
    out_csv_main = OUTPUT_DIR / "confirmatory_feature_ablations.csv"
    df_abl.to_csv(out_csv_main, index=False)

    out_json = CONF_DIR / "confirmatory_feature_ablations.json"
    with open(out_json, "w") as fp:
        json.dump(ablation_results, fp, indent=2)

    print(f"\nSaved feature ablation results to {out_csv}")
    return df_abl


if __name__ == "__main__":
    run_feature_ablations()
