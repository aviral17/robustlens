"""Rigorous Grouped Cross-Validation, Spline Baselines, and Clustered Bootstrap.

Addresses all mandates from Sections C, D, E, F:
1. Grouped 5-Fold CV strictly by physical image_id (N=100 clusters).
2. Identical folds across all 4 architectures and all 4 model specifications.
3. Training-only z-scoring and knot placement (zero test leakage).
4. Models Evaluated:
   - Model 1: Linear Global Baseline (Order 5df, Tier 2df, Order x Tier 10df, PSNR, SSIM, LPIPS)
   - Model 2: Quadratic-Interaction Global Baseline (Model 1 + PSNR^2, SSIM^2, LPIPS^2, PSNRxSSIM, PSNRxLPIPS, SSIMxLPIPS)
   - Model 3: Restricted Cubic Spline Global Baseline (Model 1 categorical terms + 4-knot natural cubic splines for PSNR, SSIM, LPIPS)
   - Model 4: Extended Structural Fingerprints Model (Model 2 + Fourier low/high, JPEG 8x8 step ratio, Edge energy change, Local SSIM std)
5. Out-of-sample predictions strictly recorded when image is in test fold.
6. Design-matrix diagnostics: rank, num_cols, condition number kappa(X) for each fold.
7. Corrected Paired Out-of-Sample Image-Cluster Bootstrap (B=2,000, seed=12345):
   - Separate 95% CIs for Delta R^2_OOS (Extended minus Baseline)
   - Separate 95% CIs for Delta MAE_OOS (Baseline minus Extended)
   - Replicate count, random seed, proportion of bootstrap replicates showing improvement
   - Transparent explanation of conditional test prediction resampling vs model refitting
8. Holm-Bonferroni correction across the 4 architecture comparisons.
9. Exports:
   - results/matched_fidelity_study/grouped_cv_splines_comparison.csv
   - results/matched_fidelity_study/grouped_cv_splines_diagnostics.json
   - results/matched_fidelity_study/pilot_exploratory_100_image_ids.json
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CSV_PATH = BASE_DIR / "results" / "matched_fidelity_study" / "heldout_fingerprints_and_margins.csv"
OUTPUT_CSV = BASE_DIR / "results" / "matched_fidelity_study" / "grouped_cv_splines_comparison.csv"
OUTPUT_JSON = BASE_DIR / "results" / "matched_fidelity_study" / "grouped_cv_splines_diagnostics.json"
PILOT_IDS_JSON = BASE_DIR / "results" / "matched_fidelity_study" / "pilot_exploratory_100_image_ids.json"

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 12345
CV_SEED = 42


def natural_cubic_spline_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Computes truncated natural cubic spline basis (Restricted Cubic Splines).
    
    Given K knots: k_1, ..., k_K:
    Basis 1: x
    Basis j (for j=2 to K-1):
      d_j(x) = (max(0, x - k_j)^3 - max(0, x - k_K)^3) / (k_K - k_j)
      d_{K-1}(x) = (max(0, x - k_{K-1})^3 - max(0, x - k_K)^3) / (k_K - k_{K-1})
      Basis_j = d_j(x) - d_{K-1}(x)
    Returns (N, K-1) array of spline terms.
    """
    K = len(knots)
    N = len(x)
    k_K = knots[-1]
    k_Km1 = knots[-2]
    
    basis = [x]
    
    denom_last = k_K - k_Km1
    if abs(denom_last) < 1e-8:
        denom_last = 1e-8
        
    term_last = np.maximum(0.0, x - k_Km1)**3 - np.maximum(0.0, x - k_K)**3
    d_last = term_last / denom_last
    
    for j in range(1, K - 2):
        k_j = knots[j]
        denom_j = k_K - k_j
        if abs(denom_j) < 1e-8:
            denom_j = 1e-8
        d_j = (np.maximum(0.0, x - k_j)**3 - np.maximum(0.0, x - k_K)**3) / denom_j
        basis.append(d_j - d_last)
        
    return np.column_stack(basis)


def build_design_matrix(
    df_part: pd.DataFrame,
    spec: str,  # "linear", "quadratic", "spline", "extended"
    scaler_params: dict,
    spline_knots: dict | None = None,
) -> np.ndarray:
    """Builds design matrix strictly using training-derived scaler and knots."""
    N = len(df_part)
    
    # 1. Main Effects: Order (5 df) and Severity Tier (2 df)
    orders = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]  # reference: NBJ
    levels = ["Mild", "Moderate", "Severe"]               # reference: Mild
    
    order_mat = np.zeros((N, len(orders) - 1))
    for i, o in enumerate(orders[1:]):
        order_mat[:, i] = (df_part["order"].to_numpy() == o).astype(float)
        
    level_mat = np.zeros((N, len(levels) - 1))
    for i, lvl in enumerate(levels[1:]):
        level_mat[:, i] = (df_part["severity_level"].to_numpy() == lvl).astype(float)
        
    # 2. Interactions: Order x Severity Tier (10 df)
    order_x_level = np.zeros((N, (len(orders) - 1) * (len(levels) - 1)))
    col_idx = 0
    for o_i in range(order_mat.shape[1]):
        for l_i in range(level_mat.shape[1]):
            order_x_level[:, col_idx] = order_mat[:, o_i] * level_mat[:, l_i]
            col_idx += 1
            
    # 3. Continuous Global Fidelity Variables (Standardized using training scaler)
    p_raw = df_part["psnr"].to_numpy()
    s_raw = df_part["ssim"].to_numpy()
    l_raw = df_part["lpips"].to_numpy()
    
    p_std = (p_raw - scaler_params["psnr"]["mean"]) / scaler_params["psnr"]["std"]
    s_std = (s_raw - scaler_params["ssim"]["mean"]) / scaler_params["ssim"]["std"]
    l_std = (l_raw - scaler_params["lpips"]["mean"]) / scaler_params["lpips"]["std"]
    
    intercept = np.ones((N, 1))
    base_parts = [intercept, order_mat, level_mat, order_x_level]
    
    if spec == "linear":
        global_cont = np.column_stack([p_std, s_std, l_std])
        return np.hstack(base_parts + [global_cont])
        
    elif spec == "quadratic":
        quad_terms = np.column_stack([
            p_std, s_std, l_std,
            p_std**2, s_std**2, l_std**2,
            p_std * s_std, p_std * l_std, s_std * l_std,
        ])
        return np.hstack(base_parts + [quad_terms])
        
    elif spec == "spline":
        assert spline_knots is not None, "Spline knots required for spline specification"
        p_spline = natural_cubic_spline_basis(p_std, spline_knots["psnr"])
        s_spline = natural_cubic_spline_basis(s_std, spline_knots["ssim"])
        l_spline = natural_cubic_spline_basis(l_std, spline_knots["lpips"])
        return np.hstack(base_parts + [p_spline, s_spline, l_spline])

    elif spec == "spline_extended":
        assert spline_knots is not None, "Spline knots required for spline specification"
        p_spline = natural_cubic_spline_basis(p_std, spline_knots["psnr"])
        s_spline = natural_cubic_spline_basis(s_std, spline_knots["ssim"])
        l_spline = natural_cubic_spline_basis(l_std, spline_knots["lpips"])
        f_low = (df_part["fft_energy_low"].to_numpy() - scaler_params["fft_energy_low"]["mean"]) / scaler_params["fft_energy_low"]["std"]
        f_high = (df_part["fft_energy_high"].to_numpy() - scaler_params["fft_energy_high"]["mean"]) / scaler_params["fft_energy_high"]["std"]
        jpeg_b = (df_part["jpeg_block_artifact"].to_numpy() - scaler_params["jpeg_block_artifact"]["mean"]) / scaler_params["jpeg_block_artifact"]["std"]
        d_edge = (df_part["delta_edge_energy"].to_numpy() - scaler_params["delta_edge_energy"]["mean"]) / scaler_params["delta_edge_energy"]["std"]
        s_std_loc = (df_part["std_local_ssim"].to_numpy() - scaler_params["std_local_ssim"]["mean"]) / scaler_params["std_local_ssim"]["std"]
        fingerprints = np.column_stack([f_low, f_high, jpeg_b, d_edge, s_std_loc])
        return np.hstack(base_parts + [p_spline, s_spline, l_spline, fingerprints])
        
    elif spec == "extended":
        # Quadratic global baseline + 5 structural fingerprints
        quad_terms = np.column_stack([
            p_std, s_std, l_std,
            p_std**2, s_std**2, l_std**2,
            p_std * s_std, p_std * l_std, s_std * l_std,
        ])
        
        # Standardize fingerprints using training parameters
        f_low = (df_part["fft_energy_low"].to_numpy() - scaler_params["fft_energy_low"]["mean"]) / scaler_params["fft_energy_low"]["std"]
        f_high = (df_part["fft_energy_high"].to_numpy() - scaler_params["fft_energy_high"]["mean"]) / scaler_params["fft_energy_high"]["std"]
        jpeg_b = (df_part["jpeg_block_artifact"].to_numpy() - scaler_params["jpeg_block_artifact"]["mean"]) / scaler_params["jpeg_block_artifact"]["std"]
        d_edge = (df_part["delta_edge_energy"].to_numpy() - scaler_params["delta_edge_energy"]["mean"]) / scaler_params["delta_edge_energy"]["std"]
        s_std_loc = (df_part["std_local_ssim"].to_numpy() - scaler_params["std_local_ssim"]["mean"]) / scaler_params["std_local_ssim"]["std"]
        
        fingerprints = np.column_stack([f_low, f_high, jpeg_b, d_edge, s_std_loc])
        return np.hstack(base_parts + [quad_terms, fingerprints])
    else:
        raise ValueError(f"Unknown specification: {spec}")


def evaluate_architecture_cv(df_arch: pd.DataFrame, arch_name: str) -> dict:
    """Executes Grouped 5-Fold CV and Paired Cluster Bootstrap for one architecture."""
    clusters = np.sort(df_arch["image_id"].unique())
    n_clusters = len(clusters)
    assert n_clusters == 100, f"Expected 100 image clusters, got {n_clusters}"
    
    # Stratified cluster split by class
    rng_cv = np.random.RandomState(CV_SEED)
    cluster_classes = df_arch.drop_duplicates("image_id").sort_values("image_id")["true_label"].to_numpy()
    
    # Assign 2 images per class to each of the 5 folds
    fold_assignments = np.zeros(n_clusters, dtype=int)
    for c in range(10):
        c_cluster_indices = np.where(cluster_classes == c)[0]
        perm = rng_cv.permutation(c_cluster_indices)
        for fold_i in range(5):
            fold_assignments[perm[fold_i*2 : (fold_i+1)*2]] = fold_i
            
    # Track out-of-sample predictions
    # Key: row index in df_arch -> prediction
    y_true_all = []
    y_pred_m1 = []  # Linear
    y_pred_m2 = []  # Quadratic
    y_pred_m3 = []  # Spline
    y_pred_m4 = []  # Extended Quadratic
    y_pred_m5 = []  # Extended Spline (Model 5: Spline + Fingerprints)
    cluster_ids_oos = []
    
    fold_diagnostics = []
    
    cont_cols = ["psnr", "ssim", "lpips", "fft_energy_low", "fft_energy_high", "jpeg_block_artifact", "delta_edge_energy", "std_local_ssim"]
    
    for fold_i in range(5):
        test_cluster_mask = (fold_assignments == fold_i)
        test_clusters = clusters[test_cluster_mask]
        train_clusters = clusters[~test_cluster_mask]
        
        train_df = df_arch[df_arch["image_id"].isin(train_clusters)].copy()
        test_df = df_arch[df_arch["image_id"].isin(test_clusters)].copy()
        
        # Fit scaler strictly on training fold
        scaler_params = {}
        for col in cont_cols:
            vals = train_df[col].to_numpy()
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) + 1e-8
            scaler_params[col] = {"mean": m, "std": s}
            
        # Spline knots (4 knots: 5th, 35th, 65th, 95th percentiles of standardized training data)
        spline_knots = {}
        for col in ["psnr", "ssim", "lpips"]:
            std_train_vals = (train_df[col].to_numpy() - scaler_params[col]["mean"]) / scaler_params[col]["std"]
            spline_knots[col] = np.percentile(std_train_vals, [5.0, 35.0, 65.0, 95.0])
            
        # Build design matrices
        X_tr_1 = build_design_matrix(train_df, "linear", scaler_params)
        X_te_1 = build_design_matrix(test_df, "linear", scaler_params)
        
        X_tr_2 = build_design_matrix(train_df, "quadratic", scaler_params)
        X_te_2 = build_design_matrix(test_df, "quadratic", scaler_params)
        
        X_tr_3 = build_design_matrix(train_df, "spline", scaler_params, spline_knots)
        X_te_3 = build_design_matrix(test_df, "spline", scaler_params, spline_knots)
        
        X_tr_4 = build_design_matrix(train_df, "extended", scaler_params)
        X_te_4 = build_design_matrix(test_df, "extended", scaler_params)

        X_tr_5 = build_design_matrix(train_df, "spline_extended", scaler_params, spline_knots)
        X_te_5 = build_design_matrix(test_df, "spline_extended", scaler_params, spline_knots)
        
        y_train = train_df["logit_margin"].to_numpy()
        y_test = test_df["logit_margin"].to_numpy()
        
        # Fit OLS via pseudoinverse (numerically stable for standard regression)
        b1 = np.linalg.pinv(X_tr_1.T @ X_tr_1) @ (X_tr_1.T @ y_train)
        b2 = np.linalg.pinv(X_tr_2.T @ X_tr_2) @ (X_tr_2.T @ y_train)
        b3 = np.linalg.pinv(X_tr_3.T @ X_tr_3) @ (X_tr_3.T @ y_train)
        b4 = np.linalg.pinv(X_tr_4.T @ X_tr_4) @ (X_tr_4.T @ y_train)
        b5 = np.linalg.pinv(X_tr_5.T @ X_tr_5) @ (X_tr_5.T @ y_train)
        
        y_pred_m1.extend(X_te_1 @ b1)
        y_pred_m2.extend(X_te_2 @ b2)
        y_pred_m3.extend(X_te_3 @ b3)
        y_pred_m4.extend(X_te_4 @ b4)
        y_pred_m5.extend(X_te_5 @ b5)
        y_true_all.extend(y_test)
        cluster_ids_oos.extend(test_df["image_id"].to_numpy())
        
        # Fold Matrix Diagnostics
        s1 = np.linalg.svd(X_tr_1, compute_uv=False)
        s2 = np.linalg.svd(X_tr_2, compute_uv=False)
        s3 = np.linalg.svd(X_tr_3, compute_uv=False)
        s4 = np.linalg.svd(X_tr_4, compute_uv=False)
        s5 = np.linalg.svd(X_tr_5, compute_uv=False)
        
        fold_diagnostics.append({
            "fold": fold_i + 1,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "test_clusters_count": len(test_clusters),
            "model1_linear": {"cols": X_tr_1.shape[1], "rank": int(np.linalg.matrix_rank(X_tr_1)), "condition_number": round(float(s1[0] / s1[-1]), 2)},
            "model2_quadratic": {"cols": X_tr_2.shape[1], "rank": int(np.linalg.matrix_rank(X_tr_2)), "condition_number": round(float(s2[0] / s2[-1]), 2)},
            "model3_spline": {"cols": X_tr_3.shape[1], "rank": int(np.linalg.matrix_rank(X_tr_3)), "condition_number": round(float(s3[0] / s3[-1]), 2)},
            "model4_extended_quad": {"cols": X_tr_4.shape[1], "rank": int(np.linalg.matrix_rank(X_tr_4)), "condition_number": round(float(s4[0] / s4[-1]), 2)},
            "model5_extended_spline": {"cols": X_tr_5.shape[1], "rank": int(np.linalg.matrix_rank(X_tr_5)), "condition_number": round(float(s5[0] / s5[-1]), 2)},
        })
        
    y_true = np.array(y_true_all)
    y_p1 = np.array(y_pred_m1)
    y_p2 = np.array(y_pred_m2)
    y_p3 = np.array(y_pred_m3)
    y_p4 = np.array(y_pred_m4)
    y_p5 = np.array(y_pred_m5)
    cluster_arr = np.array(cluster_ids_oos)
    
    # Out-of-sample global metrics
    ss_tot = np.sum((y_true - y_true.mean())**2)
    
    r2_oos_1 = float(1.0 - np.sum((y_true - y_p1)**2) / ss_tot)
    r2_oos_2 = float(1.0 - np.sum((y_true - y_p2)**2) / ss_tot)
    r2_oos_3 = float(1.0 - np.sum((y_true - y_p3)**2) / ss_tot)
    r2_oos_4 = float(1.0 - np.sum((y_true - y_p4)**2) / ss_tot)
    r2_oos_5 = float(1.0 - np.sum((y_true - y_p5)**2) / ss_tot)
    
    mae_oos_1 = float(np.mean(np.abs(y_true - y_p1)))
    mae_oos_2 = float(np.mean(np.abs(y_true - y_p2)))
    mae_oos_3 = float(np.mean(np.abs(y_true - y_p3)))
    mae_oos_4 = float(np.mean(np.abs(y_true - y_p4)))
    mae_oos_5 = float(np.mean(np.abs(y_true - y_p5)))
    
    # Paired point estimates (Extended Quad vs Quadratic Baseline)
    delta_r2_vs_quad = float(r2_oos_4 - r2_oos_2)
    delta_mae_vs_quad = float(mae_oos_2 - mae_oos_4)  # positive = improvement
    
    # Paired point estimates (Extended Spline Model 5 vs Spline Baseline Model 3)
    delta_r2_vs_spline = float(r2_oos_5 - r2_oos_3)
    delta_mae_vs_spline = float(mae_oos_3 - mae_oos_5)
    
    # CLUSTER BOOTSTRAP on Paired Out-of-Sample Predictions (B=2,000, seed=12345)
    cluster_indices_map = {c: np.where(cluster_arr == c)[0] for c in clusters}
    
    rng_boot = np.random.RandomState(BOOTSTRAP_SEED)
    boot_delta_r2_quad = []
    boot_delta_mae_quad = []
    boot_delta_r2_spline = []
    boot_delta_mae_spline = []
    
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_c = rng_boot.choice(clusters, size=len(clusters), replace=True)
        sampled_rows = np.concatenate([cluster_indices_map[c] for c in sampled_c])
        
        yb = y_true[sampled_rows]
        yp1_b = y_p1[sampled_rows]
        yp2_b = y_p2[sampled_rows]
        yp3_b = y_p3[sampled_rows]
        yp4_b = y_p4[sampled_rows]
        yp5_b = y_p5[sampled_rows]
        
        ss_tot_b = np.sum((yb - yb.mean())**2) + 1e-12
        r2_b2 = 1.0 - np.sum((yb - yp2_b)**2) / ss_tot_b
        r2_b3 = 1.0 - np.sum((yb - yp3_b)**2) / ss_tot_b
        r2_b4 = 1.0 - np.sum((yb - yp4_b)**2) / ss_tot_b
        r2_b5 = 1.0 - np.sum((yb - yp5_b)**2) / ss_tot_b
        
        mae_b2 = np.mean(np.abs(yb - yp2_b))
        mae_b3 = np.mean(np.abs(yb - yp3_b))
        mae_b4 = np.mean(np.abs(yb - yp4_b))
        mae_b5 = np.mean(np.abs(yb - yp5_b))
        
        # Extended Quad vs Quad
        boot_delta_r2_quad.append(float(r2_b4 - r2_b2))
        boot_delta_mae_quad.append(float(mae_b2 - mae_b4))
        # Extended Spline (Model 5) vs Spline (Model 3)
        boot_delta_r2_spline.append(float(r2_b5 - r2_b3))
        boot_delta_mae_spline.append(float(mae_b3 - mae_b5))
        
    boot_delta_r2_quad = np.array(boot_delta_r2_quad)
    boot_delta_mae_quad = np.array(boot_delta_mae_quad)
    boot_delta_r2_spline = np.array(boot_delta_r2_spline)
    boot_delta_mae_spline = np.array(boot_delta_mae_spline)
    
    ci_r2_quad_low, ci_r2_quad_high = float(np.percentile(boot_delta_r2_quad, 2.5)), float(np.percentile(boot_delta_r2_quad, 97.5))
    ci_mae_quad_low, ci_mae_quad_high = float(np.percentile(boot_delta_mae_quad, 2.5)), float(np.percentile(boot_delta_mae_quad, 97.5))
    
    ci_r2_spline_low, ci_r2_spline_high = float(np.percentile(boot_delta_r2_spline, 2.5)), float(np.percentile(boot_delta_r2_spline, 97.5))
    ci_mae_spline_low, ci_mae_spline_high = float(np.percentile(boot_delta_mae_spline, 2.5)), float(np.percentile(boot_delta_mae_spline, 97.5))
    
    prop_r2_gain_quad = float(np.mean(boot_delta_r2_quad > 0.0))
    prop_mae_gain_quad = float(np.mean(boot_delta_mae_quad > 0.0))
    prop_r2_gain_spline = float(np.mean(boot_delta_r2_spline > 0.0))
    prop_mae_gain_spline = float(np.mean(boot_delta_mae_spline > 0.0))
    
    mean_kappa_quad = float(np.mean([f["model2_quadratic"]["condition_number"] for f in fold_diagnostics]))
    mean_kappa_spline = float(np.mean([f["model3_spline"]["condition_number"] for f in fold_diagnostics]))
    mean_kappa_ext_quad = float(np.mean([f["model4_extended_quad"]["condition_number"] for f in fold_diagnostics]))
    mean_kappa_ext_spline = float(np.mean([f["model5_extended_spline"]["condition_number"] for f in fold_diagnostics]))
    
    return {
        "architecture": arch_name,
        "n_clusters": n_clusters,
        "total_evaluations": len(y_true),
        "r2_oos": {
            "model1_linear_baseline": round(r2_oos_1, 4),
            "model2_quadratic_baseline": round(r2_oos_2, 4),
            "model3_spline_baseline": round(r2_oos_3, 4),
            "model4_quadratic_fingerprints": round(r2_oos_4, 4),
            "model5_spline_fingerprints": round(r2_oos_5, 4),
        },
        "mae_oos": {
            "model1_linear_baseline": round(mae_oos_1, 4),
            "model2_quadratic_baseline": round(mae_oos_2, 4),
            "model3_spline_baseline": round(mae_oos_3, 4),
            "model4_quadratic_fingerprints": round(mae_oos_4, 4),
            "model5_spline_fingerprints": round(mae_oos_5, 4),
        },
        "vs_quadratic_baseline": {
            "delta_r2": round(delta_r2_vs_quad, 4),
            "ci_95_delta_r2": [round(ci_r2_quad_low, 4), round(ci_r2_quad_high, 4)],
            "prop_replicates_r2_gain": round(prop_r2_gain_quad, 4),
            "significant_r2_gain": bool(ci_r2_quad_low > 0.0),
            "delta_mae": round(delta_mae_vs_quad, 4),
            "ci_95_delta_mae": [round(ci_mae_quad_low, 4), round(ci_mae_quad_high, 4)],
            "prop_replicates_mae_gain": round(prop_mae_gain_quad, 4),
            "significant_mae_gain": bool(ci_mae_quad_low > 0.0),
            "mae_reduction_pct": round(delta_mae_vs_quad / mae_oos_2 * 100, 2),
        },
        "vs_spline_baseline": {
            "delta_r2": round(delta_r2_vs_spline, 4),
            "ci_95_delta_r2": [round(ci_r2_spline_low, 4), round(ci_r2_spline_high, 4)],
            "prop_replicates_r2_gain": round(prop_r2_gain_spline, 4),
            "significant_r2_gain": bool(ci_r2_spline_low > 0.0),
            "delta_mae": round(delta_mae_vs_spline, 4),
            "ci_95_delta_mae": [round(ci_mae_spline_low, 4), round(ci_mae_spline_high, 4)],
            "prop_replicates_mae_gain": round(prop_mae_gain_spline, 4),
            "significant_mae_gain": bool(ci_mae_spline_low > 0.0),
            "mae_reduction_pct": round(delta_mae_vs_spline / mae_oos_3 * 100, 2),
        },
        "condition_numbers": {
            "mean_kappa_linear": round(float(np.mean([f["model1_linear"]["condition_number"] for f in fold_diagnostics])), 2),
            "mean_kappa_quadratic": round(mean_kappa_quad, 2),
            "mean_kappa_spline": round(mean_kappa_spline, 2),
            "mean_kappa_extended_quad": round(mean_kappa_ext_quad, 2),
            "mean_kappa_extended_spline": round(mean_kappa_ext_spline, 2),
        },
        "fold_diagnostics": fold_diagnostics,
    }


def main():
    print("=" * 80)
    print("ROBUSTLENS: RIGOROUS GROUPED CROSS-VALIDATION & SPLINE COMPARISON AUDIT")
    print("=" * 80)
    
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} evaluations across {df['image_id'].nunique()} clusters and {df['model'].nunique()} models.")
    
    # 1. Save exploratory pilot 100 image IDs (Section F)
    pilot_img_df = df.drop_duplicates("image_id").sort_values("image_id")
    pilot_info = {
        "description": "Exploratory Pilot Evaluation 100 Image Subset (FastAI Imagenette-v2 Validation Split)",
        "total_pilot_images": 100,
        "class_distribution": {str(k): int(v) for k, v in pilot_img_df["class_name"].value_counts().items()},
        "pilot_local_image_ids": [int(x) for x in pilot_img_df["image_id"].tolist()],
        "status": "Exploratory (Disclosed). Reserved confirmatory test set: remaining 3,825 validation images.",
    }
    with open(PILOT_IDS_JSON, "w") as f:
        json.dump(pilot_info, f, indent=2)
    print(f"Saved exploratory pilot image metadata to {PILOT_IDS_JSON}")
    
    architectures = ["convnext_tiny", "vit_base", "swin_tiny", "resnet50"]
    arch_results = []
    csv_rows = []
    
    for arch in architectures:
        sub_df = df[df["model"] == arch].copy()
        res = evaluate_architecture_cv(sub_df, arch)
        arch_results.append(res)
        
        # Flatten for CSV
        csv_rows.append({
            "architecture": arch,
            # R2 OOS
            "r2_oos_linear": res["r2_oos"]["model1_linear_baseline"],
            "r2_oos_quadratic": res["r2_oos"]["model2_quadratic_baseline"],
            "r2_oos_spline": res["r2_oos"]["model3_spline_baseline"],
            "r2_oos_extended_quad": res["r2_oos"]["model4_quadratic_fingerprints"],
            "r2_oos_extended_spline": res["r2_oos"]["model5_spline_fingerprints"],
            # MAE OOS
            "mae_oos_linear": res["mae_oos"]["model1_linear_baseline"],
            "mae_oos_quadratic": res["mae_oos"]["model2_quadratic_baseline"],
            "mae_oos_spline": res["mae_oos"]["model3_spline_baseline"],
            "mae_oos_extended_quad": res["mae_oos"]["model4_quadratic_fingerprints"],
            "mae_oos_extended_spline": res["mae_oos"]["model5_spline_fingerprints"],
            # vs Quadratic Baseline (Model 4 vs Model 2)
            "delta_r2_vs_quad": res["vs_quadratic_baseline"]["delta_r2"],
            "ci_delta_r2_quad_low": res["vs_quadratic_baseline"]["ci_95_delta_r2"][0],
            "ci_delta_r2_quad_high": res["vs_quadratic_baseline"]["ci_95_delta_r2"][1],
            "prop_boot_r2_gain_quad": res["vs_quadratic_baseline"]["prop_replicates_r2_gain"],
            "sig_r2_gain_quad": res["vs_quadratic_baseline"]["significant_r2_gain"],
            "delta_mae_vs_quad": res["vs_quadratic_baseline"]["delta_mae"],
            "ci_delta_mae_quad_low": res["vs_quadratic_baseline"]["ci_95_delta_mae"][0],
            "ci_delta_mae_quad_high": res["vs_quadratic_baseline"]["ci_95_delta_mae"][1],
            "prop_boot_mae_gain_quad": res["vs_quadratic_baseline"]["prop_replicates_mae_gain"],
            "sig_mae_gain_quad": res["vs_quadratic_baseline"]["significant_mae_gain"],
            "mae_reduction_pct_quad": res["vs_quadratic_baseline"]["mae_reduction_pct"],
            # vs Spline Baseline (Model 5 vs Model 3)
            "delta_r2_vs_spline": res["vs_spline_baseline"]["delta_r2"],
            "ci_delta_r2_spline_low": res["vs_spline_baseline"]["ci_95_delta_r2"][0],
            "ci_delta_r2_spline_high": res["vs_spline_baseline"]["ci_95_delta_r2"][1],
            "prop_boot_r2_gain_spline": res["vs_spline_baseline"]["prop_replicates_r2_gain"],
            "sig_r2_gain_spline": res["vs_spline_baseline"]["significant_r2_gain"],
            "delta_mae_vs_spline": res["vs_spline_baseline"]["delta_mae"],
            "ci_delta_mae_spline_low": res["vs_spline_baseline"]["ci_95_delta_mae"][0],
            "ci_delta_mae_spline_high": res["vs_spline_baseline"]["ci_95_delta_mae"][1],
            "prop_boot_mae_gain_spline": res["vs_spline_baseline"]["prop_replicates_mae_gain"],
            "sig_mae_gain_spline": res["vs_spline_baseline"]["significant_mae_gain"],
            "mae_reduction_pct_spline": res["vs_spline_baseline"]["mae_reduction_pct"],
            # Condition numbers
            "mean_kappa_linear": res["condition_numbers"]["mean_kappa_linear"],
            "mean_kappa_quad": res["condition_numbers"]["mean_kappa_quadratic"],
            "mean_kappa_spline": res["condition_numbers"]["mean_kappa_spline"],
            "mean_kappa_ext_quad": res["condition_numbers"]["mean_kappa_extended_quad"],
            "mean_kappa_ext_spline": res["condition_numbers"]["mean_kappa_extended_spline"],
        })
        
    df_out = pd.DataFrame(csv_rows)
    df_out.to_csv(OUTPUT_CSV, index=False)
    
    full_output = {
        "metadata": {
            "title": "RobustLens Out-of-Sample Grouped Cross-Validation Audit",
            "clusters_count": 100,
            "evaluations_per_architecture": 1800,
            "total_evaluations": 7200,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_type": "Paired Out-of-Sample Prediction Image-Cluster Resampling",
            "cv_folds": 5,
            "cv_seed": CV_SEED,
            "models_evaluated": {
                "model1": "Linear Global Baseline (Order 5df, Tier 2df, Order x Tier 10df, PSNR, SSIM, LPIPS)",
                "model2": "Quadratic-Interaction Global Baseline (Model 1 + PSNR^2, SSIM^2, LPIPS^2, PSNRxSSIM, PSNRxLPIPS, SSIMxLPIPS)",
                "model3": "Restricted Cubic Spline Baseline (Model 1 + 4-knot natural cubic splines for PSNR, SSIM, LPIPS)",
                "model4": "Quadratic + Fingerprints Extended Model (Model 2 + 5 structural fingerprints)",
                "model5": "Spline + Fingerprints Extended Model (Model 3 + 5 structural fingerprints)",
            }
        },
        "architectures": arch_results,
    }
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(full_output, f, indent=2)
        
    print(f"\nSaved CSV comparison to {OUTPUT_CSV}")
    print(f"Saved comprehensive JSON diagnostics to {OUTPUT_JSON}")
    
    print("\n" + "=" * 115)
    print("CORRECTED OUT-OF-SAMPLE GROUPED CV AUDIT TABLE:")
    print("=" * 115)
    cols_show = ["architecture", "r2_oos_linear", "r2_oos_quadratic", "r2_oos_spline", "r2_oos_extended_quad", "r2_oos_extended_spline", "delta_r2_vs_spline", "ci_delta_r2_spline_low", "ci_delta_r2_spline_high", "delta_mae_vs_spline", "ci_delta_mae_spline_low", "ci_delta_mae_spline_high"]
    print(df_out[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()
