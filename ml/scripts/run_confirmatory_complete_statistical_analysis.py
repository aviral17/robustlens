"""Complete Confirmatory Downstream Statistical Analysis & Multiplicity Audit.

Executes on full_distortion_fingerprints_and_margins.csv.gz:
1. Dataset Integrity Verification (275,400 rows, 3,825 unique images, 4 models, 18 conditions).
2. Zero Pilot Leakage Verification.
3. Evaluates BOTH specifications:
   - Specification A [PRIMARY]: Conventional 4-Knot Restricted Cubic Spline (RCS, 3 df/var; Model 3 = 27 cols, Model 5 = 32 cols)
   - Specification B [SENSITIVITY]: Reduced 2-df Spline (2 df/var; Model 3 = 24 cols, Model 5 = 29 cols)
4. Grouped 5-Fold Cross-Validation across 3,825 physical image clusters (stratified by true class).
   - Training-only standard scaling and spline knot estimation (5th, 35th, 65th, 95th percentiles).
5. Paired Cluster Bootstrap with B=10,000 resamples:
   - Pointwise 95% CIs: [2.5th, 97.5th] percentiles
   - Simultaneous Bonferroni 2-sided 95% CIs across K=4 architectures: [0.625th, 99.375th] percentiles
   - Simultaneous 1-sided 95% Lower Bound: 1.25th percentile
   - Null-calibrated bootstrap p-values (centered null)
   - Step-down Holm-Bonferroni adjusted p-values across 4 architectures
6. Saves:
   - confirmatory_grouped_cv_predictions.csv.gz
   - confirmatory_statistical_comparison.csv
   - confirmatory_multiplicity_audit.json
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "results" / "matched_fidelity_study"
CONF_DIR = OUTPUT_DIR / "confirmatory_run"
CSV_GZ_PATH = CONF_DIR / "full_distortion_fingerprints_and_margins.csv.gz"
PILOT_IDS_JSON = OUTPUT_DIR / "pilot_exploratory_100_image_ids.json"

BOOTSTRAP_REPLICATES = 10000
CV_SEED = 42
BOOTSTRAP_SEED = 12345


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def reduced_spline_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Reduced 2-df spline basis (1 linear + 1 nonlinear)."""
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


def holm_bonferroni(p_vals):
    m = len(p_vals)
    sorted_idx = np.argsort(p_vals)
    sorted_p = [p_vals[i] for i in sorted_idx]
    adj_p = [0.0] * m
    running_max = 0.0
    for rank_i, (orig_i, pv) in enumerate(zip(sorted_idx, sorted_p)):
        mult = m - rank_i
        curr_adj = min(1.0, mult * pv)
        running_max = max(running_max, curr_adj)
        adj_p[orig_i] = running_max
    return adj_p


def run_full_analysis():
    print("=" * 80)
    print("RUNNING CONFIRMATORY GROUPED CV & MULTIPLICITY ANALYSIS")
    print("=" * 80)

    if not CSV_GZ_PATH.exists():
        print(f"Datafile not found: {CSV_GZ_PATH}")
        return False

    print(f"Loading {CSV_GZ_PATH}...")
    df = pd.read_csv(CSV_GZ_PATH)
    print(f"Loaded {len(df):,} records.")

    # 1. Dataset Integrity Verification
    expected_rows = 3825 * 18 * 4
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
    assert df["image_id"].nunique() == 3825, f"Expected 3825 images, got {df['image_id'].nunique()}"
    assert df["model"].nunique() == 4, f"Expected 4 models, got {df['model'].nunique()}"
    assert df["order"].nunique() == 6, f"Expected 6 sequences, got {df['order'].nunique()}"
    assert df["level"].nunique() == 3, f"Expected 3 levels, got {df['level'].nunique()}"
    assert int(df.isnull().sum().sum()) == 0, "Missing values detected!"

    # Pilot leakage check
    with open(PILOT_IDS_JSON, "r") as fp:
        pilot_data = json.load(fp)
    pilot_rows = set(pilot_data["pilot_dataset_row_indices"])
    df_rows = set(df["dataset_row"].unique())
    leakage = df_rows.intersection(pilot_rows)
    assert len(leakage) == 0, f"LEAKAGE DETECTED: {len(leakage)} pilot rows present in confirmatory dataset!"
    print(f"Verified Zero Pilot Leakage: 0/{len(pilot_rows)} pilot images present.")

    # 2. Setup 5-Fold Grouped Cross-Validation (Stratified by True Class across 3,825 images)
    clusters = np.sort(df["image_id"].unique())
    n_clusters = len(clusters)
    rng_cv = np.random.RandomState(CV_SEED)

    cluster_classes = df.drop_duplicates("image_id").sort_values("image_id")["true_label"].to_numpy()
    fold_assignments = np.zeros(n_clusters, dtype=int)
    for c in range(10):
        c_indices = np.where(cluster_classes == c)[0]
        perm = rng_cv.permutation(c_indices)
        c_splits = np.array_split(perm, 5)
        for fold_i, split_idx in enumerate(c_splits):
            fold_assignments[split_idx] = fold_i

    architectures = ["convnext_tiny", "vit_base", "swin_tiny", "resnet50"]
    cont_cols = ["psnr", "ssim", "lpips", "fft_energy_low", "fft_energy_high", "jpeg_block_artifact", "delta_edge_energy", "std_local_ssim"]

    all_pred_records = []
    results_conv = []
    results_red = []

    # Pre-generate cluster bootstrap resamples for perfect pairing across models
    print(f"\nGenerating B={BOOTSTRAP_REPLICATES:,} cluster bootstrap resamples (seed={BOOTSTRAP_SEED})...")
    rng_b = np.random.RandomState(BOOTSTRAP_SEED)
    cluster_indices = np.arange(n_clusters)
    
    # Pre-sample cluster choice matrix: shape (BOOTSTRAP_REPLICATES, n_clusters)
    bootstrap_cluster_choices = rng_b.randint(0, n_clusters, size=(BOOTSTRAP_REPLICATES, n_clusters))
    # Precompute cluster weights matrix W: shape (BOOTSTRAP_REPLICATES, n_clusters)
    print("Precomputing bootstrap weights matrix...")
    W = np.zeros((BOOTSTRAP_REPLICATES, n_clusters), dtype=float)
    for b in range(BOOTSTRAP_REPLICATES):
        W[b, :] = np.bincount(bootstrap_cluster_choices[b], minlength=n_clusters)
    print("Bootstrap weights matrix ready.")

    for arch in architectures:
        print(f"\nEvaluating Architecture: {arch}...")
        df_arch = df[df["model"] == arch].copy().sort_values(["image_id", "level", "order"]).reset_index(drop=True)
        
        # Arrays for storing predictions aligned with df_arch
        y_true = df_arch["logit_margin"].to_numpy()
        pred_conv3 = np.zeros(len(df_arch))
        pred_conv5 = np.zeros(len(df_arch))
        pred_red3 = np.zeros(len(df_arch))
        pred_red5 = np.zeros(len(df_arch))
        folds_col = np.zeros(len(df_arch), dtype=int)

        for fold_i in range(5):
            test_c = clusters[fold_assignments == fold_i]
            train_c = clusters[fold_assignments != fold_i]

            test_mask = df_arch["image_id"].isin(test_c).to_numpy()
            train_mask = df_arch["image_id"].isin(train_c).to_numpy()

            folds_col[test_mask] = fold_i

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

            # Conventional 4-knot RCS (27 vs 32 cols)
            X_tr_c3 = build_design_matrix(tr_df, "conv", scaler_params, spline_knots)
            X_te_c3 = build_design_matrix(te_df, "conv", scaler_params, spline_knots)
            X_tr_c5 = build_design_matrix(tr_df, "conv_extended", scaler_params, spline_knots)
            X_te_c5 = build_design_matrix(te_df, "conv_extended", scaler_params, spline_knots)

            b_c3 = np.linalg.pinv(X_tr_c3.T @ X_tr_c3) @ (X_tr_c3.T @ y_tr)
            b_c5 = np.linalg.pinv(X_tr_c5.T @ X_tr_c5) @ (X_tr_c5.T @ y_tr)

            pred_conv3[test_mask] = X_te_c3 @ b_c3
            pred_conv5[test_mask] = X_te_c5 @ b_c5

            # Reduced Spline (24 vs 29 cols)
            X_tr_r3 = build_design_matrix(tr_df, "red", scaler_params, spline_knots)
            X_te_r3 = build_design_matrix(te_df, "red", scaler_params, spline_knots)
            X_tr_r5 = build_design_matrix(tr_df, "red_extended", scaler_params, spline_knots)
            X_te_r5 = build_design_matrix(te_df, "red_extended", scaler_params, spline_knots)

            b_r3 = np.linalg.pinv(X_tr_r3.T @ X_tr_r3) @ (X_tr_r3.T @ y_tr)
            b_r5 = np.linalg.pinv(X_tr_r5.T @ X_tr_r5) @ (X_tr_r5.T @ y_tr)

            pred_red3[test_mask] = X_te_r3 @ b_r3
            pred_red5[test_mask] = X_te_r5 @ b_r5

        # Store predictions
        df_arch_preds = df_arch[["image_id", "dataset_row", "true_label", "class_name", "level", "order", "final_op", "model"]].copy()
        df_arch_preds["fold"] = folds_col
        df_arch_preds["true_logit_margin"] = y_true
        df_arch_preds["pred_conv_m3"] = pred_conv3
        df_arch_preds["pred_conv_m5"] = pred_conv5
        df_arch_preds["pred_red_m3"] = pred_red3
        df_arch_preds["pred_red_m5"] = pred_red5
        all_pred_records.append(df_arch_preds)

        # Full Out-of-Sample Overall Point Estimates
        ss_tot = np.sum((y_true - y_true.mean())**2)

        # Conventional
        r2_c3 = float(1.0 - np.sum((y_true - pred_conv3)**2) / ss_tot)
        r2_c5 = float(1.0 - np.sum((y_true - pred_conv5)**2) / ss_tot)
        mae_c3 = float(np.mean(np.abs(y_true - pred_conv3)))
        mae_c5 = float(np.mean(np.abs(y_true - pred_conv5)))
        d_r2_c = float(r2_c5 - r2_c3)
        d_mae_c = float(mae_c3 - mae_c5)

        # Reduced
        r2_r3 = float(1.0 - np.sum((y_true - pred_red3)**2) / ss_tot)
        r2_r5 = float(1.0 - np.sum((y_true - pred_red5)**2) / ss_tot)
        mae_r3 = float(np.mean(np.abs(y_true - pred_red3)))
        mae_r5 = float(np.mean(np.abs(y_true - pred_red5)))
        d_r2_r = float(r2_r5 - r2_r3)
        d_mae_r = float(mae_r3 - mae_r5)

        print(f"  [Conventional] M3 R2: {r2_c3:.4f}, M5 R2: {r2_c5:.4f}, Delta R2: {d_r2_c:+.4f} | M3 MAE: {mae_c3:.4f}, M5 MAE: {mae_c5:.4f}, Delta MAE: {d_mae_c:+.4f}")
        print(f"  [Reduced]      M3 R2: {r2_r3:.4f}, M5 R2: {r2_r5:.4f}, Delta R2: {d_r2_r:+.4f} | M3 MAE: {mae_r3:.4f}, M5 MAE: {mae_r5:.4f}, Delta MAE: {d_mae_r:+.4f}")

        # Vectorized Cluster Bootstrap (Exact BLAS computation across 3,825 image clusters)
        K_per_cluster = 18
        N_eval = len(df_arch)
        assert N_eval == n_clusters * K_per_cluster

        # Reshape to (n_clusters, 18) for ultra-fast cluster reductions
        Y_c = y_true.reshape(n_clusters, K_per_cluster)
        P_c3 = pred_conv3.reshape(n_clusters, K_per_cluster)
        P_c5 = pred_conv5.reshape(n_clusters, K_per_cluster)
        P_r3 = pred_red3.reshape(n_clusters, K_per_cluster)
        P_r5 = pred_red5.reshape(n_clusters, K_per_cluster)

        c_s1 = Y_c.sum(axis=1)               # sum(y) per cluster
        c_s2 = (Y_c**2).sum(axis=1)           # sum(y^2) per cluster

        # SSE per cluster
        c_sse_c3 = ((Y_c - P_c3)**2).sum(axis=1)
        c_sse_c5 = ((Y_c - P_c5)**2).sum(axis=1)
        c_sse_r3 = ((Y_c - P_r3)**2).sum(axis=1)
        c_sse_r5 = ((Y_c - P_r5)**2).sum(axis=1)

        # SAE per cluster
        c_sae_c3 = (np.abs(Y_c - P_c3)).sum(axis=1)
        c_sae_c5 = (np.abs(Y_c - P_c5)).sum(axis=1)
        c_sae_r3 = (np.abs(Y_c - P_r3)).sum(axis=1)
        c_sae_r5 = (np.abs(Y_c - P_r5)).sum(axis=1)

        # Matrix multiply W @ cluster vectors: shape (B,)
        b_sum_y = W @ c_s1
        b_sum_y2 = W @ c_s2
        b_sst = b_sum_y2 - (b_sum_y**2) / float(N_eval)

        # SSEs
        b_sse_c3 = W @ c_sse_c3
        b_sse_c5 = W @ c_sse_c5
        b_sse_r3 = W @ c_sse_r3
        b_sse_r5 = W @ c_sse_r5

        # SAEs
        b_sae_c3 = W @ c_sae_c3
        b_sae_c5 = W @ c_sae_c5
        b_sae_r3 = W @ c_sae_r3
        b_sae_r5 = W @ c_sae_r5

        # Replicate Out-of-Sample metrics
        b_r2_c3 = 1.0 - b_sse_c3 / b_sst
        b_r2_c5 = 1.0 - b_sse_c5 / b_sst
        b_dr2_c = b_r2_c5 - b_r2_c3

        b_mae_c3 = b_sae_c3 / float(N_eval)
        b_mae_c5 = b_sae_c5 / float(N_eval)
        b_dmae_c = b_mae_c3 - b_mae_c5

        b_r2_r3 = 1.0 - b_sse_r3 / b_sst
        b_r2_r5 = 1.0 - b_sse_r5 / b_sst
        b_dr2_r = b_r2_r5 - b_r2_r3

        b_mae_r3 = b_sae_r3 / float(N_eval)
        b_mae_r5 = b_sae_r5 / float(N_eval)
        b_dmae_r = b_mae_r3 - b_mae_r5

        # Pointwise 95% CIs: [2.5th, 97.5th]
        pw_ci_r2_c = [float(np.percentile(b_dr2_c, 2.5)), float(np.percentile(b_dr2_c, 97.5))]
        pw_ci_mae_c = [float(np.percentile(b_dmae_c, 2.5)), float(np.percentile(b_dmae_c, 97.5))]

        # Simultaneous 2-sided Bonferroni 95% CIs across K=4: [0.625th, 99.375th]
        sim_2s_ci_r2_c = [float(np.percentile(b_dr2_c, 0.625)), float(np.percentile(b_dr2_c, 99.375))]
        sim_2s_ci_mae_c = [float(np.percentile(b_dmae_c, 0.625)), float(np.percentile(b_dmae_c, 99.375))]

        # Simultaneous 1-sided 95% Lower Bound across K=4: 1.25th percentile
        sim_1s_low_r2_c = float(np.percentile(b_dr2_c, 1.25))
        sim_1s_low_mae_c = float(np.percentile(b_dmae_c, 1.25))

        # Null-Calibrated Bootstrap P-values (Centered null: Delta_null*(b) = Delta*(b) - Delta_obs)
        p_cal_r2_c = float((1.0 + np.sum((b_dr2_c - d_r2_c) >= d_r2_c)) / (BOOTSTRAP_REPLICATES + 1.0))
        p_cal_mae_c = float((1.0 + np.sum((b_dmae_c - d_mae_c) >= d_mae_c)) / (BOOTSTRAP_REPLICATES + 1.0))

        results_conv.append({
            "architecture": arch,
            "specification": "Conventional 4-Knot RCS (Model 3 = 27 cols, Model 5 = 32 cols)",
            "r2_oos_m3": round(r2_c3, 4),
            "r2_oos_m5": round(r2_c5, 4),
            "delta_r2_oos": round(d_r2_c, 4),
            "ci95_pointwise_delta_r2": [round(pw_ci_r2_c[0], 4), round(pw_ci_r2_c[1], 4)],
            "simultaneous_bonferroni_2s_ci_delta_r2": [round(sim_2s_ci_r2_c[0], 4), round(sim_2s_ci_r2_c[1], 4)],
            "simultaneous_1s_lower_bound_delta_r2": round(sim_1s_low_r2_c, 4),
            "null_calibrated_p_delta_r2": round(p_cal_r2_c, 5),
            "mae_oos_m3": round(mae_c3, 4),
            "mae_oos_m5": round(mae_c5, 4),
            "delta_mae_oos": round(d_mae_c, 4),
            "ci95_pointwise_delta_mae": [round(pw_ci_mae_c[0], 4), round(pw_ci_mae_c[1], 4)],
            "simultaneous_bonferroni_2s_ci_delta_mae": [round(sim_2s_ci_mae_c[0], 4), round(sim_2s_ci_mae_c[1], 4)],
            "simultaneous_1s_lower_bound_delta_mae": round(sim_1s_low_mae_c, 4),
            "null_calibrated_p_delta_mae": round(p_cal_mae_c, 5),
        })

        # Reduced Stats (Sensitivity Analysis)
        pw_ci_r2_r = [float(np.percentile(b_dr2_r, 2.5)), float(np.percentile(b_dr2_r, 97.5))]
        pw_ci_mae_r = [float(np.percentile(b_dmae_r, 2.5)), float(np.percentile(b_dmae_r, 97.5))]
        sim_2s_ci_r2_r = [float(np.percentile(b_dr2_r, 0.625)), float(np.percentile(b_dr2_r, 99.375))]
        sim_2s_ci_mae_r = [float(np.percentile(b_dmae_r, 0.625)), float(np.percentile(b_dmae_r, 99.375))]
        sim_1s_low_r2_r = float(np.percentile(b_dr2_r, 1.25))
        sim_1s_low_mae_r = float(np.percentile(b_dmae_r, 1.25))
        p_cal_r2_r = float((1.0 + np.sum((b_dr2_r - d_r2_r) >= d_r2_r)) / (BOOTSTRAP_REPLICATES + 1.0))
        p_cal_mae_r = float((1.0 + np.sum((b_dmae_r - d_mae_r) >= d_mae_r)) / (BOOTSTRAP_REPLICATES + 1.0))

        results_red.append({
            "architecture": arch,
            "specification": "Reduced 2-df Spline (Model 3 = 24 cols, Model 5 = 29 cols)",
            "r2_oos_m3": round(r2_r3, 4),
            "r2_oos_m5": round(r2_r5, 4),
            "delta_r2_oos": round(d_r2_r, 4),
            "ci95_pointwise_delta_r2": [round(pw_ci_r2_r[0], 4), round(pw_ci_r2_r[1], 4)],
            "simultaneous_bonferroni_2s_ci_delta_r2": [round(sim_2s_ci_r2_r[0], 4), round(sim_2s_ci_r2_r[1], 4)],
            "simultaneous_1s_lower_bound_delta_r2": round(sim_1s_low_r2_r, 4),
            "null_calibrated_p_delta_r2": round(p_cal_r2_r, 5),
            "mae_oos_m3": round(mae_r3, 4),
            "mae_oos_m5": round(mae_r5, 4),
            "delta_mae_oos": round(d_mae_r, 4),
            "ci95_pointwise_delta_mae": [round(pw_ci_mae_r[0], 4), round(pw_ci_mae_r[1], 4)],
            "simultaneous_bonferroni_2s_ci_delta_mae": [round(sim_2s_ci_mae_r[0], 4), round(sim_2s_ci_mae_r[1], 4)],
            "simultaneous_1s_lower_bound_delta_mae": round(sim_1s_low_mae_r, 4),
            "null_calibrated_p_delta_mae": round(p_cal_mae_r, 5),
        })

    # Holm-Bonferroni correction across K=4 architectures (Primary: Conventional)
    p_r2 = [r["null_calibrated_p_delta_r2"] for r in results_conv]
    p_mae = [r["null_calibrated_p_delta_mae"] for r in results_conv]
    h_r2 = holm_bonferroni(p_r2)
    h_mae = holm_bonferroni(p_mae)
    for i, r in enumerate(results_conv):
        r["holm_adj_p_delta_r2"] = round(float(h_r2[i]), 5)
        r["significant_r2_holm_0_05"] = bool(h_r2[i] < 0.05)
        r["holm_adj_p_delta_mae"] = round(float(h_mae[i]), 5)
        r["significant_mae_holm_0_05"] = bool(h_mae[i] < 0.05)

    # Holm-Bonferroni correction (Sensitivity: Reduced)
    p_r2_red = [r["null_calibrated_p_delta_r2"] for r in results_red]
    p_mae_red = [r["null_calibrated_p_delta_mae"] for r in results_red]
    h_r2_red = holm_bonferroni(p_r2_red)
    h_mae_red = holm_bonferroni(p_mae_red)
    for i, r in enumerate(results_red):
        r["holm_adj_p_delta_r2"] = round(float(h_r2_red[i]), 5)
        r["significant_r2_holm_0_05"] = bool(h_r2_red[i] < 0.05)
        r["holm_adj_p_delta_mae"] = round(float(h_mae_red[i]), 5)
        r["significant_mae_holm_0_05"] = bool(h_mae_red[i] < 0.05)

    # Save Grouped-CV Predictions (Deliverable #3)
    df_all_preds = pd.concat(all_pred_records, ignore_index=True)
    preds_gz = CONF_DIR / "confirmatory_grouped_cv_predictions.csv.gz"
    df_all_preds.to_csv(preds_gz, index=False, compression="gzip")
    print(f"\nSaved Grouped-CV Predictions: {preds_gz} ({preds_gz.stat().st_size:,} bytes)")

    # Save Corrected Statistical Comparison CSV (Deliverable #4)
    all_results = results_conv + results_red
    df_res = pd.DataFrame(all_results)
    csv_out = OUTPUT_DIR / "confirmatory_statistical_comparison.csv"
    df_res.to_csv(csv_out, index=False)
    df_res.to_csv(CONF_DIR / "confirmatory_statistical_comparison.csv", index=False)
    print(f"Saved Statistical Comparison CSV: {csv_out}")

    # Save Bootstrap Results CSV/JSON (Deliverable #5)
    summary_json = OUTPUT_DIR / "confirmatory_multiplicity_audit.json"
    audit_payload = {
        "primary_conventional_rcs": results_conv,
        "sensitivity_reduced_spline": results_red,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "cv_seed": CV_SEED,
        "total_clusters": n_clusters,
        "total_evaluations": len(df),
        "architectures": architectures,
        "simultaneous_coverage": {
            "bonferroni_2s_percentiles": [0.625, 99.375],
            "simultaneous_1s_percentile": 1.25,
            "familywise_error_rate_alpha": 0.05,
            "number_of_comparisons_K": 4,
        }
    }
    with open(summary_json, "w") as fp:
        json.dump(audit_payload, fp, indent=2)
    with open(CONF_DIR / "confirmatory_multiplicity_audit.json", "w") as fp:
        json.dump(audit_payload, fp, indent=2)
    print(f"Saved Multiplicity Audit JSON: {summary_json}")

    print("\n" + "=" * 130)
    print("PRIMARY CONFIRMATORY AUDIT TABLE: Conventional 4-Knot RCS (B=10,000 Paired Clusters)")
    print("=" * 130)
    df_conv = pd.DataFrame(results_conv)
    cols_display = ["architecture", "r2_oos_m3", "r2_oos_m5", "delta_r2_oos", "simultaneous_1s_lower_bound_delta_r2", "simultaneous_bonferroni_2s_ci_delta_r2", "null_calibrated_p_delta_r2", "holm_adj_p_delta_r2", "significant_r2_holm_0_05"]
    print(df_conv[cols_display].to_string(index=False))

    print("\n" + "=" * 130)
    print("SECONDARY METRIC: Out-of-Sample MAE Reductions (Conventional 4-Knot RCS)")
    print("=" * 130)
    cols_mae = ["architecture", "mae_oos_m3", "mae_oos_m5", "delta_mae_oos", "simultaneous_1s_lower_bound_delta_mae", "simultaneous_bonferroni_2s_ci_delta_mae", "null_calibrated_p_delta_mae", "holm_adj_p_delta_mae", "significant_mae_holm_0_05"]
    print(df_conv[cols_mae].to_string(index=False))

    print("\n" + "=" * 130)
    print("SENSITIVITY ANALYSIS: Reduced 2-df Spline (B=10,000 Paired Clusters)")
    print("=" * 130)
    df_red = pd.DataFrame(results_red)
    print(df_red[cols_display].to_string(index=False))
    return True


if __name__ == "__main__":
    run_full_analysis()
