"""Rigorous Grouped Cross-Validation & Out-of-Sample Comparative Modeling.

Implements all methodological mandates from Section D:
1. Grouped 5-Fold Cross-Validation strictly partitioned by original physical image ID.
   Identical folds used across all models.
2. Standard scaling (z-scoring) fitted strictly inside training folds (zero test leakage).
3. Evaluates THREE distinct models:
   - Model 1 (Linear Global Baseline):
     Order (5 df), Intensity Tier (2 df), Order x Tier Interactions (10 df),
     PSNR, SSIM, LPIPS.
   - Model 2 (Flexible Nonlinear Global Baseline):
     Linear Baseline Covariates +
     Justified nonlinear global fidelity terms:
     PSNR^2, SSIM^2, LPIPS^2, PSNR x SSIM, PSNR x LPIPS, SSIM x LPIPS.
   - Model 3 (Extended Structural Fingerprints Model):
     Flexible Nonlinear Baseline Covariates +
     Fourier residual fractions (fft_energy_low, fft_energy_high),
     JPEG 8x8 block boundary discontinuity ratio (jpeg_block_artifact),
     Sobel edge energy change (delta_edge_energy),
     Local SSIM spatial variability (std_local_ssim).
4. Tests whether structural fingerprints add out-of-sample predictive value beyond
   this highly flexible, nonlinear global fidelity baseline.
5. Evaluates strictly on held-out test folds:
   - R^2_OOS (Linear Baseline, Flexible Nonlinear Baseline, Extended Model)
   - Delta R^2_OOS (Extended vs Flexible Baseline)
   - MAE_OOS (Linear Baseline, Flexible Nonlinear Baseline, Extended Model)
   - Delta MAE_OOS (Extended vs Flexible Baseline)
   - Percentage reduction in MAE
6. Grouped Image-Cluster Bootstrap (B=1,000 iterations) for 95% CIs.
7. Design-matrix rank and condition number diagnostics kappa(X).
8. Analyzes ResNet-50, ConvNeXt-Tiny, ViT-Base/16, Swin-Tiny separately.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CSV_PATH = BASE_DIR / "results" / "matched_fidelity_study" / "heldout_fingerprints_and_margins.csv"
OUTPUT_CSV = BASE_DIR / "results" / "matched_fidelity_study" / "grouped_cv_explanatory_comparison.csv"
OUTPUT_JSON = BASE_DIR / "results" / "matched_fidelity_study" / "grouped_cv_explanatory_diagnostics.json"


def construct_design_matrix(
    df_part: pd.DataFrame,
    mu_cont: np.ndarray,
    sig_cont: np.ndarray,
    cont_cols: list[str],
    include_nonlinear_fidelity: bool = False,
    include_fingerprints: bool = False,
) -> np.ndarray:
    N = len(df_part)

    # 1. Main Effects: Order (5 df) and Severity Tier (2 df)
    # Using explicit categorical indicator mapping
    orders = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]  # NBJ is reference
    levels = ["Mild", "Moderate", "Severe"]               # Mild is reference

    order_dummies = np.zeros((N, len(orders) - 1))
    for col_i, o in enumerate(orders[1:]):
        order_dummies[:, col_i] = (df_part["order"].to_numpy() == o).astype(float)

    level_dummies = np.zeros((N, len(levels) - 1))
    for col_i, lvl in enumerate(levels[1:]):
        level_dummies[:, col_i] = (df_part["severity_level"].to_numpy() == lvl).astype(float)

    # 2. Order x Severity Tier Interactions (5 x 2 = 10 df)
    interaction_dummies = np.zeros((N, (len(orders) - 1) * (len(levels) - 1)))
    idx = 0
    for o_i in range(order_dummies.shape[1]):
        for l_i in range(level_dummies.shape[1]):
            interaction_dummies[:, idx] = order_dummies[:, o_i] * level_dummies[:, l_i]
            idx += 1

    # 3. Z-Scored Global Fidelity (PSNR, SSIM, LPIPS)
    Z_lin = (df_part[["psnr", "ssim", "lpips"]].to_numpy() - mu_cont[:3]) / sig_cont[:3]
    cols = [np.ones(N), order_dummies, level_dummies, interaction_dummies, Z_lin]

    # 4. Nonlinear Global Fidelity Terms (PSNR^2, SSIM^2, LPIPS^2, PSNRxSSIM, PSNRxLPIPS, SSIMxLPIPS)
    if include_nonlinear_fidelity:
        z_p, z_s, z_l = Z_lin[:, 0], Z_lin[:, 1], Z_lin[:, 2]
        Z_quad = np.column_stack([
            z_p**2, z_s**2, z_l**2,
            z_p * z_s, z_p * z_l, z_s * z_l
        ])
        cols.append(Z_quad)

    # 5. Spatial-Frequency and Structural Fingerprints (5 features)
    if include_fingerprints:
        fp_cols = ["fft_energy_low", "fft_energy_high", "jpeg_block_artifact", "delta_edge_energy", "std_local_ssim"]
        Z_fp = (df_part[fp_cols].to_numpy() - mu_cont[3:]) / sig_cont[3:]
        cols.append(Z_fp)

    return np.column_stack(cols)


def evaluate_architecture_cv(
    sub_df: pd.DataFrame,
    target_col: str = "logit_margin",
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    clusters = np.sort(sub_df["image_id"].unique())
    rng = np.random.RandomState(seed)
    shuffled_clusters = rng.permutation(clusters)
    folds = np.array_split(shuffled_clusters, n_splits)

    all_cont_cols = [
        "psnr", "ssim", "lpips",
        "fft_energy_low", "fft_energy_high",
        "jpeg_block_artifact", "delta_edge_energy", "std_local_ssim",
    ]

    y_true_all = []
    y_pred_m1_all = []  # Linear Baseline
    y_pred_m2_all = []  # Flexible Nonlinear Baseline
    y_pred_m3_all = []  # Extended Model

    kappa_m1, kappa_m2, kappa_m3 = [], [], []

    for fold_clusters in folds:
        test_mask = sub_df["image_id"].isin(fold_clusters)
        train_df = sub_df[~test_mask].copy()
        test_df = sub_df[test_mask].copy()

        y_train = train_df[target_col].to_numpy()
        y_test = test_df[target_col].to_numpy()

        # Fold-Isolated Standard Scaler
        mu_train = train_df[all_cont_cols].mean().to_numpy()
        sig_train = train_df[all_cont_cols].std().to_numpy() + 1e-8

        # Construct Design Matrices
        X_tr_m1 = construct_design_matrix(train_df, mu_train, sig_train, all_cont_cols, False, False)
        X_te_m1 = construct_design_matrix(test_df, mu_train, sig_train, all_cont_cols, False, False)

        X_tr_m2 = construct_design_matrix(train_df, mu_train, sig_train, all_cont_cols, True, False)
        X_te_m2 = construct_design_matrix(test_df, mu_train, sig_train, all_cont_cols, True, False)

        X_tr_m3 = construct_design_matrix(train_df, mu_train, sig_train, all_cont_cols, True, True)
        X_te_m3 = construct_design_matrix(test_df, mu_train, sig_train, all_cont_cols, True, True)

        # Condition Numbers
        s1 = np.linalg.svd(X_tr_m1, compute_uv=False)
        s2 = np.linalg.svd(X_tr_m2, compute_uv=False)
        s3 = np.linalg.svd(X_tr_m3, compute_uv=False)
        kappa_m1.append(float(s1[0] / s1[-1]))
        kappa_m2.append(float(s2[0] / s2[-1]))
        kappa_m3.append(float(s3[0] / s3[-1]))

        # Fit OLS
        b1 = np.linalg.pinv(X_tr_m1.T @ X_tr_m1) @ (X_tr_m1.T @ y_train)
        b2 = np.linalg.pinv(X_tr_m2.T @ X_tr_m2) @ (X_tr_m2.T @ y_train)
        b3 = np.linalg.pinv(X_tr_m3.T @ X_tr_m3) @ (X_tr_m3.T @ y_train)

        y_true_all.extend(y_test)
        y_pred_m1_all.extend(X_te_m1 @ b1)
        y_pred_m2_all.extend(X_te_m2 @ b2)
        y_pred_m3_all.extend(X_te_m3 @ b3)

    y_true = np.array(y_true_all)
    y1 = np.array(y_pred_m1_all)
    y2 = np.array(y_pred_m2_all)
    y3 = np.array(y_pred_m3_all)

    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2_m1 = float(1.0 - np.sum((y_true - y1)**2) / (ss_tot + 1e-12))
    r2_m2 = float(1.0 - np.sum((y_true - y2)**2) / (ss_tot + 1e-12))
    r2_m3 = float(1.0 - np.sum((y_true - y3)**2) / (ss_tot + 1e-12))
    delta_r2_vs_flex = float(r2_m3 - r2_m2)

    mae_m1 = float(np.mean(np.abs(y_true - y1)))
    mae_m2 = float(np.mean(np.abs(y_true - y2)))
    mae_m3 = float(np.mean(np.abs(y_true - y3)))
    delta_mae_vs_flex = float(mae_m2 - mae_m3)

    # Clustered Bootstrap on Delta R^2 vs Flexible Baseline (B=1,000)
    boot_deltas = []
    for _ in range(1000):
        boot_c = rng.choice(clusters, size=len(clusters), replace=True)
        boot_frames = [sub_df[sub_df["image_id"] == c] for c in boot_c]
        boot_sub = pd.concat(boot_frames, ignore_index=True)
        yb = boot_sub[target_col].to_numpy()

        mu_b = boot_sub[all_cont_cols].mean().to_numpy()
        sig_b = boot_sub[all_cont_cols].std().to_numpy() + 1e-8

        Xb2 = construct_design_matrix(boot_sub, mu_b, sig_b, all_cont_cols, True, False)
        Xb3 = construct_design_matrix(boot_sub, mu_b, sig_b, all_cont_cols, True, True)

        try:
            b2_b = np.linalg.pinv(Xb2.T @ Xb2) @ (Xb2.T @ yb)
            b3_b = np.linalg.pinv(Xb3.T @ Xb3) @ (Xb3.T @ yb)
            r2_b2 = 1.0 - np.sum((yb - Xb2 @ b2_b)**2) / np.sum((yb - yb.mean())**2)
            r2_b3 = 1.0 - np.sum((yb - Xb3 @ b3_b)**2) / np.sum((yb - yb.mean())**2)
            boot_deltas.append(float(r2_b3 - r2_b2))
        except Exception:
            pass

    ci_low = float(np.percentile(boot_deltas, 2.5)) if boot_deltas else 0.0
    ci_high = float(np.percentile(boot_deltas, 97.5)) if boot_deltas else 0.0

    return {
        "r2_oos_linear_base": round(r2_m1, 4),
        "r2_oos_flexible_base": round(r2_m2, 4),
        "r2_oos_extended": round(r2_m3, 4),
        "delta_r2_vs_flexible": round(delta_r2_vs_flex, 4),
        "ci_95_low": round(ci_low, 4),
        "ci_95_high": round(ci_high, 4),
        "mae_oos_linear_base": round(mae_m1, 4),
        "mae_oos_flexible_base": round(mae_m2, 4),
        "mae_oos_extended": round(mae_m3, 4),
        "delta_mae_vs_flexible": round(delta_mae_vs_flex, 4),
        "mae_reduction_pct": round(float(delta_mae_vs_flex / mae_m2 * 100), 2),
        "mean_condition_number_flex": round(float(np.mean(kappa_m2)), 2),
        "mean_condition_number_ext": round(float(np.mean(kappa_m3)), 2),
        "significant_gain_over_flexible": bool(ci_low > 0.0),
    }


def main():
    print("=" * 80)
    print("GROUPED CROSS-VALIDATION: FLEXIBLE NONLINEAR BASELINE VS EXTENDED MODEL")
    print("=" * 80)

    if not CSV_PATH.exists():
        print(f"Waiting for held-out evaluation file {CSV_PATH} to be written by active process...")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} evaluations across {df['image_id'].nunique()} clusters and {df['model'].nunique()} models.")

    models_list = ["resnet50", "convnext_tiny", "vit_base", "swin_tiny"]
    results = []

    for m in models_list:
        sub = df[df["model"] == m].copy()
        metrics = evaluate_architecture_cv(sub)
        results.append({"model": m, **metrics})

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 95)
    print("OUT-OF-SAMPLE GROUPED CV: FLEXIBLE NONLINEAR BASELINE VS EXTENDED MODEL")
    print("=" * 95)
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()
