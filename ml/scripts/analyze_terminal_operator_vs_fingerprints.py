"""Phase 4: Terminal Operator vs Continuous Fingerprints Disentanglement.

Tests whether structural and frequency fingerprints contain information beyond
the identity of the final corruption operator.

Compares:
1. Terminal Operator Only (3 cols: Intercept + Final_Blur + Final_Noise; Final_JPEG as reference).
2. Global Fidelity + Terminal Operator Dummies (23 cols).
3. Global Fidelity + Full Permutation Dummies (Model 3: 27 cols).
4. Full Model: Model 3 + 5 Fingerprints (Model 5: 32 cols).

Evaluates Out-of-Sample R2 and MAE across 3,825 image clusters.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_phase3_feature_ablations import conventional_rcs_basis

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "results" / "matched_fidelity_study"
CONF_DIR = OUTPUT_DIR / "confirmatory_run"
CSV_GZ_PATH = CONF_DIR / "full_distortion_fingerprints_and_margins.csv.gz"

CV_SEED = 42


def run_operator_disentanglement():
    print("=" * 80)
    print("PHASE 4: TERMINAL OPERATOR VS CONTINUOUS FINGERPRINTS")
    print("=" * 80)

    df = pd.read_csv(CSV_GZ_PATH)
    clusters = np.sort(df["image_id"].unique())
    n_clusters = len(clusters)
    cluster_classes = df.drop_duplicates("image_id").sort_values("image_id")["true_label"].to_numpy()

    rng_cv = np.random.RandomState(CV_SEED)
    fold_assignments = np.zeros(n_clusters, dtype=int)
    for c in range(10):
        c_indices = np.where(cluster_classes == c)[0]
        perm = rng_cv.permutation(c_indices)
        c_splits = np.array_split(perm, 5)
        for fold_i, split_idx in enumerate(c_splits):
            fold_assignments[split_idx] = fold_i

    architectures = ["convnext_tiny", "vit_base", "swin_tiny", "resnet50"]
    cont_cols = ["psnr", "ssim", "lpips", "fft_energy_low", "fft_energy_high", "jpeg_block_artifact", "delta_edge_energy", "std_local_ssim"]

    results = []

    for arch in architectures:
        df_arch = df[df["model"] == arch].copy().sort_values(["image_id", "level", "order"]).reset_index(drop=True)
        y_true = df_arch["logit_margin"].to_numpy()
        N = len(df_arch)
        ss_tot = np.sum((y_true - y_true.mean())**2)

        # 4 Model variants:
        # M_op: Intercept + Terminal Blur + Terminal Noise (3 cols)
        # M_fop: Global Fidelity + Terminal Operator (23 cols)
        # M_ord: Global Fidelity + Full Sequence Dummies (Model 3: 27 cols)
        # M_full: Model 3 + 5 Fingerprints (Model 5: 32 cols)

        p_op = np.zeros(N)
        p_fop = np.zeros(N)
        p_ord = np.zeros(N)
        p_full = np.zeros(N)

        for fold_i in range(5):
            test_c = clusters[fold_assignments == fold_i]
            train_c = clusters[fold_assignments != fold_i]
            test_mask = df_arch["image_id"].isin(test_c).to_numpy()
            train_mask = df_arch["image_id"].isin(train_c).to_numpy()

            tr = df_arch[train_mask]
            te = df_arch[test_mask]

            scaler_params = {c: {"mean": float(np.mean(tr[c])), "std": float(np.std(tr[c], ddof=1)) + 1e-8} for c in cont_cols}
            spline_knots = {c: np.percentile((tr[c] - scaler_params[c]["mean"]) / scaler_params[c]["std"], [5.0, 35.0, 65.0, 95.0]) for c in ["psnr", "ssim", "lpips"]}

            y_tr = tr["logit_margin"].to_numpy()

            # Operator matrices
            def get_mats(part):
                intercept = np.ones((len(part), 1))
                # Terminal op: reference = JPEG
                is_blur = (part["final_op"].to_numpy() == "Blur").astype(float)[:, None]
                is_noise = (part["final_op"].to_numpy() == "Noise").astype(float)[:, None]
                op_mat = np.hstack([intercept, is_blur, is_noise])

                # Splines
                p_sp = conventional_rcs_basis((part["psnr"].to_numpy() - scaler_params["psnr"]["mean"]) / scaler_params["psnr"]["std"], spline_knots["psnr"])
                s_sp = conventional_rcs_basis((part["ssim"].to_numpy() - scaler_params["ssim"]["mean"]) / scaler_params["ssim"]["std"], spline_knots["ssim"])
                l_sp = conventional_rcs_basis((part["lpips"].to_numpy() - scaler_params["lpips"]["mean"]) / scaler_params["lpips"]["std"], spline_knots["lpips"])
                sp_mat = np.hstack([p_sp, s_sp, l_sp])

                # Level
                levels = ["Level_1_Mild", "Level_2_Moderate", "Level_3_SevereCommon"]
                lvl_mat = np.column_stack([(part["level"].to_numpy() == lvl).astype(float) for lvl in levels[1:]])
                # Op x Level
                op_x_lvl = np.column_stack([is_blur * lvl_mat[:, 0:1], is_blur * lvl_mat[:, 1:2], is_noise * lvl_mat[:, 0:1], is_noise * lvl_mat[:, 1:2]])

                fop_mat = np.hstack([op_mat, lvl_mat, op_x_lvl, sp_mat])

                # Full Sequence order
                orders = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]
                ord_mat = np.column_stack([(part["order"].to_numpy() == o).astype(float) for o in orders[1:]])
                ord_x_lvl = np.zeros((len(part), (len(orders) - 1) * (len(levels) - 1)))
                c_idx = 0
                for o_i in range(ord_mat.shape[1]):
                    for l_i in range(lvl_mat.shape[1]):
                        ord_x_lvl[:, c_idx] = ord_mat[:, o_i] * lvl_mat[:, l_i]
                        c_idx += 1
                ord_full_mat = np.hstack([intercept, ord_mat, lvl_mat, ord_x_lvl, sp_mat])

                # Fingerprints
                f_low = (part["fft_energy_low"].to_numpy() - scaler_params["fft_energy_low"]["mean"]) / scaler_params["fft_energy_low"]["std"]
                f_high = (part["fft_energy_high"].to_numpy() - scaler_params["fft_energy_high"]["mean"]) / scaler_params["fft_energy_high"]["std"]
                jpeg_b = (part["jpeg_block_artifact"].to_numpy() - scaler_params["jpeg_block_artifact"]["mean"]) / scaler_params["jpeg_block_artifact"]["std"]
                d_edge = (part["delta_edge_energy"].to_numpy() - scaler_params["delta_edge_energy"]["mean"]) / scaler_params["delta_edge_energy"]["std"]
                s_std_loc = (part["std_local_ssim"].to_numpy() - scaler_params["std_local_ssim"]["mean"]) / scaler_params["std_local_ssim"]["std"]
                fps = np.column_stack([f_low, f_high, jpeg_b, d_edge, s_std_loc])

                full_mat = np.hstack([ord_full_mat, fps])
                return op_mat, fop_mat, ord_full_mat, full_mat

            X_tr_op, X_tr_fop, X_tr_ord, X_tr_full = get_mats(tr)
            X_te_op, X_te_fop, X_te_ord, X_te_full = get_mats(te)

            b_op = np.linalg.pinv(X_tr_op.T @ X_tr_op) @ (X_tr_op.T @ y_tr)
            b_fop = np.linalg.pinv(X_tr_fop.T @ X_tr_fop) @ (X_tr_fop.T @ y_tr)
            b_ord = np.linalg.pinv(X_tr_ord.T @ X_tr_ord) @ (X_tr_ord.T @ y_tr)
            b_full = np.linalg.pinv(X_tr_full.T @ X_tr_full) @ (X_tr_full.T @ y_tr)

            p_op[test_mask] = X_te_op @ b_op
            p_fop[test_mask] = X_te_fop @ b_fop
            p_ord[test_mask] = X_te_ord @ b_ord
            p_full[test_mask] = X_te_full @ b_full

        r2_op = float(1.0 - np.sum((y_true - p_op)**2) / ss_tot)
        r2_fop = float(1.0 - np.sum((y_true - p_fop)**2) / ss_tot)
        r2_ord = float(1.0 - np.sum((y_true - p_ord)**2) / ss_tot)
        r2_full = float(1.0 - np.sum((y_true - p_full)**2) / ss_tot)

        results.append({
            "architecture": arch,
            "r2_terminal_operator_only": round(r2_op, 4),
            "r2_fidelity_plus_terminal_operator": round(r2_fop, 4),
            "r2_fidelity_plus_full_order_model3": round(r2_ord, 4),
            "r2_model5_with_fingerprints": round(r2_full, 4),
            "delta_r2_operator_to_order": round(r2_ord - r2_fop, 4),
            "delta_r2_order_to_fingerprints": round(r2_full - r2_ord, 4),
        })
        print(f"  {arch:15s} | Terminal Op: {r2_op:.4f} | Fidelity+Op: {r2_fop:.4f} | Fidelity+Order (M3): {r2_ord:.4f} | Model 5 (+Fingerprints): {r2_full:.4f} | Fingerprint Gain over Order: {r2_full - r2_ord:+.4f}")

    df_res = pd.DataFrame(results)
    out_csv = CONF_DIR / "confirmatory_terminal_operator_disentanglement.csv"
    df_res.to_csv(out_csv, index=False)
    with open(CONF_DIR / "confirmatory_terminal_operator_disentanglement.json", "w") as fp:
        json.dump(results, fp, indent=2)
    return df_res


if __name__ == "__main__":
    run_operator_disentanglement()
