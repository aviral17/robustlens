#!/usr/bin/env python3
"""Run Full-Rank Saturated 6-Cell Order Decomposition, Prediction Instability Transitions, and Clustered Bootstrap Analysis.

This script implements the rigorous statistical analysis for RobustLens:
1. Two distinct estimands:
   - Total effect of sequence order at fixed operator parameters (Nominal-Order regime).
   - Descriptive order differences at comparable output fidelity (Approximate-Fidelity regime).
2. Full-rank saturated six-cell decomposition with a balanced parity contrast
   (intercept, 2 start contrasts, 2 end contrasts, 1 balanced parity contrast),
   plus direct marginal means and contrasts.
3. Clustered bootstrap on physical image clusters (B = 5,000 resamples).
4. Full 5-state joint prediction space decomposition:
   - Both Correct (E_11)
   - Loss (E_10)
   - Gain (E_01)
   - Same Wrong Label (E_00^same)
   - Error Migration / Different Wrong Labels (E_00^diff)
   Distinguishing Predicted-Label Churn P(Y_hat_A != Y_hat_B) from Correctness Churn P(C_A != C_B).
5. Brier scores, confidence shifts, and per-class vulnerability profiles.
6. Multiplicity corrections in log space (Holm-Bonferroni FWER, Benjamini-Hochberg FDR)
   without literal 0.0 p-values.
7. Outputs structured CSV and JSON artifacts to reports/final_research_package/analysis/.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm


# 1. Define Sequence Permutations and Contrast Design Matrix
ORDERS = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]

# Design matrix for orders in sequence: [NBJ, NJB, BNJ, BJN, JNB, JBN]
# Reference: Start = Noise (N), End = Noise (N)
# Columns: [Intercept, Start_B, Start_J, End_B, End_J, Parity]
# Group 1 (Even Permutations, Parity = +1): NBJ (Start N, End J), BJN (Start B, End N), JNB (Start J, End B)
# Group 2 (Odd Permutations, Parity = -1): NJB (Start N, End B), BNJ (Start B, End J), JBN (Start J, End N)
X_DESIGN = np.array([
    [1.0, 0.0, 0.0, 0.0, 1.0,  1.0],  # NBJ
    [1.0, 0.0, 0.0, 1.0, 0.0, -1.0],  # NJB
    [1.0, 1.0, 0.0, 0.0, 1.0, -1.0],  # BNJ
    [1.0, 1.0, 0.0, 0.0, 0.0,  1.0],  # BJN
    [1.0, 0.0, 1.0, 1.0, 0.0,  1.0],  # JNB
    [1.0, 0.0, 1.0, 0.0, 0.0, -1.0],  # JBN
], dtype=float)

INV_X = np.linalg.inv(X_DESIGN)
PARAM_NAMES = ["Intercept", "Start_Blur", "Start_JPEG", "End_Blur", "End_JPEG", "Parity_Contrast"]


def holm_bonferroni_log(log_p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted log-p-values in log space (FWER control)."""
    n = len(log_p_values)
    sort_idx = np.argsort(log_p_values)
    adj_log_p = np.zeros(n, dtype=float)
    running_max = -np.inf
    for rank, idx in enumerate(sort_idx):
        multiplier = n - rank
        raw_log_p = log_p_values[idx]
        adj = min(0.0, raw_log_p + np.log(multiplier))
        running_max = max(running_max, adj)
        adj_log_p[idx] = running_max
    return adj_log_p


def benjamini_hochberg_log(log_p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted log-p-values in log space (FDR control)."""
    n = len(log_p_values)
    sort_idx = np.argsort(log_p_values)
    adj_log_p = np.zeros(n, dtype=float)
    running_min = 0.0
    for rank in reversed(range(n)):
        idx = sort_idx[rank]
        raw_log_p = log_p_values[idx]
        multiplier_log = np.log(n) - np.log(rank + 1)
        adj = min(0.0, raw_log_p + multiplier_log)
        running_min = min(running_min, adj)
        adj_log_p[idx] = running_min
    return adj_log_p


def format_log_p_value(log_p: float) -> str:
    """Convert log_e(p) to a numerically stable scientific string without literal 0.0."""
    if log_p >= np.log(1e-4):
        p_val = float(np.exp(log_p))
        return f"{p_val:.4f}"
    
    exponent_10 = log_p / np.log(10.0)
    exp_int = int(np.floor(exponent_10))
    mantissa = 10.0 ** (exponent_10 - exp_int)
    if mantissa >= 9.995:
        mantissa /= 10.0
        exp_int += 1
    return f"{mantissa:.2f}e{exp_int}"


def compute_marginal_contrasts(cell_means: np.ndarray) -> dict:
    """Compute direct marginal means and contrasts across orders.
    
    cell_means: array of shape (6,) in order [NBJ, NJB, BNJ, BJN, JNB, JBN]
    """
    mu = dict(zip(ORDERS, cell_means))
    
    start_n = 0.5 * (mu["NBJ"] + mu["NJB"])
    start_b = 0.5 * (mu["BNJ"] + mu["BJN"])
    start_j = 0.5 * (mu["JNB"] + mu["JBN"])
    
    end_n = 0.5 * (mu["BJN"] + mu["JBN"])
    end_b = 0.5 * (mu["NJB"] + mu["JNB"])
    end_j = 0.5 * (mu["NBJ"] + mu["BNJ"])
    
    even = (mu["NBJ"] + mu["BJN"] + mu["JNB"]) / 3.0
    odd = (mu["NJB"] + mu["BNJ"] + mu["JBN"]) / 3.0
    parity_contrast = even - odd
    
    return {
        "start_Noise": start_n,
        "start_Blur": start_b,
        "start_JPEG": start_j,
        "start_Blur_vs_Noise": start_b - start_n,
        "start_JPEG_vs_Noise": start_j - start_n,
        "end_Noise": end_n,
        "end_Blur": end_b,
        "end_JPEG": end_j,
        "end_Blur_vs_Noise": end_b - end_n,
        "end_JPEG_vs_Noise": end_j - end_n,
        "even_mean": even,
        "odd_mean": odd,
        "parity_contrast": parity_contrast,
    }


def run_decomposition_and_transitions_for_regime(
    df: pd.DataFrame,
    regime_name: str,
    severity_col: str,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    """Run full saturated decomposition, 5-state prediction space transitions, and clustered bootstrap."""
    print(f"\n=======================================================")
    print(f"Running Analysis: {regime_name} (Severity col: {severity_col})")
    print(f"Total rows in raw data: {len(df)}")

    # Add Brier score: (confidence - correct)^2
    df["brier"] = (df["confidence"] - df["correct"]) ** 2

    # Filter only compound corruption orders
    df_eval = df[df["order"].isin(ORDERS)].copy()
    print(f"Compound evaluation rows: {len(df_eval)}")

    models = sorted(df_eval["model"].unique().tolist())
    image_ids = np.sort(df_eval["sample_idx"].unique())
    n_images = len(image_ids)
    print(f"Unique images: {n_images}, Models: {models}")

    # Build image-level pivot arrays for vectorized bootstrap
    img_to_idx = {img_id: idx for idx, img_id in enumerate(image_ids)}
    model_to_idx = {m: idx for idx, m in enumerate(models)}
    order_to_idx = {o: idx for idx, o in enumerate(ORDERS)}

    df_eval["img_row"] = df_eval["sample_idx"].map(img_to_idx)
    df_eval["model_row"] = df_eval["model"].map(model_to_idx)
    df_eval["order_row"] = df_eval["order"].map(order_to_idx)

    # Average over severities per image, model, order
    agg_df = df_eval.groupby(
        ["img_row", "model_row", "order_row"], as_index=False
    ).agg({
        "correct": "mean",
        "confidence": "mean",
        "psnr": "mean",
        "ssim": "mean",
        "brier": "mean",
    })

    acc_tensor = np.zeros((n_images, len(models), len(ORDERS)), dtype=float)
    conf_tensor = np.zeros((n_images, len(models), len(ORDERS)), dtype=float)
    psnr_tensor = np.zeros((n_images, len(models), len(ORDERS)), dtype=float)
    ssim_tensor = np.zeros((n_images, len(models), len(ORDERS)), dtype=float)
    brier_tensor = np.zeros((n_images, len(models), len(ORDERS)), dtype=float)

    for _, row in agg_df.iterrows():
        i = int(row["img_row"])
        m = int(row["model_row"])
        o = int(row["order_row"])
        acc_tensor[i, m, o] = row["correct"]
        conf_tensor[i, m, o] = row["confidence"]
        psnr_tensor[i, m, o] = row["psnr"]
        ssim_tensor[i, m, o] = row["ssim"]
        brier_tensor[i, m, o] = row["brier"]

    # Compute overall across-models mean tensor: (n_images, 1, 6)
    acc_all = np.mean(acc_tensor, axis=1, keepdims=True)
    conf_all = np.mean(conf_tensor, axis=1, keepdims=True)
    psnr_all = np.mean(psnr_tensor, axis=1, keepdims=True)
    ssim_all = np.mean(ssim_tensor, axis=1, keepdims=True)
    brier_all = np.mean(brier_tensor, axis=1, keepdims=True)

    all_models_tensor = np.concatenate([acc_tensor, acc_all], axis=1)
    all_conf_tensor = np.concatenate([conf_tensor, conf_all], axis=1)
    all_psnr_tensor = np.concatenate([psnr_tensor, psnr_all], axis=1)
    all_ssim_tensor = np.concatenate([ssim_tensor, ssim_all], axis=1)
    all_brier_tensor = np.concatenate([brier_tensor, brier_all], axis=1)

    model_labels = models + ["Overall"]

    # Point Estimates
    mean_cells_acc = np.mean(all_models_tensor, axis=0)  # (n_models+1, 6)
    mean_cells_conf = np.mean(all_conf_tensor, axis=0)
    mean_cells_psnr = np.mean(all_psnr_tensor, axis=0)
    mean_cells_ssim = np.mean(all_ssim_tensor, axis=0)
    mean_cells_brier = np.mean(all_brier_tensor, axis=0)

    # Betas = Inv(X) @ means.T -> shape (6, n_models+1) -> transpose to (n_models+1, 6)
    beta_acc = (INV_X @ mean_cells_acc.T).T
    beta_conf = (INV_X @ mean_cells_conf.T).T
    beta_psnr = (INV_X @ mean_cells_psnr.T).T
    beta_ssim = (INV_X @ mean_cells_ssim.T).T
    beta_brier = (INV_X @ mean_cells_brier.T).T

    # Clustered Bootstrap on Images (B = 5,000)
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n_images, size=(n_boot, n_images))

    print(f"Executing {n_boot} clustered bootstrap resamples across {n_images} physical images...")
    boot_means_acc = np.mean(all_models_tensor[boot_indices], axis=1)
    boot_betas_acc = np.einsum("bmi,ji->bmj", boot_means_acc, INV_X)

    se_acc = np.std(boot_betas_acc, axis=0)
    ci_lower_acc = np.percentile(boot_betas_acc, 2.5, axis=0)
    ci_upper_acc = np.percentile(boot_betas_acc, 97.5, axis=0)
    z_acc = np.where(se_acc > 1e-12, beta_acc / se_acc, 0.0)
    abs_z_acc = np.abs(z_acc)
    log_p_acc = np.where(abs_z_acc > 0, np.log(2.0) + norm.logsf(abs_z_acc), 0.0)

    # Secondary bootstrap metrics
    boot_means_conf = np.mean(all_conf_tensor[boot_indices], axis=1)
    boot_betas_conf = np.einsum("bmi,ji->bmj", boot_means_conf, INV_X)
    se_conf = np.std(boot_betas_conf, axis=0)

    boot_means_psnr = np.mean(all_psnr_tensor[boot_indices], axis=1)
    boot_betas_psnr = np.einsum("bmi,ji->bmj", boot_means_psnr, INV_X)
    se_psnr = np.std(boot_betas_psnr, axis=0)

    boot_means_ssim = np.mean(all_ssim_tensor[boot_indices], axis=1)
    boot_betas_ssim = np.einsum("bmi,ji->bmj", boot_means_ssim, INV_X)
    se_ssim = np.std(boot_betas_ssim, axis=0)

    boot_means_brier = np.mean(all_brier_tensor[boot_indices], axis=1)
    boot_betas_brier = np.einsum("bmi,ji->bmj", boot_means_brier, INV_X)
    se_brier = np.std(boot_betas_brier, axis=0)

    # Format coefficients table
    coeff_records = []
    log_p_primary = []

    for m_idx, m_name in enumerate(model_labels):
        for p_idx, p_name in enumerate(PARAM_NAMES):
            b = beta_acc[m_idx, p_idx]
            se = se_acc[m_idx, p_idx]
            cil = ci_lower_acc[m_idx, p_idx]
            ciu = ci_upper_acc[m_idx, p_idx]
            z = z_acc[m_idx, p_idx]
            lp = log_p_acc[m_idx, p_idx]
            log_p_primary.append(lp)

            coeff_records.append({
                "regime": regime_name,
                "model": m_name,
                "parameter": p_name,
                "metric": "Top-1 Accuracy",
                "estimate": float(b),
                "se_boot": float(se),
                "ci_lower": float(cil),
                "ci_upper": float(ciu),
                "z_score": float(z),
                "log_p_raw": float(lp),
                "p_raw": format_log_p_value(lp),
                "conf_estimate": float(beta_conf[m_idx, p_idx]),
                "conf_se": float(se_conf[m_idx, p_idx]),
                "psnr_estimate": float(beta_psnr[m_idx, p_idx]),
                "psnr_se": float(se_psnr[m_idx, p_idx]),
                "ssim_estimate": float(beta_ssim[m_idx, p_idx]),
                "ssim_se": float(se_ssim[m_idx, p_idx]),
                "brier_estimate": float(beta_brier[m_idx, p_idx]),
                "brier_se": float(se_brier[m_idx, p_idx]),
            })

    # Apply Holm-Bonferroni and Benjamini-Hochberg in log space
    log_p_primary_arr = np.array(log_p_primary)
    adj_log_p_holm = holm_bonferroni_log(log_p_primary_arr)
    adj_log_p_bh = benjamini_hochberg_log(log_p_primary_arr)

    for idx, rec in enumerate(coeff_records):
        rec["log_p_holm"] = float(adj_log_p_holm[idx])
        rec["log_p_bh"] = float(adj_log_p_bh[idx])
        rec["p_holm_fwer"] = format_log_p_value(adj_log_p_holm[idx])
        rec["p_bh_fdr"] = format_log_p_value(adj_log_p_bh[idx])

    # Direct marginal contrasts per model
    marginal_records = []
    for m_idx, m_name in enumerate(model_labels):
        m_contrasts = compute_marginal_contrasts(mean_cells_acc[m_idx])
        m_contrasts["regime"] = regime_name
        m_contrasts["model"] = m_name
        marginal_records.append(m_contrasts)

    # Cell Means Table
    cell_records = []
    cell_se_acc = np.std(boot_means_acc, axis=0)
    for m_idx, m_name in enumerate(model_labels):
        for o_idx, o_name in enumerate(ORDERS):
            cell_records.append({
                "regime": regime_name,
                "model": m_name,
                "order": o_name,
                "accuracy": float(mean_cells_acc[m_idx, o_idx]),
                "accuracy_se": float(cell_se_acc[m_idx, o_idx]),
                "confidence": float(mean_cells_conf[m_idx, o_idx]),
                "psnr": float(mean_cells_psnr[m_idx, o_idx]),
                "ssim": float(mean_cells_ssim[m_idx, o_idx]),
                "brier": float(mean_cells_brier[m_idx, o_idx]),
            })

    # 4. Joint Prediction-Space Transitions and Error Churn
    # For every pair (A, B) across the 6 orders
    churn_records = []
    pair_transition_list = []

    # Prepare pivots for fast transition calculations
    # Build a lookup dictionary per model
    model_pivots = {}
    for m_name in models:
        sub = df_eval[df_eval["model"] == m_name].sort_values(["sample_idx", severity_col, "order"])
        piv_pred = sub.pivot(index=["sample_idx", severity_col], columns="order", values="prediction")
        piv_corr = sub.pivot(index=["sample_idx", severity_col], columns="order", values="correct")
        piv_conf = sub.pivot(index=["sample_idx", severity_col], columns="order", values="confidence")
        # true label
        piv_true = sub.pivot(index=["sample_idx", severity_col], columns="order", values="true_label")
        model_pivots[m_name] = (piv_pred, piv_corr, piv_conf, piv_true)

    for m_name in models:
        piv_pred, piv_corr, piv_conf, piv_true = model_pivots[m_name]
        for i_a in range(len(ORDERS)):
            for i_b in range(i_a + 1, len(ORDERS)):
                ord_a = ORDERS[i_a]
                ord_b = ORDERS[i_b]

                pa = piv_pred[ord_a].to_numpy()
                pb = piv_pred[ord_b].to_numpy()
                ca = piv_corr[ord_a].to_numpy().astype(int)
                cb = piv_corr[ord_b].to_numpy().astype(int)
                confa = piv_conf[ord_a].to_numpy()
                confb = piv_conf[ord_b].to_numpy()

                n_matched = len(ca)

                # 5-State Joint Partition
                e11 = np.mean((ca == 1) & (cb == 1))
                e10 = np.mean((ca == 1) & (cb == 0))  # Loss
                e01 = np.mean((ca == 0) & (cb == 1))  # Gain
                e00_same = np.mean((ca == 0) & (cb == 0) & (pa == pb))
                e00_diff = np.mean((ca == 0) & (cb == 0) & (pa != pb))  # Error Migration

                # Verify partition sums to 1.0
                assert np.isclose(e11 + e10 + e01 + e00_same + e00_diff, 1.0), "Partition must sum to 1.0"

                # Churn Endpoints
                delta_acc = float(np.mean(cb) - np.mean(ca))
                correctness_churn = e10 + e01
                excess_corr_churn = 2.0 * min(e10, e01)

                label_churn = float(np.mean(pa != pb))
                excess_label_churn = label_churn - abs(delta_acc)
                agreement_rate = e11 + e00_same

                # Algebraic identity verification: Label Churn == Correctness Churn + Error Migration
                assert np.isclose(label_churn, correctness_churn + e00_diff), "Label churn identity failed"
                assert np.isclose(agreement_rate, 1.0 - label_churn), "Agreement identity failed"

                conf_shift = float(np.mean(np.abs(confa - confb)))

                churn_records.append({
                    "regime": regime_name,
                    "model": m_name,
                    "order_A": ord_a,
                    "order_B": ord_b,
                    "n_pairs": n_matched,
                    "acc_A": float(np.mean(ca)),
                    "acc_B": float(np.mean(cb)),
                    "delta_acc": delta_acc,
                    "e11_both_correct": float(e11),
                    "e10_loss": float(e10),
                    "e01_gain": float(e01),
                    "e00_same_wrong": float(e00_same),
                    "e00_diff_wrong": float(e00_diff),
                    "correctness_churn": float(correctness_churn),
                    "excess_correctness_churn": float(excess_corr_churn),
                    "label_churn": float(label_churn),
                    "excess_label_churn": float(excess_label_churn),
                    "agreement_rate": float(agreement_rate),
                    "mean_abs_conf_shift": conf_shift,
                    "identity_verified": True,
                })

    # Summary of transitions by model and overall
    df_churn_regime = pd.DataFrame(churn_records)
    transition_summary = []
    
    # By model
    for m_name in models:
        m_sub = df_churn_regime[df_churn_regime["model"] == m_name]
        transition_summary.append({
            "regime": regime_name,
            "model": m_name,
            "mean_acc": float((m_sub["acc_A"].mean() + m_sub["acc_B"].mean()) / 2.0),
            "mean_abs_delta_acc": float(np.abs(m_sub["delta_acc"]).mean()),
            "e11_both_correct": float(m_sub["e11_both_correct"].mean()),
            "e10_loss": float(m_sub["e10_loss"].mean()),
            "e01_gain": float(m_sub["e01_gain"].mean()),
            "e00_same_wrong": float(m_sub["e00_same_wrong"].mean()),
            "e00_diff_wrong": float(m_sub["e00_diff_wrong"].mean()),
            "correctness_churn": float(m_sub["correctness_churn"].mean()),
            "excess_correctness_churn": float(m_sub["excess_correctness_churn"].mean()),
            "label_churn": float(m_sub["label_churn"].mean()),
            "excess_label_churn": float(m_sub["excess_label_churn"].mean()),
            "agreement_rate": float(m_sub["agreement_rate"].mean()),
            "mean_abs_conf_shift": float(m_sub["mean_abs_conf_shift"].mean()),
        })

    # Overall across models
    transition_summary.append({
        "regime": regime_name,
        "model": "Overall",
        "mean_acc": float((df_churn_regime["acc_A"].mean() + df_churn_regime["acc_B"].mean()) / 2.0),
        "mean_abs_delta_acc": float(np.abs(df_churn_regime["delta_acc"]).mean()),
        "e11_both_correct": float(df_churn_regime["e11_both_correct"].mean()),
        "e10_loss": float(df_churn_regime["e10_loss"].mean()),
        "e01_gain": float(df_churn_regime["e01_gain"].mean()),
        "e00_same_wrong": float(df_churn_regime["e00_same_wrong"].mean()),
        "e00_diff_wrong": float(df_churn_regime["e00_diff_wrong"].mean()),
        "correctness_churn": float(df_churn_regime["correctness_churn"].mean()),
        "excess_correctness_churn": float(df_churn_regime["excess_correctness_churn"].mean()),
        "label_churn": float(df_churn_regime["label_churn"].mean()),
        "excess_label_churn": float(df_churn_regime["excess_label_churn"].mean()),
        "agreement_rate": float(df_churn_regime["agreement_rate"].mean()),
        "mean_abs_conf_shift": float(df_churn_regime["mean_abs_conf_shift"].mean()),
    })

    # 5. Per-Class Vulnerability Breakdown
    classes = sorted(df_eval["true_class_name"].unique())
    per_class_records = []
    for c_name in classes:
        sub_c = df_eval[df_eval["true_class_name"] == c_name]
        acc_c = sub_c["correct"].mean()
        conf_c = sub_c["confidence"].mean()
        brier_c = sub_c["brier"].mean()

        # Across order pairs for this class
        piv_p_c = sub_c.pivot(index=["sample_idx", "model", severity_col], columns="order", values="prediction")
        piv_c_c = sub_c.pivot(index=["sample_idx", "model", severity_col], columns="order", values="correct")

        c_label_churn = []
        c_corr_churn = []
        c_e11 = []
        c_e00_same = []
        c_e00_diff = []

        for i_a in range(len(ORDERS)):
            for i_b in range(i_a + 1, len(ORDERS)):
                oa = ORDERS[i_a]
                ob = ORDERS[i_b]
                pa = piv_p_c[oa].to_numpy()
                pb = piv_p_c[ob].to_numpy()
                ca = piv_c_c[oa].to_numpy()
                cb = piv_c_c[ob].to_numpy()

                c_label_churn.append(np.mean(pa != pb))
                c_corr_churn.append(np.mean(ca != cb))
                c_e11.append(np.mean((ca == 1) & (cb == 1)))
                c_e00_same.append(np.mean((ca == 0) & (cb == 0) & (pa == pb)))
                c_e00_diff.append(np.mean((ca == 0) & (cb == 0) & (pa != pb)))

        per_class_records.append({
            "regime": regime_name,
            "class_name": c_name,
            "accuracy": float(acc_c),
            "confidence": float(conf_c),
            "brier_score": float(brier_c),
            "label_churn": float(np.mean(c_label_churn)),
            "correctness_churn": float(np.mean(c_corr_churn)),
            "error_migration": float(np.mean(c_e00_diff)),
            "same_wrong": float(np.mean(c_e00_same)),
            "both_correct": float(np.mean(c_e11)),
        })

    return {
        "coefficients": coeff_records,
        "cell_means": cell_records,
        "marginal_contrasts": marginal_records,
        "churn": churn_records,
        "transition_summary": transition_summary,
        "per_class": per_class_records,
    }


def main():
    root_dir = Path(__file__).resolve().parents[2]
    out_dir = root_dir / "reports/final_research_package/analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Full Nominal Study
    nominal_path = root_dir / "results/order_effect_study/full/outputs/all_order_predictions.csv.gz"
    print(f"Reading Nominal data from {nominal_path}...")
    df_nominal = pd.read_csv(nominal_path)

    # Load Approximate-Fidelity Study
    matched_path = root_dir / "results/order_effect_study/matched_fidelity/outputs/all_order_predictions.csv.gz"
    print(f"Reading Matched Fidelity data from {matched_path}...")
    df_matched = pd.read_csv(matched_path)

    # Run decompositions and transitions
    res_nominal = run_decomposition_and_transitions_for_regime(
        df_nominal,
        regime_name="Nominal-Order",
        severity_col="severity",
        n_boot=5000,
        seed=101,
    )

    res_matched = run_decomposition_and_transitions_for_regime(
        df_matched,
        regime_name="Approximate-Fidelity",
        severity_col="target_idx",
        n_boot=5000,
        seed=202,
    )

    # Combine results
    all_coeffs = res_nominal["coefficients"] + res_matched["coefficients"]
    all_cells = res_nominal["cell_means"] + res_matched["cell_means"]
    all_marginals = res_nominal["marginal_contrasts"] + res_matched["marginal_contrasts"]
    all_churn = res_nominal["churn"] + res_matched["churn"]
    all_transitions = res_nominal["transition_summary"] + res_matched["transition_summary"]
    all_classes = res_nominal["per_class"] + res_matched["per_class"]

    df_coeffs = pd.DataFrame(all_coeffs)
    df_cells = pd.DataFrame(all_cells)
    df_marginals = pd.DataFrame(all_marginals)
    df_churn = pd.DataFrame(all_churn)
    df_transitions = pd.DataFrame(all_transitions)
    df_classes = pd.DataFrame(all_classes)

    # Save CSVs
    coeffs_csv = out_dir / "decomposition_coefficients.csv"
    cells_csv = out_dir / "order_cell_means.csv"
    marginals_csv = out_dir / "marginal_contrasts.csv"
    churn_csv = out_dir / "error_churn_matrix.csv"
    transitions_csv = out_dir / "prediction_instability_transitions.csv"
    classes_csv = out_dir / "per_class_instability_summary.csv"

    df_coeffs.to_csv(coeffs_csv, index=False)
    df_cells.to_csv(cells_csv, index=False)
    df_marginals.to_csv(marginals_csv, index=False)
    df_churn.to_csv(churn_csv, index=False)
    df_transitions.to_csv(transitions_csv, index=False)
    df_classes.to_csv(classes_csv, index=False)

    print(f"\nSaved Analysis CSVs:")
    print(f" - {coeffs_csv}")
    print(f" - {cells_csv}")
    print(f" - {marginals_csv}")
    print(f" - {churn_csv}")
    print(f" - {transitions_csv}")
    print(f" - {classes_csv}")

    # Compute Model-Level Sequence Gap Contraction
    print("\n--- Model-Level Sequence Gap Contraction ---")
    contraction_records = []
    models = ["resnet50", "convnext_tiny", "vit_base", "swin_tiny", "Overall"]

    for m in models:
        nom_sub = df_cells[(df_cells["regime"] == "Nominal-Order") & (df_cells["model"] == m)]
        match_sub = df_cells[(df_cells["regime"] == "Approximate-Fidelity") & (df_cells["model"] == m)]

        nom_accs = nom_sub["accuracy"].to_numpy()
        match_accs = match_sub["accuracy"].to_numpy()

        nom_gap = float(np.max(nom_accs) - np.min(nom_accs))
        match_gap = float(np.max(match_accs) - np.min(match_accs))
        contraction = float((nom_gap - match_gap) / nom_gap) * 100.0 if nom_gap > 0 else 0.0

        # Also get label churn and correctness churn contraction
        nom_tr = df_transitions[(df_transitions["regime"] == "Nominal-Order") & (df_transitions["model"] == m)].iloc[0]
        mat_tr = df_transitions[(df_transitions["regime"] == "Approximate-Fidelity") & (df_transitions["model"] == m)].iloc[0]

        rec = {
            "model": m,
            "nominal_mean_acc": float(np.mean(nom_accs)),
            "nominal_min_acc": float(np.min(nom_accs)),
            "nominal_max_acc": float(np.max(nom_accs)),
            "nominal_acc_gap": nom_gap,
            "matched_mean_acc": float(np.mean(match_accs)),
            "matched_min_acc": float(np.min(match_accs)),
            "matched_max_acc": float(np.max(match_accs)),
            "matched_acc_gap": match_gap,
            "acc_gap_contraction_pct": contraction,
            "nominal_label_churn": float(nom_tr["label_churn"]),
            "matched_label_churn": float(mat_tr["label_churn"]),
            "nominal_correctness_churn": float(nom_tr["correctness_churn"]),
            "matched_correctness_churn": float(mat_tr["correctness_churn"]),
            "nominal_error_migration": float(nom_tr["e00_diff_wrong"]),
            "matched_error_migration": float(mat_tr["e00_diff_wrong"]),
        }
        contraction_records.append(rec)
        print(f"Model: {m:20s} | Nom Acc Gap: {nom_gap*100:5.2f}% | Matched Acc Gap: {match_gap*100:5.2f}% | Contraction: {contraction:5.2f}%")

    df_contraction = pd.DataFrame(contraction_records)
    df_contraction.to_csv(out_dir / "nominal_vs_matched_gap_contraction.csv", index=False)

    # Save Full JSON Summary
    summary_data = {
        "metadata": {
            "title": "RobustLens Order Decomposition, Instability Transitions, and Clustered Bootstrap",
            "date": "2026-10-18",
            "n_clusters_images": 3925,
            "n_bootstrap_replicates": 5000,
            "design_matrix": X_DESIGN.tolist(),
            "parameter_names": PARAM_NAMES,
            "orders": ORDERS,
        },
        "gap_contraction": contraction_records,
        "prediction_instability_transitions": all_transitions,
        "marginal_contrasts": all_marginals,
        "per_class_summary": all_classes,
        "sample_coefficients": all_coeffs[:24],
    }
    with open(out_dir / "decomposition_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Summary JSON written to {out_dir / 'decomposition_summary.json'}")
    print("Done!")


if __name__ == "__main__":
    main()
