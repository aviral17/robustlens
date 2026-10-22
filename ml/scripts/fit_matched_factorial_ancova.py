"""Fit full factorial cluster-robust logistic ANCOVA on matched-fidelity predictions.

Evaluates:
  logit(P(Y=1)) = beta0 + sum(beta_order * I(order)) + sum(beta_target * I(target))
                  + sum(beta_inter * I(order)*I(target)) + gamma_psnr * PSNR + gamma_ssim * SSIM

Accounts for physical image clusters (N=3,925, 117,750 observations per model).
Computes:
1. Model 1 (Main Effects Model): Order (5 df) + PSNR + SSIM
2. Model 2 (Full Factorial Model): Order (5 df) + Target (4 df) + Order x Target (20 df) + PSNR + SSIM
3. Compares with nominal severity benchmark to measure gap contraction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MATCHED_DIR = os.path.join(REPO_ROOT, "results", "order_effect_study", "matched_fidelity", "outputs")
NOMINAL_DIR = os.path.join(REPO_ROOT, "results", "order_effect_study", "full", "outputs")


def run(matched_dir: str = MATCHED_DIR, nominal_dir: str = NOMINAL_DIR) -> pd.DataFrame:
    csv_gz = os.path.join(matched_dir, "all_order_predictions.csv.gz")
    if not os.path.exists(csv_gz):
        raise FileNotFoundError(f"Predictions file not found: {csv_gz}")

    print(f"Reading matched-fidelity predictions from {csv_gz}...")
    df = pd.read_csv(csv_gz)

    # Determine target column name (target_idx or severity)
    target_col = "target_idx" if "target_idx" in df.columns else "severity"

    models = ["convnext_tiny", "swin_tiny", "vit_base", "resnet50"]
    results = []

    for m in models:
        print(f"\n=======================================================")
        print(f"Fitting Cluster-Robust Logistic ANCOVA for {m}")
        print(f"=======================================================")
        sub = df[(df["model"] == m) & (df["order"] != "clean")].copy()
        y = sub["correct"].astype(float).values
        groups = sub["sample_idx"].values
        n_clusters = len(np.unique(groups))
        n_obs = len(y)

        order_dummies = pd.get_dummies(sub["order"], drop_first=True, dtype=float, prefix="ord")
        target_dummies = pd.get_dummies(sub[target_col].astype(str), drop_first=True, dtype=float, prefix="tgt")

        # Interaction terms: Order x Target
        inter_cols = {}
        for o_col in order_dummies.columns:
            for t_col in target_dummies.columns:
                inter_cols[f"{o_col}_x_{t_col}"] = order_dummies[o_col].values * target_dummies[t_col].values
        inter_df = pd.DataFrame(inter_cols, index=sub.index)

        # 1. Model 1: Main Effects Only (Order + PSNR + SSIM)
        X1 = pd.concat([sub[["psnr", "ssim"]], order_dummies], axis=1)
        X1 = sm.add_constant(X1)
        res1 = sm.Logit(y, X1).fit(cov_type="cluster", cov_kwds={"groups": groups}, disp=False, maxiter=100)

        # Test order (5 df) in Model 1
        r1_order = np.zeros((len(order_dummies.columns), len(X1.columns)))
        for i, c in enumerate(order_dummies.columns):
            r1_order[i, X1.columns.get_loc(c)] = 1.0
        w1_order = res1.wald_test(r1_order)

        # 2. Model 2: Full Factorial Model (Order + Target + Order x Target + PSNR + SSIM)
        X2 = pd.concat([sub[["psnr", "ssim"]], order_dummies, target_dummies, inter_df], axis=1)
        X2 = sm.add_constant(X2)
        res2 = sm.Logit(y, X2).fit(cov_type="cluster", cov_kwds={"groups": groups}, disp=False, maxiter=100)

        # Wald test for Order Main Effect (5 df)
        r2_order = np.zeros((len(order_dummies.columns), len(X2.columns)))
        for i, c in enumerate(order_dummies.columns):
            r2_order[i, X2.columns.get_loc(c)] = 1.0
        w2_order = res2.wald_test(r2_order)

        # Wald test for Target Main Effect (4 df)
        r2_tgt = np.zeros((len(target_dummies.columns), len(X2.columns)))
        for i, c in enumerate(target_dummies.columns):
            r2_tgt[i, X2.columns.get_loc(c)] = 1.0
        w2_tgt = res2.wald_test(r2_tgt)

        # Wald test for Order x Target Interaction (20 df)
        r2_inter = np.zeros((len(inter_df.columns), len(X2.columns)))
        for i, c in enumerate(inter_df.columns):
            r2_inter[i, X2.columns.get_loc(c)] = 1.0
        w2_inter = res2.wald_test(r2_inter)

        # Wald test for Joint Order Effect (Order main + Order x Target, 25 df)
        all_order_cols = list(order_dummies.columns) + list(inter_df.columns)
        r2_joint_order = np.zeros((len(all_order_cols), len(X2.columns)))
        for i, c in enumerate(all_order_cols):
            r2_joint_order[i, X2.columns.get_loc(c)] = 1.0
        w2_joint_order = res2.wald_test(r2_joint_order)

        print(f"Model 1 (Order + PSNR + SSIM): Clustered Wald stat = {w1_order.statistic:.3f} (p = {w1_order.pvalue:.4e}, 5 df)")
        print(f"Model 2 Full Factorial Pseudo R2: {res2.prsquared:.4f}")
        print(f"  Order Main Effect (5 df): Wald = {w2_order.statistic:.3f} (p = {w2_order.pvalue:.4e})")
        print(f"  Target Main Effect (4 df): Wald = {w2_tgt.statistic:.3f} (p = {w2_tgt.pvalue:.4e})")
        print(f"  Order x Target Interaction (20 df): Wald = {w2_inter.statistic:.3f} (p = {w2_inter.pvalue:.4e})")
        print(f"  Joint Order Terms (25 df): Wald = {w2_joint_order.statistic:.3f} (p = {w2_joint_order.pvalue:.4e})")
        print(f"  PSNR coef = {res2.params['psnr']:.4f} (z = {res2.tvalues['psnr']:.2f}, p = {res2.pvalues['psnr']:.4e})")
        print(f"  SSIM coef = {res2.params['ssim']:.4f} (z = {res2.tvalues['ssim']:.2f}, p = {res2.pvalues['ssim']:.4e})")

        results.append({
            "model": m,
            "pseudo_r2_m1": round(res1.prsquared, 4),
            "m1_order_wald_stat": round(float(w1_order.statistic), 3),
            "m1_order_p_value": float(w1_order.pvalue),
            "pseudo_r2_m2": round(res2.prsquared, 4),
            "m2_order_main_wald_stat": round(float(w2_order.statistic), 3),
            "m2_order_main_p_value": float(w2_order.pvalue),
            "m2_target_main_wald_stat": round(float(w2_tgt.statistic), 3),
            "m2_target_main_p_value": float(w2_tgt.pvalue),
            "m2_order_x_target_wald_stat": round(float(w2_inter.statistic), 3),
            "m2_order_x_target_p_value": float(w2_inter.pvalue),
            "m2_joint_order_wald_stat": round(float(w2_joint_order.statistic), 3),
            "m2_joint_order_p_value": float(w2_joint_order.pvalue),
            "coef_psnr": round(float(res2.params["psnr"]), 4),
            "pval_psnr": float(res2.pvalues["psnr"]),
            "coef_ssim": round(float(res2.params["ssim"]), 4),
            "pval_ssim": float(res2.pvalues["ssim"]),
            "n_observations": n_obs,
            "n_clusters": n_clusters,
        })

    res_df = pd.DataFrame(results)
    out_csv = os.path.join(matched_dir, "matched_factorial_ancova.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved factorial ANCOVA results to {out_csv}")
    return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-dir", default=MATCHED_DIR)
    parser.add_argument("--nominal-dir", default=NOMINAL_DIR)
    args = parser.parse_args()
    run(args.matched_dir, args.nominal_dir)
