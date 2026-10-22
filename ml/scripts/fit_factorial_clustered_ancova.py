"""Fit full factorial cluster-robust logistic ANCOVA.

Evaluates:
  logit(P(Y=1)) = beta0 + sum(beta_order * I(order)) + sum(beta_sev * I(sev))
                  + sum(beta_inter * I(order)*I(sev)) + gamma_psnr * PSNR + gamma_ssim * SSIM

Accounts for physical image clusters (N=3,925, 117,750 observations per model).
Computes:
1. Model 1 (Main Effects Model): Order (5 df) + PSNR + SSIM
2. Model 2 (Full Factorial Model): Order (5 df) + Severity (4 df) + Order x Severity (20 df) + PSNR + SSIM
"""

import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FULL_DIR = os.path.join(REPO_ROOT, "results", "order_effect_study", "full", "outputs")

def run():
    csv_gz = os.path.join(FULL_DIR, "all_order_predictions.csv.gz")
    print(f"Reading predictions from {csv_gz}...")
    df = pd.read_csv(csv_gz)

    models = ["convnext_tiny", "swin_tiny", "vit_base", "resnet50"]
    results = []

    for m in models:
        print(f"\n--- Fitting Cluster-Robust Logistic ANCOVA for {m} ---")
        sub = df[(df["model"] == m) & (df["order"] != "clean")].copy()
        y = sub["correct"].astype(float).values
        groups = sub["sample_idx"].values
        n_clusters = len(np.unique(groups))
        n_obs = len(y)

        # Baseline reference: order='BNJ' (lowest acc) or 'NBJ', severity=1
        order_dummies = pd.get_dummies(sub["order"], drop_first=True, dtype=float, prefix="ord")
        sev_dummies = pd.get_dummies(sub["severity"].astype(str), drop_first=True, dtype=float, prefix="sev")

        # Interaction terms
        inter_cols = {}
        for o_col in order_dummies.columns:
            for s_col in sev_dummies.columns:
                inter_cols[f"{o_col}_x_{s_col}"] = order_dummies[o_col].values * sev_dummies[s_col].values
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

        # 2. Model 2: Full Factorial Model (Order + Severity + Order x Severity + PSNR + SSIM)
        X2 = pd.concat([sub[["psnr", "ssim"]], order_dummies, sev_dummies, inter_df], axis=1)
        X2 = sm.add_constant(X2)
        res2 = sm.Logit(y, X2).fit(cov_type="cluster", cov_kwds={"groups": groups}, disp=False, maxiter=100)

        # Wald test for Order Main Effect (5 df)
        r2_order = np.zeros((len(order_dummies.columns), len(X2.columns)))
        for i, c in enumerate(order_dummies.columns):
            r2_order[i, X2.columns.get_loc(c)] = 1.0
        w2_order = res2.wald_test(r2_order)

        # Wald test for Severity Main Effect (4 df)
        r2_sev = np.zeros((len(sev_dummies.columns), len(X2.columns)))
        for i, c in enumerate(sev_dummies.columns):
            r2_sev[i, X2.columns.get_loc(c)] = 1.0
        w2_sev = res2.wald_test(r2_sev)

        # Wald test for Order x Severity Interaction (20 df)
        r2_inter = np.zeros((len(inter_df.columns), len(X2.columns)))
        for i, c in enumerate(inter_df.columns):
            r2_inter[i, X2.columns.get_loc(c)] = 1.0
        w2_inter = res2.wald_test(r2_inter)

        # Wald test for Joint Order Effect (Order main + Order x Severity, 25 df)
        all_order_cols = list(order_dummies.columns) + list(inter_df.columns)
        r2_joint_order = np.zeros((len(all_order_cols), len(X2.columns)))
        for i, c in enumerate(all_order_cols):
            r2_joint_order[i, X2.columns.get_loc(c)] = 1.0
        w2_joint_order = res2.wald_test(r2_joint_order)

        print(f"Model 1 (Order + PSNR + SSIM): Clustered Wald stat = {w1_order.statistic:.3f} (p = {w1_order.pvalue:.4e}, 5 df)")
        print(f"Model 2 Full Factorial Pseudo R2: {res2.prsquared:.4f}")
        print(f"  Order Main Effect (5 df): Wald = {w2_order.statistic:.3f} (p = {w2_order.pvalue:.4e})")
        print(f"  Severity Main Effect (4 df): Wald = {w2_sev.statistic:.3f} (p = {w2_sev.pvalue:.4e})")
        print(f"  Order x Severity Interaction (20 df): Wald = {w2_inter.statistic:.3f} (p = {w2_inter.pvalue:.4e})")
        print(f"  Joint Order Terms (25 df): Wald = {w2_joint_order.statistic:.3f} (p = {w2_joint_order.pvalue:.4e})")
        print(f"  PSNR coef = {res2.params['psnr']:.4f} (z = {res2.tvalues['psnr']:.2f}, p = {res2.pvalues['psnr']:.4e})")
        print(f"  SSIM coef = {res2.params['ssim']:.4f} (z = {res2.tvalues['ssim']:.2f}, p = {res2.pvalues['ssim']:.4e})")

        results.append({
            "model": m,
            "n_observations": n_obs,
            "n_clusters": n_clusters,
            "m1_pseudo_r2": round(float(res1.prsquared), 4),
            "m1_clustered_wald_order_5df": round(float(w1_order.statistic), 3),
            "m1_clustered_pval_order_5df": float(w1_order.pvalue),
            "m2_pseudo_r2": round(float(res2.prsquared), 4),
            "m2_wald_order_main_5df": round(float(w2_order.statistic), 3),
            "m2_pval_order_main_5df": float(w2_order.pvalue),
            "m2_wald_sev_main_4df": round(float(w2_sev.statistic), 3),
            "m2_pval_sev_main_4df": float(w2_sev.pvalue),
            "m2_wald_inter_20df": round(float(w2_inter.statistic), 3),
            "m2_pval_inter_20df": float(w2_inter.pvalue),
            "m2_wald_joint_order_25df": round(float(w2_joint_order.statistic), 3),
            "m2_pval_joint_order_25df": float(w2_joint_order.pvalue),
            "coef_psnr": round(float(res2.params["psnr"]), 4),
            "pval_psnr": float(res2.pvalues["psnr"]),
            "coef_ssim": round(float(res2.params["ssim"]), 4),
            "pval_ssim": float(res2.pvalues["ssim"]),
        })

    res_df = pd.DataFrame(results)
    out_csv = os.path.join(FULL_DIR, "factorial_ancova_results.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved comprehensive factorial ANCOVA results to {out_csv}")
    print(res_df.to_string())

if __name__ == "__main__":
    run()
