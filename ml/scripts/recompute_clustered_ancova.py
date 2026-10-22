"""Recompute distortion-controlled ANCOVA with cluster-robust standard errors.

The evaluation uses repeated measures (30 compound conditions) from N=3,925 physical images.
Standard errors and Wald hypothesis tests are clustered at the physical image level (sample_idx)
to correctly account for intra-cluster correlation.
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
    ancova_rows = []

    for m in models:
        sub = df[(df["model"] == m) & (df["order"] != "clean")].copy()
        dummies = pd.get_dummies(sub["order"], drop_first=False, dtype=float)
        if "NBJ" in dummies.columns:
            dummies = dummies.drop(columns=["NBJ"])
        order_cols = list(dummies.columns)

        X = pd.concat([sub[["psnr", "ssim"]], dummies], axis=1)
        X = sm.add_constant(X)
        y = sub["correct"].astype(float)
        groups = sub["sample_idx"]

        # Cluster-robust logistic regression clustered on image (sample_idx)
        res_cl = sm.Logit(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups}, disp=False, maxiter=100)

        # Unclustered for reference
        res_un = sm.Logit(y, X).fit(disp=False, maxiter=100)

        r_matrix = np.zeros((len(order_cols), len(X.columns)))
        for i, c in enumerate(order_cols):
            r_matrix[i, X.columns.get_loc(c)] = 1.0

        wald_cl = res_cl.wald_test(r_matrix)
        wald_un = res_un.wald_test(r_matrix)

        ancova_rows.append({
            "model": m,
            "pseudo_r2": round(float(res_cl.prsquared), 4),
            "coef_psnr": round(float(res_cl.params["psnr"]), 4),
            "pval_psnr_clustered": float(res_cl.pvalues["psnr"]),
            "coef_ssim": round(float(res_cl.params["ssim"]), 4),
            "pval_ssim_clustered": float(res_cl.pvalues["ssim"]),
            "clustered_wald_stat": round(float(wald_cl.statistic), 3),
            "clustered_wald_p_value": float(wald_cl.pvalue),
            "unclustered_wald_stat": round(float(wald_un.statistic), 3),
            "unclustered_wald_p_value": float(wald_un.pvalue),
            "order_significant_at_05": bool(wald_cl.pvalue < 0.05),
            "n_observations": len(y),
            "n_clusters": int(sub["sample_idx"].nunique()),
        })

    df_ancova = pd.DataFrame(ancova_rows)
    out_csv = os.path.join(FULL_DIR, "distortion_ancova.csv")
    df_ancova.to_csv(out_csv, index=False)
    print(f"\nSaved updated cluster-robust distortion ANCOVA to {out_csv}:")
    print(df_ancova[["model", "pseudo_r2", "clustered_wald_stat", "clustered_wald_p_value", "n_clusters"]])

    # Also update benchmark_report.json
    rep_path = os.path.join(FULL_DIR, "benchmark_report.json")
    if os.path.exists(rep_path):
        with open(rep_path, "r") as f:
            rep = json.load(f)
        rep["distortion_ancova"] = df_ancova.to_dict(orient="records")
        with open(rep_path, "w") as f:
            json.dump(rep, f, indent=2)
        print("Updated benchmark_report.json with clustered ANCOVA.")

if __name__ == "__main__":
    run()
