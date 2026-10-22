"""Comparative analysis: Nominal Severity Benchmark vs. Matched-Fidelity Benchmark.

Computes:
1. Cross-sequence accuracy variance reduction:
   Var_matched / Var_nominal
2. Maximum pairwise sequence accuracy gap contraction:
   (Gap_nominal - Gap_matched) / Gap_nominal
3. Sequence rank correlation (Spearman rho) between nominal and matched regimes.
4. Churn decoupling: Excess churn at matched fidelity vs nominal severity.
5. Logistic Wald statistic reduction (Order main effect Wald nominal vs matched).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NOMINAL_DIR = os.path.join(REPO_ROOT, "results", "order_effect_study", "full", "outputs")
MATCHED_DIR = os.path.join(REPO_ROOT, "results", "order_effect_study", "matched_fidelity", "outputs")
OUT_DIR = MATCHED_DIR


def run(nominal_dir: str = NOMINAL_DIR, matched_dir: str = MATCHED_DIR) -> dict:
    nom_summary_path = os.path.join(nominal_dir, "order_effects_summary.csv")
    match_summary_path = os.path.join(matched_dir, "order_effects_summary.csv")
    nom_ancova_path = os.path.join(nominal_dir, "distortion_ancova.csv")
    match_ancova_path = os.path.join(matched_dir, "matched_factorial_ancova.csv")
    nom_churn_path = os.path.join(nominal_dir, "error_churn_summary.csv")
    match_churn_path = os.path.join(matched_dir, "error_churn_summary.csv")

    for p in [nom_summary_path, match_summary_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing summary file: {p}")

    df_nom = pd.read_csv(nom_summary_path)
    df_match = pd.read_csv(match_summary_path)

    # Standardize column names
    if "severity" in df_match.columns and "target_idx" not in df_match.columns:
        df_match["target_idx"] = df_match["severity"]
    if "severity" not in df_nom.columns and "target_idx" in df_nom.columns:
        df_nom["severity"] = df_nom["target_idx"]

    # Exclude clean baseline for order effects comparison
    df_nom_comp = df_nom[df_nom["order"] != "clean"].copy()
    df_match_comp = df_match[df_match["order"] != "clean"].copy()

    models = ["convnext_tiny", "swin_tiny", "vit_base", "resnet50"]
    comparison_results = []

    print("=" * 75)
    print("ROBUSTLENS: NOMINAL SEVERITY VS. MATCHED FIDELITY COMPARISON")
    print("=" * 75)

    for m in models:
        nom_m = df_nom_comp[df_nom_comp["model"] == m]
        match_m = df_match_comp[df_match_comp["model"] == m]

        # Overall compound accuracy
        nom_mean_acc = nom_m["accuracy"].mean()
        match_mean_acc = match_m["accuracy"].mean()

        # By sequence accuracy across all 5 levels
        nom_seq_acc = nom_m.groupby("order")["accuracy"].mean()
        match_seq_acc = match_m.groupby("order")["accuracy"].mean()

        # Max gap across sequences
        nom_max_gap = nom_seq_acc.max() - nom_seq_acc.min()
        match_max_gap = match_seq_acc.max() - match_seq_acc.min()
        gap_contraction = (nom_max_gap - match_max_gap) / nom_max_gap if nom_max_gap > 0 else 0.0

        # Variance across sequences
        nom_var = nom_seq_acc.var()
        match_var = match_seq_acc.var()
        var_reduction = (nom_var - match_var) / nom_var if nom_var > 0 else 0.0

        # Rank correlation of sequence difficulty
        rho, rho_p = spearmanr(nom_seq_acc.values, match_seq_acc.loc[nom_seq_acc.index].values)

        # Level-by-level gaps
        level_gaps = []
        for lvl in [1, 2, 3, 4, 5]:
            n_sub = nom_m[nom_m["severity"] == lvl]
            m_sub = match_m[match_m["target_idx"] == lvl]
            n_gap = n_sub["accuracy"].max() - n_sub["accuracy"].min() if not n_sub.empty else 0.0
            m_gap = m_sub["accuracy"].max() - m_sub["accuracy"].min() if not m_sub.empty else 0.0
            level_gaps.append({
                "level": lvl,
                "nominal_gap": round(n_gap, 4),
                "matched_gap": round(m_gap, 4),
                "contraction": round((n_gap - m_gap) / n_gap, 4) if n_gap > 0 else 0.0
            })

        print(f"\nModel: {m}")
        print(f"  Nominal Compound Acc: {nom_mean_acc*100:.2f}%, Matched Compound Acc: {match_mean_acc*100:.2f}%")
        print(f"  Nominal Sequence Gap: {nom_max_gap*100:.2f}%, Matched Sequence Gap: {match_max_gap*100:.2f}%")
        print(f"  Gap Contraction: {gap_contraction*100:.1f}%")
        print(f"  Cross-Sequence Variance Reduction: {var_reduction*100:.1f}%")
        print(f"  Sequence Rank Correlation (Spearman rho): {rho:.3f} (p = {rho_p:.4f})")

        comparison_results.append({
            "model": m,
            "nominal_mean_accuracy": round(nom_mean_acc, 4),
            "matched_mean_accuracy": round(match_mean_acc, 4),
            "nominal_max_gap": round(nom_max_gap, 4),
            "matched_max_gap": round(match_max_gap, 4),
            "gap_contraction_pct": round(gap_contraction * 100.0, 2),
            "nominal_variance": round(nom_var, 6),
            "matched_variance": round(match_var, 6),
            "variance_reduction_pct": round(var_reduction * 100.0, 2),
            "rank_correlation_rho": round(rho, 4),
            "rank_correlation_pval": round(rho_p, 4),
            "level_gaps": level_gaps,
        })

    # Error churn comparison if files exist
    churn_comparison = []
    if os.path.exists(nom_churn_path) and os.path.exists(match_churn_path):
        df_nc = pd.read_csv(nom_churn_path)
        df_mc = pd.read_csv(match_churn_path)
        for m in models:
            sub_nc = df_nc[df_nc["model"] == m]
            sub_mc = df_mc[df_mc["model"] == m]
            if not sub_nc.empty and not sub_mc.empty:
                churn_comparison.append({
                    "model": m,
                    "nominal_mean_churn": round(sub_nc["error_churn"].mean(), 4),
                    "matched_mean_churn": round(sub_mc["error_churn"].mean(), 4),
                    "nominal_mean_excess": round(sub_nc["excess_churn"].mean(), 4),
                    "matched_mean_excess": round(sub_mc["excess_churn"].mean(), 4),
                    "nominal_decoupling_ratio": round((sub_nc["excess_churn"] / sub_nc["error_churn"]).mean(), 4),
                    "matched_decoupling_ratio": round((sub_mc["excess_churn"] / sub_mc["error_churn"]).mean(), 4),
                })

    out_payload = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "model_comparisons": comparison_results,
        "churn_comparisons": churn_comparison,
    }

    out_json = os.path.join(matched_dir, "nominal_vs_matched_comparison.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    df_out = pd.DataFrame([{
        "model": r["model"],
        "nominal_mean_acc": r["nominal_mean_accuracy"],
        "matched_mean_acc": r["matched_mean_accuracy"],
        "nominal_gap": r["nominal_max_gap"],
        "matched_gap": r["matched_max_gap"],
        "gap_contraction_pct": r["gap_contraction_pct"],
        "var_reduction_pct": r["variance_reduction_pct"],
        "rank_rho": r["rank_correlation_rho"],
    } for r in comparison_results])
    out_csv = os.path.join(matched_dir, "nominal_vs_matched_summary.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved comparison JSON to {out_json}")
    print(f"Saved comparison CSV to {out_csv}")
    return out_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominal-dir", default=NOMINAL_DIR)
    parser.add_argument("--matched-dir", default=MATCHED_DIR)
    args = parser.parse_args()
    run(args.nominal_dir, args.matched_dir)
