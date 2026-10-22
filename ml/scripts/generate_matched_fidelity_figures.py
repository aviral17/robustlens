"""Generate publication-grade figures for the Matched-Fidelity Benchmark paper.

Figures:
- Figure 1: Objective Quality Matching Validation (Nominal Confound vs. Matched Fidelity Support).
- Figure 2: Residual Sequence Effects & Gap Contraction across Models.
- Figure 3: Factorial Sequence x Target Interaction Profiles.
- Figure 4: Decoupling of Net Accuracy Difference and Error Churn.
- Figure 5: Calibration (ECE & Reliability Diagrams) under Matched Quality.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MATCHED_DIR = os.path.join(REPO_ROOT, "results", "order_effect_study", "matched_fidelity", "outputs")
NOMINAL_DIR = os.path.join(REPO_ROOT, "results", "order_effect_study", "full", "outputs")
REPORTS_FIG_DIR = os.path.join(REPO_ROOT, "reports", "figures")
WEB_FIG_DIR = os.path.join(REPO_ROOT, "apps", "web", "public", "figures")

os.makedirs(REPORTS_FIG_DIR, exist_ok=True)
os.makedirs(WEB_FIG_DIR, exist_ok=True)

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "font.family": "sans-serif",
})

MODEL_PALETTE = {
    "convnext_tiny": "#2563eb",  # Blue
    "swin_tiny": "#059669",      # Emerald Green
    "vit_base": "#7c3aed",       # Purple
    "resnet50": "#d97706",       # Amber/Orange
}

MODEL_NAMES = {
    "convnext_tiny": "ConvNeXt-Tiny",
    "swin_tiny": "Swin-Tiny",
    "vit_base": "ViT-Base/16",
    "resnet50": "ResNet-50",
}

ORDER_PALETTE = {
    "NBJ": "#1f77b4",
    "NJB": "#2ca02c",
    "BNJ": "#ff7f0e",
    "BJN": "#d62728",
    "JNB": "#9467bd",
    "JBN": "#8c564b",
}

ORDERS = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]


def generate_figure_1(matched_dir: str = MATCHED_DIR, nominal_dir: str = NOMINAL_DIR):
    """Figure 1: Objective Distortion Matching Validation."""
    print("Generating Figure 1: Quality Matching Diagnostic...")
    
    # Check predictions files
    nom_csv = os.path.join(nominal_dir, "all_order_predictions.csv.gz")
    match_csv = os.path.join(matched_dir, "all_order_predictions.csv.gz")

    if not os.path.exists(nom_csv) or not os.path.exists(match_csv):
        print(f"Skipping Figure 1: Missing predictions files.")
        return

    # Read subset to plot PSNR / SSIM distributions (first model, non-clean)
    nom_df = pd.read_csv(nom_csv)
    nom_sub = nom_df[(nom_df["model"] == "resnet50") & (nom_df["order"] != "clean")].copy()

    match_df = pd.read_csv(match_csv)
    match_sub = match_df[(match_df["model"] == "resnet50") & (match_df["order"] != "clean")].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A1: Nominal PSNR by Sequence across Severities
    for order in ORDERS:
        order_data = nom_sub[nom_sub["order"] == order]
        mean_psnr = order_data.groupby("severity")["psnr"].mean()
        axes[0, 0].plot(mean_psnr.index, mean_psnr.values, marker="o", label=order, color=ORDER_PALETTE.get(order), linewidth=2)
    axes[0, 0].set_title("A1: Nominal Benchmark - PSNR Confound", fontweight="bold")
    axes[0, 0].set_xlabel("Nominal Severity Level")
    axes[0, 0].set_ylabel("Mean PSNR (dB)")
    axes[0, 0].set_xticks([1, 2, 3, 4, 5])
    axes[0, 0].legend(loc="upper right", framealpha=0.9)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Panel A2: Nominal SSIM by Sequence across Severities
    for order in ORDERS:
        order_data = nom_sub[nom_sub["order"] == order]
        mean_ssim = order_data.groupby("severity")["ssim"].mean()
        axes[0, 1].plot(mean_ssim.index, mean_ssim.values, marker="s", label=order, color=ORDER_PALETTE.get(order), linewidth=2)
    axes[0, 1].set_title("A2: Nominal Benchmark - SSIM Confound", fontweight="bold")
    axes[0, 1].set_xlabel("Nominal Severity Level")
    axes[0, 1].set_ylabel("Mean SSIM")
    axes[0, 1].set_xticks([1, 2, 3, 4, 5])
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Panel B1: Matched Fidelity PSNR by Sequence across Target Levels
    tgt_col = "target_idx" if "target_idx" in match_sub.columns else "severity"
    for order in ORDERS:
        order_data = match_sub[match_sub["order"] == order]
        mean_psnr = order_data.groupby(tgt_col)["psnr"].mean()
        axes[1, 0].plot(mean_psnr.index, mean_psnr.values, marker="o", label=order, color=ORDER_PALETTE.get(order), linewidth=2)
    axes[1, 0].set_title("B1: Matched-Fidelity Benchmark - Calibrated PSNR", fontweight="bold")
    axes[1, 0].set_xlabel("Matched Fidelity Target Level")
    axes[1, 0].set_ylabel("Mean PSNR (dB)")
    axes[1, 0].set_xticks([1, 2, 3, 4, 5])
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Panel B2: Matched Fidelity SSIM by Sequence across Target Levels
    for order in ORDERS:
        order_data = match_sub[match_sub["order"] == order]
        mean_ssim = order_data.groupby(tgt_col)["ssim"].mean()
        axes[1, 1].plot(mean_ssim.index, mean_ssim.values, marker="s", label=order, color=ORDER_PALETTE.get(order), linewidth=2)
    axes[1, 1].set_title("B2: Matched-Fidelity Benchmark - Calibrated SSIM", fontweight="bold")
    axes[1, 1].set_xlabel("Matched Fidelity Target Level")
    axes[1, 1].set_ylabel("Mean SSIM")
    axes[1, 1].set_xticks([1, 2, 3, 4, 5])
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Physical Image Fidelity: Nominal Confounding vs. Matched Common Support", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_rep = os.path.join(REPORTS_FIG_DIR, "fig1_matched_fidelity_validation.png")
    out_web = os.path.join(WEB_FIG_DIR, "fig1_matched_fidelity_validation.png")
    plt.savefig(out_rep, dpi=300, bbox_inches="tight")
    plt.savefig(out_web, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Figure 1 to {out_rep}")


def generate_figure_2(matched_dir: str = MATCHED_DIR, nominal_dir: str = NOMINAL_DIR):
    """Figure 2: Residual Sequence Effects & Gap Contraction."""
    print("Generating Figure 2: Sequence Effects & Gap Contraction...")
    nom_sum_path = os.path.join(nominal_dir, "order_effects_summary.csv")
    match_sum_path = os.path.join(matched_dir, "order_effects_summary.csv")

    if not os.path.exists(nom_sum_path) or not os.path.exists(match_sum_path):
        print("Skipping Figure 2: Summary CSVs not found.")
        return

    df_nom = pd.read_csv(nom_sum_path)
    df_match = pd.read_csv(match_sum_path)

    df_nom_comp = df_nom[df_nom["order"] != "clean"]
    df_match_comp = df_match[df_match["order"] != "clean"]

    models = ["convnext_tiny", "swin_tiny", "vit_base", "resnet50"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for idx, m in enumerate(models):
        ax = axes[idx]
        n_sub = df_nom_comp[df_nom_comp["model"] == m].groupby("order")["accuracy"].mean()
        m_sub = df_match_comp[df_match_comp["model"] == m].groupby("order")["accuracy"].mean()

        x = np.arange(len(ORDERS))
        width = 0.35

        nom_vals = [n_sub.get(o, 0.0) * 100.0 for o in ORDERS]
        match_vals = [m_sub.get(o, 0.0) * 100.0 for o in ORDERS]

        rects1 = ax.bar(x - width/2, nom_vals, width, label="Nominal Severity", color="#94a3b8", edgecolor="black", alpha=0.85)
        rects2 = ax.bar(x + width/2, match_vals, width, label="Matched Fidelity", color=MODEL_PALETTE[m], edgecolor="black", alpha=0.95)

        nom_gap = max(nom_vals) - min(nom_vals)
        match_gap = max(match_vals) - min(match_vals)
        contraction = (nom_gap - match_gap) / nom_gap * 100.0 if nom_gap > 0 else 0.0

        ax.set_title(f"{MODEL_NAMES[m]} (Gap Contraction: {contraction:.1f}%)", fontweight="bold")
        ax.set_ylabel("Compound Accuracy (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(ORDERS, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Annotate gaps
        ax.text(0.03, 0.92, f"Nominal Gap: {nom_gap:.2f}%\nMatched Gap: {match_gap:.2f}%",
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9), fontsize=9)

    fig.suptitle("Robustness under Permutation: Nominal Confounded Gap vs. Matched-Fidelity Gap", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_rep = os.path.join(REPORTS_FIG_DIR, "fig2_sequence_gap_contraction.png")
    out_web = os.path.join(WEB_FIG_DIR, "fig2_sequence_gap_contraction.png")
    plt.savefig(out_rep, dpi=300, bbox_inches="tight")
    plt.savefig(out_web, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Figure 2 to {out_rep}")


def generate_figure_3(matched_dir: str = MATCHED_DIR):
    """Figure 3: Factorial Sequence x Target Interaction Profiles."""
    print("Generating Figure 3: Factorial Interaction Profiles...")
    match_sum_path = os.path.join(matched_dir, "order_effects_summary.csv")
    if not os.path.exists(match_sum_path):
        print("Skipping Figure 3: Summary CSV not found.")
        return

    df_match = pd.read_csv(match_sum_path)
    df_match_comp = df_match[df_match["order"] != "clean"]
    tgt_col = "target_idx" if "target_idx" in df_match.columns else "severity"

    models = ["convnext_tiny", "swin_tiny", "vit_base", "resnet50"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for idx, m in enumerate(models):
        ax = axes[idx]
        sub_m = df_match_comp[df_match_comp["model"] == m]

        for order in ORDERS:
            order_data = sub_m[sub_m["order"] == order]
            acc_by_tgt = order_data.groupby(tgt_col)["accuracy"].mean() * 100.0
            ax.plot(acc_by_tgt.index, acc_by_tgt.values, marker="o", label=order,
                    color=ORDER_PALETTE.get(order), linewidth=2.2, markersize=6)

        ax.set_title(f"{MODEL_NAMES[m]} - Sequence Profiles across Fidelity Targets", fontweight="bold")
        ax.set_xlabel("Matched Fidelity Target Level")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(["T1 (28dB)", "T2 (24.5dB)", "T3 (21.5dB)", "T4 (18.5dB)", "T5 (16dB)"])
        ax.legend(loc="lower left", framealpha=0.9, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Sequence x Target Factorial Interaction: Disentangling Order and Degradation Depth", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_rep = os.path.join(REPORTS_FIG_DIR, "fig3_factorial_interaction_profiles.png")
    out_web = os.path.join(WEB_FIG_DIR, "fig3_factorial_interaction_profiles.png")
    plt.savefig(out_rep, dpi=300, bbox_inches="tight")
    plt.savefig(out_web, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Figure 3 to {out_rep}")


def generate_figure_4(matched_dir: str = MATCHED_DIR):
    """Figure 4: Decoupling of Error Churn and Net Accuracy Difference."""
    print("Generating Figure 4: Churn Decoupling Analysis...")
    churn_path = os.path.join(matched_dir, "error_churn_summary.csv")
    if not os.path.exists(churn_path):
        print("Skipping Figure 4: Churn CSV not found.")
        return

    df_churn = pd.read_csv(churn_path)
    models = ["convnext_tiny", "swin_tiny", "vit_base", "resnet50"]

    fig = plt.figure(figsize=(15, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # Panel A: Net Difference vs Error Churn Scatter
    ax_scatter = fig.add_subplot(gs[0, :])
    for m in models:
        sub = df_churn[df_churn["model"] == m]
        net_col = "net_difference" if "net_difference" in sub.columns else "net_diff"
        net_diff = (sub[net_col].abs() * 100.0).values
        churn = (sub["error_churn"] * 100.0).values
        ax_scatter.scatter(net_diff, churn, color=MODEL_PALETTE[m], label=MODEL_NAMES[m], alpha=0.75, s=60, edgecolors="none")

    # Add identity line
    max_val = max(ax_scatter.get_xlim()[1], ax_scatter.get_ylim()[1])
    ax_scatter.plot([0, max_val], [0, max_val], linestyle="--", color="gray", label="Identity (Zero Excess Churn: G+L = |G-L|)")
    ax_scatter.set_title("A: Decoupling of Net Accuracy Difference and Error Churn at Matched Fidelity", fontweight="bold")
    ax_scatter.set_xlabel("Absolute Net Accuracy Difference |G - L| (%)")
    ax_scatter.set_ylabel("Pairwise Error Churn G + L (%)")
    ax_scatter.legend(loc="upper left", framealpha=0.9)
    ax_scatter.grid(True, linestyle="--", alpha=0.5)

    # Panel B: Mean Error Churn vs Mean Excess Churn by Model
    ax_bar = fig.add_subplot(gs[1, 0])
    mean_churns = [df_churn[df_churn["model"] == m]["error_churn"].mean() * 100.0 for m in models]
    mean_excess = [df_churn[df_churn["model"] == m]["excess_churn"].mean() * 100.0 for m in models]
    x = np.arange(len(models))
    width = 0.35
    ax_bar.bar(x - width/2, mean_churns, width, label="Total Churn (G + L)", color="#3b82f6", edgecolor="black")
    ax_bar.bar(x + width/2, mean_excess, width, label="Excess Churn (G + L - |G-L|)", color="#ef4444", edgecolor="black")
    ax_bar.set_title("B: Total Churn vs. Excess Churn by Architecture", fontweight="bold")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=15, ha="right")
    ax_bar.set_ylabel("Mean Rate (%)")
    ax_bar.legend(loc="upper right")
    ax_bar.grid(True, linestyle="--", alpha=0.5)

    # Panel C: Decoupling Ratio (Excess Churn / Total Churn)
    ax_ratio = fig.add_subplot(gs[1, 1])
    ratios = [(e / c) * 100.0 for e, c in zip(mean_excess, mean_churns)]
    bars = ax_ratio.bar([MODEL_NAMES[m] for m in models], ratios, color=[MODEL_PALETTE[m] for m in models], edgecolor="black")
    ax_ratio.set_title("C: Excess Churn Ratio (Fraction of Churn Unexplained by Net Difference)", fontweight="bold")
    ax_ratio.set_ylabel("Excess Churn Ratio (%)")
    ax_ratio.set_ylim(0, 100)
    for b in bars:
        h = b.get_height()
        ax_ratio.text(b.get_x() + b.get_width()/2., h + 2, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax_ratio.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=15, ha="right")

    fig.suptitle("Individual-Sample Prediction Volatility under Matched Physical Image Fidelity", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_rep = os.path.join(REPORTS_FIG_DIR, "fig4_matched_churn_decoupling.png")
    out_web = os.path.join(WEB_FIG_DIR, "fig4_matched_churn_decoupling.png")
    plt.savefig(out_rep, dpi=300, bbox_inches="tight")
    plt.savefig(out_web, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Figure 4 to {out_rep}")


def generate_figure_5(matched_dir: str = MATCHED_DIR):
    """Figure 5: Calibration & Uncertainty under Matched Quality."""
    print("Generating Figure 5: Calibration Analysis...")
    rep_path = os.path.join(matched_dir, "benchmark_report.json")
    if not os.path.exists(rep_path):
        print("Skipping Figure 5: Benchmark report not found.")
        return

    with open(rep_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    # Plot leaderboard ECE and accuracy
    leaderboard = rep.get("leaderboard", [])
    if not leaderboard:
        print("Skipping Figure 5: Leaderboard empty.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    models = [item["model"] for item in leaderboard]
    model_labels = [item.get("model_name", m) for item, m in zip(leaderboard, models)]
    accs = [item.get("compound_accuracy", 0.0) * 100.0 for item in leaderboard]
    eces = [item.get("mean_compound_ece", 0.0) * 100.0 for item in leaderboard]

    x = np.arange(len(models))
    width = 0.35

    rects1 = ax1.bar(x - width/2, accs, width, label="Compound Accuracy (%)", color="#2563eb", edgecolor="black")
    ax1.set_ylabel("Accuracy (%)", color="#2563eb", fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis="y", labelcolor="#2563eb")

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, eces, width, label="Mean Compound ECE (%)", color="#dc2626", edgecolor="black")
    ax2.set_ylabel("Expected Calibration Error (%)", color="#dc2626", fontweight="bold")
    ax2.set_ylim(0, max(eces) * 1.5 if eces else 15)
    ax2.tick_params(axis="y", labelcolor="#dc2626")

    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, fontweight="bold")
    ax1.set_title("Classifier Reliability & Calibration under Matched Image Fidelity", fontsize=14, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    out_rep = os.path.join(REPORTS_FIG_DIR, "fig5_matched_calibration.png")
    out_web = os.path.join(WEB_FIG_DIR, "fig5_matched_calibration.png")
    plt.savefig(out_rep, dpi=300, bbox_inches="tight")
    plt.savefig(out_web, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Figure 5 to {out_rep}")


def generate_all(matched_dir: str = MATCHED_DIR, nominal_dir: str = NOMINAL_DIR):
    print("=" * 70)
    print("Generating Matched-Fidelity Research Publication Figures")
    print("=" * 70)
    generate_figure_1(matched_dir, nominal_dir)
    generate_figure_2(matched_dir, nominal_dir)
    generate_figure_3(matched_dir)
    generate_figure_4(matched_dir)
    generate_figure_5(matched_dir)
    print("\nAll Matched-Fidelity publication figures generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-dir", default=MATCHED_DIR)
    parser.add_argument("--nominal-dir", default=NOMINAL_DIR)
    args = parser.parse_args()
    generate_all(args.matched_dir, args.nominal_dir)
