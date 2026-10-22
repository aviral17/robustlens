"""Generate publication-grade research figures from the full evaluation artifacts."""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

OUTPUT_DIR = "reports/figures"
RESULTS_DIR = "results/full_eval/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
    "font.family": "sans-serif",
})

# Color palette
PALETTE = {
    "convnext_tiny": "#2563eb",  # Blue
    "swin_tiny": "#059669",      # Emerald
    "resnet50": "#d97706",       # Amber
    "vit_base": "#7c3aed",       # Violet
}
DISPLAY_NAMES = {
    "convnext_tiny": "ConvNeXt-Tiny",
    "swin_tiny": "Swin-Tiny",
    "resnet50": "ResNet-50",
    "vit_base": "ViT-Base/16"
}

# 1. Figure 1: Clean vs Shifted Accuracy
def plot_figure_1(lb_df):
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    models = lb_df["model"].tolist()
    labels = [DISPLAY_NAMES.get(m, m) for m in models]
    x = np.arange(len(models))
    width = 0.35

    clean_accs = lb_df["clean_accuracy"].values * 100
    shifted_accs = lb_df["mean_shifted_accuracy"].values * 100

    rects1 = ax.bar(x - width/2, clean_accs, width, label="Clean Baseline", color="#94a3b8", edgecolor="#475569")
    rects2 = ax.bar(x + width/2, shifted_accs, width, label="Shifted (Mean across 15 Shifts)", 
                    color=[PALETTE[m] for m in models], edgecolor="#1e293b")

    ax.set_ylabel("Top-1 Accuracy (%)", fontweight="bold")
    ax.set_title("Architecture Accuracy: Clean Baseline vs. Distribution Shift (N=3,925)", pad=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="semibold")
    ax.set_ylim(50, 100)
    ax.legend(frameon=True, loc="upper right")

    # Add accuracy drops
    for i, (m, c, s) in enumerate(zip(models, clean_accs, shifted_accs)):
        drop = c - s
        ax.annotate(f"-{drop:.1f}%",
                    xy=(x[i] + width/2, s),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold", color="#dc2626")

    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "01_clean_vs_shifted_accuracy.png")
    plt.savefig(p)
    plt.close()
    print(f"Saved: {p}")

# 2. Figure 2: Corruption Severity Degradation Curves
def plot_figure_2(bk_df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, dpi=300)
    corruptions = ["gaussian_noise", "defocus_blur", "jpeg_compression"]
    titles = ["Gaussian Noise", "Defocus Blur", "JPEG Compression"]

    for ax_idx, (corr, title) in enumerate(zip(corruptions, titles)):
        ax = axes[ax_idx]
        corr_data = bk_df[bk_df["corruption"] == corr]
        
        for model in ["convnext_tiny", "swin_tiny", "resnet50", "vit_base"]:
            m_data = corr_data[corr_data["model"] == model].sort_values("severity")
            sevs = [0] + m_data["severity"].tolist()
            # Clean baseline
            clean_acc = bk_df[(bk_df["model"] == model) & (bk_df["corruption"] == "clean")]["accuracy"].iloc[0] * 100
            accs = [clean_acc] + (m_data["accuracy"] * 100).tolist()
            
            ax.plot(sevs, accs, marker="o", linewidth=2.2, label=DISPLAY_NAMES[model], color=PALETTE[model])
            
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Severity Level (0 = Clean)", fontweight="semibold")
        if ax_idx == 0:
            ax.set_ylabel("Top-1 Accuracy (%)", fontweight="bold")
            ax.legend(frameon=True, loc="lower left")
        ax.set_ylim(30, 95)
        ax.set_xticks(range(6))
        ax.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("Robustness Degradation Profiles Across Perturbation Severities (N=3,925)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "02_corruption_severity_degradation.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")

# 3. Figure 3: Per-Corruption Family Breakdown
def plot_figure_3(bk_df):
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    corruptions = ["gaussian_noise", "defocus_blur", "jpeg_compression"]
    labels = ["Gaussian Noise", "Defocus Blur", "JPEG Compression"]
    models = ["convnext_tiny", "swin_tiny", "resnet50", "vit_base"]
    x = np.arange(len(corruptions))
    width = 0.2

    for i, model in enumerate(models):
        mean_accs = []
        for corr in corruptions:
            sub = bk_df[(bk_df["model"] == model) & (bk_df["corruption"] == corr)]
            mean_accs.append(sub["accuracy"].mean() * 100)
        ax.bar(x + (i - 1.5)*width, mean_accs, width, label=DISPLAY_NAMES[model], color=PALETTE[model], edgecolor="#334155")

    ax.set_ylabel("Mean Top-1 Accuracy (%)", fontweight="bold")
    ax.set_title("Performance Breakdown by Corruption Family (Averaged Over Severities 1-5)", pad=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="semibold")
    ax.set_ylim(40, 90)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "03_per_corruption_breakdown.png")
    plt.savefig(p)
    plt.close()
    print(f"Saved: {p}")

# 4. Figure 4: Calibration Error Under Shift
def plot_figure_4(lb_df, bk_df):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    models = ["convnext_tiny", "swin_tiny", "resnet50", "vit_base"]
    x = np.arange(len(models))
    width = 0.35

    clean_eces = []
    shifted_eces = []
    for m in models:
        c_ece = bk_df[(bk_df["model"] == m) & (bk_df["corruption"] == "clean")]["ece"].iloc[0] * 100
        s_ece = bk_df[(bk_df["model"] == m) & (bk_df["corruption"] != "clean")]["ece"].mean() * 100
        clean_eces.append(c_ece)
        shifted_eces.append(s_ece)

    rects1 = ax.bar(x - width/2, clean_eces, width, label="Clean Baseline ECE", color="#94a3b8", edgecolor="#475569")
    rects2 = ax.bar(x + width/2, shifted_eces, width, label="Shifted Mean ECE", 
                    color=[PALETTE[m] for m in models], edgecolor="#1e293b")

    ax.set_ylabel("Expected Calibration Error (%)", fontweight="bold")
    ax.set_title("Calibration Degradation Under Distribution Shift (Lower is Better)", pad=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in models], fontweight="semibold")
    ax.set_ylim(0, 26)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "04_calibration_ece_shift.png")
    plt.savefig(p)
    plt.close()
    print(f"Saved: {p}")

# 5. Figure 5: Pairwise Clustered Bootstrap 95% Confidence Intervals
def plot_figure_5(pw_df):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    pw = pw_df.copy().reset_index(drop=True)
    y_pos = np.arange(len(pw))

    diffs = pw["shifted_accuracy_difference"].values * 100
    ci_low = pw["paired_ci_95_lower"].values * 100
    ci_high = pw["paired_ci_95_upper"].values * 100
    xerr = [diffs - ci_low, ci_high - diffs]

    labels = [f"{row['model_1_name']} vs. {row['model_2_name']}" for _, row in pw.iterrows()]

    ax.axvline(0, color="#dc2626", linestyle="--", linewidth=1.5, label="Null Hypothesis (Δ = 0)")
    ax.errorbar(diffs, y_pos, xerr=xerr, fmt="o", color="#2563eb", ecolor="#1e3a8a", elinewidth=2.5,
                capsize=5, capthick=2, markersize=8)

    for i, (d, cl, ch) in enumerate(zip(diffs, ci_low, ci_high)):
        ax.annotate(f"{d:+.2f}%  [{cl:+.2f}%, {ch:+.2f}%]",
                    xy=(ch, y_pos[i]),
                    xytext=(10, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontweight="semibold")
    ax.set_xlabel("Shifted Top-1 Accuracy Difference Δ (%) [95% Clustered CI]", fontweight="bold")
    ax.set_title("Pairwise Clustered Paired Bootstrap Comparisons (N=3,925 Image Clusters)", pad=14, fontweight="bold")
    ax.set_xlim(-4, 15)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(frameon=True, loc="lower left")
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "05_pairwise_clustered_confidence_intervals.png")
    plt.savefig(p)
    plt.close()
    print(f"Saved: {p}")

def main():
    lb = pd.read_csv(os.path.join(RESULTS_DIR, "leaderboard_summary.csv"))
    bk = pd.read_csv(os.path.join(RESULTS_DIR, "detailed_breakdown.csv"))
    pw = pd.read_csv(os.path.join(RESULTS_DIR, "pairwise_significance.csv"))

    plot_figure_1(lb)
    plot_figure_2(bk)
    plot_figure_3(bk)
    plot_figure_4(lb, bk)
    plot_figure_5(pw)
    print("All 5 research figures generated successfully.")

if __name__ == "__main__":
    main()
