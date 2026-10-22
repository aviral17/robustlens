"""Generate publication-grade figures for compound corruption order effects study."""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "reports", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
    "font.family": "sans-serif",
})

ORDER_PALETTE = {
    "NBJ": "#dc2626",  # Red (Noise -> Blur -> JPEG)
    "NJB": "#ea580c",  # Orange (Noise -> JPEG -> Blur)
    "BNJ": "#b45309",  # Brown (Blur -> Noise -> JPEG)
    "BJN": "#0284c7",  # Sky Blue (Blur -> JPEG -> Noise)
    "JNB": "#4f46e5",  # Indigo (JPEG -> Noise -> Blur)
    "JBN": "#059669",  # Emerald (JPEG -> Blur -> Noise)
}

MODEL_PALETTE = {
    "convnext_tiny": "#2563eb",
    "swin_tiny": "#059669",
    "resnet50": "#d97706",
    "vit_base": "#7c3aed",
}

ORDER_NAMES = {
    "NBJ": "Noise -> Blur -> JPEG",
    "NJB": "Noise -> JPEG -> Blur",
    "BNJ": "Blur -> Noise -> JPEG",
    "BJN": "Blur -> JPEG -> Noise",
    "JNB": "JPEG -> Noise -> Blur",
    "JBN": "JPEG -> Blur -> Noise",
}

MODEL_NAMES = {
    "convnext_tiny": "ConvNeXt-Tiny",
    "swin_tiny": "Swin-Tiny",
    "resnet50": "ResNet-50",
    "vit_base": "ViT-Base/16",
}


def plot_figure_06_order_by_arch(summary_df):
    """Figure 6: Accuracy across the 6 orders by architecture."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    
    # Filter compound orders (exclude clean)
    df = summary_df[summary_df["order"] != "clean"]
    models = ["convnext_tiny", "swin_tiny", "vit_base", "resnet50"]
    orders = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]
    
    x = np.arange(len(models))
    width = 0.13
    
    for i, ord_code in enumerate(orders):
        accs = []
        for m in models:
            sub = df[(df["model"] == m) & (df["order"] == ord_code)]
            accs.append(sub["accuracy"].mean() * 100.0)
            
        offset = (i - 2.5) * width
        rects = ax.bar(x + offset, accs, width, label=ord_code, color=ORDER_PALETTE[ord_code], edgecolor="#1e293b", alpha=0.9)
        
    ax.set_ylabel("Mean Top-1 Accuracy (%)", fontweight="bold")
    ax.set_title("Classification Accuracy across 6 Corruption Permutations by Architecture", pad=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES[m] for m in models], fontweight="semibold")
    ax.set_ylim(20, 85)
    ax.legend(title="Permutation Sequence", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "06_order_effects_by_architecture.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


def plot_figure_07_psnr_vs_acc(summary_df):
    """Figure 7: PSNR and SSIM vs Accuracy across orders."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    df = summary_df[summary_df["order"] != "clean"]
    orders = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]

    for ord_code in orders:
        sub = df[df["order"] == ord_code]
        ax1.scatter(sub["mean_psnr"], sub["accuracy"] * 100.0, label=ord_code, color=ORDER_PALETTE[ord_code], s=55, alpha=0.85)
        ax2.scatter(sub["mean_ssim"], sub["accuracy"] * 100.0, label=ord_code, color=ORDER_PALETTE[ord_code], s=55, alpha=0.85)

    ax1.set_xlabel("Mean PSNR (dB)", fontweight="semibold")
    ax1.set_ylabel("Top-1 Accuracy (%)", fontweight="bold")
    ax1.set_title("Fidelity (PSNR) vs. Accuracy by Order", fontweight="bold")
    ax1.legend(title="Order", loc="upper left")

    ax2.set_xlabel("Mean SSIM", fontweight="semibold")
    ax2.set_ylabel("Top-1 Accuracy (%)", fontweight="bold")
    ax2.set_title("Structural Similarity (SSIM) vs. Accuracy by Order", fontweight="bold")
    ax2.legend(title="Order", loc="upper left")

    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "07_psnr_ssim_vs_accuracy_by_order.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


def plot_figure_08_severity_interaction(summary_df):
    """Figure 8: Order accuracy divergence across severities 1 to 5."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True, dpi=300)
    models = ["convnext_tiny", "swin_tiny", "vit_base", "resnet50"]
    orders = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]

    for ax_idx, m in enumerate(models):
        ax = axes[ax_idx]
        sub_m = summary_df[(summary_df["model"] == m) & (summary_df["order"] != "clean")]
        
        for ord_code in orders:
            sub_o = sub_m[sub_m["order"] == ord_code].sort_values("severity")
            sevs = sub_o["severity"].values
            accs = sub_o["accuracy"].values * 100.0
            ax.plot(sevs, accs, marker="o", linewidth=2.0, label=ord_code, color=ORDER_PALETTE[ord_code])

        ax.set_title(MODEL_NAMES[m], fontweight="bold")
        ax.set_xlabel("Compound Severity Level", fontweight="semibold")
        ax.set_xticks([1, 2, 3, 4, 5])
        if ax_idx == 0:
            ax.set_ylabel("Top-1 Accuracy (%)", fontweight="bold")
            ax.legend(title="Order", loc="lower left", fontsize=9)

    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "08_order_severity_interaction.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


def plot_figure_09_order_ece(summary_df):
    """Figure 9: Expected Calibration Error (ECE) across orders."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    df = summary_df[summary_df["order"] != "clean"]
    orders = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]
    models = ["convnext_tiny", "swin_tiny", "vit_base", "resnet50"]

    x = np.arange(len(orders))
    width = 0.18

    for i, m in enumerate(models):
        eces = []
        for ord_code in orders:
            sub = df[(df["model"] == m) & (df["order"] == ord_code)]
            eces.append(sub["ece"].mean() * 100.0)
        offset = (i - 1.5) * width
        ax.bar(x + offset, eces, width, label=MODEL_NAMES[m], color=MODEL_PALETTE[m], edgecolor="#1e293b", alpha=0.9)

    ax.set_ylabel("Expected Calibration Error (ECE, %)", fontweight="bold")
    ax.set_title("Calibration Degradation (ECE) Across Permutation Orders", pad=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(orders, fontweight="semibold")
    ax.set_ylim(0, 24)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(title="Architecture", loc="upper left", frameon=True)

    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "09_order_calibration_ece.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates = [
        os.path.join(repo_root, "results/order_effect_study/full/outputs/order_effects_summary.csv"),
        os.path.join(repo_root, "results/order_effect_study/pilot/outputs/order_effects_summary.csv"),
        os.path.join(repo_root, "results/order_effect_study/local_pilot/order_effects_summary.csv"),
        os.path.join(repo_root, "results/order_effect_study/local_smoke/order_summary.csv"),
    ]
    summary_path = None
    for c in candidates:
        if os.path.exists(c):
            summary_path = c
            break

    output_dir = os.path.join(repo_root, "reports/figures")
    os.makedirs(output_dir, exist_ok=True)

    if summary_path is None:
        print("No order summary found yet to plot. Will generate when data is ready.")
        return

    print(f"Generating figures from {summary_path}...")
    df = pd.read_csv(summary_path)
    plot_figure_06_order_by_arch(df)
    plot_figure_07_psnr_vs_acc(df)
    plot_figure_08_severity_interaction(df)
    plot_figure_09_order_ece(df)
    print("All order figures generated successfully!")


if __name__ == "__main__":
    main()
