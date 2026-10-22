#!/usr/bin/env python3
"""Generate canonical release figures for RobustLens.

Produces:
- reports/final_research_package/figures/figure1_order_accuracy.png
- reports/final_research_package/figures/figure2_accuracy_fidelity.png
- reports/final_research_package/figures/figure3_range_contraction.png
- reports/final_research_package/figures/figure4_prediction_churn.png

All figures satisfy:
1. Figure 1: Comparable y-axes (25% to 75%), consistent ticks, categorical order representation (no misleading connecting lines).
2. Figure 2: Non-causal title "Accuracy Broadly Tracks Image Fidelity Across Nominal Orders".
3. Figure 3: Uses "PSNR/SSIM-Targeted Regime", legend placed cleanly clear of data marks.
4. Figure 4: Legend and value labels strictly non-overlapping, showing 5-state partition and churn decoupling.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPO_ROOT / "reports/final_research_package/analysis"
FIG_DIR = REPO_ROOT / "reports/final_research_package/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "font.family": "sans-serif",
})

ORDER_COLORS = {
    "NBJ": "#1e40af",  # Deep Blue
    "NJB": "#0284c7",  # Sky Blue
    "BNJ": "#0d9488",  # Teal
    "BJN": "#b45309",  # Amber/Bronze
    "JNB": "#047857",  # Emerald
    "JBN": "#be123c",  # Rose/Red
}

MODEL_DISPLAY = {
    "resnet50": "ResNet-50",
    "convnext_tiny": "ConvNeXt-Tiny",
    "vit_base": "ViT-Base/16",
    "swin_tiny": "Swin-Tiny",
    "Overall": "Across-Models Mean",
}


def make_figure1_order_accuracy():
    """Figure 1: Classification Accuracy Across 6 Corruption Permutations by Architecture.
    
    Uses comparable y-axis ranges (25% to 75%), consistent ticks, and categorical grouped bars.
    """
    df_cells = pd.read_csv(ANALYSIS_DIR / "order_cell_means.csv")
    models = ["resnet50", "convnext_tiny", "vit_base", "swin_tiny", "Overall"]
    orders = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]
    
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=300)
    
    x = np.arange(len(models))
    width = 0.13
    
    for i, ord_name in enumerate(orders):
        accs = []
        errs = []
        for m in models:
            sub = df_cells[(df_cells["regime"] == "Nominal-Order") & (df_cells["model"] == m) & (df_cells["order"] == ord_name)]
            acc = sub["accuracy"].values[0] * 100.0
            se = sub["accuracy_se"].values[0] * 100.0 if "accuracy_se" in sub else 0.0
            accs.append(acc)
            errs.append(se * 1.96)
            
        offset = (i - 2.5) * width
        rects = ax.bar(
            x + offset,
            accs,
            width,
            yerr=errs,
            capsize=2.5,
            label=f"{ord_name}",
            color=ORDER_COLORS[ord_name],
            edgecolor="#1e293b",
            linewidth=0.6,
            alpha=0.92
        )
        
    ax.set_ylabel("Mean Top-1 Accuracy (%, 95% CI)", fontweight="bold")
    ax.set_title("Figure 1: Top-1 Classification Accuracy Across Six Nominal Corruption Permutations", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in models], fontweight="semibold")
    ax.set_ylim(25, 75)
    ax.set_yticks(np.arange(25, 80, 5))
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    
    ax.legend(
        title="Permutation Sequence",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        framealpha=0.95
    )
    
    plt.tight_layout()
    out_path = FIG_DIR / "figure1_order_accuracy.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_path}")


def make_figure2_accuracy_fidelity():
    """Figure 2: Accuracy Broadly Tracks Image Fidelity Across Nominal Orders.
    
    Avoids causal overstatement while demonstrating how low-level PSNR/SSIM correlate with accuracy.
    """
    summary_path = REPO_ROOT / "results/order_effect_study/full/outputs/order_effects_summary.csv"
    df = pd.read_csv(summary_path)
    df = df[df["order"] != "clean"]
    orders = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0), dpi=300)
    
    marker_map = {"resnet50": "o", "convnext_tiny": "s", "vit_base": "^", "swin_tiny": "D"}
    
    for ord_code in orders:
        sub = df[df["order"] == ord_code]
        for m, marker in marker_map.items():
            sub_m = sub[sub["model"] == m]
            ax1.scatter(
                sub_m["mean_psnr"],
                sub_m["accuracy"] * 100.0,
                color=ORDER_COLORS[ord_code],
                marker=marker,
                s=48,
                alpha=0.85,
                edgecolors="#1e293b",
                linewidth=0.5
            )
            ax2.scatter(
                sub_m["mean_ssim"],
                sub_m["accuracy"] * 100.0,
                color=ORDER_COLORS[ord_code],
                marker=marker,
                s=48,
                alpha=0.85,
                edgecolors="#1e293b",
                linewidth=0.5
            )
            
    # Dummy handles for legend
    import matplotlib.lines as mlines
    order_handles = [mlines.Line2D([], [], color=ORDER_COLORS[o], marker='o', linestyle='', markersize=7, label=o) for o in orders]
    model_handles = [mlines.Line2D([], [], color='#475569', marker=marker_map[m], linestyle='', markersize=7, label=MODEL_DISPLAY[m]) for m in ["resnet50", "convnext_tiny", "vit_base", "swin_tiny"]]
    
    ax1.set_xlabel("Mean Peak Signal-to-Noise Ratio (PSNR, dB)", fontweight="semibold")
    ax1.set_ylabel("Top-1 Accuracy (%)", fontweight="bold")
    ax1.set_title("A. Top-1 Accuracy vs. Mean PSNR", fontweight="bold")
    ax1.set_ylim(20, 85)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    ax2.set_xlabel("Mean Structural Similarity Index (SSIM)", fontweight="semibold")
    ax2.set_ylabel("Top-1 Accuracy (%)", fontweight="bold")
    ax2.set_title("B. Top-1 Accuracy vs. Mean SSIM", fontweight="bold")
    ax2.set_ylim(20, 85)
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    # Place legends cleanly
    ax1.legend(handles=order_handles, title="Sequence", loc="upper left", frameon=True, fontsize=8, ncol=2)
    ax2.legend(handles=model_handles, title="Architecture", loc="upper left", frameon=True, fontsize=8)
    
    fig.suptitle("Figure 2: Accuracy Broadly Tracks Image Fidelity Across Nominal Orders", fontweight="bold", fontsize=13, y=0.98)
    plt.tight_layout()
    out_path = FIG_DIR / "figure2_accuracy_fidelity.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_path}")


def make_figure3_range_contraction():
    """Figure 3: Sequence Performance Gap Contraction under PSNR/SSIM-Targeted Calibration.
    
    Uses exact regime name: "PSNR/SSIM-Targeted Regime". Legend clear of data marks.
    """
    df_gaps = pd.read_csv(ANALYSIS_DIR / "nominal_vs_matched_gap_contraction.csv")
    models = ["resnet50", "convnext_tiny", "vit_base", "swin_tiny", "Overall"]
    
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=300)
    
    x = np.arange(len(models))
    width = 0.32
    
    nom_gaps = []
    match_gaps = []
    contractions = []
    
    for m in models:
        r = df_gaps[df_gaps["model"] == m].iloc[0]
        nom_gaps.append(r["nominal_acc_gap"] * 100.0)
        match_gaps.append(r["matched_acc_gap"] * 100.0)
        contractions.append(r["acc_gap_contraction_pct"])
        
    rects1 = ax.bar(x - width/2, nom_gaps, width, label="Nominal-Order Regime (Confounded)", color="#dc2626", alpha=0.88, edgecolor="#1e293b", linewidth=0.6)
    rects2 = ax.bar(x + width/2, match_gaps, width, label="PSNR/SSIM-Targeted Regime (Calibrated)", color="#2563eb", alpha=0.88, edgecolor="#1e293b", linewidth=0.6)
    
    # Annotate contraction percentages clearly above bars without overlap
    for i in range(len(models)):
        val_nom = nom_gaps[i]
        val_match = match_gaps[i]
        c_pct = contractions[i]
        ax.annotate(
            f"−{c_pct:.1f}%\nContraction",
            xy=(x[i], max(val_nom, val_match) + 0.8),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=8.5,
            color="#0f172a"
        )
        
    ax.set_ylabel("Cross-Sequence Max-Min Accuracy Gap (% points)", fontweight="bold")
    ax.set_title("Figure 3: Contraction of Sequence Accuracy Gaps in PSNR/SSIM-Targeted Regime", fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in models], fontweight="semibold")
    ax.set_ylim(0, 27)
    ax.set_yticks(np.arange(0, 30, 5))
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    
    # Place legend in upper right whitespace, strictly clear of bars
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=9.5)
    
    plt.tight_layout()
    out_path = FIG_DIR / "figure3_range_contraction.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_path}")


def make_figure4_prediction_churn():
    """Figure 4: Decoupling of Net Accuracy Differences and Prediction Churn.
    
    Non-overlapping annotations and legend showing 5-state prediction space decomposition.
    """
    df_trans = pd.read_csv(ANALYSIS_DIR / "prediction_instability_transitions.csv")
    models = ["resnet50", "convnext_tiny", "vit_base", "swin_tiny", "Overall"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300, sharey=True)
    
    # Colors for 5 states
    c_e11 = "#10b981"      # Green (Both Correct)
    c_e01 = "#3b82f6"      # Blue (Gain)
    c_e10 = "#ef4444"      # Red (Loss)
    c_e00_same = "#94a3b8" # Slate Gray (Same Incorrect)
    c_e00_diff = "#f59e0b" # Amber (Error Migration / Different Incorrect)
    
    for panel_idx, (regime, title) in enumerate([
        ("Nominal-Order", "A. Nominal-Order Regime"),
        ("Approximate-Fidelity", "B. PSNR/SSIM-Targeted Regime"),
    ]):
        ax = axes_list[panel_idx] if 'axes_list' in locals() else (ax1 if panel_idx == 0 else ax2)
        sub = df_trans[df_trans["regime"] == regime].set_index("model")
        
        y_pos = np.arange(len(models))
        
        e11 = [sub.loc[m, "e11_both_correct"] * 100 for m in models]
        e01 = [sub.loc[m, "e01_gain"] * 100 for m in models]
        e10 = [sub.loc[m, "e10_loss"] * 100 for m in models]
        e00_same = [sub.loc[m, "e00_same_wrong"] * 100 for m in models]
        e00_diff = [sub.loc[m, "e00_diff_wrong"] * 100 for m in models]
        
        p1 = ax.barh(y_pos, e11, label="Both Correct (E11)", color=c_e11, edgecolor="white", height=0.62)
        p2 = ax.barh(y_pos, e01, left=e11, label="Gain: 0 → 1 (E01)", color=c_e01, edgecolor="white", height=0.62)
        p3 = ax.barh(y_pos, e10, left=np.array(e11)+np.array(e01), label="Loss: 1 → 0 (E10)", color=c_e10, edgecolor="white", height=0.62)
        p4 = ax.barh(y_pos, e00_same, left=np.array(e11)+np.array(e01)+np.array(e10), label="Same Incorrect Class (E00_same)", color=c_e00_same, edgecolor="white", height=0.62)
        p5 = ax.barh(y_pos, e00_diff, left=np.array(e11)+np.array(e01)+np.array(e10)+np.array(e00_same), label="Error Migration: Different Incorrect (E00_diff)", color=c_e00_diff, edgecolor="white", height=0.62)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([MODEL_DISPLAY[m] for m in models], fontweight="semibold")
        ax.set_xlabel("Percentage of Evaluated Image Pairs (%)", fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        
        # Add text callouts for Predicted-Label Churn vs Correctness Churn
        for idx_m, m in enumerate(models):
            lbl_churn = sub.loc[m, "label_churn"] * 100
            corr_churn = sub.loc[m, "correctness_churn"] * 100
            # annotate inside or next to bar cleanly
            ax.text(101.5, idx_m, f"Label Churn: {lbl_churn:.1f}%\nCorr Churn: {corr_churn:.1f}%", va="center", fontsize=7.5, color="#0f172a")

    ax1.set_xlim(0, 125)
    ax2.set_xlim(0, 125)
    
    # Shared single legend at the top of the figure to prevent overlapping
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=True, fontsize=8.5)
    
    fig.suptitle("Figure 4: 5-State Prediction Space Partition & Decoupling of Label Churn from Correctness Churn", fontweight="bold", fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = FIG_DIR / "figure4_prediction_churn.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_path}")


def main():
    print("Generating canonical release figures...")
    make_figure1_order_accuracy()
    make_figure2_accuracy_fidelity()
    make_figure3_range_contraction()
    make_figure4_prediction_churn()
    print("All 4 canonical figures generated successfully!")


if __name__ == "__main__":
    main()
