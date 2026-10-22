"""Generate publication-grade Figure 10: Order-Dependent Error Churn Analysis."""

import os
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORTS_FIG_DIR = os.path.join(REPO_ROOT, "reports", "figures")
WEB_FIG_DIR = os.path.join(REPO_ROOT, "apps", "web", "public", "figures")
os.makedirs(REPORTS_FIG_DIR, exist_ok=True)
os.makedirs(WEB_FIG_DIR, exist_ok=True)

CHURN_SUMMARY_CSV = os.path.join(REPO_ROOT, "results", "order_effect_study", "full", "outputs", "error_churn_summary.csv")
CHURN_HIGH_EXCESS_CSV = os.path.join(REPO_ROOT, "results", "order_effect_study", "full", "outputs", "error_churn_high_excess_cases.csv")

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

ORDERS = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]


def build_symmetric_matrix(df_model, metric="error_churn"):
    """Build a 6x6 symmetric matrix of average pairwise metric values."""
    mat = np.zeros((6, 6))
    for i, o_a in enumerate(ORDERS):
        for j, o_b in enumerate(ORDERS):
            if i == j:
                mat[i, j] = 0.0
            else:
                match = df_model[
                    ((df_model["order_a"] == o_a) & (df_model["order_b"] == o_b)) |
                    ((df_model["order_a"] == o_b) & (df_model["order_b"] == o_a))
                ]
                if not match.empty:
                    mat[i, j] = match[metric].mean() * 100.0  # percentage
    return mat


def generate_figure_10():
    df = pd.read_csv(CHURN_SUMMARY_CSV)
    df_high = pd.read_csv(CHURN_HIGH_EXCESS_CSV)

    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.15], hspace=0.32, wspace=0.22)

    # ==========================================
    # Panel A: 2x2 sub-grid of Heatmaps (Top Left & Top Right combined or Top row)
    # ==========================================
    # We will use gs[0, :] for 4 side-by-side heatmaps or a 1x4 subgrid in the top row
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0, :], wspace=0.28)
    
    models = ["convnext_tiny", "swin_tiny", "vit_base", "resnet50"]
    vmax = 28.0  # common scale for percentage churn

    for idx, model_id in enumerate(models):
        ax_heat = fig.add_subplot(gs_top[0, idx])
        df_m = df[df["model"] == model_id]
        mat = build_symmetric_matrix(df_m, "error_churn")
        
        im = ax_heat.imshow(mat, cmap="YlOrRd", vmin=0, vmax=vmax, origin="upper")
        ax_heat.set_title(f"{MODEL_NAMES[model_id]}\nMean Churn: {mat[mat > 0].mean():.1f}%", fontsize=11, fontweight="bold", pad=8)
        ax_heat.set_xticks(range(6))
        ax_heat.set_yticks(range(6))
        ax_heat.set_xticklabels(ORDERS, fontsize=9, rotation=45)
        ax_heat.set_yticklabels(ORDERS if idx == 0 else [], fontsize=9)
        if idx == 0:
            ax_heat.set_ylabel("Order Permutation", fontsize=11)
        ax_heat.grid(False)

        # Annotate cells
        for r in range(6):
            for c in range(6):
                if r == c:
                    ax_heat.text(c, r, "-", ha="center", va="center", color="gray", fontsize=8)
                else:
                    val = mat[r, c]
                    color = "white" if val > 16.0 else "black"
                    ax_heat.text(c, r, f"{val:.1f}", ha="center", va="center", color=color, fontsize=8, fontweight="medium")

    # Add colorbar for heatmaps on the right of top row
    cbar_ax = fig.add_axes([0.92, 0.58, 0.012, 0.30])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Mean Error Churn Rate (%)", fontsize=10)

    # Title label for Panel A
    fig.text(0.08, 0.94, "(A) Pairwise Error Churn Rate Matrices by Architecture (Averaged across Severities 1–5)",
             fontsize=13, fontweight="bold")

    # ==========================================
    # Panel B: Decoupling Scatter Plot (Bottom Left)
    # ==========================================
    ax_scatter = fig.add_subplot(gs[1, 0])
    
    # Plot theoretical minimum bound: Churn >= |Delta_net|
    x_line = np.linspace(0, 42, 100)
    ax_scatter.plot(x_line, x_line, linestyle="--", color="#64748b", linewidth=1.5,
                    label=r"Lower Bound: $\mathrm{Churn} = |\Delta_{\mathrm{net}}|$ ($\mathrm{Excess} = 0$)")

    # Shaded decoupled region: |Delta_net| <= 2% and Excess Churn >= 5%
    # Note: Excess = Churn - |Delta| >= 5% => Churn >= |Delta| + 5%
    # So for x in [0, 2], Churn ranges from x + 5 to 15%
    rect = Rectangle((0, 5), 2.0, 10.0, linewidth=1.5, edgecolor="#dc2626", facecolor="#fee2e2",
                     alpha=0.45, linestyle=":", zorder=1, label=r"Decoupled Zone ($|\Delta| \leq 2\%$, Excess $\geq 5\%$)")
    ax_scatter.add_patch(rect)

    # Plot the 300 evaluations
    for model_id in models:
        df_sub = df[df["model"] == model_id]
        ax_scatter.scatter(
            df_sub["net_accuracy_diff"].abs() * 100.0,
            df_sub["error_churn"] * 100.0,
            c=MODEL_PALETTE[model_id],
            label=f"{MODEL_NAMES[model_id]} (n=75)",
            alpha=0.75,
            s=42,
            edgecolors="none",
            zorder=3
        )

    # Key callout annotations
    # 1. ViT-Base Sev 1 NBJ vs BJN: Net -0.18%, Churn 10.73%
    ax_scatter.annotate(
        "ViT-Base (Sev 1)\n$\\Delta = -0.18\\%, \\mathrm{Churn} = 10.73\\%$\n($p=0.77$, 421 flips)",
        xy=(0.18, 10.73), xytext=(3.5, 13.5),
        arrowprops=dict(facecolor="#7c3aed", edgecolor="#7c3aed", arrowstyle="->", lw=1.2),
        fontsize=8.5, fontweight="bold", color="#581c87",
        bbox=dict(boxstyle="round,pad=0.25", fc="#faf5ff", ec="#c084fc", lw=1)
    )

    # 2. ConvNeXt-Tiny Sev 4 NBJ vs JNB: Net +0.20%, Churn 10.70%
    ax_scatter.annotate(
        "ConvNeXt (Sev 4)\n$\\Delta = +0.20\\%, \\mathrm{Churn} = 10.70\\%$\n($p=0.73$, 420 flips)",
        xy=(0.20, 10.70), xytext=(4.0, 7.5),
        arrowprops=dict(facecolor="#2563eb", edgecolor="#2563eb", arrowstyle="->", lw=1.2),
        fontsize=8.5, fontweight="bold", color="#1e3a8a",
        bbox=dict(boxstyle="round,pad=0.25", fc="#eff6ff", ec="#93c5fd", lw=1)
    )

    # 3. ViT-Base Sev 1 NJB vs JBN: Exact zero difference! Delta = 0.00%, Churn = 9.99%
    ax_scatter.annotate(
        "ViT-Base (Sev 1, NJB vs JBN)\n$\\Delta = 0.00\\%, \\mathrm{Churn} = 9.99\\%$\n(Exact 0 net diff; 392 flips!)",
        xy=(0.0, 9.99), xytext=(2.5, 2.5),
        arrowprops=dict(facecolor="#dc2626", edgecolor="#dc2626", arrowstyle="->", lw=1.2),
        fontsize=8.5, fontweight="bold", color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.25", fc="#fef2f2", ec="#fca5a5", lw=1)
    )

    # 4. ResNet-50 Sev 3 BNJ vs BJN: Peak Excess Churn 14.22%
    ax_scatter.annotate(
        "ResNet-50 (Sev 3, BNJ vs BJN)\n$\\Delta = -4.15\\%, \\mathrm{Churn} = 22.52\\%$\n(Peak Excess: 14.22%, $G=9.2\\%, L=13.3\\%$)",
        xy=(4.15, 22.52), xytext=(8.0, 27.0),
        arrowprops=dict(facecolor="#d97706", edgecolor="#d97706", arrowstyle="->", lw=1.2),
        fontsize=8.5, fontweight="bold", color="#78350f",
        bbox=dict(boxstyle="round,pad=0.25", fc="#fffbeb", ec="#fcd34d", lw=1)
    )

    ax_scatter.set_title(r"(B) Decoupling of Error Churn from Net Accuracy Difference" + "\n" + r"Pearson $r = 0.965$ (Global); 39 Pairs Decoupled ($|\Delta| \leq 2\%$, Excess $\geq 5\%$)",
                         fontsize=11.5, fontweight="bold", pad=8)
    ax_scatter.set_xlabel("Absolute Net Accuracy Difference $|\\Delta_{\\mathrm{net}}| = |G - L|$ (%)", fontsize=11)
    ax_scatter.set_ylabel("Error Churn Rate $\\mathrm{Churn} = G + L$ (%)", fontsize=11)
    ax_scatter.set_xlim(-0.8, 42)
    ax_scatter.set_ylim(-0.8, 44)
    ax_scatter.legend(loc="upper left", frameon=True, fontsize=8.5)

    # ==========================================
    # Panel C: Excess Churn & Rates vs Severity (Bottom Right)
    # ==========================================
    ax_sev = fig.add_subplot(gs[1, 1])

    severities = [1, 2, 3, 4, 5]
    bar_width = 0.18
    x_base = np.arange(len(severities))

    # For each model, plot mean excess churn across severity levels
    for idx, model_id in enumerate(models):
        df_m = df[df["model"] == model_id]
        excess_means = [df_m[df_m["severity"] == s]["excess_churn"].mean() * 100.0 for s in severities]
        excess_stds = [df_m[df_m["severity"] == s]["excess_churn"].std() * 100.0 for s in severities]
        
        pos = x_base + (idx - 1.5) * bar_width
        ax_sev.bar(pos, excess_means, width=bar_width, color=MODEL_PALETTE[model_id],
                   label=f"{MODEL_NAMES[model_id]}", alpha=0.85, edgecolor="white", yerr=excess_stds, capsize=3)

    # Add overlay line for overall mean churn across severities
    overall_churn_means = [df[df["severity"] == s]["error_churn"].mean() * 100.0 for s in severities]
    overall_net_means = [df[df["severity"] == s]["net_accuracy_diff"].abs().mean() * 100.0 for s in severities]
    
    ax_sev2 = ax_sev.twinx()
    l1 = ax_sev2.plot(x_base, overall_churn_means, color="#0f172a", marker="o", linewidth=2.0,
                      label="Mean Total Churn (%)", linestyle="-")
    l2 = ax_sev2.plot(x_base, overall_net_means, color="#dc2626", marker="s", linewidth=2.0,
                      label="Mean $|\\Delta_{\\mathrm{net}}|$ (%)", linestyle=":")
    ax_sev2.set_ylabel("Overall Across Models (%)", fontsize=11, color="#0f172a")
    ax_sev2.grid(False)
    ax_sev2.set_ylim(0, 32)

    ax_sev.set_title("(C) Order-Dependent Excess Churn Dynamics Across Severities\nExcess Churn $2\\min(G, L)$ Peaks at Intermediate Severities ($S=2,3$)",
                     fontsize=11.5, fontweight="bold", pad=8)
    ax_sev.set_xlabel("Compound Corruption Severity Level", fontsize=11)
    ax_sev.set_ylabel("Excess Churn $2\\min(G, L)$ (%)", fontsize=11)
    ax_sev.set_xticks(x_base)
    ax_sev.set_xticklabels([f"Sev {s}" for s in severities], fontsize=10)
    ax_sev.set_ylim(0, 16)
    
    # Combined legend for panel C
    h1, l_1 = ax_sev.get_legend_handles_labels()
    h2, l_2 = ax_sev2.get_legend_handles_labels()
    ax_sev.legend(h1 + h2, l_1 + l_2, loc="upper left", frameon=True, fontsize=8.5, ncol=2)


    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.07, right=0.90, hspace=0.38, wspace=0.25)
    
    out_path_reports = os.path.join(REPORTS_FIG_DIR, "10_error_churn_analysis.png")
    out_path_web = os.path.join(WEB_FIG_DIR, "10_error_churn_analysis.png")
    
    plt.savefig(out_path_reports, dpi=300)
    plt.close()
    print(f"Saved Figure 10 to: {out_path_reports}")

    shutil.copyfile(out_path_reports, out_path_web)
    print(f"Copied Figure 10 to: {out_path_web}")


if __name__ == "__main__":
    generate_figure_10()
