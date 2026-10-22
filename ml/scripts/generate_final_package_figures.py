#!/usr/bin/env python3
"""Generate publication-quality figures for the final research paper.

Outputs to reports/final_research_package/figures/:
- fig1_calibration_residuals_and_tolerances.png
- fig2_nominal_vs_matched_gap_contraction.png
- fig3_saturated_decomposition_coefficients.png
- fig4_paired_error_churn_decomposition.png
- fig5_calibration_ece_and_confidence.png
"""

from pathlib import Path
import json
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
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
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


def make_fig1_calibration_residuals():
    """Figure 1: Objective Quality Calibration Diagnostics and Tolerance Violations."""
    res_csv = REPO_ROOT / "results/order_effect_study/matched_fidelity/matching_residuals.csv"
    df_res = pd.read_csv(res_csv)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # 1. PSNR Residuals
    ax1 = axes[0]
    for order in ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]:
        sub = df_res[df_res["order"] == order]
        ax1.plot(
            sub["target_idx"],
            sub["residual_psnr"],
            marker="o",
            linewidth=1.8,
            markersize=6,
            label=order,
            color=ORDER_COLORS.get(order, "#333"),
        )
    ax1.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
    ax1.axhline(0.75, color="#dc2626", linestyle="--", linewidth=1.2, label="Tolerance (±0.75 dB)")
    ax1.axhline(-0.75, color="#dc2626", linestyle="--", linewidth=1.2)
    ax1.fill_between([0.8, 5.2], -0.75, 0.75, color="#fee2e2", alpha=0.35)
    ax1.set_xlabel("Target Distortion Bin (1: Mild → 5: Severe)")
    ax1.set_ylabel("PSNR Calibration Residual (dB)\n[Achieved − Target]")
    ax1.set_title("A. PSNR Residuals vs. Stated Tolerance", fontweight="bold")
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax1.set_xlim(0.8, 5.2)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=8)

    # 2. SSIM Residuals
    ax2 = axes[1]
    for order in ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]:
        sub = df_res[df_res["order"] == order]
        ax2.plot(
            sub["target_idx"],
            sub["residual_ssim"],
            marker="s",
            linewidth=1.8,
            markersize=6,
            label=order,
            color=ORDER_COLORS.get(order, "#333"),
        )
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
    ax2.axhline(0.035, color="#dc2626", linestyle="--", linewidth=1.2, label="Tolerance (±0.035)")
    ax2.axhline(-0.035, color="#dc2626", linestyle="--", linewidth=1.2)
    ax2.fill_between([0.8, 5.2], -0.035, 0.035, color="#fee2e2", alpha=0.35)
    ax2.set_xlabel("Target Distortion Bin (1: Mild → 5: Severe)")
    ax2.set_ylabel("SSIM Calibration Residual\n[Achieved − Target]")
    ax2.set_title("B. SSIM Residuals vs. Stated Tolerance", fontweight="bold")
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.set_xlim(0.8, 5.2)
    ax2.legend(loc="lower left", framealpha=0.9, fontsize=8)

    plt.tight_layout()
    out_path = FIG_DIR / "fig1_calibration_residuals_and_tolerances.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def make_fig2_gap_contraction():
    """Figure 2: Sequence Gap Contraction (Nominal vs. Approximate-Fidelity)."""
    df_cells = pd.read_csv(ANALYSIS_DIR / "order_cell_means.csv")
    models = ["resnet50", "convnext_tiny", "vit_base", "swin_tiny", "Overall"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), dpi=300, sharey=True)

    orders = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]
    x = np.arange(len(models))
    width = 0.12

    # Panel A: Nominal
    ax1 = axes[0]
    for i, ord_name in enumerate(orders):
        accs = []
        for m in models:
            sub = df_cells[(df_cells["regime"] == "Nominal-Order") & (df_cells["model"] == m) & (df_cells["order"] == ord_name)]
            accs.append(sub["accuracy"].values[0] * 100.0)
        ax1.bar(x + (i - 2.5) * width, accs, width, label=ord_name, color=ORDER_COLORS[ord_name], alpha=0.9)

    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_DISPLAY[m] for m in models], rotation=15, ha="right")
    ax1.set_ylabel("Mean Top-1 Accuracy (%)")
    ax1.set_title("A. Nominal-Order Benchmark (Confounded Gaps: 17.1%–21.6%)", fontweight="bold")
    ax1.set_ylim(25, 75)
    ax1.legend(title="Sequence", loc="upper left", ncol=2, fontsize=8)

    # Panel B: Approximate-Fidelity
    ax2 = axes[1]
    for i, ord_name in enumerate(orders):
        accs = []
        for m in models:
            sub = df_cells[(df_cells["regime"] == "Approximate-Fidelity") & (df_cells["model"] == m) & (df_cells["order"] == ord_name)]
            accs.append(sub["accuracy"].values[0] * 100.0)
        ax2.bar(x + (i - 2.5) * width, accs, width, label=ord_name, color=ORDER_COLORS[ord_name], alpha=0.9)

    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_DISPLAY[m] for m in models], rotation=15, ha="right")
    ax2.set_title("B. Approximate-Fidelity Benchmark (Contracted Gaps: 4.7%–11.7%)", fontweight="bold")
    ax2.set_ylim(25, 75)
    ax2.legend(title="Sequence", loc="upper left", ncol=2, fontsize=8)

    plt.tight_layout()
    out_path = FIG_DIR / "fig2_nominal_vs_matched_gap_contraction.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def make_fig3_decomposition():
    """Figure 3: Forest Plot of Saturated Decomposition Parameters with 95% Bootstrap CIs."""
    df_coeffs = pd.read_csv(ANALYSIS_DIR / "decomposition_coefficients.csv")

    params_to_show = ["Start_Blur", "Start_JPEG", "End_Blur", "End_JPEG", "Parity_Contrast"]
    param_labels = [
        "Start with Blur vs. Noise",
        "Start with JPEG vs. Noise",
        "End with Blur vs. Noise",
        "End with JPEG vs. Noise",
        "Balanced Parity (Even vs. Odd Permutations)",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300, sharey=True)

    models = ["resnet50", "convnext_tiny", "vit_base", "swin_tiny"]
    colors_mod = {"resnet50": "#2563eb", "convnext_tiny": "#16a34a", "vit_base": "#9333ea", "swin_tiny": "#ea580c"}

    for panel_idx, (regime, title) in enumerate([
        ("Nominal-Order", "A. Nominal-Order Study"),
        ("Approximate-Fidelity", "B. Approximate-Fidelity Study"),
    ]):
        ax = axes[panel_idx]
        y_positions = np.arange(len(params_to_show))

        for m_idx, m in enumerate(models):
            sub = df_coeffs[(df_coeffs["regime"] == regime) & (df_coeffs["model"] == m)]
            offsets = (m_idx - 1.5) * 0.18

            est = []
            ci_l = []
            ci_u = []
            for p in params_to_show:
                row = sub[sub["parameter"] == p].iloc[0]
                est.append(row["estimate"] * 100.0)
                ci_l.append(row["ci_lower"] * 100.0)
                ci_u.append(row["ci_upper"] * 100.0)

            est = np.array(est)
            err_l = est - np.array(ci_l)
            err_u = np.array(ci_u) - est

            ax.errorbar(
                est,
                y_positions + offsets,
                xerr=[err_l, err_u],
                fmt="o",
                color=colors_mod[m],
                label=MODEL_DISPLAY[m] if panel_idx == 0 else "",
                capsize=3,
                elinewidth=1.4,
                markersize=5,
            )

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(param_labels)
        ax.set_xlabel("Effect on Top-1 Accuracy (% points, 95% Clustered CI)")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    out_path = FIG_DIR / "fig3_saturated_decomposition_coefficients.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def make_fig4_error_churn():
    """Figure 4: Decoupling of Net Accuracy Difference and Error Churn."""
    df_churn = pd.read_csv(ANALYSIS_DIR / "error_churn_matrix.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

    for panel_idx, (regime, title) in enumerate([
        ("Nominal-Order", "A. Nominal-Order Study (Confounded)"),
        ("Approximate-Fidelity", "B. Approximate-Fidelity Study (Calibrated)"),
    ]):
        ax = axes[panel_idx]
        sub = df_churn[df_churn["regime"] == regime]

        delta_pct = np.abs(sub["delta_acc"].values) * 100.0
        excess_pct = sub["excess_churn"].values * 100.0
        total_churn_pct = sub["churn_rate"].values * 100.0

        scatter = ax.scatter(
            delta_pct,
            excess_pct,
            c=total_churn_pct,
            cmap="viridis",
            s=55,
            alpha=0.85,
            edgecolors="w",
            linewidth=0.5,
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Total Error Churn (%)", fontsize=8)

        # Highlight point where delta_acc ~ 0 but excess churn is large
        near_zero = sub[np.abs(sub["delta_acc"]) < 0.03]
        if not near_zero.empty:
            max_case = near_zero.sort_values("excess_churn", ascending=False).iloc[0]
            ax.annotate(
                f"{max_case['model'][:7]}: {max_case['order_A']} vs {max_case['order_B']}\n|Δ|={abs(max_case['delta_acc'])*100:.1f}%, Churn={max_case['churn_rate']*100:.1f}%",
                xy=(abs(max_case["delta_acc"]) * 100.0, max_case["excess_churn"] * 100.0),
                xytext=(abs(max_case["delta_acc"]) * 100.0 + 3.0, max_case["excess_churn"] * 100.0 - 1.5),
                arrowprops=dict(facecolor="red", arrowstyle="->", lw=1.0),
                fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec="#f87171", lw=0.8),
            )

        ax.set_xlabel("|Net Accuracy Difference| (%)")
        ax.set_ylabel("Excess Churn [2 × min(Gain, Loss)] (%)")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = FIG_DIR / "fig4_paired_error_churn_decomposition.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def make_fig5_calibration():
    """Figure 5: ECE and Confidence Degradation across Sequence Orders."""
    df_cells = pd.read_csv(ANALYSIS_DIR / "order_cell_means.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

    models = ["resnet50", "convnext_tiny", "vit_base", "swin_tiny"]
    orders = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]

    for panel_idx, (regime, title) in enumerate([
        ("Nominal-Order", "A. Nominal-Order Confidence vs. Accuracy"),
        ("Approximate-Fidelity", "B. Approximate-Fidelity Confidence vs. Accuracy"),
    ]):
        ax = axes[panel_idx]
        sub = df_cells[df_cells["regime"] == regime]

        for m in models:
            sub_m = sub[sub["model"] == m]
            ax.scatter(
                sub_m["accuracy"] * 100.0,
                sub_m["confidence"] * 100.0,
                s=80,
                label=MODEL_DISPLAY[m],
                alpha=0.85,
                edgecolors="black",
                linewidth=0.5,
            )

        # Plot y=x perfect calibration line
        ax.plot([30, 75], [30, 75], color="red", linestyle="--", linewidth=1.0, label="Perfect Calibration (y=x)")
        ax.set_xlabel("Mean Top-1 Accuracy (%)")
        ax.set_ylabel("Mean Top-1 Confidence (%)")
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(30, 75)
        ax.set_ylim(30, 75)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = FIG_DIR / "fig5_calibration_ece_and_confidence.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def main():
    print("Generating publication-grade figures in reports/final_research_package/figures/...")
    make_fig1_calibration_residuals()
    make_fig2_gap_contraction()
    make_fig3_decomposition()
    make_fig4_error_churn()
    make_fig5_calibration()
    print("All 5 publication figures successfully generated!")


if __name__ == "__main__":
    main()
