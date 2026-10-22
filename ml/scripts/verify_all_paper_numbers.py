"""Comprehensive verification of every empirical number in the paper against raw artifacts."""

import json
from pathlib import Path
import pandas as pd
import numpy as np

def verify():
    base = Path(__file__).resolve().parents[2]
    matched_dir = base / "results" / "order_effect_study" / "matched_fidelity"
    full_dir = base / "results" / "order_effect_study" / "full"

    print("=" * 80)
    print("1. CALIBRATION RESIDUALS AND TOLERANCES")
    print("=" * 80)
    residuals_file = matched_dir / "matching_residuals.csv"
    df_res = pd.read_csv(residuals_file)
    psnr_rmse = np.sqrt((df_res["residual_psnr"] ** 2).mean())
    psnr_max_abs = df_res["residual_psnr"].abs().max()
    worst_psnr_idx = df_res["residual_psnr"].abs().idxmax()
    worst_psnr = df_res.loc[worst_psnr_idx]

    ssim_rmse = np.sqrt((df_res["residual_ssim"] ** 2).mean())
    ssim_max_abs = df_res["residual_ssim"].abs().max()
    worst_ssim_idx = df_res["residual_ssim"].abs().idxmax()
    worst_ssim = df_res.loc[worst_ssim_idx]

    print(f"PSNR Calibration RMSE: {psnr_rmse:.3f} dB")
    print(f"PSNR Max Abs Residual: {psnr_max_abs:.3f} dB (Seq: {worst_psnr['order']}, Target: {worst_psnr['target_idx']}, Res: {worst_psnr['residual_psnr']:+.3f})")
    print(f"SSIM Calibration RMSE: {ssim_rmse:.4f}")
    print(f"SSIM Max Abs Residual: {ssim_max_abs:.4f} (Seq: {worst_ssim['order']}, Target: {worst_ssim['target_idx']}, Res: {worst_ssim['residual_ssim']:+.4f})")

    print("\nResiduals per sequence and target:")
    for tgt, g in df_res.groupby("target_idx"):
        print(f"Target {tgt}: Target PSNR={g['target_psnr'].iloc[0]:.1f}, Target SSIM={g['target_ssim'].iloc[0]:.3f}")
        for _, r in g.iterrows():
            print(f"  Seq {r['order']}: Achieved PSNR={r['achieved_psnr']:.2f} (diff={r['residual_psnr']:+.2f}), Achieved SSIM={r['achieved_ssim']:.3f} (diff={r['residual_ssim']:+.3f})")

    print("\n" + "=" * 80)
    print("2. VALIDATION SET FIDELITY SPREADS")
    print("=" * 80)
    summary_file = matched_dir / "outputs" / "order_effects_summary.csv"
    df_sum = pd.read_csv(summary_file)
    if "target_idx" in df_sum.columns:
        for tgt, g in df_sum[df_sum["target_idx"] > 0].groupby("target_idx"):
            psnr_spread = g["mean_psnr"].max() - g["mean_psnr"].min()
            ssim_spread = g["mean_ssim"].max() - g["mean_ssim"].min()
            print(f"Target {tgt} Validation Spread: PSNR Spread = {psnr_spread:.2f} dB [{g['mean_psnr'].min():.2f} - {g['mean_psnr'].max():.2f}], SSIM Spread = {ssim_spread:.3f} [{g['mean_ssim'].min():.3f} - {g['mean_ssim'].max():.3f}]")

    print("\n" + "=" * 80)
    print("3. NOMINAL VS MATCHED SUMMARY & CONTRACTION")
    print("=" * 80)
    comp_file = matched_dir / "outputs" / "nominal_vs_matched_summary.csv"
    df_comp = pd.read_csv(comp_file)
    print(df_comp.to_string(index=False))

    print("\n" + "=" * 80)
    print("4. FACTORIAL ANCOVA")
    print("=" * 80)
    ancova_file = matched_dir / "outputs" / "matched_factorial_ancova.csv"
    if ancova_file.exists():
        df_anc = pd.read_csv(ancova_file)
        print(df_anc.to_string(index=False))

    print("\n" + "=" * 80)
    print("5. ERROR CHURN BY MODEL")
    print("=" * 80)
    churn_file = matched_dir / "outputs" / "error_churn_summary.csv"
    if churn_file.exists():
        df_churn = pd.read_csv(churn_file)
        churn_agg = df_churn.groupby("model")[["gain_rate", "loss_rate", "net_difference", "error_churn", "excess_churn"]].mean()
        print(churn_agg)
        
        # excess churn proportion
        df_churn["excess_ratio"] = df_churn["excess_churn"] / df_churn["error_churn"]
        print("\nExcess Churn as proportion of Total Churn:")
        print(df_churn.groupby("model")["excess_ratio"].mean())

    print("\n" + "=" * 80)
    print("6. SEQUENCE-LEVEL MEANS (MATCHED FIDELITY)")
    print("=" * 80)
    for model in ["convnext_tiny", "swin_tiny", "vit_base", "resnet50"]:
        sub_m = df_sum[(df_sum["model"] == model) & (df_sum["target_idx"] > 0)]
        print(f"\nModel: {model} (Matched Fidelity):")
        seq_means = sub_m.groupby("order")["accuracy"].mean().sort_values(ascending=False)
        for s, acc in seq_means.items():
            print(f"  {s}: {acc * 100:.2f}%")

    print("\n" + "=" * 80)
    print("7. SEQUENCE-LEVEL MEANS (NOMINAL STUDY)")
    print("=" * 80)
    full_summary = full_dir / "outputs" / "order_effects_summary.csv"
    if full_summary.exists():
        df_full = pd.read_csv(full_summary)
        for model in ["convnext_tiny", "swin_tiny", "vit_base", "resnet50"]:
            sub_m = df_full[(df_full["model"] == model) & (df_full["severity"] > 0)]
            print(f"\nModel: {model} (Nominal Study):")
            seq_means = sub_m.groupby("order")["accuracy"].mean().sort_values(ascending=False)
            for s, acc in seq_means.items():
                print(f"  {s}: {acc * 100:.2f}%")

if __name__ == "__main__":
    verify()
