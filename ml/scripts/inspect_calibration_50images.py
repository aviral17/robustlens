import json

with open("results/matched_fidelity_study/calibration_50images_verification.json", "r", encoding="utf-8") as f:
    d = json.load(f)

print("=" * 80)
print("50-IMAGE CALIBRATION VALIDATION REPORT ACROSS ALL 18 CONDITIONS")
print("=" * 80)

for lvl, l in d["level_summaries"].items():
    t_p = l["target_psnr"]
    t_s = l["target_ssim"]
    print(f"\n--- {lvl} (Target: {t_p:.2f} dB, {t_s:.3f} SSIM) ---")
    print(f"{'Order':<6} | {'Final Op':<8} | {'50-Img PSNR':<18} | {'50-Img SSIM':<18} | {'Δ Grid vs Hold':<18} | {'Pass?':<6}")
    print("-" * 80)
    for s, p in l["sequences"].items():
        a = p["all_50_images"]
        delta = p["subsample_generalization_delta"]
        final_op = "JPEG" if s.endswith("J") else ("Blur" if s.endswith("B") else "Noise")
        p_str = f"{a['psnr_mean']:.2f} ± {a['psnr_se']:.2f} (err:{a['psnr_error_vs_target']:+.2f})"
        s_str = f"{a['ssim_mean']:.3f} ± {a['ssim_se']:.3f} (err:{a['ssim_error_vs_target']:+.3f})"
        d_str = f"dP:{delta['psnr_delta']:+.2f}, dS:{delta['ssim_delta']:+.3f}"
        status = "PASS" if a["within_prespecified_tolerances"] else "FAIL"
        print(f"{s:<6} | {final_op:<8} | {p_str:<18} | {s_str:<18} | {d_str:<18} | {status:<6}")
