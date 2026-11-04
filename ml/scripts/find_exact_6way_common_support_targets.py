"""Search for valid 6-way simultaneous common-support targets where ALL 6 sequences meet tolerances.

Tolerances:
- |PSNR_k - PSNR*| <= 0.75 dB
- |SSIM_k - SSIM*| <= 0.035
All 6 sequences must simultaneously satisfy both tolerances.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
GRID_CSV = BASE_DIR / "results" / "matched_fidelity_study" / "common_support_grid_calibration.csv"
OUTPUT_JSON = BASE_DIR / "results" / "matched_fidelity_study" / "valid_simultaneous_6way_targets.json"

df = pd.read_csv(GRID_CSV)
SEQUENCES = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]

# Fine candidate mesh in (PSNR, SSIM)
psnr_candidates = np.arange(18.0, 32.1, 0.2)
ssim_candidates = np.arange(0.35, 0.901, 0.005)

valid_targets = []

for p_target in psnr_candidates:
    for s_target in ssim_candidates:
        seq_matches = {}
        all_pass = True
        for seq in SEQUENCES:
            sub = df[df["order"] == seq]
            p_diff = (sub["mean_psnr"] - p_target).abs()
            s_diff = (sub["mean_ssim"] - s_target).abs()
            pass_mask = (p_diff <= 0.75) & (s_diff <= 0.035)

            if not pass_mask.any():
                all_pass = False
                break
            else:
                best_idx = ((p_diff / 0.75)**2 + (s_diff / 0.035)**2).idxmin()
                best = sub.loc[best_idx]
                seq_matches[seq] = {
                    "noise_std": float(best["noise_std"]),
                    "blur_sigma": float(best["blur_sigma"]),
                    "jpeg_quality": int(best["jpeg_quality"]),
                    "achieved_psnr": round(float(best["mean_psnr"]), 3),
                    "achieved_ssim": round(float(best["mean_ssim"]), 4),
                    "psnr_err": round(float(best["mean_psnr"] - p_target), 3),
                    "ssim_err": round(float(best["mean_ssim"] - s_target), 4),
                }

        if all_pass:
            # Calculate maximum normalized error across all 6 sequences
            max_norm_err = max(
                np.sqrt((v["psnr_err"] / 0.75)**2 + (v["ssim_err"] / 0.035)**2)
                for v in seq_matches.values()
            )
            valid_targets.append({
                "psnr_target": round(float(p_target), 2),
                "ssim_target": round(float(s_target), 3),
                "max_normalized_error": round(float(max_norm_err), 4),
                "sequences": seq_matches,
            })

print(f"Total candidate points tested: {len(psnr_candidates) * len(ssim_candidates)}")
print(f"Total valid 6-way simultaneous common-support targets found: {len(valid_targets)}")

if valid_targets:
    df_targets = pd.DataFrame([
        {
            "psnr": t["psnr_target"],
            "ssim": t["ssim_target"],
            "max_norm_err": t["max_normalized_error"],
        }
        for t in valid_targets
    ])
    print(f"PSNR range of valid 6-way targets: [{df_targets['psnr'].min():.1f}, {df_targets['psnr'].max():.1f}] dB")
    print(f"SSIM range of valid 6-way targets: [{df_targets['ssim'].min():.3f}, {df_targets['ssim'].max():.3f}]")

    # Select 3 well-separated robust targets:
    # 1. High fidelity (Mild): PSNR ~ 28.0 dB
    # 2. Medium fidelity (Moderate): PSNR ~ 24.5 dB
    # 3. Lower fidelity (Common-Severe): PSNR ~ 21.0 - 22.0 dB
    sub_mild = [t for t in valid_targets if t["psnr_target"] >= 27.0]
    sub_mod = [t for t in valid_targets if 23.5 <= t["psnr_target"] <= 25.5]
    sub_sev = [t for t in valid_targets if 20.0 <= t["psnr_target"] <= 22.5]

    best_mild = min(sub_mild, key=lambda x: x["max_normalized_error"]) if sub_mild else None
    best_mod = min(sub_mod, key=lambda x: x["max_normalized_error"]) if sub_mod else None
    best_sev = min(sub_sev, key=lambda x: x["max_normalized_error"]) if sub_sev else None

    selected_3 = {
        "Level_1_Mild": best_mild,
        "Level_2_Moderate": best_mod,
        "Level_3_SevereCommon": best_sev,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "all_valid_targets_count": len(valid_targets),
            "selected_3_robust_levels": selected_3,
            "all_valid_targets": valid_targets,
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("SELECTED 3 ROBUST FULLY-MATCHED 6-WAY COMMON-SUPPORT TARGETS:")
    print("=" * 80)
    for lvl, data in selected_3.items():
        if data:
            print(f"\n--- {lvl}: Target PSNR {data['psnr_target']:.2f} dB, Target SSIM {data['ssim_target']:.3f} (Max Norm Err: {data['max_normalized_error']:.3f}) ---")
            print(f"{'Order':<6} | {'Noise':<6} | {'Blur':<6} | {'JPEG':<6} | {'Achieved PSNR':<14} | {'Achieved SSIM':<14} | {'PSNR Err':<10} | {'SSIM Err':<10}")
            print("-" * 88)
            for s, p in data["sequences"].items():
                print(f"{s:<6} | {p['noise_std']:<6.3f} | {p['blur_sigma']:<6.1f} | {p['jpeg_quality']:<6d} | {p['achieved_psnr']:<14.2f} | {p['achieved_ssim']:<14.3f} | {p['psnr_err']:<+10.2f} | {p['ssim_err']:<+10.4f}")
else:
    print("NO valid 6-way simultaneous common support targets found in coarse grid!")
