"""Extract exact sequence-specific parameter triples for 3 matched common-support levels.

Level 1 (Mild): PSNR ~ 28.0 dB, SSIM ~ 0.850
Level 2 (Mod-Mild): PSNR ~ 24.5 dB, SSIM ~ 0.740
Level 3 (Moderate): PSNR ~ 21.5 dB, SSIM ~ 0.620

All 3 levels fall strictly inside the validated 6-way common support:
PSNR in [17.5, 32.3] dB, SSIM in [0.335, 0.895].
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
GRID_CSV = BASE_DIR / "results" / "matched_fidelity_study" / "common_support_grid_calibration.csv"
OUTPUT_JSON = BASE_DIR / "results" / "matched_fidelity_study" / "calibrated_common_support_3levels.json"

SEQUENCES = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]
TARGETS = {
    "Level_1_Mild": (28.0, 0.850),
    "Level_2_ModMild": (24.5, 0.740),
    "Level_3_Moderate": (21.5, 0.620),
}

df = pd.read_csv(GRID_CSV)
results = {}

for lvl_name, (t_p, t_s) in TARGETS.items():
    results[lvl_name] = {
        "target_psnr": t_p,
        "target_ssim": t_s,
        "sequences": {},
    }
    for seq in SEQUENCES:
        sub = df[df["order"] == seq].copy()
        norm_dist = np.sqrt(((sub["mean_psnr"] - t_p) / 0.75)**2 + ((sub["mean_ssim"] - t_s) / 0.035)**2)
        best_idx = norm_dist.idxmin()
        best = sub.loc[best_idx]

        results[lvl_name]["sequences"][seq] = {
            "noise_std": float(best["noise_std"]),
            "blur_sigma": float(best["blur_sigma"]),
            "jpeg_quality": int(best["jpeg_quality"]),
            "achieved_psnr": round(float(best["mean_psnr"]), 3),
            "achieved_ssim": round(float(best["mean_ssim"]), 4),
            "psnr_err": round(float(best["mean_psnr"] - t_p), 3),
            "ssim_err": round(float(best["mean_ssim"] - t_s), 4),
            "within_tolerance": bool(abs(best["mean_psnr"] - t_p) <= 0.75 and abs(best["mean_ssim"] - t_s) <= 0.035),
        }

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print("Extracted Sequence-Specific Parameters for 3 Common-Support Levels:")
for lvl, data in results.items():
    print(f"\n--- {lvl} (Target: {data['target_psnr']} dB, {data['target_ssim']} SSIM) ---")
    print(f"{'Order':<6} | {'Noise':<6} | {'Blur':<6} | {'JPEG':<6} | {'Achieved PSNR':<14} | {'Achieved SSIM':<14} | {'Within Tol?':<11}")
    print("-" * 75)
    for s, p in data["sequences"].items():
        print(f"{s:<6} | {p['noise_std']:<6.3f} | {p['blur_sigma']:<6.1f} | {p['jpeg_quality']:<6d} | {p['achieved_psnr']:<14.2f} | {p['achieved_ssim']:<14.3f} | {'YES' if p['within_tolerance'] else 'NO':<11}")
