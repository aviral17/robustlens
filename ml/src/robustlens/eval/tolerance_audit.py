"""Tolerance auditing module for multi-stage robustness benchmarking.

Audits whether calibrated corruption parameters meet prespecified physical matching
tolerances (PSNR +/- 0.75 dB, SSIM +/- 0.035).

Enforces scientific classification:
- If both PSNR and SSIM tolerances are satisfied: 'exact_matched'
- If tolerances are violated: 'approximate_fidelity_comparison'
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, Any
import numpy as np
import pandas as pd


def audit_matching_tolerances(
    residuals_path: str,
    summary_path: str,
    config_path: str,
) -> Dict[str, Any]:
    if not os.path.exists(residuals_path):
        raise FileNotFoundError(f"Missing residuals file: {residuals_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    tolerances = config["metadata"]["tolerances"]
    max_psnr_tol = float(tolerances["max_psnr_tolerance_db"])
    max_ssim_tol = float(tolerances["max_ssim_tolerance"])

    df_res = pd.read_csv(residuals_path)
    
    psnr_rmse = float(np.sqrt((df_res["residual_psnr"] ** 2).mean()))
    psnr_max_abs = float(df_res["residual_psnr"].abs().max())
    worst_psnr_row = df_res.loc[df_res["residual_psnr"].abs().idxmax()]

    ssim_rmse = float(np.sqrt((df_res["residual_ssim"] ** 2).mean()))
    ssim_max_abs = float(df_res["residual_ssim"].abs().max())
    worst_ssim_row = df_res.loc[df_res["residual_ssim"].abs().idxmax()]

    psnr_pass = bool(psnr_max_abs <= max_psnr_tol and psnr_rmse <= max_psnr_tol)
    ssim_pass = bool(ssim_max_abs <= max_ssim_tol and ssim_rmse <= max_ssim_tol)
    overall_exact_match_pass = psnr_pass and ssim_pass

    # Validation set spread check
    val_spreads = {}
    if os.path.exists(summary_path):
        df_sum = pd.read_csv(summary_path)
        if "target_idx" in df_sum.columns and df_sum["target_idx"].max() > 0:
            for tgt, g in df_sum[df_sum["target_idx"] > 0].groupby("target_idx"):
                p_spread = float(g["mean_psnr"].max() - g["mean_psnr"].min())
                s_spread = float(g["mean_ssim"].max() - g["mean_ssim"].min())
                val_spreads[int(tgt)] = {
                    "psnr_spread_db": round(p_spread, 2),
                    "ssim_spread": round(s_spread, 3),
                }

    report = {
        "prespecified_tolerances": {
            "max_psnr_tolerance_db": max_psnr_tol,
            "max_ssim_tolerance": max_ssim_tol,
        },
        "measured_calibration_residuals": {
            "psnr_rmse_db": round(psnr_rmse, 3),
            "psnr_max_abs_residual_db": round(psnr_max_abs, 3),
            "worst_psnr_condition": f"{worst_psnr_row['order']} Target {worst_psnr_row['target_idx']} ({worst_psnr_row['residual_psnr']:+.3f} dB)",
            "ssim_rmse": round(ssim_rmse, 4),
            "ssim_max_abs_residual": round(ssim_max_abs, 4),
            "worst_ssim_condition": f"{worst_ssim_row['order']} Target {worst_ssim_row['target_idx']} ({worst_ssim_row['residual_ssim']:+.4f})",
        },
        "validation_set_spreads": val_spreads,
        "criteria_verdict": {
            "psnr_matched_pass": psnr_pass,
            "ssim_matched_pass": ssim_pass,
            "exact_matching_achieved": overall_exact_match_pass,
            "scientific_classification": "exact_matched" if overall_exact_match_pass else "approximate_fidelity_comparison",
        },
        "scientific_guidance": (
            "PASSED: Strict fidelity matching verified."
            if overall_exact_match_pass
            else (
                "FAILED STRICT MATCHING: Calibration residuals exceed prespecified tolerances. "
                "The study MUST be scientifically classified and reported as an 'Approximate-Fidelity Comparison'. "
                "Do NOT claim that physical confounding has been eliminated or that fidelity is strictly matched."
            )
        ),
    }
    return report
