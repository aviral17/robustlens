"""Automated verification of fidelity matching tolerances.

Audits whether the calibrated multi-stage corruption benchmark meets the
prespecified physical matching tolerances (PSNR +/- 0.75 dB, SSIM +/- 0.035).

If tolerances are violated, prevents any downstream script or report from
claiming 'strict matching' or 'eliminated confounding'.
"""

from __future__ import annotations

import json
import sys

from robustlens.eval.tolerance_audit import audit_matching_tolerances


def main():
    print("=" * 80)
    print("ROBUSTNESS BENCHMARK: FIDELITY MATCHING TOLERANCE AUDIT")
    print("=" * 80)
    from pathlib import Path

    def resolve(p_str: str) -> str:
        p = Path(p_str)
        if p.exists():
            return str(p)
        repo_root = Path(__file__).resolve().parents[2]
        alt = repo_root / p_str
        if alt.exists():
            return str(alt)
        return p_str

    report = audit_matching_tolerances(
        residuals_path=resolve("results/order_effect_study/matched_fidelity/matching_residuals.csv"),
        summary_path=resolve("results/order_effect_study/matched_fidelity/outputs/order_effects_summary.csv"),
        config_path=resolve("results/order_effect_study/matched_fidelity/calibration_config.json"),
    )
    print(json.dumps(report, indent=2))
    print("=" * 80)
    
    if not report["criteria_verdict"]["exact_matching_achieved"]:
        print("\n[SCIENTIFIC AUDIT NOTICE]:", file=sys.stderr)
        print(f"  * PSNR tolerance: {report['prespecified_tolerances']['max_psnr_tolerance_db']} dB | Measured RMSE: {report['measured_calibration_residuals']['psnr_rmse_db']} dB (Max: {report['measured_calibration_residuals']['psnr_max_abs_residual_db']} dB)", file=sys.stderr)
        print(f"  * SSIM tolerance: {report['prespecified_tolerances']['max_ssim_tolerance']} | Measured RMSE: {report['measured_calibration_residuals']['ssim_rmse']} (Max: {report['measured_calibration_residuals']['ssim_max_abs_residual']})", file=sys.stderr)
        print(f"  * Classification: {report['criteria_verdict']['scientific_classification'].upper()}", file=sys.stderr)
        print(f"  * Guidance: {report['scientific_guidance']}\n", file=sys.stderr)
        sys.exit(2)
    else:
        print("\n[SUCCESS]: Prespecified matching criteria passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
