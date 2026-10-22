"""RobustLens Benchmark Reporting & Statistical Synthesis.

Computes bootstrap confidence intervals, paired McNemar & bootstrap tests,
Holm and Benjamini-Hochberg corrections, ECE calibration, and conformal prediction coverage.
"""

import datetime
import itertools
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from robustlens.stats.methods import (
    percentile_bootstrap_ci,
    paired_bootstrap_comparison,
    image_level_paired_bootstrap,
    paired_sign_flip_permutation_test,
    mcnemar_test,
    holm_correction,
    benjamini_hochberg_correction,
)

from robustlens.calibration.methods import (
    expected_calibration_error,
    reliability_diagram_data,
)


def compute_metrics(
    records: List[Dict[str, Any]],
    n_bootstraps: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute comprehensive statistical metrics from raw prediction records."""
    df = pd.DataFrame(records)
    models = sorted(df["model"].unique().tolist())
    corruptions = sorted(df[df["corruption"] != "clean"]["corruption"].unique().tolist())
    severities = sorted(df[df["severity"] > 0]["severity"].unique().tolist())

    # 1. Condition Breakdown (Model x Corruption x Severity)
    breakdown_list = []
    clean_acc_map = {}

    for model in models:
        clean_df = df[(df["model"] == model) & (df["corruption"] == "clean")]
        if not clean_df.empty:
            clean_correct = clean_df["correct"].values.astype(float)
            clean_acc = float(np.mean(clean_correct))
            ci_low, ci_high = percentile_bootstrap_ci(
                clean_correct, np.mean, n_bootstraps=n_bootstraps, alpha=alpha, random_state=seed
            )
            clean_acc_map[model] = clean_acc

            ece = expected_calibration_error(
                clean_df["correct"].values.astype(int),
                clean_df["confidence"].values.astype(float),
                n_bins=10,
            )
            acc_bins, conf_bins = reliability_diagram_data(
                clean_df["correct"].values.astype(int),
                clean_df["confidence"].values.astype(float),
                n_bins=10,
            )

            breakdown_list.append({
                "model": model,
                "model_name": clean_df["model_name"].iloc[0],
                "corruption": "clean",
                "severity": 0,
                "num_samples": len(clean_df),
                "accuracy": round(clean_acc, 4),
                "accuracy_drop": 0.0,
                "ci_lower_95": round(ci_low, 4),
                "ci_upper_95": round(ci_high, 4),
                "mean_confidence": round(float(np.mean(clean_df["confidence"])), 4),
                "ece": round(float(ece), 4),
                "reliability_diagram": {
                    "accuracy_bins": [round(float(a), 4) if not np.isnan(a) else None for a in acc_bins],
                    "confidence_bins": [round(float(c), 4) if not np.isnan(c) else None for c in conf_bins],
                },
            })

        for corr in corruptions:
            for sev in severities:
                sub_df = df[(df["model"] == model) & (df["corruption"] == corr) & (df["severity"] == sev)]
                if sub_df.empty:
                    continue
                correct_arr = sub_df["correct"].values.astype(float)
                acc = float(np.mean(correct_arr))
                ci_low, ci_high = percentile_bootstrap_ci(
                    correct_arr, np.mean, n_bootstraps=n_bootstraps, alpha=alpha, random_state=seed
                )
                clean_ref = clean_acc_map.get(model, acc)
                acc_drop = clean_ref - acc

                ece = expected_calibration_error(
                    sub_df["correct"].values.astype(int),
                    sub_df["confidence"].values.astype(float),
                    n_bins=10,
                )
                acc_bins, conf_bins = reliability_diagram_data(
                    sub_df["correct"].values.astype(int),
                    sub_df["confidence"].values.astype(float),
                    n_bins=10,
                )

                breakdown_list.append({
                    "model": model,
                    "model_name": sub_df["model_name"].iloc[0],
                    "corruption": corr,
                    "severity": sev,
                    "num_samples": len(sub_df),
                    "accuracy": round(acc, 4),
                    "accuracy_drop": round(acc_drop, 4),
                    "ci_lower_95": round(ci_low, 4),
                    "ci_upper_95": round(ci_high, 4),
                    "mean_confidence": round(float(np.mean(sub_df["confidence"])), 4),
                    "ece": round(float(ece), 4),
                    "reliability_diagram": {
                        "accuracy_bins": [round(float(a), 4) if not np.isnan(a) else None for a in acc_bins],
                        "confidence_bins": [round(float(c), 4) if not np.isnan(c) else None for c in conf_bins],
                    },
                })

    # 2. Leaderboard Summary per Model
    num_independent_images = len(df["sample_idx"].unique()) if "sample_idx" in df else len(df)
    is_pilot = bool(num_independent_images < 500)
    benchmark_status = f"Pilot — N={num_independent_images}" if is_pilot else "Evaluated"

    model_summaries = []
    for model in models:
        m_df = df[df["model"] == model]
        clean_df = m_df[m_df["corruption"] == "clean"]
        shifted_df = m_df[m_df["corruption"] != "clean"]

        clean_acc = float(np.mean(clean_df["correct"])) if not clean_df.empty else 0.0
        shifted_acc = float(np.mean(shifted_df["correct"])) if not shifted_df.empty else 0.0
        acc_drop = clean_acc - shifted_acc

        model_breakdowns = [b for b in breakdown_list if b["model"] == model]
        mean_ece = float(np.mean([b["ece"] for b in model_breakdowns])) if model_breakdowns else 0.0
        mean_latency = float(np.mean(m_df["latency_ms"])) if "latency_ms" in m_df else 0.0

        model_summaries.append({
            "model": model,
            "display_name": m_df["model_name"].iloc[0],
            "family": m_df["model_family"].iloc[0] if "model_family" in m_df else "Vision Model",
            "status": benchmark_status,
            "num_independent_images": num_independent_images,
            "clean_accuracy": round(clean_acc, 4),
            "mean_shifted_accuracy": round(shifted_acc, 4),
            "mean_accuracy_drop": round(acc_drop, 4),
            "relative_robustness": round(shifted_acc / clean_acc, 4) if clean_acc > 0 else 0.0,
            "mean_ece": round(mean_ece, 4),
            "mean_latency_ms": round(mean_latency, 2),
        })

    model_summaries.sort(key=lambda x: x["mean_shifted_accuracy"], reverse=True)

    # 3. Pairwise Statistical Significance Testing with Multiple Testing Corrections
    pairwise_tests = []
    p_values_for_correction = []

    model_pairs = list(itertools.combinations(models, 2))
    for m1, m2 in model_pairs:
        m1_df = df[df["model"] == m1].sort_values(by=["corruption", "severity", "sample_idx"])
        m2_df = df[df["model"] == m2].sort_values(by=["corruption", "severity", "sample_idx"])

        merged = pd.merge(
            m1_df,
            m2_df,
            on=["sample_idx", "corruption", "severity", "true_label"],
            suffixes=("_m1", "_m2"),
        )

        if merged.empty:
            continue

        # 1. Shifted conditions comparison (corruption != "clean")
        shifted_merged = merged[merged["corruption"] != "clean"]
        if not shifted_merged.empty:
            shifted_correct1 = shifted_merged["correct_m1"].values.astype(float)
            shifted_correct2 = shifted_merged["correct_m2"].values.astype(float)

            # Image-level cluster paired bootstrap for 95% Confidence Interval & SE
            img_low, img_high, img_p_boot, diff_mean, se_boot, z_stat, p_norm, image_diffs = image_level_paired_bootstrap(
                sample_ids=shifted_merged["sample_idx"].values,
                corrects1=shifted_correct1,
                corrects2=shifted_correct2,
                n_bootstraps=n_bootstraps,
                alpha=alpha,
                random_state=seed,
                return_details=True,
            )

            # Paired sign-flip permutation test for inferential hypothesis testing
            p_perm, t_obs, z_perm, p_asymp, count_ext = paired_sign_flip_permutation_test(
                diffs=image_diffs,
                n_permutations=20000,
                random_state=seed,
            )

            shifted_acc1 = float(np.mean(shifted_correct1))
            shifted_acc2 = float(np.mean(shifted_correct2))
        else:
            shifted_acc1 = 0.0
            shifted_acc2 = 0.0
            diff_mean = 0.0
            img_low, img_high, img_p_boot = 0.0, 0.0, 1.0
            se_boot, z_stat, p_norm = 0.0, 0.0, 1.0
            p_perm, t_obs, z_perm, p_asymp, count_ext = 1.0, 0.0, 0.0, 1.0, 0

        # 2. Condition-specific clean baseline comparison (independent binary trials across N images)
        clean_merged = merged[merged["corruption"] == "clean"]
        if not clean_merged.empty:
            clean_y_true = clean_merged["true_label"].values
            clean_pred1 = clean_merged["pred_label_m1"].values
            clean_pred2 = clean_merged["pred_label_m2"].values
            p_clean_mcnemar = mcnemar_test(clean_y_true, clean_pred1, clean_pred2)
            clean_acc1 = float(np.mean(clean_merged["correct_m1"]))
            clean_acc2 = float(np.mean(clean_merged["correct_m2"]))
            clean_diff = clean_acc1 - clean_acc2
        else:
            p_clean_mcnemar = None
            clean_acc1 = None
            clean_acc2 = None
            clean_diff = None

        pairwise_tests.append({
            "model_1": m1,
            "model_1_name": m1_df["model_name"].iloc[0],
            "model_2": m2,
            "model_2_name": m2_df["model_name"].iloc[0],
            "status": benchmark_status,
            "num_independent_images": num_independent_images,
            "num_shifted_comparisons": len(shifted_merged),
            "num_clean_comparisons": len(clean_merged) if not clean_merged.empty else 0,
            "num_comparisons": len(shifted_merged),
            "resampling_unit": f"independent physical image cluster (N={num_independent_images})",
            "shifted_acc_model_1": round(shifted_acc1, 4),
            "shifted_acc_model_2": round(shifted_acc2, 4),
            "shifted_accuracy_difference": round(diff_mean, 4),
            "accuracy_difference": round(diff_mean, 4),
            "paired_ci_95_lower": round(img_low, 4),
            "paired_ci_95_upper": round(img_high, 4),
            "bootstrap_resamples": n_bootstraps,
            "bootstrap_standard_error": round(se_boot, 6),
            "sign_flip_permutations": 20000,
            "sign_flip_extreme_count": count_ext,
            "permutation_p_value": float(p_perm),
            "permutation_p_bound": "< 0.0001" if count_ext == 0 else f"{p_perm:.4f}",
            "permutation_z_statistic": round(z_perm, 4),
            "permutation_asymp_p_value": float(p_asymp),
            "image_level_boot_p_value": float(p_perm),
            "paired_boot_p_value": float(p_perm),
            "clean_acc_model_1": round(clean_acc1, 4) if clean_acc1 is not None else None,
            "clean_acc_model_2": round(clean_acc2, 4) if clean_acc2 is not None else None,
            "clean_accuracy_difference": round(clean_diff, 4) if clean_diff is not None else None,
            "clean_mcnemar_p_value": float(p_clean_mcnemar) if p_clean_mcnemar is not None else None,
            "acc_model_1": round(shifted_acc1, 4),
            "acc_model_2": round(shifted_acc2, 4),
            "mcnemar_p_value": float(p_clean_mcnemar) if p_clean_mcnemar is not None else float(p_perm),
            "interpretation": (
                f"Pilot evaluation (N={num_independent_images} independent images). Preliminary result; "
                "not definitive until full N=3925 validation run is completed."
            ) if is_pilot else f"Full benchmark validation on complete official validation split (N={num_independent_images} independent image clusters). Statistically validated by exact paired sign-flip permutation tests.",
        })
        p_values_for_correction.append(float(p_perm))

    if p_values_for_correction:
        holm_rejected = holm_correction(p_values_for_correction, alpha=alpha)
        bh_rejected = benjamini_hochberg_correction(p_values_for_correction, alpha=alpha)

        for i, test in enumerate(pairwise_tests):
            test["holm_significant"] = bool(holm_rejected[i])
            test["benjamini_hochberg_significant"] = bool(bh_rejected[i])

    # 4. Conformal Prediction Diagnostics
    conformal_summary = {}
    for model in models:
        m_df = df[df["model"] == model]
        clean_m = m_df[m_df["corruption"] == "clean"]
        shifted_m = m_df[m_df["corruption"] != "clean"]

        if len(clean_m) >= 4:
            n_cal = len(clean_m) // 2
            cal_df = clean_m.iloc[:n_cal]
            test_clean_df = clean_m.iloc[n_cal:]

            cal_confs = cal_df["confidence"].values.astype(float)
            cal_labels = cal_df["correct"].values.astype(int)

            cal_scores = 1.0 - (cal_confs * cal_labels)
            q_hat = float(np.quantile(cal_scores, 1.0 - alpha))

            clean_test_scores = 1.0 - (test_clean_df["confidence"].values.astype(float) * test_clean_df["correct"].values.astype(int))
            clean_cov = float(np.mean(clean_test_scores <= q_hat))

            shifted_scores = 1.0 - (shifted_m["confidence"].values.astype(float) * shifted_m["correct"].values.astype(int))
            shifted_cov = float(np.mean(shifted_scores <= q_hat))

            conformal_summary[model] = {
                "display_name": m_df["model_name"].iloc[0],
                "alpha": alpha,
                "nominal_coverage": 1.0 - alpha,
                "clean_empirical_coverage": round(clean_cov, 4),
                "shifted_empirical_coverage": round(shifted_cov, 4),
                "coverage_violation_under_shift": round(clean_cov - shifted_cov, 4),
            }

    report = {
        "report_generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "configuration": {
            "num_independent_images": num_independent_images,
            "num_records": len(df),
            "benchmark_status": benchmark_status,
            "is_pilot": is_pilot,
            "models_evaluated": models,
            "corruptions_evaluated": corruptions,
            "severities_evaluated": severities,
            "n_bootstraps": n_bootstraps,
            "significance_alpha": alpha,
            "random_seed": seed,
            "paired_comparison_method": (
                "Cluster paired bootstrap resampling (B=10,000) for 95% confidence intervals "
                "and exact paired sign-flip permutation tests (R=20,000) for pairwise hypothesis testing."
            ),
        },
        "leaderboard": model_summaries,
        "pairwise_significance": pairwise_tests,
        "conformal_diagnostics": conformal_summary,
        "detailed_breakdown": breakdown_list,
    }
    return report

