"""Generate statistical benchmark reports from raw evaluation results.

Computes bootstrap confidence intervals, McNemar's significance tests,
Holm and Benjamini-Hochberg multiple-testing corrections, ECE calibration,
and conformal coverage analysis.

Usage:
    uv run python scripts/make_report.py --input ../results/raw/ --output ../results/
"""

import argparse
import json
import os
from typing import Dict, List, Any

import pandas as pd

from robustlens.eval.reporting import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Generate statistical analysis report from evaluation predictions.")
    parser.add_argument(
        "--input",
        type=str,
        default="../results/raw/",
        help="Directory containing all_predictions.json and manifest.json.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../results/",
        help="Directory to save final benchmark_report.json and summary tables.",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=10000,
        help="Number of bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05 for 95%% CIs).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap reproducibility.",
    )
    return parser.parse_args()


def load_predictions(input_dir: str) -> List[Dict[str, Any]]:
    preds_file = os.path.join(input_dir, "all_predictions.json")
    if not os.path.exists(preds_file):
        raise FileNotFoundError(f"Missing predictions file: {preds_file}. Run run_eval.py first.")
    with open(preds_file, "r") as f:
        return json.load(f)


def main():
    args = parse_args()
    print("=" * 60)
    print("RobustLens Statistical Report Generator")
    print("=" * 60)
    print(f"Input dir:  {args.input}")
    print(f"Output dir: {args.output}")

    records = load_predictions(args.input)
    print(f"Loaded {len(records)} prediction records.")

    print("Computing bootstrap CIs, paired tests, corrections, and calibration...")
    report = compute_metrics(
        records=records,
        n_bootstraps=args.n_bootstraps,
        alpha=args.alpha,
        seed=args.seed,
    )

    os.makedirs(args.output, exist_ok=True)

    # 1. Save JSON benchmark report
    report_path = os.path.join(args.output, "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Generated benchmark JSON report: {report_path}")

    # 2. Save CSV summary tables
    df_leaderboard = pd.DataFrame(report["leaderboard"])
    csv_lb_path = os.path.join(args.output, "leaderboard_summary.csv")
    df_leaderboard.to_csv(csv_lb_path, index=False)
    print(f"Generated CSV leaderboard:       {csv_lb_path}")

    if report["pairwise_significance"]:
        df_pairwise = pd.DataFrame(report["pairwise_significance"])
        csv_pair_path = os.path.join(args.output, "pairwise_significance.csv")
        df_pairwise.to_csv(csv_pair_path, index=False)
        print(f"Generated CSV pairwise tests:    {csv_pair_path}")

    df_breakdown = pd.DataFrame(report["detailed_breakdown"])
    csv_bk_path = os.path.join(args.output, "detailed_breakdown.csv")
    df_breakdown.to_csv(csv_bk_path, index=False)
    print(f"Generated CSV condition table:   {csv_bk_path}")

    print("\nSummary Leaderboard:")
    print(df_leaderboard[["display_name", "clean_accuracy", "mean_shifted_accuracy", "mean_accuracy_drop", "mean_ece"]])
    print("=" * 60)


if __name__ == "__main__":
    main()
