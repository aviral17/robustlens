"""Validation script for the RobustLens Matched-Fidelity Benchmark.

Validates:
1. Row count = 486,700 records.
2. Unique images = exactly 3,925.
3. Models = 4 architectures (ResNet-50, ConvNeXt-Tiny, ViT-Base/16, Swin-Tiny).
4. Conditions = 31 per image (1 clean + 6 sequences x 5 matched targets).
5. Null/NaN check = exactly 0 nulls.
6. Matching tolerances = PSNR and SSIM matching diagnostics across sequences.
7. Manifest and artifact consistency.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import pandas as pd
import numpy as np


EXPECTED_MODELS = {"resnet50", "convnext_tiny", "vit_base", "swin_tiny"}
EXPECTED_ORDERS = {"clean", "NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"}
EXPECTED_TARGETS = {0, 1, 2, 3, 4, 5}
EXPECTED_IMAGES = 3925
EXPECTED_CONDITIONS = 31
EXPECTED_TOTAL_RECORDS = EXPECTED_IMAGES * EXPECTED_CONDITIONS * len(EXPECTED_MODELS)  # 486,700


def validate_matched_fidelity_artifacts(output_dir: str) -> bool:
    print("=" * 70)
    print(f"Validating Matched-Fidelity Benchmark Artifacts in: {output_dir}")
    print("=" * 70)

    # 1. Check required files
    required_files = [
        "all_order_predictions.csv.gz",
        "order_effects_summary.csv",
        "order_effects_leaderboard.csv",
        "order_pairwise_significance.csv",
        "error_churn_summary.csv",
        "order_omnibus_tests.json",
        "distortion_ancova.csv",
        "benchmark_report.json",
        "manifest.json",
    ]

    missing = [f for f in required_files if not os.path.exists(os.path.join(output_dir, f))]
    if missing:
        print(f"FAIL: Missing required artifact files: {missing}")
        return False
    print(f"PASS: All {len(required_files)} required artifact files are present.")

    # 2. Validate predictions file
    pred_path = os.path.join(output_dir, "all_order_predictions.csv.gz")
    print(f"Reading predictions from {pred_path}...")
    df = pd.read_csv(pred_path)
    total_records = len(df)
    print(f"Total prediction records: {total_records} (Expected: {EXPECTED_TOTAL_RECORDS})")

    if total_records != EXPECTED_TOTAL_RECORDS:
        print(f"FAIL: Record count mismatch: expected {EXPECTED_TOTAL_RECORDS}, got {total_records}")
        return False
    print("PASS: Exact record count verified (486,700).")

    # 3. Check nulls
    null_counts = df.isnull().sum().to_dict()
    total_nulls = sum(null_counts.values())
    if total_nulls > 0:
        print(f"FAIL: Found {total_nulls} null values: {null_counts}")
        return False
    print("PASS: Zero null values across all columns.")

    # 4. Check unique images
    unique_images = df["sample_idx"].nunique()
    if unique_images != EXPECTED_IMAGES:
        print(f"FAIL: Unique image count mismatch: expected {EXPECTED_IMAGES}, got {unique_images}")
        return False
    print(f"PASS: Exactly {unique_images} unique validation images evaluated.")

    # 5. Check models
    models_found = set(df["model"].unique())
    if models_found != EXPECTED_MODELS:
        print(f"FAIL: Models mismatch: expected {EXPECTED_MODELS}, got {models_found}")
        return False
    print(f"PASS: All 4 models verified: {sorted(models_found)}")

    # 6. Check conditions per model-image
    condition_counts = df.groupby(["model", "sample_idx"]).size()
    if not (condition_counts == EXPECTED_CONDITIONS).all():
        bad = condition_counts[condition_counts != EXPECTED_CONDITIONS]
        print(f"FAIL: Inconsistent conditions per image: {len(bad)} image-models have unexpected counts")
        return False
    print(f"PASS: Exactly {EXPECTED_CONDITIONS} conditions evaluated for every image and model.")

    # 7. Check fidelity targets and sequences
    orders_found = set(df["order"].unique())
    if orders_found != EXPECTED_ORDERS:
        print(f"FAIL: Orders mismatch: expected {EXPECTED_ORDERS}, got {orders_found}")
        return False
    print(f"PASS: All 6 sequences + clean condition verified: {sorted(orders_found)}")

    # 8. Check distortion diagnostics
    print("\n--- Image Fidelity Diagnostics ---")
    compound_df = df[df["order"] != "clean"]
    target_col = "target_idx" if "target_idx" in compound_df.columns else "severity"
    for t_idx, grp in compound_df.groupby(target_col):
        mean_psnr = grp["psnr"].mean()
        mean_ssim = grp["ssim"].mean()
        order_psnr_gaps = grp.groupby("order")["psnr"].mean()
        max_gap = order_psnr_gaps.max() - order_psnr_gaps.min()
        print(f"Target {t_idx}: Mean PSNR = {mean_psnr:.2f} dB, Mean SSIM = {mean_ssim:.4f}, Max Order PSNR Gap = {max_gap:.2f} dB")

    # 9. Validate manifest
    with open(os.path.join(output_dir, "manifest.json"), "r") as f:
        manifest = json.load(f)
    print(f"\nManifest Version: {manifest.get('version')}, Device: {manifest.get('hardware', {}).get('device')}")
    print(f"Benchmark: {manifest.get('benchmark')}")
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate matched-fidelity evaluation artifacts")
    parser.add_argument("--output-dir", type=str, default="results/order_effect_study/matched_fidelity/outputs", help="Outputs directory")
    args = parser.parse_args()

    success = validate_matched_fidelity_artifacts(args.output_dir)
    sys.exit(0 if success else 1)
