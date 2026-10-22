"""Validation script for full order effects benchmark output artifacts."""

import argparse
import json
import os
import pandas as pd
import numpy as np

def validate_full_benchmark(full_dir: str = "results/order_effect_study/full/outputs/", expected_images: int = 3925):
    print(f"=== Validating Full Order-Effects Benchmark in {full_dir} ===")
    
    pred_path = os.path.join(full_dir, "all_order_predictions.csv.gz")
    if not os.path.exists(pred_path):
        pred_path_uncompressed = os.path.join(full_dir, "all_order_predictions.csv")
        if os.path.exists(pred_path_uncompressed):
            pred_path = pred_path_uncompressed
        else:
            raise FileNotFoundError(f"Predictions file not found in {full_dir}")

    print(f"Reading predictions from {pred_path}...")
    df = pd.read_csv(pred_path)
    print(f"1. Loaded predictions: {len(df)} rows")

    # Check expected counts
    expected_models = {"resnet50", "convnext_tiny", "vit_base", "swin_tiny"}
    expected_orders = {"clean", "NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"}
    expected_severities = {0, 1, 2, 3, 4, 5}
    expected_conditions = 31  # 1 clean + 6 * 5
    expected_total = expected_images * expected_conditions * len(expected_models)

    assert len(df) == expected_total, f"Expected {expected_total} rows, got {len(df)}"
    print(f"   [PASS] Row count matches exactly {expected_total:,} ({expected_images:,} images x 31 conditions x 4 models)")

    # Check nulls
    null_counts = df.isnull().sum().to_dict()
    assert sum(null_counts.values()) == 0, f"Found null values: {null_counts}"
    print(f"   [PASS] 0 null values detected across all columns")

    # Check unique images
    unique_images = df["sample_idx"].unique()
    assert len(unique_images) == expected_images, f"Expected {expected_images} unique images, got {len(unique_images)}"
    print(f"   [PASS] Exactly {expected_images} unique validation images evaluated")

    # Check models
    found_models = set(df["model"].unique())
    assert found_models == expected_models, f"Models mismatch: {found_models} vs {expected_models}"
    print(f"   [PASS] All 4 architectures present: {sorted(list(found_models))}")

    # Check orders and severities
    found_orders = set(df["order"].unique())
    assert found_orders == expected_orders, f"Orders mismatch: {found_orders} vs {expected_orders}"
    found_sevs = set(df["severity"].unique())
    assert found_sevs == expected_severities, f"Severities mismatch: {found_sevs} vs {expected_severities}"
    print(f"   [PASS] All 6 permutation orders and 5 severities + clean baseline verified")

    # Check PSNR & SSIM determinism
    grouped_dist = df.groupby(["sample_idx", "order", "severity"])[["psnr", "ssim"]].nunique()
    assert (grouped_dist["psnr"] == 1).all(), "PSNR varies across models for the same image-condition!"
    assert (grouped_dist["ssim"] == 1).all(), "SSIM varies across models for the same image-condition!"
    print(f"   [PASS] Deterministic transformation verified: identical PSNR and SSIM across models for every condition")

    # Check non-commutativity: distinct orders produce different distortion
    mean_dist_by_order = df[df["order"] != "clean"].groupby("order")[["psnr", "ssim"]].mean()
    psnr_spread = mean_dist_by_order["psnr"].max() - mean_dist_by_order["psnr"].min()
    ssim_spread = mean_dist_by_order["ssim"].max() - mean_dist_by_order["ssim"].min()
    assert psnr_spread > 1.0, f"Expected order-dependent PSNR divergence, got spread {psnr_spread:.2f} dB"
    assert ssim_spread > 0.05, f"Expected order-dependent SSIM divergence, got spread {ssim_spread:.3f}"
    print(f"   [PASS] Operator non-commutativity confirmed: mean PSNR spread = {psnr_spread:.2f} dB, SSIM spread = {ssim_spread:.3f}")

    # Verify summary files exist
    required_files = [
        "order_effects_summary.csv",
        "order_effects_leaderboard.csv",
        "order_pairwise_significance.csv",
        "order_omnibus_tests.json",
        "distortion_ancova.csv",
        "benchmark_report.json",
        "manifest.json"
    ]
    for rf in required_files:
        fpath = os.path.join(full_dir, rf)
        assert os.path.exists(fpath), f"Missing required artifact: {rf}"
        assert os.path.getsize(fpath) > 0, f"Artifact is empty: {rf}"
        print(f"   [PASS] Artifact verified: {rf} ({os.path.getsize(fpath):,} bytes)")

    print(f"\nALL FULL BENCHMARK CHECKS PASSED FOR {full_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="results/order_effect_study/full/outputs/")
    parser.add_argument("--images", type=int, default=3925)
    args = parser.parse_args()
    validate_full_benchmark(args.dir, args.images)
