"""CLI script for running compound corruption order effects evaluation locally.

Usage:
    uv run python scripts/run_order_eval.py --subset-size 10 --output ../results/order_effect_study/local_pilot/
"""

from __future__ import annotations

import argparse
import datetime
import io
import itertools
import json
import math
import os
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import timm
from datasets import load_dataset
from PIL import Image

from robustlens.corruptions.order_effects import (
    ORDER_PERMUTATIONS,
    apply_compound_corruption,
    compute_psnr,
    compute_ssim,
    generate_matched_noise,
)
from robustlens.stats.order_analysis import (
    benjamini_hochberg_correction,
    clustered_paired_bootstrap_ci,
    compute_ece,
    compute_friedman_omnibus,
    distortion_controlled_ancova,
    holm_bonferroni_correction,
    paired_sign_flip_test,
)

IMAGENETTE_TO_IMAGENET = {
    0: 0, 1: 217, 2: 482, 3: 491, 4: 497,
    5: 566, 6: 569, 7: 571, 8: 574, 9: 701,
}

CLASS_INDEX_URL = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"

MODEL_REGISTRY = {
    "resnet50": {
        "timm_id": "resnet50.a1_in1k",
        "display_name": "ResNet-50",
        "family": "CNN"
    },
    "convnext_tiny": {
        "timm_id": "convnext_tiny.in12k_ft_in1k",
        "display_name": "ConvNeXt-Tiny",
        "family": "Modern CNN"
    },
    "vit_base": {
        "timm_id": "vit_base_patch16_224.augreg_in21k_ft_in1k",
        "display_name": "ViT-Base/16",
        "family": "Transformer"
    },
    "swin_tiny": {
        "timm_id": "swin_tiny_patch4_window7_224.ms_in1k",
        "display_name": "Swin-Tiny",
        "family": "Hierarchical ViT"
    }
}


def main():
    parser = argparse.ArgumentParser(description="Run compound corruption order effects evaluation.")
    parser.add_argument("--subset-size", type=int, default=10, help="Number of validation images to evaluate")
    parser.add_argument("--models", nargs="+", default=["resnet50", "convnext_tiny", "vit_base", "swin_tiny"])
    parser.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--output", type=str, default="results/order_effect_study/local_pilot/")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device)
    print(f"Running on device: {device}")
    print(f"Target models: {args.models}")
    print(f"Subset size: {args.subset_size} images")

    # Load dataset
    print("Loading Imagenette validation split...")
    full_ds = load_dataset("johnowhitaker/imagenette2-320", split="train")
    total_rows = len(full_ds)
    assert total_rows == 13394, f"Expected 13,394 total rows, got {total_rows}"
    val_ds = full_ds.select(range(9469, total_rows))
    assert len(val_ds) == 3925, f"Expected 3,925 validation rows, got {len(val_ds)}"

    sample_count = min(args.subset_size, len(val_ds)) if args.subset_size is not None else len(val_ds)
    g = torch.Generator().manual_seed(args.random_seed)
    selected_indices = torch.randperm(len(val_ds), generator=g)[:sample_count].tolist()

    spatial_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor()
    ])
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    clean_tensors = []
    images_meta = []
    for idx in selected_indices:
        item = val_ds[idx]
        img = item["image"].convert("RGB")
        clean_tensors.append(spatial_transform(img))
        orig_lbl = item["label"]
        images_meta.append({
            "sample_idx": idx,
            "orig_label": orig_lbl,
            "true_label": IMAGENETTE_TO_IMAGENET[orig_lbl]
        })

    # Conditions: clean (s0) + 6 permutations x 5 severities = 31 conditions
    conditions = [("clean", 0)]
    for order in ORDER_PERMUTATIONS:
        for sev in args.severities:
            conditions.append((order, sev))

    print(f"Total conditions: {len(conditions)}")
    expected_records = sample_count * len(conditions) * len(args.models)
    print(f"Expected prediction records: {sample_count} x {len(conditions)} x {len(args.models)} = {expected_records}")

    # Precompute distortion metrics
    print("Precomputing objective distortion metrics (PSNR, SSIM)...")
    distortion_cache = {}
    for order, sev in conditions:
        for i, meta in enumerate(images_meta):
            s_idx = meta["sample_idx"]
            clean_t = clean_tensors[i]
            if order == "clean":
                distortion_cache[(s_idx, "clean", 0)] = (100.0, 1.0)
            else:
                corr_t = apply_compound_corruption(clean_t, order, sev, sample_idx=s_idx, base_seed=args.random_seed)
                p = compute_psnr(clean_t, corr_t)
                s = compute_ssim(clean_t, corr_t)
                distortion_cache[(s_idx, order, sev)] = (round(p, 3), round(s, 4))

    all_records = []
    t_start = time.time()
    for m_key in args.models:
        if m_key not in MODEL_REGISTRY:
            continue
        m_meta = MODEL_REGISTRY[m_key]
        print(f"Evaluating {m_meta['display_name']} ({m_meta['timm_id']})...")
        model = timm.create_model(m_meta["timm_id"], pretrained=True)
        model = model.to(device)
        model.eval()

        for order, sev in conditions:
            for b_start in range(0, sample_count, args.batch_size):
                b_end = min(b_start + args.batch_size, sample_count)
                batch_meta = images_meta[b_start:b_end]
                batch_clean = clean_tensors[b_start:b_end]

                if order == "clean":
                    batch_corr = batch_clean
                else:
                    batch_corr = [
                        apply_compound_corruption(t, order, sev, sample_idx=m["sample_idx"], base_seed=args.random_seed)
                        for t, m in zip(batch_clean, batch_meta)
                    ]

                batch_dev = torch.stack(batch_corr).to(device)
                batch_norm = normalize(batch_dev)

                with torch.no_grad():
                    logits = model(batch_norm)
                    probs = torch.softmax(logits, dim=-1)
                    confidences, preds = torch.max(probs, dim=-1)

                    # Entropy
                    log_probs = torch.log(probs + 1e-12)
                    entropy = -torch.sum(probs * log_probs, dim=-1)

                confidences = confidences.cpu().numpy()
                preds = preds.cpu().numpy()
                entropy = entropy.cpu().numpy()

                for i, meta in enumerate(batch_meta):
                    s_idx = meta["sample_idx"]
                    p_label = int(preds[i])
                    conf = float(confidences[i])
                    corr = int(p_label == meta["true_label"])
                    ent = float(entropy[i])
                    p_val, s_val = distortion_cache[(s_idx, order, sev)]

                    all_records.append({
                        "sample_idx": s_idx,
                        "model": m_key,
                        "model_name": m_meta["display_name"],
                        "model_family": m_meta["family"],
                        "order": order,
                        "severity": sev,
                        "true_label": meta["true_label"],
                        "prediction": p_label,
                        "confidence": round(conf, 5),
                        "entropy": round(ent, 5),
                        "correct": corr,
                        "psnr": p_val,
                        "ssim": s_val,
                    })

        del model

    df_all = pd.DataFrame(all_records)
    csv_path = os.path.join(args.output, "all_order_predictions.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"Saved {len(df_all)} predictions to {csv_path}")

    # 1. Condition Summary
    summary_rows = []
    for (model_key, order, sev), grp in df_all.groupby(["model", "order", "severity"]):
        model_name = grp["model_name"].iloc[0]
        model_family = grp["model_family"].iloc[0]
        acc = float(grp["correct"].mean())
        conf = float(grp["confidence"].mean())
        psnr_m = float(grp["psnr"].mean())
        ssim_m = float(grp["ssim"].mean())
        ent_m = float(grp["entropy"].mean())
        ece = compute_ece(grp["confidence"].values, grp["correct"].values, n_bins=10)

        summary_rows.append({
            "model": model_key,
            "model_name": model_name,
            "model_family": model_family,
            "order": order,
            "severity": sev,
            "accuracy": round(acc, 4),
            "mean_confidence": round(conf, 4),
            "ece": round(ece, 4),
            "mean_entropy": round(ent_m, 4),
            "mean_psnr": round(psnr_m, 2),
            "mean_ssim": round(ssim_m, 3),
            "n_samples": len(grp)
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(args.output, "order_effects_summary.csv"), index=False)

    # 2. Leaderboard
    leaderboard_rows = []
    for model_key, grp in df_all.groupby("model"):
        clean_grp = grp[grp["order"] == "clean"]
        compound_grp = grp[grp["order"] != "clean"]
        clean_acc = float(clean_grp["correct"].mean())
        compound_acc = float(compound_grp["correct"].mean())
        order_means = compound_grp.groupby("order")["correct"].mean()

        leaderboard_rows.append({
            "model": model_key,
            "model_name": grp["model_name"].iloc[0],
            "model_family": grp["model_family"].iloc[0],
            "clean_accuracy": round(clean_acc, 4),
            "compound_accuracy": round(compound_acc, 4),
            "order_accuracy_variance": round(float(order_means.var()), 6),
            "order_max_accuracy_gap": round(float(order_means.max() - order_means.min()), 4),
            "mean_compound_ece": round(float(df_summary[(df_summary["model"] == model_key) & (df_summary["order"] != "clean")]["ece"].mean()), 4),
        })
    df_leaderboard = pd.DataFrame(leaderboard_rows).sort_values("compound_accuracy", ascending=False)
    df_leaderboard.to_csv(os.path.join(args.output, "order_effects_leaderboard.csv"), index=False)

    # 3. Repeated-Measures Friedman Omnibus Tests across Orders
    omnibus_results = {}
    for model_key in args.models:
        omnibus_results[model_key] = {}
        for sev in args.severities:
            sub = df_all[(df_all["model"] == model_key) & (df_all["severity"] == sev)]
            piv = sub.pivot(index="sample_idx", columns="order", values="correct")[ORDER_PERMUTATIONS].dropna()
            res_omni = compute_friedman_omnibus(piv.values)
            omnibus_results[model_key][f"severity_{sev}"] = res_omni

    with open(os.path.join(args.output, "order_omnibus_tests.json"), "w") as f:
        json.dump(omnibus_results, f, indent=2)

    # 4. Pairwise Sign-Flip Permutation Tests & Clustered Bootstrap CIs
    pairwise_rows = []
    for model_key in args.models:
        for sev in args.severities:
            sub = df_all[(df_all["model"] == model_key) & (df_all["severity"] == sev)]
            piv = sub.pivot(index="sample_idx", columns="order", values="correct")[ORDER_PERMUTATIONS].dropna()

            for ord1, ord2 in itertools.combinations(ORDER_PERMUTATIONS, 2):
                diffs = (piv[ord1] - piv[ord2]).values
                diff_m, ci_l, ci_u = clustered_paired_bootstrap_ci(piv[ord1].values, piv[ord2].values, n_resamples=1000, seed=args.random_seed)
                p_perm = paired_sign_flip_test(diffs, n_permutations=5000, seed=args.random_seed)

                pairwise_rows.append({
                    "model": model_key,
                    "severity": sev,
                    "order_a": ord1,
                    "order_b": ord2,
                    "diff_mean": round(float(diff_m), 4),
                    "ci_lower_95": round(float(ci_l), 4),
                    "ci_upper_95": round(float(ci_u), 4),
                    "p_value_raw": p_perm,
                })

    df_pairwise = pd.DataFrame(pairwise_rows)
    # Multiplicity corrections
    adjusted_rows = []
    for model_key, grp in df_pairwise.groupby("model"):
        pvals = grp["p_value_raw"].values.tolist()
        holm = holm_bonferroni_correction(pvals)
        bh = benjamini_hochberg_correction(pvals)
        grp = grp.copy()
        grp["p_value_holm"] = [round(p, 5) for p in holm]
        grp["p_value_bh"] = [round(p, 5) for p in bh]
        adjusted_rows.append(grp)
    df_pairwise_adj = pd.concat(adjusted_rows, ignore_index=True)
    df_pairwise_adj.to_csv(os.path.join(args.output, "order_pairwise_significance.csv"), index=False)

    # 5. Distortion-Controlled ANCOVA
    ancova_rows = []
    for model_key in args.models:
        sub_df = df_all[(df_all["model"] == model_key) & (df_all["order"] != "clean")].copy()
        res_ancova = distortion_controlled_ancova(sub_df, reference_order="NBJ")
        if res_ancova.get("status") == "success":
            ancova_rows.append({
                "model": model_key,
                "pseudo_r2": round(float(res_ancova["pseudo_r2"]), 4),
                "coef_psnr": round(float(res_ancova["coefficients"].get("psnr", 0)), 4),
                "pval_psnr": float(res_ancova["p_values"].get("psnr", 1)),
                "coef_ssim": round(float(res_ancova["coefficients"].get("ssim", 0)), 4),
                "pval_ssim": float(res_ancova["p_values"].get("ssim", 1)),
                "order_wald_stat": round(float(res_ancova["order_joint_wald_stat"]), 3),
                "order_wald_p_value": float(res_ancova["order_joint_p_value"]),
                "order_significant_at_05": bool(res_ancova["order_joint_p_value"] < 0.05),
                "n_observations": res_ancova["n_obs"]
            })
    df_ancova = pd.DataFrame(ancova_rows)
    df_ancova.to_csv(os.path.join(args.output, "distortion_ancova.csv"), index=False)

    # 6. Manifest
    total_time = time.time() - t_start
    manifest = {
        "benchmark": "RobustLens Order Effects Local Fallback Pilot",
        "version": "1.0.0",
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "runtime_seconds": round(total_time, 2),
        "hardware": {"device": str(device)},
        "dataset": {
            "name": "johnowhitaker/imagenette2-320",
            "split": "validation (rows 9469..13393)",
            "n_images": sample_count,
        },
        "parameters": {
            "permutations": ORDER_PERMUTATIONS,
            "severities": args.severities,
            "n_conditions_per_image": len(conditions),
            "n_models": len(args.models),
            "n_total_predictions": len(df_all),
            "random_seed": args.random_seed,
        },
        "artifacts": {
            "all_predictions": "all_order_predictions.csv",
            "summary": "order_effects_summary.csv",
            "leaderboard": "order_effects_leaderboard.csv",
            "pairwise_significance": "order_pairwise_significance.csv",
            "omnibus_tests": "order_omnibus_tests.json",
            "distortion_ancova": "distortion_ancova.csv",
        }
    }
    with open(os.path.join(args.output, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Local pilot evaluation successfully completed in {total_time:.1f}s.")


if __name__ == "__main__":
    main()
