"""Run robustness evaluation pipeline.

Evaluates pretrained vision models on clean and corrupted datasets,
saving per-image predictions, labels, correctness, confidences, and latencies.

Usage:
    uv run python scripts/run_eval.py --models resnet50 convnext_tiny vit_base swin_tiny \
        --subset-size 50 --severities 1 2 3 --output ../results/raw/
"""

import argparse
import datetime
import json
import os
import sys
import time
from typing import List, Dict, Any

import torch
from tqdm import tqdm

from robustlens.eval.engine import (
    MODEL_REGISTRY,
    resolve_model_key,
    build_model_and_transforms,
    evaluate_batch,
    get_imagenet_classes,
)
from robustlens.eval.dataset import ImagenetteDataset, get_imagenette_loader, DATASET_PROVENANCE
from robustlens.corruptions.transforms import get_corruption_transform, CORRUPTIONS



def parse_args():
    parser = argparse.ArgumentParser(
        description="Run robustness evaluation of pretrained vision models under distribution shift."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet50", "convnext_tiny", "vit_base", "swin_tiny"],
        help="Model architectures to evaluate (resnet50, convnext_tiny, vit_base, swin_tiny).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenette",
        help="Dataset name to evaluate on (default: imagenette).",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Number of images to evaluate from validation set (default: full set, ~3925 images).",
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Corruption severity levels (1-5).",
    )
    parser.add_argument(
        "--corruptions",
        nargs="+",
        default=["gaussian_noise", "defocus_blur", "jpeg_compression", "brightness", "contrast"],
        help="Corruption types to apply.",
    )
    parser.add_argument(
        "--include-clean",
        action="store_true",
        default=True,
        help="Always include clean (unperturbed) baseline evaluation (severity 0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda or cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sample selection and noise.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../results/raw/",
        help="Output directory to save evaluation metrics and predictions.",
    )
    return parser.parse_args()


def run_evaluation(args) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print("=" * 60)
    print("RobustLens Evaluation Pipeline")
    print("=" * 60)
    print(f"Device:       {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Models:       {args.models}")
    print(f"Corruptions:  {args.corruptions}")
    print(f"Severities:   {args.severities}")
    print(f"Subset Size:  {args.subset_size or 'Full validation split'}")
    print(f"Batch Size:   {args.batch_size}")
    print(f"Output Dir:   {args.output}")
    print("=" * 60)

    os.makedirs(args.output, exist_ok=True)
    classes = get_imagenet_classes()

    # Normalize requested models
    resolved_model_keys = [resolve_model_key(m) for m in args.models]

    # Pre-load dataset items into memory / cache using base pre-transform
    # We use a dummy transform first to load images cleanly
    print(f"\nLoading {args.dataset} validation set...")
    raw_dataset = ImagenetteDataset(split="validation")
    total_available = len(raw_dataset)

    sample_count = min(args.subset_size, total_available) if args.subset_size else total_available
    g = torch.Generator().manual_seed(args.seed)
    chosen_indices = torch.randperm(total_available, generator=g)[:sample_count].tolist()

    print(f"Selected {len(chosen_indices)} images deterministically with seed {args.seed}.")

    # Gather images and ground truth
    images_data = []
    for idx in chosen_indices:
        item = raw_dataset.dataset[idx]
        pil_img = item["image"].convert("RGB")
        orig_label = item["label"]
        imagenet_label = raw_dataset.IMAGENETTE_TO_IMAGENET[orig_label]
        images_data.append({
            "sample_idx": idx,
            "image": pil_img,
            "true_label": imagenet_label,
            "true_class_name": classes[imagenet_label],
            "orig_label": orig_label,
        })

    all_predictions: List[Dict[str, Any]] = []

    # Outer loop: model
    for model_key in resolved_model_keys:
        model_info = MODEL_REGISTRY[model_key]
        display_name = model_info["display_name"]
        print(f"\nEvaluating Model: {display_name} ({model_info['timm_id']})")
        
        model, pre_transform, post_transform = build_model_and_transforms(model_key, device)

        # Preprocess PIL images to [0, 1] tensors
        preprocessed_tensors = [pre_transform(d["image"]) for d in images_data]

        conditions = []
        if args.include_clean:
            conditions.append(("clean", 0))
        for corr in args.corruptions:
            for sev in args.severities:
                conditions.append((corr, sev))

        # Evaluate each condition
        for corr_name, sev in conditions:
            desc = f"  {display_name} | {corr_name} (s{sev})"
            
            # Apply corruption if not clean
            if corr_name == "clean":
                corrupted_tensors = preprocessed_tensors
            else:
                # Map name to transform function
                transform_name = "gaussian_blur" if corr_name == "defocus_blur" else corr_name
                corr_func = get_corruption_transform(transform_name, sev)
                corrupted_tensors = [corr_func(t.clone()) for t in preprocessed_tensors]

            # Batch inference
            pbar = tqdm(range(0, len(corrupted_tensors), args.batch_size), desc=desc, leave=False)
            for b_idx in pbar:
                batch_tensors = corrupted_tensors[b_idx : b_idx + args.batch_size]
                batch_meta = images_data[b_idx : b_idx + args.batch_size]

                batch_preds = evaluate_batch(
                    model=model,
                    tensors=batch_tensors,
                    post_transform=post_transform,
                    device=device,
                    classes=classes,
                )

                for meta, pred in zip(batch_meta, batch_preds):
                    is_correct = bool(pred["pred_label"] == meta["true_label"])
                    record = {
                        "sample_idx": meta["sample_idx"],
                        "model": model_key,
                        "model_name": display_name,
                        "model_family": model_info["family"],
                        "corruption": corr_name,
                        "severity": sev,
                        "true_label": meta["true_label"],
                        "true_class_name": meta["true_class_name"],
                        "pred_label": pred["pred_label"],
                        "pred_class_name": pred["pred_class_name"],
                        "confidence": pred["confidence"],
                        "correct": is_correct,
                        "top_5_classes": pred["top_5_classes"],
                        "top_5_probs": pred["top_5_probs"],
                        "latency_ms": round(pred["latency_ms"], 2),
                    }
                    all_predictions.append(record)

    # Save per-image predictions
    preds_file = os.path.join(args.output, "all_predictions.json")
    with open(preds_file, "w") as f:
        json.dump(all_predictions, f, indent=2)
    print(f"\nSaved {len(all_predictions)} prediction records to {preds_file}")

    # Save manifest
    manifest = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "benchmark_status": f"Pilot — N={len(images_data)}" if len(images_data) < 500 else f"Full Benchmark — N={len(images_data)}",
        "is_pilot": bool(len(images_data) < 500),
        "num_independent_images": len(images_data),
        "num_prediction_records": len(all_predictions),
        "conditions_per_image": len(all_predictions) // (len(images_data) * len(resolved_model_keys)) if images_data and resolved_model_keys else 1,
        "dataset": DATASET_PROVENANCE,
        "seed": args.seed,
        "device": str(device),
        "models": [MODEL_REGISTRY[k] for k in resolved_model_keys],
        "corruptions": args.corruptions,
        "severities": args.severities,
        "total_records": len(all_predictions),
        "predictions_file": preds_file,
    }
    manifest_file = os.path.join(args.output, "manifest.json")
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved experiment manifest to {manifest_file}")

    return manifest



def main():
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
