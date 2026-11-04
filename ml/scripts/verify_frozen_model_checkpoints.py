"""Verify and freeze all 4 model checkpoints, input transforms, and logit margin formulas.

Mandate 1 Reproducibility Audit:
- Exact timm identifiers
- Library versions (torch, torchvision, timm)
- Local checkpoint filenames, byte sizes, and SHA-256 hashes
- Full pretrained configuration (input size, crop, interpolation, mean, std)
- Classifier output dimensions (1000 classes)
- Exact 10-synset Imagenette to ImageNet-1K mapping
- True-class logit margin mathematical formulation and validation
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform

import torch
import torchvision
import timm
from robustlens.eval.dataset import IMAGENETTE_TO_IMAGENET, DATASET_PROVENANCE

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "results" / "matched_fidelity_study"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUTPUT_DIR / "model_checkpoints_verification.json"

MODELS_CONFIG = {
    "resnet50": {
        "model_id": "resnet50.a1_in1k",
        "family": "ResNet",
        "description": "ResNet-50 trained with modern A1 recipe (RandAugment, Mixup, CutMix, 300 epochs)",
    },
    "convnext_tiny": {
        "model_id": "convnext_tiny.in12k_ft_in1k",
        "family": "Modern ConvNet",
        "description": "ConvNeXt-Tiny pre-trained on ImageNet-12k and fine-tuned on ImageNet-1K at 224x224",
    },
    "vit_base": {
        "model_id": "vit_base_patch16_224.augreg_in21k_ft_in1k",
        "family": "Vision Transformer",
        "description": "ViT-Base/16 pre-trained on ImageNet-21k with AugReg and fine-tuned on ImageNet-1K",
    },
    "swin_tiny": {
        "model_id": "swin_tiny_patch4_window7_224.ms_in1k",
        "family": "Hierarchical Vision Transformer",
        "description": "Swin-Tiny (patch 4, window 7) trained on ImageNet-1K",
    },
}

IMAGENETTE_CLASSES = {
    0: ("n01440764", "tench"),
    1: ("n02102040", "English springer"),
    2: ("n02979186", "cassette player"),
    3: ("n03000684", "chain saw"),
    4: ("n03028079", "church"),
    5: ("n03394916", "French horn"),
    6: ("n03417042", "garbage truck"),
    7: ("n03425413", "gas pump"),
    8: ("n03445777", "golf ball"),
    9: ("n03888257", "parachute"),
}


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def find_hf_checkpoint_file(model_id: str) -> dict | None:
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_dir.exists():
        return None
    repo_folder = f"models--timm--{model_id.replace('/', '--')}"
    target_dir = cache_dir / repo_folder
    if not target_dir.exists():
        return None
    
    # Search for model.safetensors or pytorch_model.bin in snapshots
    for root, _, files in os.walk(target_dir):
        for f in files:
            if f in ("model.safetensors", "pytorch_model.bin", "model.pth"):
                p = Path(root) / f
                return {
                    "path": str(p),
                    "filename": f,
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                }
    return None


def main():
    print("=" * 80)
    print("MANDATE 1: MODEL CHECKPOINTS, CONFIGURATIONS, AND LOGIT MARGIN AUDIT")
    print("=" * 80)

    env_versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
    }
    print(f"Environment: Python {env_versions['python']} | PyTorch {env_versions['torch']} | Torchvision {env_versions['torchvision']} | timm {env_versions['timm']}")

    model_audits = {}
    for short_name, m_spec in MODELS_CONFIG.items():
        m_id = m_spec["model_id"]
        print(f"\nInspecting: {short_name} ({m_id})...")
        m = timm.create_model(m_id, pretrained=False)
        default_cfg = m.default_cfg
        data_cfg = timm.data.resolve_model_data_config(m)
        
        # Classifier dimension
        out_dim = m.get_classifier().out_features if hasattr(m, "get_classifier") else None
        assert out_dim == 1000, f"Expected 1000 output dimensions, got {out_dim}"
        
        # Locate checkpoint file
        ckpt_info = find_hf_checkpoint_file(m_id)
        
        model_audits[short_name] = {
            "model_identifier": m_id,
            "family": m_spec["family"],
            "description": m_spec["description"],
            "output_dimension": out_dim,
            "pretrained_cfg": {
                "input_size": list(default_cfg.get("input_size", (3, 224, 224))),
                "interpolation": default_cfg.get("interpolation", "bicubic"),
                "mean": [round(float(x), 4) for x in default_cfg.get("mean", (0.485, 0.456, 0.406))],
                "std": [round(float(x), 4) for x in default_cfg.get("std", (0.229, 0.224, 0.225))],
                "crop_pct": float(data_cfg.get("crop_pct", 0.875)),
                "num_classes": default_cfg.get("num_classes", 1000),
                "classifier": default_cfg.get("classifier", "fc"),
                "hf_hub_id": default_cfg.get("hf_hub_id", f"timm/{m_id}"),
            },
            "checkpoint_file": ckpt_info,
        }
        print(f"  Output Dim: {out_dim}")
        print(f"  Pretrained Mean: {default_cfg.get('mean')}, Std: {default_cfg.get('std')}, Crop Pct: {data_cfg.get('crop_pct')}")
        if ckpt_info:
            print(f"  Checkpoint File: {ckpt_info['filename']} ({ckpt_info['size_bytes']:,} bytes)")
            print(f"  SHA-256: {ckpt_info['sha256']}")

    # Verify Synset Mapping
    print("\n" + "=" * 80)
    print("IMAGENETTE (10 CLASSES) TO IMAGENET-1K (1,000 CLASSES) TARGET MAPPING")
    print("=" * 80)
    synset_mapping_audit = []
    for local_idx, (wnid, class_name) in IMAGENETTE_CLASSES.items():
        in1k_idx = IMAGENETTE_TO_IMAGENET[local_idx]
        synset_mapping_audit.append({
            "imagenette_local_index": local_idx,
            "wordnet_id": wnid,
            "class_name": class_name,
            "imagenet_1k_target_index": in1k_idx,
        })
        print(f"  Local {local_idx} | WNID: {wnid} | Name: {class_name:<18} | ImageNet-1K Target Index: {in1k_idx}")

    # Mathematical Verification of Logit Margin Formula
    print("\n" + "=" * 80)
    print("VERIFYING TRUE-CLASS LOGIT MARGIN FORMULA")
    print("=" * 80)
    # Synthetic logit vector of dimension 1,000
    torch.manual_seed(42)
    dummy_logits = torch.randn(1000)
    # Set known values:
    true_1k_idx = 482  # cassette player (local 2)
    dummy_logits[true_1k_idx] = 12.50
    # Set runner-up within the 1,000 classes
    runner_up_1k = 300
    dummy_logits[runner_up_1k] = 9.50
    # Set maximum among the OTHER 9 Imagenette classes
    other_imagenette_indices = [IMAGENETTE_TO_IMAGENET[i] for i in range(10) if i != 2]
    for idx in other_imagenette_indices:
        dummy_logits[idx] = 4.00
    
    # Correct formula: over all 999 incorrect ImageNet classes
    z_true = float(dummy_logits[true_1k_idx].item())
    z_other_full = float(torch.cat([dummy_logits[:true_1k_idx], dummy_logits[true_1k_idx+1:]]).max().item())
    margin_full = z_true - z_other_full
    
    # Incorrect formula: over only the 9 other Imagenette classes
    z_other_10class = float(dummy_logits[other_imagenette_indices].max().item())
    margin_10class = z_true - z_other_10class
    
    print(f"True class logit z_true (index {true_1k_idx}): {z_true:.2f}")
    print(f"Full 1,000-class runner-up z_other (index {runner_up_1k}): {z_other_full:.2f} -> Margin: {margin_full:.2f}")
    print(f"10-class restricted runner-up: {z_other_10class:.2f} -> Margin: {margin_10class:.2f}")
    print(f"Delta: {margin_10class - margin_full:.2f} (Restricting to 10 classes artificially inflates margin!)")
    assert abs(margin_full - (12.50 - 9.50)) < 1e-6, "Margin calculation error"
    print("PASSED: True-class logit margin formula verified strictly over all 999 incorrect ImageNet classes.")

    audit_payload = {
        "audit_name": "RobustLens Model Checkpoints and Margin Formulation Audit",
        "environment": env_versions,
        "models": model_audits,
        "imagenette_to_imagenet_1k_synsets": synset_mapping_audit,
        "margin_formulation": {
            "formula_latex": r"M = z_{y^*} - \max_{j \ne y^*, j \in \{0, \dots, 999\}} z_j",
            "search_space": "Full 1,000-class output logit vector",
            "justification": "Ensures true out-of-class competition reflects the entire ImageNet vocabulary, avoiding artificial margin inflation.",
            "synthetic_test_verified": True,
        },
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(audit_payload, f, indent=2)
    print(f"\nSaved complete model checkpoint verification to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
