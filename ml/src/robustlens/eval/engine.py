"""RobustLens Core Evaluation Engine.

Evaluates pretrained vision models on clean and genuinely corrupted datasets,
recording reproducible predictions, confidences, top-k distributions, and latencies.
"""

import time
import os
import urllib.request
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torchvision.transforms as T
import timm

from robustlens.corruptions.transforms import get_corruption_transform, CORRUPTIONS
from robustlens.eval.dataset import ImagenetteDataset, get_imagenette_loader, IMAGENETTE_TO_IMAGENET

# Canonical model identifiers in timm
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "resnet50": {
        "timm_id": "resnet50.a1_in1k",
        "display_name": "ResNet-50",
        "family": "CNN",
    },
    "convnext_tiny": {
        "timm_id": "convnext_tiny.in12k_ft_in1k",
        "display_name": "ConvNeXt-Tiny",
        "family": "Modern CNN",
    },
    "vit_base": {
        "timm_id": "vit_base_patch16_224.augreg_in21k_ft_in1k",
        "display_name": "ViT-Base/16",
        "family": "Transformer",
    },
    "swin_tiny": {
        "timm_id": "swin_tiny_patch4_window7_224.ms_in1k",
        "display_name": "Swin-Tiny",
        "family": "Hierarchical ViT",
    },
}

# Synonyms for flexible CLI parsing
MODEL_ALIASES: Dict[str, str] = {
    "resnet50": "resnet50",
    "resnet-50": "resnet50",
    "resnet": "resnet50",
    "convnext_tiny": "convnext_tiny",
    "convnext-tiny": "convnext_tiny",
    "convnext": "convnext_tiny",
    "vit_base_patch16_224": "vit_base",
    "vit_base": "vit_base",
    "vit-base": "vit_base",
    "vit": "vit_base",
    "swin_tiny_patch4_window7_224": "swin_tiny",
    "swin_tiny": "swin_tiny",
    "swin-tiny": "swin_tiny",
    "swin": "swin_tiny",
}


def get_imagenet_classes() -> List[str]:
    """Retrieve the standard 1000 ImageNet class labels."""
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RobustLens/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            classes = [line.decode("utf-8").strip() for line in response.readlines()]
            if len(classes) == 1000:
                return classes
    except Exception:
        pass
    # Fallback to generic class labels if offline
    return [f"class_{i}" for i in range(1000)]


def resolve_model_key(name: str) -> str:
    """Normalize model name to canonical key."""
    norm = name.strip().lower()
    if norm in MODEL_ALIASES:
        return MODEL_ALIASES[norm]
    for key, info in MODEL_REGISTRY.items():
        if norm == info["timm_id"].lower() or norm == info["display_name"].lower():
            return key
    raise ValueError(f"Unknown model: '{name}'. Supported choices: {list(MODEL_REGISTRY.keys())}")


def build_model_and_transforms(
    model_key: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, T.Compose, T.Normalize]:
    """Load pretrained model and separate spatial preprocessing from normalization.

    Separating preprocessing ensures corruptions are applied to un-normalized [0, 1]
    pixel tensors before channel normalization, matching standard ImageNet-C protocol.
    """
    key = resolve_model_key(model_key)
    timm_id = MODEL_REGISTRY[key]["timm_id"]

    model = timm.create_model(timm_id, pretrained=True).eval().to(device)
    cfg = timm.data.resolve_model_data_config(model)

    input_size = cfg.get("input_size", (3, 224, 224))
    img_size = input_size[1:] if len(input_size) == 3 else (224, 224)
    interpolation = cfg.get("interpolation", "bicubic")
    interp_mode = getattr(T.InterpolationMode, interpolation.upper(), T.InterpolationMode.BICUBIC)

    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))

    # Pre-transform: PIL -> 224x224 RGB Tensor in [0, 1]
    pre_transform = T.Compose([
        T.Resize(int(img_size[0] * 1.14), interpolation=interp_mode),
        T.CenterCrop(img_size),
        T.ToTensor(),
    ])

    # Post-transform: Normalize tensor with model-specific mean and std
    post_transform = T.Normalize(mean=mean, std=std)

    return model, pre_transform, post_transform


def evaluate_batch(
    model: torch.nn.Module,
    tensors: torch.Tensor,
    post_transform: T.Normalize,
    device: torch.device,
    classes: List[str],
) -> List[Dict[str, Any]]:
    """Run forward pass and extract top predictions and probabilities."""
    normalized = torch.stack([post_transform(t) for t in tensors]).to(device)
    
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(normalized)
        probs = torch.softmax(logits, dim=-1)
    latency_ms = (time.perf_counter() - t0) * 1000.0 / len(tensors)

    top5_probs, top5_indices = torch.topk(probs, k=5, dim=-1)
    
    top5_probs = top5_probs.cpu().numpy()
    top5_indices = top5_indices.cpu().numpy()

    batch_results = []
    for i in range(len(tensors)):
        pred_idx = int(top5_indices[i][0])
        conf = float(top5_probs[i][0])
        top_5_classes = [classes[idx] for idx in top5_indices[i]]
        top_5_p = [float(p) for p in top5_probs[i]]

        batch_results.append({
            "pred_label": pred_idx,
            "pred_class_name": classes[pred_idx],
            "confidence": conf,
            "top_5_classes": top_5_classes,
            "top_5_probs": top_5_p,
            "latency_ms": latency_ms,
        })
    return batch_results
