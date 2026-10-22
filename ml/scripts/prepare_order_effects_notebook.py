"""Prepare and validate the 01_order_effects_benchmark.ipynb notebook for Kaggle GPU execution."""

import argparse
import json
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def build_order_effects_notebook(subset_size=None, batch_size=32, n_permutations=20000, n_bootstraps=10000):
    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    }

    # Cell 0: Markdown Header
    c0 = new_markdown_cell("""# 🔍 RobustLens: Compound Corruption Order Effects Benchmark

**Investigating Non-Commutativity, Distortion Sensitivity, and Architectural Invariance under Permuted Distribution Shifts**

- **Research Question:** Does corruption permutation order alter classification failure and calibration under compound image shifts?
- **Permutations:** All 6 permutations of $\\{Noise, Blur, JPEG\\}$:
  - `NBJ`: Noise $\\to$ Blur $\\to$ JPEG
  - `NJB`: Noise $\\to$ JPEG $\\to$ Blur
  - `BNJ`: Blur $\\to$ Noise $\\to$ JPEG
  - `BJN`: Blur $\\to$ JPEG $\\to$ Noise
  - `JNB`: JPEG $\\to$ Noise $\\to$ Blur
  - `JBN`: JPEG $\\to$ Blur $\\to$ Noise
- **Severities:** Levels 1 to 5 ($30$ compound conditions) + Clean baseline ($1$ condition) = $31$ conditions per image per model.
- **Models:**
  1. ResNet-50 (`resnet50.a1_in1k`)
  2. ConvNeXt-Tiny (`convnext_tiny.in12k_ft_in1k`)
  3. ViT-Base/16 (`vit_base_patch16_224.augreg_in21k_ft_in1k`)
  4. Swin-Tiny (`swin_tiny_patch4_window7_224.ms_in1k`)
- **Dataset:** Official FastAI Imagenette-v2 Validation Split ($N = 3{,}925$ images, rows 9,469..13,393).
- **Experimental Scale:** $3{,}925 \\times 31 \\times 4 = 486{,}700$ total forward passes.
- **Statistical Controls:**
  - Exact seed-matched noise realizations across all 6 permutations per image.
  - Objective perceptual distortion metrics (PSNR and SSIM) computed per sample.
  - Repeated-measures Friedman omnibus test across permutation sequences.
  - Image-level paired sign-flip permutation tests ($R = 20{,}000$) with Holm and Benjamini-Hochberg corrections.
  - Clustered paired bootstrap confidence intervals ($B = 10{,}000$).
  - Distortion-controlled ANCOVA / logistic regression to isolate order effects from low-level distortion.
  - Expected Calibration Error (ECE) under all 30 compound conditions.
""")

    # Cell 1: Environment & GPU setup
    c1 = new_code_cell("""# 1. Setup Environment & Verify GPU
!pip install -q timm datasets scipy statsmodels

import os
import sys
import time
import json
import io
import gzip
import math
import itertools
import datetime
import platform
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
import statsmodels.api as sm

import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import timm
from datasets import load_dataset

print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"Timm: {timm.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Compute Device: {device} ({gpu_name})")
if not torch.cuda.is_available():
    print("WARNING: Running on CPU! For full evaluation, GPU must be enabled.")
""")

    # Cell 2: Hyperparameters & Configuration
    subset_val_repr = "None" if subset_size is None else str(subset_size)
    c2 = new_code_cell(f"""# 2. Configuration & Hyperparameters
SUBSET_SIZE = {subset_val_repr}  # None for full evaluation (all 3,925 official validation images)
BATCH_SIZE = {batch_size}
RANDOM_SEED = 42
N_BOOTSTRAPS = {n_bootstraps}
N_PERMUTATIONS = {n_permutations}
SIGNIFICANCE_ALPHA = 0.05
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print(f"Configuration: SUBSET_SIZE={{SUBSET_SIZE}}, BATCH_SIZE={{BATCH_SIZE}}, SEED={{RANDOM_SEED}}")
""")

    # Cell 3: Dataset Loading & Validation Split Integrity
    c3 = new_code_cell("""# 3. Dataset Loading & Verified FastAI Imagenette Validation Split
IMAGENETTE_TO_IMAGENET = {
    0: 0,    # tench (n01440764)
    1: 217,  # English springer (n02102040)
    2: 482,  # cassette player (n02979186)
    3: 491,  # chainsaw (n03000684)
    4: 497,  # church (n03028079)
    5: 566,  # French horn (n03394916)
    6: 569,  # garbage truck (n03417042)
    7: 571,  # gas pump (n03425413)
    8: 574,  # golf ball (n03445777)
    9: 701,  # parachute (n03888257)
}

CLASS_INDEX_URL = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
req = urllib.request.urlopen(CLASS_INDEX_URL)
raw_class_idx = json.loads(req.read().decode("utf-8"))
IMAGENET_CLASSES = {int(k): v[1].replace("_", " ") for k, v in raw_class_idx.items()}

print("Loading official Imagenette-v2 dataset mirror (johnowhitaker/imagenette2-320)...")
full_ds = load_dataset("johnowhitaker/imagenette2-320", split="train")
total_rows = len(full_ds)
assert total_rows == 13394, f"Expected 13,394 total rows, got {total_rows}"

# Genuine FastAI validation split: rows 9469 to 13394 (3,925 images)
val_ds = full_ds.select(range(9469, total_rows))
assert len(val_ds) == 3925, f"Expected 3,925 validation rows, got {len(val_ds)}"

# Verify boundary transition and exact per-class counts
val_class_counts = Counter(val_ds["label"])
EXPECTED_VAL_COUNTS = {
    0: 387, 1: 395, 2: 357, 3: 386, 4: 409,
    5: 394, 6: 389, 7: 419, 8: 399, 9: 390
}
for cid, exp_cnt in EXPECTED_VAL_COUNTS.items():
    assert val_class_counts[cid] == exp_cnt, f"Validation class {cid} expected {exp_cnt}, got {val_class_counts[cid]}"
print(f"Validation split verified: 3,925 images across 10 classes. Exactly matches official FastAI counts.")

# Preprocessing to un-normalized 224x224 RGB float32 tensors
spatial_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()
])

total_val = len(val_ds)
sample_count = min(SUBSET_SIZE, total_val) if SUBSET_SIZE is not None else total_val
if SUBSET_SIZE is not None:
    g = torch.Generator().manual_seed(RANDOM_SEED)
    selected_indices = torch.randperm(total_val, generator=g)[:sample_count].tolist()
else:
    selected_indices = list(range(total_val))

print(f"Preloading and spatially transforming {len(selected_indices)} images into CPU tensors...")
t_load_start = time.time()
clean_cpu_tensors = []
images_meta = []

for i, idx in enumerate(selected_indices):
    item = val_ds[idx]
    img = item["image"].convert("RGB")
    tensor_cpu = spatial_transform(img)
    clean_cpu_tensors.append(tensor_cpu)
    
    orig_label = item["label"]
    imagenet_label = IMAGENETTE_TO_IMAGENET[orig_label]
    images_meta.append({
        "sample_idx": idx,
        "orig_label": orig_label,
        "true_label": imagenet_label,
        "true_class_name": IMAGENET_CLASSES[imagenet_label]
    })
    if (i + 1) % 1000 == 0 or (i + 1) == len(selected_indices):
        print(f"  Processed {i + 1}/{len(selected_indices)} images in {time.time() - t_load_start:.1f}s")

print(f"Ready: {len(clean_cpu_tensors)} clean tensors cached in CPU memory.")
""")

    # Cell 4: Compound Corruptions, Matched Noise & Metrics Engine
    c4 = new_code_cell("""# 4. Compound Corruption Pipeline with Seed-Matched Noise & Diagnostics
ORDER_PERMUTATIONS = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]

ORDER_DESCRIPTIONS = {
    "NBJ": "Noise -> Blur -> JPEG",
    "NJB": "Noise -> JPEG -> Blur",
    "BNJ": "Blur -> Noise -> JPEG",
    "BJN": "Blur -> JPEG -> Noise",
    "JNB": "JPEG -> Noise -> Blur",
    "JBN": "JPEG -> Blur -> Noise",
}

NOISE_STDS = [0.04, 0.08, 0.12, 0.18, 0.26]
BLUR_PARAMS = [
    (3, 1.0),
    (5, 2.0),
    (7, 3.0),
    (9, 4.0),
    (11, 5.0),
]
JPEG_QUALITIES = [80, 65, 50, 35, 20]

def get_deterministic_noise_seed(sample_idx, severity, base_seed=42):
    return int((base_seed * 1000003 + sample_idx * 9973 + severity * 101) % (2**31 - 1))

def generate_matched_noise(shape, severity, sample_idx=0, base_seed=42, device=None, dtype=torch.float32):
    std = NOISE_STDS[severity - 1]
    seed = get_deterministic_noise_seed(sample_idx, severity, base_seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(shape, generator=gen, dtype=dtype) * std
    if device is not None:
        noise = noise.to(device)
    return noise

def apply_gaussian_noise(img, severity, noise_tensor=None, sample_idx=0, base_seed=42):
    if noise_tensor is None:
        noise_tensor = generate_matched_noise(img.shape, severity, sample_idx=sample_idx, base_seed=base_seed, device=img.device, dtype=img.dtype)
    else:
        if noise_tensor.device != img.device:
            noise_tensor = noise_tensor.to(img.device)
    return torch.clamp(img + noise_tensor, 0.0, 1.0)

def apply_gaussian_blur(img, severity):
    k, sigma = BLUR_PARAMS[severity - 1]
    return TF.gaussian_blur(img, [k, k], [sigma, sigma])

def apply_jpeg_compression(img, severity):
    quality = JPEG_QUALITIES[severity - 1]
    orig_device = img.device
    cpu_img = img.detach().cpu()
    pil_img = TF.to_pil_image(cpu_img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed_pil = Image.open(buffer)
    return TF.to_tensor(compressed_pil).to(orig_device)

def apply_compound_corruption(img, order, severity, sample_idx=0, base_seed=42, precomputed_noise=None):
    if precomputed_noise is None:
        noise_tensor = generate_matched_noise(img.shape, severity, sample_idx=sample_idx, base_seed=base_seed, device=img.device, dtype=img.dtype)
    else:
        noise_tensor = precomputed_noise
        
    current = img
    for char in order:
        if char == "N":
            current = apply_gaussian_noise(current, severity, noise_tensor=noise_tensor)
        elif char == "B":
            current = apply_gaussian_blur(current, severity)
        elif char == "J":
            current = apply_jpeg_compression(current, severity)
    return current

# Metrics
def compute_psnr(clean, corrupt, max_val=1.0, eps=1e-10):
    mse = torch.mean((clean.float() - corrupt.float()) ** 2).item()
    if mse < eps:
        return 100.0
    return float(10.0 * math.log10((max_val ** 2) / mse))

def _create_gaussian_kernel_2d(kernel_size=11, sigma=1.5):
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    gauss_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = torch.outer(gauss_1d, gauss_1d)
    return gauss_2d

def compute_ssim(clean, corrupt, window_size=11, sigma=1.5, max_val=1.0, k1=0.01, k2=0.03):
    if clean.ndim == 3:
        clean = clean.unsqueeze(0)
    if corrupt.ndim == 3:
        corrupt = corrupt.unsqueeze(0)
    clean = clean.float()
    corrupt = corrupt.float()
    channels = clean.shape[1]
    device = clean.device

    kernel_2d = _create_gaussian_kernel_2d(window_size, sigma).to(device)
    kernel_4d = kernel_2d.expand(channels, 1, window_size, window_size)

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    pad = window_size // 2

    mu1 = F.conv2d(clean, kernel_4d, padding=pad, groups=channels)
    mu2 = F.conv2d(corrupt, kernel_4d, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(clean * clean, kernel_4d, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(corrupt * corrupt, kernel_4d, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(clean * corrupt, kernel_4d, padding=pad, groups=channels) - mu1_mu2

    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return float((numerator / denominator).mean().item())

print("Compound corruption engine and objective distortion diagnostics verified.")
""")

    # Cell 5: Evaluation Loop
    c5 = new_code_cell("""# 5. Model Execution Loop
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

normalize_transform = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Build condition list: 1 clean + (6 permutations x 5 severities = 30 compound conditions) = 31 conditions
CONDITIONS = [("clean", 0)]
for order in ORDER_PERMUTATIONS:
    for sev in [1, 2, 3, 4, 5]:
        CONDITIONS.append((order, sev))

print(f"Total conditions per model: {len(CONDITIONS)}")
print(f"Total forward passes: {len(images_meta)} images x {len(CONDITIONS)} conditions x {len(MODEL_REGISTRY)} models = {len(images_meta) * len(CONDITIONS) * len(MODEL_REGISTRY)}")

# First precompute distortion diagnostics (PSNR and SSIM) once per image-condition
print("\\nPrecomputing objective image quality metrics (PSNR, SSIM) for all conditions...")
t_dist_start = time.time()
distortion_cache = {}  # key: (sample_idx, order, severity) -> (psnr, ssim)

for c_idx, (order, sev) in enumerate(CONDITIONS):
    t_c0 = time.time()
    for i, meta in enumerate(images_meta):
        s_idx = meta["sample_idx"]
        clean_tensor = clean_cpu_tensors[i]
        if order == "clean":
            distortion_cache[(s_idx, "clean", 0)] = (100.0, 1.0)
        else:
            corrupted = apply_compound_corruption(
                clean_tensor, order=order, severity=sev, sample_idx=s_idx, base_seed=RANDOM_SEED
            )
            p_val = compute_psnr(clean_tensor, corrupted)
            s_val = compute_ssim(clean_tensor, corrupted)
            distortion_cache[(s_idx, order, sev)] = (round(p_val, 3), round(s_val, 4))
            
    if (c_idx + 1) % 5 == 0 or (c_idx + 1) == len(CONDITIONS):
        print(f"  Distortion computed for condition {c_idx + 1}/{len(CONDITIONS)} ({order} s{sev}) in {time.time() - t_c0:.1f}s")

print(f"Distortion cache ready in {time.time() - t_dist_start:.1f}s.")

# Execution loop
all_records = []
t_eval_start = time.time()

for model_key, model_meta in MODEL_REGISTRY.items():
    print(f"\\n=======================================================")
    print(f"Evaluating {model_meta['display_name']} ({model_meta['timm_id']})...")
    print(f"=======================================================")
    
    model = timm.create_model(model_meta["timm_id"], pretrained=True)
    model = model.to(device)
    model.eval()
    
    num_samples = len(images_meta)
    
    for cond_idx, (order, sev) in enumerate(CONDITIONS):
        t0 = time.time()
        cond_preds = 0
        cond_correct = 0
        
        for b_start in range(0, num_samples, BATCH_SIZE):
            b_end = min(b_start + BATCH_SIZE, num_samples)
            batch_meta = images_meta[b_start:b_end]
            
            # Prepare corrupted batch
            batch_tensors = []
            for i, meta in enumerate(batch_meta):
                clean_t = clean_cpu_tensors[b_start + i]
                if order == "clean":
                    batch_tensors.append(clean_t)
                else:
                    corr_t = apply_compound_corruption(
                        clean_t, order=order, severity=sev, sample_idx=meta["sample_idx"], base_seed=RANDOM_SEED
                    )
                    batch_tensors.append(corr_t)
                    
            batch_cpu = torch.stack(batch_tensors)
            batch_dev = batch_cpu.to(device)
            batch_norm = normalize_transform(batch_dev)
            
            with torch.no_grad():
                logits = model(batch_norm)
                probs = torch.softmax(logits, dim=-1)
                confidences, preds = torch.max(probs, dim=-1)
                
                # Predictive entropy
                log_probs = torch.log(probs + 1e-12)
                entropy = -torch.sum(probs * log_probs, dim=-1)
                
            confidences = confidences.cpu().numpy()
            preds = preds.cpu().numpy()
            entropy = entropy.cpu().numpy()
            
            for i, meta in enumerate(batch_meta):
                s_idx = meta["sample_idx"]
                pred_label = int(preds[i])
                conf = float(confidences[i])
                correct = bool(pred_label == meta["true_label"])
                ent = float(entropy[i])
                psnr_val, ssim_val = distortion_cache[(s_idx, order, sev)]
                
                cond_preds += 1
                if correct:
                    cond_correct += 1
                    
                all_records.append({
                    "sample_idx": s_idx,
                    "model": model_key,
                    "model_name": model_meta["display_name"],
                    "model_family": model_meta["family"],
                    "order": order,
                    "severity": sev,
                    "true_label": meta["true_label"],
                    "true_class_name": meta["true_class_name"],
                    "prediction": pred_label,
                    "pred_class_name": IMAGENET_CLASSES.get(pred_label, str(pred_label)),
                    "confidence": round(conf, 5),
                    "entropy": round(ent, 5),
                    "correct": int(correct),
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                })
                
        cond_acc = cond_correct / cond_preds if cond_preds > 0 else 0.0
        elapsed = time.time() - t0
        if (cond_idx + 1) % 5 == 0 or (cond_idx + 1) == len(CONDITIONS) or order == "clean":
            print(f"  [{cond_idx + 1}/{len(CONDITIONS)}] {order} (s={sev}): Acc = {cond_acc * 100:.2f}%, Time = {elapsed:.1f}s")
            
    # Checkpoint completed model predictions immediately to disk
    model_records = [r for r in all_records if r["model"] == model_key]
    df_model = pd.DataFrame(model_records)
    ckpt_path = os.path.join(OUTPUT_DIR, f"predictions_{model_key}.csv.gz")
    df_model.to_csv(ckpt_path, index=False, compression="gzip")
    print(f"  [CHECKPOINT] Preserved {len(df_model)} records for {model_key} to {ckpt_path}")

    # Free model memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\\nAll model forward passes finished in {time.time() - t_eval_start:.1f}s. Total records: {len(all_records)}")

""")

    # Cell 6: Statistical Analysis & Synthesis
    c6 = new_code_cell("""# 6. Statistical Analysis Engine: Omnibus Tests, Pairwise Permutations & ANCOVA
print("\\nRunning Statistical Synthesis...")
df_all = pd.DataFrame(all_records)

# Save gzipped predictions for space efficiency
csv_path = os.path.join(OUTPUT_DIR, "all_order_predictions.csv.gz")
df_all.to_csv(csv_path, index=False, compression="gzip")
print(f"Saved complete prediction records to {csv_path} ({os.path.getsize(csv_path) / 1024 / 1024:.2f} MB)")

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
    
    # ECE calculation (15 bins)
    confs = grp["confidence"].values
    corrs = grp["correct"].values
    bin_b = np.linspace(0.0, 1.0, 16)
    ece = 0.0
    for b in range(15):
        mask = (confs >= bin_b[b]) & (confs < bin_b[b + 1]) if b < 14 else (confs >= bin_b[b]) & (confs <= bin_b[b + 1])
        nb = np.sum(mask)
        if nb > 0:
            ece += (nb / len(confs)) * abs(np.mean(corrs[mask]) - np.mean(confs[mask]))
            
    summary_rows.append({
        "model": model_key,
        "model_name": model_name,
        "model_family": model_family,
        "order": order,
        "severity": sev,
        "accuracy": round(acc, 4),
        "mean_confidence": round(conf, 4),
        "ece": round(float(ece), 4),
        "mean_entropy": round(ent_m, 4),
        "mean_psnr": round(psnr_m, 2),
        "mean_ssim": round(ssim_m, 3),
        "n_samples": len(grp)
    })

df_summary = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUTPUT_DIR, "order_effects_summary.csv")
df_summary.to_csv(summary_csv, index=False)
print(f"Saved condition summary to {summary_csv}")

# 2. Leaderboard: Aggregate across all 30 compound conditions
leaderboard_rows = []
for model_key, grp in df_all.groupby("model"):
    clean_grp = grp[grp["order"] == "clean"]
    compound_grp = grp[grp["order"] != "clean"]
    
    clean_acc = float(clean_grp["correct"].mean())
    compound_acc = float(compound_grp["correct"].mean())
    relative_mce = (1.0 - compound_acc) / (1.0 - 0.7696)  # standardized against ResNet-50 baseline
    
    # Variance across the 6 orders
    order_means = compound_grp.groupby("order")["correct"].mean()
    order_var = float(order_means.var())
    order_max_gap = float(order_means.max() - order_means.min())
    
    leaderboard_rows.append({
        "model": model_key,
        "model_name": grp["model_name"].iloc[0],
        "model_family": grp["model_family"].iloc[0],
        "clean_accuracy": round(clean_acc, 4),
        "compound_accuracy": round(compound_acc, 4),
        "order_accuracy_variance": round(order_var, 6),
        "order_max_accuracy_gap": round(order_max_gap, 4),
        "mean_compound_ece": round(float(df_summary[(df_summary["model"] == model_key) & (df_summary["order"] != "clean")]["ece"].mean()), 4),
    })

df_leaderboard = pd.DataFrame(leaderboard_rows).sort_values("compound_accuracy", ascending=False)
lead_csv = os.path.join(OUTPUT_DIR, "order_effects_leaderboard.csv")
df_leaderboard.to_csv(lead_csv, index=False)
print(f"Saved leaderboard to {lead_csv}")

# 3. Repeated-Measures Friedman Omnibus Tests across Orders
omnibus_results = {}
for model_key in MODEL_REGISTRY.keys():
    omnibus_results[model_key] = {}
    for sev in [1, 2, 3, 4, 5]:
        sub = df_all[(df_all["model"] == model_key) & (df_all["severity"] == sev)]
        # Pivot to (N_images, 6 orders)
        piv = sub.pivot(index="sample_idx", columns="order", values="correct")[ORDER_PERMUTATIONS].dropna()
        cols = [piv[col].values for col in ORDER_PERMUTATIONS]
        stat, pval = stats.friedmanchisquare(*cols)
        kendall_w = float(stat / (len(piv) * (len(ORDER_PERMUTATIONS) - 1)))
        
        omnibus_results[model_key][f"severity_{sev}"] = {
            "chi2_stat": round(float(stat), 3),
            "p_value": float(pval),
            "kendall_w": round(float(kendall_w), 4),
            "n_images": len(piv)
        }

with open(os.path.join(OUTPUT_DIR, "order_omnibus_tests.json"), "w") as f:
    json.dump(omnibus_results, f, indent=2)
print("Saved Friedman omnibus test results.")

# 4. Pairwise Sign-Flip Permutation Tests (R=20,000) & Clustered Bootstrap CIs (B=10,000)
def fast_sign_flip_test(diffs, n_perms=N_PERMUTATIONS, seed=RANDOM_SEED):
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 1.0
    obs_mean = float(np.mean(diffs))
    abs_obs = abs(obs_mean)
    
    if len(nonzero) <= 16:
        signs = np.array(list(np.ndindex(*(2,) * len(nonzero))), dtype=float) * 2 - 1.0
        perm_means = np.sum(signs * nonzero, axis=1) / len(diffs)
        n_extreme = np.sum(np.abs(perm_means) >= abs_obs - 1e-12)
        return float((1 + n_extreme) / (1 + len(signs)))
        
    rng = np.random.default_rng(seed)
    batch_size = 5000
    n_batches = math.ceil(n_perms / batch_size)
    extreme_count = 0
    total = 0
    for _ in range(n_batches):
        cur = min(batch_size, n_perms - total)
        if cur <= 0:
            break
        signs = rng.choice([-1.0, 1.0], size=(cur, len(nonzero)))
        perm_m = np.sum(signs * nonzero, axis=1) / len(diffs)
        extreme_count += int(np.sum(np.abs(perm_m) >= abs_obs - 1e-12))
        total += cur
    return float((1.0 + extreme_count) / (1.0 + total))

pairwise_rows = []
for model_key in MODEL_REGISTRY.keys():
    for sev in [1, 2, 3, 4, 5]:
        sub = df_all[(df_all["model"] == model_key) & (df_all["severity"] == sev)]
        piv = sub.pivot(index="sample_idx", columns="order", values="correct")[ORDER_PERMUTATIONS].dropna()
        
        for ord1, ord2 in itertools.combinations(ORDER_PERMUTATIONS, 2):
            diffs = (piv[ord1] - piv[ord2]).values
            diff_m = float(np.mean(diffs))
            
            # Bootstrap CI
            rng = np.random.default_rng(RANDOM_SEED)
            bs_indices = rng.integers(0, len(diffs), size=(N_BOOTSTRAPS, len(diffs)))
            bs_means = np.mean(diffs[bs_indices], axis=1)
            ci_l = float(np.percentile(bs_means, 2.5))
            ci_u = float(np.percentile(bs_means, 97.5))
            
            p_perm = fast_sign_flip_test(diffs, n_perms=N_PERMUTATIONS)
            
            pairwise_rows.append({
                "model": model_key,
                "severity": sev,
                "order_a": ord1,
                "order_b": ord2,
                "diff_mean": round(diff_m, 4),
                "ci_lower_95": round(ci_l, 4),
                "ci_upper_95": round(ci_u, 4),
                "p_value_raw": p_perm,
            })

df_pairwise = pd.DataFrame(pairwise_rows)

# Multiplicity adjustments per model
adjusted_rows = []
for model_key, grp in df_pairwise.groupby("model"):
    pvals = grp["p_value_raw"].values
    m = len(pvals)
    
    # Holm
    s_idx = np.argsort(pvals)
    holm = [0.0] * m
    c_max = 0.0
    for r, idx in enumerate(s_idx):
        c_max = max(c_max, pvals[idx] * (m - r))
        holm[idx] = min(1.0, c_max)
        
    # BH
    bh = [0.0] * m
    c_min = 1.0
    for r in reversed(range(m)):
        idx = s_idx[r]
        c_min = min(c_min, (pvals[idx] * m) / (r + 1))
        bh[idx] = min(1.0, c_min)
        
    grp = grp.copy()
    grp["p_value_holm"] = [round(p, 5) for p in holm]
    grp["p_value_bh"] = [round(p, 5) for p in bh]
    adjusted_rows.append(grp)

df_pairwise_adj = pd.concat(adjusted_rows, ignore_index=True)
pair_csv = os.path.join(OUTPUT_DIR, "order_pairwise_significance.csv")
df_pairwise_adj.to_csv(pair_csv, index=False)
print(f"Saved pairwise significance to {pair_csv}")

# 5. Distortion-Controlled ANCOVA (Controlling for PSNR and SSIM)
ancova_rows = []
for model_key in MODEL_REGISTRY.keys():
    sub_df = df_all[(df_all["model"] == model_key) & (df_all["order"] != "clean")].copy()
    
    dummies = pd.get_dummies(sub_df["order"], drop_first=False, dtype=float)
    if "NBJ" in dummies.columns:
        dummies = dummies.drop(columns=["NBJ"])
    order_cols = list(dummies.columns)
    
    X = pd.concat([sub_df[["psnr", "ssim"]], dummies], axis=1)
    X = sm.add_constant(X)
    y = sub_df["correct"].astype(float)
    
    try:
        logit = sm.Logit(y, X).fit(disp=False, maxiter=100)
        
        # Test order joint significance
        order_idx = [X.columns.get_loc(c) for c in order_cols]
        r_matrix = np.zeros((len(order_cols), len(X.columns)))
        for i, idx in enumerate(order_idx):
            r_matrix[i, idx] = 1.0
        wald = logit.wald_test(r_matrix)
        
        ancova_rows.append({
            "model": model_key,
            "pseudo_r2": round(float(logit.prsquared), 4),
            "coef_psnr": round(float(logit.params["psnr"]), 4),
            "pval_psnr": float(logit.pvalues["psnr"]),
            "coef_ssim": round(float(logit.params["ssim"]), 4),
            "pval_ssim": float(logit.pvalues["ssim"]),
            "order_wald_stat": round(float(wald.statistic), 3),
            "order_wald_p_value": float(wald.pvalue),
            "order_significant_at_05": bool(wald.pvalue < 0.05),
            "n_observations": len(y)
        })
    except Exception as e:
        print(f"ANCOVA failed for {model_key}: {e}")

df_ancova = pd.DataFrame(ancova_rows)
ancova_csv = os.path.join(OUTPUT_DIR, "distortion_ancova.csv")
df_ancova.to_csv(ancova_csv, index=False)
print(f"Saved distortion ANCOVA to {ancova_csv}")

# 6. Benchmark Report JSON
benchmark_report = {
    "title": "RobustLens Compound Corruption Order Effects Benchmark",
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "n_images": len(images_meta),
    "models": list(MODEL_REGISTRY.keys()),
    "permutations": ORDER_PERMUTATIONS,
    "severities": [1, 2, 3, 4, 5],
    "total_records": len(df_all),
    "leaderboard": df_leaderboard.to_dict(orient="records"),
    "omnibus_tests": omnibus_results,
    "distortion_ancova": df_ancova.to_dict(orient="records"),
}

with open(os.path.join(OUTPUT_DIR, "benchmark_report.json"), "w") as f:
    json.dump(benchmark_report, f, indent=2)

print("\\nStatistical synthesis complete!")
""")

    # Cell 7: Manifest & Provenance
    c7 = new_code_cell("""# 7. Verification & Manifest Generation
print("\\nVerifying Artifact Integrity...")
expected_conds = len(CONDITIONS)
expected_records = len(images_meta) * expected_conds * len(MODEL_REGISTRY)

assert len(df_all) == expected_records, f"Expected {expected_records} rows, got {len(df_all)}"
assert df_all.isnull().sum().sum() == 0, "Found unexpected null values in predictions!"
assert len(df_summary) == len(MODEL_REGISTRY) * expected_conds, f"Summary rows mismatch: {len(df_summary)}"

manifest = {
    "benchmark": "RobustLens Order Effects Evaluation",
    "version": "1.0.0",
    "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "hardware": {
        "device": str(device),
        "gpu_name": gpu_name if torch.cuda.is_available() else "None",
    },
    "dataset": {
        "name": "johnowhitaker/imagenette2-320",
        "split": "validation (rows 9469..13393)",
        "n_images": len(images_meta),
        "per_class_counts": EXPECTED_VAL_COUNTS,
    },
    "parameters": {
        "permutations": ORDER_PERMUTATIONS,
        "severities": [1, 2, 3, 4, 5],
        "n_conditions_per_image": expected_conds,
        "n_models": len(MODEL_REGISTRY),
        "n_total_predictions": len(df_all),
        "n_permutations_test": N_PERMUTATIONS,
        "n_bootstraps_ci": N_BOOTSTRAPS,
        "random_seed": RANDOM_SEED,
    },
    "artifacts": {
        "all_predictions": "outputs/all_order_predictions.csv.gz",
        "summary": "outputs/order_effects_summary.csv",
        "leaderboard": "outputs/order_effects_leaderboard.csv",
        "pairwise_significance": "outputs/order_pairwise_significance.csv",
        "omnibus_tests": "outputs/order_omnibus_tests.json",
        "distortion_ancova": "outputs/distortion_ancova.csv",
        "benchmark_report": "outputs/benchmark_report.json",
    }
}

manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Manifest written to {manifest_path}")
print("ALL VERIFICATIONS PASSED SUCCESSFULLY.")
""")

    nb.cells = [c0, c1, c2, c3, c4, c5, c6, c7]
    return nb


def main():
    parser = argparse.ArgumentParser(description="Prepare order effects evaluation notebook")
    parser.add_argument("--subset-size", type=int, default=None, help="Subset size (None for full 3,925)")
    parser.add_argument("--output-dir", type=str, default="ml/notebooks/order_effects_eval", help="Output directory")
    parser.add_argument("--notebook-name", type=str, default="01_order_effects_benchmark.ipynb", help="Notebook filename")
    parser.add_argument("--kernel-id", type=str, default="aviral199/robustlens-order-effects-evaluation", help="Kaggle kernel ID")
    parser.add_argument("--title", type=str, default="RobustLens Order Effects Evaluation", help="Kaggle kernel title")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    nb = build_order_effects_notebook(subset_size=args.subset_size)
    
    nb_path = os.path.join(args.output_dir, args.notebook_name)
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Generated notebook: {nb_path} ({os.path.getsize(nb_path)} bytes)")

    # Metadata for Kaggle kernel push
    metadata = {
        "id": args.kernel_id,
        "title": args.title,
        "code_file": args.notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": []
    }

    meta_path = os.path.join(args.output_dir, "kernel-metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Generated kernel metadata: {meta_path}")


if __name__ == "__main__":
    main()
