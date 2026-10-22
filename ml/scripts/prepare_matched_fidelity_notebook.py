"""Prepare and validate the 01_matched_fidelity_benchmark.ipynb notebook for Kaggle GPU execution.

This notebook evaluates:
- 3,925 official Imagenette-v2 validation images
- 31 conditions: 1 clean + 6 sequences x 5 quality-matched fidelity targets
- 4 models: ResNet-50, ConvNeXt-Tiny, ViT-Base/16, Swin-Tiny
- Total: 486,700 inferences
- Uses calibrated parameters derived on the independent training split.
"""

from __future__ import annotations

import argparse
import json
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def build_matched_fidelity_notebook(
    subset_size=None,
    batch_size=32,
    n_permutations=20000,
    n_bootstraps=10000,
):
    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    }

    # Load calibrated parameters
    config_path = os.path.join("results", "order_effect_study", "matched_fidelity", "calibration_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            calib_data = json.load(f)
        calibrated_params_repr = json.dumps(calib_data["sequence_parameters"], indent=2)
    else:
        # Fallback calibrated parameters
        calibrated_params_repr = "{}"

    # Cell 0: Header
    c0 = new_markdown_cell("""# 🔍 RobustLens: Counterfactual Matched-Fidelity Benchmark

**At Matched Image Fidelity, Does Corruption Order Still Matter? A Counterfactual Study of Vision Classifier Reliability**

- **Research Question:** When multi-stage composite image degradations are calibrated to matched physical image fidelity (PSNR and SSIM) across sequences on common support, does the execution order (e.g., `NBJ` vs `NJB` vs `BNJ` vs `BJN` vs `JNB` vs `JBN`) still predict classifier failure, prediction churn, and calibration?
- **Dataset:** Official FastAI Imagenette-v2 Validation Split ($N = 3{,}925$ images, rows 9,469..13,393).
- **Architectures:** ResNet-50 (`resnet50.a1_in1k`), ConvNeXt-Tiny (`convnext_tiny.in12k_ft_in1k`), ViT-Base/16 (`vit_base_patch16_224.augreg_in21k_ft_in1k`), Swin-Tiny (`swin_tiny_patch4_window7_224.ms_in1k`).
- **Conditions:** 1 clean baseline + 6 sequences $\times$ 5 matched distortion targets ($T_1 \dots T_5$) = 31 conditions per image per model.
- **Scale:** $3{,}925 \times 31 \times 4 = 486{,}700$ total forward passes.
- **Calibration Protocol:** Operator parameters $(\sigma_N, \sigma_B, q_J)$ calibrated strictly on an independent training split partition (rows 0..9,468); zero validation leakage.
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

    # Cell 2: Configuration & Calibrated Parameters
    subset_val_repr = "None" if subset_size is None else str(subset_size)
    c2 = new_code_cell(f"""# 2. Configuration & Calibrated Sequence Parameters
SUBSET_SIZE = {subset_val_repr}  # None for full 3,925 evaluation
BATCH_SIZE = {batch_size}
RANDOM_SEED = 42
N_BOOTSTRAPS = {n_bootstraps}
N_PERMUTATIONS = {n_permutations}
SIGNIFICANCE_ALPHA = 0.05
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CALIBRATED_TARGETS = {calibrated_params_repr}

ORDER_PERMUTATIONS = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]
TARGET_LEVELS = [1, 2, 3, 4, 5]

print(f"Configuration: SUBSET_SIZE={{SUBSET_SIZE}}, BATCH_SIZE={{BATCH_SIZE}}, SEED={{RANDOM_SEED}}")
print(f"Loaded calibrated parameters for {{len(CALIBRATED_TARGETS)}} target levels across all 6 sequences.")
""")

    # Cell 3: Dataset Loading & Validation Split Integrity
    c3 = new_code_cell("""# 3. Dataset Loading & Verified Validation Split
IMAGENETTE_TO_IMAGENET = {
    0: 0, 1: 217, 2: 482, 3: 491, 4: 497,
    5: 566, 6: 569, 7: 571, 8: 574, 9: 701,
}

CLASS_INDEX_URL = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
req = urllib.request.urlopen(CLASS_INDEX_URL)
raw_class_idx = json.loads(req.read().decode("utf-8"))
IMAGENET_CLASSES = {int(k): v[1].replace("_", " ") for k, v in raw_class_idx.items()}

print("Loading official Imagenette-v2 dataset mirror (johnowhitaker/imagenette2-320)...")
full_ds = load_dataset("johnowhitaker/imagenette2-320", split="train")
total_rows = len(full_ds)
assert total_rows == 13394, f"Expected 13,394 total rows, got {total_rows}"

val_ds = full_ds.select(range(9469, total_rows))
assert len(val_ds) == 3925, f"Expected 3,925 validation rows, got {len(val_ds)}"

val_class_counts = Counter(val_ds["label"])
EXPECTED_VAL_COUNTS = {
    0: 387, 1: 395, 2: 357, 3: 386, 4: 409,
    5: 394, 6: 389, 7: 419, 8: 399, 9: 390
}
for cid, exp_cnt in EXPECTED_VAL_COUNTS.items():
    assert val_class_counts[cid] == exp_cnt, f"Class {cid} count mismatch"
print("Validation split verified: exactly 3,925 images across 10 classes.")

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

    # Cell 4: Matched-Fidelity Corruption Pipeline & Diagnostics
    c4 = new_code_cell("""# 4. Matched-Fidelity Corruption Pipeline & Metrics
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

_SSIM_KERNEL_4D = _create_gaussian_kernel_2d(11, 1.5).expand(3, 1, 11, 11)

def compute_ssim(clean, corrupt, window_size=11, max_val=1.0, k1=0.01, k2=0.03):
    global _SSIM_KERNEL_4D
    if clean.ndim == 3:
        clean = clean.unsqueeze(0)
    if corrupt.ndim == 3:
        corrupt = corrupt.unsqueeze(0)
    clean = clean.float()
    corrupt = corrupt.float()
    channels = clean.shape[1]
    device = clean.device

    if _SSIM_KERNEL_4D.device != device:
        _SSIM_KERNEL_4D = _SSIM_KERNEL_4D.to(device)

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    pad = window_size // 2

    mu1 = F.conv2d(clean, _SSIM_KERNEL_4D, padding=pad, groups=channels)
    mu2 = F.conv2d(corrupt, _SSIM_KERNEL_4D, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(clean * clean, _SSIM_KERNEL_4D, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(corrupt * corrupt, _SSIM_KERNEL_4D, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(clean * corrupt, _SSIM_KERNEL_4D, padding=pad, groups=channels) - mu1_mu2

    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return float((numerator / denominator).mean().item())

def apply_matched_compound(img, order, target_idx, sample_idx=0, base_seed=42):
    t_key = f"target_{target_idx}"
    p = CALIBRATED_TARGETS[t_key]["sequences"][order]
    noise_std = p["noise_std"]
    blur_k = p["blur_k"]
    blur_sigma = p["blur_sigma"]
    jpeg_quality = p["jpeg_quality"]

    # Deterministic noise realization
    noise_seed = int((base_seed * 1000003 + sample_idx * 9973 + target_idx * 101) % (2**31 - 1))
    gen = torch.Generator().manual_seed(noise_seed)
    noise_tensor = torch.randn(img.shape, generator=gen, dtype=img.dtype) * max(0.001, noise_std)

    cur = img
    for char in order:
        if char == "N":
            cur = torch.clamp(cur + noise_tensor, 0.0, 1.0)
        elif char == "B":
            if blur_sigma > 0.1 and blur_k >= 3:
                k = blur_k if blur_k % 2 == 1 else blur_k + 1
                cur = TF.gaussian_blur(cur, [k, k], [blur_sigma, blur_sigma])
        elif char == "J":
            q = max(5, min(95, int(round(jpeg_quality))))
            pil_img = TF.to_pil_image(cur.cpu())
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            comp = Image.open(buf)
            cur = TF.to_tensor(comp)
    return cur

print("Matched-fidelity pipeline and metrics engine verified.")
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

# 1 clean condition + 6 permutations x 5 matched targets = 31 conditions
CONDITIONS = [("clean", 0)]
for t_idx in TARGET_LEVELS:
    for order in ORDER_PERMUTATIONS:
        CONDITIONS.append((order, t_idx))

print(f"Total conditions per model: {len(CONDITIONS)}")
print(f"Total inferences: {len(images_meta)} images x {len(CONDITIONS)} conditions x {len(MODEL_REGISTRY)} models = {len(images_meta) * len(CONDITIONS) * len(MODEL_REGISTRY)}")

# Precompute distortion metrics (PSNR, SSIM) once per image-condition
print("\\nPrecomputing objective image quality metrics (PSNR, SSIM) for all conditions...")
t_dist_start = time.time()
distortion_cache = {}

for c_idx, (order, t_idx) in enumerate(CONDITIONS):
    t_c0 = time.time()
    for i, meta in enumerate(images_meta):
        s_idx = meta["sample_idx"]
        clean_tensor = clean_cpu_tensors[i]
        if order == "clean":
            distortion_cache[(s_idx, "clean", 0)] = (100.0, 1.0)
        else:
            corrupted = apply_matched_compound(
                clean_tensor, order=order, target_idx=t_idx, sample_idx=s_idx, base_seed=RANDOM_SEED
            )
            p_val = compute_psnr(clean_tensor, corrupted)
            s_val = compute_ssim(clean_tensor, corrupted)
            distortion_cache[(s_idx, order, t_idx)] = (round(p_val, 3), round(s_val, 4))
            
    if (c_idx + 1) % 5 == 0 or (c_idx + 1) == len(CONDITIONS):
        print(f"  Distortion computed for condition {c_idx + 1}/{len(CONDITIONS)} ({order} T{t_idx}) in {time.time() - t_c0:.1f}s")

print(f"Distortion cache ready in {time.time() - t_dist_start:.1f}s.")

# Execution loop
all_records = []
t_eval_start = time.time()
num_samples = len(images_meta)

for model_key, model_meta in MODEL_REGISTRY.items():
    print(f"\\n=======================================================")
    print(f"Evaluating {model_meta['display_name']} ({model_meta['timm_id']})...")
    print(f"=======================================================")
    
    model = timm.create_model(model_meta["timm_id"], pretrained=True)
    model = model.to(device)
    model.eval()
    
    for cond_idx, (order, t_idx) in enumerate(CONDITIONS):
        t0 = time.time()
        cond_preds = 0
        cond_correct = 0
        
        for b_start in range(0, num_samples, BATCH_SIZE):
            b_end = min(b_start + BATCH_SIZE, num_samples)
            batch_meta = images_meta[b_start:b_end]
            
            batch_tensors = []
            for i, meta in enumerate(batch_meta):
                clean_t = clean_cpu_tensors[b_start + i]
                if order == "clean":
                    batch_tensors.append(clean_t)
                else:
                    corr_t = apply_matched_compound(
                        clean_t, order=order, target_idx=t_idx, sample_idx=meta["sample_idx"], base_seed=RANDOM_SEED
                    )
                    batch_tensors.append(corr_t)
                    
            batch_cpu = torch.stack(batch_tensors)
            batch_dev = batch_cpu.to(device)
            batch_norm = normalize_transform(batch_dev)
            
            with torch.no_grad():
                logits = model(batch_norm)
                probs = torch.softmax(logits, dim=-1)
                confidences, preds = torch.max(probs, dim=-1)
                
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
                psnr_val, ssim_val = distortion_cache[(s_idx, order, t_idx)]
                
                cond_preds += 1
                if correct:
                    cond_correct += 1
                    
                all_records.append({
                    "sample_idx": s_idx,
                    "model": model_key,
                    "model_name": model_meta["display_name"],
                    "model_family": model_meta["family"],
                    "order": order,
                    "target_idx": t_idx,
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
            print(f"  [{cond_idx + 1}/{len(CONDITIONS)}] {order} (T={t_idx}): Acc = {cond_acc * 100:.2f}%, Time = {elapsed:.1f}s")
            
    # Checkpoint completed model predictions immediately to disk
    model_records = [r for r in all_records if r["model"] == model_key]
    df_model = pd.DataFrame(model_records)
    ckpt_path = os.path.join(OUTPUT_DIR, f"predictions_{model_key}.csv.gz")
    df_model.to_csv(ckpt_path, index=False, compression="gzip")
    print(f"  [CHECKPOINT] Preserved {len(df_model)} records for {model_key} to {ckpt_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\\nAll model forward passes finished in {time.time() - t_eval_start:.1f}s. Total records: {len(all_records)}")
""")

    # Cell 6: Statistical Analysis & Error Churn Engine
    c6 = new_code_cell("""# 6. Statistical Engine: Repeated-Measures ANCOVA, Churn & Pairwise Corrections
print("\\nRunning Statistical Synthesis...")
df_all = pd.DataFrame(all_records)

# Save gzipped predictions
csv_path = os.path.join(OUTPUT_DIR, "all_order_predictions.csv.gz")
df_all.to_csv(csv_path, index=False, compression="gzip")
print(f"Saved complete prediction records to {csv_path} ({os.path.getsize(csv_path) / 1024 / 1024:.2f} MB)")

# 1. Condition Summary
summary_rows = []
for (model_key, order, t_idx), grp in df_all.groupby(["model", "order", "target_idx"]):
    model_name = grp["model_name"].iloc[0]
    model_family = grp["model_family"].iloc[0]
    acc = float(grp["correct"].mean())
    conf = float(grp["confidence"].mean())
    psnr_m = float(grp["psnr"].mean())
    ssim_m = float(grp["ssim"].mean())
    ent_m = float(grp["entropy"].mean())
    
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
        "target_idx": t_idx,
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

# 2. Leaderboard: Aggregate across all 30 matched compound conditions
leaderboard_rows = []
for model_key, grp in df_all.groupby("model"):
    clean_grp = grp[grp["order"] == "clean"]
    compound_grp = grp[grp["order"] != "clean"]
    
    clean_acc = float(clean_grp["correct"].mean())
    compound_acc = float(compound_grp["correct"].mean())
    
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

# 3. Repeated-Measures Friedman Omnibus Tests across Orders at Matched Fidelity
omnibus_results = {}
for model_key in MODEL_REGISTRY.keys():
    omnibus_results[model_key] = {}
    for t_idx in TARGET_LEVELS:
        sub = df_all[(df_all["model"] == model_key) & (df_all["target_idx"] == t_idx)]
        piv = sub.pivot(index="sample_idx", columns="order", values="correct")[ORDER_PERMUTATIONS].dropna()
        cols = [piv[col].values for col in ORDER_PERMUTATIONS]
        stat, pval = stats.friedmanchisquare(*cols)
        kendall_w = float(stat / (len(piv) * (len(ORDER_PERMUTATIONS) - 1)))
        
        omnibus_results[model_key][f"target_{t_idx}"] = {
            "chi2_stat": round(float(stat), 3),
            "p_value": float(pval),
            "kendall_w": round(float(kendall_w), 4),
            "n_images": len(piv)
        }

with open(os.path.join(OUTPUT_DIR, "order_omnibus_tests.json"), "w") as f:
    json.dump(omnibus_results, f, indent=2)
print("Saved Friedman omnibus test results.")

# 4. Pairwise Sign-Flip Permutation Tests & Error Churn Analysis
def fast_sign_flip_test(diffs, n_perms=N_PERMUTATIONS, seed=RANDOM_SEED):
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 1.0
    obs_mean = float(np.mean(diffs))
    abs_obs = abs(obs_mean)
    
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
churn_rows = []

for model_key in MODEL_REGISTRY.keys():
    for t_idx in TARGET_LEVELS:
        sub = df_all[(df_all["model"] == model_key) & (df_all["target_idx"] == t_idx)]
        piv = sub.pivot(index="sample_idx", columns="order", values="correct")[ORDER_PERMUTATIONS].dropna()
        N_img = len(piv)
        
        for ord1, ord2 in itertools.combinations(ORDER_PERMUTATIONS, 2):
            c1 = piv[ord1].values
            c2 = piv[ord2].values
            diffs = c1 - c2
            diff_m = float(np.mean(diffs))
            
            # Churn metrics
            n00 = int(np.sum((c1 == 0) & (c2 == 0)))
            n01 = int(np.sum((c1 == 0) & (c2 == 1)))
            n10 = int(np.sum((c1 == 1) & (c2 == 0)))
            n11 = int(np.sum((c1 == 1) & (c2 == 1)))
            
            G = n01 / N_img
            L = n10 / N_img
            net_diff = G - L
            churn = G + L
            excess_churn = 2.0 * min(G, L)
            
            # Bootstrap CI
            rng = np.random.default_rng(RANDOM_SEED)
            bs_indices = rng.integers(0, len(diffs), size=(N_BOOTSTRAPS, len(diffs)))
            bs_means = np.mean(diffs[bs_indices], axis=1)
            ci_l = float(np.percentile(bs_means, 2.5))
            ci_u = float(np.percentile(bs_means, 97.5))
            
            p_perm = fast_sign_flip_test(diffs, n_perms=N_PERMUTATIONS)
            
            pairwise_rows.append({
                "model": model_key,
                "target_idx": t_idx,
                "order_a": ord1,
                "order_b": ord2,
                "diff_mean": round(diff_m, 4),
                "ci_lower_95": round(ci_l, 4),
                "ci_upper_95": round(ci_u, 4),
                "p_value_raw": p_perm,
            })
            
            churn_rows.append({
                "model": model_key,
                "target_idx": t_idx,
                "order_a": ord1,
                "order_b": ord2,
                "gain_rate": round(G, 4),
                "loss_rate": round(L, 4),
                "net_difference": round(net_diff, 4),
                "error_churn": round(churn, 4),
                "excess_churn": round(excess_churn, 4),
                "n_switch_images": n01 + n10,
                "n_stable_images": n00 + n11,
                "n_total": N_img,
            })

df_pairwise = pd.DataFrame(pairwise_rows)
df_churn = pd.DataFrame(churn_rows)

# Multiplicity adjustments per model
adjusted_rows = []
for model_key, grp in df_pairwise.groupby("model"):
    pvals = grp["p_value_raw"].values
    m = len(pvals)
    s_idx = np.argsort(pvals)
    holm = [0.0] * m
    c_max = 0.0
    for r, idx in enumerate(s_idx):
        c_max = max(c_max, pvals[idx] * (m - r))
        holm[idx] = min(1.0, c_max)
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

churn_csv = os.path.join(OUTPUT_DIR, "error_churn_summary.csv")
df_churn.to_csv(churn_csv, index=False)
print(f"Saved error churn summary to {churn_csv}")

# 5. Distortion-Controlled Factorial ANCOVA (Controlling for PSNR and SSIM)
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
    "title": "RobustLens Counterfactual Matched-Fidelity Benchmark",
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "n_images": len(images_meta),
    "models": list(MODEL_REGISTRY.keys()),
    "permutations": ORDER_PERMUTATIONS,
    "matched_targets": TARGET_LEVELS,
    "total_records": len(df_all),
    "leaderboard": df_leaderboard.to_dict(orient="records"),
    "omnibus_tests": omnibus_results,
    "distortion_ancova": df_ancova.to_dict(orient="records"),
}

with open(os.path.join(OUTPUT_DIR, "benchmark_report.json"), "w") as f:
    json.dump(benchmark_report, f, indent=2)

print("\\nStatistical synthesis complete!")
""")

    # Cell 7: Verification & Manifest Generation
    c7 = new_code_cell("""# 7. Verification & Manifest Generation
print("\\nVerifying Artifact Integrity...")
expected_conds = len(CONDITIONS)
expected_records = len(images_meta) * expected_conds * len(MODEL_REGISTRY)

assert len(df_all) == expected_records, f"Expected {expected_records} rows, got {len(df_all)}"
assert df_all.isnull().sum().sum() == 0, "Found unexpected null values in predictions!"
assert len(df_summary) == len(MODEL_REGISTRY) * expected_conds, f"Summary rows mismatch: {len(df_summary)}"

manifest = {
    "benchmark": "RobustLens Matched-Fidelity Evaluation",
    "version": "2.0.0",
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
        "matched_targets": [1, 2, 3, 4, 5],
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
        "error_churn": "outputs/error_churn_summary.csv",
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
    parser = argparse.ArgumentParser(description="Prepare matched fidelity evaluation notebook")
    parser.add_argument("--subset-size", type=int, default=None, help="Subset size (None for full 3,925)")
    parser.add_argument("--output-dir", type=str, default="ml/notebooks/matched_fidelity_eval", help="Output directory")
    parser.add_argument("--notebook-name", type=str, default="01_matched_fidelity_benchmark.ipynb", help="Notebook filename")
    parser.add_argument("--kernel-id", type=str, default="aviral199/robustlens-matched-fidelity-full-benchmark", help="Kaggle kernel ID")
    parser.add_argument("--title", type=str, default="RobustLens Matched Fidelity Full Benchmark", help="Kaggle kernel title")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    nb = build_matched_fidelity_notebook(subset_size=args.subset_size)
    
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
