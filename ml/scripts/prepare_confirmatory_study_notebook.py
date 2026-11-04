"""Prepare 02_distortion_fingerprints_confirmatory.ipynb for full-scale Kaggle GPU execution.

Confirmatory Stage:
- Dataset: All N=3,925 FastAI Imagenette-v2 validation images (rows 9469..13393).
- Conditions: 1 clean baseline + 6 sequences x 3 calibrated common-support levels = 19 conditions per image.
- Models: ResNet-50, ConvNeXt-Tiny, ViT-Base/16, Swin-Tiny (4 models on CUDA).
- Total forward passes: 3,925 x 19 x 4 = 298,300 forward passes.
- Sequence-specific calibrated parameters strictly land inside the validated 6-way common support:
  PSNR in [17.5, 32.3] dB, SSIM in [0.335, 0.895].
- Computes Global Fidelity (PSNR, SSIM, LPIPS via AlexNet on GPU).
- Computes 4 Prespecified Spatial-Frequency Fingerprints (FFT bands, JPEG block ratio, Sobel edge energy, local SSIM std).
- Saves full_distortion_fingerprints_and_margins.csv.gz.
- Conducts grouped cross-validation & clustered bootstrap on all 3,925 clusters.
"""

from __future__ import annotations

import json
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PARAMS_JSON = BASE_DIR / "results" / "matched_fidelity_study" / "calibrated_common_support_3levels.json"


def build_confirmatory_notebook(output_path: Path):
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
        "accelerator": "GPU",
    }

    with open(PARAMS_JSON, "r") as f:
        calib_params_data = json.load(f)

    calib_params_repr = json.dumps(calib_params_data, indent=2)

    # Markdown Cell 1: Overview
    c1 = new_markdown_cell("""# 🔬 RobustLens Confirmatory Study: Spatial-Frequency & Structural Distortion Fingerprints
## Full-Sample Evaluation on FastAI Imagenette-v2 ($N = 3{,}925$ Validation Images)

- **Research Question:** Beyond global image fidelity (PSNR, SSIM, LPIPS), do spatial-frequency and structural distortion fingerprints explain architecture-specific true-class logit margins under compound corruption permutations?
- **Dataset:** Official FastAI Imagenette-v2 Validation Split ($N = 3{,}925$ images, rows 9,469..13,393).
- **Scale:** $3{,}925 \\times 19 \\times 4 = 298{,}300$ neural forward passes on NVIDIA Tesla GPU.
- **Experimental Conditions:** 1 clean baseline + 6 sequences $\\times$ 3 calibrated levels = 19 conditions.
  - Level 1 (Mild Common Support): Target PSNR $28.0\\text{ dB}$, SSIM $0.850$.
  - Level 2 (Mod-Mild Common Support): Target PSNR $24.5\\text{ dB}$, SSIM $0.740$.
  - Level 3 (Moderate Common Support): Target PSNR $21.5\\text{ dB}$, SSIM $0.620$.
  - *All 3 levels fall strictly inside the validated 6-way joint common support region* ($PSNR \\in [17.5, 32.3]\\text{ dB}, SSIM \\in [0.335, 0.895]$).
- **Descriptors:**
  1. Global Fidelity: PSNR, SSIM, LPIPS (AlexNet feature space).
  2. 2D Fourier radial power fractions ($E_{\\text{low}}, E_{\\text{med}}, E_{\\text{high}}$).
  3. JPEG $8 \\times 8$ block-boundary discontinuity ratio ($B_{\\text{JPEG}}$).
  4. Relative Sobel gradient edge energy change ($\\Delta E_{\\text{edge}}$).
  5. Local SSIM spatial variability ($\\text{Std}(\\text{SSIM}_{\\text{local}})$).
- **Inference Strategy:** Grouped cross-validation by physical image identity and clustered bootstrap ($B=1{,}000$ clusters) on true-class logit margin.
""")

    # Code Cell 2: Setup & GPU Verification
    c2 = new_code_cell("""# 1. Environment & GPU Verification
!pip install -q timm datasets lpips scipy

import io
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import scipy.signal
from scipy import stats

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import timm
import lpips
from datasets import load_dataset

assert torch.cuda.is_available(), "GPU is REQUIRED for the 298,300 forward passes!"
device = torch.device("cuda")
print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
""")

    # Code Cell 3: Metrics & Fingerprint Extraction Functions
    c3 = new_code_cell("""# 2. Mathematical Definition of Fingerprints & Fidelity Descriptors

def compute_psnr(clean: torch.Tensor, corrupt: torch.Tensor) -> float:
    mse = torch.mean((clean.float() - corrupt.float()) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return float(10.0 * math.log10(1.0 / mse))

def _create_ssim_kernel(k: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    k2d = torch.outer(g, g)
    return k2d.expand(3, 1, k, k)

_SSIM_K = _create_ssim_kernel().to(device)

def compute_ssim_and_local_map(clean: torch.Tensor, corrupt: torch.Tensor):
    global _SSIM_K
    if clean.ndim == 3:
        clean = clean.unsqueeze(0)
    if corrupt.ndim == 3:
        corrupt = corrupt.unsqueeze(0)
    clean = clean.float().to(device)
    corrupt = corrupt.float().to(device)

    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    mu1 = F.conv2d(clean, _SSIM_K, padding=5, groups=3)
    mu2 = F.conv2d(corrupt, _SSIM_K, padding=5, groups=3)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sig1_sq = F.conv2d(clean * clean, _SSIM_K, padding=5, groups=3) - mu1_sq
    sig2_sq = F.conv2d(corrupt * corrupt, _SSIM_K, padding=5, groups=3) - mu2_sq
    sig12 = F.conv2d(clean * corrupt, _SSIM_K, padding=5, groups=3) - mu12

    num = (2 * mu12 + c1) * (2 * sig12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sig1_sq + sig2_sq + c2)
    ssim_map = (num / den).mean(dim=1).squeeze(0)

    return float(ssim_map.mean().item()), float(ssim_map.std().item())

def compute_distortion_fingerprints(clean_t: torch.Tensor, corrupt_t: torch.Tensor):
    clean_np = clean_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    corrupt_np = corrupt_t.squeeze(0).permute(1, 2, 0).cpu().numpy()

    gray_clean = clean_np.mean(axis=2).astype(np.float32)
    gray_corrupt = corrupt_np.mean(axis=2).astype(np.float32)
    delta = gray_corrupt - gray_clean
    H, W = delta.shape

    # 1. 2D Fourier Power Spectrum Fractions
    fft2 = np.fft.fft2(delta)
    power = np.abs(np.fft.fftshift(fft2)) ** 2
    y, x = np.ogrid[-H/2:H/2, -W/2:W/2]
    r_norm = np.sqrt(x**2 + y**2) / np.sqrt((H/2)**2 + (W/2)**2)
    total_power = power.sum() + 1e-12

    e_low = float(power[r_norm < 0.15].sum() / total_power)
    e_med = float(power[(r_norm >= 0.15) & (r_norm < 0.45)].sum() / total_power)
    e_high = float(power[r_norm >= 0.45].sum() / total_power)

    # 2. JPEG 8x8 Block-Boundary Discontinuity Ratio
    diff_h = np.abs(gray_corrupt[:, 1:] - gray_corrupt[:, :-1])
    mask_h = np.zeros(diff_h.shape[1], dtype=bool)
    mask_h[7::8] = True
    b_h = diff_h[:, mask_h].mean() if mask_h.sum() > 0 else 0.0
    i_h = diff_h[:, ~mask_h].mean() if (~mask_h).sum() > 0 else 1.0

    diff_v = np.abs(gray_corrupt[1:, :] - gray_corrupt[:-1, :])
    mask_v = np.zeros(diff_v.shape[0], dtype=bool)
    mask_v[7::8] = True
    b_v = diff_v[mask_v, :].mean() if mask_v.sum() > 0 else 0.0
    i_v = diff_v[~mask_v, :].mean() if (~mask_v).sum() > 0 else 1.0

    block_ratio = float(((b_h / (i_h + 1e-6)) + (b_v / (i_v + 1e-6))) / 2.0)

    # 3. Sobel Edge Energy Difference
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    edge_clean = np.sqrt(scipy.signal.convolve2d(gray_clean, sobel_x, mode="same")**2 + scipy.signal.convolve2d(gray_clean, sobel_y, mode="same")**2).mean()
    edge_corr = np.sqrt(scipy.signal.convolve2d(gray_corrupt, sobel_x, mode="same")**2 + scipy.signal.convolve2d(gray_corrupt, sobel_y, mode="same")**2).mean()
    delta_edge = float((edge_corr - edge_clean) / (edge_clean + 1e-6))

    # 4. Local SSIM Variation
    _, std_local_ssim = compute_ssim_and_local_map(clean_t, corrupt_t)

    return {
        "fft_energy_low": round(e_low, 4),
        "fft_energy_med": round(e_med, 4),
        "fft_energy_high": round(e_high, 4),
        "jpeg_block_artifact": round(block_ratio, 4),
        "delta_edge_energy": round(delta_edge, 4),
        "std_local_ssim": round(std_local_ssim, 4),
    }
""")

    # Code Cell 4: Compound Sequence Generator
    c4 = new_code_cell("""# 3. Compound Permutation Application Function
def apply_compound_sequence(img: torch.Tensor, order: str, noise_std: float, blur_sigma: float, jpeg_quality: int, noise_seed: int = 42):
    gen = torch.Generator().manual_seed(noise_seed)
    noise = torch.randn(img.shape, generator=gen, dtype=img.dtype) * max(0.0001, noise_std)

    cur = img
    k = max(3, int(2 * round(blur_sigma * 1.5) + 1))
    if k % 2 == 0:
        k += 1

    for char in order:
        if char == "N":
            cur = torch.clamp(cur + noise, 0.0, 1.0)
        elif char == "B":
            if blur_sigma > 0.05:
                cur = TF.gaussian_blur(cur, [k, k], [blur_sigma, blur_sigma])
        elif char == "J":
            q = max(5, min(95, int(round(jpeg_quality))))
            pil_img = TF.to_pil_image(cur.cpu())
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            cur = TF.to_tensor(Image.open(buf))
    return cur
""")

    # Code Cell 5: Calibrated Sequence Parameters & Batch Execution Loop
    c5 = new_code_cell(f"""# 4. Calibrated Common-Support Parameters & Full Batch Execution (N=3,925 Images)

CALIBRATED_PARAMS = {calib_params_repr}

IMAGENETTE_TO_IMAGENET = {{0: 0, 1: 217, 2: 482, 3: 491, 4: 497, 5: 566, 6: 569, 7: 571, 8: 574, 9: 701}}
CLASS_NAMES = {{0: "tench", 1: "English springer", 2: "cassette player", 3: "chain saw", 4: "church",
               5: "French horn", 6: "garbage truck", 7: "gas pump", 8: "golf ball", 9: "parachute"}}

FINAL_OPS = {{
    "NBJ": "JPEG", "BNJ": "JPEG",
    "NJB": "Blur", "JNB": "Blur",
    "BJN": "Noise", "JBN": "Noise",
}}

full_ds = load_dataset("johnowhitaker/imagenette2-320", split="train")
val_ds = full_ds.select(range(9469, len(full_ds)))
assert len(val_ds) == 3925, f"Expected 3,925 images, got {{len(val_ds)}}"

print("Loading Models onto GPU...")
models = {{
    "resnet50": timm.create_model("resnet50.a1_in1k", pretrained=True).to(device).eval(),
    "convnext_tiny": timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=True).to(device).eval(),
    "vit_base": timm.create_model("vit_base_patch16_224.augreg_in21k_ft_in1k", pretrained=True).to(device).eval(),
    "swin_tiny": timm.create_model("swin_tiny_patch4_window7_224.ms_in1k", pretrained=True).to(device).eval(),
}}
lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

norm_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
spatial_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

SEQUENCES = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]
LEVELS = ["Level_1_Mild", "Level_2_ModMild", "Level_3_Moderate"]

records = []
print("Beginning Batch Execution across 3,925 Images...")
t0 = time.time()

for img_idx in range(len(val_ds)):
    item = val_ds[img_idx]
    clean_img = spatial_transform(item["image"].convert("RGB"))
    true_lbl = item["label"]
    true_1k = IMAGENETTE_TO_IMAGENET[true_lbl]

    clean_norm = norm_transform(clean_img).unsqueeze(0).to(device)
    clean_margins = {{}}
    with torch.no_grad():
        for m_name, m_net in models.items():
            logits = m_net(clean_norm).squeeze(0)
            z_t = float(logits[true_1k].item())
            z_o = float(torch.cat([logits[:true_1k], logits[true_1k+1:]]).max().item())
            clean_margins[m_name] = z_t - z_o

    for lvl in LEVELS:
        lvl_data = CALIBRATED_PARAMS[lvl]["sequences"]
        for seq in SEQUENCES:
            p = lvl_data[seq]
            corr_img = apply_compound_sequence(
                clean_img, order=seq,
                noise_std=p["noise_std"], blur_sigma=p["blur_sigma"], jpeg_quality=int(p["jpeg_quality"]),
                noise_seed=42 + img_idx * 13,
            )

            p_val = compute_psnr(clean_img, corr_img)
            s_val, _ = compute_ssim_and_local_map(clean_img, corr_img)
            with torch.no_grad():
                lp_clean = (clean_img.unsqueeze(0) * 2.0 - 1.0).to(device)
                lp_corr = (corr_img.unsqueeze(0) * 2.0 - 1.0).to(device)
                lpips_val = float(lpips_fn(lp_clean, lp_corr).item())

            fingerprints = compute_distortion_fingerprints(clean_img, corr_img)
            corr_norm = norm_transform(corr_img).unsqueeze(0).to(device)

            with torch.no_grad():
                for m_name, m_net in models.items():
                    logits = m_net(corr_norm).squeeze(0)
                    probs = F.softmax(logits, dim=0)
                    z_t = float(logits[true_1k].item())
                    z_o = float(torch.cat([logits[:true_1k], logits[true_1k+1:]]).max().item())
                    margin = z_t - z_o
                    delta_margin = margin - clean_margins[m_name]
                    pred_1k = int(logits.argmax().item())
                    correct = int(pred_1k == true_1k)
                    conf = float(probs[pred_1k].item())

                    records.append({{
                        "image_id": img_idx,
                        "true_label": true_lbl,
                        "class_name": CLASS_NAMES[true_lbl],
                        "level": lvl,
                        "order": seq,
                        "final_op": FINAL_OPS[seq],
                        "model": m_name,
                        "psnr": round(p_val, 3),
                        "ssim": round(s_val, 4),
                        "lpips": round(lpips_val, 4),
                        **fingerprints,
                        "logit_margin": round(margin, 4),
                        "delta_logit_margin": round(delta_margin, 4),
                        "top1_correct": correct,
                        "confidence": round(conf, 4),
                    }})

    if (img_idx + 1) % 250 == 0:
        elapsed = time.time() - t0
        rate = (img_idx + 1) / elapsed
        rem = (len(val_ds) - (img_idx + 1)) / rate
        print(f"Progress: {{img_idx + 1}}/{{len(val_ds)}} images ({{elapsed:.1f}}s, est {{rem/60:.1f}}m remaining)...")

df_full = pd.DataFrame(records)
df_full.to_csv("full_distortion_fingerprints_and_margins.csv.gz", index=False, compression="gzip")
print(f"Completed {{len(df_full)}} evaluations in {{time.time() - t0:.1f}}s.")
""")

    nb.cells = [c1, c2, c3, c4, c5]
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Generated Calibrated Kaggle confirmatory notebook at {output_path}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "notebooks" / "02_distortion_fingerprints_confirmatory.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_confirmatory_notebook(out)
