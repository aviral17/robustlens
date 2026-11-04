"""Build complete confirmatory notebook with post-execution verification and statistical multiplicity audit.

Generates:
ml/notebooks/02_distortion_fingerprints_confirmatory.ipynb
and
ml/notebooks/confirmatory_distortion_fingerprints/02_distortion_fingerprints_confirmatory.ipynb
plus kernel-metadata.json
"""

from __future__ import annotations

import json
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PARAMS_JSON = BASE_DIR / "results" / "matched_fidelity_study" / "calibrated_common_support_3levels.json"
PILOT_IDS_JSON = BASE_DIR / "results" / "matched_fidelity_study" / "pilot_exploratory_100_image_ids.json"


def build_notebook():
    with open(PARAMS_JSON, "r") as f:
        calib_data = json.load(f)
    calib_repr = json.dumps(calib_data, indent=2)

    with open(PILOT_IDS_JSON, "r") as f:
        pilot_data = json.load(f)
    pilot_indices = pilot_data.get("pilot_dataset_row_indices", pilot_data["pilot_local_image_ids"])
    pilot_indices_repr = json.dumps(pilot_indices)

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

    # Cell 0: Markdown Introduction
    c0 = new_markdown_cell("""# 🔬 RobustLens Confirmatory Study: Spatial-Frequency & Structural Distortion Fingerprints
## Full Confirmatory Evaluation on FastAI Imagenette-v2 ($N = 3{,}825$ Reserved Images)

- **Research Question:** Beyond global image fidelity (PSNR, SSIM, LPIPS), do spatial-frequency and structural distortion fingerprints explain architecture-specific true-class logit margins under compound corruption permutations?
- **Dataset:** Official FastAI Imagenette-v2 Validation Split ($N = 3{,}825$ reserved confirmatory images, rows 9,469..13,393; 100 pilot images strictly quarantined).
- **Scale:** $3{,}825 \\times 19 \\times 4 = 290{,}700$ neural forward passes on NVIDIA GPU.
- **Experimental Conditions:** 1 clean baseline + 6 sequences $\\times$ 3 calibrated levels = 19 conditions per image.
  - Level 1 (Mild Common Support): Target PSNR $27.2\\text{ dB}$, SSIM $0.810$.
  - Level 2 (Moderate Common Support): Target PSNR $23.8\\text{ dB}$, SSIM $0.650$.
  - Level 3 (Severe Common Support): Target PSNR $21.8\\text{ dB}$, SSIM $0.500$.
- **Descriptors:**
  1. Global Fidelity: PSNR, SSIM, LPIPS (AlexNet feature space).
  2. 2D Fourier radial power fractions ($E_{\\text{low}}, E_{\\text{med}}, E_{\\text{high}}$).
  3. JPEG $8 \\times 8$ block-boundary discontinuity ratio ($B_{\\text{JPEG}}$).
  4. Relative Sobel gradient edge energy change ($\\Delta E_{\\text{edge}}$).
  5. Local SSIM spatial variability ($\\text{Std}(\\text{SSIM}_{\\text{local}})$).
- **Primary Model Outcome:** True-class logit margin $M = z_{\\text{true}} - \\max_{j \\ne \\text{true}, j \\in 0..999} z_j$.
- **Confirmatory Inference:** Grouped 5-fold CV by physical image cluster, paired cluster bootstrap ($B=2{,}000$), step-down Holm-Bonferroni correction across 4 architectures.
""")

    # Cell 1: Environment & Setup
    c1 = new_code_cell("""# 1. Environment & GPU Verification
!pip install -q timm datasets lpips scipy statsmodels

import io
import json
import math
import os
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import scipy.signal
from scipy import stats

import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import timm
import lpips
from datasets import load_dataset

assert torch.cuda.is_available(), "GPU is REQUIRED for the 290,700 forward passes!"
device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"CUDA Device: {gpu_name} ({gpu_mem:.2f} GB)")
print(f"PyTorch: {torch.__version__} | Torchvision: {torchvision.__version__} | timm: {timm.__version__}")
""")

    # Cell 2: Fingerprint Mathematical Definitions
    c2 = new_code_cell("""# 2. Mathematical Definition of Distortion Fingerprints & Fidelity Descriptors

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

    # 1. 2D Fourier Power Spectrum Fractions (Low, Med, High radial bands)
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

    # 4. Local SSIM Spatial Variation
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

    # Cell 3: Compound Corruption Transform
    c3 = new_code_cell("""# 3. Compound Permutation Application Function
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

    # Cell 4: Batch Evaluation Loop (N=3,825 images, 290,700 forward passes)
    c4 = new_code_cell(f"""# 4. Confirmatory Evaluation Loop (N=3,825 Images x 19 Conditions x 4 Models = 290,700 Passes)
import json

CALIBRATED_PARAMS = json.loads(r'''{calib_repr}''')
PILOT_ROW_INDICES = set({pilot_indices_repr})

IMAGENETTE_TO_IMAGENET = {{0: 0, 1: 217, 2: 482, 3: 491, 4: 497, 5: 566, 6: 569, 7: 571, 8: 574, 9: 701}}
CLASS_NAMES = {{0: "tench", 1: "English springer", 2: "cassette player", 3: "chain saw", 4: "church",
               5: "French horn", 6: "garbage truck", 7: "gas pump", 8: "golf ball", 9: "parachute"}}

FINAL_OPS = {{
    "NBJ": "JPEG", "BNJ": "JPEG",
    "NJB": "Blur", "JNB": "Blur",
    "BJN": "Noise", "JBN": "Noise",
}}

full_ds = load_dataset("johnowhitaker/imagenette2-320", split="train")

# Strictly select the 3,825 untouched confirmatory validation images (rows 9469..13393 excluding 100 pilot rows)
confirmatory_indices = [r for r in range(9469, 13394) if r not in PILOT_ROW_INDICES]
val_ds = full_ds.select(confirmatory_indices)
assert len(val_ds) == 3825, f"Expected 3,825 confirmatory images, got {{len(val_ds)}}"
print(f"Loaded {{len(val_ds)}} untouched confirmatory validation images (100 exploratory pilot images strictly quarantined).")

print("Loading 4 Frozen Vision Models onto GPU...")
models = {{
    "resnet50": timm.create_model("resnet50.a1_in1k", pretrained=True).to(device).eval(),
    "convnext_tiny": timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=True).to(device).eval(),
    "vit_base": timm.create_model("vit_base_patch16_224.augreg_in21k_ft_in1k", pretrained=True).to(device).eval(),
    "swin_tiny": timm.create_model("swin_tiny_patch4_window7_224.ms_in1k", pretrained=True).to(device).eval(),
}}
lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

# Model-specific normalization transforms
model_norms = {{
    "resnet50": T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    "convnext_tiny": T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    "vit_base": T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    "swin_tiny": T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
}}

spatial_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

SEQUENCES = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]
LEVELS = ["Level_1_Mild", "Level_2_Moderate", "Level_3_SevereCommon"]

records = []
print("Beginning Confirmatory Batch Execution across 3,825 Images...")
t0 = time.time()

for img_idx in range(len(val_ds)):
    item = val_ds[img_idx]
    clean_img = spatial_transform(item["image"].convert("RGB"))
    true_lbl = item["label"]
    true_1k = IMAGENETTE_TO_IMAGENET[true_lbl]

    clean_margins = {{}}
    with torch.no_grad():
        for m_name, m_net in models.items():
            norm_fn = model_norms[m_name]
            clean_norm = norm_fn(clean_img).unsqueeze(0).to(device)
            logits = m_net(clean_norm).squeeze(0)
            z_t = float(logits[true_1k].item())
            z_o = float(torch.cat([logits[:true_1k], logits[true_1k+1:]]).max().item())
            clean_margins[m_name] = z_t - z_o

    for lvl in LEVELS:
        lvl_data = CALIBRATED_PARAMS["levels"][lvl]["sequences"]
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

            with torch.no_grad():
                for m_name, m_net in models.items():
                    norm_fn = model_norms[m_name]
                    corr_norm = norm_fn(corr_img).unsqueeze(0).to(device)
                    logits = m_net(corr_norm).squeeze(0)
                    probs = F.softmax(logits, dim=0)

                    z_t = float(logits[true_1k].item())
                    # Full 1,000-class competitor search:
                    z_o = float(torch.cat([logits[:true_1k], logits[true_1k+1:]]).max().item())
                    margin = z_t - z_o
                    delta_margin = margin - clean_margins[m_name]
                    pred_1k = int(logits.argmax().item())
                    correct = int(pred_1k == true_1k)
                    conf = float(probs[pred_1k].item())

                    records.append({{
                        "image_id": img_idx,
                        "dataset_row": confirmatory_indices[img_idx],
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

    if (img_idx + 1) % 250 == 0 or (img_idx + 1) == len(val_ds):
        elapsed = time.time() - t0
        rate = (img_idx + 1) / elapsed
        rem = (len(val_ds) - (img_idx + 1)) / rate
        print(f"Progress: {{img_idx + 1}}/{{len(val_ds)}} images ({{elapsed:.1f}}s, est {{rem/60:.1f}}m remaining)...")

df_full = pd.DataFrame(records)
csv_out_path = "full_distortion_fingerprints_and_margins.csv.gz"
df_full.to_csv(csv_out_path, index=False, compression="gzip")
total_time = time.time() - t0
print(f"CONFIRMATORY RUN COMPLETE: Saved {{len(df_full):,}} records to {{csv_out_path}} in {{total_time:.1f}}s.")
""")

    # Cell 5: Post-Execution Verification and Confirmatory Grouped-CV Audit
    c5 = new_code_cell("""# 5. Confirmatory Audit: Multiplicity Resolution & Grouped Cross-Validation
import json
import platform
import numpy as np
import pandas as pd
import torch
import torchvision
import timm

print("=" * 80)
print("CONFIRMATORY AUDIT: REPRODUCIBILITY VERIFICATION & GROUPED CV")
print("=" * 80)

# 1. Output Integrity Checks
expected_rows = 3825 * 18 * 4  # 18 corrupted conditions x 4 models
print(f"Verifying complete output row count: {len(df_full):,} (expected {expected_rows:,})...")
assert len(df_full) == expected_rows, f"Row count mismatch: got {len(df_full)}, expected {expected_rows}"
assert df_full["image_id"].nunique() == 3825, f"Expected 3825 unique images, got {df_full['image_id'].nunique()}"
assert df_full["model"].nunique() == 4, f"Expected 4 models, got {df_full['model'].nunique()}"
assert df_full["order"].nunique() == 6, f"Expected 6 sequences, got {df_full['order'].nunique()}"
assert df_full["level"].nunique() == 3, f"Expected 3 levels, got {df_full['level'].nunique()}"

missing_count = int(df_full.isnull().sum().sum())
print(f"Missing values across entire dataset: {missing_count}")
assert missing_count == 0, "Missing values detected in confirmatory predictions!"

# 2. Hardware and Environment Manifest
manifest = {
    "evaluation_title": "RobustLens Confirmatory Study Evaluation Manifest",
    "total_records": len(df_full),
    "unique_images": df_full["image_id"].nunique(),
    "conditions_per_image": 18,
    "models_evaluated": list(models.keys()),
    "gpu_device": torch.cuda.get_device_name(0),
    "gpu_total_memory_gb": round(float(torch.cuda.get_device_properties(0).total_memory / (1024**3)), 2),
    "execution_time_seconds": round(total_time, 2),
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "torchvision_version": torchvision.__version__,
    "timm_version": timm.__version__,
    "zero_missing_values_verified": True,
}
with open("confirmatory_execution_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("Saved confirmatory_execution_manifest.json")

# 3. Restricted Cubic Spline Basis Function
def natural_cubic_spline_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    K = len(knots)
    k_K = knots[-1]
    k_Km1 = knots[-2]
    basis = [x]
    denom_last = max(1e-8, abs(k_K - k_Km1))
    term_last = np.maximum(0.0, x - k_Km1)**3 - np.maximum(0.0, x - k_K)**3
    d_last = term_last / denom_last
    for j in range(1, K - 2):
        k_j = knots[j]
        denom_j = max(1e-8, abs(k_K - k_j))
        d_j = (np.maximum(0.0, x - k_j)**3 - np.maximum(0.0, x - k_K)**3) / denom_j
        basis.append(d_j - d_last)
    return np.column_stack(basis)

def build_design_matrix(df_part, spec, scaler_params, spline_knots=None):
    N = len(df_part)
    orders = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]
    levels = ["Level_1_Mild", "Level_2_Moderate", "Level_3_SevereCommon"]

    order_mat = np.zeros((N, len(orders) - 1))
    for i, o in enumerate(orders[1:]):
        order_mat[:, i] = (df_part["order"].to_numpy() == o).astype(float)

    level_mat = np.zeros((N, len(levels) - 1))
    for i, lvl in enumerate(levels[1:]):
        level_mat[:, i] = (df_part["level"].to_numpy() == lvl).astype(float)

    order_x_level = np.zeros((N, (len(orders) - 1) * (len(levels) - 1)))
    col_idx = 0
    for o_i in range(order_mat.shape[1]):
        for l_i in range(level_mat.shape[1]):
            order_x_level[:, col_idx] = order_mat[:, o_i] * level_mat[:, l_i]
            col_idx += 1

    p_raw = df_part["psnr"].to_numpy()
    s_raw = df_part["ssim"].to_numpy()
    l_raw = df_part["lpips"].to_numpy()

    p_std = (p_raw - scaler_params["psnr"]["mean"]) / scaler_params["psnr"]["std"]
    s_std = (s_raw - scaler_params["ssim"]["mean"]) / scaler_params["ssim"]["std"]
    l_std = (l_raw - scaler_params["lpips"]["mean"]) / scaler_params["lpips"]["std"]

    intercept = np.ones((N, 1))
    base_parts = [intercept, order_mat, level_mat, order_x_level]

    if spec == "spline":
        p_spline = natural_cubic_spline_basis(p_std, spline_knots["psnr"])
        s_spline = natural_cubic_spline_basis(s_std, spline_knots["ssim"])
        l_spline = natural_cubic_spline_basis(l_std, spline_knots["lpips"])
        return np.hstack(base_parts + [p_spline, s_spline, l_spline])

    elif spec == "spline_extended":
        p_spline = natural_cubic_spline_basis(p_std, spline_knots["psnr"])
        s_spline = natural_cubic_spline_basis(s_std, spline_knots["ssim"])
        l_spline = natural_cubic_spline_basis(l_std, spline_knots["lpips"])
        f_low = (df_part["fft_energy_low"].to_numpy() - scaler_params["fft_energy_low"]["mean"]) / scaler_params["fft_energy_low"]["std"]
        f_high = (df_part["fft_energy_high"].to_numpy() - scaler_params["fft_energy_high"]["mean"]) / scaler_params["fft_energy_high"]["std"]
        jpeg_b = (df_part["jpeg_block_artifact"].to_numpy() - scaler_params["jpeg_block_artifact"]["mean"]) / scaler_params["jpeg_block_artifact"]["std"]
        d_edge = (df_part["delta_edge_energy"].to_numpy() - scaler_params["delta_edge_energy"]["mean"]) / scaler_params["delta_edge_energy"]["std"]
        s_std_loc = (df_part["std_local_ssim"].to_numpy() - scaler_params["std_local_ssim"]["mean"]) / scaler_params["std_local_ssim"]["std"]
        fingerprints = np.column_stack([f_low, f_high, jpeg_b, d_edge, s_std_loc])
        return np.hstack(base_parts + [p_spline, s_spline, l_spline, fingerprints])

def holm_bonferroni(p_vals):
    m = len(p_vals)
    sorted_idx = np.argsort(p_vals)
    sorted_p = [p_vals[i] for i in sorted_idx]
    adj_p = [0.0] * m
    running_max = 0.0
    for rank_i, (orig_i, pv) in enumerate(zip(sorted_idx, sorted_p)):
        mult = m - rank_i
        curr_adj = min(1.0, mult * pv)
        running_max = max(running_max, curr_adj)
        adj_p[orig_i] = running_max
    return adj_p

# 4. Run Grouped CV & Paired Cluster Bootstrap
print("Running Grouped 5-Fold Cross-Validation on Confirmatory Dataset...")
clusters = np.sort(df_full["image_id"].unique())
n_clusters = len(clusters)
rng_cv = np.random.RandomState(42)

cluster_classes = df_full.drop_duplicates("image_id").sort_values("image_id")["true_label"].to_numpy()
fold_assignments = np.zeros(n_clusters, dtype=int)
for c in range(10):
    c_indices = np.where(cluster_classes == c)[0]
    perm = rng_cv.permutation(c_indices)
    c_splits = np.array_split(perm, 5)
    for fold_i, split_idx in enumerate(c_splits):
        fold_assignments[split_idx] = fold_i

architectures = ["convnext_tiny", "vit_base", "swin_tiny", "resnet50"]
cont_cols = ["psnr", "ssim", "lpips", "fft_energy_low", "fft_energy_high", "jpeg_block_artifact", "delta_edge_energy", "std_local_ssim"]

cv_results = []
for arch in architectures:
    df_arch = df_full[df_full["model"] == arch].copy()
    y_true_all, y_p3_all, y_p5_all, c_ids = [], [], [], []

    for fold_i in range(5):
        test_mask = (fold_assignments == fold_i)
        test_c = clusters[test_mask]
        train_c = clusters[~test_mask]

        tr_df = df_arch[df_arch["image_id"].isin(train_c)].copy()
        te_df = df_arch[df_arch["image_id"].isin(test_c)].copy()

        scaler_params = {}
        for col in cont_cols:
            vals = tr_df[col].to_numpy()
            scaler_params[col] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) + 1e-8}

        spline_knots = {}
        for col in ["psnr", "ssim", "lpips"]:
            std_v = (tr_df[col].to_numpy() - scaler_params[col]["mean"]) / scaler_params[col]["std"]
            spline_knots[col] = np.percentile(std_v, [5.0, 35.0, 65.0, 95.0])

        X_tr3 = build_design_matrix(tr_df, "spline", scaler_params, spline_knots)
        X_te3 = build_design_matrix(te_df, "spline", scaler_params, spline_knots)
        X_tr5 = build_design_matrix(tr_df, "spline_extended", scaler_params, spline_knots)
        X_te5 = build_design_matrix(te_df, "spline_extended", scaler_params, spline_knots)

        y_tr = tr_df["logit_margin"].to_numpy()
        y_te = te_df["logit_margin"].to_numpy()

        b3 = np.linalg.pinv(X_tr3.T @ X_tr3) @ (X_tr3.T @ y_tr)
        b5 = np.linalg.pinv(X_tr5.T @ X_tr5) @ (X_tr5.T @ y_tr)

        y_p3_all.extend(X_te3 @ b3)
        y_p5_all.extend(X_te5 @ b5)
        y_true_all.extend(y_te)
        c_ids.extend(te_df["image_id"].to_numpy())

    yt = np.array(y_true_all)
    yp3 = np.array(y_p3_all)
    yp5 = np.array(y_p5_all)
    c_arr = np.array(c_ids)

    ss_tot = np.sum((yt - yt.mean())**2)
    r2_3 = float(1.0 - np.sum((yt - yp3)**2) / ss_tot)
    r2_5 = float(1.0 - np.sum((yt - yp5)**2) / ss_tot)
    mae_3 = float(np.mean(np.abs(yt - yp3)))
    mae_5 = float(np.mean(np.abs(yt - yp5)))

    d_r2 = float(r2_5 - r2_3)
    d_mae = float(mae_3 - mae_5)

    # Paired cluster bootstrap (B=2,000)
    c_map = {c: np.where(c_arr == c)[0] for c in clusters}
    rng_b = np.random.RandomState(12345)
    b_dr2, b_dmae = [], []

    for _ in range(2000):
        sb_c = rng_b.choice(clusters, size=len(clusters), replace=True)
        sb_rows = np.concatenate([c_map[c] for c in sb_c])
        y_b = yt[sb_rows]
        yp3_b = yp3[sb_rows]
        yp5_b = yp5[sb_rows]

        ss_b = np.sum((y_b - y_b.mean())**2) + 1e-12
        r2_3b = 1.0 - np.sum((y_b - yp3_b)**2) / ss_b
        r2_5b = 1.0 - np.sum((y_b - yp5_b)**2) / ss_b
        mae_3b = np.mean(np.abs(y_b - yp3_b))
        mae_5b = np.mean(np.abs(y_b - yp5_b))

        b_dr2.append(float(r2_5b - r2_3b))
        b_dmae.append(float(mae_3b - mae_5b))

    b_dr2 = np.array(b_dr2)
    b_dmae = np.array(b_dmae)

    # Pointwise 95% CIs
    ci95_r2 = [float(np.percentile(b_dr2, 2.5)), float(np.percentile(b_dr2, 97.5))]
    ci95_mae = [float(np.percentile(b_dmae, 2.5)), float(np.percentile(b_dmae, 97.5))]

    # Simultaneous Bonferroni 98.75% CIs (family coverage >= 95% across 4 architectures)
    simul_ci_r2 = [float(np.percentile(b_dr2, 1.25)), float(np.percentile(b_dr2, 98.75))]
    simul_ci_mae = [float(np.percentile(b_dmae, 1.25)), float(np.percentile(b_dmae, 98.75))]

    # One-sided empirical p-values: H0: delta <= 0 vs H1: delta > 0
    raw_p_r2 = float((1.0 + np.sum(b_dr2 <= 0.0)) / (len(b_dr2) + 1.0))
    raw_p_mae = float((1.0 + np.sum(b_dmae <= 0.0)) / (len(b_dmae) + 1.0))

    cv_results.append({
        "architecture": arch,
        "r2_oos_spline": round(r2_3, 4),
        "r2_oos_spline_fingerprints": round(r2_5, 4),
        "delta_r2_oos": round(d_r2, 4),
        "raw_p_delta_r2": round(raw_p_r2, 5),
        "ci95_pointwise_delta_r2": [round(ci95_r2[0], 4), round(ci95_r2[1], 4)],
        "simultaneous_bonferroni_ci_delta_r2": [round(simul_ci_r2[0], 4), round(simul_ci_r2[1], 4)],
        "mae_oos_spline": round(mae_3, 4),
        "mae_oos_spline_fingerprints": round(mae_5, 4),
        "delta_mae_oos": round(d_mae, 4),
        "raw_p_delta_mae": round(raw_p_mae, 5),
        "ci95_pointwise_delta_mae": [round(ci95_mae[0], 4), round(ci95_mae[1], 4)],
        "simultaneous_bonferroni_ci_delta_mae": [round(simul_ci_mae[0], 4), round(simul_ci_mae[1], 4)],
    })

# Compute step-down Holm-Bonferroni adjusted p-values
p_r2_list = [r["raw_p_delta_r2"] for r in cv_results]
p_mae_list = [r["raw_p_delta_mae"] for r in cv_results]

holm_r2 = holm_bonferroni(p_r2_list)
holm_mae = holm_bonferroni(p_mae_list)

for i, r in enumerate(cv_results):
    r["holm_adj_p_delta_r2"] = round(float(holm_r2[i]), 5)
    r["significant_r2_holm_0_05"] = bool(holm_r2[i] < 0.05)
    r["holm_adj_p_delta_mae"] = round(float(holm_mae[i]), 5)
    r["significant_mae_holm_0_05"] = bool(holm_mae[i] < 0.05)

df_cv_summary = pd.DataFrame(cv_results)
df_cv_summary.to_csv("confirmatory_grouped_cv_results.csv", index=False)
with open("confirmatory_grouped_cv_diagnostics.json", "w") as f:
    json.dump({"confirmatory_results": cv_results}, f, indent=2)

print("\\nCONFIRMATORY GROUPED CV & MULTIPLICITY SUMMARY TABLE:")
print(df_cv_summary[["architecture", "r2_oos_spline", "r2_oos_spline_fingerprints", "delta_r2_oos", "raw_p_delta_r2", "holm_adj_p_delta_r2", "significant_r2_holm_0_05"]].to_string(index=False))
print("\\nALL EVALUATIONS AND CONFIRMATORY AUDIT SUCCESSFULLY COMPLETED.")
""")

    nb.cells = [c0, c1, c2, c3, c4, c5]
    return nb


def main():
    nb = build_notebook()
    
    # 1. Save to ml/notebooks/02_distortion_fingerprints_confirmatory.ipynb
    out1 = BASE_DIR / "ml" / "notebooks" / "02_distortion_fingerprints_confirmatory.ipynb"
    with open(out1, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Generated {out1} ({out1.stat().st_size:,} bytes)")

    # 2. Save to ml/notebooks/confirmatory_distortion_fingerprints/02_distortion_fingerprints_confirmatory.ipynb
    target_dir = BASE_DIR / "ml" / "notebooks" / "confirmatory_distortion_fingerprints"
    target_dir.mkdir(parents=True, exist_ok=True)
    out2 = target_dir / "02_distortion_fingerprints_confirmatory.ipynb"
    with open(out2, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Generated {out2} ({out2.stat().st_size:,} bytes)")

    # 3. Kernel metadata
    meta = {
        "id": "aviral199/robustlens-distortion-fingerprints-confirmatory",
        "title": "RobustLens Distortion Fingerprints Confirmatory",
        "code_file": "02_distortion_fingerprints_confirmatory.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    meta_path = target_dir / "kernel-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Generated {meta_path}")


if __name__ == "__main__":
    main()
