"""
CINS Phase III: Commutator-Induced Neural Susceptibility
Standalone Kaggle / Google Colab Cloud GPU Runner

Executes full PyTorch autograd backpropagation and 2D FFT cross-correlation
across canonical vision architectures:
- ResNet-50
- ConvNeXt-Tiny
- ViT-Base/16
- Swin-Tiny

Author: RobustLens Research Initiative
Date: November 2026
"""

import os
import sys
import json
import time
from io import BytesIO
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

# ==============================================================================
# SECTION 1: CINS MATHEMATICAL OPERATORS
# ==============================================================================

def operator_noise(img_np, seed=None, sigma=0.08):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, img_np.shape).astype(np.float32)
    return np.clip(img_np + noise, 0.0, 1.0), noise

def operator_blur(img_np, radius=2.0):
    blurred = np.empty_like(img_np)
    for c in range(img_np.shape[2]):
        blurred[:, :, c] = gaussian_filter(img_np[:, :, c], sigma=radius, mode="reflect")
    return np.clip(blurred, 0.0, 1.0)

def operator_jpeg(img_np, quality=75):
    img_uint8 = np.clip(img_np * 255.0, 0.0, 255.0).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    decompressed = Image.open(buf)
    return np.asarray(decompressed, dtype=np.float32) / 255.0

def generate_cins_pair(img_np, seed=42, sigma=0.08, radius=2.0, quality=75):
    # u = J(B(N(x, eps)))
    n_x, eps = operator_noise(img_np, seed=seed, sigma=sigma)
    bn_x = operator_blur(n_x, radius=radius)
    u = operator_jpeg(bn_x, quality=quality)

    # v = J(N(B(x), eps))
    b_x = operator_blur(img_np, radius=radius)
    nb_x = np.clip(b_x + eps, 0.0, 1.0)
    v = operator_jpeg(nb_x, quality=quality)

    d = (u - v).astype(np.float32)
    c = 0.5 * (u + v).astype(np.float32)
    return u, v, d, c, eps

def circular_shift(arr, ty, tx):
    return np.roll(arr, shift=(ty, tx), axis=(0, 1))

def compute_fft_response_map(grad, d, alpha=1.0):
    H, W, C = grad.shape
    accum = np.zeros((H, W), dtype=np.float64)
    for c in range(C):
        G = np.fft.fftn(grad[:, :, c].astype(np.float64))
        D = np.fft.fftn(d[:, :, c].astype(np.float64))
        cross_spec = G * np.conj(D)
        accum += np.real(np.fft.ifftn(cross_spec))
    return (alpha * accum).astype(np.float32)

# ==============================================================================
# SECTION 2: PYTORCH MODEL LOADER & GRADIENT EXTRACTOR
# ==============================================================================

class PyTorchModelEvaluator:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.model = self._load_model().to(device)
        self.model.eval()

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def _load_model(self):
        print(f"Loading {self.model_name}...")
        if self.model_name == "ResNet-50":
            return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif self.model_name == "ConvNeXt-Tiny":
            return models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        elif self.model_name == "ViT-Base":
            return models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        elif self.model_name == "Swin-Tiny":
            return models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def get_prediction_and_gradient(self, img_np, target_class=None):
        tensor = torch.tensor(img_np, dtype=torch.float32, device=self.device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor.requires_grad_(True)

        norm_tensor = self.normalize(tensor.squeeze(0)).unsqueeze(0)
        logits = self.model(norm_tensor)
        probs = torch.softmax(logits, dim=-1)

        pred_class = int(torch.argmax(logits, dim=-1)[0].item())
        pred_conf = float(probs[0, pred_class].item())

        if target_class is None:
            target_class = pred_class

        target_logit = logits[0, target_class]
        grad = torch.autograd.grad(target_logit, tensor)[0]
        grad_np = grad.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        return {
            "pred_class": pred_class,
            "pred_conf": pred_conf,
            "target_class": target_class,
            "target_logit": float(target_logit.item()),
            "gradient": grad_np
        }

    def evaluate_logits(self, batch_np, target_class):
        # batch_np: [B, H, W, C]
        tensor = torch.tensor(batch_np, dtype=torch.float32, device=self.device)
        tensor = tensor.permute(0, 3, 1, 2)
        norm_tensors = torch.stack([self.normalize(t) for t in tensor])

        with torch.no_grad():
            logits = self.model(norm_tensors)
            target_logits = logits[:, target_class].cpu().numpy()
            probs = torch.softmax(logits, dim=-1)[:, target_class].cpu().numpy()

        return target_logits, probs

# ==============================================================================
# SECTION 3: EXPERIMENTAL PIPELINE RUNNER
# ==============================================================================

def run_cins_experiment():
    print("=" * 70)
    print("CINS PHASE III: KAGGLE/COLAB CLOUD GPU BENCHMARK")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")

    # Create synthetic natural photographic benchmark image if no file provided
    np.random.seed(42)
    H, W = 224, 224
    base_grad_x = np.linspace(0.2, 0.8, W, dtype=np.float32)
    base_grad_y = np.linspace(0.2, 0.8, H, dtype=np.float32)
    xx, yy = np.meshgrid(base_grad_x, base_grad_y)
    img_np = np.zeros((H, W, 3), dtype=np.float32)
    img_np[:, :, 0] = xx
    img_np[:, :, 1] = yy
    img_np[:, :, 2] = 0.5 * (xx + yy)

    # Add Gaussian circular blob (synthetic object)
    cy, cx = 112, 112
    dist_sq = (np.arange(H)[:, None] - cy)**2 + (np.arange(W)[None, :] - cx)**2
    blob = np.exp(-dist_sq / (2 * (35**2)))
    img_np[:, :, 0] = np.clip(img_np[:, :, 0] + 0.3 * blob, 0.05, 0.95)
    img_np[:, :, 1] = np.clip(img_np[:, :, 1] - 0.2 * blob, 0.05, 0.95)

    u, v, d, c, eps = generate_cins_pair(img_np, seed=42)

    # Apply Protocol B: Interior-Anchor Contraction
    eta = 0.05
    c_eta = (1.0 - 2.0 * eta) * c + eta

    models_to_test = ["ResNet-50", "ConvNeXt-Tiny", "ViT-Base", "Swin-Tiny"]
    alpha_levels = [0.10, 0.25, 0.50]

    # Sample a 32-point translation grid
    grid_offsets = []
    step = 56
    for gy in range(0, 224, step):
        for gx in range(0, 224, step):
            grid_offsets.append((gy, gx))

    full_results = {}

    for m_name in models_to_test:
        print("\n" + "-" * 60)
        print(f"BENCHMARKING ARCHITECTURE: {m_name}")
        print("-" * 60)

        evaluator = PyTorchModelEvaluator(m_name, device)

        # Step 1: Base prediction & exact autograd gradient at c_eta
        base_eval = evaluator.get_prediction_and_gradient(c_eta)
        target_class = base_eval["target_class"]
        grad = base_eval["gradient"]

        print(f"Anchor Prediction: Class {target_class} (Conf: {base_eval['pred_conf']*100:.2f}%, Logit: {base_eval['target_logit']:.2f})")
        print(f"Gradient L2 Norm: {np.linalg.norm(grad):.4f}")

        full_results[m_name] = {
            "target_class": target_class,
            "anchor_logit": base_eval["target_logit"],
            "anchor_conf": base_eval["pred_conf"],
            "alphas": {}
        }

        for alpha in alpha_levels:
            print(f"\n--- Testing Intervention Strength alpha = {alpha} ---")
            t_start = time.time()
            fft_map = compute_fft_response_map(grad, d, alpha=alpha)
            fft_time_ms = (time.time() - t_start) * 1000.0

            # Collect predicted vs actual responses on grid offsets
            predicted_vals = []
            actual_logit_deltas = []
            actual_prob_deltas = []

            # Batch evaluate all grid shifts
            batch_plus = []
            batch_minus = []
            for ty, tx in grid_offsets:
                d_tau = circular_shift(d, ty, tx)
                x_plus = c_eta + 0.5 * alpha * d_tau
                x_minus = c_eta - 0.5 * alpha * d_tau
                batch_plus.append(x_plus)
                batch_minus.append(x_minus)
                predicted_vals.append(float(fft_map[ty, tx]))

            batch_plus_np = np.stack(batch_plus, axis=0)
            batch_minus_np = np.stack(batch_minus, axis=0)

            logits_plus, probs_plus = evaluator.evaluate_logits(batch_plus_np, target_class)
            logits_minus, probs_minus = evaluator.evaluate_logits(batch_minus_np, target_class)

            actual_logit_deltas = (logits_plus - logits_minus).tolist()
            actual_prob_deltas = (probs_plus - probs_minus).tolist()

            # Compute correlation metrics
            r_pearson, p_val = pearsonr(predicted_vals, actual_logit_deltas)
            r_spearman, sp_val = spearmanr(predicted_vals, actual_logit_deltas)
            sign_agreement = np.mean(np.sign(predicted_vals) == np.sign(actual_logit_deltas))

            print(f"FFT Map Time: {fft_time_ms:.2f} ms")
            print(f"Pearson Correlation (rho): {r_pearson:.4f} (p={p_val:.2e})")
            print(f"Spearman Rank Correlation: {r_spearman:.4f}")
            print(f"Directional Sign Agreement: {sign_agreement*100:.1f}%")

            full_results[m_name]["alphas"][str(alpha)] = {
                "fft_time_ms": fft_time_ms,
                "pearson_rho": float(r_pearson),
                "pearson_p": float(p_val),
                "spearman_rho": float(r_spearman),
                "sign_agreement": float(sign_agreement),
                "grid_sample_count": len(grid_offsets),
                "predicted_vals": predicted_vals,
                "actual_logit_deltas": actual_logit_deltas,
                "actual_prob_deltas": actual_prob_deltas
            }

    out_path = "cins_kaggle_results.json"
    with open(out_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print("\n" + "=" * 70)
    print(f"ALL BENCHMARKS COMPLETE. Results saved to: {out_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_cins_experiment()
