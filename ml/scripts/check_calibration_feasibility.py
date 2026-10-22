"""Feasibility study on joint PSNR and SSIM matching for multi-stage corruptions.

Evaluates whether the 6 sequence permutations can jointly achieve:
PSNR tolerance: +/- 0.75 dB
SSIM tolerance: +/- 0.035
across the 5 target distortion levels on independent training images (indices 0..9468).
"""

import io
import math
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from datasets import load_dataset

def compute_psnr_batch(clean: torch.Tensor, corrupt: torch.Tensor) -> float:
    mse = torch.mean((clean.float() - corrupt.float()) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return float(10.0 * math.log10(1.0 / mse))

def _create_gaussian_kernel_2d(k=11, s=1.5):
    coords = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * s ** 2))
    g = g / g.sum()
    return torch.outer(g, g)

_SSIM_K = _create_gaussian_kernel_2d(11, 1.5).expand(3, 1, 11, 11)

def compute_ssim_batch(clean: torch.Tensor, corrupt: torch.Tensor) -> float:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu1 = F.conv2d(clean.float(), _SSIM_K, padding=5, groups=3)
    mu2 = F.conv2d(corrupt.float(), _SSIM_K, padding=5, groups=3)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1 * mu2
    sig1_sq = F.conv2d(clean.float() * clean.float(), _SSIM_K, padding=5, groups=3) - mu1_sq
    sig2_sq = F.conv2d(corrupt.float() * corrupt.float(), _SSIM_K, padding=5, groups=3) - mu2_sq
    sig12 = F.conv2d(clean.float() * corrupt.float(), _SSIM_K, padding=5, groups=3) - mu12
    num = (2 * mu12 + c1) * (2 * sig12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sig1_sq + sig2_sq + c2)
    return float((num / den).mean().item())

def apply_seq(img: torch.Tensor, order: str, n_std: float, b_sig: float, q_jpg: int, seed: int = 42) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(img.shape, generator=gen) * max(0.0001, n_std)
    cur = img
    k = max(3, int(2 * round(b_sig * 1.5) + 1))
    for char in order:
        if char == "N":
            cur = torch.clamp(cur + noise, 0.0, 1.0)
        elif char == "B":
            if b_sig > 0.05:
                cur = TF.gaussian_blur(cur, [k, k], [b_sig, b_sig])
        elif char == "J":
            pil_img = TF.to_pil_image(cur.cpu())
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=int(round(q_jpg)))
            buf.seek(0)
            cur = TF.to_tensor(Image.open(buf))
    return cur

def eval_condition(batch: torch.Tensor, order: str, n_std: float, b_sig: float, q_jpg: int):
    B = batch.shape[0]
    corrupt = torch.stack([apply_seq(batch[i], order, n_std, b_sig, q_jpg, seed=1000 + i*13) for i in range(B)])
    return compute_psnr_batch(batch, corrupt), compute_ssim_batch(batch, corrupt)

def main():
    print("Loading dataset (training split rows 0..9468)...", flush=True)
    ds = load_dataset("johnowhitaker/imagenette2-320", split="train")
    spatial_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    rng = np.random.RandomState(42)
    indices = rng.choice(range(0, 9469), size=15, replace=False).tolist()
    batch = torch.stack([spatial_transform(ds[i]["image"].convert("RGB")) for i in indices])
    print(f"Loaded batch of {len(batch)} images.", flush=True)

    targets = {
        1: (28.0, 0.850, "Target 1 (Mild)"),
        2: (24.5, 0.740, "Target 2 (Mod-Mild)"),
        3: (21.5, 0.620, "Target 3 (Moderate)"),
        4: (18.5, 0.480, "Target 4 (Mod-Severe)"),
        5: (16.0, 0.360, "Target 5 (Severe)"),
    }
    sequences = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]
    tol_p = 0.75
    tol_s = 0.035

    print("\n" + "=" * 80, flush=True)
    print("EXPLORING JOINT PSNR/SSIM FEASIBILITY ACROSS ALL 6 SEQUENCES", flush=True)
    print("Target tolerances: PSNR +/- 0.75 dB, SSIM +/- 0.035", flush=True)
    print("=" * 80, flush=True)

    for t_idx, (t_p, t_s, t_name) in targets.items():
        print(f"\n---> Analyzing {t_name}: Target PSNR = {t_p:.1f} dB, Target SSIM = {t_s:.3f}", flush=True)
        for seq in sequences:
            # Let's search over a grid around reasonable values to find the best compromise
            # or see if any parameter set hits both tolerances
            best_loss = 1e9
            best_res = None
            found_within_tol = False

            # Define search ranges depending on target
            if t_idx == 1:
                n_list = [0.015, 0.025, 0.035, 0.05, 0.07]
                b_list = [0.4, 0.7, 1.0, 1.3]
                q_list = [70, 80, 88, 95]
            elif t_idx == 3:
                n_list = [0.04, 0.07, 0.10, 0.14, 0.18]
                b_list = [1.2, 1.8, 2.4, 3.0]
                q_list = [30, 45, 60, 75]
            elif t_idx == 5:
                n_list = [0.10, 0.16, 0.24, 0.35, 0.45]
                b_list = [2.5, 3.8, 5.0, 6.5]
                q_list = [8, 15, 25, 40]
            else:
                n_list = [0.03, 0.06, 0.10, 0.15]
                b_list = [0.8, 1.5, 2.2, 3.0]
                q_list = [40, 55, 70, 85]

            for n in n_list:
                for b in b_list:
                    for q in q_list:
                        p, s = eval_condition(batch, seq, n, b, q)
                        dp = p - t_p
                        ds = s - t_s
                        # Check tolerance
                        if abs(dp) <= tol_p and abs(ds) <= tol_s:
                            found_within_tol = True
                            print(f"  [SUCCESS] {seq}: n={n:.3f}, b={b:.1f}, q={q:2d} -> PSNR={p:.2f} dB (d={dp:+.2f}), SSIM={s:.4f} (d={ds:+.4f})", flush=True)
                            break
                        loss = (dp / tol_p)**2 + (ds / tol_s)**2
                        if loss < best_loss:
                            best_loss = loss
                            best_res = (n, b, q, p, s, dp, ds)
                    if found_within_tol:
                        break
                if found_within_tol:
                    break

            if not found_within_tol:
                n, b, q, p, s, dp, ds = best_res
                print(f"  [CANNOT MATCH BOTH] {seq}: Closest n={n:.3f}, b={b:.1f}, q={q:2d} -> PSNR={p:.2f} (d={dp:+.2f} dB), SSIM={s:.4f} (d={ds:+.4f})", flush=True)

if __name__ == "__main__":
    main()
