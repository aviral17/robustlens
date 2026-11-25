"""
Pixel-Validity Feasibility Audit Script for CINS
Calculates exact bounds for alpha across real test images, evaluates anchor saturation,
and tests feasibility for both circular translations and interiorized controls.
"""

import os
import json
import numpy as np
from PIL import Image

def get_image_paths():
    base_dir = r"apps\web\public\examples"
    clean_paths = []
    for i in range(5):
        p = os.path.join(base_dir, f"img_{i}_clean.jpg")
        if os.path.exists(p):
            clean_paths.append(p)
    
    # Also find imagenet_v2 images
    v2_dir = r"data\imagenet_v2\MatchedFrequency"
    if os.path.exists(v2_dir):
        for root, _, files in os.walk(v2_dir):
            for f in files:
                if f.endswith(".jpeg") or f.endswith(".jpg"):
                    clean_paths.append(os.path.join(root, f))
                    if len(clean_paths) >= 10:
                        break
            if len(clean_paths) >= 10:
                break
    return clean_paths[:10]

def simulate_nbj_bnj(img_np, sigma=0.08, blur_radius=2, jpeg_quality=75, seed=42):
    """
    Fixed-parameter NBJ vs BNJ on [0, 1] float image.
    Uses exact same random noise realization epsilon for both sequences.
    """
    H, W, C = img_np.shape
    rng = np.random.RandomState(seed)
    epsilon = rng.normal(0, sigma, img_np.shape).astype(np.float32)

    # Simple separable 2D Gaussian blur kernel
    k_size = 2 * blur_radius + 1
    ax = np.arange(-blur_radius, blur_radius + 1)
    kernel_1d = np.exp(-0.5 * (ax / 1.5) ** 2)
    kernel_1d /= kernel_1d.sum()
    
    def apply_blur(x):
        out = np.zeros_like(x)
        for c in range(C):
            # Convolve rows then cols
            temp = np.apply_along_axis(lambda m: np.convolve(m, kernel_1d, mode='same'), axis=0, arr=x[:, :, c])
            out[:, :, c] = np.apply_along_axis(lambda m: np.convolve(m, kernel_1d, mode='same'), axis=1, arr=temp)
        return np.clip(out, 0.0, 1.0)

    # 1. Sequence NBJ: Noise -> Blur -> JPEG
    x_n = np.clip(img_np + epsilon, 0.0, 1.0)
    x_nb = apply_blur(x_n)
    # Simulate JPEG by 8x8 block DCT quantization or PIL save/load
    from io import BytesIO
    def apply_jpeg(x, q):
        pil_img = Image.fromarray((x * 255.0).astype(np.uint8))
        buf = BytesIO()
        pil_img.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        reloaded = Image.open(buf)
        return np.asarray(reloaded, dtype=np.float32) / 255.0

    u = apply_jpeg(x_nb, jpeg_quality)

    # 2. Sequence BNJ: Blur -> Noise -> JPEG (EXACT SAME EPSILON)
    x_b = apply_blur(img_np)
    x_bn = np.clip(x_b + epsilon, 0.0, 1.0)
    v = apply_jpeg(x_bn, jpeg_quality)

    d = u - v
    c = (u + v) / 2.0
    return u, v, d, c

def audit_image(img_path, seed=42):
    img = Image.open(img_path).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    img_np = np.asarray(img, dtype=np.float32) / 255.0

    u, v, d, c = simulate_nbj_bnj(img_np, seed=seed)

    # Saturation in anchor c
    sat_zeros = (c <= 1e-4)
    sat_ones = (c >= 1.0 - 1e-4)
    sat_total = np.logical_or(sat_zeros, sat_ones)
    frac_saturated = float(np.mean(sat_total))

    # Min headroom in c
    headroom = np.minimum(c, 1.0 - c)
    min_headroom = float(np.min(headroom))
    mean_headroom = float(np.mean(headroom))
    q01_headroom = float(np.percentile(headroom, 0.1))

    # Residual magnitude
    max_d = float(np.max(np.abs(d)))
    mean_d = float(np.mean(np.abs(d)))
    rms_d = float(np.sqrt(np.mean(d ** 2)))

    # Compute alpha bounds across a uniform sample of 32 spatial translations
    translations = []
    # Grid of 32 uniformly spaced translations
    ys = np.linspace(0, 223, 6, dtype=int)
    xs = np.linspace(0, 223, 6, dtype=int)
    for y in ys:
        for x in xs:
            if len(translations) < 32:
                translations.append((int(y), int(x)))

    alpha_max_list = []
    alpha_q999_list = []

    for ty, tx in translations:
        d_tau = np.roll(np.roll(d, ty, axis=0), tx, axis=1)
        nonzero = np.abs(d_tau) > 1e-5
        if np.any(nonzero):
            bounds = 2.0 * headroom[nonzero] / np.abs(d_tau[nonzero])
            alpha_max_list.append(float(np.min(bounds)))
            alpha_q999_list.append(float(np.percentile(bounds, 0.1)))
        else:
            alpha_max_list.append(10.0)
            alpha_q999_list.append(10.0)

    # Interiorized anchor control c_eta with eta = 0.05
    eta = 0.05
    c_eta = (1.0 - 2.0 * eta) * c + eta
    headroom_eta = np.minimum(c_eta, 1.0 - c_eta)
    min_headroom_eta = float(np.min(headroom_eta))
    
    alpha_max_eta_list = []
    for ty, tx in translations:
        d_tau = np.roll(np.roll(d, ty, axis=0), tx, axis=1)
        nonzero = np.abs(d_tau) > 1e-5
        if np.any(nonzero):
            bounds_eta = 2.0 * headroom_eta[nonzero] / np.abs(d_tau[nonzero])
            alpha_max_eta_list.append(float(np.min(bounds_eta)))
        else:
            alpha_max_eta_list.append(10.0)

    return {
        "filename": os.path.basename(img_path),
        "frac_saturated_anchor_pixels": frac_saturated,
        "min_anchor_headroom": min_headroom,
        "mean_anchor_headroom": mean_headroom,
        "q01_anchor_headroom": q01_headroom,
        "max_residual_d": max_d,
        "mean_residual_d": mean_d,
        "rms_residual_d": rms_d,
        "min_feasible_alpha_unconstrained": float(np.min(alpha_max_list)),
        "median_feasible_alpha_unconstrained": float(np.median(alpha_max_list)),
        "feasible_alpha_q01_headroom": float(np.min(alpha_q999_list)),
        "min_feasible_alpha_interior_eta005": float(np.min(alpha_max_eta_list)),
        "median_feasible_alpha_interior_eta005": float(np.median(alpha_max_eta_list)),
    }

def main():
    paths = get_image_paths()
    print(f"Found {len(paths)} images for audit.")
    results = []
    for p in paths:
        res = audit_image(p)
        results.append(res)
        print(f"Image {res['filename']}: Saturated={res['frac_saturated_anchor_pixels']*100:.3f}%, "
              f"Max|d|={res['max_residual_d']:.4f}, MinAlpha={res['min_feasible_alpha_unconstrained']:.4f}, "
              f"MinAlpha(eta=0.05)={res['min_feasible_alpha_interior_eta005']:.4f}")

    out_path = r"research\phase3_cins\pixel_validity_audit_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
