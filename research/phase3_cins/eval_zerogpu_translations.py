"""
Evaluates genuine neural vision models on Hugging Face ZeroGPU (NVIDIA A10G)
across translated intervention endpoints to measure:
1. Actual neural response variation across translations.
2. Sign agreement and extreme ranking with predicted responses.
3. Architecture differences (ResNet-50, ConvNeXt-Tiny, ViT-Base, Swin-Tiny).
"""

import os
import sys
import json
import time
import base64
import urllib.request
from io import BytesIO
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cins_engine import (
    generate_noise_blur_jpeg_pair,
    circular_shift,
    compute_fft_response_map
)

def numpy_to_base64_jpeg(arr_np):
    arr_uint8 = np.clip(arr_np * 255.0, 0.0, 255.0).astype(np.uint8)
    pil_img = Image.fromarray(arr_uint8)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return "data:image/jpeg;base64," + base64.b64encode(buf.read()).decode("utf-8")

def query_zerogpu(img_b64):
    req = urllib.request.Request(
        "https://robustlens.vercel.app/api/inference",
        data=json.dumps({"image": img_b64, "model": "all", "corruption": "none", "severity": 1}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=45) as res:
        data = json.loads(res.read().decode("utf-8"))
        return data.get("result", {})

def run_zerogpu_eval():
    img_path = r"apps\web\public\examples\img_0_clean.jpg"
    img = Image.open(img_path).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    img_np = np.asarray(img, dtype=np.float32) / 255.0

    u, v, d, c, eps = generate_noise_blur_jpeg_pair(img_np, seed=42)

    # Use interior anchor eta=0.05 for guaranteed headroom
    eta = 0.05
    c_eta = (1.0 - 2.0 * eta) * c + eta

    # Sobel gradient proxy
    lum = 0.2989 * c_eta[:, :, 0] + 0.5870 * c_eta[:, :, 1] + 0.1140 * c_eta[:, :, 2]
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    sx = np.zeros_like(lum)
    sy = np.zeros_like(lum)
    for i in range(1, 223):
        for j in range(1, 223):
            patch = lum[i-1:i+2, j-1:j+2]
            sx[i, j] = np.sum(patch * kx)
            sy[i, j] = np.sum(patch * ky)
    s_mag = np.sqrt(sx**2 + sy**2)
    proxy_grad = np.repeat(s_mag[:, :, np.newaxis], 3, axis=2)

    fft_map = compute_fft_response_map(proxy_grad, d, alpha=1.0)
    flat_max = np.argmax(fft_map)
    flat_min = np.argmin(fft_map)
    tau_max = (int(flat_max // 224), int(flat_max % 224))
    tau_min = (int(flat_min // 224), int(flat_min % 224))

    tested_shifts = [
        {"name": "unshifted_origin", "tau": (0, 0)},
        {"name": "predicted_max", "tau": tau_max},
        {"name": "predicted_min", "tau": tau_min},
        {"name": "grid_shift_44", "tau": (44, 44)},
        {"name": "grid_shift_88", "tau": (88, 88)},
        {"name": "grid_shift_132", "tau": (132, 132)},
    ]

    alpha = 0.25 # moderate intervention strength
    records = []

    print(f"Querying ZeroGPU NVIDIA A10G for {len(tested_shifts)} translation pairs at alpha={alpha}...")

    # First evaluate anchor baseline c_eta
    c_b64 = numpy_to_base64_jpeg(c_eta)
    anchor_res = query_zerogpu(c_b64)
    anchor_models = {m["model"]: m for m in anchor_res.get("results", [])}
    print("Anchor Baseline Predictions:")
    for m_name, m_data in anchor_models.items():
        print(f"  {m_name}: {m_data['prediction']} ({m_data['confidence']*100:.1f}%)")

    for shift in tested_shifts:
        ty, tx = shift["tau"]
        d_tau = circular_shift(d, ty, tx)
        x_plus = c_eta + 0.5 * alpha * d_tau
        x_minus = c_eta - 0.5 * alpha * d_tau

        print(f"\nEvaluating shift: {shift['name']} tau=({ty}, {tx})...")
        t0 = time.time()
        res_plus = query_zerogpu(numpy_to_base64_jpeg(x_plus))
        res_minus = query_zerogpu(numpy_to_base64_jpeg(x_minus))
        eval_time = time.time() - t0

        models_plus = {m["model"]: m for m in res_plus.get("results", [])}
        models_minus = {m["model"]: m for m in res_minus.get("results", [])}

        pred_val = float(fft_map[ty, tx] * alpha)

        shift_record = {
            "shift_name": shift["name"],
            "tau_y": ty,
            "tau_x": tx,
            "predicted_response_proxy": pred_val,
            "eval_time_sec": eval_time,
            "models": {}
        }

        for m_name in ["ResNet-50", "ConvNeXt-Tiny", "ViT-Base", "Swin-Tiny"]:
            p_data = models_plus.get(m_name, {})
            m_data = models_minus.get(m_name, {})
            conf_plus = p_data.get("confidence", 0.0)
            conf_minus = m_data.get("confidence", 0.0)
            conf_delta = conf_plus - conf_minus

            shift_record["models"][m_name] = {
                "pred_plus": p_data.get("prediction"),
                "conf_plus": conf_plus,
                "pred_minus": m_data.get("prediction"),
                "conf_minus": conf_minus,
                "actual_response_conf_delta": conf_delta
            }
            print(f"  [{m_name}] Plus: {p_data.get('prediction')} ({conf_plus*100:.1f}%), "
                  f"Minus: {m_data.get('prediction')} ({conf_minus*100:.1f}%), Delta: {conf_delta*100:+.2f}%")

        records.append(shift_record)

    out_file = r"research\phase3_cins\zerogpu_empirical_translations.json"
    with open(out_file, "w") as f:
        json.dump({
            "image": "img_0_clean.jpg (Tench)",
            "alpha": alpha,
            "anchor_baseline": anchor_models,
            "evaluations": records
        }, f, indent=2)
    print(f"\nEmpirical ZeroGPU results saved to {out_file}")

if __name__ == "__main__":
    run_zerogpu_eval()
