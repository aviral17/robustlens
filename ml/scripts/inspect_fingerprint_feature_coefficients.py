"""Inspect Model 5 fingerprint regression coefficients across folds."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_confirmatory_complete_statistical_analysis import build_design_matrix

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONF_DIR = BASE_DIR / "results" / "matched_fidelity_study" / "confirmatory_run"
CSV_GZ_PATH = CONF_DIR / "full_distortion_fingerprints_and_margins.csv.gz"

df = pd.read_csv(CSV_GZ_PATH)
clusters = np.sort(df["image_id"].unique())
cluster_classes = df.drop_duplicates("image_id").sort_values("image_id")["true_label"].to_numpy()

rng_cv = np.random.RandomState(42)
fold_assignments = np.zeros(len(clusters), dtype=int)
for c in range(10):
    c_indices = np.where(cluster_classes == c)[0]
    perm = rng_cv.permutation(c_indices)
    c_splits = np.array_split(perm, 5)
    for fold_i, split_idx in enumerate(c_splits):
        fold_assignments[split_idx] = fold_i

architectures = ["convnext_tiny", "vit_base", "swin_tiny", "resnet50"]
cont_cols = ["psnr", "ssim", "lpips", "fft_energy_low", "fft_energy_high", "jpeg_block_artifact", "delta_edge_energy", "std_local_ssim"]
fp_names = ["fft_energy_low", "fft_energy_high", "jpeg_block_artifact", "delta_edge_energy", "std_local_ssim"]

coef_summary = []

for arch in architectures:
    df_arch = df[df["model"] == arch].copy().sort_values(["image_id", "level", "order"]).reset_index(drop=True)
    arch_fp_coefs = []
    
    for fold_i in range(5):
        test_c = clusters[fold_assignments == fold_i]
        train_c = clusters[fold_assignments != fold_i]
        tr_df = df_arch[df_arch["image_id"].isin(train_c)]
        
        scaler_params = {}
        for col in cont_cols:
            vals = tr_df[col].to_numpy()
            scaler_params[col] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) + 1e-8}
        
        spline_knots = {}
        for col in ["psnr", "ssim", "lpips"]:
            std_v = (tr_df[col].to_numpy() - scaler_params[col]["mean"]) / scaler_params[col]["std"]
            spline_knots[col] = np.percentile(std_v, [5.0, 35.0, 65.0, 95.0])
        
        y_tr = tr_df["logit_margin"].to_numpy()
        X_tr_c5 = build_design_matrix(tr_df, "conv_extended", scaler_params, spline_knots)
        b_c5 = np.linalg.pinv(X_tr_c5.T @ X_tr_c5) @ (X_tr_c5.T @ y_tr)
        
        # Last 5 coefficients correspond to fingerprints
        arch_fp_coefs.append(b_c5[-5:])
    
    mean_coefs = np.mean(arch_fp_coefs, axis=0)
    std_coefs = np.std(arch_fp_coefs, axis=0)
    
    row = {"architecture": arch}
    for name, m, s in zip(fp_names, mean_coefs, std_coefs):
        row[f"{name}_mean"] = round(float(m), 4)
        row[f"{name}_std"] = round(float(s), 4)
    coef_summary.append(row)

df_c = pd.DataFrame(coef_summary)
print("\nModel 5 (Conventional RCS) Fingerprint Coefficients (Mean across 5 Folds):")
print(df_c.to_string(index=False))

out_p = CONF_DIR / "confirmatory_fingerprint_coefficients.json"
with open(out_p, "w") as fp:
    json.dump(coef_summary, fp, indent=2)
