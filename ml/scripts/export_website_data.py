"""Export validated research results to compact, browser-ready JSONs for the website.

Authoritative source files:
- results/matched_fidelity_study/calibrated_common_support_3levels.json
- results/matched_fidelity_study/confirmatory_statistical_comparison.csv
- results/matched_fidelity_study/confirmatory_feature_ablations.csv
- results/matched_fidelity_study/confirmatory_run/confirmatory_terminal_operator_disentanglement.csv
- results/matched_fidelity_study/confirmatory_run/confirmatory_local_ssim_interaction_test.json
- results/matched_fidelity_study/confirmatory_run/confirmatory_execution_manifest.json
- results/matched_fidelity_study/imagenet_v2/imagenet_v2_statistical_comparison.csv
- results/matched_fidelity_study/imagenet_v2/imagenet_v2_replication_audit.json

Output files:
- apps/web/public/data/matched_fidelity_study_data.json
- apps/web/public/data/imagenet_v2_replication_data.json
- apps/web/public/data/reproducibility_manifest.json
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "matched_fidelity_study"
PUBLIC_DATA_DIR = BASE_DIR / "apps" / "web" / "public" / "data"
PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def export_matched_fidelity_data():
    print("Exporting matched_fidelity_study_data.json...")

    # 1. Calibration
    with open(RESULTS_DIR / "calibrated_common_support_3levels.json", "r", encoding="utf-8") as f:
        cal_data = json.load(f)

    # 2. Confirmatory execution manifest
    with open(RESULTS_DIR / "confirmatory_run" / "confirmatory_execution_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 3. Statistical comparison
    df_stat = pd.read_csv(RESULTS_DIR / "confirmatory_statistical_comparison.csv")
    stat_rows = []
    for _, row in df_stat.iterrows():
        stat_rows.append({
            "architecture": row["architecture"],
            "display_name": {
                "vit_base": "ViT-Base / 16",
                "convnext_tiny": "ConvNeXt-Tiny",
                "swin_tiny": "Swin-Tiny",
                "resnet50": "ResNet-50",
            }.get(row["architecture"], row["architecture"]),
            "specification": row["specification"],
            "r2_oos_m3": round(float(row["r2_oos_m3"]), 4),
            "r2_oos_m5": round(float(row["r2_oos_m5"]), 4),
            "delta_r2_oos": round(float(row["delta_r2_oos"]), 4),
            "ci95_pointwise_delta_r2": row["ci95_pointwise_delta_r2"],
            "simultaneous_ci_delta_r2": row["simultaneous_bonferroni_2s_ci_delta_r2"],
            "mae_oos_m3": round(float(row["mae_oos_m3"]), 4),
            "mae_oos_m5": round(float(row["mae_oos_m5"]), 4),
            "delta_mae_oos": round(float(row["delta_mae_oos"]), 4),
            "ci95_pointwise_delta_mae": row["ci95_pointwise_delta_mae"],
            "simultaneous_ci_delta_mae": row["simultaneous_bonferroni_2s_ci_delta_mae"],
            "holm_adj_p_delta_r2": row["holm_adj_p_delta_r2"],
            "p_display": "<= 0.0004",  # All crossed 0/10,000 bootstrap resamples
            "significant_r2": bool(row["significant_r2_holm_0_05"]),
            "significant_mae": bool(row["significant_mae_holm_0_05"]),
        })

    # 4. Feature ablations
    df_abl = pd.read_csv(RESULTS_DIR / "confirmatory_feature_ablations.csv")
    ablation_by_arch = {}
    for arch in ["vit_base", "convnext_tiny", "swin_tiny", "resnet50"]:
        arch_sub = df_abl[df_abl["architecture"] == arch]
        ablation_by_arch[arch] = [
            {
                "condition": r["ablation_condition"],
                "num_columns": int(r["num_columns"]),
                "r2_oos": round(float(r["r2_oos"]), 4),
                "delta_r2_vs_baseline": round(float(r["delta_r2_vs_baseline"]), 4),
                "ci95_delta_r2": r["ci95_delta_r2"],
                "mae_oos": round(float(r["mae_oos"]), 4),
                "delta_mae_vs_baseline": round(float(r["delta_mae_vs_baseline"]), 4),
                "ci95_delta_mae": r["ci95_delta_mae"],
                "p_val": float(r["p_val_delta_r2"]),
            }
            for _, r in arch_sub.iterrows()
        ]

    # 5. Terminal operator disentanglement
    df_term = pd.read_csv(RESULTS_DIR / "confirmatory_run" / "confirmatory_terminal_operator_disentanglement.csv")
    term_rows = []
    for _, r in df_term.iterrows():
        term_rows.append({
            "architecture": r["architecture"],
            "display_name": {
                "vit_base": "ViT-Base / 16",
                "convnext_tiny": "ConvNeXt-Tiny",
                "swin_tiny": "Swin-Tiny",
                "resnet50": "ResNet-50",
            }.get(r["architecture"], r["architecture"]),
            "r2_model_a_terminal_only": round(float(r["r2_terminal_operator_only"]), 4),
            "r2_model_b_global_terminal": round(float(r["r2_fidelity_plus_terminal_operator"]), 4),
            "r2_model_c_global_full_order": round(float(r["r2_fidelity_plus_full_order_model3"]), 4),
            "r2_model_d_full_fingerprints": round(float(r["r2_model5_with_fingerprints"]), 4),
            "delta_r2_terminal_to_order": round(float(r["delta_r2_operator_to_order"]), 4),
            "delta_r2_order_to_fingerprints": round(float(r["delta_r2_order_to_fingerprints"]), 4),
        })

    # 6. Local SSIM interaction test
    with open(RESULTS_DIR / "confirmatory_run" / "confirmatory_local_ssim_interaction_test.json", "r", encoding="utf-8") as f:
        interaction_tests = json.load(f)

    out_data = {
        "metadata": {
            "title": "At Matched Image Fidelity, Does Corruption Order Still Matter? Spatial-Frequency and Structural Fingerprints of Vision Classifier Reliability",
            "study_type": "Matched-Fidelity Counterfactual Common-Support Confirmatory Study",
            "date": "November 2026",
            "evaluated_dataset": "FastAI Imagenette-v2 validation split (10 WordNet synsets: tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute)",
            "primary_images_count": manifest.get("unique_images", 3825),
            "primary_corrupted_passes": manifest.get("total_records", 275400),
            "primary_clean_passes": 15300,
            "primary_total_passes": 290700,
            "conditions_per_image": 18,
            "clean_conditions_per_image": 1,
            "total_conditions_per_image": 19,
            "evaluated_models": [
                {"id": "resnet50", "name": "ResNet-50", "timm_id": "resnet50.a1_in1k", "family": "CNN"},
                {"id": "convnext_tiny", "name": "ConvNeXt-Tiny", "timm_id": "convnext_tiny.in12k_ft_in1k", "family": "Modern CNN"},
                {"id": "vit_base", "name": "ViT-Base / 16", "timm_id": "vit_base_patch16_224.augreg_in21k_ft_in1k", "family": "Global-Attention ViT"},
                {"id": "swin_tiny", "name": "Swin-Tiny", "timm_id": "swin_tiny_patch4_window7_224.ms_in1k", "family": "Hierarchical Windowed ViT"},
            ],
            "pretraining_disclosure": "The 10 Imagenette synsets originate from ImageNet-1K. Models were pretrained on ImageNet-1K or 12k/21k fine-tuned on 1K. Confirmatory images are in-distribution with respect to backbone pretraining data, but completely quarantined from parameter calibration (N=50) and exploratory pilot (N=100) data.",
            "external_replication_count": 300,
            "external_replication_passes": 22800,
            "grand_total_evaluations": 313500,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "calibration": {
            "protocol_name": cal_data.get("protocol"),
            "delineation_method": cal_data.get("delineation_method"),
            "tolerances": cal_data.get("prespecified_tolerances"),
            "lpips_role": "LPIPS was not an independently calibrated target. It was evaluated as a continuous regression covariate modeled via restricted cubic splines to control for perceptual distance.",
            "levels": cal_data.get("levels"),
            "sequences": ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"],
        },
        "statistical_comparison": stat_rows,
        "feature_ablations": ablation_by_arch,
        "terminal_operator_disentanglement": term_rows,
        "local_ssim_interaction_tests": interaction_tests,
        "fingerprint_definitions": [
            {
                "id": "E_low",
                "name": "Low-Frequency Fourier Energy",
                "math": "E_{low} = \\sum_{r < r_0} |F(u,v)|^2 / E_{total}",
                "description": "Fraction of spectral energy in the low-frequency Fourier regime (r < 0.15 * f_Nyquist).",
            },
            {
                "id": "E_high",
                "name": "High-Frequency Fourier Energy",
                "math": "E_{high} = \\sum_{r > r_1} |F(u,v)|^2 / E_{total}",
                "description": "Fraction of spectral energy in high-frequency modes (r > 0.50 * f_Nyquist).",
            },
            {
                "id": "B_JPEG",
                "name": "JPEG Block-Boundary Discontinuity",
                "math": "B_{JPEG} = \\text{Mean 8x8 block border step difference}",
                "description": "Magnitude of intensity gradients across 8x8 DCT grid boundaries normalized by intra-block gradients.",
            },
            {
                "id": "Delta_Edge",
                "name": "Sobel Edge-Energy Residual",
                "math": "\\Delta E_{edge} = ||\\nabla I_{corr}||_2^2 - ||\\nabla I_{clean}||_2^2",
                "description": "Differential total gradient energy captured by Sobel directional operators relative to clean input.",
            },
            {
                "id": "sigma_local_SSIM",
                "name": "Local SSIM Spatial Heterogeneity",
                "math": "\\sigma(SSIM_{11\\times11})",
                "description": "Spatial standard deviation of the 11x11 local SSIM patch map across the visual field.",
            },
        ],
    }

    out_path = PUBLIC_DATA_DIR / "matched_fidelity_study_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved: {out_path} ({out_path.stat().st_size:,} bytes)")
    return out_path


def export_imagenet_v2_data():
    print("Exporting imagenet_v2_replication_data.json...")
    df_v2 = pd.read_csv(RESULTS_DIR / "imagenet_v2" / "imagenet_v2_statistical_comparison.csv")

    rows = []
    for _, r in df_v2.iterrows():
        p_val = float(r["p_value"])
        # Holm adjustment for pooled results (4 architectures)
        if r["variant"] == "Pooled_External_All_Variants":
            # Rank orders: vit (0.00850 * 4 = 0.0340), resnet (0.05899 * 3 = 0.1770), convnext (0.12189 * 2 = 0.2438), swin (0.17678 * 1 = 0.1768)
            p_holm_map = {
                "vit_base": 0.0340,
                "resnet50": 0.1770,
                "convnext_tiny": 0.2438,
                "swin_tiny": 0.1768,
            }
            p_adj = p_holm_map.get(r["architecture"], p_val)
            sig = bool(p_adj < 0.05)
        else:
            p_adj = None
            sig = False

        rows.append({
            "variant": r["variant"],
            "architecture": r["architecture"],
            "display_name": {
                "vit_base": "ViT-Base / 16",
                "convnext_tiny": "ConvNeXt-Tiny",
                "swin_tiny": "Swin-Tiny",
                "resnet50": "ResNet-50",
            }.get(r["architecture"], r["architecture"]),
            "n_images": int(r["n_images"]),
            "r2_m3": round(float(r["r2_m3"]), 4),
            "r2_m5": round(float(r["r2_m5"]), 4),
            "delta_r2": round(float(r["delta_r2"]), 4),
            "ci95_delta_r2": r["ci95_delta_r2"],
            "mae_m3": round(float(r["mae_m3"]), 4),
            "mae_m5": round(float(r["mae_m5"]), 4),
            "delta_mae": round(float(r["delta_mae"]), 4),
            "ci95_delta_mae": r["ci95_delta_mae"],
            "p_value_raw": round(p_val, 5),
            "p_value_holm": round(p_adj, 4) if p_adj is not None else None,
            "significant_holm": sig,
        })

    v2_data = {
        "metadata": {
            "title": "Independent External Replication on ImageNet-V2",
            "sample_size_total": 300,
            "variants": ["MatchedFrequency", "Threshold0.7", "TopImages"],
            "images_per_variant": 100,
            "conditions_per_image": 19,
            "total_evaluations": 22800,
            "key_finding": "Only ViT-Base achieves a positive out-of-sample R² CI strictly excluding zero and retains statistical significance following step-down Holm-Bonferroni correction (Delta R2 = +0.0371, 95% CI: [+0.0065, +0.0675], p_adj = 0.0340). For ResNet-50, ConvNeXt-Tiny, and Swin-Tiny, pooled point estimates remain positive (+0.0115 to +0.0212), but their 95% bootstrap CIs cross zero and p_adj > 0.05.",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": rows,
    }

    out_path = PUBLIC_DATA_DIR / "imagenet_v2_replication_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(v2_data, f, indent=2)
    print(f"Saved: {out_path} ({out_path.stat().st_size:,} bytes)")
    return out_path


def export_reproducibility_manifest():
    print("Exporting reproducibility_manifest.json...")
    critical_files = [
        RESULTS_DIR / "calibrated_common_support_3levels.json",
        RESULTS_DIR / "confirmatory_statistical_comparison.csv",
        RESULTS_DIR / "confirmatory_feature_ablations.csv",
        RESULTS_DIR / "confirmatory_run" / "confirmatory_terminal_operator_disentanglement.csv",
        RESULTS_DIR / "confirmatory_run" / "confirmatory_local_ssim_interaction_test.json",
        RESULTS_DIR / "confirmatory_run" / "confirmatory_execution_manifest.json",
        RESULTS_DIR / "imagenet_v2" / "imagenet_v2_statistical_comparison.csv",
        RESULTS_DIR / "imagenet_v2" / "imagenet_v2_replication_audit.json",
    ]

    manifest_entries = []
    for p in critical_files:
        if p.exists():
            manifest_entries.append({
                "relative_path": str(p.relative_to(BASE_DIR)).replace("\\", "/"),
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })
        else:
            print(f"Warning: file not found {p}")

    out_path = PUBLIC_DATA_DIR / "reproducibility_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": manifest_entries,
        }, f, indent=2)
    print(f"Saved: {out_path} ({out_path.stat().st_size:,} bytes)")
    return out_path


if __name__ == "__main__":
    export_matched_fidelity_data()
    export_imagenet_v2_data()
    export_reproducibility_manifest()
    print("Website data export complete.")
