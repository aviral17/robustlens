"""Export full validated order effects benchmark artifacts to apps/web/public/ for the interactive dashboard."""

import os
import shutil
import json
import pandas as pd
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FULL_OUTPUTS = os.path.join(REPO_ROOT, "results", "order_effect_study", "full", "outputs")
WEB_PUBLIC = os.path.join(REPO_ROOT, "apps", "web", "public")
WEB_FIGURES = os.path.join(WEB_PUBLIC, "figures")

os.makedirs(WEB_PUBLIC, exist_ok=True)
os.makedirs(WEB_FIGURES, exist_ok=True)

def main():
    print("Reading full benchmark outputs...")
    df_summary = pd.read_csv(os.path.join(FULL_OUTPUTS, "order_effects_summary.csv"))
    df_leaderboard = pd.read_csv(os.path.join(FULL_OUTPUTS, "order_effects_leaderboard.csv"))
    df_pairwise = pd.read_csv(os.path.join(FULL_OUTPUTS, "order_pairwise_significance.csv"))
    df_ancova = pd.read_csv(os.path.join(FULL_OUTPUTS, "distortion_ancova.csv"))
    factorial_ancova_path = os.path.join(FULL_OUTPUTS, "factorial_ancova_results.csv")
    df_factorial_ancova = pd.read_csv(factorial_ancova_path) if os.path.exists(factorial_ancova_path) else None
    
    with open(os.path.join(FULL_OUTPUTS, "order_omnibus_tests.json"), "r") as f:
        omnibus_tests = json.load(f)
    with open(os.path.join(FULL_OUTPUTS, "manifest.json"), "r") as f:
        manifest = json.load(f)

    # 1. Sequence Rankings aggregated across all 4 architectures
    compound_only = df_summary[df_summary["order"] != "clean"]
    order_rankings = []
    
    order_names = {
        "NJB": "Noise -> JPEG -> Blur",
        "JNB": "JPEG -> Noise -> Blur",
        "NBJ": "Noise -> Blur -> JPEG",
        "JBN": "JPEG -> Blur -> Noise",
        "BJN": "Blur -> JPEG -> Noise",
        "BNJ": "Blur -> Noise -> JPEG",
    }
    
    for order_code, grp in compound_only.groupby("order"):
        order_rankings.append({
            "order": order_code,
            "description": order_names.get(order_code, order_code),
            "mean_accuracy": round(float(grp["accuracy"].mean() * 100.0), 2),
            "mean_confidence": round(float(grp["mean_confidence"].mean() * 100.0), 2),
            "mean_ece": round(float(grp["ece"].mean()), 4),
            "mean_psnr": round(float(grp["mean_psnr"].mean()), 2),
            "mean_ssim": round(float(grp["mean_ssim"].mean()), 3),
        })
    order_rankings.sort(key=lambda x: x["mean_accuracy"], reverse=True)

    # 2. Per-model severity degradation curves
    severity_curves = {}
    for model_key in ["convnext_tiny", "swin_tiny", "vit_base", "resnet50"]:
        severity_curves[model_key] = {}
        sub = compound_only[compound_only["model"] == model_key]
        for ord_code in ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"]:
            ord_sub = sub[sub["order"] == ord_code].sort_values("severity")
            severity_curves[model_key][ord_code] = [
                {
                    "severity": int(r["severity"]),
                    "accuracy": round(float(r["accuracy"] * 100.0), 2),
                    "ece": round(float(r["ece"]), 4),
                    "psnr": round(float(r["mean_psnr"]), 2),
                    "ssim": round(float(r["mean_ssim"]), 3),
                }
                for _, r in ord_sub.iterrows()
            ]

    # Load Error Churn artifacts
    churn_report_path = os.path.join(FULL_OUTPUTS, "error_churn_report.json")
    churn_high_path = os.path.join(FULL_OUTPUTS, "error_churn_high_excess_cases.csv")
    churn_summary_path = os.path.join(FULL_OUTPUTS, "error_churn_summary.csv")

    churn_report = {}
    if os.path.exists(churn_report_path):
        with open(churn_report_path, "r") as f:
            churn_report = json.load(f)

    df_churn_high = pd.read_csv(churn_high_path) if os.path.exists(churn_high_path) else None
    df_churn_summary = pd.read_csv(churn_summary_path) if os.path.exists(churn_summary_path) else None

    # Load Matched Fidelity artifacts if available
    matched_outputs = os.path.join(REPO_ROOT, "results", "order_effect_study", "matched_fidelity", "outputs")
    matched_data = None
    if os.path.exists(matched_outputs):
        m_lead_path = os.path.join(matched_outputs, "order_effects_leaderboard.csv")
        m_ancova_path = os.path.join(matched_outputs, "matched_factorial_ancova.csv")
        m_comp_path = os.path.join(matched_outputs, "nominal_vs_matched_summary.csv")
        m_churn_path = os.path.join(matched_outputs, "error_churn_summary.csv")
        m_summary_path = os.path.join(matched_outputs, "order_effects_summary.csv")

        if os.path.exists(m_lead_path) and os.path.exists(m_summary_path):
            df_m_lead = pd.read_csv(m_lead_path)
            df_m_sum = pd.read_csv(m_summary_path)
            df_m_ancova = pd.read_csv(m_ancova_path) if os.path.exists(m_ancova_path) else None
            df_m_comp = pd.read_csv(m_comp_path) if os.path.exists(m_comp_path) else None
            df_m_churn = pd.read_csv(m_churn_path) if os.path.exists(m_churn_path) else None

            # Rank order sequences under matched fidelity
            m_comp_only = df_m_sum[df_m_sum["order"] != "clean"]
            m_rankings = []
            for order_code, grp in m_comp_only.groupby("order"):
                m_rankings.append({
                    "order": order_code,
                    "description": order_names.get(order_code, order_code),
                    "mean_accuracy": round(float(grp["accuracy"].mean() * 100.0), 2),
                    "mean_psnr": round(float(grp["mean_psnr"].mean()), 2),
                    "mean_ssim": round(float(grp["mean_ssim"].mean()), 3),
                })
            m_rankings.sort(key=lambda x: x["mean_accuracy"], reverse=True)

            matched_data = {
                "leaderboard": df_m_lead.to_dict(orient="records"),
                "sequence_rankings": m_rankings,
                "factorial_ancova": df_m_ancova.to_dict(orient="records") if df_m_ancova is not None else [],
                "comparison_summary": df_m_comp.to_dict(orient="records") if df_m_comp is not None else [],
                "churn_summary": df_m_churn.head(50).to_dict(orient="records") if df_m_churn is not None else [],
            }

    # Structure complete payload
    web_data = {
        "meta": {
            "title": "RobustLens Compound Corruption Order Effects Benchmark",
            "benchmark_status": "Full Benchmark — N=3,925 (Validated)",
            "dataset": "Imagenette-v2 (Validation Split, 10 Canonical Classes)",
            "n_images": 3925,
            "n_conditions": 31,
            "n_models": 4,
            "total_records": 486700,
            "hardware": "NVIDIA Tesla T4 GPU",
            "inference_cluster_unit": "Physical Image (sample_idx)",
            "generated_at": manifest.get("date", ""),
        },
        "leaderboard": df_leaderboard.to_dict(orient="records"),
        "sequence_rankings": order_rankings,
        "condition_summary": df_summary.to_dict(orient="records"),
        "severity_curves": severity_curves,
        "distortion_ancova": df_ancova.to_dict(orient="records"),
        "factorial_ancova": df_factorial_ancova.to_dict(orient="records") if df_factorial_ancova is not None else [],
        "omnibus_tests": omnibus_tests,
        "pairwise_significance": df_pairwise.to_dict(orient="records"),
        "error_churn": {
            "report": churn_report,
            "high_excess_cases": df_churn_high.to_dict(orient="records") if df_churn_high is not None else [],
            "all_cases": df_churn_summary.to_dict(orient="records") if df_churn_summary is not None else [],
        },
        "matched_fidelity": matched_data
    }

    out_json = os.path.join(WEB_PUBLIC, "order_effects_data.json")
    with open(out_json, "w") as f:
        json.dump(web_data, f, indent=2)
    print(f"Exported dashboard JSON to {out_json} ({os.path.getsize(out_json):,} bytes)")

    # Copy publication figures to apps/web/public/figures/
    fig_src = os.path.join(REPO_ROOT, "reports", "figures")
    for f in os.listdir(fig_src):
        if f.endswith(".png"):
            src_f = os.path.join(fig_src, f)
            dst_f = os.path.join(WEB_FIGURES, f)
            shutil.copy2(src_f, dst_f)
            print(f"  Copied figure: {f} -> {dst_f}")

    print("Dashboard data sync complete!")

if __name__ == "__main__":
    main()
