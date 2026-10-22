"""Unit tests for statistical report generation and result schemas."""

import numpy as np
import pytest
from robustlens.eval.reporting import compute_metrics


def test_compute_metrics_schema():
    """Verify that compute_metrics produces complete, statistically sound output schemas."""
    # Synthetic toy predictions with 2 models across clean and 1 corruption
    toy_records = []
    models = [("resnet50", "ResNet-50", "CNN"), ("vit_base", "ViT-Base/16", "Transformer")]

    np.random.seed(42)
    for sample_idx in range(20):
        true_lbl = sample_idx % 10
        for m_key, m_name, m_fam in models:
            # Clean condition
            pred_clean = true_lbl if np.random.rand() > 0.1 else (true_lbl + 1) % 10
            toy_records.append({
                "sample_idx": sample_idx,
                "model": m_key,
                "model_name": m_name,
                "model_family": m_fam,
                "corruption": "clean",
                "severity": 0,
                "true_label": true_lbl,
                "true_class_name": f"class_{true_lbl}",
                "pred_label": pred_clean,
                "pred_class_name": f"class_{pred_clean}",
                "confidence": 0.95 if pred_clean == true_lbl else 0.40,
                "correct": bool(pred_clean == true_lbl),
                "top_5_classes": [f"class_{i}" for i in range(5)],
                "top_5_probs": [0.6, 0.2, 0.1, 0.05, 0.05],
                "latency_ms": 15.0,
            })

            # Shifted condition
            pred_shift = true_lbl if np.random.rand() > 0.5 else (true_lbl + 2) % 10
            toy_records.append({
                "sample_idx": sample_idx,
                "model": m_key,
                "model_name": m_name,
                "model_family": m_fam,
                "corruption": "gaussian_noise",
                "severity": 3,
                "true_label": true_lbl,
                "true_class_name": f"class_{true_lbl}",
                "pred_label": pred_shift,
                "pred_class_name": f"class_{pred_shift}",
                "confidence": 0.70 if pred_shift == true_lbl else 0.35,
                "correct": bool(pred_shift == true_lbl),
                "top_5_classes": [f"class_{i}" for i in range(5)],
                "top_5_probs": [0.4, 0.3, 0.15, 0.1, 0.05],
                "latency_ms": 15.0,
            })

    report = compute_metrics(toy_records, n_bootstraps=200, alpha=0.05, seed=42)

    # Schema checks
    assert "leaderboard" in report
    assert "pairwise_significance" in report
    assert "conformal_diagnostics" in report
    assert "detailed_breakdown" in report

    # Leaderboard checks
    assert len(report["leaderboard"]) == 2
    for lb in report["leaderboard"]:
        assert 0.0 <= lb["clean_accuracy"] <= 1.0
        assert 0.0 <= lb["mean_shifted_accuracy"] <= 1.0
        assert "mean_ece" in lb
        assert "status" in lb
        assert "num_independent_images" in lb

    # Configuration checks
    assert report["configuration"]["num_independent_images"] == 20
    assert report["configuration"]["num_records"] == 80
    assert "Pilot" in report["configuration"]["benchmark_status"]


    # Pairwise tests checks
    assert len(report["pairwise_significance"]) == 1
    pair = report["pairwise_significance"][0]
    assert 0.0 <= pair["mcnemar_p_value"] <= 1.0
    assert 0.0 <= pair["paired_boot_p_value"] <= 1.0
    assert "image_level_boot_p_value" in pair
    assert "holm_significant" in pair
    assert "benjamini_hochberg_significant" in pair
    assert pair["paired_ci_95_lower"] <= pair["paired_ci_95_upper"]
    assert "interpretation" in pair

