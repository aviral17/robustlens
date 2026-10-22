"""Unit tests for the Matched-Fidelity Benchmark module."""

import json
import math
import os
import pytest
import torch
import torch.nn.functional as F
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CALIB_JSON = os.path.join(REPO_ROOT, "results", "order_effect_study", "matched_fidelity", "calibration_config.json")


def _create_gaussian_kernel_2d(kernel_size=11, sigma=1.5):
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    gauss_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = torch.outer(gauss_1d, gauss_1d)
    return gauss_2d


_SSIM_KERNEL_4D = _create_gaussian_kernel_2d(11, 1.5).expand(3, 1, 11, 11)


def compute_ssim(clean, corrupt, window_size=11, max_val=1.0, k1=0.01, k2=0.03):
    global _SSIM_KERNEL_4D
    if clean.ndim == 3:
        clean = clean.unsqueeze(0)
    if corrupt.ndim == 3:
        corrupt = corrupt.unsqueeze(0)
    clean = clean.float()
    corrupt = corrupt.float()
    channels = clean.shape[1]
    device = clean.device

    if _SSIM_KERNEL_4D.device != device:
        _SSIM_KERNEL_4D = _SSIM_KERNEL_4D.to(device)

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    pad = window_size // 2

    mu1 = F.conv2d(clean, _SSIM_KERNEL_4D, padding=pad, groups=channels)
    mu2 = F.conv2d(corrupt, _SSIM_KERNEL_4D, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(clean * clean, _SSIM_KERNEL_4D, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(corrupt * corrupt, _SSIM_KERNEL_4D, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(clean * corrupt, _SSIM_KERNEL_4D, padding=pad, groups=channels) - mu1_mu2

    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return float((numerator / denominator).mean().item())


def compute_psnr(clean, corrupt, max_val=1.0, eps=1e-10):
    mse = float(torch.mean((clean.float() - corrupt.float()) ** 2).item())
    if mse < eps:
        return 100.0
    return float(10.0 * math.log10((max_val ** 2) / mse))


class TestCalibrationConfig:
    def test_calibration_file_exists(self):
        assert os.path.exists(CALIB_JSON), f"Missing {CALIB_JSON}"

    def test_calibration_structure(self):
        with open(CALIB_JSON, "r") as f:
            data = json.load(f)

        assert "sequence_parameters" in data
        targets = data["sequence_parameters"]
        assert len(targets) == 5, f"Expected 5 targets, got {len(targets)}"

        expected_orders = {"NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"}
        for t_key, t_data in targets.items():
            assert "target_psnr" in t_data
            assert "target_ssim" in t_data
            assert "sequences" in t_data
            seqs = t_data["sequences"]
            assert set(seqs.keys()) == expected_orders

            for seq, params in seqs.items():
                assert 0.001 <= params["noise_std"] <= 0.40
                assert 1 <= params["blur_k"] <= 45
                assert 0.0 <= params["blur_sigma"] <= 12.0
                assert 5 <= params["jpeg_quality"] <= 95
                assert "achieved_psnr" in params
                assert "achieved_ssim" in params


class TestDistortionMetrics:
    def test_psnr_identical_images(self):
        img = torch.rand(3, 64, 64)
        psnr = compute_psnr(img, img)
        assert psnr >= 99.0

    def test_ssim_identical_images(self):
        img = torch.rand(3, 64, 64)
        ssim = compute_ssim(img, img)
        assert abs(ssim - 1.0) < 1e-4

    def test_ssim_symmetry(self):
        img_a = torch.rand(3, 64, 64)
        img_b = torch.rand(3, 64, 64)
        ssim_ab = compute_ssim(img_a, img_b)
        ssim_ba = compute_ssim(img_b, img_a)
        assert abs(ssim_ab - ssim_ba) < 1e-5

    def test_ssim_bounded(self):
        img_a = torch.rand(3, 64, 64)
        img_b = torch.rand(3, 64, 64)
        ssim = compute_ssim(img_a, img_b)
        assert -1.0 <= ssim <= 1.0


class TestErrorChurnInvariants:
    def test_churn_identities(self):
        # Create mock predictions
        np.random.seed(42)
        n = 500
        c_a = np.random.binomial(1, 0.65, n)
        c_b = np.random.binomial(1, 0.60, n)

        g = np.mean((c_a == 0) & (c_b == 1))
        l = np.mean((c_a == 1) & (c_b == 0))
        net_diff = g - l
        churn = g + l
        excess_churn = churn - abs(net_diff)

        assert abs(net_diff - (np.mean(c_b) - np.mean(c_a))) < 1e-10
        assert excess_churn >= -1e-10
        assert churn >= abs(net_diff) - 1e-10


class TestToleranceAudit:
    def test_tolerance_audit_execution_and_classification(self):
        from robustlens.eval.tolerance_audit import audit_matching_tolerances

        
        residuals_path = os.path.join(REPO_ROOT, "results", "order_effect_study", "matched_fidelity", "matching_residuals.csv")
        summary_path = os.path.join(REPO_ROOT, "results", "order_effect_study", "matched_fidelity", "outputs", "order_effects_summary.csv")
        config_path = os.path.join(REPO_ROOT, "results", "order_effect_study", "matched_fidelity", "calibration_config.json")
        
        report = audit_matching_tolerances(residuals_path, summary_path, config_path)
        
        assert "prespecified_tolerances" in report
        assert "measured_calibration_residuals" in report
        assert "criteria_verdict" in report
        
        # When calibration residuals exceed tolerances, study must be classified as approximate
        res = report["measured_calibration_residuals"]
        tol = report["prespecified_tolerances"]
        if res["psnr_rmse_db"] > tol["max_psnr_tolerance_db"] or res["ssim_rmse"] > tol["max_ssim_tolerance"]:
            assert report["criteria_verdict"]["exact_matching_achieved"] is False
            assert report["criteria_verdict"]["scientific_classification"] == "approximate_fidelity_comparison"

