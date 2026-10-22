"""Unit tests for compound corruption order effects and quality diagnostics."""

import pytest
import torch

from robustlens.corruptions.order_effects import (
    ORDER_PERMUTATIONS,
    apply_compound_corruption,
    compute_psnr,
    compute_ssim,
    generate_matched_noise,
    get_deterministic_noise_seed,
)


def test_order_permutations_structure():
    assert len(ORDER_PERMUTATIONS) == 6
    assert set(ORDER_PERMUTATIONS) == {"NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"}
    for p in ORDER_PERMUTATIONS:
        assert sorted(list(p)) == ["B", "J", "N"]


def test_matched_noise_determinism():
    shape = (3, 64, 64)
    # Same sample_idx and severity must yield identical noise
    noise1 = generate_matched_noise(shape, severity=2, sample_idx=10)
    noise2 = generate_matched_noise(shape, severity=2, sample_idx=10)
    assert torch.equal(noise1, noise2)

    # Different severity must differ
    noise3 = generate_matched_noise(shape, severity=3, sample_idx=10)
    assert not torch.equal(noise1, noise3)

    # Different sample_idx must differ
    noise4 = generate_matched_noise(shape, severity=2, sample_idx=11)
    assert not torch.equal(noise1, noise4)


def test_compound_corruption_output_shape_and_bounds():
    torch.manual_seed(123)
    img = torch.rand(3, 64, 64)
    
    for order in ORDER_PERMUTATIONS:
        corrupt = apply_compound_corruption(img, order=order, severity=2, sample_idx=5)
        assert corrupt.shape == img.shape
        assert corrupt.min() >= 0.0
        assert corrupt.max() <= 1.0


def test_operator_non_commutativity():
    """Verify that different permutations of identical operations yield distinct tensors."""
    torch.manual_seed(999)
    # A structured test pattern (gradient + texture)
    x = torch.linspace(0, 1, 64).repeat(64, 1)
    y = x.t()
    img = torch.stack([x, y, (x + y) / 2.0], dim=0)

    # Compare NBJ vs JBN vs BNJ with matched noise
    out_nbj = apply_compound_corruption(img, order="NBJ", severity=3, sample_idx=42)
    out_jbn = apply_compound_corruption(img, order="JBN", severity=3, sample_idx=42)
    out_bnj = apply_compound_corruption(img, order="BNJ", severity=3, sample_idx=42)

    # Mathematical non-commutativity: out_nbj != out_jbn
    diff_nbj_jbn = (out_nbj - out_jbn).abs().max().item()
    diff_nbj_bnj = (out_nbj - out_bnj).abs().max().item()

    assert diff_nbj_jbn > 0.05, f"NBJ and JBN should differ significantly, got diff={diff_nbj_jbn}"
    assert diff_nbj_bnj > 0.02, f"NBJ and BNJ should differ significantly, got diff={diff_nbj_bnj}"


def test_psnr_and_ssim_metrics():
    torch.manual_seed(42)
    clean = torch.rand(3, 64, 64)
    
    # Clean with clean: PSNR=100 (cap), SSIM=1.0
    psnr_clean = compute_psnr(clean, clean)
    ssim_clean = compute_ssim(clean, clean)
    assert psnr_clean == 100.0
    assert abs(ssim_clean - 1.0) < 1e-4

    # Distorted image
    corrupt = apply_compound_corruption(clean, order="NBJ", severity=3, sample_idx=1)
    psnr_corrupt = compute_psnr(clean, corrupt)
    ssim_corrupt = compute_ssim(clean, corrupt)

    # Sanity checks
    assert 5.0 < psnr_corrupt < 45.0
    assert 0.1 < ssim_corrupt < 0.98
