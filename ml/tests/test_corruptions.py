"""Unit tests for image corruption transforms."""

import torch
from torchvision import transforms
from PIL import Image

from robustlens.corruptions.transforms import get_corruption_transform, CORRUPTIONS


def test_corruption_types_and_severities():
    """Verify all corruptions run on image tensors and preserve valid [0, 1] bounds."""
    test_tensor = torch.full((3, 64, 64), 0.5)

    for corr_name in CORRUPTIONS:
        for sev in [1, 3, 5]:
            transform = get_corruption_transform(corr_name, sev)
            out = transform(test_tensor.clone())

            assert out.shape == test_tensor.shape, f"{corr_name} s{sev} changed tensor shape: {out.shape}"
            assert not torch.isnan(out).any(), f"{corr_name} s{sev} produced NaN"
            assert not torch.isinf(out).any(), f"{corr_name} s{sev} produced Inf"
            assert out.min() >= 0.0, f"{corr_name} s{sev} produced negative value: {out.min()}"
            assert out.max() <= 1.0, f"{corr_name} s{sev} exceeded 1.0: {out.max()}"


def test_gaussian_noise_monotonicity():
    """Verify severity 5 applies higher variance perturbation than severity 1."""
    torch.manual_seed(42)
    base = torch.full((3, 100, 100), 0.5)

    noise_s1 = get_corruption_transform("gaussian_noise", 1)(base.clone())
    noise_s5 = get_corruption_transform("gaussian_noise", 5)(base.clone())

    diff_s1 = torch.abs(noise_s1 - base).mean().item()
    diff_s5 = torch.abs(noise_s5 - base).mean().item()

    assert diff_s5 > diff_s1, f"Severity 5 perturbation ({diff_s5}) should exceed severity 1 ({diff_s1})"
