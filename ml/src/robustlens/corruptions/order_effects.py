"""Compound corruption order effect pipelines and image-quality diagnostics.

This module implements the 6 permutations of the 3 canonical image degradations:
- N: Gaussian Noise (additive high-frequency stochastic perturbation)
- B: Gaussian Blur (spatial low-pass convolution)
- J: JPEG Compression (non-linear DCT quantization and blocking)

Crucially, it supports seed-matched noise generation so that for any given image
and severity level, the exact same noise realization epsilon is injected regardless
of where 'N' appears in the pipeline order.
"""

from __future__ import annotations

import io
import math
from typing import Tuple, List, Dict, Optional, Union

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# Canonical order permutations (3! = 6)
ORDER_PERMUTATIONS: List[str] = [
    "NBJ",  # Noise -> Blur -> JPEG
    "NJB",  # Noise -> JPEG -> Blur
    "BNJ",  # Blur -> Noise -> JPEG
    "BJN",  # Blur -> JPEG -> Noise
    "JNB",  # JPEG -> Noise -> Blur
    "JBN",  # JPEG -> Blur -> Noise
]

ORDER_DESCRIPTIONS: Dict[str, str] = {
    "NBJ": "Noise -> Blur -> JPEG",
    "NJB": "Noise -> JPEG -> Blur",
    "BNJ": "Blur -> Noise -> JPEG",
    "BJN": "Blur -> JPEG -> Noise",
    "JNB": "JPEG -> Noise -> Blur",
    "JBN": "JPEG -> Blur -> Noise",
}

# Parameter definitions matched across severities 1..5
NOISE_STDS: List[float] = [0.04, 0.08, 0.12, 0.18, 0.26]
BLUR_PARAMS: List[Tuple[int, float]] = [
    (3, 1.0),
    (5, 2.0),
    (7, 3.0),
    (9, 4.0),
    (11, 5.0),
]
JPEG_QUALITIES: List[int] = [80, 65, 50, 35, 20]


def get_deterministic_noise_seed(sample_idx: int, severity: int, base_seed: int = 42) -> int:
    """Generate a stable, distinct seed for a given image index and severity."""
    # Linear congruential / hashing combination to avoid collision
    return int((base_seed * 1000003 + sample_idx * 9973 + severity * 101) % (2**31 - 1))


def generate_matched_noise(
    shape: torch.Size,
    severity: int,
    sample_idx: int = 0,
    base_seed: int = 42,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Deterministically generate the matched additive Gaussian noise tensor.
    
    This guarantees that the exact same noise realization epsilon is injected
    across all 6 permutations for image sample_idx at severity.
    """
    if severity < 1 or severity > 5:
        raise ValueError(f"Severity must be in 1..5, got {severity}")
    std = NOISE_STDS[severity - 1]
    seed = get_deterministic_noise_seed(sample_idx, severity, base_seed)
    
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(shape, generator=gen, dtype=dtype) * std
    if device is not None:
        noise = noise.to(device)
    return noise


def apply_gaussian_noise(
    img: torch.Tensor,
    severity: int,
    noise_tensor: Optional[torch.Tensor] = None,
    sample_idx: int = 0,
    base_seed: int = 42,
) -> torch.Tensor:
    """Apply additive Gaussian noise clamped to [0, 1]."""
    if noise_tensor is None:
        noise_tensor = generate_matched_noise(
            img.shape, severity, sample_idx=sample_idx, base_seed=base_seed, device=img.device, dtype=img.dtype
        )
    else:
        if noise_tensor.shape != img.shape:
            raise ValueError(f"Noise shape {noise_tensor.shape} != image shape {img.shape}")
        if noise_tensor.device != img.device:
            noise_tensor = noise_tensor.to(img.device)
            
    return torch.clamp(img + noise_tensor, 0.0, 1.0)


def apply_gaussian_blur(img: torch.Tensor, severity: int) -> torch.Tensor:
    """Apply spatial Gaussian blur."""
    if severity < 1 or severity > 5:
        raise ValueError(f"Severity must be in 1..5, got {severity}")
    kernel_size, sigma = BLUR_PARAMS[severity - 1]
    return TF.gaussian_blur(img, [kernel_size, kernel_size], [sigma, sigma])


def apply_jpeg_compression(img: torch.Tensor, severity: int) -> torch.Tensor:
    """Apply lossy JPEG compression via PIL."""
    if severity < 1 or severity > 5:
        raise ValueError(f"Severity must be in 1..5, got {severity}")
    quality = JPEG_QUALITIES[severity - 1]
    
    # Ensure CPU tensor for PIL conversion
    original_device = img.device
    cpu_img = img.detach().cpu()
    pil_img = TF.to_pil_image(cpu_img)
    
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    
    compressed_pil = Image.open(buffer)
    compressed_tensor = TF.to_tensor(compressed_pil).to(original_device)
    return compressed_tensor


def apply_compound_corruption(
    img: torch.Tensor,
    order: str,
    severity: int,
    sample_idx: int = 0,
    base_seed: int = 42,
    precomputed_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply the compound corruption according to the specified 3-letter order string.
    
    Args:
        img: Input image tensor (3, H, W) with values in [0.0, 1.0].
        order: Permutation code, one of ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"].
        severity: Severity level 1..5.
        sample_idx: Physical image index for deterministic seed matching.
        base_seed: Base random seed.
        precomputed_noise: Optional precomputed matched noise tensor.
        
    Returns:
        Corrupted tensor (3, H, W) with values in [0.0, 1.0].
    """
    order = order.upper().strip()
    if order not in ORDER_PERMUTATIONS:
        raise ValueError(f"Invalid order '{order}'. Must be one of {ORDER_PERMUTATIONS}")
    if severity < 1 or severity > 5:
        raise ValueError(f"Severity must be in 1..5, got {severity}")

    # Generate or reuse matched noise tensor
    if precomputed_noise is None:
        noise_tensor = generate_matched_noise(
            img.shape, severity, sample_idx=sample_idx, base_seed=base_seed, device=img.device, dtype=img.dtype
        )
    else:
        noise_tensor = precomputed_noise

    current = img
    for char in order:
        if char == "N":
            current = apply_gaussian_noise(current, severity, noise_tensor=noise_tensor)
        elif char == "B":
            current = apply_gaussian_blur(current, severity)
        elif char == "J":
            current = apply_jpeg_compression(current, severity)
        else:
            raise ValueError(f"Unrecognized corruption primitive '{char}' in order '{order}'")

    return current


# =========================================================================
# Objective Image Quality Metrics: PSNR and SSIM
# =========================================================================

def compute_psnr(
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    max_val: float = 1.0,
    eps: float = 1e-10,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    
    Args:
        clean: Clean image tensor (C, H, W) or (B, C, H, W) in [0, 1].
        corrupt: Corrupted image tensor with same shape.
        max_val: Maximum dynamic range of image (1.0 for normalized tensors).
        eps: Small constant to prevent log(0).
        
    Returns:
        PSNR value in dB as float.
    """
    mse = torch.mean((clean.float() - corrupt.float()) ** 2).item()
    if mse < eps:
        return 100.0  # Cap for essentially identical images
    return float(10.0 * math.log10((max_val ** 2) / mse))


def _create_gaussian_kernel_2d(kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Create a 1D Gaussian kernel and outer product to 2D."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    gauss_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = torch.outer(gauss_1d, gauss_1d)
    return gauss_2d


def compute_ssim(
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    max_val: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Compute Structural Similarity Index (SSIM) between clean and corrupt tensors.
    
    Follows Wang et al. (2004) standard specification.
    
    Args:
        clean: Tensor of shape (C, H, W) or (1, C, H, W) in [0, 1].
        corrupt: Tensor of same shape.
        window_size: Gaussian window size (default: 11).
        sigma: Gaussian standard deviation (default: 1.5).
        max_val: Dynamic range (1.0).
        k1, k2: Stability constants.
        
    Returns:
        Mean SSIM scalar float in [-1.0, 1.0] (typically [0, 1]).
    """
    # Normalize dimensions to (B, C, H, W)
    if clean.ndim == 3:
        clean = clean.unsqueeze(0)
    if corrupt.ndim == 3:
        corrupt = corrupt.unsqueeze(0)
        
    clean = clean.float()
    corrupt = corrupt.float()
    channels = clean.shape[1]
    device = clean.device

    # Build 2D Gaussian filter kernel per channel
    kernel_2d = _create_gaussian_kernel_2d(window_size, sigma).to(device)
    kernel_4d = kernel_2d.expand(channels, 1, window_size, window_size)

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    # Local means
    pad = window_size // 2
    mu1 = F.conv2d(clean, kernel_4d, padding=pad, groups=channels)
    mu2 = F.conv2d(corrupt, kernel_4d, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Local variances and covariance
    sigma1_sq = F.conv2d(clean * clean, kernel_4d, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(corrupt * corrupt, kernel_4d, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(clean * corrupt, kernel_4d, padding=pad, groups=channels) - mu1_mu2

    # SSIM formula
    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / denominator

    return float(ssim_map.mean().item())
