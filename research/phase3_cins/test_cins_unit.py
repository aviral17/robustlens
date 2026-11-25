"""
Unit Test Suite for Commutator-Induced Neural Susceptibility (CINS)
Tests:
1. Operator Commutator identities (linear ideal vs. nonlinear actual).
2. Exact Spectral & Energy Invariants under circular spatial translation.
3. 2D FFT Circular Cross-Correlation vs. Brute-Force Spatial Inner Products.
4. Taylor Symmetric Difference Quadratic Cancellation Theorem on synthetic models.
5. Timing benchmark: 2D FFT vs. Brute-force full grid.
"""

import sys
import os
import time
import json
import numpy as np

# Ensure local module is loadable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cins_engine import (
    generate_noise_blur_jpeg_pair,
    circular_shift,
    verify_spectral_invariants,
    compute_fft_response_map,
    brute_force_spatial_inner_product,
    verify_fft_vs_brute_force,
    SyntheticDifferentiableModel
)

def test_linear_commutator_identity():
    print("\n--- Test 1: Linear Noise-Blur Commutator Identity ---")
    H, W = 64, 64
    x = np.random.RandomState(10).uniform(0.2, 0.8, (H, W, 1)).astype(np.float32)
    sigma = 0.05
    rng = np.random.RandomState(11)
    eps = rng.normal(0, sigma, (H, W, 1)).astype(np.float32)

    # 1D kernel
    k = np.array([0.05, 0.25, 0.4, 0.25, 0.05], dtype=np.float32)
    def blur_lin(im):
        t1 = np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'), axis=0, arr=im[:, :, 0])
        t2 = np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'), axis=1, arr=t1)
        return t2[:, :, np.newaxis]

    # Without clipping:
    # B(N(x)) = B(x + eps) = B(x) + B(eps)
    # N(B(x)) = B(x) + eps
    # d = B(N(x)) - N(B(x)) = B(eps) - eps = (K - I)*eps
    bn_x = blur_lin(x + eps)
    nb_x = blur_lin(x) + eps
    d_actual = bn_x - nb_x
    d_theory = blur_lin(eps) - eps

    max_diff = np.max(np.abs(d_actual - d_theory))
    dc_mean = np.mean(d_actual)
    print(f"Max difference between [B, N](x) and (K - I)eps: {max_diff:.2e}")
    print(f"Spatial DC mean of ideal commutator: {dc_mean:.2e}")
    assert max_diff < 1e-6, "Linear commutator identity failed!"
    assert abs(dc_mean) < 1e-4, "Ideal commutator DC component should be approximately zero!"
    return {"passed": True, "max_discrepancy": float(max_diff), "dc_mean": float(dc_mean)}

def test_spectral_invariants():
    print("\n--- Test 2: Exact Spectral & Energy Invariants ---")
    H, W, C = 224, 224, 3
    x = np.random.RandomState(42).uniform(0.1, 0.9, (H, W, C)).astype(np.float32)
    u, v, d, c, eps = generate_noise_blur_jpeg_pair(x, seed=42)

    # Test random translation
    res = verify_spectral_invariants(d, tau_y=37, tau_x=89)
    print("Spectral Invariance Discrepancies under tau=(37, 89):")
    for k, v in res.items():
        print(f"  {k}: {v:.2e}")
        assert v < 1e-10, f"Invariant {k} violated! Val: {v}"
    return res

def test_fft_cross_correlation_sign_and_precision():
    print("\n--- Test 3: 2D FFT Circular Cross-Correlation vs. Brute Force ---")
    H, W, C = 224, 224, 3
    gradient = np.random.RandomState(101).randn(H, W, C).astype(np.float64)
    residual = np.random.RandomState(102).randn(H, W, C).astype(np.float64)

    res = verify_fft_vs_brute_force(gradient, residual, num_test_points=50, alpha=0.5)
    print(f"Tested {res['num_tested_coordinates']} spatial coordinates.")
    print(f"Max absolute discrepancy between FFT and brute-force: {res['max_absolute_error']:.2e}")
    print(f"Mean relative discrepancy: {res['mean_relative_error']:.2e}")
    assert res['max_absolute_error'] < 1e-12, "FFT cross-correlation failed machine precision test!"
    return res

def test_taylor_quadratic_cancellation():
    print("\n--- Test 4: Symmetric Taylor Quadratic Cancellation ---")
    model = SyntheticDifferentiableModel(in_dim=(16, 16, 1), mode="quadratic")
    H, W, C = 16, 16, 1
    c = np.random.RandomState(201).uniform(0.3, 0.7, (H, W, C))
    d = np.random.RandomState(202).normal(0, 0.1, (H, W, C))
    grad = model.gradient_margin(c)

    # Test for several shrinking alphas: 0.1, 0.05, 0.025, 0.0125
    alphas = [0.2, 0.1, 0.05, 0.025]
    symmetric_errors = []
    asymmetric_errors = []

    for a in alphas:
        # Symmetrically perturbed: x_+ = c + a/2 * d, x_- = c - a/2 * d
        x_plus = c + 0.5 * a * d
        x_minus = c - 0.5 * a * d
        actual_sym = model.margin(x_plus) - model.margin(x_minus)
        pred_sym = a * np.sum(grad * d)
        err_sym = abs(actual_sym - pred_sym)
        symmetric_errors.append(err_sym)

        # Asymmetrically perturbed: x_+ = c + a * d vs c
        x_asym = c + a * d
        actual_asym = model.margin(x_asym) - model.margin(c)
        pred_asym = a * np.sum(grad * d)
        err_asym = abs(actual_asym - pred_asym)
        asymmetric_errors.append(err_asym)

    # In a quadratic model, the symmetric difference error should be strictly ZERO (since third-order is 0),
    # while the asymmetric difference error should scale as O(a^2)!
    print("Intervention Strengths (alpha):", alphas)
    print("Symmetric Errors (x_+ - x_-):", [f"{e:.2e}" for e in symmetric_errors])
    print("Asymmetric Errors (x_+ - c):", [f"{e:.2e}" for e in asymmetric_errors])

    # Check that symmetric error is near floating-point zero for quadratic network
    assert symmetric_errors[-1] < 1e-10, "Symmetric error on quadratic model must be zero!"
    # Check that asymmetric error is quadratic O(alpha^2)
    ratio_asym = asymmetric_errors[0] / asymmetric_errors[1] # should be ~ (0.2 / 0.1)^2 = 4.0
    print(f"Asymmetric error scaling ratio (0.2 vs 0.1): {ratio_asym:.2f} (expected ~ 4.0)")
    assert 3.8 < ratio_asym < 4.2, "Asymmetric error did not scale quadratically!"

    return {
        "alphas": alphas,
        "symmetric_errors": [float(e) for e in symmetric_errors],
        "asymmetric_errors": [float(e) for e in asymmetric_errors],
        "asymmetric_ratio": float(ratio_asym),
    }

def test_timing_benchmark():
    print("\n--- Test 5: Timing Benchmark (2D FFT vs. Brute-Force) ---")
    H, W, C = 224, 224, 3
    grad = np.random.randn(H, W, C).astype(np.float32)
    res = np.random.randn(H, W, C).astype(np.float32)

    # 1. FFT full 50,176 map
    t0 = time.perf_counter()
    fft_map = compute_fft_response_map(grad, res, alpha=0.5)
    t_fft = time.perf_counter() - t0
    print(f"2D FFT Full Map (50,176 predictions): {t_fft*1000:.2f} ms")

    # 2. Brute-force 100 sample positions
    num_samples = 100
    t0 = time.perf_counter()
    for _ in range(num_samples):
        brute_force_spatial_inner_product(grad, res, 15, 23, alpha=0.5)
    t_bf_sample = time.perf_counter() - t0
    t_per_sample = t_bf_sample / num_samples
    t_bf_extrapolated = t_per_sample * 50176
    print(f"Brute-force per position: {t_per_sample*1000:.3f} ms")
    print(f"Brute-force Extrapolated 50,176 positions: {t_bf_extrapolated:.2f} seconds ({t_bf_extrapolated/60:.2f} minutes)")
    speedup = t_bf_extrapolated / t_fft
    print(f"Measured algorithmic speedup: {speedup:.1f}x")

    return {
        "fft_full_map_ms": float(t_fft * 1000),
        "brute_force_per_point_ms": float(t_per_sample * 1000),
        "brute_force_extrapolated_50176_sec": float(t_bf_extrapolated),
        "empirical_speedup_ratio": float(speedup),
    }

def main():
    print("=" * 70)
    print("COMMUTATOR-INDUCED NEURAL SUSCEPTIBILITY (CINS) — UNIT TEST SUITE")
    print("=" * 70)
    t_start = time.time()
    
    report = {
        "test_1_linear_commutator": test_linear_commutator_identity(),
        "test_2_spectral_invariants": test_spectral_invariants(),
        "test_3_fft_vs_brute_force": test_fft_cross_correlation_sign_and_precision(),
        "test_4_taylor_quadratic_cancellation": test_taylor_quadratic_cancellation(),
        "test_5_timing_benchmark": test_timing_benchmark(),
        "total_test_runtime_sec": time.time() - t_start,
        "overall_status": "ALL_TESTS_PASSED"
    }

    report_path = r"research\phase3_cins\unit_test_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"ALL 5 TESTS PASSED SUCCESSFULLY! Report written to {report_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
