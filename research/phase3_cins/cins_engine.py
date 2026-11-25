"""
Commutator-Induced Neural Susceptibility (CINS) Core Engine
Provides:
1. Degradation operators (NBJ and BNJ with identical pseudo-random noise realization).
2. Spectral invariance verification (Fourier magnitude, power, L2, histogram).
3. 2D FFT circular cross-correlation engine for full 50,176-point response maps.
4. Mathematical verification comparing brute-force spatial inner products vs. FFT.
5. Synthetic differentiable models for Taylor quadratic cancellation proofs.
"""

import numpy as np
from io import BytesIO
from PIL import Image

def generate_noise_blur_jpeg_pair(img_np, sigma=0.08, blur_radius=2, jpeg_quality=75, seed=42):
    """
    Simulates NBJ and BNJ compound degradations on [0, 1] RGB image (H x W x C).
    MANDATORY: Uses the EXACT same noise realization epsilon for both sequences.
    """
    H, W, C = img_np.shape
    rng = np.random.RandomState(seed)
    epsilon = rng.normal(0, sigma, img_np.shape).astype(np.float32)

    # 1D normalized Gaussian kernel
    ax = np.arange(-blur_radius, blur_radius + 1)
    kernel_1d = np.exp(-0.5 * (ax / 1.5) ** 2)
    kernel_1d /= kernel_1d.sum()

    def apply_blur(x):
        out = np.zeros_like(x)
        for c in range(C):
            temp = np.apply_along_axis(lambda m: np.convolve(m, kernel_1d, mode='same'), axis=0, arr=x[:, :, c])
            out[:, :, c] = np.apply_along_axis(lambda m: np.convolve(m, kernel_1d, mode='same'), axis=1, arr=temp)
        return np.clip(out, 0.0, 1.0)

    def apply_jpeg(x, q):
        pil_img = Image.fromarray((x * 255.0).astype(np.uint8))
        buf = BytesIO()
        pil_img.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        reloaded = Image.open(buf)
        return np.asarray(reloaded, dtype=np.float32) / 255.0

    # Sequence NBJ: Noise -> Blur -> JPEG
    x_n = np.clip(img_np + epsilon, 0.0, 1.0)
    x_nb = apply_blur(x_n)
    u = apply_jpeg(x_nb, jpeg_quality)

    # Sequence BNJ: Blur -> Noise -> JPEG (SAME EPSILON)
    x_b = apply_blur(img_np)
    x_bn = np.clip(x_b + epsilon, 0.0, 1.0)
    v = apply_jpeg(x_bn, jpeg_quality)

    d = u - v
    c = (u + v) / 2.0
    return u, v, d, c, epsilon

def circular_shift(arr, tau_y, tau_x):
    """
    Applies circular 2D spatial translation across (axis 0, axis 1).
    arr[y, x] -> arr[(y - tau_y) % H, (x - tau_x) % W]
    """
    return np.roll(np.roll(arr, tau_y, axis=0), tau_x, axis=1)

def verify_spectral_invariants(d, tau_y=17, tau_x=43):
    """
    Numerically verifies all 5 spectral and energy invariants between d and circular_shift(d, tau_y, tau_x).
    Computes in double precision (float64) to assess true mathematical invariance.
    """
    d_64 = d.astype(np.float64)
    H, W, C = d_64.shape
    d_tau = circular_shift(d_64, tau_y, tau_x)

    # 1. Per-channel Fourier magnitude discrepancy
    max_mag_diff = 0.0
    max_power_diff = 0.0
    for c in range(C):
        F_d = np.fft.fft2(d_64[:, :, c])
        F_tau = np.fft.fft2(d_tau[:, :, c])
        mag_diff = np.max(np.abs(np.abs(F_tau) - np.abs(F_d)))
        power_diff = np.max(np.abs(np.abs(F_tau)**2 - np.abs(F_d)**2))
        max_mag_diff = max(max_mag_diff, float(mag_diff))
        max_power_diff = max(max_power_diff, float(power_diff))

    # 2. Parseval L2 energy discrepancy
    l2_orig = float(np.sqrt(np.sum(d_64 ** 2)))
    l2_tau = float(np.sqrt(np.sum(d_tau ** 2)))
    l2_diff = abs(l2_orig - l2_tau)

    # 3. Pixel histogram invariance (Wasserstein / sorted absolute difference)
    hist_diff = 0.0
    for c in range(C):
        sorted_orig = np.sort(d_64[:, :, c].flatten())
        sorted_tau = np.sort(d_tau[:, :, c].flatten())
        hist_diff = max(hist_diff, float(np.max(np.abs(sorted_orig - sorted_tau))))

    # 4. Midpoint invariance (c + alpha/2 * d_tau + c - alpha/2 * d_tau) / 2 == c
    alpha = 0.25
    c_dummy = np.full_like(d_64, 0.5)
    x_plus = c_dummy + 0.5 * alpha * d_tau
    x_minus = c_dummy - 0.5 * alpha * d_tau
    midpoint_diff = float(np.max(np.abs(0.5 * (x_plus + x_minus) - c_dummy)))

    return {
        "max_fourier_magnitude_discrepancy": max_mag_diff,
        "max_fourier_power_discrepancy": max_power_diff,
        "l2_energy_discrepancy": l2_diff,
        "pixel_histogram_sorted_discrepancy": hist_diff,
        "midpoint_invariance_discrepancy": midpoint_diff,
    }

def compute_fft_response_map(gradient, residual, alpha=1.0):
    """
    Computes full H x W predicted response map via 2D circular cross-correlation.
    gradient: H x W x C (spatial gradient of margin at anchor c)
    residual: H x W x C (untranslated residual d = u - v)
    alpha: intervention scaling factor

    Returns:
    response_map: H x W float array where response_map[ty, tx] = alpha * <gradient, circular_shift(residual, ty, tx)>
    """
    H, W, C = gradient.shape
    cross_spec_sum = np.zeros((H, W), dtype=np.complex128)

    for c in range(C):
        G_c = np.fft.fft2(gradient[:, :, c])
        D_c = np.fft.fft2(residual[:, :, c])
        # Cross-correlation in frequency domain: G * conj(D)
        cross_spec_sum += G_c * np.conj(D_c)

    # Inverse 2D FFT to recover spatial cross-correlation
    spatial_corr = np.real(np.fft.ifft2(cross_spec_sum))
    return alpha * spatial_corr

def brute_force_spatial_inner_product(gradient, residual, tau_y, tau_x, alpha=1.0):
    """
    Brute-force O(H x W x C) calculation of alpha * <gradient, circular_shift(residual, ty, tx)>
    """
    d_tau = circular_shift(residual, tau_y, tau_x)
    return alpha * float(np.sum(gradient * d_tau))

def verify_fft_vs_brute_force(gradient, residual, num_test_points=25, alpha=1.0):
    """
    Verifies that the 2D FFT circular cross-correlation matches brute-force spatial inner products
    at random and extreme spatial translations to within floating-point machine precision.
    """
    H, W, _ = gradient.shape
    fft_map = compute_fft_response_map(gradient, residual, alpha=alpha)

    # Test random translations
    rng = np.random.RandomState(1337)
    test_y = rng.randint(0, H, size=num_test_points)
    test_x = rng.randint(0, W, size=num_test_points)

    max_abs_err = 0.0
    relative_errs = []

    for ty, tx in zip(test_y, test_x):
        val_bf = brute_force_spatial_inner_product(gradient, residual, ty, tx, alpha=alpha)
        val_fft = float(fft_map[ty, tx])
        err = abs(val_bf - val_fft)
        max_abs_err = max(max_abs_err, err)
        if abs(val_bf) > 1e-8:
            relative_errs.append(err / abs(val_bf))

    return {
        "max_absolute_error": max_abs_err,
        "mean_relative_error": float(np.mean(relative_errs)) if relative_errs else 0.0,
        "num_tested_coordinates": num_test_points,
    }

class SyntheticDifferentiableModel:
    """
    A controllable synthetic model to verify:
    1. Perfect linear network: error is identically zero.
    2. Quadratic/C3 smooth network: second-order error strictly cancels for symmetric differences!
    3. Piecewise linear (ReLU) network: activation boundary crossings cause error.
    """
    def __init__(self, in_dim=(16, 16, 1), mode="quadratic"):
        self.H, self.W, self.C = in_dim
        self.D = self.H * self.W * self.C
        self.mode = mode
        rng = np.random.RandomState(42)
        self.W1 = rng.normal(0, 0.1, (self.D, 16)).astype(np.float64)
        self.W2 = rng.normal(0, 0.1, (16, 2)).astype(np.float64) # 2 classes: class 0 (true), class 1 (competing)

    def forward_logits(self, x_flat):
        # x_flat: (N, D)
        h = x_flat @ self.W1
        if self.mode == "linear":
            pass
        elif self.mode == "quadratic":
            h = h + 0.1 * (h ** 2)
        elif self.mode == "cubic":
            h = h + 0.1 * (h ** 3)
        elif self.mode == "relu":
            h = np.maximum(0.0, h)
        logits = h @ self.W2
        return logits

    def margin(self, x):
        # x: (H, W, C)
        x_flat = x.flatten()[np.newaxis, :]
        logits = self.forward_logits(x_flat)[0]
        return logits[0] - logits[1]

    def gradient_margin(self, x, eps=1e-6):
        # Numerical gradient via central differences
        grad = np.zeros_like(x, dtype=np.float64)
        flat_x = x.copy().flatten()
        for i in range(len(flat_x)):
            x_plus = flat_x.copy()
            x_minus = flat_x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            m_plus = self.forward_logits(x_plus[np.newaxis, :])[0]
            m_minus = self.forward_logits(x_minus[np.newaxis, :])[0]
            margin_plus = m_plus[0] - m_plus[1]
            margin_minus = m_minus[0] - m_minus[1]
            grad.flat[i] = (margin_plus - margin_minus) / (2.0 * eps)
        return grad
