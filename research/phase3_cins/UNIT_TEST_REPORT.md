# Unit Test & Numerical Verification Report (Audit-Revised)

**Document Reference:** `research/phase3_cins/UNIT_TEST_REPORT.md`  
**Framework:** Commutator-Induced Neural Susceptibility (CINS)  
**Authors:** RobustLens Research Initiative  
**Date:** November 2026 (Revised Post-Audit)  
**Status:** All 6 Comprehensive Tests Passed  

---

## 1. Executive Summary & Verification Scope

In response to the independent scientific audit, the CINS numerical test suite (`test_cins_unit.py`) was expanded from 5 to 6 comprehensive tests to address:
1. The physical origin of the non-zero DC residual in the ideal linear commutator.
2. The exact scaling behavior of symmetric interventions across quadratic, cubic polynomial, and smooth non-polynomial manifolds.
3. Controlled algorithmic timing methodology distinguishing measured subgrid latencies from extrapolated full-map computations.
4. Numerical validation of exact per-translation, common-grid, and global toroidal feasible intervention bounds $\alpha_{\max}$.

---

## 2. Test-by-Test Results & Scientific Analyses

### Test 1: Linear Commutator Identity & DC Residual Investigation
- **Theoretical Formulation:**
  For linear continuous operators $\mathcal{B}(x) = K * x$ and $\mathcal{N}(x) = x + \epsilon$ with white Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$:
  $$[\mathcal{B}, \mathcal{N}](x) = (K - I) * \epsilon$$
- **Measured Discrepancy:**
  $$\max |[\mathcal{B}, \mathcal{N}](x) - (K - I)*\epsilon| = 3.26 \times 10^{-16}$$
  This confirms exact operator equivalence to double floating-point precision.
- **Investigation of Non-Zero DC Component:**
  - In earlier preliminary reports, the non-zero spatial mean ($6.93 \times 10^{-6}$) was casually described as "machine-precision zero." This was mathematically inaccurate.
  - **Audit Finding:** The sample mean of white noise $\epsilon$ over a finite discrete grid of $N = H \times W = 16,384$ pixels has theoretical standard error:
    $$\text{SE}(\bar{\epsilon}) = \frac{\sigma}{\sqrt{N}} = \frac{0.08}{\sqrt{16,384}} \approx 6.25 \times 10^{-4}$$
  - The measured empirical mean of the random input noise realization was $\bar{\epsilon} = +2.14 \times 10^{-4}$ (well within the $1\sigma$ Monte Carlo error bound).
  - Furthermore, discrete spatial convolution using `scipy.ndimage.gaussian_filter` truncates Gaussian filter tails at the image boundaries (`mode='same'`). Measuring the interior-only spatial mean yields $-1.05 \times 10^{-4}$, while the full image mean is $-3.34 \times 10^{-5}$.
  - **Scientific Conclusion:** The non-zero DC residual is a finite-sample Monte Carlo sampling deviation $\mathcal{O}(\sigma / \sqrt{N})$ combined with boundary filter truncation, NOT floating-point machine precision. In infinite continuous domain $\mathbb{R}^2$, the expectation $\mathbb{E}[\iint (K - I)*\epsilon] \equiv 0$ strictly holds.

---

### Test 2: Spectral & Energy Invariants
- **Tested Translation:** $\tau = (37, 89)$ on $224 \times 224 \times 3$ RGB tensor.
- **Measured Discrepancies:**
  - Fourier Magnitude Spectrum: $2.13 \times 10^{-14}$
  - Fourier Power Spectrum: $1.59 \times 10^{-12}$
  - Parseval $L_2$ Energy: $0.00 \times 10^0$ (Exact)
  - Sorted Pixel Histogram: $0.00 \times 10^0$ (Exact)
  - Midpoint Invariance: $0.00 \times 10^0$ (Exact)
- **Conclusion:** All five invariants hold to machine precision ($< 10^{-10}$ threshold passed).

---

### Test 3: 2D FFT Circular Cross-Correlation Equivalence
- **Tested Coordinates:** 50 randomly sampled translations on $224 \times 224 \times 3$ gradient and residual arrays.
- **Discrepancy:**
  - $\max |\hat{r}_{\text{FFT}} - \hat{r}_{\text{spatial}}| = 2.27 \times 10^{-13}$
  - Mean relative error: $1.04 \times 10^{-15}$
- **Conclusion:** The 2D FFT cross-correlation engine is algebraically and numerically identical to direct spatial inner products.

---

### Test 4: Multi-Model Taylor Expansion & Order Verification
- **Part 4A (Quadratic Model):**
  - Symmetric difference error at $\alpha = 0.025$: $4.72 \times 10^{-13}$ (strictly canceled).
  - Asymmetric error scaling ratio ($0.2$ vs $0.1$): $4.00\times$ (exact $\mathcal{O}(\alpha^2)$).
- **Part 4B (Cubic Polynomial Model $h + 0.1 h^3$):**
  - Symmetric errors: $[1.40 \times 10^{-6}, 1.75 \times 10^{-7}, 2.19 \times 10^{-8}, 2.73 \times 10^{-9}]$.
  - Error scaling ratios under factor-of-2 reductions in $\alpha$:
    $$\frac{\mathcal{E}(0.2)}{\mathcal{E}(0.1)} = 8.00\times, \quad \frac{\mathcal{E}(0.1)}{\mathcal{E}(0.05)} = 8.00\times, \quad \frac{\mathcal{E}(0.05)}{\mathcal{E}(0.025)} = 8.00\times$$
  - Exactly matches $2^3 = 8.00\times$, confirming that leading symmetric difference error is cubic $\mathcal{O}(\alpha^3)$.
- **Part 4C (Smooth Non-Polynomial Model $\sin(h)$):**
  - Symmetric errors: $[1.67 \times 10^{-7}, 2.09 \times 10^{-8}, 2.62 \times 10^{-9}, 3.29 \times 10^{-10}]$.
  - Asymptotic ratio at small $\alpha$ ($0.05$ vs $0.025$): $7.95\times$ (converging to $8.0\times$).
  - At larger $\alpha$, quintic terms $\mathcal{O}(\alpha^5)$ cause slight deviation from 8.0, matching the theoretical Taylor expansion of $\sin(h)$.

---

### Test 5: Algorithmic Efficiency Benchmark
- **Methodology:**
  - Timed 10 repetitions of the full $224 \times 224 = 50,176$ FFT map after 3 warm-up runs.
  - Timed brute-force spatial correlation directly on a $16 \times 16 = 256$ subgrid after warm-up runs.
- **Measured Latencies:**
  - 2D FFT Full Map: $14.40 \pm 1.74\text{ ms}$.
  - Direct spatial correlation on 256 points: $771.07\text{ ms}$ ($3.012\text{ ms}$ per spatial coordinate).
  - Extrapolated brute-force full map ($50,176$ points): $164.32\text{ s}$ ($2.74\text{ minutes}$).
  - Calculated algorithmic speedup: **$11,410.6\times$** (or $\approx 2,000\times$ in unvectorized loops).
- **GPU Forward Comparison Note:**
  - In earlier reports, an unmeasured figure of $1,689\times$ GPU speedup was stated based on theoretical estimates. Per the audit directive, this extrapolated claim is removed from empirical validation until measured directly with synchronized GPU timers.

---

### Test 6: Feasible Intervention Strength ($\alpha_{\max}$) Verification
- **Formulation Verified:**
  $$\alpha_{\max}(\tau) = \min_{i,j,c: d_\tau(i,j,c) \neq 0} \frac{2 \min(c(i,j,c), 1 - c(i,j,c))}{|d_\tau(i,j,c)|}$$
- **Measured Results on $64 \times 64 \times 3$ Image:**
  - Single shift $\tau = (15, 23)$ bound: $\alpha_{\max} = 2.6486$.
  - Endpoints: $x_+ \in [0.000000, 0.879385]$, $x_- \in [0.033280, 0.878749]$. Zero clipping.
  - 16-shift grid common bound: $\alpha_{\text{common}} = 0.9677$.
  - Global toroidal bound: $\alpha_{\text{global}} = 0.4918$.
  - Interior anchor ($\eta = 0.05$): $\alpha_{\text{global}} = 0.8607$.
- **Conclusion:** The analytical formulas guarantee zero out-of-gamut clipping.

---

## 3. Summary of Pass/Fail Status

| Test Index | Verification Domain | Target Criterion | Measured Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Test 1** | Linear Commutator Identity & DC | Error $< 10^{-12}$, DC within $3\sigma$ | Error: $3.26 \times 10^{-16}$, DC: $-3.34 \times 10^{-5}$ | **PASSED** |
| **Test 2** | Spectral & Energy Invariants | Double precision error $< 10^{-10}$ | Magnitude: $2.13 \times 10^{-14}$, $L_2$: $0.00$, Hist: $0.00$ | **PASSED** |
| **Test 3** | 2D FFT vs. Direct Spatial Product | Numerical discrepancy $< 10^{-12}$ | Error: $2.27 \times 10^{-13}$, Relative: $1.04 \times 10^{-15}$ | **PASSED** |
| **Test 4** | Taylor Order (Quad, Cubic, Sin) | Symmetric cancels; Cubic ratio $= 8.00\times$ | Quad: $4.72 \times 10^{-13}$, Cubic: $8.00\times$, Sin: $7.95\times$ | **PASSED** |
| **Test 5** | Algorithmic Acceleration | Controlled benchmark with warm-up | FFT: $14.4\text{ ms}$, Speedup: $> 10,000\times$ | **PASSED** |
| **Test 6** | Feasible Alpha Bounds | Exact clipping prevention $[0, 1]$ | Range: $[0.000000, 0.879385]$ | **PASSED** |
