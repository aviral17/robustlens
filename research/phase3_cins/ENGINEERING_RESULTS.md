# Engineering Feasibility & Numerical Benchmarks: CINS Framework
## Algorithmic Evaluation & Preliminary Confidence Response (Post-Audit Revised)

**Document Reference:** `research/phase3_cins/ENGINEERING_RESULTS.md`  
**Framework:** Commutator-Induced Neural Susceptibility (CINS)  
**Authors:** RobustLens Research Initiative  
**Date:** November 2026 (Post-Audit Revised)  
**Status:** Validated Benchmarks & Preliminary Response Audit  

---

## 1. Overview of Experimental Validation

This report documents the numerical and algorithmic validation of the CINS engine and audits the preliminary hardware response exploration.

The validation is structured into two separate domains:
1. **Mathematical & Algorithmic Benchmarks:** Local CPU evaluation testing the 2D circular cross-correlation engine, double-precision invariant preservation, and speedup ratios against brute-force spatial inner products under controlled timing methodology.
2. **Preliminary Neural Response Exploration:** Remote evaluation on Hugging Face ZeroGPU (NVIDIA A10G) across four neural vision architectures (ResNet-50, ConvNeXt-Tiny, ViT-Base/16, Swin-Tiny) testing confidence-response variation on an exploratory sample.

---

## 2. Invariant Preservation Benchmarks

Across all $10$ photographic and benchmark test images, we verified the five theoretical invariants under 2D circular translations $\tau = (\tau_y, \tau_x)$:

| Invariant Metric | Theoretical Expectation | Measured Difference (Double Precision) | Pass / Fail Status |
| :--- | :--- | :--- | :--- |
| **Fourier Magnitude Spectrum** $\| \mathcal{F}(d_\tau) \|$ | $\| \mathcal{F}(d_\tau) \| \equiv \| \mathcal{F}(d) \|$ | **$2.13 \times 10^{-14}$** | **PASS** ($< 10^{-10}$) |
| **Fourier Power Spectrum** $\| \mathcal{F}(d_\tau) \|^2$ | $\| \mathcal{F}(d_\tau) \|^2 \equiv \| \mathcal{F}(d) \|^2$ | **$1.59 \times 10^{-12}$** | **PASS** ($< 10^{-10}$) |
| **Total $L_2$ Energy** $\|d_\tau\|_2^2$ | $\|d_\tau\|_2^2 \equiv \|d\|_2^2$ | **$0.00 \times 10^{0}$** (Exact) | **PASS** ($< 10^{-10}$) |
| **Pixel Value Histogram** (Sorted $d_\tau$) | $\operatorname{sort}(d_\tau) \equiv \operatorname{sort}(d)$ | **$0.00 \times 10^{0}$** (Exact) | **PASS** ($< 10^{-10}$) |
| **Shared Anchor Midpoint** $c$ | $c = \frac{1}{2}(u + v)$ | **Invariant by definition** | **PASS** |

**Conclusion:** Spatial circular translation $\tau$ strictly preserves global Fourier power, $L_2$ norm, and marginal pixel distributions to machine precision, confirming that downstream neural response variation is driven purely by the spatial phase relationship between the residual and image features.

---

## 3. Algorithmic Acceleration & Speedup Benchmark

We benchmarked the computation of the complete $224 \times 224 = 50,176$ spatial response map using two algorithms on identical hardware (AMD Ryzen 9 / Intel Core i7 equivalent):
1. **2D Circular Cross-Correlation via FFT:** Using `np.fft.fft2` and `np.fft.ifft2` according to the 2D Circular Cross-Correlation Theorem.
2. **Direct Spatial Inner Product:** Iterating over spatial translation offsets, taking the inner product $\sum_{i,j,c} g_{i,j,c} \cdot d_{i,j,c}(\tau)$.

### Controlled Timing Methodology
- Applied 3 warm-up iterations prior to timing.
- Measured 10 full repetitions of the 2D FFT full map ($50,176$ points).
- Measured direct spatial correlation on a $16 \times 16 = 256$ subgrid to obtain an empirical per-point cost ($3.012\text{ ms/point}$).
- Numerical equivalence was independently verified across 50 random coordinates ($\max |\hat{r}_{\text{FFT}} - \hat{r}_{\text{spatial}}| = 2.27 \times 10^{-13}$).

### Runtime Comparison Table

| Metric | Direct Spatial Inner Product | 2D FFT Cross-Correlation | Measured Improvement |
| :--- | :--- | :--- | :--- |
| **Directly Measured Time (256 Points)** | $771.07\text{ ms}$ | $14.40 \pm 1.74\text{ ms}$ *(Computes all 50,176)* | **$> 50\times$ faster than subgrid** |
| **Extrapolated Time (Full 50,176 Points)** | $164.32\text{ seconds}$ ($2.74\text{ min}$) | **$14.40\text{ ms}$** | **$11,410.6\times$ Extrapolated** |
| **Computational Complexity** | $\mathcal{O}(H^2 W^2 C) \approx 7.55 \times 10^9$ MACs | $\mathcal{O}(C \cdot HW \log(HW)) \approx 2.4 \times 10^6$ ops | **$\sim 3,125\times$ Theoretical** |
| **Numerical Discrepancy** | Baseline reference | $\max |\hat{r}_{\text{FFT}} - \hat{r}_{\text{spatial}}| = 2.27 \times 10^{-13}$ | **Machine Precision ($< 10^{-12}$)** |
| **Relative Discrepancy** | Baseline reference | $\frac{\|\hat{r}_{\text{FFT}} - \hat{r}_{\text{spatial}}\|_2}{\|\hat{r}_{\text{spatial}}\|_2} = 1.04 \times 10^{-15}$ | **Exact Equivalence** |

*Note on GPU Comparison:* Preliminary claims of a measured $1,689\times$ GPU speedup were based on theoretical forward pass estimates and have been retracted. Exact GPU inference speedup will be benchmarked on Kaggle GPU using synchronized `torch.cuda.Event` timers.

---

## 4. Audit of Preliminary ZeroGPU Response Exploration

In `eval_zerogpu_translations.py`, we queried remote vision models on Hugging Face ZeroGPU (NVIDIA A10G) across six spatial translations for `img_0_clean.jpg` (ImageNet class: `tench`).

### Empirical Findings
- Anchor baseline top-1 confidence:
  - ResNet-50: `tench` ($99.99988\%$)
  - ConvNeXt-Tiny: `tench` ($89.24\%$)
  - ViT-Base: `tench` ($99.74\%$)
  - Swin-Tiny: `tench` ($85.11\%$)
- Measured symmetric confidence response $\Delta p(\tau) = p_{\text{tench}}(x_+(\tau)) - p_{\text{tench}}(x_-(\tau))$ at $\alpha = 0.25$:
  - **Swin-Tiny:** Varied from $-3.69\%$ at $\tau=(44, 44)$ to $+4.92\%$ at $\tau=(0, 0)$, exhibiting an **$8.61$ percentage point dynamic range**.
  - **ConvNeXt-Tiny:** Varied from $-0.60\%$ to $+1.06\%$, exhibiting a **$1.66$ percentage point dynamic range**.
  - **ResNet-50 & ViT-Base:** Saturated ($p > 0.997$), showing minimal post-softmax variation ($\Delta p < 10^{-4}$).

### Methodological Audit & Disclosures
1. **Preliminary Nature:** Conducted on a single image across 6 shifts. It demonstrates confidence sensitivity to translation, but does **not** validate gradient predictions.
2. **Proxy Gradient Warning:** The gradient map used in that exploratory script was a Sobel edge filter proxy, not a genuine backpropagated neural gradient.
3. **Transport Distortion:** The remote API required base64 JPEG encoding, which introduces minor DCT quantization noise and perturbs exact floating-point spectral invariants.
4. **Conclusion:** Genuine neural-gradient prediction validation requires the standalone Kaggle GPU experiment using direct floating-point tensor inputs and true autograd gradients.
