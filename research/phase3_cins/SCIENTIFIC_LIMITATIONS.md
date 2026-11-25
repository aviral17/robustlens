# Scientific Limitations & Negative Results Disclosure: CINS Framework
## Canonical Integrity & Audit Disclosure (Post-Audit Revised)

**Document Reference:** `research/phase3_cins/SCIENTIFIC_LIMITATIONS.md`  
**Framework:** Commutator-Induced Neural Susceptibility (CINS)  
**Authors:** RobustLens Research Initiative  
**Date:** November 2026 (Post-Audit Revised)  
**Status:** Canonical Scientific Integrity, Methodological Audit & Limitations Report  

---

## 1. Ethical & Epistemological Stance

In strict adherence to academic rigor, this document provides an unvarnished audit of the preliminary experiments, theoretical limitations, and operational boundaries of the **Commutator-Induced Neural Susceptibility (CINS)** framework.

We do **not** claim CINS as an established universal predictor of vision robustness, nor do we claim "world first" priority without verified proof. We explicitly distinguish preliminary feasibility explorations from completed neural-gradient validations, and identify all sources of approximation error and experimental caveats.

---

## 2. Audit of the Preliminary ZeroGPU Experiment

The earlier preliminary experiment reported in `eval_zerogpu_translations.py` and `zerogpu_empirical_translations.json` had significant methodological constraints that must be explicitly understood:

1. **Exploratory Sample Size:**  
   The experiment was conducted on exactly **one image** (`img_0_clean.jpg`) and across only **six spatial translations**. It demonstrated that neural confidence responds to residual translation, but it is **not** a completed statistical validation of neural-gradient response predictions.
2. **Proxy Gradient vs. Genuine Neural Gradient:**  
   The gradient map used in `eval_zerogpu_translations.py` was a **Sobel edge magnitude filter**, used as a crude geometric proxy. It was **not** a genuine backpropagated neural gradient $\nabla_x m(c)$ from PyTorch autograd. Therefore, agreement or disagreement in that script does not reflect the predictive performance of true neural gradients.
3. **Lossy Image Transport & Invariant Degradation:**  
   The ZeroGPU HTTP query serialized input tensors into base64-encoded JPEG strings (`quality=95`).  
   *Critical Finding:* Lossy JPEG compression applied after generating $x_+(\tau)$ and $x_-(\tau)$ introduces discrete cosine transform (DCT) quantization noise into the input tensor. Consequently, the exact spectral and energy invariants verified in floating-point numpy were slightly perturbed at the input of the remote model. True invariant testing requires direct floating-point tensor feeding without intermediate image re-encoding.
4. **Post-Softmax Saturation vs. Un-Normalized Logits:**  
   The remote API returned only top-1 post-softmax probabilities. For ResNet-50 and ViT-Base ($p > 0.997$), softmax sigmoid plateauing compressed response deltas to fractions of a percent ($10^{-4}$ to $10^{-6}$), masking internal feature susceptibility. Genuine validation requires pre-softmax logits and margins.
5. **Hardware Provenance & Checkpoint Verification:**  
   The remote endpoint `/api/inference` calls an external Hugging Face Space. While functional for interactive demos, the exact model checkpoint weights (e.g., torchvision vs. timm), preprocessing transforms (resizing, interpolation modes), and floating-point precision cannot be independently audited or guaranteed through an HTTP API.

---

## 3. Core Theoretical & Methodological Limitations

```
                                    ┌────────────────────────────────────┐
                                    │     CINS Framework Limitations     │
                                    └─────────────────┬──────────────────┘
                                                      │
         ┌──────────────────────────────┬─────────────┴────────────────┬──────────────────────────────┐
         │                              │                              │                              │
         ▼                              ▼                              ▼                              ▼
┌──────────────────┐          ┌──────────────────┐           ┌──────────────────┐           ┌──────────────────┐
│ Circular Boundary│          │ Gamut Saturation │           │ Non-Smoothness   │           │ Single-Anchor    │
│ Toroidal Wrap    │          │ Headroom Limits  │           │ Argmax / ReLUs   │           │ Path Trajectory  │
└──────────────────┘          └──────────────────┘           └──────────────────┘           └──────────────────┘
```

### 3.1 The Circular Translation (Toroidal) Boundary Assumption
- **The Limitation:**  
  The $2,000\times$ algorithmic speedup relies on the 2D Discrete Fourier Transform (DFT), which enforces circular (periodic) boundary conditions:
  $$d_\tau(y, x) = d((y - \tau_y) \bmod H, (x - \tau_x) \bmod W)$$
- **Physical Reality:**  
  Physical camera corruptions (lens raindrops, sensor read noise) do not wrap toroidally across image boundaries. When $\tau$ is large, boundary pixels wrap into the center.  
- **Mitigation:** In confirmatory experiments, evaluation should report small localized offsets ($\tau \in [-16, 16]^2$) separately from large boundary-wrapping offsets.

### 3.2 Photographic Gamut Headroom & Boundary Clipping
- **The Limitation:**  
  In natural RGB images containing saturated specular highlights ($c = 1.0$) or shadow pixels ($c = 0.0$), the unconstrained global headroom is:
  $$\alpha_{\text{global}} = \frac{2 \min_{i,j,c} \min(c, 1-c)}{\max_{i,j,c} |d|}$$
  For images with any pixels touching $0.0$ or $1.0$, unconstrained $\alpha_{\text{global}} \to 0$.
- **Audit Clarification on Protocol B:**  
  Interior-anchor contraction $c_\eta = (1 - 2\eta)c + \eta$ with $\eta = 0.05$ guarantees minimum headroom $\delta_c \ge 0.05$.  
  The bound is $\alpha_{\text{global}} \ge \frac{0.10}{\max |d|}$.  
  *Audit Correction:* $\alpha \ge 0.46$ is **not** a universal mathematical guarantee for all images. If an image produces a residual with $\max |d| = 0.25$, then $\alpha_{\text{global}} \ge 0.40$. The feasible strength depends on the residual norm and must be computed explicitly per sample.

### 3.3 Regularity Assumptions & Failure Across Non-Smooth Boundaries
- **The Limitation:**  
  The cubic error bound $\mathcal{E} = \mathcal{O}(\alpha^3)$ requires $\mathcal{C}^5$ Fréchet differentiability on the open segment $[c - h, c + h]$.
- **Reality:**  
  Vision networks violate this across:
  1. **ReLU activation hyperplanes:** Where gradients undergo jump discontinuities.
  2. **Top-2 competitor logit intersections (the argmax cliff):** Where true-class margin $m(x) = z_{y^*} - \max_{j \ne y^*} z_j$ is non-differentiable.
  3. **Hypothesis Status:** Claims that single-anchor predictions are "highly accurate at $\alpha \le 0.25$" or "fail at $\alpha \ge 0.50$" remain empirical hypotheses to be systematically measured, not established facts.

### 3.4 Disambiguation from Established DSP Literature
- **What is NOT Novel:** The algebraic non-commutativity of non-linear filters (e.g., median filtering and morphological operations) is a standard topic in classical digital signal processing (DSP).
- **What IS Novel in CINS:** Isolating the specific algebraic commutator residual of noise, blur, and JPEG compression under spectrally invariant 2D circular shifts, and constructing FFT-accelerated neural decision response maps to characterize architectural susceptibility in deep neural networks.

---

## 4. Retracted or Corrected Claims Summary

| Preliminary Claim | Audit Correction & Current Status |
| :--- | :--- |
| *"Measured 1,689× GPU speedup"* | **Retracted.** Unmeasured theoretical extrapolation. Only CPU FFT vs. CPU brute-force ($14.4\text{ ms}$ vs. $164.3\text{ s}$, $\sim 11,400\times$) has been directly measured. |
| *"Single-anchor predictions guaranteed accurate at $\alpha \le 0.25$"* | **Retracted as an established fact.** Reclassified as a testable scientific hypothesis (H1) for the 10-image GPU experiment. |
| *"GELU networks are piecewise affine"* | **Corrected.** GELU is infinitely differentiable ($\mathcal{C}^\infty$) and smooth non-linear, not piecewise affine. Pure ReLU networks can be piecewise affine. |
| *"Swin window size is $7\times 7$ pixels"* | **Corrected.** Swin window size $M=7$ refers to tokens, corresponding to $28\times 28$ input pixels in Stage 1 with patch size $4\times 4$. |
| *"Measured DC residual of $6.93\times 10^{-6}$ is machine-precision zero"* | **Corrected.** It is a finite-sample Monte Carlo error $\mathcal{O}(\sigma / \sqrt{N})$ combined with FIR kernel boundary truncation, not machine epsilon ($10^{-16}$). |
| *"Universal $\alpha \ge 0.46$ guarantee under Protocol B"* | **Corrected.** The lower bound is $\frac{0.10}{\max |d|}$, which varies with sample residual magnitude. |
