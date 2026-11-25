# Scientific Limitations & Negative Results Disclosure: CINS Framework

**Document Reference:** `research/phase3_cins/SCIENTIFIC_LIMITATIONS.md`  
**Framework:** Commutator-Induced Neural Susceptibility (CINS)  
**Authors:** RobustLens Research Initiative  
**Date:** November 2026  
**Status:** Canonical Scientific Integrity & Limitations Report  

---

## 1. Ethical & Epistemological Stance

In adherence to the highest standards of scientific integrity, this document provides an unvarnished, transparent disclosure of the theoretical constraints, empirical edge cases, negative results, and operational boundaries of the **Commutator-Induced Neural Susceptibility (CINS)** framework.

We do **not** claim CINS as an unqualified "world first" or universal panacea for vision robustness. Rather, we clearly delimit where the mathematical guarantees hold, where empirical approximations begin to diverge, and where further foundational research is necessary before submitting to peer-reviewed conferences (e.g., CVPR, NeurIPS, ICML).

---

## 2. Core Methodological Limitations

```
                                    ┌────────────────────────────────────┐
                                    │     CINS Framework Limitations     │
                                    └─────────────────┬──────────────────┘
                                                      │
         ┌──────────────────────────────┬─────────────┴────────────────┬──────────────────────────────┐
         │                              │                              │                              │
         ▼                              ▼                              ▼                              ▼
┌──────────────────┐          ┌──────────────────┐           ┌──────────────────┐           ┌──────────────────┐
│ Circular Boundary│          │ Gamut Saturation │           │ Softmax Logit    │           │ Single-Anchor    │
│ Toroidal Artifact│          │ Clipping Bottlenk│           │ Compression      │           │ Path Deviation   │
└──────────────────┘          └──────────────────┘           └──────────────────┘           └──────────────────┘
```

### 2.1 The Circular Translation (Toroidal) Boundary Assumption

**The Limitation:**  
The $2,000\times$ algorithmic acceleration achieved by CINS relies fundamentally on the 2D Discrete Fourier Transform (DFT), which assumes periodic boundary conditions. That is, a circular shift by $\tau = (\tau_y, \tau_x)$:
$$d_\tau(y, x) = d((y - \tau_y) \bmod H, (x - \tau_x) \bmod W)$$
wraps pixels from the bottom edge to the top edge, and from the right edge to the left edge.

**Scientific Impact:**
- In real-world physical imaging, corruptions (e.g., lens raindrops, sensor noise patterns, camera shake) do not wrap toroidally around image borders.
- When $\tau$ approaches half the image dimension ($\tau_y \approx H/2, \tau_x \approx W/2$), residual energy originally near the image perimeter is injected into the semantic center of the image, while center features are displaced to the perimeter.
- **Remediation & Boundary Mitigation:**  
  In the confirmatory protocol, circular boundary artifacts can be suppressed by applying a Hann or Tukey spatial taper window $\mathcal{W}(y, x)$ to the residual $d$ prior to circular translation, or by evaluating only small localized spatial displacements $\tau \in [-16, +16]^2$ where perimeter wrapping is negligible.

---

### 2.2 Photographic Gamut Headroom & Boundary Clipping

**The Limitation:**  
Natural RGB photographic images frequently contain saturated specular highlights ($x_{i,j,c} = 1.0$) or clipped shadow regions ($x_{i,j,c} = 0.0$).
Under the unconstrained additive formulation:
$$x_+(\tau) = c + \frac{1}{2}\alpha d_\tau$$
if $c_{i,j,c} = 1.0$ and $d_{\tau, i,j,c} > 0$, the resulting pixel exceeds the display gamut $[0, 1]$. If standard clipping $\operatorname{clip}(x, 0, 1)$ is applied, the linearity of the intervention is destroyed, and the residual Fourier invariants are violated.

**Empirical Audit Finding:**  
In our audit across $10$ photographic images (`audit_pixel_validity.py`), even a tiny fraction of saturated pixels ($0.001\% - 0.037\%$) reduces the unconstrained global headroom $\alpha_{\max}$ to exactly zero.

**Protocol Resolution & Its Trade-Off:**  
To preserve exact invariant preservation without clipping, we introduced **Protocol B: Interior-Anchor Contraction**:
$$c_\eta = (1 - 2\eta)c + \eta, \quad \eta = 0.05$$
This contracts the anchor into $[0.05, 0.95]$, guaranteeing headroom $\alpha \ge 0.46$ for $100\%$ of natural images.  
*Limitation of Protocol B:* The anchor $c_\eta$ has slightly lower contrast than the raw anchor $c$. While neural predictions remain consistent (e.g., all models correctly classified `tench`), it introduces a minor domain shift.

---

### 2.3 Post-Softmax Probability Compression vs. Un-Normalized Logits

**The Limitation:**  
In our empirical ZeroGPU evaluation (`eval_zerogpu_translations.py`), ResNet-50 and ViT-Base were highly confident on the anchor image ($p_{\text{tench}} = 99.99988\%$ and $99.742\%$, respectively).
Because the derivative of the softmax function is:
$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$
for $p_i \approx 1.0$, the gradient approaches zero:
$$\frac{\partial p_i}{\partial z_i} \approx 1.0 \times (1.0 - 1.0) = 0$$

**Empirical Result:**  
ResNet-50 showed virtually zero variation in post-softmax confidence ($\Delta p < 10^{-6}$), not because the network's internal features were insensitive to the shift, but because the extreme confidence compressed the logits onto the flat plateau of the softmax sigmoid.

**Scientific Recommendation:**  
Future Phase III confirmatory studies must log raw un-normalized pre-softmax logits $z_{y^*}$ or log-odds $\log \frac{p}{1-p}$ to accurately measure internal neural susceptibility without non-linear output compression.

---

### 2.4 Single-Anchor Linearization vs. Non-Convex Decision Landscapes

**The Limitation:**  
The CINS prediction $\hat{r}(\tau) = \alpha \langle \nabla m(c), d_\tau \rangle$ evaluates the gradient at a single point in pixel space: the anchor midpoint $c$.
If the distance between $x_+$ and $x_-$ is large ($\alpha > 0.50$), or if the model's decision surface has high-frequency ripples (as observed in Swin-Transformer's $7\times 7$ window boundaries), the straight-line trajectory between $x_-$ and $x_+$ intersects multiple activation boundaries.

**Empirical Consequence:**  
As documented in `APPROXIMATION_ERROR_ANALYSIS.md`, while symmetric differences strictly cancel the quadratic Hessian term $\nabla^2 m(c)$, cubic terms $\nabla^3 m(c)$ begin to dominate for $\alpha \ge 0.50$. Practitioners cannot extrapolate first-order response maps to arbitrary perturbation magnitudes without recalibrating the gradient.

---

### 2.5 Host Environment Security & Compute Architecture Constraints

**The Limitation:**  
On the current Windows development workstation, the host security architecture enforces an **Application Control Policy (WDAC / AppLocker)** that blocks the execution of untrusted C-extensions and binary DLLs (`torch_python.dll`) in user directories (`OSError: [WinError 4551]`).

**Methodological Solution:**
1. The mathematical framework, invariant verifications, 2D FFT cross-correlation engine, and numerical unit tests were implemented in pure, highly optimized NumPy, SciPy, and Pillow—which execute with zero security restrictions and achieved a $1,972\times$ acceleration.
2. Genuine deep vision model forward inferences were executed via the live production Hugging Face ZeroGPU endpoint (NVIDIA A10G).
3. For full autograd backpropagation gradient extraction across large batches, a standalone, reproducible Kaggle GPU notebook (`cins_phase3_kaggle_experiment.ipynb`) and runner script (`cins_phase3_kaggle_runner.py`) are provided, enabling zero-friction cloud GPU execution.

---

## 3. Disambiguation from Prior Art & What We Do NOT Claim

To maintain absolute academic credibility:

1. **We DO NOT claim to have discovered operator non-commutativity:**  
   The non-commutativity of non-linear filters (e.g., median filtering and morphological operators) has been studied in classical digital signal processing for decades. Our contribution is isolating the specific algebraic commutator residual of noise, blur, and JPEG compression and demonstrating that its spatial translation reveals hidden architectural susceptibility in deep neural networks.
2. **We DO NOT claim to replace adversarial robustness methods:**  
   CINS is not an adversarial attack framework. Adversarial attacks optimize perturbations to minimize accuracy; CINS investigates the natural structural commutativity of standard physical corruptions under invariant spectral energy.
3. **We DO NOT claim perfect linear correlation for all architectures:**  
   As shown in our empirical ZeroGPU results, Swin-Tiny displays complex window-straddling dynamics that produce non-linear ripples, yielding lower correlation with smooth linear gradient proxies than ConvNeXt-Tiny.

---

## 4. Summary Table of Addressed vs. Open Challenges

| Challenge / Vulnerability | Status in Current Implementation | Future Research Pathway |
| :--- | :--- | :--- |
| **Spectral Invariant Verification** | **Completely Solved** (Verified to $< 10^{-13}$ precision) | Complete |
| **Curvature Cancellation** | **Completely Solved** (Rigorous Taylor proof of $\mathcal{O}(\alpha^3)$ error) | Complete |
| **Photographic Headroom** | **Solved via Protocol B** ($c_\eta$ contraction) | Investigate adaptive local contrast scaling |
| **Algorithmic Latency** | **Completely Solved** ($20.9\text{ ms}$ for $50,176$ evaluations) | Implement PyTorch CUDA 2D FFT kernel |
| **Softmax Saturation** | **Characterized** (Documented in Error Analysis) | Extract pre-softmax logits in Kaggle runner |
| **Window Boundary Aliasing (Swin)** | **Characterized** (Observed 8.61% empirical swing) | Formulate window-aware attention gradient kernels |
