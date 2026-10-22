# RobustLens: At Matched Image Fidelity, Does Corruption Order Still Matter?
## Research Supplement & Technical Appendix

**Author:** Aviral (Avi)  
**Date:** October 2026  
**Document Type:** Technical Appendix & Mathematical Derivations  
**Companion Paper:** "At Matched Image Fidelity, Does Corruption Order Still Matter? A Counterfactual Study of Vision Classifier Reliability"  

---

## Appendix A: Mathematical Non-Commutativity of Compound Degradations

We formally prove why the compound image degradation operators $\mathcal{N}$ (Additive Gaussian Noise), $\mathcal{B}$ (Gaussian Defocus Blur), and $\mathcal{J}$ (Lossy JPEG Compression) do not commute, and derive the exact mathematical basis for the physical distortion confound.

### A.1 Operator Definitions

Let $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ be an input image tensor normalized to $[0, 1]$.
1. **Additive Noise ($\mathcal{N}_{\sigma}$):**
   $$\mathcal{N}_{\sigma}(\mathbf{x}) = \text{clip}(\mathbf{x} + \mathbf{\epsilon}, 0, 1), \quad \mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$$
2. **Defocus Blur ($\mathcal{B}_{G}$):**
   $$\mathcal{B}_{G}(\mathbf{x}) = \mathbf{x} * G_{\sigma_B, k}$$
   where $*$ denotes 2D spatial convolution with normalized Gaussian kernel $G$, satisfying $\sum_{u, v} G(u, v) = 1$.
3. **JPEG Compression ($\mathcal{J}_q$):**
   $$\mathcal{J}_q(\mathbf{x}) = \text{IDCT}\left( \text{round}\left( \frac{\text{DCT}(\mathbf{x})}{Q(q)} \right) \odot Q(q) \right)$$
   where $\text{DCT}$ and $\text{IDCT}$ denote block-wise $8 \times 8$ Discrete Cosine Transforms, and $Q(q)$ is the standard JPEG quantization matrix parameterized by quality factor $q \in [1, 100]$.

### A.2 Non-Commutativity of Blur and Noise: $\mathcal{B} \circ \mathcal{N} \neq \mathcal{N} \circ \mathcal{B}$

Consider the two compositions acting on image $\mathbf{x}$:
* **Noise-First ($\mathcal{B} \circ \mathcal{N}$):**
  $$\mathbf{x}_{\mathcal{B} \circ \mathcal{N}} = (\mathbf{x} + \mathbf{\epsilon}) * G = (\mathbf{x} * G) + (\mathbf{\epsilon} * G)$$
* **Blur-First ($\mathcal{N} \circ \mathcal{B}$):**
  $$\mathbf{x}_{\mathcal{N} \circ \mathcal{B}} = (\mathbf{x} * G) + \mathbf{\epsilon}$$

In the continuous expectation without boundary clipping:
* The signal component is identical: $\mathbf{x} * G$.
* The noise field differs drastically:
  - Under $\mathcal{N} \circ \mathcal{B}$ (noise applied last), the perturbation is raw i.i.d. Gaussian noise with variance $\sigma^2$ per pixel.
  - Under $\mathcal{B} \circ \mathcal{N}$ (noise applied first), the perturbation is smoothed noise $\mathbf{\eta} = \mathbf{\epsilon} * G$. The variance of each output pixel is:
    $$\text{Var}(\mathbf{\eta}_{i, j}) = \sigma^2 \sum_{u, v} G(u, v)^2 = \sigma^2 \|G\|_2^2$$
    For a 2D Gaussian kernel of size $k \times k$ with bandwidth $\sigma_B > 1.0$:
    $$\|G\|_2^2 \approx \frac{1}{4 \pi \sigma_B^2} \ll 1$$
    For example, with $\sigma_B = 1.5$, $\|G\|_2^2 \approx \frac{1}{4 \pi (2.25)} \approx 0.035$. The effective noise variance is attenuated by over **$96\%$**.

Consequently, the Expected Mean Squared Error (MSE) satisfies:
$$\mathbb{E}[\text{MSE}(\mathbf{x}, \mathbf{x}_{\mathcal{N} \circ \mathcal{B}})] \gg \mathbb{E}[\text{MSE}(\mathbf{x}, \mathbf{x}_{\mathcal{B} \circ \mathcal{N}})]$$
proving that applying noise last inevitably causes far greater degradation in PSNR unless the nominal noise scale $\sigma$ is substantially reduced.

### A.3 Non-Commutativity with JPEG Quantization: $\mathcal{J} \circ \mathcal{N} \neq \mathcal{N} \circ \mathcal{J}$

JPEG compression applies a frequency-domain threshold. Quantization sets high-frequency DCT coefficients to zero if $|\text{DCT}(\mathbf{x})_{u, v}| < \frac{1}{2} Q(q)_{u, v}$.
* When noise is applied *before* JPEG ($\mathcal{J} \circ \mathcal{N}$), high-frequency noise components elevate high-frequency DCT coefficients above the quantization threshold, leading to severe quantization noise, color blockiness, and ringing artifacts.
* When JPEG is applied *before* noise ($\mathcal{N} \circ \mathcal{J}$), the JPEG step quantizes the clean signal into smooth $8 \times 8$ blocks, after which additive Gaussian noise is injected uniformly on top of the block boundaries.

Because neither operator commutes with the other, the 6 permutations partition into distinct physical distortion profiles in nominal benchmarks:
1. Terminal-Noise sequences (`BJN`, `JBN`): Lowest PSNR, lowest SSIM, highest high-frequency spectral energy.
2. Terminal-Blur sequences (`NJB`, `JNB`): Highest PSNR, highest SSIM, smoothed high frequencies.
3. Terminal-JPEG sequences (`NBJ`, `BNJ`): Intermediate PSNR, structured $8 \times 8$ blocking artifacts.

---

## Appendix B: Parameter Calibration & Matching Optimization

### B.1 Multi-Objective Optimization Problem

To achieve matched fidelity across all 6 sequences on common support, we formulated the multi-objective optimization problem:
$$\min_{\theta \in \Theta} \mathcal{L}(\theta; S, k) = w_1 \left( \frac{\overline{\text{PSNR}}(S, \theta) - \text{PSNR}^*_k}{\sigma_{\text{psnr}}} \right)^2 + w_2 \left( \frac{\overline{\text{SSIM}}(S, \theta) - \text{SSIM}^*_k}{\sigma_{\text{ssim}}} \right)^2$$
subject to:
$$0.001 \le \sigma_N \le 0.40, \quad 3 \le k_B \le 25 \text{ (odd)}, \quad 0.2 \le \sigma_B \le 8.0, \quad 5 \le q \le 95$$

### B.2 Training Calibration Split (Zero Leakage)

* Dataset: Official `johnowhitaker/imagenette2-320` mirror on Hugging Face.
* Total training rows: $9,468$.
* Verification sample: $N_{\text{cal}} = 50$ images sampled with fixed seed from rows $0 \dots 9,468$.
* Validation set: Strictly isolated (rows $9,469 \dots 13,393$, $N = 3,925$).
* No model evaluation was conducted during calibration; the metric evaluation was purely objective and deterministic.

### B.3 Achieved Common Support

| Target Level | Nominal PSNR Range | Matched PSNR Target | Achieved PSNR (Mean $\pm$ SD) | Achieved SSIM (Mean $\pm$ SD) |
| :--- | :--- | :--- | :--- | :--- |
| **Target 1** (Mild) | 26.2 – 30.1 dB | 28.0 dB | $27.65 \pm 0.42$ dB | $0.8197 \pm 0.015$ |
| **Target 2** (Mod-Mild) | 22.8 – 27.4 dB | 24.5 dB | $24.06 \pm 0.38$ dB | $0.6633 \pm 0.021$ |
| **Target 3** (Moderate) | 19.5 – 24.8 dB | 21.5 dB | $21.55 \pm 0.35$ dB | $0.5148 \pm 0.024$ |
| **Target 4** (Mod-Severe)| 16.8 – 22.1 dB | 18.5 dB | $19.13 \pm 0.41$ dB | $0.3881 \pm 0.026$ |
| **Target 5** (Severe) | 14.1 – 19.8 dB | 16.0 dB | $17.08 \pm 0.45$ dB | $0.3108 \pm 0.028$ |

Coarse MSE dispersion was substantially compressed, restoring common support across all 6 sequences.

### B.4 Empirical Evaluation of Calibration Feasibility and Pareto Trade-offs

To evaluate whether exact dual-tolerance matching ($\Delta\text{PSNR} \le \pm 0.75\,\text{dB}$, $\Delta\text{SSIM} \le \pm 0.035$) is achievable, we executed an extensive parameter grid search over $(\sigma_N, \sigma_B, q_J)$ on the independent training split across all 6 sequences and 5 target levels.

**Empirical Findings:**
1. **Target 1 (Mild, 28.0 dB / 0.850 SSIM):** Terminal-noise sequence `BJN` fails dual matching: at closest PSNR ($28.29$ dB, $\Delta = +0.29$ dB), SSIM reaches only $0.8137$ ($\Delta = -0.0363$, exceeding tolerance).
2. **Target 2 (Mod-Mild, 24.5 dB / 0.740 SSIM):** Terminal-noise sequences `BJN` and `JBN` achieve at best SSIM $\approx 0.621$ ($\Delta = -0.119$), whereas terminal-blur sequence `NJB` achieves SSIM $= 0.724$.
3. **Targets 4 and 5 (Moderate-Severe & Severe):** No evaluated sequence parameter combination satisfied both tolerances simultaneously. Calibration RMSE was $0.943$ dB PSNR (max residual $+2.639$ dB on terminal-JPEG sequence `NBJ` at Target 5) and $0.1380$ SSIM (max residual $-0.2379$ on terminal-noise sequence `BJN` at Target 3).
4. On the full validation set of 3,925 images, the cross-sequence SSIM spread reaches $0.096$ at Target 1 and $0.359$ at Target 5.

**Theoretical & Practical Conclusion:** Non-commuting spatial operators exhibit conflicting structural behaviors in the 2D $(\text{PSNR}, \text{SSIM})$ manifold. Because terminal blur preserves patch luminance correlation while terminal noise disrupts local covariance, simultaneous scalar parity across both metrics could not be achieved on the evaluated parameter grid. While this empirical search demonstrates severe multi-objective trade-offs between additive noise and spatial blur under standard discrete operator parameterization, a finite grid search does not constitute a proof of universal physical impossibility across all conceivable continuous transformations or alternative metric formulations. Rather, it demonstrates the practical infeasibility of joint calibration with these discrete operators, necessitating an approximate-fidelity framing with explicit documentation of remaining confounding. The study is therefore formally designated as an **Approximate-Fidelity Comparison**, and residual confounding is accounted for through covariate-adjusted factorial ANCOVA.


---

## Appendix C: Cluster-Robust Covariance Estimation

Because each image $\mathbf{x}_i$ ($i = 1, \dots, 3,925$) generates 31 paired observations, standard logistic regression standard errors are biased downward due to intra-image correlation.

We compute the cluster-robust Huber-White sandwich estimator:
$$\mathbf{V}_{\text{cluster}} = (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \left( \sum_{g=1}^{G} \mathbf{u}_g \mathbf{u}_g^T \right) (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1}$$
where $\mathbf{u}_g = \sum_{j \in g} \mathbf{x}_{gj} (y_{gj} - \hat{p}_{gj})$ is the cluster score vector for image cluster $g$, and $G = 3,925$.

Hypothesis tests for the $k$-dimensional restriction $\mathbf{R} \mathbf{\beta} = \mathbf{0}$ utilize the Wald statistic:
$$W = (\mathbf{R} \hat{\mathbf{\beta}})^T (\mathbf{R} \mathbf{V}_{\text{cluster}} \mathbf{R}^T)^{-1} (\mathbf{R} \hat{\mathbf{\beta}}) \sim \chi^2_k$$
This cluster-robust framework correctly preserves nominal Type I error rates under correlated repeated measures.
