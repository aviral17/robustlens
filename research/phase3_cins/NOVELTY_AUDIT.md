# Commutator-Induced Neural Susceptibility (CINS)
## Comprehensive Academic Novelty Audit & Prior Art Disambiguation

**Date:** November 2026  
**Branch:** `avi1444320-phase3`  
**Baseline Git Commit:** `fce29ff566b43e7951c440659e3e5815c16d91cb`  
**Repository:** https://github.com/aviral17/robustlens  
**Authorship Note:** Strictly decoupled from manuscript authorship and institutional affiliations per research directive.

---

### Executive Summary

This novelty audit evaluates whether the proposed framework—**Commutator-Induced Neural Susceptibility (CINS)**—occupies an unaddressed, scientifically distinct niche in computer vision robustness, or whether it substantially duplicates existing literature. 

Our investigation examined seven adjacent literatures:
1. **Compositional Robustness & Compound Corruptions** (e.g., ImageNet-C, ImageNet-P, Mason et al. 2023, TESTNAV 2024, RobustLens Phase I & II).
2. **Frequency-Domain Robustness & Fourier Sensitivity** (e.g., Yin et al. NeurIPS 2019, Saikia et al. 2021).
3. **Phase-Preserving / Phase-Scrambling Visual Attribution** (e.g., Oppenheim & Lim 1981, Geirhos et al. 2019, MACO NeurIPS 2023).
4. **Projected Displacement (PD) & Localized Threat Models** (e.g., Muthukumar, Pal, Sulam, Vidal, CVPR 2025).
5. **Adversarial Patch Translation & Location Sensitivity** (e.g., Brown et al. 2017, Karmon et al. 2018, PatchMap 2023).
6. **Gradient Attribution & First-Order Cross-Correlation** (e.g., Simonyan et al. 2013, Sundararajan et al. 2017, Fast Fourier Correlation / FFC).
7. **Local Linearity & Taylor Approximation in Deep Networks** (e.g., Goodfellow et al. 2015, Fawzi et al. 2018, Moosavi-Dezfooli et al. 2019).

**Verdict:** The core proposed mechanism of CINS—**translating contextual image operator order-swap residuals $d = \mathcal{O}_1(\mathcal{O}_2(x)) - \mathcal{O}_2(\mathcal{O}_1(x))$ under exact residual spectral invariance, and predicting the full $H \times W$ spatial margin response landscape via a single anchor gradient cross-correlated through 2D FFT**—does **not** exist in published literature. However, key components (such as FFT cross-correlation in classical signal processing, and gradient projection in adversarial attacks) are well-established. Therefore, CINS must be positioned not as a generic "world-first spatial analysis", but specifically as an **operator-commutator causal spatial intervention framework**.

---

### Detailed Comparative Analysis Across Research Strands

#### 1. Compositional Robustness & Operator Non-Commutativity
* **Prior Art:**
  - *Hendrycks & Dietterich (ICLR 2019)* introduced ImageNet-C evaluating 15 isolated, unary corruptions at 5 discrete severities.
  - *Mintun et al. (ICML 2021)* introduced ImageNet-P measuring temporal perturbation stability under continuous camera transforms.
  - *Mason et al. (OpenReview 2023)* tested compositional generalization of corruptions, arguing for modular sub-networks over data augmentation invariance.
  - *TESTNAV (2024)* used multi-objective Pareto optimization (NSGA-II) to balance corruption severity against perceptual fidelity (SSIM/KID).
  - *RobustLens Phase I (October 2026)* performed 973,400 forward passes establishing that corruption order matters ($NBJ \ne BNJ$) and drives paired prediction churn.
  - *RobustLens Phase II (November 2026)* performed 313,500 confirmatory forward passes on joint common support, identifying that Sobel edge energy change ($\Delta E_{\text{edge}}$) is the strongest observational fingerprint ($R^2$ up to $+0.0199$ on ViT-Base).
* **The Novelty Gap:**
  - Existing compositional robustness benchmarks treat corruption order as an *observational* condition: they measure accuracy drops for sequence $A \to B$ versus $B \to A$.
  - Phase II established *statistical associations* between physical fingerprints (Sobel, Fourier radial bands) and model margins, but could not establish *causality* because changing order simultaneously alters spatial edge alignment, block boundaries, and frequency balance.
  - **CINS Contribution:** CINS isolates the spatial alignment of the order discrepancy from its frequency content by translating the residual $d$ across the image plane while holding its Fourier power spectrum strictly constant.

#### 2. Frequency-Domain Robustness vs. Residual Spectral Invariance
* **Prior Art:**
  - *Yin et al. (NeurIPS 2019)* ("A Fourier Perspective on Model Robustness in Computer Vision") systematically probed CNNs by adding synthetic 2D Fourier basis noise across spatial frequencies, discovering that CNNs are sensitive to high-frequency noise and that standard data augmentations bias models toward low-frequency reliance.
  - *Saikia et al. (2021)* and subsequent works investigated frequency filtering (bandpass, low-pass, high-pass) on robustness.
  - *Phase Scrambling Literature (Oppenheim & Lim 1981, Geirhos et al. 2019)*: Randomizing phase angles while preserving Fourier magnitude yields unrecognizable textures, demonstrating human phase reliance vs. CNN texture bias.
  - *MACO (NeurIPS 2023)*: Magnitude-Constrained Optimization in feature visualization freezes natural Fourier magnitude and optimizes phase angles to produce realistic attribution maps.
* **The Novelty Gap:**
  - Yin et al. and related works modify the image's overall Fourier spectrum (adding energy to specific $(u, v)$ coordinates or filtering bands).
  - Phase scrambling alters relative phase across frequencies, destroying spatial coherence.
  - **CINS Distinction:** CINS does **not** scramble phases or inject synthetic frequency noise. Instead, circular spatial translation of the real compound order residual $d_\tau = \text{roll}(d, \tau)$ applies a linear phase ramp $e^{-i 2\pi (u \tau_y / H + v \tau_x / W)}$ to the residual. This guarantees:
    $$|\mathcal{F}(d_\tau)| \equiv |\mathcal{F}(d)|, \quad \|\mathcal{F}(d_\tau)\|^2 \equiv \|\mathcal{F}(d)\|^2, \quad \|d_\tau\|_2 \equiv \|d\|_2$$
    Residual power, $L_2$ norm, and pixel histogram remain invariant, while the residual is spatially repositioned across semantic image features.

#### 3. Projected Displacement (PD) Threat Model (CVPR 2025)
* **Prior Art:**
  - *Muthukumar, Pal, Sulam, Vidal (CVPR 2025, arXiv:2501.18098)* ("Disentangling Safe and Unsafe Corruptions via Anisotropy and Locality"):
    - Proposes Projected Displacement (PD) to replace isotropic $\ell_p$ balls.
    - Defines a threat model based on projecting input perturbations onto local "unsafe directions" derived from training data label boundaries.
    - Demonstrates that high-$\ell_p$ natural corruptions (blur, noise, compression) are "safe" if their projection onto unsafe directions is small, whereas adversarial attacks align directly with unsafe directions.
* **The Novelty Gap:**
  - PD is a *threat model definition* that projects arbitrary perturbations onto class-discriminative subspaces.
  - PD does not investigate operator non-commutativity, does not examine compound corruption sequencing, does not translate residuals spatially, and does not use 2D FFT cross-correlation to construct response landscapes.
  - **CINS Distinction:** CINS investigates where the physical non-commutativity of real image operators ($NBJ$ vs. $BNJ$) aligns with the network's decision boundary gradient, using spatial translation interventions.

#### 4. Adversarial Patch Location Sensitivity & Sliding Window Evaluations
* **Prior Art:**
  - *Brown et al. (2017)*, *Karmon et al. (2018)*: Showed that localized adversarial patches can force misclassification regardless of background.
  - *PatchMap (2023)*: Systematically slid an adversarial patch across every grid position $(x, y)$ of an image to map model vulnerability hot-spots.
  - *Limitation:* Exhaustive sliding window evaluation requires $H \times W$ full neural network forward passes per image (e.g., $224 \times 224 = 50,176$ forward passes per image). For 100 images, this requires $> 5,000,000$ forward passes.
* **The Novelty Gap:**
  - Patch evaluations apply synthetic, unconstrained adversarial patches designed to maximize loss.
  - PatchMap and brute-force sliding evaluations are computationally expensive ($\mathcal{O}(H \cdot W)$ network forward passes).
  - **CINS Distinction:**
    1. CINS evaluates the *natural physical residual* arising from operator non-commutativity, not an artificial adversarial patch.
    2. CINS computes the **entire 50,176-point response map in a single step ($\sim 1\text{ ms}$)** by cross-correlating the shared-anchor gradient with the residual via 2D FFT ($\mathcal{O}(H W \log(HW))$), eliminating brute-force inference.
    3. CINS explicitly investigates the mathematical breakdown of this first-order FFT prediction against actual neural inference.

#### 5. Gradient Attribution & Cross-Correlation
* **Prior Art:**
  - Classical saliency maps (Simonyan et al. 2013) compute input gradients $g = \nabla_x z_c(x)$.
  - Integrated Gradients (Sundararajan et al. 2017) integrates gradients along a straight path from baseline to input.
  - Convolution Theorem in Deep Learning: standard forward/backward passes in convolutional layers use FFT or Winograd algorithms to accelerate spatial convolutions.
* **The Novelty Gap:**
  - Attribution methods explain model predictions at a *fixed* input image.
  - They do not construct a continuous spatial intervention field of counterfactual inputs generated by operator commutators.
  - **CINS Distinction:** CINS uses the anchor gradient as a spatial linear functional $\langle g, d_\tau \rangle$ to predict how the decision margin will react to translating an order-swap residual to any position $\tau \in \{0, \dots, H-1\} \times \{0, \dots, W-1\}$.

---

### Exact Boundaries of the Proposed CINS Research Contribution

To maintain scientific integrity and prevent overclaiming, CINS must explicitly state what it **is** and what it **is not**:

1. **What CINS IS:**
   - A causal intervention methodology that translates physical compound corruption order residuals across an image under exact spectral and energy invariance.
   - An analytical framework that connects operator non-commutativity ($[N, B] \ne 0$) to neural decision response landscapes.
   - A fast computational screening tool ($\approx 1\text{ ms}$ via 2D FFT) that predicts spatial hot-spots and cold-spots for order sensitivity without running 50,176 forward passes.
   - An empirical investigation into the validity boundaries of local linearity (activation boundary crossings, competing class switches, and intervention strength $\alpha$).

2. **What CINS IS NOT:**
   - It is **not** a claim that FFT cross-correlation is mathematically new (cross-correlation via FFT is a classical 19th/20th-century theorem).
   - It is **not** a claim that neural networks are globally linear (the paper explicitly studies approximation *failure*).
   - It is **not** a universal defense or certified robustness guarantee.
   - It does **not** claim that circular translations reflect real camera physics at the image borders (boundary-aware non-circular controls are explicitly analyzed).

---

### Conclusion & Scientific Go/No-Go Decision

**Scientific Verdict: GO.**  
The proposed research occupies a genuine, defensible novelty gap at the intersection of:
- Empirical computer vision corruption non-commutativity.
- Spectrally invariant spatial translation interventions.
- First-order FFT-accelerated neural response prediction.
- Quantitative characterization of neural approximation breakdown.

Proceeding with the formal mathematical derivation and experimental protocol.
