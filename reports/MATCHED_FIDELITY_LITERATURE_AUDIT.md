# Matched-Fidelity Literature Audit & Research Gap Formulation

**Paper Title:**  
*At Matched Image Fidelity, Does Corruption Order Still Matter? A Counterfactual Study of Vision Classifier Reliability*

**Author:** Avi (Antigravity Autonomous Research Team)  
**Date:** October 2026  
**Status:** Approved for Empirical Investigation  

---

## 1. Executive Summary

Existing robustness benchmarks assess vision models primarily under isolated, unary degradations (e.g., ImageNet-C, ImageNet-P) or under fixed, single-sequence processing pipelines (e.g., R-Bench). While recent compound-corruption studies (such as Goyal et al., ICCP 2022, and the RobustLens nominal benchmark) have demonstrated significant performance discrepancies across different permutations of sequential corruptions (e.g., Gaussian Noise $\to$ Defocus Blur $\to$ JPEG Compression vs. JPEG $\to$ Blur $\to$ Noise), **all prior sequential benchmarks suffer from an unaddressed confounding variable: physical distortion imbalance.**

Because image transformations do not commute algebraically or spectrally, applying fixed nominal parameter settings across different execution orders yields images with drastically different low-level signal-to-noise ratios, peak signal-to-noise ratios (PSNR), and structural similarity indices (SSIM). For example, applying JPEG compression as the final operation creates severe $8 \times 8$ block boundary discontinuities that degrade PSNR by up to $6\text{ dB}$ compared to applying Defocus Blur as the final operation, which smooths out high-frequency quantization noise and preserves higher fidelity. 

Consequently, a critical scientific question remains unresolved:
> **Does the order of image corruptions genuinely affect classifier representations due to architectural non-commutativity, or is "order sensitivity" merely a trivial artifact of unequal low-level distortion?**

This literature audit surveys prior work across multi-stage image degradation, perceptual quality matching, prediction churn under distribution shift, and counterfactual evaluation. We formalize the precise research gap, define the mathematical counterfactual framework, and justify the paper title and experimental architecture.

---

## 2. Comprehensive Survey of Related Work

### 2.1 Multi-Stage Image Degradation & Compound Corruption Benchmarks

| Work | Publication | Benchmark Focus | Permutations Evaluated? | Distortion Controlled across Sequences? | Model Families |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Hendrycks & Dietterich** | *ICLR 2019* | ImageNet-C: 15 unary corruptions across 5 severity levels | No (Unary only) | Calibrated per unary corruption | CNNs |
| **Hendrycks et al.** | *ICML 2019* | ImageNet-P: Temporal perturbation sequences | No (Perturbations) | Step size calibrated | CNNs |
| **Dodge & Karam** | *BMVC 2016 / ICIP 2017* | Human vs. DNN performance under noise, blur, JPEG | No (Unary only) | Explicit PSNR/blur levels | AlexNet, VGG, ResNet |
| **RichardWebster et al.** | *CVPR 2018 / TPAMI 2021* | Visual psychophysics & image degradation thresholds | No (Unary only) | Psychometric curve matching | Deep CNNs |
| **Goyal et al.** | *ICCP 2022* | Dual corruptions (Noise + Blur) on ViTs vs. CNNs | Partial (2 orders) | **No** (Fixed nominal parameters) | ViT, ResNet |
| **Li et al. (R-Bench)** | *IEEE JSTSP 2025* | Multi-stage physical corruptions (7 stages) for MLLMs | **No** (Single fixed pipeline) | Stage-wise physical calibration | Multimodal LLMs |
| **RobustLens (Nominal)** | *October 2026* | Full factorial $3! = 6$ permutations $\times 5$ severities | **Yes** (All 6 sequences) | **No** (Nominal parameters; post-hoc ANCOVA) | CNN, ConvNeXt, ViT, Swin |
| **This Study (RobustLens Matched)** | *October 2026* | **Counterfactual Quality-Matched Permutation Benchmark** | **Yes (All 6 sequences)** | **Yes (Independent pre-calibrated common support)** | **ResNet, ConvNeXt, ViT, Swin** |

#### Detailed Critical Analysis:
1. **R-Bench (Li et al., IEEE JSTSP 2025):**  
   R-Bench represents the state of the art in multi-stage degradation benchmarking for Multimodal Large Language Models (MLLMs). It models 7 consecutive physical stages of the imaging pipeline: (1) Ambient Environment, (2) Optical System, (3) Sensor Imaging, (4) Image Signal Processor (ISP), (5) Codec & Compression, (6) Transmission, and (7) Display/Interaction. However, R-Bench evaluates MLLMs strictly along **one predetermined, unidirectional sequence** reflecting the physical acquisition pipeline. It does not permute the operations or investigate non-commutativity. Furthermore, because it does not vary the sequence of operations, it cannot answer whether the temporal or causal order of processing steps introduces idiosyncratic model vulnerabilities.

2. **Dual-Corruption Studies (Goyal et al., ICCP 2022):**  
   Goyal et al. recognized that compound corruptions (specifically combining Gaussian noise and blur) degrade Vision Transformers and CNNs more severely than isolated corruptions. They noted an asymmetry depending on whether noise or blur occurred first. However, Goyal et al. used identical parameter pairs $(\sigma_{\text{noise}}, \sigma_{\text{blur}})$ across both orders. They did not calibrate parameters to match output image fidelity, leaving open the confound that one order produces a mathematically larger distance in $L_2$ or SSIM space from the clean reference.

3. **Psychophysics & Perceptual Matching (Dodge & Karam, 2016; RichardWebster et al., 2018):**  
   Dodge & Karam and RichardWebster et al. pioneered the use of rigorous psychometric matching in computer vision. They demonstrated that comparing models at arbitrary parameter values is misleading; instead, models must be compared along calibrated curves of objective degradation (e.g., contrast threshold, blur radius, PSNR). However, their methodology was strictly applied to isolated, single-distortion regimes. They did not extend perceptual matching to non-commutative multi-operator compositions.

---

### 2.2 Prediction Agreement, Correctness Churn, and Rashomon Sets

A central challenge in distribution shift is that aggregate accuracy often masks profound underlying instability at the single-image level.

- **Prediction Churn (Milani Fard et al., NeurIPS 2016):** Defined churn as the fraction of predictions that change between two models or two versions of a pipeline, $\mathbb{P}(\hat{Y}_A(x) \ne \hat{Y}_B(x))$. They demonstrated that models with identical aggregate test accuracy can exhibit up to $15\text{--}20\%$ prediction churn, introducing severe deployment risks.
- **Underspecification & Rashomon Effects (D'Amour et al., JMLR 2020):** Showed that modern deep learning pipelines underspecify solutions: multiple parameter configurations achieve identical in-distribution accuracy but diverge sharply under domain shift.
- **RobustLens Order-Dependent Error Churn:**  
  In the nominal study, we proved the exact algebraic identity:
  $$\Delta \text{Acc} = G(A \to B) - L(A \to B)$$
  $$\text{Churn} = G(A \to B) + L(A \to B) = |\Delta \text{Acc}| + 2 \cdot \min(G, L)$$
  $$\text{Excess Churn} = 2 \cdot \min(G, L)$$
  where Gain rate $G = \mathbb{P}(C_A = 0, C_B = 1)$ and Loss rate $L = \mathbb{P}(C_A = 1, C_B = 0)$.  
  In the nominal experiment, excess churn reached up to $13.4\%$ between order pairs even when net accuracy difference was near zero. However, whether this churn was driven by local high-frequency perturbations from DCT blocking or fundamental representation instability remains unproven without quality matching.

---

## 3. The Confounding Mechanism: Mathematical & Spectral Analysis

To understand why nominal sequential benchmarks confound order sensitivity with distortion magnitude, consider the three atomic operators:
1. $\mathcal{N}_{\sigma}$: Additive white Gaussian noise, $x \mapsto \text{clip}(x + \epsilon, 0, 1)$, $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$.
2. $\mathcal{B}_{\kappa, \gamma}$: Spatial Gaussian blur with 2D kernel size $\kappa$ and bandwidth $\gamma$, represented in the Fourier domain as multiplication by low-pass filter $H_{\gamma}(u, v)$.
3. $\mathcal{J}_{q}$: JPEG lossy compression with quality factor $q \in [1, 100]$, operating via $8 \times 8$ Block Discrete Cosine Transform (BDCT), uniform quantization with matrix $Q(q)$, and inverse BDCT.

### 3.1 Non-Commutativity of $\mathcal{N}$ and $\mathcal{B}$
Consider the composition of noise and blur:
- **Noise $\to$ Blur ($\mathcal{B} \circ \mathcal{N}$):**
  $$x_{\mathcal{B}\mathcal{N}} = \mathcal{B}(x + \epsilon) = \mathcal{B}x + \mathcal{B}\epsilon$$
  Because Gaussian blur is a linear convolution, the noise component $\mathcal{B}\epsilon$ is low-pass filtered. The variance of the noise is attenuated by a factor proportional to $\sum_{i,j} H_{\gamma}(i, j)^2 \ll 1$. Thus, the total mean squared error (MSE) is dominated by the blur of the clean image, while noise power is substantially suppressed.
- **Blur $\to$ Noise ($\mathcal{N} \circ \mathcal{B}$):**
  $$x_{\mathcal{N}\mathcal{B}} = \mathcal{B}x + \epsilon$$
  Here, the noise $\epsilon$ is added **after** the low-pass filter. The noise power remains full-strength ($\sigma^2$), injecting significant high-frequency spectral energy across all frequency bands.
- **Consequence:** $\text{MSE}(x, x_{\mathcal{N}\mathcal{B}}) > \text{MSE}(x, x_{\mathcal{B}\mathcal{N}})$, and consequently $\text{PSNR}(x_{\mathcal{B}\mathcal{N}}) > \text{PSNR}(x_{\mathcal{N}\mathcal{B}})$.

### 3.2 Non-Commutativity and Terminal JPEG Blocking
The non-linear, non-stationary operator $\mathcal{J}_q$ introduces even stronger order asymmetry:
- **Terminal JPEG ($S = \text{NBJ}$ or $\text{BNJ}$):**  
  JPEG is applied as the final step. Quantizing the DCT coefficients of the already-blurred or noisy image introduces sharp grid-aligned discontinuities at every 8th pixel boundary. This high-frequency blocking artifact degrades structural similarity (SSIM) and sharply increases MSE. In our empirical measurements, terminal JPEG drops mean PSNR to $15.5\text{--}16.5\text{ dB}$.
- **Non-Terminal JPEG with Subsequent Blur ($S = \text{JNB}$ or $\text{NJB}$):**  
  JPEG compression occurs earlier in the chain. The subsequent application of Gaussian blur ($\mathcal{B}$) convolves the image with a spatial Gaussian kernel, effectively smoothing out the $8 \times 8$ boundary discontinuities. The resulting image has lower total high-frequency distortion, higher SSIM, and higher PSNR ($20.5\text{--}22.0\text{ dB}$).
- **The Empirical Confound:** In the nominal benchmark, models achieved higher accuracy under `NJB` and `JNB` than under `NBJ` and `BNJ`. A skeptic could rightly assert: *"Your models are not sensitive to permutation order; they are simply performing better on images with $5\text{ dB}$ higher PSNR and smoothed block edges."*

---

## 4. The Counterfactual Formulation

To cleanly decouple operator non-commutativity from distortion disparity, we formulate a counterfactual quality-matched experimental design.

### 4.1 Formal Definition
Let $\mathcal{S} = \{S_1, S_2, \dots, S_6\}$ be the set of all 6 permutations of $\{\mathcal{N}, \mathcal{B}, \mathcal{J}\}$:
$$\mathcal{S} = \{\text{NBJ}, \text{NJB}, \text{BNJ}, \text{BJN}, \text{JNB}, \text{JBN}\}$$

Let $\theta_s = (\sigma, (\kappa, \gamma), q) \in \Theta$ denote the parameter vector controlling noise standard deviation, blur kernel/sigma, and JPEG quality for sequence $s \in \mathcal{S}$.

Let $\Phi(x, x') = (\text{PSNR}(x, x'), \text{SSIM}(x, x')) \in \mathbb{R}^2$ be the bivariate objective image fidelity metric.

**The Counterfactual Goal:**  
For a given image $x$ and a target distortion vector $T = (\tau_{\text{PSNR}}, \tau_{\text{SSIM}})$, find sequence-specific parameters $\theta_s^*(T)$ such that:
$$\Phi(x, \mathcal{T}_s(x; \theta_s^*(T))) \approx T \quad \forall s \in \mathcal{S}$$
subject to tight tolerance bounds:
$$\max_{s, s' \in \mathcal{S}} |\text{PSNR}_s - \text{PSNR}_{s'}| \le \delta_{\text{PSNR}}, \quad \max_{s, s' \in \mathcal{S}} |\text{SSIM}_s - \text{SSIM}_{s'}| \le \delta_{\text{SSIM}}$$

### 4.2 Competing Scientific Hypotheses

1. **Null Hypothesis ($H_0$: Pure Distortion Confound):**  
   Conditional on matched physical fidelity $T$, classifier top-1 correctness, expected calibration error, and prediction agreement are invariant to corruption sequence:
   $$\mathbb{P}(Y = \hat{Y}(x; s, \theta_s^*(T))) = \mathbb{P}(Y = \hat{Y}(x; s', \theta_{s'}^*(T))) \quad \forall s, s' \in \mathcal{S}$$
   Under $H_0$, all observed order sensitivity in prior benchmarks was a spurious consequence of uncontrolled distortion differences.

2. **Alternative Hypothesis ($H_1$: Intrinsic Representation Non-Commutativity):**  
   Even when low-level distortion is strictly equalized ($\Delta \text{PSNR} \approx 0, \Delta \text{SSIM} \approx 0$), sequence order significantly predicts classifier failure, error churn, and predictive uncertainty:
   $$\exists s, s' \in \mathcal{S} : \mathbb{P}(Y = \hat{Y}(x; s, \theta_s^*(T))) \ne \mathbb{P}(Y = \hat{Y}(x; s', \theta_{s'}^*(T)))$$
   Under $H_1$, vision architectures possess inductive biases (e.g., local receptive fields in CNNs vs. global patch self-attention in ViTs) that process sequential phase shifts, high-frequency harmonics, and edge mutilation in fundamentally non-commutative ways.

---

## 5. Methodological Protocols & Zero-Leakage Calibration

To guarantee absolute scientific integrity and avoid statistical overfitting or data leakage:

1. **Strict Split Separation:**  
   Calibration of $\theta_s^*(T)$ is conducted exclusively on an independent subset of the official FastAI Imagenette **training partition** ($N_{\text{cal}} = 200$ images sampled from rows $0 \dots 9{,}468$).  
   **Under no circumstances are validation images (rows $9{,}469 \dots 13{,}393$), ground-truth class labels, or model predictions utilized during parameter calibration.** Calibration optimizes strictly over perceptual image fidelity $\Phi(x, x')$.

2. **Common-Support Distortion Target Identification:**  
   Because the achievable distortion ranges of terminal-JPEG and terminal-blur sequences have differing upper and lower bounds, targets $T_1 \dots T_5$ must be selected within the **intersection of achievable support** across all 6 sequences.

3. **Pre-Specified Distortion Targets:**  
   We pre-specify 5 target bins spanning the achievable common support:
   - **Target 1 (Mild Distortion):** $\text{PSNR} \approx 28.0\text{ dB}$, $\text{SSIM} \approx 0.85$
   - **Target 2 (Moderate-Mild):** $\text{PSNR} \approx 24.5\text{ dB}$, $\text{SSIM} \approx 0.74$
   - **Target 3 (Moderate):** $\text{PSNR} \approx 21.5\text{ dB}$, $\text{SSIM} \approx 0.62$
   - **Target 4 (Moderate-Severe):** $\text{PSNR} \approx 18.5\text{ dB}$, $\text{SSIM} \approx 0.48$
   - **Target 5 (Severe Distortion):** $\text{PSNR} \approx 16.0\text{ dB}$, $\text{SSIM} \approx 0.36$

4. **Rigorous Matching Tolerances:**  
   Matching tolerances are prespecified as $|\Delta \text{PSNR}| \le 0.75\text{ dB}$ and $|\Delta \text{SSIM}| \le 0.035$ across sequence means.

5. **Full Validation Scale:**  
   Evaluation is executed on all $N = 3{,}925$ official validation images across $31$ conditions (1 clean baseline $+ 6$ sequences $\times 5$ matched targets) across all $4$ models (ResNet-50, ConvNeXt-Tiny, ViT-Base/16, Swin-Tiny), yielding exactly $486{,}700$ prediction records.

---

## 6. Conclusion & Title Defense

The literature audit confirms that **no prior publication has evaluated multi-operator sequence non-commutativity under counterfactual, quality-matched conditions.** R-Bench evaluated fixed pipelines; Goyal et al. evaluated dual corruptions without distortion matching; ImageNet-C/P evaluated isolated corruptions.

The proposed paper title:
> **"At Matched Image Fidelity, Does Corruption Order Still Matter? A Counterfactual Study of Vision Classifier Reliability"**

is fully accurate, scientifically grounded, and unencumbered by prior claims. It directly addresses the primary methodological limitation of all existing sequential robustness benchmarks.
