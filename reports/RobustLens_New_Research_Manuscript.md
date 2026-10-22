# Towards Matched-Fidelity Robustness Benchmarking: An Approximate-Fidelity Counterfactual Study of Multi-Stage Vision Classifier Vulnerabilities

**Author:** Aviral (Avi)  
**Affiliation:** Independent Research in Computer Vision & Statistical Machine Learning  
**Date:** October 2026  
**Repository & Reproducibility:** [github.com/aviral199/robustlens](https://github.com/aviral199/robustlens)  
**Dataset:** Imagenette-v2 Validation Split ($N = 3,925$ images, 10 canonical ImageNet classes)  
**Scale:** 4 Pretrained Architectures $\times$ 31 Evaluated Conditions $\times$ 3,925 Physical Images = **486,700 Balanced Prediction Records**  

---

## Abstract

Evaluating the robustness of deep computer vision models against multi-stage, composite corruptions is critical for safety-critical autonomous systems. Recent benchmarks demonstrate substantial *order effects*: permuting the execution sequence of identical image degradation operators (Gaussian Noise $\mathcal{N}$, Defocus Blur $\mathcal{B}$, and JPEG Compression $\mathcal{J}$) induces compound top-1 accuracy disparities of up to 21 percentage points across standard vision architectures. However, these nominal benchmarks suffer from an unaddressed physical confounding mechanism: the mathematical operators do not commute algebraically ($\mathcal{N} \circ \mathcal{B} \circ \mathcal{J} \neq \mathcal{B} \circ \mathcal{J} \circ \mathcal{N}$), resulting in drastically divergent physical image fidelity. Terminal-noise sequences (`BJN`, `JBN`) inject unattenuated high-frequency noise spikes, depressing Peak Signal-to-Noise Ratio (PSNR) by over 4 dB, whereas terminal-blur sequences (`NJB`, `JNB`) smooth both sensor noise and JPEG block edges, yielding higher Structural Similarity (SSIM). Sequences ending in JPEG compression (`NBJ`, `BNJ`) exhibit structured block quantization boundaries.

In this study, we formulate a counterfactual evaluation framework to investigate whether sequence effects persist when degradations are calibrated toward common physical fidelity. We conduct an independent-partition parameter calibration on training images ($N_{\text{cal}} \subset [0, 9468]$, zero validation leakage) across all 6 permutations targeting 5 fidelity levels (28.0 dB down to 16.0 dB PSNR). Across an extensive parameter grid search on the calibration partition, **the prespecified joint matching criteria ($\pm 0.75$ dB PSNR, $\pm 0.035$ SSIM) could not be met simultaneously across sequences**, reflecting severe multi-objective trade-offs between additive noise and spatial blur. While this empirical search demonstrates the practical infeasibility of joint calibration with these discrete operators, a finite grid search does not constitute a proof of universal physical impossibility across all conceivable continuous transformations. Consequently, we evaluate an **approximate-fidelity design** on common support across exactly 486,700 inferences on 3,925 physical images across ConvNeXt-Tiny, Swin-Tiny, ViT-Base/16, and ResNet-50.

Our empirical findings reveal:
1. **Substantial Gap Contraction:** Calibrating to approximate common support contracts the nominal cross-sequence accuracy gap by **$77.1\%$ in ConvNeXt-Tiny** ($20.45\% \to 4.67\%$) and **$72.6\%$ in ResNet-50** ($21.55\% \to 5.90\%$), demonstrating that nominal benchmarks heavily confound sequence ordering with differential distortion magnitude.
2. **Persistent Architectural Sensitivity & Residual Confounding:** In Vision Transformers, sequence gaps persist more strongly (ViT-Base: $10.03\%$; Swin-Tiny: $11.68\%$). Clustered factorial logistic ANCOVA reveals that sequence remains a statistically significant factor ($p < 10^{-23}$). However, residual SSIM differences ($0.096$ to $0.359$ spread) demonstrate that complete physical decoupling was not achieved via scalar parameter tuning alone.
3. **The Churn Decoupling Phenomenon:** Despite compressed aggregate accuracy gaps, individual-sample error churn ($G + L = P(C_A \neq C_B)$) averages **$10.9\%$ to $13.4\%$**, with excess churn accounting for **$56.9\%$ to $75.5\%$** of all prediction shifts.

---

## 1. Introduction & Physical Confounding

Deep neural networks deployed in real-world visual systems (e.g., automated driving perception, aerial robotics, and clinical diagnostics) must process multi-stage image degradations. Photons incident on a sensor encounter optical defocus blur ($\mathcal{B}$), Poisson-Gaussian sensor noise ($\mathcal{N}$), and lossy compression serialization ($\mathcal{J}$).

Standard benchmarks such as ImageNet-C [@hendrycks2019benchmarking] evaluate isolated unary degradations. To reflect real-world composite shifts, recent benchmarks like R-Bench [@li2025rbench] introduced sequential compound pipelines. In nominal compound evaluations, operators are applied using fixed nominal parameters across permutation sequences. Evaluating 4 canonical architectures across all 6 permutations of $\{\mathcal{N}, \mathcal{B}, \mathcal{J}\}$ on 3,925 images (486,700 predictions) reveals severe nominal accuracy gaps:
* ConvNeXt-Tiny achieved $66.44\%$ on terminal-blur sequence `NJB` but collapsed to $45.99\%$ on terminal-noise sequence `BJN` (gap: $20.45\%$; terminal-JPEG `BNJ` reached $48.63\%$).
* ResNet-50 achieved $57.27\%$ on terminal-blur sequence `JNB` ($56.88\%$ on `NJB`) but collapsed to $35.72\%$ on terminal-noise sequence `BJN` (gap: $21.55\%$; terminal-JPEG `BNJ` reached $37.30\%$).
* Swin-Tiny reached $62.94\%$ on `JNB` vs. $45.81\%$ on `BNJ` (gap: $17.13\%$), and ViT-Base reached $60.14\%$ on `NJB` vs. $39.77\%$ on `BNJ` (gap: $20.37\%$).
* In nominal benchmarking, sequences ending in JPEG (`NBJ`, `BNJ`) and noise (`BJN`, `JBN`) suffered the greatest degradation, whereas sequences ending in blur (`NJB`, `JNB`) maintained the highest accuracy.

### 1.1 The Physical Confounding Mechanism

Before attributing these disparities to an innate architectural vulnerability to corruption order, one must examine the physical signal processing:

Let $\mathbf{x} \in [0, 1]^{H \times W \times C}$ be a clean image. Consider permutations `NJB` (terminal blur) and `JBN` (terminal noise):
$$\mathbf{x}_{\text{NJB}} = \mathcal{B}(\mathcal{J}(\mathbf{x} + \mathbf{\epsilon})), \quad \mathbf{\epsilon} \sim \mathcal{N}(0, \sigma^2)$$
$$\mathbf{x}_{\text{JBN}} = \mathcal{B}(\mathcal{J}(\mathbf{x})) + \mathbf{\epsilon}$$

Because spatial convolution ($\mathcal{B}$) and lossy JPEG quantization ($\mathcal{J}$) do not commute with additive noise ($\mathcal{N}$):
1. In `NJB` (and `JNB`), the terminal Gaussian blur $\mathcal{B}$ convolves the image with a 2D Gaussian kernel $G_\sigma$. Linear convolution acts as a low-pass filter, attenuating high-frequency noise variance by $\|G_\sigma\|_2^2 \ll 1$ and smoothing $8 \times 8$ JPEG block boundaries.
2. In `JBN` (and `BJN`), noise $\mathbf{\epsilon}$ is injected *last*. It is never low-pass filtered. High-frequency pixel perturbations remain unattenuated.
3. In `NBJ` and `BNJ`, JPEG compression $\mathcal{J}$ is applied *last*, quantizing DCT coefficients and leaving distinct $8 \times 8$ grid blocking artifacts.

Empirical measurements on nominal severity 3 images demonstrate the distortion gap: `NJB` achieves a Mean PSNR of **$25.2$ dB** and SSIM of **$0.76$**, whereas `JBN` collapses to **$21.1$ dB** PSNR and **$0.58$** SSIM—a physical distortion discrepancy of **$4.1$ dB PSNR** and **$0.18$ SSIM**. Nominal benchmarks thus fundamentally confound execution sequence with total physical degradation.

---

## 2. Counterfactual Design & The Feasibility Barrier

To disentangle sequence ordering from distortion magnitude, we formulated a counterfactual calibration protocol.

### 2.1 Prespecified Matching Protocol & Hypotheses
Let $S \in \{\text{NBJ}, \text{NJB}, \text{BNJ}, \text{BJN}, \text{JNB}, \text{JBN}\}$ denote the sequence, governed by parameters $\theta(S) = (\sigma_N, k_B, \sigma_B, q_J)$.
We established 5 target fidelity levels on common support:
* Target 1 (Mild): Target PSNR = $28.0$ dB, Target SSIM = $0.850$
* Target 2 (Moderate-Mild): Target PSNR = $24.5$ dB, Target SSIM = $0.740$
* Target 3 (Moderate): Target PSNR = $21.5$ dB, Target SSIM = $0.620$
* Target 4 (Moderate-Severe): Target PSNR = $18.5$ dB, Target SSIM = $0.480$
* Target 5 (Severe): Target PSNR = $16.0$ dB, Target SSIM = $0.360$

Prespecified tolerances were set to: $\Delta \text{PSNR} \le \pm 0.75\,\text{dB}$ and $\Delta \text{SSIM} \le \pm 0.035$.
Calibration was executed exclusively on an independent training partition ($N_{\text{cal}} = 50$, rows $0..9468$), completely blind to validation images, model logits, and ground-truth labels.

### 2.2 Calibration Feasibility & Trade-offs Under Non-Commuting Operators
A critical scientific finding of this investigation is that **the prespecified dual-tolerance criteria ($\Delta \text{PSNR} \le \pm 0.75\,\text{dB}$, $\Delta \text{SSIM} \le \pm 0.035$) could not be satisfied jointly across all six sequences**.

A systematic grid search over $(\sigma_N, \sigma_B, q_J)$ on the independent training split confirmed that:
* At Target 1, terminal-noise sequence `BJN` fails dual matching: at PSNR = $28.29$ dB, SSIM reaches only $0.8137$ ($\Delta = -0.0363$, exceeding tolerance).
* At Target 2, terminal-noise sequences `BJN` and `JBN` achieve at best SSIM $\approx 0.621$ ($\Delta = -0.119$) when PSNR is near $25.4$ dB, whereas terminal-blur sequence `NJB` achieves SSIM $= 0.724$.
* At Targets 4 and 5, no evaluated sequence parameter combination satisfied both tolerances simultaneously. Across all conditions, the calibration PSNR RMSE was **$0.943$ dB** (maximum residual $+2.639$ dB on terminal-JPEG sequence `NBJ` at Target 5) and SSIM RMSE was **$0.1380$** (maximum residual $-0.2379$ on terminal-noise sequence `BJN` at Target 3).
* On the held-out validation set ($N = 3,925$), the resulting cross-sequence spread reached $1.35$ dB PSNR and $0.096$ SSIM at Target 1, widening to $3.10$ dB PSNR and $0.359$ SSIM at Target 5.

**Structural Analysis:** The non-commutativity of spatial convolution ($\mathcal{B}$), non-linear DCT quantization ($\mathcal{J}$), and high-frequency additive noise ($\mathcal{N}$) imposes distinct structural trade-offs:
1. Terminal-blur sequences (`NJB`, `JNB`) end in spatial convolution, which low-pass filters high frequencies and smooths JPEG block edges, elevating SSIM relative to MSE.
2. Terminal-noise sequences (`BJN`, `JBN`) end in additive Gaussian noise, which disrupts local patch luminance and covariance without attenuation, sharply depressing SSIM.
3. Terminal-JPEG sequences (`NBJ`, `BNJ`) end in block quantization, creating structured $8 \times 8$ grid discontinuities.

While this extensive grid search demonstrates that standard scalar parameter tuning across discrete degradation operators cannot simultaneously meet both tolerances, a finite grid search does not constitute a proof of universal physical impossibility across all conceivable continuous transformations or alternative perceptual quality metrics. We therefore designate this benchmark as an **Approximate-Fidelity Comparison** and explicitly evaluate remaining confounding via statistical covariates.

### 2.3 Scientific Reclassification: Approximate-Fidelity Comparison
In accordance with methodological integrity, we designate this experiment as an **Approximate-Fidelity Comparison** rather than an exact matched-fidelity study. While exact dual-metric parity within prespecified tolerances was not achieved, the calibrated parameter grid successfully compresses the nominal $4.1$ dB PSNR spread into common target bands ($28\text{ dB} \to 16\text{ dB}$), establishing common support across all 6 sequences while making residual SSIM disparities explicit.

| Target Level | Target PSNR / SSIM | Sequence | Noise $\sigma$ | Blur $(k, \sigma)$ | JPEG $q$ | Achieved PSNR (dB) | Achieved SSIM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Target 1** (Mild) | 28.0 dB / 0.850 | NBJ | 0.050 | $k=3, \sigma=0.85$ | 80 | 28.49 | 0.8691 |
| | | NJB | 0.046 | $k=5, \sigma=1.05$ | 76 | 26.88 | 0.8499 |
| | | BNJ | 0.040 | $k=3, \sigma=0.70$ | 85 | 28.07 | 0.7875 |
| | | BJN | 0.024 | $k=3, \sigma=0.85$ | 84 | 27.58 | 0.7814 |
| | | JNB | 0.046 | $k=5, \sigma=1.00$ | 76 | 27.01 | 0.8390 |
| | | JBN | 0.024 | $k=3, \sigma=0.85$ | 84 | 27.86 | 0.7914 |
| **Target 2** (Mod-Mild) | 24.5 dB / 0.740 | NBJ | 0.082 | $k=5, \sigma=1.45$ | 65 | 24.98 | 0.7671 |
| | | NJB | 0.084 | $k=7, \sigma=1.68$ | 56 | 23.95 | 0.7354 |
| | | BNJ | 0.060 | $k=5, \sigma=1.15$ | 76 | 24.32 | 0.6197 |
| | | BJN | 0.042 | $k=5, \sigma=1.35$ | 72 | 23.56 | 0.5642 |
| | | JNB | 0.085 | $k=7, \sigma=1.68$ | 55 | 23.89 | 0.7236 |
| | | JBN | 0.042 | $k=5, \sigma=1.35$ | 72 | 23.66 | 0.5701 |
| **Target 3** (Moderate) | 21.5 dB / 0.620 | NBJ | 0.130 | $k=7, \sigma=2.20$ | 48 | 22.76 | 0.6628 |
| | | NJB | 0.142 | $k=9, \sigma=2.65$ | 36 | 21.83 | 0.6240 |
| | | BNJ | 0.095 | $k=7, \sigma=1.75$ | 64 | 21.16 | 0.4163 |
| | | BJN | 0.064 | $k=7, \sigma=1.95$ | 58 | 20.84 | 0.3821 |
| | | JNB | 0.145 | $k=9, \sigma=2.65$ | 35 | 21.80 | 0.6177 |
| | | JBN | 0.064 | $k=7, \sigma=1.95$ | 58 | 20.89 | 0.3856 |
| **Target 4** (Mod-Severe)| 18.5 dB / 0.480 | NBJ | 0.220 | $k=11, \sigma=3.20$ | 32 | 20.49 | 0.5513 |
| | | NJB | 0.260 | $k=13, \sigma=4.10$ | 18 | 19.49 | 0.5170 |
| | | BNJ | 0.145 | $k=9, \sigma=2.60$ | 46 | 18.48 | 0.2593 |
| | | BJN | 0.095 | $k=9, \sigma=2.70$ | 44 | 18.41 | 0.2427 |
| | | JNB | 0.265 | $k=13, \sigma=4.10$ | 17 | 19.46 | 0.5138 |
| | | JBN | 0.095 | $k=9, \sigma=2.70$ | 44 | 18.44 | 0.2444 |
| **Target 5** (Severe) | 16.0 dB / 0.360 | NBJ | 0.320 | $k=15, \sigma=4.40$ | 18 | 18.64 | 0.4761 |
| | | NJB | 0.380 | $k=17, \sigma=5.20$ | 10 | 17.71 | 0.4629 |
| | | BNJ | 0.220 | $k=11, \sigma=3.55$ | 31 | 16.04 | 0.1575 |
| | | BJN | 0.135 | $k=11, \sigma=3.50$ | 30 | 16.20 | 0.1533 |
| | | JNB | 0.385 | $k=17, \sigma=5.20$ | 9 | 17.70 | 0.4608 |
| | | JBN | 0.135 | $k=11, \sigma=3.50$ | 30 | 16.22 | 0.1542 |

*Table 1: Calibrated multi-stage corruption parameters and achieved fidelity on the independent training split ($N_{\text{cal}}=50$).*

![Figure 1: Quality Matching Validation](file:///c:/STATISTICS%20&%20CV/robustlens/reports/figures/fig1_matched_fidelity_validation.png)
*Figure 1: Objective Distortion Matching. Top: Nominal severity confound where terminal noise causes an uncalibrated 4 dB PSNR drop. Bottom: Calibrated approximate-fidelity support.*

---

## 3. Experimental Architecture & Statistical Methodology

We evaluate four canonical pretrained vision architectures:
1. **ConvNeXt-Tiny** (`convnext_tiny.in12k_ft_in1k`, 28.6M params): Modern pure convolutional network.
2. **Swin-Tiny** (`swin_tiny_patch4_window7_224.ms_in1k`, 28.3M params): Hierarchical Vision Transformer.
3. **ViT-Base/16** (`vit_base_patch16_224.augreg_in21k_ft_in1k`, 86.6M params): Isotropic Vision Transformer.
4. **ResNet-50** (`resnet50.a1_in1k`, 25.6M params): Classical residual CNN baseline.

Evaluation was performed on the official **Imagenette-v2 validation partition** ($N = 3,925$ images). Each model was evaluated on 1 clean condition and 30 compound conditions (6 sequences $\times$ 5 target levels), producing **121,675 inferences per model** and **486,700 total balanced prediction records**.

### 3.1 Clustered Factorial Logistic ANCOVA
To handle repeated measurements on identical physical images without inflating degrees of freedom, we fit cluster-robust logistic regression models ($N = 117,750$ compound observations per model, clustered on $3,925$ image IDs):
$$\text{logit}(P(Y_{ij} = 1)) = \beta_0 + \sum_{s} \alpha_s \cdot \text{Seq}_{ij}^{(s)} + \sum_{k} \gamma_k \cdot \text{Tgt}_{ij}^{(k)} + \sum_{s, k} \delta_{sk} (\text{Seq} \times \text{Tgt})_{ij} + \beta_1 \text{PSNR}_{ij} + \beta_2 \text{SSIM}_{ij}$$
Covariate adjustment explicitly accounts for residual PSNR and SSIM variations across sequences.

---

## 4. Empirical Findings

### 4.1 Sequence Gap Contraction
Calibrating multi-stage degradations to common support dramatically shrinks nominal sequence accuracy gaps:
* **ConvNeXt-Tiny:** Maximum sequence gap collapsed from **$20.45\%$ down to $4.67\%$** (a **$77.1\%$ gap contraction**, with cross-sequence accuracy variance contracting by **$95.9\%$**).
* **ResNet-50:** Sequence gap collapsed from **$21.55\%$ down to $5.90\%$** (**$72.6\%$ gap contraction**, **$94.9\%$ variance reduction**).
* **ViT-Base/16:** Sequence gap contracted from **$20.37\%$ to $10.03\%$** (**$50.7\%$ gap contraction**, **$86.9\%$ variance reduction**).
* **Swin-Tiny:** Sequence gap contracted from **$17.13\%$ to $11.68\%$** (**$31.8\%$ gap contraction**, **$55.1\%$ variance reduction**).

| Architecture | Clean Accuracy | Approximate Compound Accuracy | Nominal Gap | Approximate Gap | Gap Contraction | Variance Reduction | Mean ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ConvNeXt-Tiny** | $87.13\%$ | $58.11\%$ | $20.45\%$ | **$4.67\%$** | **$77.1\%$** | $95.9\%$ | $0.0653$ |
| **Swin-Tiny** | $86.04\%$ | $56.27\%$ | $17.13\%$ | **$11.68\%$** | **$31.8\%$** | $55.1\%$ | $0.1039$ |
| **ViT-Base/16** | $80.25\%$ | $53.31\%$ | $20.37\%$ | **$10.03\%$** | **$50.7\%$** | $86.9\%$ | $0.0420$ |
| **ResNet-50** | $87.36\%$ | $48.47\%$ | $21.55\%$ | **$5.90\%$** | **$72.6\%$** | $94.9\%$ | $0.0919$ |

*Table 2: Comparison of Nominal Severity vs. Approximate-Fidelity performance, demonstrating major gap contraction in CNNs and persistent dispersion in Transformers.*

![Figure 2: Sequence Gap Contraction](file:///c:/STATISTICS%20&%20CV/robustlens/reports/figures/fig2_sequence_gap_contraction.png)
*Figure 2: Accuracy by sequence under Nominal Severity vs. Approximate-Fidelity support.*

### 4.2 Inversion of Nominal Degradation Hierarchy & Residual Confounding
In nominal benchmarks, terminal-noise sequences (`BJN`, `JBN`) and terminal-JPEG sequence `BNJ` suffered the lowest accuracies. Under approximate fidelity calibration, an apparent hierarchy inversion emerges:
* In Swin-Tiny: terminal-noise sequences `JBN` ($62.12\%$) and `BJN` ($61.23\%$) maintained higher accuracy than terminal-blur sequences `NJB` ($50.44\%$) and `JNB` ($52.31\%$), while terminal-JPEG sequences achieved intermediate accuracies (`NBJ`: $57.61\%$, `BNJ`: $53.93\%$).
* In ResNet-50: terminal-noise sequences `JBN` ($51.08\%$) and `BJN` ($50.63\%$) outperformed terminal-blur sequences `NJB` ($46.76\%$) and `JNB` ($47.05\%$), with terminal-JPEG sequences at `NBJ` ($50.14\%$) and `BNJ` ($45.18\%$).

**Critical Scientific Caveat:** This inversion must not be interpreted as proof that vision architectures are inherently more robust to noise than blur. Rather, to force terminal-blur sequences (`NJB`, `JNB`) down to target PSNRs of $18.5$ and $16.0$ dB, blur kernel radii had to be expanded drastically ($k=13..17$, $\sigma=4.1..5.2$). Heavy blur eliminates high-frequency spatial edge transitions and phase information that patch embeddings require for token discrimination. Simultaneously, residual SSIM in terminal blur remained substantially higher ($0.46$ vs $0.15$ in `BJN`). The apparent ranking inversion is thus an artifact of trading spatial edge clarity against additive noise variance under scalar PSNR matching.

### 4.3 Factorial Clustered ANCOVA Results
Even when controlling for continuous image-level PSNR and SSIM, sequence main effects and sequence $\times$ target interactions remain overwhelmingly significant:

| Model | Pseudo $R^2$ | Order Main (5 df) | Target Main (4 df) | Order $\times$ Target (20 df) | Joint Order (25 df) | PSNR Coef ($z$) | SSIM Coef ($z$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ConvNeXt-Tiny** | $0.1379$ | $W=119.4$ ($p=4.2 \times 10^{-24}$) | $W=395.3$ ($p=2.8 \times 10^{-84}$) | $W=272.0$ ($p=4.0 \times 10^{-46}$) | $W=611.0$ ($p=6.1 \times 10^{-113}$) | $+0.032$ ($z=2.33$) | $+1.528$ ($z=6.78$) |
| **Swin-Tiny** | $0.1283$ | $W=159.7$ ($p=1.2 \times 10^{-32}$) | $W=227.7$ ($p=4.2 \times 10^{-48}$) | $W=651.1$ ($p=4.8 \times 10^{-125}$) | $W=1436.9$ ($p=5.0 \times 10^{-288}$) | $+0.046$ ($z=3.36$) | $+1.850$ ($z=7.95$) |
| **ViT-Base/16** | $0.0879$ | $W=130.1$ ($p=2.3 \times 10^{-26}$) | $W=344.9$ ($p=2.2 \times 10^{-73}$) | $W=547.6$ ($p=3.0 \times 10^{-103}$) | $W=1276.5$ ($p=8.8 \times 10^{-254}$) | $+0.072$ ($z=5.46$) | $+0.577$ ($z=2.61$) |
| **ResNet-50** | $0.2034$ | $W=268.9$ ($p=4.8 \times 10^{-56}$) | $W=774.7$ ($p=2.3 \times 10^{-166}$) | $W=239.9$ ($p=1.2 \times 10^{-39}$) | $W=794.4$ ($p=1.8 \times 10^{-151}$) | $+0.063$ ($z=4.87$) | $+1.680$ ($z=6.92$) |

*Table 3: Full Factorial Clustered Logistic ANCOVA results ($N=117,750$ compound observations per model, clustered on 3,925 physical images).*

![Figure 3: Factorial Interaction Profiles](file:///c:/STATISTICS%20&%20CV/robustlens/reports/figures/fig3_factorial_interaction_profiles.png)
*Figure 3: Sequence degradation trajectories across the 5 target fidelity levels.*

### 4.4 Error Churn & Microscopic Prediction Volatility
Evaluating models solely by aggregate top-1 accuracy conceals substantial sample-level instability. Across 300 matched condition pairs, we compute:
* Net accuracy difference: $\Delta = |G - L|$
* Total error churn: $\text{Churn} = G + L = P(C_A \neq C_B)$
* Excess error churn: $\text{Excess} = (G + L) - |G - L| = 2 \min(G, L)$

| Architecture | Mean Net Diff ($|G - L|$) | Mean Error Churn ($G + L$) | Mean Excess Churn | Excess Churn Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **ConvNeXt-Tiny** | $2.71\%$ | $10.93\%$ | $8.22\%$ | **$75.5\%$** |
| **Swin-Tiny** | $5.94\%$ | $11.92\%$ | $5.97\%$ | **$56.9\%$** |
| **ViT-Base/16** | $4.65\%$ | $13.40\%$ | $8.75\%$ | **$67.0\%$** |
| **ResNet-50** | $3.77\%$ | $13.10\%$ | $9.33\%$ | **$70.9\%$** |

*Table 4: Error churn and excess churn across 300 sequence pairs.*

Even under calibrated common support, classifiers flip predictions on **$11\%$ to $13\%$** of physical images simply due to operator reordering. Over **$57\%$ to $75\%$** of this churn is excess volatility that produces zero net change in benchmark accuracy.

![Figure 4: Churn Decoupling Analysis](file:///c:/STATISTICS%20&%20CV/robustlens/reports/figures/fig4_matched_churn_decoupling.png)
*Figure 4: Decoupling of Net Accuracy Difference and Error Churn under approximate physical fidelity.*

![Figure 5: Calibration Analysis](file:///c:/STATISTICS%20&%20CV/robustlens/reports/figures/fig5_matched_calibration.png)
*Figure 5: Expected Calibration Error (ECE) and compound accuracy across architectures.*

---

## 5. Limitations & Recommendations for Robustness Benchmarking

1. **Remaining Confounding Under Approximate Fidelity:** Scalar parameter tuning across discrete degradation operators failed to meet the prespecified joint matching criteria ($\pm 0.75$ dB PSNR, $\pm 0.035$ SSIM), leaving substantial residual SSIM spread ($0.096$ to $0.359$). While this empirical grid search demonstrates severe practical trade-offs between additive noise and spatial blur, a finite grid search does not prove universal physical impossibility across all conceivable continuous transformations or alternative perceptual metrics. Multi-stage robustness studies must report multi-metric diagnostics (PSNR, SSIM, and perceptual metrics) transparently and avoid claiming that confounding has been eliminated.
2. **Convolution vs. Attention Inductive Biases:** Pure convolutional architectures show greater stability under common support (gap contracting by over $72\%$), whereas Vision Transformers retain substantial sensitivity ($10\%–12\%$ gaps). The interaction between fixed patch partitioning grids and JPEG $8 \times 8$ block boundaries creates an intrinsic structural sensitivity in ViTs.
3. **Mandatory Reporting of Error Churn:** In safety-critical deployments, aggregate benchmark scores are insufficient. Robustness evaluations must report sample-level prediction churn alongside top-1 accuracy.

---

## References

1. D. Hendrycks and T. Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations," *International Conference on Learning Representations (ICLR)*, 2019.
2. D. Hendrycks, N. Mu, E. D. Cubuk, B. Zoph, J. Gilmer, and B. Lakshminarayanan, "AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty," *International Conference on Learning Representations (ICLR)*, 2020.
3. H. Li, Y. Wen, Y. Long, Y. Li, and B. Li, "R-Bench: Evaluating Model Robustness under Multi-Stage Composed Corruptions," *IEEE Journal of Selected Topics in Signal Processing*, 2025.
4. P. Goyal, D. Mahajan, A. Gupta, and D. Caruana, "Evaluating the Robustness of Vision Transformers under Compound Distortions," *IEEE International Conference on Computational Photography (ICCP)*, 2022.
5. S. Dodge and L. Karam, "Understanding how image quality affects deep neural networks," *IEEE International Conference on Quality of Multimedia Experience (QoMEX)*, 2016.
6. B. RichardWebster, S. E. Anthony, and W. J. Scheirer, "Visual Psychophysics for Making Evaluation in Deep Learning," *European Conference on Computer Vision (ECCV)*, 2018.
7. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," *International Conference on Machine Learning (ICML)*, 2017.
8. A. N. Angelopoulos and S. Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification," *arXiv:2107.07511*, 2021.
