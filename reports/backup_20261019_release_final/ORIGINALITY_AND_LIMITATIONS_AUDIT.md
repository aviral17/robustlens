# Originality, Prior-Art Survey, and Limitations Audit: Order Has Consequences

**Project:** RobustLens Research Benchmark  
**Title:** Order Has Consequences: Total Effects and Fidelity-Conditioned Prediction Instability under Compound Image Corruptions  
**Artifact:** Final Research Audit & Literature Positioning  
**Author:** Avi (RobustLens Research Lead)  
**Date:** October 2026  
**Repository Path:** `reports/final_research_package/ORIGINALITY_AND_LIMITATIONS_AUDIT.md`  

---

## 1. Executive Summary & Research Positioning

Robustness benchmarking in computer vision has evolved from single-distortion perturbations (e.g., ImageNet-C, ImageNet-P) toward multi-stage and compound degradations. However, when multiple image processing operations are composed sequentially, their mathematical non-commutativity ($\mathcal{T}_A \circ \mathcal{T}_B \neq \mathcal{T}_B \circ \mathcal{T}_A$) introduces a fundamental experimental dilemma that requires separating two distinct research estimands:

1. **Estimand 1: Total Operational Effect of Sequence Order (Nominal Regime):**  
   Applying identical nominal parameter vectors across different permutation orders creates unequal low-level distortion (e.g., terminal JPEG compression generates severe $8 \times 8$ block boundary artifacts, whereas subsequent Gaussian blur smooths those discontinuities, yielding a $\sim 5\text{ dB}$ PSNR difference). Under this operational estimand, output fidelity disparities are downstream consequences of operator non-commutativity, forming part of the compound treatment pathway.
2. **Estimand 2: Order Differences at Comparable Output Fidelity (Approximate-Fidelity Regime):**  
   Calibrating operator parameters on an independent training set to target common perceptual fidelity bins (PSNR and SSIM) contracts accuracy gaps substantially (by $63.18\%$ overall, and up to $77.15\%$ in ConvNeXt). However, because prespecified tolerances ($\pm 0.75\text{ dB}$ PSNR, $\pm 0.035$ SSIM) were not achieved over the tested discrete grid, this regime provides an **approximate-fidelity descriptive comparison**, rather than an exact causal effect at matched fidelity. Confounding was reduced, but not eliminated.

This audit establishes the exact boundaries of prior art, specifies the precise methodological and empirical contributions of RobustLens, and provides a transparent accounting of experimental limitations. In compliance with strict academic integrity standards:
- We distinguish measured empirical facts from statistical interpretations.
- We separate the **nominal-order study** (confounded regime) from the **approximate-fidelity study** (calibrated common-support regime).
- We explicitly report that prespecified matching tolerances were **not met** over the tested parameter grid, avoiding unsupported claims of "eliminated confounding" or "causal effects."
- We do not claim global priority or "world-first" status, grounding all findings against published and contemporary literature.

---

## 2. Comprehensive Survey of Related Work

The following matrix compares RobustLens against foundational, recent, and contemporary literature across multi-stage image degradation, operator chain forensics, prediction churn, and perceptual quality matching.

### Table 1: Systematic Prior-Art Comparison Matrix

| Benchmark / Literature | Core Objective | Deg. Operators & Sequence Control | Permutation Design | Fidelity Control across Orders | Metrics Evaluated | Architectural Scope | Key Limitation Addressed by RobustLens |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **Hendrycks & Dietterich (ICLR 2019)**<br>*ImageNet-C* | Benchmark common corruption robustness. | 15 unary corruptions across 4 categories (noise, blur, weather, digital). | None (Unary only) | Calibrated per unary corruption against humans. | Top-1 Error, mCE. | ResNet, AlexNet, VGG. | Zero compound or sequential operators; cannot evaluate operator interaction or sequence order. |
| **Hendrycks et al. (ICML 2019)**<br>*ImageNet-P* | Measure prediction stability across video frame perturbations. | Perturbation trajectories (affine, noise, blur). | Forward temporal sequences only. | Step sizes calibrated per trajectory. | Flip rate, Top-5 stability. | ResNet, DenseNet. | Evaluates temporal frame-to-frame shifts, not operator permutation order on static inputs. |
| **Milani Fard et al. (NeurIPS 2016)**<br>*Launch & Iterate: Reducing Prediction Churn* | Quantify and reduce prediction instability (churn) in iterative machine learning models. | Model retraining and pipeline iterations. | N/A | N/A | Churn $\mathbb{P}(\hat{Y}_A \neq \hat{Y}_B)$. | Tabular / DNN classifiers. | Formulated prediction churn to reduce churn between model revisions, not for permutation-induced input shifts under identical frozen weights. |
| **Dodge & Karam (QoMEX 2016 / ICIP 2017)** | Human vs. DNN performance under noise, blur, compression. | Unary Gaussian noise, Gaussian blur, JPEG. | None (Unary only) | Matched to psychometric curves and PSNR. | Accuracy vs. objective distortion. | AlexNet, VGG-16, ResNet-50. | Restricted entirely to isolated unary degradations; no multi-operator compositions. |
| **Liao et al. (IEEE JSTSP 2020)**<br>*Operator Chain Forensics* | Forensic detection of image processing operator execution order. | Multi-operator chains (filtering, resizing, compression). | Sequential chains evaluated for detection. | Fixed operator parameters. | Detection accuracy of chain ordering. | Forensics CNNs. | Focuses on forensic detection of processing history; does not evaluate semantic classifier robustness, fidelity matching, or prediction churn across architectures. |
| **Mintun et al. (NeurIPS 2021)**<br>*On Interaction Between Augmentations & Corruptions* | Interaction between data augmentations and corruptions in natural corruption robustness. | Data augmentations paired with corruptions. | Fixed pipeline sequences. | Evaluated across fixed ImageNet-C severities. | Effective robustness, accuracy. | ResNet, Modern CNNs. | Evaluates interactions under fixed pipelines, not the non-commutative permutation group of operators. |
| **Goyal et al. (ICCP 2022)**<br>*Dual Corruptions* | Robustness of vision models under noise-blur dual corruptions. | Pairwise combinations (Noise + Blur in low-light imaging). | 2 orders evaluated ($\mathcal{N}\circ\mathcal{B}$ vs $\mathcal{B}\circ\mathcal{N}$). | **No** (Identical nominal parameters used). | Top-1 Accuracy. | ResNet-50, ViT-Base. | Limited to dual operators; no 3-way factorial design, no fidelity matching, no churn decomposition. |
| **Li et al. (IEEE JSTSP 2025)**<br>*R-Bench* | Multi-stage physical corruptions for Multimodal LLMs. | 7 sequential stages modeling capture-to-display. | None (Single fixed pipeline order). | Stage-wise physical calibration. | MLLM benchmark scores. | Multimodal LLMs (GPT-4V, LLaVA). | Fixed unidirectional sequence reflecting acquisition; does not permute operators or test order invariance. |
| **Wei et al. (arXiv:2609.35583, 2026)**<br>*Paired Evaluations* | Explores what paired evaluations reveal under visual perturbations. | Unary synthetic corruptions & physical recapture. | None (Clean vs. perturbed pairs). | Synthetic ImageNet-C severities and camera prints. | Lost acceptance, true-class prob drop, bounds. | 44 Vision Classifiers. | Evaluates clean-to-corrupted instance pairs for sample prioritization; does not permute multi-stage compound operators. |
| **Erkara & Mayer (Neurocomputing 2026)**<br>*Blind Confusion & ACD* | Black-box robustness and confidence divergence under common and structured corruptions. | 15 unary corruptions (Gaussian, speckle, salt & pepper, random lines, confusion blocks). | None (Unary only) | Evaluated across fixed corruption severity levels. | Top-1 Error, Corruption Error (CE), Accuracy Confidence Divergence (ACD). | 37 Architectures (CNNs, Noise-Robust CNNs, ViT, DeiT, Swin). | Evaluates isolated common and structured unary corruptions; no sequential multi-operator compositions, no permutation group testing, no churn decomposition. |
| **RobustLens: Nominal Study (Ours, 2026)** | Benchmark vision models under full permutation set of 3 canonical operators. | Full $3! = 6$ permutations of {Noise, Blur, JPEG} $\times 5$ nominal severities. | **Full $3! = 6$ Factorial** | **No** (Nominal parameters; total operational effect). | Accuracy, Brier, Churn ($G, L$), Clustered Bootstrap. | ResNet-50, ConvNeXt-Tiny, ViT-Base, Swin-Tiny. | Demonstrates massive nominal gaps (up to $21.55\%$), separating the total operational effect from baseline confounding. |
| **RobustLens: Approx-Fidelity Study (Ours, 2026)** | Counterfactual evaluation targeting matched PSNR & SSIM support across permutations. | Full $3! = 6$ permutations across 5 calibrated target distortion bins. | **Full $3! = 6$ Factorial** | **Approximate** (Pre-calibrated on train set; tolerances failed). | Accuracy, 5-State Transitions, Saturated 6-Cell Contrasts. | ResNet-50, ConvNeXt-Tiny, ViT-Base, Swin-Tiny. | Decouples nominal scaling; quantifies 31.8%–77.2% gap contraction while transparently documenting residual spreads. |

---

## 3. Demarcation of Prior Art vs. Novelty

To prevent inflated academic claims, we explicitly delineate what is established prior art versus the original contributions of this work:

### 3.1 Established Prior Art (Not Claimed as Novel)
1. **Operator Non-Commutativity in Image Processing:** The fact that 2D spatial convolution ($\mathcal{B}$), additive stochastic perturbation ($\mathcal{N}$), and block discrete cosine transform quantization ($\mathcal{J}$) do not commute is a textbook property of digital signal processing (Oppenheim & Schafer; Gonzalez & Woods).
2. **Forensic Traceability of Operator Sequences:** Liao et al. (IEEE JSTSP 2020) demonstrated that CNNs can detect the sequential order of image operations from forensic artifacts.
3. **Prediction Churn Concept:** Defining churn as prediction disagreement $\mathbb{P}(\hat{Y}_A \neq \hat{Y}_B)$ originated in machine learning stability literature (Milani Fard et al., NeurIPS 2016, "Launch and Iterate: Reducing Prediction Churn").
4. **Dual-Corruption Asymmetry:** Observing that Noise $\to$ Blur behaves differently from Blur $\to$ Noise was reported by Goyal et al. (ICCP 2022).
5. **ImageNet-C Corruption Formulas:** The mathematical formulations of Gaussian noise, Gaussian blur, and JPEG compression follow Hendrycks & Dietterich (ICLR 2019).
6. **Unary Corruption Benchmarks and Confidence Divergence:** Benchmarking classification models under common/structured unary perturbations and measuring accuracy-confidence divergence (e.g., Erkara & Mayer, Neurocomputing 2026) are established in the literature.

### 3.2 Original Contributions of RobustLens
1. **Full Factorial $3! = 6$ Permutation Benchmark with Seed-Matched Noise:**  
   Unlike prior dual-corruption or fixed-pipeline studies, RobustLens evaluates the complete symmetric group $\mathfrak{S}_3$ (6 permutations) across 5 severities on identical physical images with strictly seed-matched noise realizations, generating $486{,}700$ evaluated records per study across 4 distinct vision architectures (ResNet-50, ConvNeXt-Tiny, ViT-Base/16, Swin-Tiny).
2. **Complete 5-State Joint Prediction Space Decomposition:**  
   We formalize the exact joint prediction partition:
   $$\mathbb{P}(E_{11}) + \mathbb{P}(E_{10}) + \mathbb{P}(E_{01}) + \mathbb{P}(E_{00}^{\text{same}}) + \mathbb{P}(E_{00}^{\text{diff}}) = 1$$
   proving that:
   $$\text{Label Churn} = \text{Correctness Churn} + \mathbb{P}(E_{00}^{\text{diff}})$$
   We demonstrate that Error Migration ($E_{00}^{\text{diff}}$) averages $23.85\%$ (nominal) and $22.40\%$ (matched), causing Predicted-Label Churn ($34\text{--}40\%$) to be more than double Correctness Churn ($12\text{--}17\%$). We reveal severe class-level error migration (e.g., $39.41\%$ for 'church').
3. **Full-Rank Saturated Six-Cell Decomposition with Balanced Parity:**  
   We implement a full-rank linear decomposition spanning all 6 sequence cell means:
   $$\mu = X \beta$$
   demonstrating that terminal blur effects flip from positive ($+12.27\%$) to negative ($-6.82\%$) upon low-level calibration, while balanced parity contrasts between even and odd permutations remain statistically significant ($p < 10^{-100}$). We report direct marginal means and contrasts alongside reference-coded parameters.
4. **Empirical Common-Support Calibration and Gap Contraction Analysis:**  
   We present the first empirical investigation attempting to pre-calibrate compound sequence parameters against joint PSNR and SSIM targets on a held-out training partition. We document how equalizing low-level distortion contracts nominal sequence gaps by $31.8\%$ to $77.2\%$, while rigorously documenting the residual failure modes of joint calibration.

---

## 4. Methodological Analysis: Finite Grid Search vs. Universal Impossibility

A key finding in our approximate-fidelity investigation is that prespecified matching tolerances ($|\Delta\text{PSNR}| \le 0.75\text{ dB}$, $|\Delta\text{SSIM}| \le 0.035$) were **not satisfied** across the six permutations:
- Achieved PSNR RMSE: $0.943\text{ dB}$ (maximum absolute residual: $+2.639\text{ dB}$ on `NBJ` Target 5).
- Achieved SSIM RMSE: $0.1380$ (maximum absolute residual: $-0.2379$ on `BJN` Target 3).
- Validation cross-sequence PSNR spread reached $3.10\text{ dB}$ at Target 5; SSIM spread reached $0.359$.

### Crucial Scientific Clarification
In earlier working notes, this outcome was colloquially referred to as "proving the physical impossibility of joint PSNR and SSIM matching." **This phrasing is scientifically imprecise and overbroad.**

A discrete grid search over a constrained three-parameter family $(\sigma, (\kappa, \gamma), q)$ evaluated on $N_{\text{cal}} = 50$ training images demonstrates only that:
> **Joint matching within tight tolerances ($\pm 0.75\text{ dB}$, $\pm 0.035$ SSIM) was not achieved over the tested parameter grid and functional forms.**

It does **not** prove mathematical or physical impossibility across all possible parameter combinations, continuous optimization surfaces, or alternate corruption models (e.g., continuous JPEG approximations, bilateral filters, or generalized Gaussian noise). In this final package, we strictly frame this result as:
1. An empirical failure of the prespecified discrete calibration procedure to satisfy its target tolerances.
2. An illustrative demonstration of the structural trade-off between frequency-domain blur attenuation and block-quantization artifacts.

---

## 5. Explicit Limitations and Scope Boundaries

To ensure complete transparency for faculty reviewers, PhD admissions committees, and the broader community, the limitations of this benchmark are explicitly stated:

1. **Approximate Fidelity, Not Strict Matching:**  
   The calibrated study must be characterized as an *approximate-fidelity comparison*. Residual distortion differences remain across sequences (PSNR spread up to $3.10\text{ dB}$, SSIM spread up to $0.359$). Therefore, remaining performance differences cannot be attributed solely to architectural representation effects without acknowledging residual distortion imbalance.
2. **Dataset Scale and Diversity:**  
   Evaluations were conducted on the FastAI Imagenette-v2 10-synset benchmark ($N = 3{,}925$ validation images, $N = 9{,}468$ training images) rather than the full 1,000-class ImageNet-1k dataset. While Imagenette represents 10 natural synsets spanning coarse and fine visual features, results should be replicated on full ImageNet-1k before claiming universal benchmark dominance.
3. **Task Scope:**  
   The study evaluates image classification models exclusively. Dense spatial prediction tasks (object detection, semantic segmentation, depth estimation) and multimodal vision-language models were not evaluated.
4. **Synthetic vs. In-the-Wild Corruptions:**  
   Corruptions were synthesized using canonical mathematical operators (Gaussian noise, Gaussian blur, standard JPEG encoding). Real-world sensor artifacts (e.g., Bayer demosaicing, motion blur, rolling shutter, atmospheric turbulence) involve more complex physical dynamics.
5. **No Claims of Publication Acceptance or Global Superiority:**  
   This package represents an independent, rigorous research investigation prepared for graduate admissions and subsequent peer review. No claims of paper acceptance, conference presentation, or global novelty are asserted.
