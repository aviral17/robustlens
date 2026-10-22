# Order Has Consequences: Total Effects and Fidelity-Conditioned Prediction Instability under Compound Image Corruptions

**Author:** Avi  
**Affiliation:** Independent Research in Statistical Machine Learning & Computer Vision  
**Date:** October 2026  
**Document Status:** Unpublished Independent Research Working Paper / Manuscript Draft in Preparation  
**Target Purpose:** Independent Research Writing Sample for Fall 2027 U.S. PhD Applications  
**Codebase & Data Integrity:** Local repository verified; all findings traceable to raw prediction records (`results/order_effect_study/*/outputs/all_order_predictions.csv.gz`).  

---

## Abstract

Deep neural networks for computer vision are widely evaluated under synthetic distribution shifts, but standard benchmarks treat corruptions primarily as isolated unary operations (e.g., ImageNet-C) or along fixed, unidirectional pipelines (e.g., R-Bench). In real-world imaging pipelines, degradations occur sequentially through multi-stage physical and digital processes governed by non-commutative mathematical operators ($\mathcal{T}_A \circ \mathcal{T}_B \neq \mathcal{T}_B \circ \mathcal{T}_A$). In this work, we conduct a systematic paired evaluation of the complete symmetric group of three canonical image degradations—Gaussian Noise ($\mathcal{N}$), Defocus Blur ($\mathcal{B}$), and JPEG Compression ($\mathcal{J}$)—yielding all $3! = 6$ permutation sequences across five severity levels evaluated on four prominent vision architectures (ResNet-50, ConvNeXt-Tiny, ViT-Base/16, Swin-Tiny). Our study comprises $973{,}400$ total forward-pass evaluations across two distinct experimental regimes ($486{,}700$ records per regime on $3{,}925$ validation images from FastAI Imagenette-v2, paired with deterministically seed-matched noise realizations).

We formalize and analyze two distinct estimands:
1. **Total Operational Effect of Sequence Order under Fixed Nominal Parameters:** When operators run at standard ImageNet-C severity parameters, low-level physical fidelity varies systematically as a downstream consequence of operator non-commutativity (e.g., terminal blur attenuates prior high-frequency noise and JPEG block discontinuities, boosting PSNR by up to $6.2\text{ dB}$). Under this nominal regime, top-1 classification accuracy varies by up to $21.55\%$ (ResNet-50) and $20.45\%$ (ConvNeXt-Tiny) on identical physical images.
2. **Order Differences under Approximate Low-Level Fidelity:** To test whether sequence sensitivity persists when output distortion is constrained, we pre-calibrate operator parameters on an independent training partition ($N_{\text{cal}} = 50$) targeting common (PSNR, SSIM) support across sequences. Prespecified calibration tolerances ($\pm 0.75\text{ dB}$ PSNR, $\pm 0.035$ SSIM) were not achieved over the tested discrete parameter grid (PSNR RMSE = $0.943\text{ dB}$, SSIM RMSE = $0.1380$; validation cross-sequence PSNR spread reaches $3.10\text{ dB}$), establishing this regime strictly as an *approximate-fidelity descriptive comparison* rather than an exact causal effect at matched fidelity. Crucially, equalizing low-level distortion contracts cross-sequence performance gaps substantially—by $77.15\%$ in ConvNeXt, $72.62\%$ in ResNet, $50.75\%$ in ViT, and $31.80\%$ in Swin (the cross-sequence range of the overall model mean contracts by $63.18\%$, dropping from $18.43\%$ down to $6.79\%$, while the arithmetic average of the four individual architecture gap contractions is $58.08\%$).

Beyond aggregate accuracy, we demonstrate that standard benchmark metrics severely obscure model instability. We formulate a complete 5-state joint prediction space partition decomposing paired transitions into Both Correct ($E_{11}$), Loss ($E_{10}$), Gain ($E_{01}$), Same Incorrect Class ($E_{00}^{\text{same}}$), and Error Migration ($E_{00}^{\text{diff}}$, different incorrect classes). We prove that **Predicted-Label Churn** ($\mathbb{P}(\hat{Y}_A \neq \hat{Y}_B)$) is more than double **Correctness-State Churn** ($\mathbb{P}(C_A \neq C_B) = L + G$), because Error Migration accounts for $23.85\%$ of all evaluations in nominal pipelines and $22.40\%$ under approximate fidelity. Sequence pairs with near-identical aggregate accuracy ($|\Delta\text{Acc}| < 2.5\%$) exhibit up to $10.97\%$ excess churn and over $40\%$ label churn, demonstrating that sequence order reshuffles decision boundaries and error topologies across identical visual instances.

---

## 1. Introduction & Research Questions

Evaluating the robustness of computer vision architectures under distribution shift is essential for reliable deployment in safety-critical domains such as autonomous vehicles, medical diagnostics, and automated inspection. Since the introduction of ImageNet-C (Hendrycks & Dietterich, 2019), benchmarks have established that deep convolutional neural networks (CNNs) and Vision Transformers (ViTs) suffer dramatic performance degradation when subjected to high-frequency noise, spatial blurring, and lossy compression.

Despite widespread adoption, the standard benchmarking paradigm exhibits a critical structural limitation: **it treats corruptions as isolated, unary operations.** In physical imaging pipelines, degradations rarely occur in isolation. An autonomous driving camera captures an image subjected to atmospheric scattering (defocus blur), sensor read noise (Gaussian noise), and edge transmission encoding (JPEG compression). Crucially, the mathematical operators governing digital image formation are **fundamentally non-commutative**:
$$\mathcal{T}_A \circ \mathcal{T}_B \neq \mathcal{T}_B \circ \mathcal{T}_A$$
Spatial convolution is a linear low-pass filter, sensor noise is an additive stochastic field, and JPEG compression is a non-linear, grid-discontinuous block quantization process.

### 1.1 The Two Distinct Estimands
When evaluating sequence order in compound image corruptions, conflating experimental objectives leads to severe misinterpretation. In this study, we explicitly formalize and separate two distinct estimands:

1. **Estimand 1: Total Operational Effect at Fixed Parameters:**  
   What is the total impact of operator execution sequence on classification performance when degradations occur at fixed operational parameters? Under this estimand, low-level fidelity disparities (e.g., higher PSNR when blur follows JPEG) are not baseline confounders to be adjusted away; they are the direct, downstream physical consequence of operator non-commutativity. The measured accuracy gap reflects the total operational effect of sequence order in real-world processing pipelines.
2. **Estimand 2: Fidelity-Conditioned Descriptive Comparison:**  
   When output images are constrained to comparable low-level fidelity (approximate common support), to what extent do cross-sequence accuracy disparities persist? Because our discrete calibration procedure did not satisfy prespecified tolerances ($\pm 0.75\text{ dB}$ PSNR, $\pm 0.035$ SSIM), this estimand provides a *descriptive comparison under approximate common support*, rather than an exact causal effect at matched fidelity. Confounding was mitigated, but not eliminated.

### 1.2 Research Questions
We formulate three concrete research questions:
- **RQ1 (Total Operational Effect):** How severely does execution order alter classifier accuracy, confidence, and calibration when operators run at standard nominal parameter settings?
- **RQ2 (Fidelity-Conditioned Effect & Gap Contraction):** How much of the nominal sequence gap contracts when operator parameters are calibrated to target common PSNR/SSIM support, and do architecture-specific residual gaps persist?
- **RQ3 (Prediction Instability & Error Migration):** Does sequence order alter the specific failure sets and predicted class assignments for individual images, even when aggregate benchmark scores appear invariant?

### 1.3 Contributions of this Work
Through $973{,}400$ total model evaluations ($486{,}700$ records per regime) across four representative vision architectures—ResNet-50 (He et al., 2016), ConvNeXt-Tiny (Liu et al., 2022), ViT-Base/16 (Dosovitskiy et al., 2021), and Swin-Tiny (Liu et al., 2021)—we deliver the following contributions:

1. **Full Factorial $3! = 6$ Permutation Design with Seed-Matched Noise:** We evaluate all six permutations of Gaussian Noise, Defocus Blur, and JPEG Compression across five severity levels on $3{,}925$ validation images from FastAI Imagenette-v2. All comparisons are conducted on identical physical images with deterministically seed-matched noise realizations.
2. **Identification of the Terminal Smoothing Confound:** We demonstrate that in nominal evaluations, accuracy gaps of up to $21.55\%$ are primarily driven by terminal operator identity: sequences ending in blur preserve accuracy by attenuating prior high-frequency artifacts, whereas sequences ending in JPEG degrade accuracy by introducing boundary discontinuities.
3. **Approximate-Fidelity Calibration and Gap Contraction:** We formulate an optimization procedure on an independent training partition ($N_{\text{cal}} = 50$) targeting five pre-specified (PSNR, SSIM) common-support bins. We transparently document that prespecified tolerances ($\pm 0.75\text{ dB}$ PSNR, $\pm 0.035$ SSIM) were not achieved over the tested discrete grid. Nevertheless, equalizing low-level distortion contracts cross-sequence performance gaps by $31.8\%$ to $77.2\%$.
4. **Full-Rank Saturated Six-Cell Decomposition with Balanced Parity:** We formulate a full-rank linear decomposition partitioning sequence effects into intercept, entry (Start Blur, Start JPEG), exit (End Blur, End JPEG), and a balanced parity contrast between even and odd permutations. We report direct marginal contrasts alongside reference-coded parameters, showing that terminal operator effects invert upon fidelity calibration while balanced parity remains statistically significant ($p < 10^{-100}$).
5. **Decoupling of Predicted-Label Churn and Correctness Churn:** We formulate a complete 5-state joint prediction space partition ($E_{11}, E_{10}, E_{01}, E_{00}^{\text{same}}, E_{00}^{\text{diff}}$). We prove mathematically and empirically that Predicted-Label Churn ($40.78\%$ nominal, $34.74\%$ matched) is more than double Correctness-State Churn ($16.93\%$ nominal, $12.34\%$ matched) due to massive Error Migration ($E_{00}^{\text{diff}} = 23.85\%$ nominal, $22.40\%$ matched). We document extreme per-class vulnerability profiles across the 10 Imagenette classes.

---

## 2. Related Work & Novelty Demarcation

### 2.1 Benchmarking Vision Robustness
Synthetic robustness benchmarks trace back to ImageNet-C and ImageNet-P (Hendrycks & Dietterich, 2019). While ImageNet-C evaluated 15 distinct degradations, each was applied in isolation. ImageNet-P introduced perturbation sequences across video frames, but evaluated temporal trajectories rather than operator permutations on static images. Dodge & Karam (2016, 2019) and RichardWebster et al. (2018) pioneered psychometric and distortion matching in vision, but restricted their analyses to unary corruptions. Erkara & Mayer (2026) evaluated 37 neural network architectures across 15 common and structured image corruptions on ImageNet, proposing Accuracy Confidence Divergence to evaluate calibration under single perturbations, but did not evaluate compound sequential compositions or operator permutations.

Recent work has increasingly recognized compound corruptions. Kang et al. (2019) evaluated unforeseen adversaries and composite perturbations, but utilized fixed, unpermuted pipelines. Mintun et al. (2021) investigated interaction effects between data augmentations and corruptions in natural corruption robustness under fixed sequences. Goyal et al. (2022) studied dual corruptions (noise and motion blur in low-light imaging) on vision models, observing an asymmetry between Noise $\to$ Blur and Blur $\to$ Noise; however, they evaluated only dual-operator combinations without calibrating distortion parameters to match physical fidelity across the full permutation group. Li et al. (2025) introduced R-Bench, modeling a 7-stage physical imaging pipeline for Multimodal LLMs, but followed a single unidirectional acquisition sequence without permuting operator order. Liao et al. (2020) demonstrated in digital forensics that the sequential execution order of image processing operators leaves distinct, detectable forensic traces in deep feature representations, confirming the non-commutative physical reality of operator sequences. Recently, Wei et al. (2026) demonstrated the utility of paired single-image evaluations under visual perturbations, establishing classwise correctness bounds and prioritizing failure-prone instances for physical capture testing. However, their analysis evaluates unary synthetic corruptions or real camera recapture without permuting sequential multi-stage operators, without joint PSNR/SSIM common-support pre-calibration across permutations, and without a six-cell entry/exit/parity decomposition.

### 2.2 Prediction Churn and Model Instability
Prediction churn—the rate at which predictions differ between model iterations or pipeline versions—was formalized by Milani Fard, Cormier, Canini, & Gupta (2016) ("Launch and Iterate: Reducing Prediction Churn") in the context of model retraining. D'Amour et al. (2020) demonstrated that underspecification in deep networks leads to models with identical in-distribution accuracy that diverge sharply under distribution shift. In this work, we adapt the concept of prediction churn to **counterfactual input permutations** under identical, frozen model weights, and resolve the mathematical distinction between predicted-label churn and correctness-state churn.

### 2.3 Explicit Demarcation of Prior Art vs. Novelty
To adhere to rigorous academic integrity standards, we explicitly delineate prior art from our novel contributions:
- **Prior Art:** Operator non-commutativity is a classical digital signal processing property. Detecting operator chain order in image forensics (Liao et al., 2020), defining prediction churn for model iterations (Milani Fard et al., 2016), ImageNet-C corruptions (Hendrycks & Dietterich, 2019), dual-corruption asymmetry (Goyal et al., 2022), and unary black-box confusion metrics (Erkara & Mayer, 2026) are established in prior literature.
- **Novelty of RobustLens:** We present the first full factorial $3! = 6$ permutation study across four modern vision architectures evaluated on identical physical images with seed-matched noise, the complete 5-state joint prediction space decomposition separating predicted-label churn from correctness churn, a full-rank saturated six-cell decomposition with direct marginal contrasts and balanced parity, and an empirical common-support calibration study that transparently documents tolerance diagnostics and gap contraction.

---

## 3. Mathematical Framework & Statistical Methodology

### 3.1 Operator Definitions and Sequence Non-Commutativity
Let $x \in [0, 1]^{3 \times H \times W}$ denote a clean RGB image. We consider three degradation operators:
1. **Gaussian Noise ($\mathcal{N}_{\sigma}$):** $x \mapsto \text{clip}(x + \epsilon, 0, 1)$, where $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$.
2. **Defocus Blur ($\mathcal{B}_{\kappa, \gamma}$):** Spatial 2D Gaussian convolution with kernel size $\kappa$ and bandwidth $\gamma$.
3. **JPEG Compression ($\mathcal{J}_q$):** $8 \times 8$ block discrete cosine transform (BDCT) quantization with quality factor $q \in [1, 100]$.

The full symmetric group $\mathfrak{S}_3$ consists of 6 permutation sequences applied strictly from left to right:
$$\mathcal{S} = \{\text{NBJ}, \text{NJB}, \text{BNJ}, \text{BJN}, \text{JNB}, \text{JBN}\}$$
- `NBJ`: $\mathcal{J} \circ \mathcal{B} \circ \mathcal{N}$ (Noise $\to$ Blur $\to$ JPEG; **ends in JPEG**)
- `NJB`: $\mathcal{B} \circ \mathcal{J} \circ \mathcal{N}$ (Noise $\to$ JPEG $\to$ Blur; **ends in Blur**)
- `BNJ`: $\mathcal{J} \circ \mathcal{N} \circ \mathcal{B}$ (Blur $\to$ Noise $\to$ JPEG; **ends in JPEG**)
- `BJN`: $\mathcal{N} \circ \mathcal{J} \circ \mathcal{B}$ (Blur $\to$ JPEG $\to$ Noise; **ends in Noise**)
- `JNB`: $\mathcal{B} \circ \mathcal{N} \circ \mathcal{J}$ (JPEG $\to$ Noise $\to$ Blur; **ends in Blur**)
- `JBN`: $\mathcal{N} \circ \mathcal{B} \circ \mathcal{J}$ (JPEG $\to$ Blur $\to$ Noise; **ends in Noise**)

### 3.2 Full 5-State Joint Prediction Space Decomposition
Let $\hat{Y}_A(i), \hat{Y}_B(i) \in \{1, \dots, K\}$ denote the model's predicted class, and let $Y(i) \in \{1, \dots, K\}$ denote the ground-truth class label for image $i$. Binary correctness indicators are $C_A(i) = \mathbb{I}(\hat{Y}_A(i) = Y(i))$ and $C_B(i) = \mathbb{I}(\hat{Y}_B(i) = Y(i))$.

The joint prediction space partitions into five mutually exclusive, exhaustive transitions:
1. **$E_{11}$ (Both Correct):** $\hat{Y}_A = Y$ and $\hat{Y}_B = Y$.
2. **$E_{10}$ (Loss / $L$):** $\hat{Y}_A = Y$ and $\hat{Y}_B \neq Y$.
3. **$E_{01}$ (Gain / $G$):** $\hat{Y}_A \neq Y$ and $\hat{Y}_B = Y$.
4. **$E_{00}^{\text{same}}$ (Same Wrong Label):** $\hat{Y}_A = \hat{Y}_B \neq Y$.
5. **$E_{00}^{\text{diff}}$ (Error Migration / Different Wrong Labels):** $\hat{Y}_A \neq Y$, $\hat{Y}_B \neq Y$, and $\hat{Y}_A \neq \hat{Y}_B$.

#### Conservation of Probability and Churn Identities
These five states partition the sample space completely:
$$\mathbb{P}(E_{11}) + \mathbb{P}(E_{10}) + \mathbb{P}(E_{01}) + \mathbb{P}(E_{00}^{\text{same}}) + \mathbb{P}(E_{00}^{\text{diff}}) = 1$$

We can now define the two distinct notions of churn:
- **Correctness-State Churn:** The probability that correctness flips:
  $$\text{Correctness Churn} = \mathbb{P}(C_A \neq C_B) = \mathbb{P}(E_{10}) + \mathbb{P}(E_{01}) = L + G$$
- **Predicted-Label Churn:** The probability that the predicted class changes:
  $$\text{Label Churn} = \mathbb{P}(\hat{Y}_A \neq \hat{Y}_B) = \mathbb{P}(E_{10}) + \mathbb{P}(E_{01}) + \mathbb{P}(E_{00}^{\text{diff}}) = \text{Correctness Churn} + \mathbb{P}(E_{00}^{\text{diff}})$$
- **Prediction Agreement Rate:**
  $$\text{Agreement Rate} = \mathbb{P}(\hat{Y}_A = \hat{Y}_B) = \mathbb{P}(E_{11}) + \mathbb{P}(E_{00}^{\text{same}}) = 1 - \text{Label Churn}$$
- **Net Accuracy Difference:**
  $$\Delta\text{Acc}(A \to B) = \mathbb{E}[C_B] - \mathbb{E}[C_A] = G - L$$
- **Excess Correctness Churn:**
  $$\text{Excess Correctness Churn} = \text{Correctness Churn} - |\Delta\text{Acc}| = 2 \cdot \min(G, L)$$
- **Excess Label Churn:**
  $$\text{Excess Label Churn} = \text{Label Churn} - |\Delta\text{Acc}| = 2 \cdot \min(G, L) + \mathbb{P}(E_{00}^{\text{diff}})$$

**Key Theoretical Consequence:** When models make distinct incorrect predictions ($\mathbb{P}(E_{00}^{\text{diff}}) > 0$), Correctness Churn strictly understates classifier instability. In our experiments, Error Migration is so large ($\sim 23\%$) that Predicted-Label Churn is more than double Correctness Churn.

We also measure decision confidence shifts and Brier scores:
$$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^N (\text{conf}_i - C_i)^2, \qquad \Delta\text{Conf} = \frac{1}{N}\sum_{i=1}^N |\text{conf}_A(i) - \text{conf}_B(i)|$$

### 3.3 Full-Rank Saturated Six-Cell Decomposition with Balanced Parity
To rigorously decompose sequence variance across all six cell means $\mu = [\mu_{\text{NBJ}}, \mu_{\text{NJB}}, \mu_{\text{BNJ}}, \mu_{\text{BJN}}, \mu_{\text{JNB}}, \mu_{\text{JBN}}]^T$, we specify the linear model $\mu = X \beta$.
Choosing reference levels $\text{Start} = \text{Noise}$ and $\text{End} = \text{Noise}$, the design matrix $X$ is:
$$X = \begin{bmatrix}
1 & 0 & 0 & 0 & 1 & +1 \\
1 & 0 & 0 & 1 & 0 & -1 \\
1 & 1 & 0 & 0 & 1 & -1 \\
1 & 1 & 0 & 0 & 0 & +1 \\
1 & 0 & 1 & 1 & 0 & +1 \\
1 & 0 & 1 & 0 & 0 & -1
\end{bmatrix}$$
Because $\det(X) = -6 \neq 0$, $X$ has full rank ($\text{rank}(X) = 6$), and $\beta = X^{-1} \mu$ is uniquely determined.

#### Clarification on Reference Level and Orthogonality
1. **Unobserved Reference Level:** In reference coding, the baseline corresponds to Start Noise and End Noise. However, a 3-operator sequence without repetition cannot both start and end with Noise. Thus, the reference cell is unobserved, and the Intercept $\beta_0$ represents an extrapolated linear parameter rather than an observable baseline condition.
2. **Non-Orthogonal Design:** The columns of $X$ are not mutually orthogonal ($X^T X \neq I$). Consequently, reference-coded coefficients must be interpreted alongside direct marginal means and contrasts.
3. **Balanced Parity Contrast:** The parity coefficient $\beta_5$ is derived from row 6 of $X^{-1}$:
   $$\beta_5 = \frac{1}{6} \left[ (\mu_{\text{NBJ}} + \mu_{\text{BJN}} + \mu_{\text{JNB}}) - (\mu_{\text{NJB}} + \mu_{\text{BNJ}} + \mu_{\text{JBN}}) \right] = \frac{1}{2} (\bar{\mu}_{\text{Even}} - \bar{\mu}_{\text{Odd}})$$
   In Group 1 (Even, $A_3$: `NBJ`, `BJN`, `JNB`), initial operators are $\{\text{N}, \text{B}, \text{J}\}$ and terminal operators are $\{\text{J}, \text{N}, \text{B}\}$. In Group 2 (Odd: `NJB`, `BNJ`, `JBN`), initial operators are $\{\text{N}, \text{B}, \text{J}\}$ and terminal operators are $\{\text{B}, \text{J}, \text{N}\}$. Every degradation operator appears exactly once at entry and exit in both groups. Thus, the parity contrast is perfectly balanced against marginal entry and exit distributions.

### 3.4 Clustered Bootstrap and Multiplicity Control
Because all conditions for a given image share the same underlying visual content, observations within an image are correlated. We perform **clustered bootstrap resampling** with $B = 5{,}000$ iterations clustered on physical image IDs ($N_{\text{clusters}} = 3{,}925$).
- **Primary Family (Top-1 Accuracy Decomposition & Contrasts):** Adjusted using the step-down Holm-Bonferroni procedure in log space to guarantee strong Family-Wise Error Rate (FWER) control at $\alpha = 0.05$.
- **Secondary Family (Confidence, PSNR, SSIM, Brier, Churn):** Adjusted using the Benjamini-Hochberg procedure in log space for False Discovery Rate (FDR) control at $q = 0.05$. Asymptotic two-sided Wald p-values are computed as $\log p = \log(2) + \text{logsf}(|z|)$ using the standard normal survival function with cluster-bootstrap standard errors to eliminate truncation to literal `0.0`.

---

## 4. Regime 1: Nominal-Order Benchmark (Estimand 1)

In the nominal-order regime, identical ImageNet-C parameters are applied across all six sequences across five severities ($s \in \{1, \dots, 5\}$).

### Table 1: Nominal-Order Model Accuracy, Brier Scores, and Sequence Disparities
Evaluated across $3{,}925$ validation images ($19{,}625$ evaluations per sequence per model; total $471{,}000$ compound records). Top-1 Accuracy shown as %, Brier score in parentheses.

| Model | Architecture | `NBJ` | `NJB` | `BNJ` | `BJN` | `JNB` | `JBN` | Mean Acc | Min Acc | Max Acc | Acc Gap ($\Delta$) | Mean Brier |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50** | CNN | 53.44% | 56.88% | 37.30% | 35.71% | **57.27%** | 35.79% | 46.06% | 35.71% | 57.27% | **21.55%** | 0.1587 |
| **ConvNeXt-Tiny** | Modern CNN | 63.98% | **66.44%** | 48.63% | 45.99% | 65.44% | 46.33% | 56.14% | 45.99% | 66.44% | **20.45%** | 0.1265 |
| **ViT-Base/16** | Transformer | 58.39% | **60.14%** | 39.77% | 41.64% | 59.19% | 42.23% | 50.23% | 39.77% | 60.14% | **20.37%** | 0.1481 |
| **Swin-Tiny** | Hierarchical ViT | 59.81% | 61.78% | 45.81% | 50.24% | **62.94%** | 50.63% | 55.20% | 45.81% | 62.94% | **17.13%** | 0.1286 |
| **Overall Mean** | — | 58.90% | **61.31%** | 42.88% | 43.40% | 61.21% | 43.75% | 51.91% | 42.88% | 61.31% | **18.43%** | 0.1405 |

### Direct Marginal Contrasts (Nominal Overall):
- **Marginal Start Means:** Start Noise = $60.11\%$, Start Blur = $43.14\%$, Start JPEG = $52.48\%$.
  - Start Blur vs. Noise: **$-16.97\%$** ($p < 10^{-100}$)
  - Start JPEG vs. Noise: **$-7.63\%$** ($p < 10^{-100}$)
- **Marginal End Means:** End Noise = $43.57\%$, End Blur = $61.26\%$, End JPEG = $50.89\%$.
  - End Blur vs. Noise: **$+17.69\%$** ($p < 10^{-100}$)
  - End JPEG vs. Noise: **$+7.32\%$** ($p < 10^{-100}$)
- **Balanced Parity:** Even Mean = $54.50\%$, Odd Mean = $49.31\%$. Parity Contrast = **$+5.19\%$** ($2 \beta_5 = +5.19\%$, $p < 10^{-100}$).

### Empirical Finding:
In nominal evaluations, the massive $18.43\%$ sequence gap is dominated by terminal operator identity: ending in Blur confers a $+17.69\%$ marginal boost over terminal noise because spatial convolution attenuates prior high-frequency noise and DCT block artifacts, boosting PSNR by $\sim 6.2\text{ dB}$ (mean PSNR $24.6\text{ dB}$ for terminal blur vs. $18.3\text{ dB}$ for terminal noise).

---

## 5. Regime 2: Approximate-Fidelity Descriptive Comparison (Estimand 2)

To assess order differences under constrained fidelity, parameters were calibrated on $N_{\text{cal}} = 50$ training images across 5 pre-specified common-support targets.

### Table 2: Measured Calibration Diagnostics and Stated Tolerance Violations
Evaluated on the $N_{\text{cal}} = 50$ training calibration partition across all 30 conditions (6 sequences $\times$ 5 targets).

| Target | Target PSNR (dB) | Target SSIM | Max Absolute PSNR Residual (dB) | PSNR Tol ($\le 0.75\text{ dB}$) | Max Absolute SSIM Residual | SSIM Tol ($\le 0.035$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Target 1** | 28.0 | 0.850 | 1.117 (`NJB`) | **Failed** | 0.0686 (`BJN`) | **Failed** |
| **Target 2** | 24.5 | 0.740 | 0.936 (`BJN`) | **Failed** | 0.1758 (`BJN`) | **Failed** |
| **Target 3** | 21.5 | 0.620 | 1.260 (`NBJ`) | **Failed** | 0.2379 (`BJN`) | **Failed** |
| **Target 4** | 18.5 | 0.480 | 1.989 (`NBJ`) | **Failed** | 0.2373 (`BJN`) | **Failed** |
| **Target 5** | 16.0 | 0.360 | 2.639 (`NBJ`) | **Failed** | 0.2067 (`BJN`) | **Failed** |
| **Overall** | — | — | **2.639 dB (Max)** | **RMSE: 0.943 dB (Failed)** | **0.2379 (Max)** | **RMSE: 0.1380 (Failed)** |

On the $3{,}925$ validation images, cross-sequence PSNR spread reached $3.10\text{ dB}$ at Target 5, and SSIM spread reached $0.359$.

> [!WARNING]
> **Methodological Boundary:** Prespecified calibration tolerances were not achieved over the tested grid. This failure stems from the functional trade-off between smoothing and edge preservation in discrete parameter spaces. It does not prove universal physical impossibility across continuous spaces, but it confirms that this regime is strictly an **approximate-fidelity comparison**. Confounding was reduced, not eliminated.

### Table 3: Accuracy, Brier Scores, and Gap Contraction under Approximate Fidelity
Evaluated across $3{,}925$ validation images ($471{,}000$ compound records). Top-1 Accuracy shown as %, Brier score in parentheses.

| Model | `NBJ` | `NJB` | `BNJ` | `BJN` | `JNB` | `JBN` | Matched Mean | Matched Gap | Nominal Gap | Contraction | Mean Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50** | 50.15% | 46.76% | 45.18% | 50.63% | 47.05% | **51.08%** | 48.47% | **5.90%** | 21.55% | **72.62%** | 0.1558 |
| **ConvNeXt-Tiny** | **60.65%** | 56.46% | 55.97% | 59.22% | 56.47% | 59.86% | 58.11% | **4.67%** | 20.45% | **77.15%** | 0.1245 |
| **ViT-Base/16** | **56.54%** | 53.39% | 46.51% | 54.72% | 53.01% | 55.68% | 53.31% | **10.03%** | 20.37% | **50.75%** | 0.1444 |
| **Swin-Tiny** | 57.61% | 50.44% | 53.93% | 61.23% | 52.31% | **62.12%** | 56.27% | **11.68%** | 17.13% | **31.80%** | 0.1433 |
| **Overall Mean** | 56.23% | 51.76% | 50.40% | 56.45% | 52.21% | **57.18%** | 54.04% | **6.79%** | 18.43% | **63.18%** | 0.1420 |

### Direct Marginal Contrasts (Approximate Fidelity Overall):
- **Marginal Start Means:** Start Noise = $54.00\%$, Start Blur = $53.42\%$, Start JPEG = $54.70\%$.
  - Start Blur vs. Noise: **$-0.58\%$** ($p = 0.089$)
  - Start JPEG vs. Noise: **$+0.70\%$** ($p = 0.041$)
- **Marginal End Means:** End Noise = $56.82\%$, End Blur = $51.99\%$, End JPEG = $53.32\%$.
  - End Blur vs. Noise: **$-4.83\%$** ($p < 10^{-100}$)
  - End JPEG vs. Noise: **$-3.50\%$** ($p < 10^{-100}$)
- **Balanced Parity:** Even Mean = $54.97\%$, Odd Mean = $53.11\%$. Parity Contrast = **$+1.85\%$** ($2 \beta_5 = +1.85\%$, $p < 10^{-100}$).

### Empirical Insights:
1. **Dramatic Gap Contraction:** Equalizing low-level distortion contracts the cross-sequence range of the overall model mean from $18.43\%$ down to $6.79\%$ (a $63.18\%$ reduction in the overall mean range; the arithmetic average of the four individual architecture gap contractions is $58.08\%$, contracting by $77.15\%$ in ConvNeXt, $72.62\%$ in ResNet, $50.75\%$ in ViT, and $31.80\%$ in Swin).
2. **Reversal of the End Operator Effect:** Under approximate fidelity, End Blur flips from $+17.69\%$ to **$-4.83\%$**. When low-level PSNR is balanced, terminal blur removes discriminative high-frequency edges without conferring an artifact-smoothing advantage.
3. **Architectural Differentiation:** ConvNeXt and ResNet contract by $>70\%$, whereas Swin-Tiny retains an $11.68\%$ residual gap (contracting by only $31.80\%$), showing heightened sensitivity to multi-stage spatial transformations.

---

## 6. Saturated Linear Decomposition Results

Table 4 compares the fitted parameters across regimes ($B = 5{,}000$ bootstrap resamples, Holm FWER-adjusted).

### Table 4: Full-Rank Saturated Six-Cell Decomposition Coefficients (Overall Model Mean)
Reference levels: Start Noise, End Noise. Point estimates and CIs reported in percentage points of Top-1 Accuracy.

| Parameter | Nominal Estimate (%) | Nominal 95% Clustered CI | Nominal $p_{\text{Holm}}$ | Matched Estimate (%) | Matched 95% Clustered CI | Matched $p_{\text{Holm}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\beta_0$ (Intercept)** | +51.63 | [+50.71, +52.52] | $< 10^{-100}$ | +59.51 | [+58.51, +60.51] | $< 10^{-100}$ |
| **$\beta_1$ (Start Blur)** | -10.83 | [-11.14, -10.53] | $< 10^{-100}$ | -3.99 | [-4.22, -3.75] | $< 10^{-100}$ |
| **$\beta_2$ (Start JPEG)** | -5.29 | [-5.50, -5.08] | $< 10^{-100}$ | -1.40 | [-1.58, -1.23] | $< 10^{-50}$ |
| **$\beta_3$ (End Blur)** | **+12.27** | [+11.88, +12.67] | $< 10^{-100}$ | **-6.82** | [-7.18, -6.46] | $< 10^{-100}$ |
| **$\beta_4$ (End JPEG)** | **+4.67** | [+4.35, +5.00] | $< 10^{-100}$ | **-4.20** | [-4.50, -3.91] | $< 10^{-100}$ |
| **$\beta_5$ (Parity Contrast)**| **+2.60** | [+2.52, +2.67] | $< 10^{-100}$ | **+0.93** | [+0.86, +0.99] | $< 10^{-100}$ |

Even after controlling for entry and exit operators, the balanced parity contrast $\beta_5$ remains highly statistically significant in both regimes ($+2.60\%$ nominal, $+0.93\%$ matched, $p < 10^{-100}$).

---

## 7. Prediction Instability, Error Migration, and Per-Class Breakdown

A cornerstone finding of RobustLens is that **aggregate accuracy obscures massive per-image prediction churn and error migration.**

### Table 5: Complete 5-State Prediction Space Transitions by Model and Overall
Averaged across all 15 pairwise order combinations per model ($N = 19{,}625$ image-severity pairs per comparison). All transition rates in %.

| Regime | Model | Both Correct ($E_{11}$) | Loss ($E_{10}$) | Gain ($E_{01}$) | Same Wrong ($E_{00}^{\text{same}}$) | Error Migration ($E_{00}^{\text{diff}}$) | Correctness Churn | Excess Corr Churn | **Label Churn** | Agreement Rate | Mean $|\Delta\text{Conf}|$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal** | ConvNeXt | 47.63 | 11.64 | 5.38 | 14.99 | **20.36** | 17.02 | 5.36 | **37.38** | 62.62 | 0.1662 |
| **Nominal** | ResNet-50 | 36.96 | 12.06 | 6.15 | 14.90 | **29.93** | 18.20 | 5.72 | **48.14** | 51.86 | 0.1850 |
| **Nominal** | Swin-Tiny | 47.98 | 8.49 | 5.96 | 16.06 | **21.51** | 14.45 | 5.82 | **35.96** | 64.04 | 0.1835 |
| **Nominal** | ViT-Base | 41.20 | 11.76 | 6.31 | 17.17 | **23.57** | 18.06 | 6.68 | **41.63** | 58.37 | 0.1732 |
| **Nominal** | **Overall** | **43.44** | **10.99** | **5.95** | **15.78** | **23.85** | **16.93** | **5.90** | **40.78** | **59.22** | **0.1770** |
| **Matched** | ConvNeXt | 52.64 | 5.49 | 5.44 | 17.44 | **19.00** | 10.93 | 8.51 | **29.92** | 70.08 | 0.1163 |
| **Matched** | ResNet-50 | 41.92 | 6.18 | 6.92 | 15.54 | **29.44** | 13.10 | 10.15 | **42.54** | 57.46 | 0.1481 |
| **Matched** | Swin-Tiny | 50.31 | 4.78 | 7.14 | 18.03 | **19.74** | 11.92 | 5.99 | **31.66** | 68.34 | 0.1307 |
| **Matched** | ViT-Base | 46.61 | 6.61 | 6.79 | 18.55 | **21.44** | 13.40 | 9.44 | **34.85** | 65.15 | 0.1314 |
| **Matched** | **Overall** | **47.87** | **5.76** | **6.57** | **17.39** | **22.40** | **12.34** | **8.52** | **34.74** | **65.26** | **0.1316** |

### Key Findings from Prediction Space Transitions:
1. **Label Churn is More Than Double Correctness Churn:** Across all models, Error Migration ($E_{00}^{\text{diff}}$) averages $23.85\%$ in nominal and $22.40\%$ in matched. Because the model frequently jumps between different wrong classes across orders, measuring only correctness churn ($16.93\%$ nominal, $12.34\%$ matched) ignores over $60\%$ of total prediction instability!
2. **ResNet-50 Exhibits Highest Instability:** ResNet-50 has the highest Error Migration ($29.93\%$ nominal, $29.44\%$ matched) and highest Predicted-Label Churn ($48.14\%$ nominal, $42.54\%$ matched). Nearly half of all predictions flip class under sequence order changes.
3. **Decoupling of Aggregate Accuracy and Error Churn:** In condition pairs with near-zero accuracy difference (e.g., ConvNeXt matched `NJB` vs. `JNB`, where $|\Delta\text{Acc}| = 0.01\%$), $8.23\%$ of predictions flip ($4.12\%$ gain, $4.11\%$ loss), generating $8.22\%$ excess churn ($1{,}613$ physical images).

### Table 6: Per-Class Vulnerability Breakdown Across 10 Imagenette Classes
Ranked from highest to lowest accuracy under Nominal-Order compound corruptions.

| Class Name | Nom Acc (%) | Nom Conf (%) | Nom Brier | Nom Label Churn (%) | Nom Corr Churn (%) | Nom Error Mig (%) | Matched Acc (%) | Matched Label Churn (%) | Matched Error Mig (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **tench** | 73.57 | 66.25 | 0.0921 | 29.09 | 17.56 | 11.54 | 77.22 | 21.54 | 11.91 |
| **golf ball** | 72.45 | 66.65 | 0.1088 | 28.86 | 17.89 | 10.97 | 75.07 | 21.88 | 10.83 |
| **parachute** | 68.01 | 61.06 | 0.1224 | 31.19 | 16.22 | 14.97 | 68.97 | 28.00 | 15.75 |
| **chain saw** | 55.56 | 51.44 | 0.1170 | 41.93 | 17.42 | 24.51 | 59.32 | 34.59 | 22.64 |
| **French horn** | 51.98 | 52.58 | 0.1053 | 41.77 | 14.21 | 27.56 | 53.81 | 36.00 | 27.08 |
| **garbage truck**| 49.61 | 50.66 | 0.1528 | 43.23 | 19.56 | 23.67 | 51.25 | 36.10 | 21.24 |
| **gas pump** | 48.57 | 48.14 | 0.1253 | 46.52 | 17.51 | 29.01 | 50.56 | 40.01 | 28.27 |
| **cassette player**| 41.35 | 42.38 | 0.1988 | 45.62 | 20.64 | 24.98 | 44.40 | 39.95 | 22.40 |
| **English springer**| 40.48 | 40.22 | 0.1786 | 48.72 | 17.85 | 30.87 | 41.61 | 42.85 | 28.25 |
| **church** | **18.35** | 38.21 | 0.2065 | **50.45** | **11.04** | **39.41** | **19.22** | **46.07** | **34.70** |

### Per-Class Insights:
For 'church', top-1 accuracy collapses to $18.35\%$, and correctness churn is only $11.04\%$. If an evaluation only inspected correctness churn, it would report 'church' as the *most stable* class! However, Error Migration is an astounding **$39.41\%$**, pushing Predicted-Label Churn to **$50.45\%$**. Under sequence permutations, the model constantly switches among different incorrect class predictions.

---

## 8. Calibration & Uncertainty Degradation

We evaluated Expected Calibration Error (ECE, 10 equal-width bins) and confidence distributions across all conditions directly from raw predictions and logits:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
1. **Overconfidence under Severe Shift:** Across all models, mean confidence degrades substantially more slowly than accuracy. Under Target 5, ResNet-50 achieves $31.2\%$ accuracy while maintaining $54.8\%$ mean confidence, resulting in severe miscalibration ($\text{ECE} > 22\%$).
2. **Terminal Noise Maximizes Miscalibration:** In nominal evaluations, sequences ending in Gaussian Noise (`BJN`, `JBN`) produce the highest calibration error (up to $18.58\%$ in Swin-Tiny). High-frequency noise perturbs localized activation maps while preserving softmax peakiness. Under approximate fidelity, calibration curves compress, but Swin-Tiny retains higher calibration volatility across sequences than ConvNeXt-Tiny.

---

## 9. Comprehensive Limitations & Scope Boundaries

In accordance with academic rigor, we explicitly state the limitations of this benchmark:

1. **Approximate Fidelity, Not Exact Matching:**  
   Prespecified tolerances ($\pm 0.75\text{ dB}$, $\pm 0.035$ SSIM) were not achieved over the tested discrete grid. Residual PSNR spread reaches $3.10\text{ dB}$ at Target 5. Therefore, residual performance differences cannot be interpreted as pure architectural effects without acknowledging residual distortion imbalance.
2. **Finite Grid Search Boundary:**  
   The failure of our discrete parameter search does **not** prove universal mathematical or physical impossibility across continuous parameter spaces or alternate corruption formulations.
3. **Dataset Scope:**  
   Evaluations were conducted on FastAI Imagenette-v2 (10 classes, $3{,}925$ validation images). Full ImageNet-1k evaluations remain necessary before generalizing benchmark conclusions.
4. **Task Domain:**  
   This study focuses exclusively on image classification. Dense prediction tasks (object detection, segmentation) and vision-language models were not evaluated.
5. **Absence of External ImageNet-V2 Validation:**  
   ImageNet-V2 was not pre-cached locally. Re-running a full 486,700-inference GPU evaluation was excluded to maintain bounded execution and data integrity.

---

## 10. Conclusion

By evaluating all six permutations of three canonical digital corruptions across both nominal and approximate-fidelity regimes, RobustLens reveals that standard robustness benchmarks suffer from a profound physical confound: **nominal sequence gaps (up to $21.55\%$) are primarily driven by terminal low-pass filter smoothing.**

When low-level distortion is pre-calibrated to common support, the cross-sequence range of the overall model mean contracts by $63.18\%$ (with individual architecture gap contractions averaging $58.08\%$ and reaching up to $77.15\%$ in ConvNeXt). However, paired single-image analysis proves that sequence order continues to matter: distinct sequences exhibit up to $10.97\%$ excess prediction churn despite near-identical aggregate accuracy. Furthermore, we establish the fundamental distinction between Predicted-Label Churn and Correctness-State Churn, proving that Error Migration ($E_{00}^{\text{diff}} \approx 23\%$) causes label churn to exceed $34\text{--}40\%$. Vision architectures exhibit non-commutative input-output response functions that alter single-image failure modes under multi-stage degradation pipelines, shifting specific misclassified instances even when aggregate accuracy is invariant. Future robustness benchmarks must account for both operator permutation order and physical distortion balancing.

---

## References

1. **Bhojanapalli, S., Chakrabarti, A., Glasner, D., Li, D., Unterthiner, T., & Veit, A.** (2021). Understanding robustness of transformers for image classification. *IEEE/CVF International Conference on Computer Vision (ICCV)*, 10231–10241. DOI: [10.1109/ICCV48922.2021.01006](https://doi.org/10.1109/ICCV48922.2021.01006).
2. **D'Amour, A., Heller, K., Moldovan, D., Adlam, B., Alipanahi, B., Beutel, A., ... & Sculley, D.** (2020). Underspecification presents challenges for credibility in modern machine learning. *Journal of Machine Learning Research (JMLR)*, 23(221), 1–61.
3. **Dodge, S., & Karam, L.** (2016). Understanding how image quality affects deep neural networks. *Eighth International Conference on Quality of Multimedia Experience (QoMEX)*, 1–6. DOI: [10.1109/QoMEX.2016.7498955](https://doi.org/10.1109/QoMEX.2016.7498955).
4. **Dodge, S., & Karam, L.** (2019). Human and DNN classification performance on images with quality distortions: A comparative study. *ACM Transactions on Applied Perception (TAP)*, 16(2), Article 7, 1–17. DOI: [10.1145/3306241](https://doi.org/10.1145/3306241).
5. **Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N.** (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *International Conference on Learning Representations (ICLR)*.
6. **Erkara, A. O., & Mayer, M.** (2026). Blind confusion of classification networks: A black box evaluation under common and structured image corruptions. *Neurocomputing*, 686, 133678. DOI: [10.1016/j.neucom.2025.133678](https://doi.org/10.1016/j.neucom.2025.133678).
7. **Goyal, B., Lalonde, J.-F., Li, Y., & Gupta, M.** (2022). Robust scene inference under noise-blur dual corruptions. *IEEE International Conference on Computational Photography (ICCP)*, 1–12. DOI: [10.1109/ICCP54855.2022.9887772](https://doi.org/10.1109/ICCP54855.2022.9887772). arXiv: [2207.11643](https://arxiv.org/abs/2207.11643).
8. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). Deep residual learning for image recognition. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778. DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90).
9. **Hendrycks, D., & Dietterich, T.** (2019). Benchmarking neural network robustness to common corruptions and perturbations. *International Conference on Learning Representations (ICLR)*.
10. **Kang, D., Sun, Y., Hendrycks, D., Brown, T., & Steinhardt, J.** (2019). Testing robustness against unforeseen adversaries. *arXiv preprint arXiv:1908.08016*.
11. **Li, Y., et al.** (2025). R-Bench: Benchmarking multi-stage physical degradation in multimodal large language models. *IEEE Journal of Selected Topics in Signal Processing*, DOI: [10.1109/JSTSP.2025.3528442](https://doi.org/10.1109/JSTSP.2025.3528442). arXiv: [2406.14088](https://arxiv.org/abs/2406.14088).
12. **Liao, X., Li, K., Zhu, X., & Liu, K. J. R.** (2020). Robust detection of image operator chain with two-stream convolutional neural network. *IEEE Journal of Selected Topics in Signal Processing*, 14(5), 955–968. DOI: [10.1109/JSTSP.2020.3002391](https://doi.org/10.1109/JSTSP.2020.3002391).
13. **Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B.** (2021). Swin transformer: Hierarchical vision transformer using shifted windows. *IEEE/CVF International Conference on Computer Vision (ICCV)*, 10012–10022. DOI: [10.1109/ICCV48922.2021.00986](https://doi.org/10.1109/ICCV48922.2021.00986).
14. **Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S.** (2022). A convnet for the 2020s. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 11976–11986. DOI: [10.1109/CVPR52688.2022.01167](https://doi.org/10.1109/CVPR52688.2022.01167).
15. **Milani Fard, M., Cormier, Q., Canini, K., & Gupta, M. R.** (2016). Launch and iterate: Reducing prediction churn. *Advances in Neural Information Processing Systems (NeurIPS)*, 29, 3179–3187.
16. **Mintun, E., Kirillov, A., Gunasekar, S., Mu, S., & Xie, S.** (2021). On interaction between augmentations and corruptions in natural corruption robustness. *Advances in Neural Information Processing Systems (NeurIPS)*, 34, 3571–3583.
17. **RichardWebster, B., Anthony, S. E., & Scheirer, W. J.** (2018). Visual psychophysics for deep neural networks. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2800–2809.
18. **Robey, A., Chamon, L. F., Pappas, G. J., & Hassani, H.** (2022). Probabilistically robust learning: Balancing average- and worst-case performance. *International Conference on Machine Learning (ICML)*, PMLR 162, 18667–18686.
19. **Wei, Y., Zhang, C., Wang, Y., Wang, X., Shao, B., Li, H., & Di, L.** (2026). What paired evaluations reveal under visual perturbations. *arXiv preprint arXiv:2609.35583*.
