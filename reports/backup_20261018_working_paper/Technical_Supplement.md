# Technical Supplement: Order Has Consequences

**Title:** Order Has Consequences: Total Effects and Fidelity-Conditioned Prediction Instability under Compound Image Corruptions  
**Artifact:** Supplementary Materials, Mathematical Proofs & Empirical Diagnostic Tables  
**Author:** Avi (RobustLens Research Lead)  
**Date:** October 2026  
**Repository Path:** `reports/final_research_package/Technical_Supplement.md`  

---

## S0. Experimental Estimands & Methodological Framing

To ensure complete clarity and academic rigor, we distinguish two experimental regimes and their associated estimands:
1. **Estimand 1 (Total Effect of Sequence Order at Fixed Parameters):**  
   Evaluates vision classifiers under standard nominal corruption parameters (ImageNet-C settings). Under this estimand, low-level fidelity differences (such as terminal blur smoothing prior JPEG block boundaries) are downstream consequences of operator non-commutativity, forming part of the operational treatment effect.
2. **Estimand 2 (Order Differences at Comparable Output Fidelity):**  
   Evaluates vision classifiers across sequences pre-calibrated to target common (PSNR, SSIM) support. Because prespecified calibration tolerances ($\pm 0.75\text{ dB}$ PSNR, $\pm 0.035$ SSIM) were not achieved over the tested discrete grid, this regime is strictly an **approximate-fidelity descriptive comparison**. It documents how much of the nominal sequence gap contracts when low-level distortion is equalized, but does not claim that fidelity confounding was eliminated or that causal invariance was proven.

---

## S1. Mathematical Foundations & Statistical Methodology

### S1.1 Saturated Six-Cell Decomposition, Rank Proof, and Marginal Contrasts
Let $\mathcal{S} = (\text{NBJ}, \text{NJB}, \text{BNJ}, \text{BJN}, \text{JNB}, \text{JBN})^T$ be the ordered vector of six sequence permutations. Let $\mu \in \mathbb{R}^6$ denote the vector of population cell means for any evaluated metric (e.g., top-1 accuracy, confidence, Brier score, PSNR, SSIM).

We define reference levels $\text{Start} = \text{Noise}$ and $\text{End} = \text{Noise}$, and introduce a balanced permutation-parity contrast between even and odd permutations. The design matrix $X \in \mathbb{R}^{6 \times 6}$ is:
$$X = \begin{bmatrix}
1 & 0 & 0 & 0 & 1 & +1 \\
1 & 0 & 0 & 1 & 0 & -1 \\
1 & 1 & 0 & 0 & 1 & -1 \\
1 & 1 & 0 & 0 & 0 & +1 \\
1 & 0 & 1 & 1 & 0 & +1 \\
1 & 0 & 1 & 0 & 0 & -1
\end{bmatrix}$$
where columns correspond to:
1. Column 1 ($\mathbf{1}$): Intercept
2. Column 2 ($\text{Start\_B}$): Indicator that the initial operator is Defocus Blur ($\mathcal{B}$).
3. Column 3 ($\text{Start\_J}$): Indicator that the initial operator is JPEG Compression ($\mathcal{J}$).
4. Column 4 ($\text{End\_B}$): Indicator that the final operator is Defocus Blur ($\mathcal{B}$).
5. Column 5 ($\text{End\_J}$): Indicator that the final operator is JPEG Compression ($\mathcal{J}$).
6. Column 6 ($\text{Parity}$): Balanced contrast indicator ($+1$ for Group 1: `NBJ`, `BJN`, `JNB`; $-1$ for Group 2: `NJB`, `BNJ`, `JBN`).

#### Lemma S1: Non-Singularity of $X$
The determinant of $X$ is given by:
$$\det(X) = -6 \neq 0$$
Consequently, $\text{rank}(X) = 6$, and $X$ is invertible. Every vector of six cell means $\mu$ has a unique linear representation $\beta = X^{-1} \mu$.

#### Analytic Form of $X^{-1}$
The exact analytic inverse $X^{-1}$ is:
$$X^{-1} = \begin{bmatrix}
1/2 & 1/2 & -1/2 & 1/2 & -1/2 & 1/2 \\
-2/3 & -1/3 & 2/3 & 1/3 & 1/3 & -1/3 \\
-1/3 & -2/3 & 1/3 & -1/3 & 2/3 & 1/3 \\
-1/3 & 1/3 & 1/3 & -1/3 & 2/3 & -2/3 \\
1/3 & -1/3 & 2/3 & -2/3 & 1/3 & -1/3 \\
1/6 & -1/6 & -1/6 & 1/6 & 1/6 & -1/6
\end{bmatrix}$$

#### Unobserved Reference Level and Non-Orthogonality
1. **Unobserved Reference Level:** In reference coding, the baseline cell corresponds to Start Noise and End Noise. However, a 3-operator sequence without repetition cannot both start and end with Noise. Hence, this reference condition is unobserved, and the Intercept $\beta_0$ represents an extrapolated linear baseline rather than an observable physical sequence.
2. **Non-Orthogonal Columns:** The columns of $X$ are not mutually orthogonal ($X^T X \neq I$). Therefore, reference-coded $\beta$ parameters should be analyzed alongside direct marginal contrasts.
3. **Balanced Parity Contrast:** Row 6 of $X^{-1}$ gives:
   $$\beta_5 = \frac{1}{6} \left[ (\mu_{\text{NBJ}} + \mu_{\text{BJN}} + \mu_{\text{JNB}}) - (\mu_{\text{NJB}} + \mu_{\text{BNJ}} + \mu_{\text{JBN}}) \right] = \frac{1}{2} (\bar{\mu}_{\text{Even}} - \bar{\mu}_{\text{Odd}})$$
   In both Group 1 (Even, $A_3$) and Group 2 (Odd, transpositions), each of the three operators $\{\text{N}, \text{B}, \text{J}\}$ appears exactly once at entry and once at exit. Thus, $\beta_5$ is perfectly balanced with respect to marginal entry and exit distributions.

#### Direct Marginal Contrasts
To provide unconstrained marginal estimates, we define:
- $\bar{\mu}_{\text{Start N}} = \frac{1}{2}(\mu_{\text{NBJ}} + \mu_{\text{NJB}})$, $\bar{\mu}_{\text{Start B}} = \frac{1}{2}(\mu_{\text{BNJ}} + \mu_{\text{BJN}})$, $\bar{\mu}_{\text{Start J}} = \frac{1}{2}(\mu_{\text{JNB}} + \mu_{\text{JBN}})$
- $\bar{\mu}_{\text{End N}} = \frac{1}{2}(\mu_{\text{BJN}} + \mu_{\text{JBN}})$, $\bar{\mu}_{\text{End B}} = \frac{1}{2}(\mu_{\text{NJB}} + \mu_{\text{JNB}})$, $\bar{\mu}_{\text{End J}} = \frac{1}{2}(\mu_{\text{NBJ}} + \mu_{\text{BNJ}})$
- Marginal Start Contrasts: $\Delta_{\text{Start B}} = \bar{\mu}_{\text{Start B}} - \bar{\mu}_{\text{Start N}}$, $\Delta_{\text{Start J}} = \bar{\mu}_{\text{Start J}} - \bar{\mu}_{\text{Start N}}$
- Marginal End Contrasts: $\Delta_{\text{End B}} = \bar{\mu}_{\text{End B}} - \bar{\mu}_{\text{End N}}$, $\Delta_{\text{End J}} = \bar{\mu}_{\text{End J}} - \bar{\mu}_{\text{End N}}$
- Balanced Parity Contrast: $\Delta_{\text{Parity}} = \bar{\mu}_{\text{Even}} - \bar{\mu}_{\text{Odd}} = 2 \beta_5$.

---

### S1.2 5-State Joint Prediction Space and Churn Decomposition
Let $\hat{Y}_A(i), \hat{Y}_B(i) \in \{1, \dots, K\}$ denote the model's predicted class, and let $Y(i) \in \{1, \dots, K\}$ denote the ground-truth label on physical image $i$. Binary correctness is $C_A(i) = \mathbb{I}(\hat{Y}_A(i) = Y(i))$ and $C_B(i) = \mathbb{I}(\hat{Y}_B(i) = Y(i))$.

The joint prediction space partitions into five mutually exclusive and exhaustive transitions:
1. $E_{11}$ (Both Correct): $\hat{Y}_A = Y$ and $\hat{Y}_B = Y$
2. $E_{10}$ (Loss / $L$): $\hat{Y}_A = Y$ and $\hat{Y}_B \neq Y$
3. $E_{01}$ (Gain / $G$): $\hat{Y}_A \neq Y$ and $\hat{Y}_B = Y$
4. $E_{00}^{\text{same}}$ (Same Wrong Label): $\hat{Y}_A = \hat{Y}_B \neq Y$
5. $E_{00}^{\text{diff}}$ (Error Migration / Different Wrong Labels): $\hat{Y}_A \neq Y$, $\hat{Y}_B \neq Y$, and $\hat{Y}_A \neq \hat{Y}_B$.

#### Theorem S1 (Decomposition of Prediction Instability):
1. **Total Probability Conservation:**
   $$\mathbb{P}(E_{11}) + \mathbb{P}(E_{10}) + \mathbb{P}(E_{01}) + \mathbb{P}(E_{00}^{\text{same}}) + \mathbb{P}(E_{00}^{\text{diff}}) = 1$$
2. **Correctness-State Churn:**
   $$\text{Correctness Churn} = \mathbb{P}(C_A \neq C_B) = \mathbb{P}(E_{10}) + \mathbb{P}(E_{01}) = L + G$$
3. **Predicted-Label Churn:**
   $$\text{Label Churn} = \mathbb{P}(\hat{Y}_A \neq \hat{Y}_B) = \mathbb{P}(E_{10}) + \mathbb{P}(E_{01}) + \mathbb{P}(E_{00}^{\text{diff}}) = \text{Correctness Churn} + \mathbb{P}(E_{00}^{\text{diff}})$$
4. **Prediction Agreement Rate:**
   $$\text{Agreement Rate} = \mathbb{P}(\hat{Y}_A = \hat{Y}_B) = \mathbb{P}(E_{11}) + \mathbb{P}(E_{00}^{\text{same}}) = 1 - \text{Label Churn}$$
5. **Net Accuracy Difference and Excess Churn:**
   $$\Delta\text{Acc}(A \to B) = G - L$$
   $$\text{Excess Correctness Churn} = \text{Correctness Churn} - |\Delta\text{Acc}| = 2 \cdot \min(G, L)$$
   $$\text{Excess Label Churn} = \text{Label Churn} - |\Delta\text{Acc}| = 2 \cdot \min(G, L) + \mathbb{P}(E_{00}^{\text{diff}})$$

*Proof:*  
The event $\hat{Y}_A = \hat{Y}_B$ occurs if and only if both predictions are correct ($\hat{Y}_A = Y$ and $\hat{Y}_B = Y$, which is $E_{11}$) or both predictions are wrong with identical class assignments ($\hat{Y}_A = \hat{Y}_B \neq Y$, which is $E_{00}^{\text{same}}$). Thus, $\mathbb{P}(\hat{Y}_A = \hat{Y}_B) = \mathbb{P}(E_{11}) + \mathbb{P}(E_{00}^{\text{same}})$.  
The complement event $\hat{Y}_A \neq \hat{Y}_B$ is Predicted-Label Churn, which consists of all instances where correctness differs ($E_{10}$ and $E_{01}$) plus instances where both are wrong but predict different classes ($E_{00}^{\text{diff}}$). Hence, $\text{Label Churn} = \mathbb{P}(E_{10}) + \mathbb{P}(E_{01}) + \mathbb{P}(E_{00}^{\text{diff}}) = \text{Correctness Churn} + \mathbb{P}(E_{00}^{\text{diff}})$.  
Substituting $\text{Correctness Churn} = |\Delta\text{Acc}| + 2\min(G, L)$ gives $\text{Excess Label Churn} = 2\min(G, L) + \mathbb{P}(E_{00}^{\text{diff}})$. $\square$

---

### S1.3 Hypothesis Testing and Multiplicity Adjustments in Log Space

#### Two-Sided Wald Hypothesis Testing
For each estimated parameter $\beta_j$ ($j \in \{0, \dots, 5\}$), we test:
$$H_0: \beta_j = 0 \quad \text{versus} \quad H_1: \beta_j \neq 0$$
All tests were strictly prespecified as non-directional, two-sided Wald tests. The Wald statistic is:
$$z_j = \frac{\hat{\beta}_j}{\widehat{\text{SE}}_{\text{boot}}(\beta_j)}$$
where $\widehat{\text{SE}}_{\text{boot}}$ is estimated via clustered bootstrap ($B = 5{,}000$ resamples on $N = 3{,}925$ physical images).

To prevent numerical underflow and eliminate arbitrary truncation to literal `0.0`, log p-values are computed via the Gaussian survival function:
$$\log p_j = \log(2) + \text{logsf}(|z_j|)$$
where $\text{logsf}(x) = \log(1 - \Phi(x))$.

#### Multiplicity Adjustments in Log Space
1. **Holm-Bonferroni (FWER control for primary accuracy family):**
   $$\log p_{(k)}^{\text{Holm}} = \max_{1 \le j \le k} \min \left( 0, \log p_{(j)} + \log(m - j + 1) \right)$$
2. **Benjamini-Hochberg (FDR control for secondary family):**
   $$\log p_{(k)}^{\text{BH}} = \min_{k \le j \le m} \min \left( 0, \log p_{(j)} + \log(m) - \log(j) \right)$$

---

## S2. Calibration Protocol, Sample Size Audit, and Residual Equations

### S2.1 Objective Calibration Loss Function
For sequence $s \in \mathcal{S}$ and target bin $k \in \{1, \dots, 5\}$, parameter vectors $\theta_s(k) = (\sigma_s(k), \kappa_s(k), \gamma_s(k), q_s(k))$ were selected by evaluating discrete candidate grids on an independent training partition ($N_{\text{cal}} = 50$ training images sampled from rows $0 \dots 9{,}468$ of FastAI Imagenette-v2 with seed 12345). Under no circumstances were validation images ($N = 3{,}925$) or model predictions used in calibration.

The objective function minimized was:
$$\theta_s^*(k) = \arg\min_{\theta \in \Theta} \left[ \lambda_P \left( \overline{\text{PSNR}}_s(\theta) - T_P(k) \right)^2 + \lambda_S \left( \overline{\text{SSIM}}_s(\theta) - T_S(k) \right)^2 \right]$$
with $\lambda_P = 1.0$ and $\lambda_S = 100.0$.

### S2.2 Calibrated Parameter Table Across All 30 Conditions

| Target | Sequence | Noise $\sigma$ | Blur Kernel $\kappa$ | Blur $\gamma$ | JPEG Quality $q$ | Achieved PSNR (dB) | Achieved SSIM | PSNR Res (dB) | SSIM Res |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T1 (28.0 dB, 0.85)** | `NBJ` | 0.050 | 3 | 0.85 | 80 | 28.492 | 0.8691 | +0.492 | +0.0191 |
| T1 | `NJB` | 0.046 | 5 | 1.05 | 76 | 26.883 | 0.8499 | -1.117 | -0.0001 |
| T1 | `BNJ` | 0.040 | 3 | 0.70 | 85 | 28.067 | 0.7875 | +0.067 | -0.0625 |
| T1 | `BJN` | 0.024 | 3 | 0.85 | 84 | 27.575 | 0.7814 | -0.425 | -0.0686 |
| T1 | `JNB` | 0.046 | 5 | 1.00 | 76 | 27.008 | 0.8390 | -0.992 | -0.0110 |
| T1 | `JBN` | 0.024 | 3 | 0.85 | 84 | 27.863 | 0.7914 | -0.137 | -0.0586 |
| **T2 (24.5 dB, 0.74)** | `NBJ` | 0.082 | 5 | 1.45 | 65 | 24.983 | 0.7671 | +0.483 | +0.0271 |
| T2 | `NJB` | 0.084 | 7 | 1.68 | 56 | 23.952 | 0.7354 | -0.548 | -0.0046 |
| T2 | `BNJ` | 0.060 | 5 | 1.15 | 76 | 24.319 | 0.6197 | -0.181 | -0.1203 |
| T2 | `BJN` | 0.042 | 5 | 1.35 | 72 | 23.564 | 0.5642 | -0.936 | -0.1758 |
| T2 | `JNB` | 0.085 | 7 | 1.68 | 55 | 23.886 | 0.7236 | -0.614 | -0.0164 |
| T2 | `JBN` | 0.042 | 5 | 1.35 | 72 | 23.657 | 0.5701 | -0.843 | -0.1699 |
| **T3 (21.5 dB, 0.62)** | `NBJ` | 0.130 | 7 | 2.20 | 48 | 22.760 | 0.6628 | +1.260 | +0.0428 |
| T3 | `NJB` | 0.142 | 9 | 2.65 | 36 | 21.833 | 0.6240 | +0.333 | +0.0040 |
| T3 | `BNJ` | 0.095 | 7 | 1.75 | 64 | 21.156 | 0.4163 | -0.344 | -0.2037 |
| T3 | `BJN` | 0.064 | 7 | 1.95 | 58 | 20.836 | 0.3821 | -0.664 | -0.2379 |
| T3 | `JNB` | 0.145 | 9 | 2.65 | 35 | 21.801 | 0.6177 | +0.301 | -0.0023 |
| T3 | `JBN` | 0.064 | 7 | 1.95 | 58 | 20.887 | 0.3856 | -0.613 | -0.2344 |
| **T4 (18.5 dB, 0.48)** | `NBJ` | 0.220 | 11 | 3.20 | 32 | 20.489 | 0.5513 | +1.989 | +0.0713 |
| T4 | `NJB` | 0.260 | 13 | 4.10 | 18 | 19.486 | 0.5170 | +0.986 | +0.0370 |
| T4 | `BNJ` | 0.145 | 9 | 2.60 | 46 | 18.484 | 0.2593 | -0.016 | -0.2207 |
| T4 | `BJN` | 0.095 | 9 | 2.70 | 44 | 18.406 | 0.2427 | -0.094 | -0.2373 |
| T4 | `JNB` | 0.265 | 13 | 4.10 | 17 | 19.457 | 0.5138 | +0.957 | +0.0338 |
| T4 | `JBN` | 0.095 | 9 | 2.70 | 44 | 18.436 | 0.2444 | -0.064 | -0.2356 |
| **T5 (16.0 dB, 0.36)** | `NBJ` | 0.320 | 15 | 4.40 | 18 | 18.639 | 0.4761 | +2.639 | +0.1161 |
| T5 | `NJB` | 0.380 | 17 | 5.20 | 10 | 17.710 | 0.4629 | +1.710 | +0.1029 |
| T5 | `BNJ` | 0.220 | 11 | 3.55 | 31 | 16.043 | 0.1575 | +0.043 | -0.2025 |
| T5 | `BJN` | 0.135 | 11 | 3.50 | 30 | 16.195 | 0.1533 | +0.195 | -0.2067 |
| T5 | `JNB` | 0.385 | 17 | 5.20 | 9 | 17.700 | 0.4608 | +1.700 | +0.1008 |
| T5 | `JBN` | 0.135 | 11 | 3.50 | 30 | 16.215 | 0.1542 | +0.215 | -0.2058 |

---

## S3. Complete Architectural Saturated Decomposition Tables

Evaluated on $N = 3{,}925$ physical validation images ($B = 5{,}000$ bootstrap iterations, Holm FWER-adjusted).

### S3.1 ResNet-50
| Parameter | Nominal Est (%) | Nominal 95% CI | $p_{\text{Holm}}$ | Matched Est (%) | Matched 95% CI | $p_{\text{Holm}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Intercept ($\beta_0$) | +43.62 | [+42.72, +44.53] | $< 10^{-100}$ | +53.19 | [+52.13, +54.21] | $< 10^{-100}$ |
| Start Blur ($\beta_1$) | -10.65 | [-11.06, -10.22] | $< 10^{-100}$ | -3.36 | [-3.75, -2.99] | $< 10^{-100}$ |
| Start JPEG ($\beta_2$) | -5.09 | [-5.41, -4.75] | $< 10^{-100}$ | -1.31 | [-1.61, -1.03] | $< 10^{-30}$ |
| End Blur ($\beta_3$) | **+16.00** | [+15.32, +16.67] | $< 10^{-100}$ | **-5.63** | [-6.30, -4.92] | $< 10^{-50}$ |
| End JPEG ($\beta_4$) | **+7.07** | [+6.55, +7.61] | $< 10^{-100}$ | **-3.85** | [-4.35, -3.33] | $< 10^{-45}$ |
| Parity Contrast ($\beta_5$) | **+2.74** | [+2.63, +2.86] | $< 10^{-100}$ | **+0.80** | [+0.69, +0.91] | $< 10^{-40}$ |

### S3.2 ConvNeXt-Tiny
| Parameter | Nominal Est (%) | Nominal 95% CI | $p_{\text{Holm}}$ | Matched Est (%) | Matched 95% CI | $p_{\text{Holm}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Intercept ($\beta_0$) | +54.34 | [+53.30, +55.35] | $< 10^{-100}$ | +61.87 | [+60.78, +62.96] | $< 10^{-100}$ |
| Start Blur ($\beta_1$) | -10.68 | [-11.14, -10.25] | $< 10^{-100}$ | -3.33 | [-3.69, -2.95] | $< 10^{-65}$ |
| Start JPEG ($\beta_2$) | -5.67 | [-6.02, -5.33] | $< 10^{-100}$ | -1.34 | [-1.63, -1.05] | $< 10^{-18}$ |
| End Blur ($\beta_3$) | **+14.43** | [+13.84, +15.04] | $< 10^{-100}$ | **-4.74** | [-5.26, -4.22] | $< 10^{-68}$ |
| End JPEG ($\beta_4$) | **+7.31** | [+6.75, +7.88] | $< 10^{-100}$ | **-1.90** | [-2.41, -1.41] | $1.49 \times 10^{-13}$ |
| Parity Contrast ($\beta_5$) | **+2.33** | [+2.22, +2.45] | $< 10^{-100}$ | **+0.67** | [+0.57, +0.78] | $< 10^{-35}$ |

### S3.3 ViT-Base/16
| Parameter | Nominal Est (%) | Nominal 95% CI | $p_{\text{Holm}}$ | Matched Est (%) | Matched 95% CI | $p_{\text{Holm}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Intercept ($\beta_0$) | +51.72 | [+50.61, +52.83] | $< 10^{-100}$ | +60.40 | [+59.13, +61.63] | $< 10^{-100}$ |
| Start Blur ($\beta_1$) | -12.93 | [-13.49, -12.36] | $< 10^{-100}$ | -7.13 | [-7.65, -6.65] | $< 10^{-100}$ |
| Start JPEG ($\beta_2$) | -6.64 | [-7.05, -6.21] | $< 10^{-100}$ | -3.27 | [-3.65, -2.91] | $< 10^{-60}$ |
| End Blur ($\beta_3$) | **+11.27** | [+10.61, +11.92] | $< 10^{-100}$ | **-5.56** | [-6.19, -4.95] | $< 10^{-65}$ |
| End JPEG ($\beta_4$) | **+3.83** | [+3.25, +4.37] | $< 10^{-35}$ | **-5.31** | [-5.86, -4.76] | $< 10^{-75}$ |
| Parity Contrast ($\beta_5$) | **+2.85** | [+2.71, +2.98] | $< 10^{-100}$ | **+1.45** | [+1.33, +1.57] | $< 10^{-100}$ |

### S3.4 Swin-Tiny
| Parameter | Nominal Est (%) | Nominal 95% CI | $p_{\text{Holm}}$ | Matched Est (%) | Matched 95% CI | $p_{\text{Holm}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Intercept ($\beta_0$) | +56.85 | [+55.70, +57.99] | $< 10^{-100}$ | +62.58 | [+61.40, +63.79] | $< 10^{-100}$ |
| Start Blur ($\beta_1$) | -9.07 | [-9.50, -8.65] | $< 10^{-100}$ | -2.12 | [-2.51, -1.72] | $< 10^{-24}$ |
| Start JPEG ($\beta_2$) | -3.76 | [-4.09, -3.44] | $< 10^{-100}$ | +0.32 | [+0.01, +0.62] | $0.0399$ |
| End Blur ($\beta_3$) | **+7.39** | [+6.84, +7.94] | $< 10^{-100}$ | **-11.36** | [-11.92, -10.82] | $< 10^{-100}$ |
| End JPEG ($\beta_4$) | +0.49 | [-0.03, +1.01] | $0.0672$ | **-5.75** | [-6.28, -5.24] | $< 10^{-100}$ |
| Parity Contrast ($\beta_5$) | **+2.46** | [+2.34, +2.59] | $< 10^{-100}$ | **+0.78** | [+0.66, +0.90] | $< 10^{-35}$ |

---

## S4. Vectorized Bootstrap Algorithm

To execute $B = 5{,}000$ bootstrap iterations over $471{,}000$ compound records in seconds:
1. Aggregate predictions across severities/targets per physical image, model, and sequence into a 3D tensor $Y \in \mathbb{R}^{N \times M \times 6}$ ($N = 3{,}925$ physical images, $M = 4$ models, 6 sequences).
2. Sample cluster indices with replacement: $I_{\text{boot}} \in \{0, \dots, N-1\}^{B \times N}$.
3. Index the tensor vectorizedly: $Y_{\text{resampled}} = Y[I_{\text{boot}}]$, shape $(B, N, M, 6)$.
4. Compute bootstrap sample means along axis 1: $\overline{Y}^* = \frac{1}{N} \sum_{i=1}^N Y_{\text{resampled}}$, shape $(B, M, 6)$.
5. Apply matrix transformation via Einstein summation: $\beta^* = \text{einsum}(\text{'bmi,ji}\to\text{bmj'}, \overline{Y}^*, X^{-1})$.
6. Percentile confidence intervals are computed via `np.percentile(beta*, [2.5, 97.5], axis=0)`.

---

## References

1. **Cormier, Q., Milani Fard, M., Canini, K., & Gupta, M. R.** (2016). Launch and iterate: Reducing prediction churn. *Advances in Neural Information Processing Systems (NeurIPS)*, 29, 3179–3187.
2. **Erkara, A. O., & Mayer, M.** (2026). Blind confusion of classification networks. *Neurocomputing*, 686, 133678.
3. **Goyal, B., Lalonde, J.-F., Li, Y., & Gupta, M.** (2022). Robust scene inference under noise-blur dual corruptions. *IEEE ICCP*, 1–12.
4. **Hendrycks, D., & Dietterich, T.** (2019). Benchmarking neural network robustness to common corruptions. *ICLR*.
5. **Li, Y., et al.** (2025). R-Bench: Benchmarking multi-stage physical degradation in multimodal large language models. *IEEE JSTSP*.
6. **Liao, X., Li, K., Zhu, X., & Liu, K. J. R.** (2020). Robust detection of image operator chain with two-stream convolutional neural network. *IEEE Journal of Selected Topics in Signal Processing*, 14(5), 955–968.
7. **Mintun, E., Kirillov, A., Gunasekar, S., Mu, S., & Xie, S.** (2021). On interaction between augmentations and corruptions in natural corruption robustness. *NeurIPS*, 34, 3571–3583.
8. **Wei, Y., et al.** (2026). What paired evaluations reveal under visual perturbations. *arXiv:2609.35583*.
