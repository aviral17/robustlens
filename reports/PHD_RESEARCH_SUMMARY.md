# RobustLens: Empirical Evaluation of Vision Models Under Distribution Shift
## Research Portfolio & PhD Application Summary (Fall 2027)

**Author:** Avi  
**Primary Research Focus:** Empirical Deep Learning, Computer Vision Robustness, Applied Statistics & Uncertainty Quantification  
**Codebase:** `https://github.com/aviral199/robustlens`  
**Primary Research Package (October 2026):**
1. **PhD Research Working Paper / Writing Sample:** [`reports/final_research_package/RobustLens_PhD_Working_Paper.pdf`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/final_research_package/RobustLens_PhD_Working_Paper.pdf) & [`RobustLens_PhD_Working_Paper.md`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/final_research_package/RobustLens_PhD_Working_Paper.md)  
   *"Order Has Consequences: Total Effects and Fidelity-Conditioned Prediction Instability under Compound Image Corruptions"*
2. **Technical Supplement & Mathematical Proofs:** [`reports/final_research_package/Technical_Supplement.md`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/final_research_package/Technical_Supplement.md)
3. **Literature Review & Prior-Art Audit Matrix:** [`reports/final_research_package/LITERATURE_REVIEW_MATRIX.md`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/final_research_package/LITERATURE_REVIEW_MATRIX.md)
4. **Scientific Audit & Issue Resolution Log:** [`reports/final_research_package/FINAL_REVIEW_ISSUES.md`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/final_research_package/FINAL_REVIEW_ISSUES.md)
5. **Originality & Scope Boundaries Audit:** [`reports/final_research_package/ORIGINALITY_AND_LIMITATIONS_AUDIT.md`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/final_research_package/ORIGINALITY_AND_LIMITATIONS_AUDIT.md)
6. **Reproducibility Protocol:** [`reports/final_research_package/REPRODUCIBILITY.md`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/final_research_package/REPRODUCIBILITY.md)
7. **Executive Completion Report:** [`reports/final_research_package/FINAL_COMPLETION_REPORT.md`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/final_research_package/FINAL_COMPLETION_REPORT.md)  
**Interactive Dashboard:** Next.js 16 (Turbopack), TypeScript, HeroUI v3, Tailwind CSS v4, Recharts (`apps/web`)  
**Grand Inference Scale:** **973,400 Total Inferences** (486,700 nominal-order evaluations + 486,700 approximate-fidelity evaluations) on $N = 3{,}925$ physical images across 4 architectures.

---

### 1. Research Motivation & Core Questions

Standard evaluation benchmarks measure vision model performance on independent and identically distributed (i.i.d.) test sets. However, real-world deployment exposes models to non-commutative multi-stage degradations (e.g., Sensor Noise $\mathcal{N}$, Optical Blur $\mathcal{B}$, and JPEG Compression $\mathcal{J}$).

This research investigates two fundamental questions:
1. **The Distortion Confounding Question:** In sequential multi-operator benchmarks, does execution sequence predict classifier failure because non-commuting operators produce unequal physical degradation?
2. **The Counterfactual Distortion Question:** When parameters are calibrated to common physical fidelity (PSNR/SSIM common support), does corruption sequence still predict classifier accuracy, error churn, and predictive uncertainty?

---

### 2. Key Scientific Findings & Empirical Discoveries

#### Finding 1: The Terminal Smoothing Confound in Nominal Benchmarks
In standard nominal evaluations (applying canonical ImageNet-C parameters across sequences), accuracy varies by up to **$21.55\%$ in ResNet-50** and **$20.45\%$ in ConvNeXt-Tiny** across the six permutations on identical physical images with seed-matched noise. However, we discovered that these gaps are dominated by terminal low-pass smoothing: sequences ending in blur (`NJB`, `JNB`) smooth out prior JPEG $8 \times 8$ DCT block boundaries, conferring up to a $5.0\text{ dB}$ PSNR boost over sequences ending in JPEG (`NBJ`, `BNJ`).

#### Finding 2: Approximate-Fidelity Calibration and Substantial Gap Contraction
Calibrating parameters on an independent training partition ($N_{\text{cal}} = 50$, zero validation leakage) across five target distortion bins revealed that **prespecified joint matching criteria ($\pm 0.75\text{ dB}$ PSNR, $\pm 0.035$ SSIM) were not met over the tested discrete grid** (PSNR RMSE = $0.943\text{ dB}$, SSIM RMSE = $0.1380$). 
Nonetheless, equalizing distortion on common support produced a dramatic contraction of cross-sequence accuracy gaps:
- **ConvNeXt-Tiny:** Sequence gap contracts from **$20.45\%$ down to $4.67\%$** (**$77.15\%$ contraction**).
- **ResNet-50:** Sequence gap contracts from **$21.55\%$ down to $5.90\%$** (**$72.62\%$ contraction**).
- **ViT-Base/16:** Sequence gap contracts from **$20.37\%$ down to $10.03\%$** (**$50.75\%$ contraction**).
- **Swin-Tiny:** Sequence gap contracts from **$17.13\%$ down to $11.68\%$** (**$31.80\%$ contraction**).
- **Overall Mean:** Sequence gap contracts from **$18.43\%$ down to $6.79\%$** (**$63.18\%$ contraction**).

#### Finding 3: Full-Rank Saturated Six-Cell Decomposition with Balanced Parity
We formulated a full-rank saturated six-cell decomposition ($\det(X) = -6$) with a balanced parity contrast between even and odd permutations ($\beta_5$). Clustered bootstrap inference ($B = 5{,}000$ on physical images, Holm FWER-controlled) showed:
- In nominal evaluations, terminal blur conferred a $+12.27\%$ accuracy boost.
- Under calibrated fidelity, the terminal blur effect reversed to **$-6.82\%$** ($p < 10^{-100}$): when PSNR is balanced, terminal blur strips high-frequency discriminative features.
- The balanced parity contrast remained statistically significant ($+0.93\%$, $p < 10^{-100}$), confirming that permutation parity independently influences classification accuracy.

#### Finding 4: The 5-State Prediction Space & Error Migration Decoupling
Microscopic paired single-image analysis on matched observations established the 5-state joint prediction space ($E_{11}, E_{10}, E_{01}, E_{00}^{\text{same}}, E_{00}^{\text{diff}}$). We proved that **Predicted-Label Churn** ($34.74\%–40.78\%$) is more than double **Correctness-State Churn** ($12.34\%–16.93\%$) because **Error Migration** ($E_{00}^{\text{diff}}$, where models predict different wrong classes) averages $\sim 23\%$ across regimes:
- In ConvNeXt under calibrated fidelity, `NJB` (56.46%) and `JNB` (56.47%) differed by only **$0.01\%$**, yet exhibited **$8.22\%$ excess churn** ($1{,}613$ flipped predictions).
- In Swin-Tiny, `BJN` (61.23%) and `JBN` (62.12%) differed by **$0.89\%$**, with **$8.84\%$ excess churn**.
- For vulnerable classes like 'church', top-1 accuracy collapses to $18.35\%$ while Error Migration reaches **$39.41\%$** (pushing Label Churn to $50.45\%$, nearly $4\times$ its correctness churn). Aggregate accuracy masks severe single-image failure set reshuffling.

---

### 3. Quantitative Comparison Table

| Architecture | Clean Acc | Matched Mean Acc | Nominal Gap | Matched Gap | Gap Contraction | Mean Churn | Excess Churn Ratio | Mean ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ConvNeXt-Tiny** | 87.13% | **58.11%** | 20.45% | **4.67%** | **77.15%** | 10.93% | **75.5%** | 0.0653 |
| **Swin-Tiny** | 86.04% | 56.27% | **17.13%** | 11.68% | 31.80% | 11.92% | 56.9% | 0.1039 |
| **ViT-Base/16** | 80.25% | 53.31% | 20.37% | 10.03% | 50.75% | 13.40% | 67.0% | **0.0420** |
| **ResNet-50** | **87.36%** | 48.47% | 21.55% | 5.90% | 72.62% | 13.10% | 70.9% | 0.0919 |

---

### 4. Technical Rigor & Quality Verification

- **Code Quality:** Automated unit tests in `pytest` verify split integrity, parameter calibration, design matrix rank ($\det(X) = -6$), and cell reconstruction identities.
- **Production Build:** Next.js 16 web application compiles with 0 errors (`next build`, 13 static routes).
- **Artifact Verification:** 486,700 records in each regime verified with zero null values across 3,925 images, 4 models, and 31 conditions.
- **Inferential Rigor:** Clustered bootstrap on physical images ($B = 5{,}000$), Holm-Bonferroni FWER control for primary family, and Benjamini-Hochberg FDR control for secondary metrics.
