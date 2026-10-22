# RobustLens: Empirical Evaluation of Vision Models Under Distribution Shift
## Research Portfolio & PhD Application Summary (Fall 2027)

**Author:** Aviral (Avi)  
**Primary Research Focus:** Empirical Deep Learning, Computer Vision Robustness, Applied Statistics & Uncertainty Quantification  
**Codebase:** `https://github.com/aviral199/robustlens`  
**Core Manuscripts:**
1. **New Counterfactual Paper:** [`reports/RobustLens_New_Research_Manuscript.pdf`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/RobustLens_New_Research_Manuscript.pdf) & [`reports/RobustLens_New_Research_Manuscript.md`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/RobustLens_New_Research_Manuscript.md)  
   *"Towards Matched-Fidelity Robustness Benchmarking: An Approximate-Fidelity Counterfactual Study of Multi-Stage Vision Classifier Vulnerabilities"*

2. **Foundational Robustness Paper:** [`reports/RobustLens_Research_Manuscript.pdf`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/RobustLens_Research_Manuscript.pdf)  
   *"RobustLens: Statistical Evaluation of Vision Models Under Distribution Shift"*
3. **Technical Supplement & Proofs:** [`reports/RobustLens_New_Research_Supplement.md`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/RobustLens_New_Research_Supplement.md)
4. **Literature Audit & Novelty Defense:** [`reports/MATCHED_FIDELITY_LITERATURE_AUDIT.md`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/MATCHED_FIDELITY_LITERATURE_AUDIT.md)  
**Interactive Dashboard:** Next.js 16 (Turbopack), TypeScript, HeroUI v3, Tailwind CSS v4, Recharts (`apps/web`)  
**Grand Inference Scale:** **737,900 Total Inferences** (251,200 primary benchmark + 486,700 counterfactual matched-fidelity benchmark) on $N=3{,}925$ physical images across 4 architectures.

---

### 1. Research Motivation & Core Questions

Standard evaluation benchmarks measure vision model performance on independent and identically distributed (i.i.d.) test sets. However, real-world deployment exposes models to non-commutative multi-stage degradations (e.g., Sensor Noise $\mathcal{N}$, Optical Blur $\mathcal{B}$, and JPEG Compression $\mathcal{J}$).

This research investigates two fundamental questions:
1. **The Statistical Evaluation Question:** How do modern depthwise convolutional networks (ConvNeXt) compare against classical CNNs (ResNet) and Vision Transformers (ViT, Swin) under controlled distribution shifts when evaluated with strict cluster-level statistical hypothesis testing?
2. **The Counterfactual Distortion Question:** In multi-stage corruption benchmarks, does execution sequence predict classifier failure simply because non-commuting operators produce unequal physical degradation (a 4 dB PSNR spread)? **Can degradations be calibrated to common physical fidelity, and what are the mathematical limits of such matching?**

---

### 2. Key Scientific Findings & Empirical Discoveries

#### Finding 1: Calibration Trade-offs & Approximate Gap Contraction
In nominal benchmarks, applying noise last (`BJN`, `JBN`) depresses PSNR by up to 4 dB and collapses SSIM because Gaussian noise is never low-pass filtered. Conversely, terminal-blur sequences (`NJB`, `JNB`) smooth prior perturbations, and terminal-JPEG sequences (`NBJ`, `BNJ`) exhibit structured block quantization.
Across an extensive calibration grid search on an independent training partition ($N_{\text{cal}}=50$, zero validation/label leakage), **the prespecified joint matching criteria ($\pm 0.75$ dB PSNR, $\pm 0.035$ SSIM) could not be met simultaneously across sequences**, reflecting severe multi-objective trade-offs between additive noise and spatial blur.
Under an **approximate-fidelity calibration** targeting 5 common fidelity levels (28 dB to 16 dB PSNR):
- **ConvNeXt-Tiny:** Sequence accuracy gap collapses from **20.45% down to 4.67%** (**77.1% gap contraction**, **95.9% variance reduction**).
- **ResNet-50:** Sequence accuracy gap collapses from **21.55% down to 5.90%** (**72.6% gap contraction**, **94.9% variance reduction**).
- **ViT-Base/16:** Sequence gap contracts from **20.37% to 10.03%** (**50.7% gap contraction**, **86.9% variance reduction**).
- **Swin-Tiny:** Sequence gap contracts from **17.13% to 11.68%** (**31.8% gap contraction**, **55.1% variance reduction**).

*Conclusion:* In pure CNNs, over 70% of the nominal "order effect" was an artifact of coarse MSE confounding. In Vision Transformers, sequence sensitivity persists more strongly due to the non-commutative interaction between JPEG block quantization and self-attention patch tokenization.

#### Finding 2: Inversion of Nominal Rankings & Residual Confounding Caveats
Under approximate fidelity calibration, an apparent hierarchy inversion emerges: terminal-noise sequences (`JBN`, `BJN`) outperformed terminal-blur sequences (`NJB`, `JNB`) in Swin-Tiny ($62.1\%$ vs $50.4\%$) and ResNet-50 ($51.1\%$ vs $46.8\%$), while terminal-JPEG sequences (`NBJ`, `BNJ`) achieved intermediate accuracies ($57.6\%$ and $53.9\%$ in Swin; $50.1\%$ and $45.2\%$ in ResNet).
*Critical Scientific Caveat:* Rather than proving an intrinsic preference for noise over blur, this inversion illustrates the residual confounding inherent in multi-stage calibration: forcing terminal blur down to low PSNR requires large blur kernels ($k=13..17$, $\sigma=4.1..5.2$), destroying edge phase structures while leaving SSIM higher ($0.46$ vs $0.15$). Thus, sequence effects cannot be completely decoupled through scalar calibration alone.


#### Finding 3: Persistent Order Effects in Factorial Clustered ANCOVA
Fitting cluster-robust logistic regressions ($N = 117,750$ per model, clustered on $3,925$ physical images):
- Order Main Effect (5 df): ConvNeXt $W=119.4$ ($p = 4.2 \times 10^{-24}$); Swin $W=159.7$ ($p = 1.2 \times 10^{-32}$); ViT $W=130.1$ ($p = 2.3 \times 10^{-26}$); ResNet $W=268.9$ ($p = 4.8 \times 10^{-56}$).
- Order $\times$ Target Interaction (20 df): All models significant with $p < 10^{-38}$.
- *Conclusion:* Even after controlling for continuous PSNR and SSIM, corruption order remains an active determinant of classifier failure.

#### Finding 4: The Churn Decoupling Phenomenon
Microscopic paired analysis ($N = 300$ condition pairs) reveals that classifiers flip predictions on **10.9% to 13.4% of individual images** between sequences, even when net accuracy differences are near zero (2.7% in ConvNeXt).
- **Excess Churn ($2 \min(G, L)$):** Accounts for **56.9% to 75.5%** of all prediction shifts.
- Evaluating vision models purely by aggregate accuracy masks substantial instance-level volatility.

---

### 3. Quantitative Comparison Table

| Architecture | Clean Acc | Matched Compound Acc | Nominal Gap | Matched Gap | Gap Contraction | Mean Churn | Excess Churn Ratio | Mean ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ConvNeXt-Tiny** | 87.13% | **58.11%** | 20.45% | **4.67%** | **77.1%** | 10.93% | **75.5%** | 0.0653 |
| **Swin-Tiny** | 86.04% | 56.27% | **17.13%** | 11.68% | 31.8% | 11.92% | 56.9% | 0.1039 |
| **ViT-Base/16** | 80.25% | 53.31% | 20.37% | 10.03% | 50.7% | 13.40% | 67.0% | **0.0420** |
| **ResNet-50** | **87.36%** | 48.47% | 21.55% | 5.90% | 72.6% | 13.10% | 70.9% | 0.0919 |

---

### 4. Technical Rigor & Quality Verification

- **Code Quality:** **43 / 43 automated unit tests passing** in `pytest` across statistics, calibration, conformal prediction, dataset split integrity, matched-fidelity parameters, and error churn identities.
- **Production Build:** Next.js 16 web application compiles with 0 errors (`next build`, 13 static routes).
- **Artifact Verification:** 486,700 records in `all_order_predictions.csv.gz` verified with zero null values, exactly 3,925 images, 4 models, and 31 conditions.
- **Inferential Rigor:** Clustered Huber-White sandwich covariance estimation, Monte Carlo paired sign-flip permutation tests ($R=20{,}000$), Holm-Bonferroni and Benjamini-Hochberg multiplicity corrections.
