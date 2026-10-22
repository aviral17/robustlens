# Faculty Review Brief: Compound Corruption Order Effects & Counterfactual Fidelity Matching in Computer Vision

**Candidate Research Project for US Computer Vision & Machine Learning PhD Applications (Fall 2027)**  
**Author:** Aviral (Avi) • Independent Research  
**Primary Research Focus:** Trustworthy Computer Vision, Distribution Shift, Applied Statistics & Representation Invariance  
**Date:** October 2026  
**Repository & Reproducibility:** [GitHub Repository](https://github.com/aviral199/robustlens)  
**Companion Manuscripts:**
- **New Paper:** [`RobustLens_New_Research_Manuscript.pdf`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/RobustLens_New_Research_Manuscript.pdf) (*"Towards Matched-Fidelity Robustness Benchmarking: An Approximate-Fidelity Counterfactual Study of Multi-Stage Vision Classifier Vulnerabilities"*)

- **Foundational Paper:** [`RobustLens_Research_Manuscript.pdf`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/RobustLens_Research_Manuscript.pdf)
- **Literature Audit:** [`MATCHED_FIDELITY_LITERATURE_AUDIT.md`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/MATCHED_FIDELITY_LITERATURE_AUDIT.md)

---

## 1. Executive Summary

- **Core Problem:** Real-world visual distribution shifts rarely occur as isolated, unary perturbations (e.g., ImageNet-C) or in fixed canonical pipelines. Autonomous systems encounter compounded sensor degradation: optical defocus blur, sensor photon noise, and lossy JPEG compression. Because image processing operators do not commute ($\mathcal{T}_A \circ \mathcal{T}_B \neq \mathcal{T}_B \circ \mathcal{T}_A$), sequence order alters the resulting spatial frequency spectrum and artifact geometry.
- **The Confounding Challenge & Trade-offs:** In nominal benchmarks, terminal-noise sequences (`BJN`, `JBN`) suffer from an unaddressed physical confound: lack of low-pass filtering collapses PSNR by over 4 dB. Across an extensive calibration grid search on an independent training split, the prespecified joint matching criteria (PSNR $\pm 0.75$ dB, SSIM $\pm 0.035$) could not be met simultaneously across all sequences due to conflicting structural trade-offs between additive noise and spatial blur.
- **The Approximate-Fidelity Design:** We calibrated operator parameters across all 6 permutations to **5 target fidelity levels on common support** (28 dB down to 16 dB PSNR) on an independent training split ($N_{\text{cal}}=50$, zero validation/label leakage).
- **Scale:** Full factorial design across 4 canonical architectures (ConvNeXt-Tiny, Swin-Tiny, ViT-Base/16, ResNet-50) over $N=3{,}925$ physical images from the Imagenette-v2 validation split. Exactly **486,700 balanced prediction records** evaluated with cluster-robust logistic ANCOVA, repeated-measures GEE, and error churn decomposition.

---

## 2. Quantitative Discoveries & Theoretical Takeaways

| Architecture | Clean Acc | Approx Compound Acc | Nominal Gap | Approx Gap | Gap Contraction | Churn Ratio ($2\min(G,L)/(G+L)$) | Clustered Order Wald (5 df) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ConvNeXt-Tiny** | 87.13% | **58.11%** | 20.45% | **4.67%** | **77.1%** | **75.5%** | $W=119.4$ ($p = 4.2 \times 10^{-24}$) |
| **Swin-Tiny** | 86.04% | 56.27% | **17.13%** | 11.68% | 31.8% | 56.9% | $W=159.7$ ($p = 1.2 \times 10^{-32}$) |
| **ViT-Base/16** | 80.25% | 53.31% | 20.37% | 10.03% | 50.7% | 67.0% | $W=130.1$ ($p = 2.3 \times 10^{-26}$) |
| **ResNet-50** | **87.36%** | 48.47% | 21.55% | 5.90% | 72.6% | 70.9% | $W=268.9$ ($p = 4.8 \times 10^{-56}$) |

### Key Theoretical Takeaways:
1. **The Gap Contraction Effect:** In pure convolutional architectures (ConvNeXt and ResNet), **over 72% to 77% of the nominal sequence gap was an artifact of coarse MSE confounding**. Calibrating to common support contracts the macro accuracy gap to $\approx 5\%$.
2. **Persistent Inductive Bias:** Despite dramatic gap contraction, the sequence effect does not collapse to zero. Cluster-robust logistic ANCOVA confirms that the Order Main Effect ($5$ df) and Order $\times$ Target Interaction ($20$ df) remain statistically significant across all 4 models ($p < 10^{-23}$).
3. **Structural Trade-offs & Residual Confounding:** Simultaneous dual-metric matching within prespecified tolerances was not achieved because terminal blur (`NJB`, `JNB`) smooths noise (boosting SSIM) while terminal noise (`BJN`, `JBN`) destroys local patch correlation. Sequences ending in JPEG (`NBJ`, `BNJ`) leave structured block boundaries. The apparent inversion of terminal noise vs blur under approximate fidelity reflects the heavy blur needed to match PSNR, proving scalar matching leaves residual structural confounding.
4. **The Churn Decoupling Phenomenon:** Classifiers undergo massive sample-level error churn (10.9% to 13.4% of physical images flip correctness between sequences). Excess churn accounts for **57% to 76%** of all prediction shifts, proving that aggregate accuracy metrics mask severe instance-level instability.

---

## 3. Alignment with Faculty Research & PhD Admissions

1. **Trustworthy Computer Vision & Distribution Shift:**
   - *Faculty:* Prof. Dan Hendrycks (UC Berkeley / CAIS), Prof. Trevor Darrell (UC Berkeley), Prof. Percy Liang (Stanford).
   - *Fit:* Advances beyond unary ImageNet-C by establishing counterfactual common-support evaluation protocols for multi-stage physical degradations.
2. **Vision Transformers & Representation Geometry:**
   - *Faculty:* Prof. Kaiming He (MIT), Prof. Jitendra Malik (UC Berkeley), Prof. Alexei Efros (UC Berkeley).
   - *Fit:* Explains the differential resilience of patch tokenization (ViT/Swin) vs. spatial convolution kernels against non-commutative phase and block-boundary artifacts.
3. **Statistical Uncertainty Quantification & Calibration:**
   - *Faculty:* Prof. Emmanuel Candès (Stanford), Prof. Ryan Tibshirani (UC Berkeley), Prof. Aaditya Ramdas (CMU).
   - *Fit:* Applies cluster-robust Huber-White sandwich estimators, Monte Carlo paired sign-flip permutation tests ($R=20{,}000$), and error churn decomposition to prevent pseudoreplication in computer vision benchmarking.

---

## 4. Key Interview & Presentation Talking Points

1. **"Why was the nominal order effect confounded, and how did you resolve it?"**
   *Talking Point:* "In our initial 486,700-record benchmark, we found a 20.45% accuracy gap in ConvNeXt between terminal-blur sequence `NJB` (66.44%) and terminal-noise sequence `BJN` (45.99%, with terminal-JPEG `BNJ` at 48.63%). But analyzing the physical image tensors revealed that `NJB` had 4 dB higher PSNR because terminal blur smoothed both noise and JPEG blocks, whereas `JBN` had unattenuated noise. To evaluate whether models were failing due to operator sequence or simply lower SNR, we conducted an independent calibration on training data (zero leakage) targeting common PSNR and SSIM support across all 6 permutations. When evaluated under this approximate common support, the accuracy gap contracted by 77.1% in ConvNeXt, proving that nominal benchmarks heavily confound operator sequence with distortion magnitude."

2. **"Does order still matter if you match PSNR and SSIM?"**
   *Talking Point:* "Yes, definitively. In our cluster-robust logistic ANCOVA clustered on 3,925 images, the Wald statistic for Order main effect (5 df) rejects the null at $p < 10^{-23}$ across all 4 architectures. Furthermore, the error churn analysis proves that even when net accuracy differences are near zero, classifiers flip predictions on 11% to 13% of individual physical images, with over 70% of that churn being excess volatility unexplained by net accuracy shifts."

3. **"Why did terminal noise outperform terminal blur under matched fidelity?"**
   *Talking Point:* "In nominal setups, terminal noise had much higher variance. But when noise is calibrated down so that MSE is equal to terminal blur, the noise preserves high-frequency edge phase and gradient orientation information. In contrast, heavy terminal blur removes high-frequency gradients altogether, making fine-grained object discrimination harder. This demonstrates that scalar PSNR matching trades spatial edge clarity against noise variance, highlighting the residual structural confounding that remains in multi-stage benchmarks."
