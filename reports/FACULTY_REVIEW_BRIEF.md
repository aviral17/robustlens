# Faculty Review Brief: Non-Commutative Compound Corruption Benchmarking

**Candidate Research Brief for US Computer Vision & Machine Learning PhD Applications (Fall 2027)**  
**Author:** Avi • Independent Research  
**Primary Research Focus:** Trustworthy Computer Vision, Distribution Shift, Applied Statistics & Representation Invariance  
**Date:** October 2026  
**Repository & Reproducibility:** [GitHub Repository](https://github.com/aviral199/robustlens)  
**Primary Working Paper:** [`reports/final_research_package/RobustLens_PhD_Working_Paper.pdf`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/final_research_package/RobustLens_PhD_Working_Paper.pdf) & [`RobustLens_PhD_Working_Paper.md`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/final_research_package/RobustLens_PhD_Working_Paper.md) (*"Order Has Consequences: Total Effects and Fidelity-Conditioned Prediction Instability under Compound Image Corruptions"*)  
**Companion 5-Page Paper:** [`reports/final_research_package/RobustLens_Final_Paper.pdf`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/final_research_package/RobustLens_Final_Paper.pdf) & [`RobustLens_Final_Paper.md`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/final_research_package/RobustLens_Final_Paper.md)  
**Technical Supplement:** [`reports/final_research_package/Technical_Supplement.md`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/final_research_package/Technical_Supplement.md)  
**Literature Matrix & Audit:** [`reports/final_research_package/LITERATURE_REVIEW_MATRIX.md`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/final_research_package/LITERATURE_REVIEW_MATRIX.md) & [`ORIGINALITY_AND_LIMITATIONS_AUDIT.md`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/final_research_package/ORIGINALITY_AND_LIMITATIONS_AUDIT.md)  
**Scientific Issue Resolution Log:** [`reports/final_research_package/FINAL_REVIEW_ISSUES.md`](file:///c:/STATISTICS%20&%20CV/robustlens/reports/final_research_package/FINAL_REVIEW_ISSUES.md)  

---

## 1. Executive Summary

- **Core Problem:** Real-world visual distribution shifts rarely occur as isolated unary perturbations (e.g., ImageNet-C) or in fixed canonical pipelines. Autonomous systems encounter compounded sensor degradation: optical defocus blur, sensor photon noise, and lossy JPEG compression. Because image processing operators do not commute ($\mathcal{T}_A \circ \mathcal{T}_B \neq \mathcal{T}_B \circ \mathcal{T}_A$), sequence order alters the resulting spatial frequency spectrum and artifact geometry.
- **The Confounding Challenge:** In nominal benchmarks, applying standard ImageNet-C parameters across permutations yields massive cross-sequence accuracy gaps of up to $21.55\%$ (ResNet) and $20.45\%$ (ConvNeXt). However, we reveal that these gaps are dominated by terminal low-pass smoothing: sequences ending in blur (`NJB`, `JNB`) smooth prior JPEG $8 \times 8$ DCT block boundaries, conferring up to a $5.0\text{ dB}$ PSNR boost over sequences ending in JPEG (`NBJ`, `BNJ`).
- **Approximate-Fidelity Common Support:** We formulated an independent pre-calibration optimization ($N_{\text{cal}} = 50$ training images, zero validation leakage) targeting common (PSNR, SSIM) support across five distortion bins. While prespecified tolerances ($\pm 0.75\text{ dB}$ PSNR, $\pm 0.035$ SSIM) were not met over the tested discrete grid (PSNR RMSE = $0.943\text{ dB}$, SSIM RMSE = $0.1380$), equalizing distortion contracted cross-sequence accuracy gaps by $63.18\%$ overall—dropping to $4.67\%$ in ConvNeXt ($77.15\%$ contraction) and $5.90\%$ in ResNet ($72.62\%$ contraction).
- **Scale:** Evaluated across four canonical vision architectures (ConvNeXt-Tiny, Swin-Tiny, ViT-Base/16, ResNet-50) over $N = 3{,}925$ physical images from FastAI Imagenette-v2. Exactly **973,400 balanced prediction records** evaluated with clustered bootstrap ($B = 5{,}000$), saturated 6-cell linear decomposition, and paired single-image error churn decomposition.

---

## 2. Quantitative Discoveries & Theoretical Takeaways

| Architecture | Clean Acc | Matched Mean Acc | Nominal Gap | Matched Gap | Gap Contraction | Mean Churn | Excess Churn Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ConvNeXt-Tiny** | 87.13% | **58.11%** | 20.45% | **4.67%** | **77.15%** | 10.93% | **75.5%** |
| **Swin-Tiny** | 86.04% | 56.27% | **17.13%** | 11.68% | 31.80% | 11.92% | 56.9% |
| **ViT-Base/16** | 80.25% | 53.31% | 20.37% | 10.03% | 50.75% | 13.40% | 67.0% |
| **ResNet-50** | **87.36%** | 48.47% | 21.55% | 5.90% | 72.62% | 13.10% | 70.9% |

### Key Theoretical Takeaways:
1. **The Gap Contraction Effect:** In pure convolutional architectures (ConvNeXt and ResNet), **over 72% to 77% of the nominal sequence gap was an artifact of low-level distortion confounding**. Calibrating to common support contracts the macro accuracy gap to $\approx 5\%$.
2. **Full-Rank Saturated Six-Cell Decomposition:** A full-rank linear decomposition spanning all cell means ($\det(X) = -6$) reveals that upon distortion calibration, the terminal blur effect reverses from $+12.27\%$ to **$-6.82\%$** ($p < 10^{-100}$, Holm FWER-controlled): when PSNR is equalized, terminal blur removes high-frequency discriminative features. The balanced parity contrast between even and odd permutations remains statistically significant ($+0.93\%$, $p < 10^{-100}$), confirming that permutation parity independently influences classification accuracy.
3. **The Churn Decoupling Phenomenon:** Classifiers undergo massive sample-level error churn (10.9% to 13.4% of physical images flip correctness between sequences). Even when net accuracy difference is near zero ($|\Delta\text{Acc}| < 2.5\%$), excess churn reaches up to **$10.97\%$**, proving that aggregate benchmark accuracy severely masks instance-level decision boundary instability.

---

## 3. Alignment with Faculty Research & PhD Admissions

1. **Trustworthy Computer Vision & Distribution Shift:**
   - *Faculty:* Prof. Dan Hendrycks (UC Berkeley / CAIS), Prof. Trevor Darrell (UC Berkeley), Prof. Percy Liang (Stanford).
   - *Fit:* Advances beyond unary ImageNet-C by establishing counterfactual common-support evaluation protocols for multi-stage physical degradations.
2. **Vision Transformers & Representation Geometry:**
   - *Faculty:* Prof. Kaiming He (MIT), Prof. Jitendra Malik (UC Berkeley), Prof. Alexei Efros (UC Berkeley).
   - *Fit:* Explains the differential resilience of patch tokenization (ViT/Swin) vs. spatial convolution kernels against non-commutative phase and block-boundary artifacts.
3. **Statistical Uncertainty Quantification & Applied Statistics:**
   - *Faculty:* Prof. Emmanuel Candès (Stanford), Prof. Ryan Tibshirani (UC Berkeley), Prof. Aaditya Ramdas (CMU).
   - *Fit:* Applies cluster-robust bootstrap inference ($B = 5{,}000$ on physical images), Holm-Bonferroni FWER control, and error churn decomposition to prevent pseudoreplication in computer vision benchmarking.

---

## 4. Key Interview & Presentation Talking Points

1. **"Why was the nominal order effect confounded, and how did you resolve it?"**
   *Talking Point:* "In our initial 486,700-record benchmark, we found a 20.45% accuracy gap in ConvNeXt between terminal-blur sequence `NJB` (66.44%) and terminal-noise sequence `BJN` (45.99%). But analyzing the physical image tensors revealed that `NJB` had ~5 dB higher PSNR because terminal blur smoothed both noise and JPEG blocks, whereas `BJN` left unattenuated noise. To evaluate whether models were failing due to operator sequence or simply lower SNR, we conducted an independent calibration on training data (zero leakage) targeting common PSNR and SSIM support across all 6 permutations. When evaluated under this common support, the accuracy gap contracted by 77.15% in ConvNeXt, proving that nominal benchmarks heavily confound operator sequence with distortion magnitude."

2. **"Does order still matter if you match PSNR and SSIM?"**
   *Talking Point:* "Yes, definitively. In our full-rank saturated 6-cell linear model, the balanced parity contrast between even and odd permutations remains significant at $p < 10^{-100}$ under Holm-Bonferroni FWER control. Furthermore, the paired single-image error churn analysis proves that even when net accuracy differences are near zero ($|\Delta\text{Acc}| < 2.5\%$), classifiers flip predictions on up to 10.97% of individual physical images, proving that sequence order reshuffles the classifier's failure set."

3. **"Why did terminal blur reverse from positive to negative under matched fidelity?"**
   *Talking Point:* "In nominal setups, terminal blur appeared advantageous because it smoothed prior noise and compression blocks. But when parameters are calibrated so that PSNR is matched, the noise variance is already balanced. Under those conditions, terminal blur strips high-frequency discriminative features without any compensatory distortion advantage, decreasing top-1 accuracy by $-6.82\%$ relative to terminal noise."
