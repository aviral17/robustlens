# RobustLens: Proposed Research Portfolio & CV Entries

**Target Use Case:** US Computer Vision / Machine Learning PhD Applications (Fall 2027)  
**Author:** Aviral (Avi)  

---

### Option 1: Curriculum Vitae (CV) — 3-Bullet Academic Format

**RobustLens: Statistical Evaluation of Vision Architecture Robustness Under Distribution Shift**  
*Independent Research Project | Python, PyTorch, timm, SciPy, Next.js, TypeScript*  
- Formulated an empirical robustness evaluation pipeline eliminating pseudoreplication in corruption benchmarks via image-level clustered paired bootstrap resampling ($B = 10{,}000$) and Monte Carlo paired sign-flip permutation tests ($R = 20{,}000$).
- Evaluated 251,200 forward passes across 4 representative architectures (ConvNeXt, Swin, ResNet, ViT) and 16 conditions on the official FastAI Imagenette-v2 validation set ($N = 3{,}925$); proved that ConvNeXt-Tiny achieves a statistically significant $+5.03\%$ lead under distribution shift over ResNet-50 ($p < 0.0001$) despite identical clean performance (asymptotic McNemar $p = 0.6402$).
- Conducted a large-scale compound corruption order-effects study (486,700 forward passes across 6 sequence permutations and 5 severities), demonstrating that transformation application sequence induces a 17.13% to 21.55% accuracy gap between best (`NJB`) and worst (`BNJ`) orders, remaining strongly associated with correctness (clustered Wald $p \ll 10^{-100}$) after conditioning on continuous PSNR and SSIM.
- Analyzed calibration degradation and conformal prediction exchangeability breakdown under covariate shift; engineered a public interactive Next.js analytics dashboard and authored a complete academic manuscript with reproducibility pipeline.

---

### Option 2: Curriculum Vitae (CV) — 2-Bullet Condensed Format

**RobustLens: Empirical Robustness & Uncertainty Benchmarking for Vision Models**  
- Developed an end-to-end evaluation pipeline over 737,900 total forward passes across 4 architectures; implemented cluster paired bootstrap CIs and Monte Carlo sign-flip permutation tests ($R = 20{,}000$) with Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR) multiplicity control.
- Demonstrated that modernized depthwise CNNs (ConvNeXt) outperform hierarchical window transformers (Swin) and classical residual CNNs under corruption shifts; proved via clustered ANCOVA ($N_{\text{clusters}} = 3{,}925$) that compound corruption order significantly affects classification accuracy (17.13%–21.55% maximum sequence gap) independent of continuous image distortion metrics.

---

### Option 3: Statement of Purpose (SoP) / Research Statement Snippet

> *"To explore empirical robustness and uncertainty quantification, I conceived and implemented **RobustLens**, a benchmark evaluating architectural inductive biases under controlled distribution shifts across 251,200 forward passes on the official FastAI Imagenette validation split ($N = 3{,}925$). Recognizing that standard corruption benchmarks introduce pseudoreplication by treating repeated evaluations of the same source image as independent trials, I formulated image-level clustered paired bootstrap confidence intervals alongside Monte Carlo paired sign-flip permutation tests ($R = 20{,}000$). My findings revealed a clean vs. shifted reversal: while ResNet-50 and ConvNeXt-Tiny achieved indistinguishable clean accuracy ($p = 0.6402$), ConvNeXt established a $+5.03\%$ advantage under distribution shift ($p < 0.0001$). Furthermore, extending this to compound corruptions (486,700 forward passes), I discovered that the sequence of corruption application induces massive classification shifts (maximum sequence accuracy gaps of 17.13% to 21.55% across architectures; clustered Wald $p \ll 10^{-100}$) even after conditioning on continuous PSNR and SSIM. This project strengthened my commitment to pursuing doctoral research at the intersection of out-of-distribution generalization, uncertainty quantification, and rigorous statistical methodology in computer vision."*

---

### Option 4: GitHub / Portfolio Summary (One Paragraph)

> **RobustLens** is an open-source research benchmark and interactive dashboard evaluating the robustness and predictive uncertainty of four canonical vision architectures (ResNet-50, ConvNeXt-Tiny, ViT-Base/16, Swin-Tiny) across single and compound corruptions (over 737,000 forward passes on $N=3{,}925$ independent validation images). It replaces naive un-clustered pooling with image-level clustered bootstrap confidence intervals ($B=10{,}000$) and Monte Carlo paired sign-flip permutation tests ($R=20{,}000$) with Holm-Bonferroni and Benjamini-Hochberg corrections, alongside cluster-robust ANCOVA, Expected Calibration Error (ECE, $M=10$ bins), and split conformal prediction exchangeability diagnostics.
