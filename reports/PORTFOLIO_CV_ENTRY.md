# RobustLens: Research Portfolio, CV, and Statement of Purpose Entries

**Target Use Case:** US Computer Vision / Machine Learning PhD Applications (Fall 2027) & Faculty Review  
**Author:** Avi  
**Date:** October 2026  
**Artifact Path:** `reports/PORTFOLIO_CV_ENTRY.md`  

---

### Option 1: Curriculum Vitae (CV) — 3-Bullet Academic Format

**Order Has Consequences: Total Effects and Fidelity-Conditioned Prediction Instability under Compound Image Corruptions (RobustLens)**  
*Independent Research Project | Python, PyTorch, timm, SciPy, Next.js, TypeScript*  
- Formulated a two-regime empirical benchmark (973,400 total forward passes; 486,700 per regime) evaluating the full symmetric group of 3 canonical image degradations ($3! = 6$ permutations $\times 5$ severities) across 4 architectures (ResNet-50, ConvNeXt-Tiny, ViT-Base/16, Swin-Tiny) on FastAI Imagenette-v2 ($N = 3{,}925$ validation images with seed-matched noise realizations).
- Discovered that massive nominal-order accuracy gaps (up to $21.55\%$ in ResNet and $20.45\%$ in ConvNeXt) are heavily confounded by terminal operator smoothing; developed an independent training calibration protocol ($N_{\text{cal}} = 50$) targeting common-support PSNR/SSIM, demonstrating that equalizing distortion contracts cross-sequence gaps by $77.15\%$ in ConvNeXt (down to $4.67\%$) and $72.62\%$ in ResNet (down to $5.90\%$), while transparently reporting that prespecified calibration tolerances were not met over the tested discrete grid.
- Formulated a full-rank saturated six-cell decomposition with direct marginal contrasts and a balanced parity contrast ($\det(X) = -6$), alongside a complete 5-state joint prediction space decomposition; proved that Predicted-Label Churn ($34.74\%–40.78\%$) is more than double Correctness Churn ($12.34\%–16.93\%$) due to Error Migration ($22.40\%–23.85\%$, where models predict different wrong classes across orders), with sequence pairs showing up to $10.97\%$ excess churn despite near-identical aggregate accuracy ($p < 10^{-100}$, Holm FWER-controlled).
- Engineered a production Next.js analytics dashboard with interactive single-image shift exploration, live error churn matrices, and authored a 5-page camera-ready research manuscript with full reproducibility pipelines.

---

### Option 2: Curriculum Vitae (CV) — 2-Bullet Condensed Format

**Order Has Consequences: Total Effects and Fidelity-Conditioned Prediction Instability (RobustLens)**  
- Designed and executed a 973,400-evaluation paired benchmark across 4 vision architectures to separate the total operational effect of multi-stage degradation order from fidelity-conditioned effects under approximate common support.
- Proved that nominal accuracy gaps of up to $21.55\%$ are dominated by terminal low-pass filter smoothing; demonstrated that distortion balancing contracts gaps by $63.18\%$ overall, while 5-state prediction space analysis proves that sequence permutations drive $34.74\%–40.78\%$ label churn and massive error migration (~23%) despite near-invariant aggregate accuracy ($p < 10^{-100}$, Holm FWER-controlled).

---

### Option 3: Statement of Purpose (SoP) / Research Statement Snippet

> *"To interrogate how architectural inductive biases govern model reliability under cascading real-world distribution shifts, I designed and conducted **RobustLens**, a benchmark evaluating non-commutative image degradation across 973,400 forward passes on four canonical vision architectures (ResNet-50, ConvNeXt-Tiny, ViT-Base/16, Swin-Tiny) on FastAI Imagenette-v2 ($N = 3{,}925$). While standard benchmarks evaluate corruptions as isolated unary operations, real-world sensing involves sequential processing pipelines whose underlying mathematical operators do not commute ($\mathcal{T}_A \circ \mathcal{T}_B \neq \mathcal{T}_B \circ \mathcal{T}_A$). Evaluating the full symmetric group of Gaussian Noise, Defocus Blur, and JPEG Compression ($3! = 6$ permutations), I found that standard nominal evaluations exhibit massive cross-sequence accuracy gaps of up to $21.55\%$. However, through systematic spectral and distortion analysis, I recognized that nominal benchmarks conflate architectural order robustness with physical distortion imbalance: sequences where blur succeeds JPEG smooth out DCT blocking, conferring a $\sim 5\text{ dB}$ PSNR advantage. To decouple these factors, I developed an independent pre-calibration optimization targeting common PSNR and SSIM support. Equalizing distortion contracted accuracy gaps by $77.15\%$ in ConvNeXt (dropping to $4.67\%$) and $72.62\%$ in ResNet (dropping to $5.90\%$). Crucially, using a full-rank saturated six-cell decomposition with a balanced parity contrast and paired single-image error churn analysis ($\text{Churn} \equiv |\Delta\text{Acc}| + 2\min(G, L)$), I proved that aggregate accuracy severely understates instability: condition pairs with near-zero accuracy difference exhibited up to $10.97\%$ excess error churn, proving that operator sequence reshuffles classifier failure sets. This project solidified my research objective: developing rigorous statistical frameworks and distortion-aware methodologies for trustworthy vision under complex, sequential distribution shifts."*

---

### Option 4: GitHub / Portfolio Summary (One Paragraph)

> **RobustLens** is an open-source statistical research benchmark evaluating the non-commutativity of sequential digital image corruptions across four prominent vision architectures (ResNet-50, ConvNeXt-Tiny, ViT-Base/16, Swin-Tiny) spanning 973,400 evaluated predictions on the official FastAI Imagenette-v2 partition. It provides the first systematic comparison between nominal-order evaluations (where sequence gaps reach $21.55\%$ due to terminal low-pass filtering) and calibrated common-support evaluations (where distortion balancing contracts gaps by $63.18\%$ overall and up to $77.15\%$ in ConvNeXt). The codebase features a full-rank saturated six-cell decomposition with a balanced parity contrast between even and odd permutations, paired single-image error churn analysis, clustered bootstrap resampling ($B = 5{,}000$ on physical images) with Holm-Bonferroni FWER control, and an interactive Next.js dashboard.
