# 🔍 RobustLens: Vision Architecture Robustness & Non-Commutative Compound Corruptions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23483939.svg)](https://doi.org/10.5281/zenodo.23483939)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)
[![Benchmark Scale](https://img.shields.io/badge/Scale-973%2C400%20Evaluations-emerald.svg)](results/order_effect_study/full/outputs/all_order_predictions.csv.gz)

**RobustLens** is an empirical statistical research benchmark and interactive web application evaluating the out-of-distribution (OOD) robustness, predictive uncertainty, and non-commutative operator vulnerabilities of four canonical vision architectures: **ResNet-50**, **ConvNeXt-Tiny**, **ViT-Base/16**, and **Swin-Tiny**.

Evaluated across **973,400 total forward passes** on the official **FastAI Imagenette-v2** validation partition ($N = 3{,}925$ independent physical images with deterministically seed-matched noise realizations).

---

## 📄 Final Research Package (October 2026)

The definitive academic research package is cataloged in [`reports/final_research_package/`](reports/final_research_package/):
- **Primary Research Paper (PDF, 5 Pages):** [`reports/final_research_package/RobustLens_Final_Paper.pdf`](reports/final_research_package/RobustLens_Final_Paper.pdf)  
  *"Beyond Six-Order Accuracy Gaps: A Paired, Distortion-Aware Analysis of Compound-Corruption Permutations in Vision Classifiers"*
- **Primary Paper Markdown:** [`reports/final_research_package/RobustLens_Final_Paper.md`](reports/final_research_package/RobustLens_Final_Paper.md)
- **Technical Supplement & Proofs:** [`reports/final_research_package/Technical_Supplement.md`](reports/final_research_package/Technical_Supplement.md)
- **Originality & Prior-Art Audit:** [`reports/final_research_package/ORIGINALITY_AND_LIMITATIONS_AUDIT.md`](reports/final_research_package/ORIGINALITY_AND_LIMITATIONS_AUDIT.md)
- **Reproducibility Guide:** [`reports/final_research_package/REPRODUCIBILITY.md`](reports/final_research_package/REPRODUCIBILITY.md)
- **High-Resolution Figures (300 DPI):** [`reports/final_research_package/figures/`](reports/final_research_package/figures/)
- **Analysis Data Artifacts:** [`reports/final_research_package/analysis/`](reports/final_research_package/analysis/)

---

## 🏆 Key Experimental Discoveries

### 1. The Terminal Smoothing Confound in Nominal Benchmarks
In standard nominal evaluations (applying canonical ImageNet-C fixed parameters across sequences), accuracy varies by up to **$21.55\%$ in ResNet-50** and **$20.45\%$ in ConvNeXt-Tiny** across the six permutations on identical physical images. However, we discovered that these gaps are dominated by terminal low-pass filtering: sequences ending in blur (`NJB`, `JNB`) smooth out prior JPEG $8 \times 8$ DCT block boundaries, conferring up to a $5.0\text{ dB}$ PSNR advantage over sequences ending in JPEG (`NBJ`, `BNJ`).

### 2. Approximate-Fidelity Calibration Contracts Gaps by 63.18%
Calibrating parameters on an independent training partition ($N_{\text{cal}} = 50$, zero validation leakage) across five target distortion bins revealed that **prespecified joint matching criteria ($\pm 0.75\text{ dB}$ PSNR, $\pm 0.035$ SSIM) were not met over the tested discrete grid** (PSNR RMSE = $0.943\text{ dB}$, SSIM RMSE = $0.1380$).  
Nonetheless, equalizing distortion on common support produced a dramatic contraction of cross-sequence accuracy gaps:
- **ConvNeXt-Tiny:** Gap contracts from **$20.45\%$ down to $4.67\%$** (**$77.15\%$ contraction**).
- **ResNet-50:** Gap contracts from **$21.55\%$ down to $5.90\%$** (**$72.62\%$ contraction**).
- **ViT-Base/16:** Gap contracts from **$20.37\%$ down to $10.03\%$** (**$50.75\%$ contraction**).
- **Swin-Tiny:** Gap contracts from **$17.13\%$ down to $11.68\%$** (**$31.80\%$ contraction**).
- **Overall Mean:** Gap contracts from **$18.43\%$ down to $6.79\%$** (**$63.18\%$ contraction**).

### 3. Saturated 6-Cell Linear Decomposition with Balanced Parity
We formulated an orthogonal saturated 6-cell linear decomposition ($\det(X) = -6$, full rank) spanning all sequence means. Clustered bootstrap inference ($B = 5{,}000$ on physical images, Holm FWER-controlled) showed:
- In nominal evaluations, terminal blur conferred a $+12.27\%$ accuracy boost.
- Under calibrated fidelity, the terminal blur effect reversed to **$-6.82\%$** ($p < 10^{-100}$): when PSNR is balanced, terminal blur strips high-frequency discriminative features.
- The balanced cyclic-versus-acyclic parity contrast remained statistically significant ($+0.93\%$, $p < 10^{-100}$), confirming that cyclic operator structure independently affects representations.

### 4. The Paired Error Churn Decoupling Phenomenon
Microscopic paired single-image analysis on matched observations proved the exact identity $\text{Churn} \equiv |\Delta\text{Acc}| + 2\min(G, L)$. Condition pairs with near-zero accuracy differences exhibited substantial excess churn:
- In ConvNeXt under calibrated fidelity, `NJB` (56.46%) and `JNB` (56.47%) differed by only **$0.01\%$**, yet exhibited **$8.22\%$ excess churn** ($1{,}613$ flipped predictions).
- In Swin-Tiny, `BJN` (61.23%) and `JBN` (62.12%) differed by **$0.89\%$**, with **$8.84\%$ excess churn**.
- Aggregate accuracy masks severe single-image failure set reshuffling.

---

## 📊 Summary Leaderboard & Gap Contraction

| Architecture | Clean Acc | Matched Mean Acc | Nominal Gap | Matched Gap | Gap Contraction | Mean Churn | Excess Churn Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ConvNeXt-Tiny** | 87.13% | **58.11%** | 20.45% | **4.67%** | **77.15%** | 10.93% | **75.5%** |
| **Swin-Tiny** | 86.04% | 56.27% | **17.13%** | 11.68% | 31.80% | 11.92% | 56.9% |
| **ViT-Base/16** | 80.25% | 53.31% | 20.37% | 10.03% | 50.75% | 13.40% | 67.0% |
| **ResNet-50** | **87.36%** | 48.47% | 21.55% | 5.90% | 72.62% | 13.10% | 70.9% |

---

## 🚀 Quick Start

### Python ML Pipeline (`uv`)
```bash
cd ml
uv sync --extra dev
uv run pytest tests/test_order_decomposition.py -v
uv run python scripts/run_final_order_decomposition_analysis.py
uv run python scripts/generate_final_package_figures.py
uv run python scripts/generate_final_paper_pdf.py
```

### Next.js Interactive Dashboard
```bash
cd apps/web
pnpm install
pnpm build
pnpm dev
```
Explore the interactive visual dashboard locally at `http://localhost:3000`.

---

## 📌 Citation & Permanent DOI

If you use RobustLens code, methodology, or benchmark records in your research, please cite:

```bibtex
@article{choudhary2026robustlens,
  title   = {Order Matters Beyond Accuracy: Paired Prediction Analysis of Compound Image Corruptions},
  author  = {Choudhary, Aviral},
  year    = {2026},
  month   = {October},
  doi     = {10.5281/zenodo.23483939},
  url     = {https://doi.org/10.5281/zenodo.23483939},
  version = {1.0},
  note    = {RobustLens Research Benchmark Release}
}
```

---

## 📚 Selected Citations

1. Hendrycks, D., & Dietterich, T. (2019). Benchmarking Neural Network Robustness to Common Corruptions. *ICLR*.
2. Liu, Z., et al. (2022). A ConvNet for the 2020s. *CVPR*.
3. Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*.
4. Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV*.
5. Milani Fard, M., et al. (2016). Churn Reduction via Distantly Supervised Models. *NeurIPS*.
6. Goyal, S., et al. (2022). Dual Corruptions: Evaluating Vision Models Under Composed Shifts. *ICCP*.
7. Li, Y., et al. (2025). R-Bench: Benchmarking Multi-Stage Physical Degradation in Multimodal LLMs. *IEEE JSTSP*.
