# 🔍 RobustLens: Vision Architecture Robustness & Non-Commutative Compound Corruptions

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)
[![Benchmark Scale](https://img.shields.io/badge/Scale-1%2C286%2C900%20Evaluations-emerald.svg)](results/)

**RobustLens** is an empirical statistical research framework and interactive evaluation suite investigating operator non-commutativity, paired prediction transitions, matched global image fidelity, spatial-frequency/structural fingerprints, and vision-model decision margins across four canonical computer vision architectures: **ResNet-50**, **ConvNeXt-Tiny**, **ViT-Base/16**, and **Swin-Tiny**.

Evaluated across **1,286,900 total forward passes** across two comprehensive research phases:
1. **November 2026 Matched-Fidelity Confirmatory & Replication Study (313,500 passes):** Evaluates $N = 3{,}825$ held-out Imagenette validation images ($290{,}700$ primary passes) across 3 calibrated common-support tiers (Mild: 27.20 dB / 0.810; Moderate: 23.80 dB / 0.650; Severe: 21.80 dB / 0.500) plus an independent $N = 300$ ImageNet-V2 external replication cohort ($22{,}800$ passes).
2. **October 2026 Foundational Compound-Corruption Benchmark (973,400 passes):** Evaluates $N = 3{,}925$ validation images across Nominal and Approximate-Fidelity regimes across all 6 sequence permutations.

---

## 📄 Research Publications & Academic Deliverables

### 🔬 Canonical Scientific Manuscript (November 2026)
- **Title:** *Spatial-Frequency and Structural Distortion Fingerprints Beyond Global Image Fidelity: Predicting Vision-Model Margins under Compound Image Corruption*
- **Authors:** **Suryakanth V. Gangashetty**$^1$ and **Aviral Choudhary**$^2$  
  $^1$Department of Computer Science and Engineering, Koneru Lakshmaiah Education Foundation, India  
  $^2$Independent Researcher, India
- **Canonical Manuscript (PDF with Vector Figures, 8 Pages):** [`apps/web/public/downloads/RobustLens_Academic_Manuscript.pdf`](apps/web/public/downloads/RobustLens_Academic_Manuscript.pdf)
- **Comprehensive Research Report (PDF, 10 Pages):** [`reports/final_research_package/RobustLens_Final_Research_Report.pdf`](reports/final_research_package/RobustLens_Final_Research_Report.pdf)
- **Editable Academic Manuscript (DOCX):** [`reports/final_research_package/RobustLens_Academic_Manuscript.docx`](reports/final_research_package/RobustLens_Academic_Manuscript.docx)
- **LaTeX Source (.tex):** [`apps/web/public/downloads/RobustLens_Academic_Manuscript.tex`](apps/web/public/downloads/RobustLens_Academic_Manuscript.tex)
- **Complete LaTeX Source & Vector Figures Archive (.zip):** [`apps/web/public/downloads/RobustLens_Matched_Fidelity_Final_Source.zip`](apps/web/public/downloads/RobustLens_Matched_Fidelity_Final_Source.zip)
- **Reproducibility Manifest & SHA-256 Hashes:** [`results/matched_fidelity_study/REPRODUCIBILITY_MANIFEST.json`](results/matched_fidelity_study/REPRODUCIBILITY_MANIFEST.json)

### 📚 Foundational Benchmark Paper (October 2026)
- **Title:** *Order Matters Beyond Accuracy: Paired Prediction Analysis of Compound Image Corruptions*
- **Authors:** **Suryakanth V. Gangashetty**$^1$ and **Aviral Choudhary**$^2$  
  $^1$Department of Computer Science and Engineering, Koneru Lakshmaiah Education Foundation, India  
  $^2$Independent Researcher, India
- **Foundational Paper (PDF, 6 Pages):** [`apps/web/public/downloads/RobustLens_Final_Paper.pdf`](apps/web/public/downloads/RobustLens_Final_Paper.pdf)
- **Foundational LaTeX Source (.tex):** [`apps/web/public/downloads/RobustLens_Foundational_Paper.tex`](apps/web/public/downloads/RobustLens_Foundational_Paper.tex)
- **Complete Research Package (.zip):** [`apps/web/public/downloads/RobustLens_Research_Package.zip`](apps/web/public/downloads/RobustLens_Research_Package.zip)
- **Technical Supplement & Proofs:** [`reports/final_research_package/Technical_Supplement.md`](reports/final_research_package/Technical_Supplement.md)

---

## 🏆 Key Experimental Discoveries (Across Both Research Phases)

### 1. Spatial-Frequency & Structural Fingerprints Add Significant Out-of-Sample Predictive Value (November 2026)
When global image fidelity is strictly calibrated across sequences, five spatial-frequency and structural fingerprints ($\Delta E_{\text{edge}}$, $E_{\text{low}}$, $E_{\text{high}}$, $B_{\text{JPEG}}$, $\sigma_{\text{local SSIM}}$) provide statistically significant out-of-sample improvements in predicting logit margins across all four architectures:
- **ResNet-50:** $\Delta R^2 = +0.0199$ (95% CI $[0.0163, 0.0234]$, $p_{\text{adj}} \le 0.0004$)
- **ConvNeXt-Tiny:** $\Delta R^2 = +0.0177$ (95% CI $[0.0142, 0.0212]$, $p_{\text{adj}} \le 0.0004$)
- **ViT-Base/16:** $\Delta R^2 = +0.0094$ (95% CI $[0.0069, 0.0120]$, $p_{\text{adj}} \le 0.0004$)
- **Swin-Tiny:** $\Delta R^2 = +0.0118$ (95% CI $[0.0089, 0.0146]$, $p_{\text{adj}} \le 0.0004$)

### 2. Sobel Edge-Energy is the Strongest Predictive Contributor (November 2026)
Feature ablation experiments identify Sobel edge-energy change ($\Delta E_{\text{edge}}$) as the single largest contributor to margin prediction across all architectures, followed by Fourier residual-energy terms ($E_{\text{low}}, E_{\text{high}}$). Full sequence permutations and continuous fingerprints add complementary, non-redundant predictive structure.

### 3. Independent External Replication on ImageNet-V2 (November 2026)
On an independent 300-image ImageNet-V2 cohort (22,800 forward passes), **ViT-Base/16** replicates with a statistically significant pooled out-of-sample $R^2$ gain after step-down Holm correction ($\Delta R^2 = +0.0371$, 95% CI $[0.0065, 0.0675]$, $p_{\text{adj}} = 0.0340$). For ResNet-50, ConvNeXt-Tiny, and Swin-Tiny, point estimates remain positive (+0.0115 to +0.0212), but their bootstrap intervals cross zero ($p_{\text{adj}} > 0.05$). Statistically significant external $R^2$ improvement is claimed only for ViT-Base.

### 4. Terminal Operator Effects & Order Sensitivity in Nominal Benchmarks (October 2026)
In the tested nominal regime, sequences ending in blur achieved higher average accuracy than sequences ending in noise (e.g. +17.69% boost across architectures), partly by smoothing prior high-frequency artifacts. Crucially, terminal operator identity alone does not explain all sequence variation: substantial accuracy differences persist even among sequences sharing the identical terminal operator (e.g., `NBJ` at 58.90% vs. `BNJ` at 42.88% in ConvNeXt-Tiny—a 16.02 percentage point divergence despite both terminating in JPEG). Sequence processing order remains a critical independent determinant of model resilience.

### 5. Descriptive Gap Contraction vs. Calibrated Common Support
In the October approximate-fidelity benchmark, equalizing shared (PSNR, SSIM) bins contracted the cross-sequence accuracy range of the four-model mean by a descriptive **63.18%** (from 18.43% down to 6.79%, with architecture-specific contractions averaging 58.08%). Because this exploratory regime did not satisfy prescribed tolerances (PSNR RMSE 0.943 dB, SSIM RMSE 0.1380), the November confirmatory study instituted a rigorously validated common-support protocol (18/18 configurations validated within $\pm 0.75$ dB PSNR, $\pm 0.035$ SSIM).

### 6. The Paired Error Churn Decoupling Phenomenon (October 2026)
Paired single-image analysis proved the exact identity $\text{Churn} \equiv |\Delta\text{Acc}| + 2\min(G, L)$. Condition pairs with near-zero accuracy differences exhibited substantial excess churn:
- In ConvNeXt under calibrated fidelity, `NJB` (56.46%) and `JNB` (56.47%) differed by only **$0.01\%$**, yet exhibited **$8.22\%$ excess churn** ($1{,}613$ flipped predictions).
- In Swin-Tiny, `BJN` (61.23%) and `JBN` (62.12%) differed by **$0.89\%$**, with **$8.84\%$ excess churn**.
- Aggregate accuracy masks severe single-image failure set reshuffling and error migration ($E_{00\_\text{diff}}$).

---

## 📊 Summary Leaderboard & Performance Metrics

| Architecture | Clean Acc | Matched Mean Acc | Nominal Gap | Matched Gap | Descriptive Contraction | Mean Churn | Excess Churn Ratio |
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
uv run pytest tests/ -v
```

### Next.js Interactive Dashboard
```bash
cd apps/web
pnpm install
pnpm build
pnpm dev
```
Explore the interactive visual dashboard locally at `http://localhost:3000` or on the live production deployment at [robustlens.vercel.app](https://robustlens.vercel.app).

---

## 📌 Academic Citations

If you use RobustLens code, empirical datasets, or statistical protocols in your research, please cite:

```bibtex
@article{gangashetty2026spatialfrequency,
  title   = {Spatial-Frequency and Structural Distortion Fingerprints Beyond Global Image Fidelity: Predicting Vision-Model Margins under Compound Image Corruption},
  author  = {Gangashetty, Suryakanth V. and Choudhary, Aviral},
  year    = {2026},
  month   = {November},
  journal = {RobustLens Research Initiative Working Paper},
  note    = {Confirmatory study N=3,825 held-out Imagenette-v2 images, 290,700 primary passes + ImageNet-V2 replication}
}

@article{gangashetty2026ordermatters,
  title   = {Order Matters Beyond Accuracy: Paired Prediction Analysis of Compound Image Corruptions},
  author  = {Gangashetty, Suryakanth V. and Choudhary, Aviral},
  year    = {2026},
  month   = {October},
  journal = {RobustLens Research Initiative Working Paper},
  version = {1.0},
  note    = {Foundational compound-corruption benchmark: 973,400 forward passes on 3,925 Imagenette-v2 images}
}
```

---

## 📚 Selected References

1. Hendrycks, D., & Dietterich, T. (2019). Benchmarking Neural Network Robustness to Common Corruptions. *ICLR*.
2. Liu, Z., et al. (2022). A ConvNet for the 2020s. *CVPR*.
3. Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*.
4. Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV*.
5. Milani Fard, M., et al. (2016). Churn Reduction via Distantly Supervised Models. *NeurIPS*.
6. Goyal, S., et al. (2022). Dual Corruptions: Evaluating Vision Models Under Composed Shifts. *ICCP*.
7. Li, Y., et al. (2025). R-Bench: Benchmarking Multi-Stage Physical Degradation in Multimodal LLMs. *IEEE JSTSP*.
