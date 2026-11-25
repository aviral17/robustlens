# Strategic Scientific Decision: Go / Conditional-Go / No-Go Protocol
## Phase III Steering Recommendation (Post-Audit Revised)

**Document Reference:** `research/phase3_cins/NEXT_RESEARCH_DECISION.md`  
**Framework:** Commutator-Induced Neural Susceptibility (CINS)  
**Authors:** RobustLens Research Initiative  
**Date:** November 2026 (Post-Audit Revised)  
**Status:** Official Research Steering Recommendation  

---

## 1. Executive Scientific Decision

Following comprehensive independent scientific audit of the Phase III preliminary reports and unit test suite:

$$\mathbf{DECISION:\;GO\;FOR\;10\text{-}IMAGE\;GPU\;FEASIBILITY\;VALIDATION}$$
$$\mathbf{DECISION:\;NOT\;YET\;GO\;FOR\;250\text{-}IMAGE\;CONFIRMATORY\;SCALE\text{-}UP}$$

The preliminary evidence confirms mathematical rigor, exact invariant preservation, and verified algorithmic acceleration ($> 10,000\times$ CPU speedup over brute-force). However, empirical neural-gradient predictive validity has not yet been demonstrated on genuine autograd gradients across a controlled image sample.

Therefore, full 250-image confirmatory scale-up is paused until the **10-image GPU engineering feasibility experiment** is executed and analyzed.

---

## 2. Gate Criteria: From Feasibility Validation to Confirmatory Scale-Up

To achieve an unconditional "GO" for the 250-image confirmatory experiment, the 10-image feasibility experiment must satisfy four quantitative gates:

```
                  ┌────────────────────────────────────────────────────────────┐
                  │        Phase III Quantitative Evaluation Gates             │
                  └─────────────────────────────┬──────────────────────────────┘
                                                │
         ┌──────────────────────────────┬───────┴──────────────────────┬──────────────────────────────┐
         │                              │                              │                              │
         ▼                              ▼                              ▼                              ▼
┌──────────────────┐          ┌──────────────────┐           ┌──────────────────┐           ┌──────────────────┐
│ Gate 1: Pearson  │          │ Gate 2: Error    │           │ Gate 3: Invariant│           │ Gate 4: Swin     │
│ Correlation      │          │ Monotonicity     │           │ Verification     │           │ Window Contrast  │
│ ρ ≥ 0.60         │          │ Error(0.8) >     │           │ Exact Tensor     │           │ Distinguish Token│
│ ConvNeXt/ResNet  │          │ Error(0.2)       │           │ Feeding (No JPEG)│           │ vs Pixel Effects │
└──────────────────┘          └──────────────────┘           └──────────────────┘           └──────────────────┘
```

1. **Gate 1 (Neural-Gradient Predictive Correlation):**  
   At intervention strength $\alpha = 0.20 \cdot \alpha_{\text{common}}$, the within-image Pearson correlation between predicted response $\hat{r}(\tau)$ and observed response $r(\tau)$ must achieve $\rho \ge 0.60$ for at least one convolutional architecture (ConvNeXt-Tiny or ResNet-50) on non-degenerate samples.
2. **Gate 2 (Error Scaling Monotonicity):**  
   Mean absolute error $\text{MAE}(\alpha)$ must increase monotonically with $\alpha \in \{0.2, 0.5, 0.8\} \times \alpha_{\text{common}}$, reflecting the theoretical expansion remainder.
3. **Gate 3 (Pure Invariant Verification):**  
   All forward passes must feed floating-point tensors directly to pretrained models with zero intermediate JPEG compression or serialization, ensuring that spectral invariants remain strictly preserved at model input.
4. **Gate 4 (Architectural Differentiation):**  
   The experiment must empirically measure whether Vision Transformers (ViT, Swin) exhibit higher approximation error or distinct spatial response spectra compared to ConvNets, providing concrete evidence for architectural hypotheses.

---

## 3. Scope of the 10-Image GPU Feasibility Experiment

- **Dataset:** 10 diverse development images with ground-truth ImageNet-1K labels.
- **Architectures:** Original timm checkpoints:
  - `resnet50.a1_in1k`
  - `convnext_tiny.in12k_ft_in1k`
  - `vit_base_patch16_224.augreg_in21k_ft_in1k`
  - `swin_tiny_patch4_window7_224.ms_in1k`
- **Translations:** 32 uniformly sampled circular translation positions (identical coordinates across models and strengths).
- **Intervention Strengths:** 3 levels ($\alpha \in \{0.2, 0.5, 0.8\} \times \alpha_{\text{common}}$).
- **Endpoint Budget:** $10 \text{ images} \times 4 \text{ models} \times 3 \text{ alphas} \times 32 \text{ shifts} \times 2 \text{ endpoints} = \mathbf{7,680 \text{ forward passes}}$.
- **Gradient Computation:** Autograd backpropagation $\nabla_x m(c)$ w.r.t. unnormalized input pixels.
- **Metrics Logged:**
  - True-class max-competitor margin $m(x) = z_y - \max_{j \ne y} z_j$.
  - Fixed anchor-competitor margin $m_{\text{fixed}}(x) = z_y - z_{j^*(c)}$.
  - Smooth log-sum-exp margin $m_{\text{LSE}}(x; T=1.0)$.
  - Pearson $\rho$, Spearman $\rho_s$, MAE, RMSE, $R^2$, and sign agreement.

---

## 4. Execution Vehicle & Instructions

The feasibility experiment is packaged in:
- Standalone runner script: `research/phase3_cins/cins_phase3_kaggle_runner.py`
- Executable Kaggle/Colab notebook: `research/phase3_cins/cins_phase3_kaggle_experiment.ipynb`

Upon successful execution of this 7,680-pass feasibility benchmark on GPU, results will be returned for independent statistical analysis prior to any decision on the 250-image confirmatory study.
