# Phase III Experimental Protocol: CINS 10-Image GPU Feasibility Validation

**Document Reference:** `research/phase3_cins/EXPERIMENT_PROTOCOL.md`  
**Framework:** Commutator-Induced Neural Susceptibility (CINS)  
**Authors:** RobustLens Research Initiative  
**Date:** November 2026 (Post-Audit Revised)  
**Status:** Approved Experimental Specification  

---

## 1. Experimental Objectives

The 10-image GPU engineering feasibility experiment evaluates whether first-order 2D circular cross-correlation response maps:
$$\hat{r}(\tau) = \alpha \langle \nabla_x m(c), d_\tau \rangle$$
predict the spatial distribution of actual neural decision responses:
$$r(\tau) = m\left(c + \frac{1}{2}\alpha d_\tau\right) - m\left(c - \frac{1}{2}\alpha d_\tau\right)$$
across 10 diverse development images and four canonical vision architectures under spectrally invariant compound-corruption residuals.

---

## 2. Model Specifications & Checkpoint Continuity

To maintain exact continuity with RobustLens Phase II, all evaluations use official pretrained weights from the `timm` library:

| Architecture | Canonical `timm` Model Identifier | Preprocessing Normalization | Input Resolution | Architecture Family |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-50** | `resnet50.a1_in1k` | Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]` | $224 \times 224$ | Canonical Residual ConvNet |
| **ConvNeXt-Tiny** | `convnext_tiny.in12k_ft_in1k` | Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]` | $224 \times 224$ | Modernized Depthwise ConvNet |
| **ViT-Base/16** | `vit_base_patch16_224.augreg_in21k_ft_in1k` | Mean: `[0.5, 0.5, 0.5]`, Std: `[0.5, 0.5, 0.5]` (AugReg) | $224 \times 224$ | Global Multi-Head Attention |
| **Swin-Tiny** | `swin_tiny_patch4_window7_224.ms_in1k` | Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]` | $224 \times 224$ | Shifted-Window Hierarchical ViT |

### Mandatory Execution Directives:
1. All models must be set to evaluation mode: `model.eval()`.
2. Model parameters must be frozen: `param.requires_grad = False`.
3. Model data configurations must be resolved directly via `timm.data.resolve_model_data_config(model)` to prevent normalization discrepancies.
4. Model weights SHA-256 hashes must be recorded in the execution manifest.

---

## 3. Input Gradient Autograd & Tensor Integrity Protocol

### 3.1 Gradient w.r.t. Floating-Point RGB Input Pixels
Gradients must be computed with respect to the unnormalized input tensor $x_{\text{pixel}} \in [0, 1]^{1 \times 3 \times H \times W}$:

```python
x_pixel = torch.tensor(c_np, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
x_pixel.requires_grad_(True)

# Normalization applied inside differentiable graph
x_norm = (x_pixel - mean_tensor) / std_tensor
logits = model(x_norm)

# Margin computation
margin = logits[0, target_class] - torch.max(torch.cat([logits[0, :target_class], logits[0, target_class+1:]]))

# Autograd backpropagation
grad = torch.autograd.grad(margin, x_pixel)[0].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
```
*Verification:* This correctly propagates gradients through the channel normalization operation $\frac{\partial x_{\text{norm}}}{\partial x_{\text{pixel}}} = \frac{1}{\text{std}}$.

### 3.2 Pure Tensor Transport (Zero Serialization Distortion)
All endpoints $x_+(\tau)$ and $x_-(\tau)$ must be constructed as direct floating-point arrays and converted into PyTorch tensors.
- **PROHIBITED:** Re-encoding into JPEG, PNG, WEBP, or base64 strings after constructing the intervention.
- **GUARANTEE:** The input tensor received by `model.forward()` contains the exact floating-point values $c \pm \frac{1}{2}\alpha d_\tau$, preserving all spectral and energy invariants to machine precision.

---

## 4. Sampling Design & Endpoint Budget

- **Image Sample:** 10 diverse development images with known ImageNet-1K labels.
- **Degradation Parameters:** Fixed canonical setting ($\sigma = 0.08, r = 2, q = 75$) with coupled pseudo-random noise realization $\epsilon$.
- **Translation Coordinates ($\mathcal{T}$):** Exactly 32 uniformly spaced 2D circular translation positions across the $224 \times 224$ grid:
  $$\mathcal{T} = \left\{ (y, x) \;\middle|\; y \in \{0, 28, 56, 84, 112, 140, 168, 196\}, \; x \in \{0, 56, 112, 168\} \right\}$$
  *Rule:* The same 32 translation coordinates are evaluated across all four models and all three intervention strengths.
- **Feasible Intervention Strengths:**
  $$\alpha \in \{0.20 \cdot \alpha_{\text{common}}, \; 0.50 \cdot \alpha_{\text{common}}, \; 0.80 \cdot \alpha_{\text{common}}\}$$
- **Endpoints Per Translation:** 2 endpoints ($x_+$ and $x_-$).

### Exact Computation Budget:
$$\text{Endpoint Forward Passes} = 10 \text{ images} \times 4 \text{ models} \times 3 \text{ alphas} \times 32 \text{ shifts} \times 2 \text{ endpoints} = \mathbf{7,680}$$
$$\text{Anchor Gradient Passes} = 10 \text{ images} \times 4 \text{ models} = \mathbf{40 \text{ autograd passes}}$$
$$\text{Anchor Baseline Forward Passes} = 10 \text{ images} \times 4 \text{ models} = \mathbf{40 \text{ forward passes}}$$

---

## 5. Metrics & Statistical Reporting

For each architecture and intervention strength, the runner exports:
1. **Decision Margins:**
   - True-class max-competitor margin: $m(x) = z_{y^*} - \max_{j \ne y^*} z_j$.
   - Fixed anchor-competitor margin: $m_{\text{fixed}}(x) = z_{y^*} - z_{j^*(c)}$.
   - Smooth log-sum-exp margin: $m_{\text{LSE}}(x; T=1.0)$.
2. **Correlation & Goodness of Fit:**
   - Within-image Pearson correlation $\rho(\hat{r}, r)$ and p-value.
   - Within-image Spearman rank correlation $\rho_s(\hat{r}, r)$.
   - Mean Absolute Error ($\text{MAE}$) and Root Mean Squared Error ($\text{RMSE}$).
   - Coefficient of determination $R^2 = 1 - \frac{\sum (r - \hat{r})^2}{\sum r^2}$ relative to the zero-response baseline.
3. **Directional Accuracy:**
   - Sign agreement percentage: $\operatorname{mean}(\operatorname{sign}(\hat{r}) == \operatorname{sign}(r))$ for responses exceeding threshold $|r| > 0.05$.
4. **Competitor Dynamics:**
   - Top-1 correctness flag at $x_+$ and $x_-$.
   - Leading competitor class index at $c, x_+$, and $x_-$.
   - Runner-up class switching indicator: $\mathbb{I}[j^*(x_+) \neq j^*(c) \text{ or } j^*(x_-) \neq j^*(c)]$.
5. **Computational Benchmarks:**
   - Model loading time.
   - Anchor forward and backward pass latency.
   - FFT response map generation latency.
   - Batched endpoint forward pass throughput (images/sec).
