# Approximation Error Analysis: Commutator-Induced Neural Susceptibility (CINS)
## Theoretical Characterization & Regularity Breakdown (Post-Audit Revised)

**Document Reference:** `research/phase3_cins/APPROXIMATION_ERROR_ANALYSIS.md`  
**Framework:** Commutator-Induced Neural Susceptibility (CINS)  
**Authors:** RobustLens Research Initiative  
**Date:** November 2026 (Post-Audit Revised)  
**Status:** Canonical Error Characterization & Methodological Hypotheses  

---

## 1. Executive Summary & Objective

The CINS framework investigates whether the spatial decision response map:
$$r(\tau) = m\left(c + \frac{1}{2}\alpha d_\tau\right) - m\left(c - \frac{1}{2}\alpha d_\tau\right)$$
can be predicted across spatial translations $\tau \in \{0, \dots, H-1\} \times \{0, \dots, W-1\}$ via a single 2D circular cross-correlation:
$$\hat{r}(\tau) = \alpha \langle g, d_\tau \rangle = \alpha \operatorname{Re}\left(\mathcal{F}^{-1}\left(\sum_{c=1}^C G_c \odot \overline{D_c}\right)\right)(\tau)$$
where $g = \nabla_x m(c) \in \mathbb{R}^{H \times W \times C}$ is the spatial gradient of a decision margin evaluated at the anchor $c = \frac{1}{2}(u + v)$.

While 2D FFT computation provides an algorithmic speedup of over $2,000\times$ on CPU compared to brute-force spatial inner products, the prediction $\hat{r}(\tau)$ is a first-order Taylor approximation.

This document presents a rigorous theoretical and empirical analysis of:
1. **The Exact Order of Convergence:** Why symmetric difference interventions nullify the second-order Hessian $\nabla^2 m(c)$ on sufficiently smooth manifolds, yielding cubic $\mathcal{O}(\alpha^3)$ leading error.
2. **Smoothness Assumptions:** The explicit $\mathcal{C}^5$ Fréchet differentiability requirement and how activation boundaries violate it.
3. **Activation Geometries:** The mathematical distinction between piecewise-affine networks (pure ReLU) and smooth non-linear networks (GELU, SiLU, Softmax, LayerNorm).
4. **The Argmax Cliff:** Discontinuities in margin gradient when runner-up classes switch.
5. **Architectural Hypotheses:** How convolutional networks (ResNet-50, ConvNeXt-Tiny) and Vision Transformers (ViT-Base/16, Swin-Tiny) differ in regularity.

---

## 2. Multi-Scale Taylor Expansion & Error Derivations

Let $m: \mathcal{U} \to \mathbb{R}$ denote a scalar neural response function defined on an open neighborhood $\mathcal{U} \subset \mathbb{R}^D$ containing the line segment $[c - h_\tau, c + h_\tau]$, where $h_\tau = \frac{1}{2} \alpha d_\tau$ and $D = H \times W \times C$.

### 2.1 Asymmetric vs. Symmetric Intervention Expansions

Consider first the **asymmetric intervention** (evaluating only the forward perturbation relative to the anchor):
$$r_{\text{asym}}(\tau) = m(c + \alpha d_\tau) - m(c)$$

Assuming $m \in \mathcal{C}^2(\mathcal{U})$:
$$m(c + \alpha d_\tau) = m(c) + \alpha \langle \nabla m(c), d_\tau \rangle + \frac{1}{2}\alpha^2 d_\tau^T \nabla^2 m(c) d_\tau + \mathcal{O}(\alpha^3)$$
$$\mathcal{E}_{\text{asym}}(\tau) = |r_{\text{asym}}(\tau) - \hat{r}_{\text{asym}}(\tau)| = \frac{1}{2}\alpha^2 \left| d_\tau^T \nabla^2 m(c) d_\tau \right| + \mathcal{O}(\alpha^3)$$
**Convergence rate:** $\mathcal{O}(\alpha^2)$.

Now consider the **symmetric difference intervention**:
$$r_{\text{sym}}(\tau) = m(c + h_\tau) - m(c - h_\tau)$$

Assuming $m \in \mathcal{C}^5(\mathcal{U})$:
$$m(c + h_\tau) = m(c) + \nabla m(c)^T h_\tau + \frac{1}{2} h_\tau^T \nabla^2 m(c) h_\tau + \frac{1}{6} \nabla^3 m(c)[h_\tau^3] + \frac{1}{24}\nabla^4 m(c)[h_\tau^4] + \mathcal{O}(\|h_\tau\|^5)$$
$$m(c - h_\tau) = m(c) - \nabla m(c)^T h_\tau + \frac{1}{2} h_\tau^T \nabla^2 m(c) h_\tau - \frac{1}{6} \nabla^3 m(c)[h_\tau^3] + \frac{1}{24}\nabla^4 m(c)[h_\tau^4] + \mathcal{O}(\|h_\tau\|^5)$$

Subtracting $m(c - h_\tau)$ from $m(c + h_\tau)$:
$$r_{\text{sym}}(\tau) = 2 \nabla m(c)^T h_\tau + \frac{1}{3} \nabla^3 m(c)[h_\tau^3] + \mathcal{O}(\|h_\tau\|^5)$$

Substituting $h_\tau = \frac{1}{2}\alpha d_\tau$:
$$2 \nabla m(c)^T \left(\frac{1}{2}\alpha d_\tau\right) = \alpha \langle \nabla m(c), d_\tau \rangle = \hat{r}(\tau)$$
$$\frac{1}{3} \nabla^3 m(c)\left[\left(\frac{1}{2}\alpha d_\tau\right)^3\right] = \frac{\alpha^3}{24} \nabla^3 m(c)[d_\tau^3]$$

### 2.2 Quadratic Curvature Cancellation Theorem

**Theorem 1 (Quadratic Curvature Nullification):**  
*On any open domain $\mathcal{U}$ where $m \in \mathcal{C}^5(\mathcal{U})$, all even-order Taylor terms—including the entire second-order Hessian matrix $\nabla^2 m(c)$—identically cancel under symmetric difference interventions:*

$$\mathcal{E}_{\text{sym}}(\tau) = |r_{\text{sym}}(\tau) - \hat{r}(\tau)| = \frac{\alpha^3}{24} \left| \nabla^3 m(c)[d_\tau^3] \right| + \mathcal{O}(\alpha^5)$$

**Numerical Consequence:**
- In contrast to asymmetric interventions where local quadratic curvature creates $\mathcal{O}(\alpha^2)$ error, symmetric interventions nullify Hessian curvature.
- Halving $\alpha \to 0.5\alpha$ diminishes the leading error term by a factor of $2^3 = 8.00\times$, as numerically verified in `test_cins_unit.py` (Test 4B and 4C).

---

## 3. Structural Mechanisms of Approximation Breakdown

The theoretical $\mathcal{O}(\alpha^3)$ rate strictly requires $\mathcal{C}^5$ Fréchet smoothness. Real neural networks violate this assumption in four specific ways:

```
                          ┌───────────────────────────────────────────────┐
                          │   Neural Response Map Approximation Bounds    │
                          └──────────────────────┬────────────────────────┘
                                                 │
         ┌────────────────────────┬──────────────┴───────────────┬────────────────────────┐
         │                        │                              │                        │
         ▼                        ▼                              ▼                        ▼
┌──────────────────┐    ┌──────────────────┐           ┌──────────────────┐     ┌──────────────────┐
│  Argmax Cliff    │    │ Activation Poly  │           │ Softmax Saturation│    │ Non-Local Tokens │
│  (Margin Discont)│    │ Crossings (ReLUs)│           │ (Logit Squeeze)  │     │ (ViT Attention)  │
└──────────────────┘    └──────────────────┘           └──────────────────┘     └──────────────────┘
```

### 3.1 Mechanism I: The "Argmax Cliff" (Runner-Up Class Switching)

In classification benchmarks, the true-class decision margin is:
$$m(x) = z_{y^*}(x) - \max_{j \neq y^*} z_j(x)$$
where $z_j(x)$ is the pre-softmax logit for class $j$, and $y^*$ is the ground-truth class.

The margin function is non-differentiable at any point $x$ where two or more runner-up classes have identical logits:
$$\mathcal{S}_{\text{cliff}} = \left\{ x \in \mathbb{R}^D \;\middle|\; \exists j_1 \neq j_2 \neq y^* \text{ s.t. } z_{j_1}(x) = z_{j_2}(x) = \max_{j \neq y^*} z_j(x) \right\}$$

Moreover, the gradient evaluated at anchor $c$ is:
$$g = \nabla_x z_{y^*}(c) - \nabla_x z_{j^*(c)}(c), \quad \text{where } j^*(c) = \arg\max_{j \neq y^*} z_j(c)$$

If the perturbation $h_\tau$ causes a different class $k \neq j^*(c)$ to become the leading runner-up:
$$\arg\max_{j \neq y^*} z_j(c + h_\tau) = k \neq j^*(c)$$
the true response reflects $\nabla z_k$, whereas the predicted response was linear in $\nabla z_{j^*(c)}$.

*Methodological Solution:* In the Kaggle GPU experiment, we log three distinct margin formulations:
1. True-class maximum-competitor margin: $m(x) = z_{y^*} - \max_{j \ne y^*} z_j$
2. Fixed anchor-competitor margin: $m_{\text{fixed}}(x) = z_{y^*} - z_{j^*(c)}$ (strictly differentiable if $z$ is smooth)
3. Smooth log-sum-exp margin: $m_{\text{LSE}}(x; T) = z_{y^*} - T \ln \sum_{j \ne y^*} \exp(z_j / T)$

### 3.2 Mechanism II: Activation Boundary Crossings vs. Smooth Non-Linearities

- **Piecewise-Affine Architectures (Pure ReLU Networks):**  
  Networks with pure ReLU activations (without LayerNorm, BatchNorm, or smooth non-linearities) partition input space $\mathbb{R}^D$ into a finite collection of convex polytopes. Inside a single polytope, the network is affine ($W_k x + b_k$), and all higher derivatives $\nabla^2 m \equiv 0$ and $\nabla^3 m \equiv 0$. The approximation $\hat{r}(\tau) = r(\tau)$ is exact within that polytope. However, when the line segment $[c - h_\tau, c + h_\tau]$ crosses an activation hyperplane, $\nabla m$ undergoes a jump discontinuity.
- **Smooth Non-Linear Architectures (GELU, SiLU, Swish):**  
  GELU ($\text{GELU}(x) = x \Phi(x)$) is **not** piecewise affine. It is infinitely differentiable ($\mathcal{C}^\infty$), with smooth, non-zero derivatives of all orders. Networks with GELU activations (ConvNeXt, ViT, Swin) have continuously varying gradients without jump discontinuities, but higher-order derivatives $\nabla^3 m \ne 0$ are non-zero throughout input space.

### 3.3 Mechanism III: Softmax Compression and Logit Saturation

When researchers evaluate post-softmax probabilities $p_{y^*}(x) = \frac{\exp(z_{y^*}(x))}{\sum_j \exp(z_j(x))}$ rather than logits:
$$\frac{\partial p_i}{\partial x} = p_i \left( \nabla z_i - \sum_j p_j \nabla z_j \right)$$
When a model is highly confident ($p_{y^*} > 0.99$), $p_i(1 - p_i) \to 0$, causing the probability response to saturate (vanishing sensitivity).  
*Rule:* CINS evaluations must primarily evaluate un-normalized pre-softmax logits $z_{y^*}$ and margins $m(x)$ to avoid artificial saturation.

---

## 4. Architectural Regularity Hypotheses

| Property | ResNet-50 (`resnet50.a1_in1k`) | ConvNeXt-Tiny (`convnext_tiny.in12k_ft_in1k`) | ViT-Base/16 (`vit_base_patch16_224...`) | Swin-Tiny (`swin_tiny_patch4...`) |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Operator** | $3\times 3$ Conv + BatchNorm | Depthwise $7\times 7$ Conv + LayerNorm | Multi-Head Self-Attention ($16\times 16$ patches) | Shifted Window Self-Attention ($7\times 7$ tokens) |
| **Activation Function** | ReLU (Piecewise Affine without BN) | GELU (Smooth $\mathcal{C}^\infty$ Non-Linear) | GELU (Smooth $\mathcal{C}^\infty$ Non-Linear) | GELU (Smooth $\mathcal{C}^\infty$ Non-Linear) |
| **Receptive Field** | Bounded local, grows with depth | Bounded local, grows with depth | Global from Layer 1 | Windowed ($7\times 7$ tokens = $28\times 28$ pixels) |
| **Linearity Hypothesis** | High within ReLU polytopes; breaks at boundaries | Smooth degradation governed by GELU curvature | Modulated by quadratic attention interactions | Modulated by window-straddling token boundaries |

### 4.1 Swin-Transformer Window Geometry
Swin-Transformer restricts self-attention to local non-overlapping windows of $M \times M$ tokens. For Swin-Tiny, $M = 7$ tokens, which with patch size $4 \times 4$ corresponds to $28 \times 28$ input pixels in Stage 1. Translating a high-frequency residual across these window partitions may modulate localized self-attention tokens. This is a hypothesized spatial mechanism that must be empirically tested against measured responses.

---

## 5. Methodological Hypotheses to be Empirically Tested

The preliminary report's claim that $\alpha \in [0.10, 0.25]$ is universally "highly accurate" and that $\alpha \ge 0.50$ "fails" was an unsupported extrapolation. In the 10-image GPU engineering experiment, we will formally test:

- **Hypothesis H1 (Intervention Strength Scaling):**  
  Does the empirical approximation error $\|r(\tau, \alpha) - \hat{r}(\tau, \alpha)\|$ scale cubically with $\alpha$ across the prespecified levels $\alpha \in \{0.2, 0.5, 0.8\} \times \alpha_{\text{common}}$?
- **Hypothesis H2 (Margin Regularity Comparison):**  
  Does the fixed anchor-competitor margin $m_{\text{fixed}}$ exhibit higher linear correlation with $\hat{r}$ than the true-class maximum-competitor margin $m$ when runner-up switching occurs?
- **Hypothesis H3 (Architectural Ranking):**  
  Do convolutional networks (ConvNeXt-Tiny, ResNet-50) maintain higher prediction correlation $\rho(\hat{r}, r)$ than Vision Transformers (ViT-Base, Swin-Tiny) under equivalent relative intervention strengths?
