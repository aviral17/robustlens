# Commutator-Induced Neural Susceptibility (CINS)
## Complete Mathematical Formulation & Theoretical Analysis (Post-Audit Revised)

**Framework:** Commutator-Induced Neural Susceptibility (CINS)  
**Document:** `MATHEMATICAL_FRAMEWORK.md`  
**Date:** November 2026 (Audit Revised)  
**Repository Branch:** `avi1444320-phase3`  

---

### 1. Contextual Image Operator Non-Commutativity

Let $\mathcal{X} = [0, 1]^{H \times W \times C}$ denote the bounded pixel space of digital images with spatial dimensions $H \times W$ and $C=3$ color channels.

Consider two elementary degradation operators:
1. **Gaussian Noise Operator $\mathcal{N}$:**
   $$\mathcal{N}(x; \sigma, \epsilon) = \operatorname{clip}(x + \epsilon, 0, 1), \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$
2. **Gaussian Defocus Blur Operator $\mathcal{B}$:**
   $$\mathcal{B}(x; k) = \operatorname{clip}(K * x, 0, 1)$$
   where $K \in \mathbb{R}^{(2r+1) \times (2r+1)}$ is a normalized spatial Gaussian kernel satisfying $K_{p, q} \ge 0$ and $\sum_{p, q} K_{p, q} = 1$.
3. **JPEG Compression Operator $\mathcal{J}$:**
   $$\mathcal{J}(x; q) = \operatorname{clip}\left( \operatorname{IDCT}\left( \operatorname{Round}\left( \frac{\operatorname{DCT}(x)}{Q(q)} \right) \odot Q(q) \right), 0, 1 \right)$$
   where $\operatorname{DCT}$ denotes block-wise $8 \times 8$ Discrete Cosine Transform and $Q(q)$ is the standard JPEG quantization table for quality factor $q$.

#### 1.1 The Idealized Linear Noise-Blur Commutator
Under idealized unconstrained linear imaging (ignoring boundary clipping $\operatorname{clip}(\cdot, 0, 1)$):
$$\mathcal{B}_{\text{lin}}(\mathcal{N}_{\text{lin}}(x)) = K * (x + \epsilon) = K * x + K * \epsilon$$
$$\mathcal{N}_{\text{lin}}(\mathcal{B}_{\text{lin}}(x)) = (K * x) + \epsilon$$

The unconstrained linear commutator difference is:
$$d_{\text{lin}} = \mathcal{B}_{\text{lin}}(\mathcal{N}_{\text{lin}}(x)) - \mathcal{N}_{\text{lin}}(\mathcal{B}_{\text{lin}}(x)) = (K * \epsilon) - \epsilon = (K - I) * \epsilon$$

**Fundamental Properties of the Idealized Commutator:**
1. **Image Invariance:** $d_{\text{lin}}$ is **strictly independent of the underlying scene $x$**. It depends exclusively on the noise realization $\epsilon$ and the blur kernel $K$.
2. **Fourier Transfer Function:** In the 2D discrete Fourier domain:
   $$\mathcal{F}(d_{\text{lin}})[u, v] = (\hat{K}[u, v] - 1) \cdot \mathcal{F}(\epsilon)[u, v]$$
3. **DC Component & Finite-Grid Deviation:**  
   In the continuous infinite domain $\mathbb{R}^2$, because the blur kernel is normalized ($\iint K = 1$), the DC gain is $\hat{K}[0, 0] = 1$, yielding $\mathbb{E}[\iint d_{\text{lin}}] \equiv 0$.  
   *Audit Clarification on Non-Zero Measured DC:*  
   On a discrete finite grid of $N = H \times W$ pixels with finite random noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$, the empirical sample mean of the noise has standard error:
   $$\text{SE}(\bar{\epsilon}) = \frac{\sigma}{\sqrt{N}}$$
   Furthermore, discrete spatial convolution using finite FIR kernels truncates the Gaussian tails at image boundaries (`mode='same'`). Consequently, the empirical DC residual $\frac{1}{N}\sum d_{\text{lin}}$ is not floating-point machine epsilon ($\approx 10^{-16}$), but a statistically expected finite sample fluctuation of order $\mathcal{O}(\sigma / \sqrt{N})$ combined with boundary truncation (measured at $\approx 10^{-5}$ for $N=16,384$, well within the $3\sigma$ confidence interval).

#### 1.2 Nonlinear Reality: Contextual Operator Coupling
In actual computer vision pipelines, image processing operations are nonlinear due to:
- **Gamut boundary clipping:** $\operatorname{clip}(z, 0, 1) = \max(0, \min(1, z))$.
- **Color-space conversions:** sRGB $\leftrightarrow$ YCbCr with sub-sampling.
- **Quantization:** Rounding DCT coefficients to integers.

For compound 3-stage sequences:
$$u = \operatorname{NBJ}(x) = \mathcal{J}(\mathcal{B}(\mathcal{N}(x; \sigma, \epsilon); k); q)$$
$$v = \operatorname{BNJ}(x) = \mathcal{J}(\mathcal{N}(\mathcal{B}(x; k); \sigma, \epsilon); q)$$

We define:
- **Contextual Order-Swap Residual:**
  $$d = u - v \in [-1, 1]^{H \times W \times C}$$
- **Shared Midpoint Anchor:**
  $$c = \frac{u + v}{2} \in [0, 1]^{H \times W \times C}$$

Because of clipping and downstream DCT quantization, $d$ does **not** vanish, and unlike the linear idealization, it couples with image structure $x$ near saturated regions ($x(p) \approx 0$ or $x(p) \approx 1$).

---

### 2. Spectrally Invariant Spatial Translations & Feasible Strengths

To investigate whether the spatial arrangement of the residual $d$ drives neural susceptibility—independently of its frequency energy—we apply circular spatial translations.

#### 2.1 Definition of Circular Translation
Let $\tau = (\tau_y, \tau_x) \in \{0, \dots, H-1\} \times \{0, \dots, W-1\}$ denote a 2D integer spatial translation vector.  
The circularly translated residual $d_\tau$ is defined component-wise by:
$$d_\tau[y, x, c] = d[(y - \tau_y) \bmod H, (x - \tau_x) \bmod W, c]$$
applied identically across all color channels $c \in \{1, 2, 3\}$.

#### 2.2 Symmetric Counterfactual Intervention Endpoints
Given an intervention scaling factor $\alpha > 0$, we construct the symmetric counterfactual pair:
$$x_+(\tau, \alpha) = c + \frac{\alpha}{2} d_\tau$$
$$x_-(\tau, \alpha) = c - \frac{\alpha}{2} d_\tau$$

#### 2.3 Mathematical Proofs of Residual Invariants
Under circular translation $\tau$:
1. **Per-Channel Fourier Magnitude Invariance:**  
   By the Fourier shift theorem:
   $$\mathcal{F}(d_\tau)[u, v, c] = \mathcal{F}(d)[u, v, c] \cdot e^{-i 2\pi \left( \frac{u \tau_y}{H} + \frac{v \tau_x}{W} \right)}$$
   Since $|e^{-i \theta}| = 1$ for all $\theta \in \mathbb{R}$:
   $$|\mathcal{F}(d_\tau)[u, v, c]| \equiv |\mathcal{F}(d)[u, v, c]| \quad \forall u, v, c$$
2. **Per-Channel Fourier Power Spectrum Invariance:**  
   $$P_{d_\tau}[u, v, c] = |\mathcal{F}(d_\tau)[u, v, c]|^2 \equiv |\mathcal{F}(d)[u, v, c]|^2 = P_d[u, v, c]$$
3. **Parseval's $L_2$ Energy Invariance:**  
   $$\|d_\tau\|_2^2 = \sum_{y, x, c} |d_\tau[y, x, c]|^2 = \frac{1}{HW} \sum_{u, v, c} |\mathcal{F}(d_\tau)[u, v, c]|^2 = \|d\|_2^2$$
4. **Pixel Histogram Invariance:**  
   The set of pixel values in $d_\tau$ is a spatial permutation of the values in $d$. Therefore, all empirical moments (mean, variance, skewness, kurtosis) and the exact marginal pixel histogram are strictly invariant.
5. **Shared Midpoint Invariance:**  
   $$\frac{x_+(\tau, \alpha) + x_-(\tau, \alpha)}{2} = \frac{(c + \frac{\alpha}{2} d_\tau) + (c - \frac{\alpha}{2} d_\tau)}{2} \equiv c$$

> **Crucial Distinction:** The invariants hold for the residual $d_\tau$ and the difference $x_+ - x_- = \alpha d_\tau$. They do **not** imply that the composite images $x_+$ or $x_-$ have invariant Fourier spectra. The spectrum $\mathcal{F}(x_+) = \mathcal{F}(c) + \frac{\alpha}{2} \mathcal{F}(d_\tau)$ depends on the relative phase between $c$ and $d_\tau$, precisely reflecting how the translated residual interacts with underlying image features.

#### 2.4 Exact Pixel-Validity Headroom Formulations
To ensure that $x_+(\tau) \in [0, 1]$ and $x_-(\tau) \in [0, 1]$ without destructive clipping, the intervention strength $\alpha$ must respect the distance of anchor pixels $c$ to the gamut boundaries:

1. **Exact Translation-Specific Feasible Bound:**
   $$\alpha_{\max}(\tau) = \min_{\substack{i, j, c \\ |d_\tau(i, j, c)| > 0}} \frac{2 \min(c(i, j, c), 1 - c(i, j, c))}{|d_\tau(i, j, c)|}$$
2. **Common Feasible Bound Over Evaluation Grid $\mathcal{T}$:**
   $$\alpha_{\text{common}}(\mathcal{T}) = \min_{\tau \in \mathcal{T}} \alpha_{\max}(\tau)$$
3. **Global Toroidal Bound Over All Circular Shifts:**
   $$\alpha_{\text{global}} = \frac{2 \min_{i, j, c} \min(c(i, j, c), 1 - c(i, j, c))}{\max_{i, j, c} |d(i, j, c)|}$$
4. **Interior-Anchor Contraction (Protocol B):**  
   For natural photographic images with boundary pixels ($c \approx 0$ or $c \approx 1$), unconstrained $\alpha_{\text{global}} \to 0$. We define the interior anchor:
   $$c_\eta = (1 - 2\eta)c + \eta, \quad \eta = 0.05 \text{ (provisional setting)}$$
   This guarantees that $\min(c_\eta, 1 - c_\eta) \ge \eta = 0.05$. Therefore:
   $$\alpha_{\text{global}}(c_\eta) \ge \frac{2 \eta}{\max_{i, j, c} |d(i, j, c)|} = \frac{0.10}{\max |d|}$$
   *Important Qualification:* $\alpha \ge 0.46$ is not a universal constant; it holds specifically when $\max |d| \le 0.217$. The bound must be evaluated empirically for each sample image.

---

### 3. FFT-Accelerated Neural Response Landscapes

Let $f: \mathcal{X} \to \mathbb{R}^K$ denote a pretrained vision model outputting unnormalized logit vectors $z(x) \in \mathbb{R}^K$.  
For an image with ground-truth class $y \in \{1, \dots, K\}$, we define three distinct margin metrics:

1. **True-Class Maximum-Competitor Margin:**
   $$m(x) = z_y(x) - \max_{j \ne y} z_j(x)$$
2. **Fixed Anchor-Competitor Margin:**
   $$m_{\text{fixed}}(x) = z_y(x) - z_{j^*(c)}(x), \quad \text{where } j^*(c) = \arg\max_{j \ne y} z_j(c)$$
3. **Smooth Log-Sum-Exp Margin:**
   $$m_{\text{LSE}}(x; T) = z_y(x) - T \ln \sum_{j \ne y} \exp\left( \frac{z_j(x)}{T} \right)$$
   where $\lim_{T \to 0^+} m_{\text{LSE}}(x; T) = m(x)$.

#### 3.1 First-Order Taylor Response Prediction
The actual symmetric neural margin response to the translated intervention is:
$$r(\tau, \alpha) = m(x_+(\tau, \alpha)) - m(x_-(\tau, \alpha))$$

Taking the first-order Taylor expansion around the shared anchor $c$:
$$r(\tau, \alpha) \approx \alpha \langle g, d_\tau \rangle$$
where $g = \nabla_x m(c) \in \mathbb{R}^{H \times W \times C}$ is the spatial gradient of the margin evaluated once at the anchor $c$.

We define the **first-order predicted response**:
$$\hat{r}(\tau, \alpha) = \alpha \langle g, d_\tau \rangle = \alpha \sum_{c=1}^C \sum_{y=0}^{H-1} \sum_{x=0}^{W-1} g[y, x, c] \cdot d[(y - \tau_y) \bmod H, (x - \tau_x) \bmod W, c]$$

#### 3.2 2D FFT Circular Cross-Correlation Derivation
Using the **Circular Convolution & Cross-Correlation Theorem**:
Let $G_c[u, v] = \mathcal{F}_2(g[:, :, c])[u, v]$ and $D_c[u, v] = \mathcal{F}_2(d[:, :, c])[u, v]$.

The inner product $\langle g_c, (d_c)_\tau \rangle$ is equivalent to circular cross-correlation:
$$\mathcal{F}_2(g_c \star d_c)[u, v] = G_c[u, v] \cdot \overline{D_c[u, v]}$$
Summing across color channels and taking the inverse 2D transform:
$$R_{g, d}[\tau_y, \tau_x] = \operatorname{Re}\left( \mathcal{F}_2^{-1} \left( \sum_{c=1}^C G_c \odot \overline{D_c} \right) \right) [\tau_y, \tau_x]$$
$$\hat{\mathbf{R}} = \alpha \cdot R_{g, d} \in \mathbb{R}^{H \times W}$$

**Computational Complexity:**
$$\mathcal{O}(C \cdot HW \log(HW)) \approx 3 \times 50,176 \times 16 \approx 2.4 \times 10^6 \text{ operations}$$
versus $\mathcal{O}(H^2 W^2 C) \approx 7.55 \times 10^9$ operations for brute force.

#### 3.3 Distributional Moments of the Predicted Response Map
1. **Spatial Mean Response:**
   $$\mu_{\hat{r}} = \frac{1}{HW} \sum_{\tau_y, \tau_x} \hat{r}(\tau, \alpha) = \frac{\alpha}{HW} \Psi[0, 0] = \alpha \sum_{c=1}^C \bar{g}_c \bar{d}_c$$
2. **Spatial Variance Response (Parseval's Cross-Spectral Power):**
   $$\sigma^2_{\hat{r}} = \frac{1}{HW} \sum_{\tau_y, \tau_x} (\hat{r}(\tau) - \mu_{\hat{r}})^2 = \frac{\alpha^2}{(HW)^2} \sum_{(u, v) \ne (0, 0)} |\Psi[u, v]|^2$$

---

### 4. Symmetric Taylor Cancellation & Regularity Assumptions

#### 4.1 Quadratic Cancellation Theorem
**Theorem (Symmetric Quadratic Curvature Cancellation):**  
*Let $\mathcal{U} \subset \mathcal{X}$ be an open neighborhood containing the line segment $[c - h, c + h]$, where $h = \frac{\alpha}{2} d_\tau$. If the scalar response functional $m: \mathcal{U} \to \mathbb{R}$ is $\mathcal{C}^5$ continuously Fréchet differentiable on $\mathcal{U}$, then:*

$$m(c + h) = m(c) + \nabla m(c)^T h + \frac{1}{2} h^T \nabla^2 m(c) h + \frac{1}{6} \nabla^3 m(c)[h^3] + \frac{1}{24}\nabla^4 m(c)[h^4] + \mathcal{O}(\|h\|^5)$$
$$m(c - h) = m(c) - \nabla m(c)^T h + \frac{1}{2} h^T \nabla^2 m(c) h - \frac{1}{6} \nabla^3 m(c)[h^3] + \frac{1}{24}\nabla^4 m(c)[h^4] + \mathcal{O}(\|h\|^5)$$

Subtracting:
$$r(\tau, \alpha) = m(c + h) - m(c - h) = 2 \nabla m(c)^T h + \frac{1}{3} \nabla^3 m(c)[h^3] + \mathcal{O}(\|h\|^5)$$
Substituting $h = \frac{\alpha}{2} d_\tau$:
$$r(\tau, \alpha) = \alpha \langle \nabla m(c), d_\tau \rangle + \frac{\alpha^3}{24} \nabla^3 m(c)[d_\tau^3] + \mathcal{O}(\alpha^5)$$

$$\hat{r}(\tau, \alpha) = \alpha \langle \nabla m(c), d_\tau \rangle$$
$$e(\tau, \alpha) = r(\tau, \alpha) - \hat{r}(\tau, \alpha) = \frac{\alpha^3}{24} \nabla^3 m(c)[d_\tau^3] + \mathcal{O}(\alpha^5)$$

*Consequence:* All even-order terms—including the entire second-order Hessian $\nabla^2 m(c)$—cancel identically. The leading-order approximation error is cubic $\mathcal{O}(\alpha^3)$, compared to quadratic $\mathcal{O}(\alpha^2)$ for asymmetric interventions.

#### 4.2 Essential Regularity Caveats & Structural Breakdown
The theorem strictly requires $\mathcal{C}^5$ Fréchet smoothness. Modern deep vision networks violate this assumption in specific ways:
1. **Activation Non-Smoothness:**  
   - Networks with pure ReLU activations without smooth normalizations are piecewise affine. Inside a single linear polytope, $\nabla^2 m \equiv 0$ and $\nabla^3 m \equiv 0$ (exact linearity). However, when the line segment $[c - h, c + h]$ intersects an activation hyperplane, $\nabla m$ is discontinuous and Taylor's theorem cannot be applied across the boundary.
   - Networks with GELU, SiLU, or Swish are smooth ($\mathcal{C}^\infty$) non-linearities, but are **not** piecewise affine. They have continuous non-zero higher-order derivatives at all points.
2. **LayerNorm, Softmax, and Self-Attention:**  
   Input-dependent normalization (LayerNorm, BatchNorm) and self-attention ($A = \operatorname{softmax}(Q K^T / \sqrt{d}) V$) introduce quadratic, bilinear, and transcendental token interactions that invalidate piecewise affinity.
3. **Margin Competitor Switching (The Argmax Ridge):**  
   The true margin $m(x) = z_y(x) - \max_{j \ne y} z_j(x)$ has non-differentiable ridge points whenever runner-up classes switch ($z_{j_1} = z_{j_2}$). If the intervention causes a different class to become the runner-up, the gradient vector jumps discontinuously from $\nabla z_y - \nabla z_{j_1}$ to $\nabla z_y - \nabla z_{j_2}$.
4. **Swin-Transformer Window Geometry:**  
   Swin-Transformer partitions tokens into non-overlapping local windows of $M \times M$ tokens (where $M = 7$ tokens corresponds to $7 \times 4 = 28$ input pixels in Stage 1 with patch size $4 \times 4$). Window straddling represents a hypothesized spatial modulation mechanism, not an established causal explanation.
