# Commutator-Induced Neural Susceptibility (CINS)
## Complete Mathematical Formulation & Theoretical Analysis

**Framework:** Commutator-Induced Neural Susceptibility (CINS)  
**Document:** `MATHEMATICAL_FRAMEWORK.md`  
**Date:** November 2026  
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
3. **Zero DC Component:** Because the blur kernel is normalized ($\sum K = 1$), its DC gain is $\hat{K}[0, 0] = 1$. Therefore:
   $$\hat{K}[0, 0] - 1 = 0 \implies \mathcal{F}(d_{\text{lin}})[0, 0] = 0$$
   The idealized commutator has **strictly zero spatial mean** and acts as a pure high-pass filter on the noise realization $\epsilon$.

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

### 2. Spectrally Invariant Spatial Translations

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

---

### 3. FFT-Accelerated Neural Response Landscapes

Let $f: \mathcal{X} \to \mathbb{R}^K$ denote a pretrained vision model outputting unnormalized logit vectors $z(x) \in \mathbb{R}^K$.  
For an image with ground-truth class $y \in \{1, \dots, K\}$, the **true-class decision margin** is:
$$m(x) = z_y(x) - \max_{j \ne y} z_j(x)$$
Alternatively, the **smooth log-sum-exp margin** with temperature parameter $T > 0$ is:
$$m_{\text{LSE}}(x; T) = z_y(x) - T \ln \sum_{j \ne y} \exp\left( \frac{z_j(x)}{T} \right)$$
Note that $\lim_{T \to 0^+} m_{\text{LSE}}(x; T) = m(x)$.

#### 3.1 First-Order Taylor Response Prediction
The actual symmetric neural margin response to the translated intervention is:
$$r(\tau, \alpha) = m(x_+(\tau, \alpha)) - m(x_-(\tau, \alpha))$$

Taking the first-order Taylor expansion around the shared anchor $c$:
$$m(x_+) = m(c) + \nabla m(c)^T \left( \frac{\alpha}{2} d_\tau \right) + \mathcal{R}_+(c, \alpha d_\tau)$$
$$m(x_-) = m(c) - \nabla m(c)^T \left( \frac{\alpha}{2} d_\tau \right) + \mathcal{R}_-(c, \alpha d_\tau)$$

Subtracting:
$$r(\tau, \alpha) \approx \alpha \langle g, d_\tau \rangle$$
where $g = \nabla_x m(c) \in \mathbb{R}^{H \times W \times C}$ is the spatial gradient of the margin evaluated once at the anchor $c$.

We define the **first-order predicted response**:
$$\hat{r}(\tau, \alpha) = \alpha \langle g, d_\tau \rangle = \alpha \sum_{c=1}^C \sum_{y=0}^{H-1} \sum_{x=0}^{W-1} g[y, x, c] \cdot d[(y - \tau_y) \bmod H, (x - \tau_x) \bmod W, c]$$

#### 3.2 2D FFT Circular Cross-Correlation Derivation
Direct evaluation of $\hat{r}(\tau, \alpha)$ for all $H \times W$ spatial positions via spatial inner products requires:
$$\mathcal{O}(H^2 W^2 C) \text{ operations}$$
For $H=W=224$ and $C=3$, this is $224^4 \times 3 \approx 7.55 \times 10^9$ multiply-accumulate operations per image.

Using the **Circular Convolution & Cross-Correlation Theorem**:
Let $G_c[u, v] = \mathcal{F}_2(g[:, :, c])[u, v]$ and $D_c[u, v] = \mathcal{F}_2(d[:, :, c])[u, v]$.

The inner product $\langle g_c, (d_c)_\tau \rangle$ is equivalent to circular cross-correlation:
$$(g_c \star d_c)[\tau_y, \tau_x] = \sum_{y=0}^{H-1} \sum_{x=0}^{W-1} g_c[y, x] \cdot d_c[(y - \tau_y) \bmod H, (x - \tau_x) \bmod W]$$

Taking the 2D discrete Fourier transform of $(g_c \star d_c)$:
$$\mathcal{F}_2(g_c \star d_c)[u, v] = G_c[u, v] \cdot \overline{D_c[u, v]}$$
where $\overline{D_c[u, v]}$ denotes the complex conjugate.

Summing across color channels and taking the inverse 2D transform:
$$R_{g, d}[\tau_y, \tau_x] = \operatorname{Re}\left( \mathcal{F}_2^{-1} \left( \sum_{c=1}^C G_c \odot \overline{D_c} \right) \right) [\tau_y, \tau_x]$$

Therefore, the complete predicted response map is:
$$\hat{\mathbf{R}} = \alpha \cdot R_{g, d} \in \mathbb{R}^{H \times W}$$
**Computational Complexity:**
$$\mathcal{O}(C \cdot HW \log(HW)) \approx 3 \times 50,176 \times 16 \approx 2.4 \times 10^6 \text{ operations}$$
This achieves a **theoretical speedup of $>3,000\times$** over spatial brute-force cross-correlation and computes all 50,176 response predictions in $<1\text{ millisecond}$.

#### 3.3 Distributional Moments of the Predicted Response Map
From the cross-spectral matrix $\Psi[u, v] = \sum_{c=1}^C G_c[u, v] \overline{D_c[u, v]}$:

1. **Spatial Mean Response:**
   $$\mu_{\hat{r}} = \frac{1}{HW} \sum_{\tau_y, \tau_x} \hat{r}(\tau, \alpha) = \frac{\alpha}{HW} \Psi[0, 0] = \alpha \sum_{c=1}^C \bar{g}_c \bar{d}_c$$
   where $\bar{g}_c = \frac{1}{HW} \sum_{y, x} g[y, x, c]$ and $\bar{d}_c = \frac{1}{HW} \sum_{y, x} d[y, x, c]$.  
   Since residual mean $\bar{d}_c \approx 0$, $\mu_{\hat{r}} \approx 0$.
2. **Spatial Variance Response (Parseval's Cross-Spectral Power):**
   $$\sigma^2_{\hat{r}} = \frac{1}{HW} \sum_{\tau_y, \tau_x} (\hat{r}(\tau) - \mu_{\hat{r}})^2 = \frac{\alpha^2}{(HW)^2} \sum_{(u, v) \ne (0, 0)} |\Psi[u, v]|^2$$
   This proves that the spatial variance of the predicted response map is directly determined by the energy of the cross-spectrum between the gradient and residual.

---

### 4. Symmetric Taylor Cancellation & Approximation Breakdown

#### 4.1 Quadratic Cancellation in Symmetric Perturbations
Assume $m \in C^3(\mathcal{X})$. For perturbation $h = \frac{\alpha}{2} d_\tau$:
$$m(c + h) = m(c) + \nabla m(c)^T h + \frac{1}{2} h^T \nabla^2 m(c) h + \frac{1}{6} \nabla^3 m(c)[h, h, h] + \mathcal{O}(\|h\|^4)$$
$$m(c - h) = m(c) - \nabla m(c)^T h + \frac{1}{2} h^T \nabla^2 m(c) h - \frac{1}{6} \nabla^3 m(c)[h, h, h] + \mathcal{O}(\|h\|^4)$$

Subtracting:
$$r(\tau, \alpha) = m(c + h) - m(c - h) = 2 \nabla m(c)^T h + \frac{1}{3} \nabla^3 m(c)[h, h, h] + \mathcal{O}(\|h\|^5)$$

Substituting $h = \frac{\alpha}{2} d_\tau$:
$$r(\tau, \alpha) = \alpha \langle \nabla m(c), d_\tau \rangle + \frac{\alpha^3}{24} \nabla^3 m(c)[d_\tau, d_\tau, d_\tau] + \mathcal{O}(\alpha^5)$$

$$\hat{r}(\tau, \alpha) = \alpha \langle \nabla m(c), d_\tau \rangle$$

The approximation error is:
$$e(\tau, \alpha) = r(\tau, \alpha) - \hat{r}(\tau, \alpha) = \frac{\alpha^3}{24} \nabla^3 m(c)[d_\tau, d_\tau, d_\tau] + \mathcal{O}(\alpha^5)$$

> **Theorem (Symmetric Quadratic Cancellation):**  
> For any $C^3$ function, the second-order Hessian curvature term $h^T \nabla^2 m(c) h$ identically cancels in symmetric difference interventions. The leading-order error term is cubic:
> $$e(\tau, \alpha) = \mathcal{O}(\alpha^3)$$
> whereas for asymmetric interventions $m(c + \alpha d_\tau) - m(c)$, the leading-order error is quadratic: $\mathcal{O}(\alpha^2)$.

#### 4.2 Mechanisms of Breakdown in Real Vision Networks
Despite quadratic cancellation, deep networks violate global $C^3$ smoothness:
1. **Piecewise-Affine Boundaries (ReLU / GELU):**  
   Activations partition input space into linear regions separated by hyperplanes. If $c + h$ and $c - h$ cross activation boundaries, the gradient $\nabla m$ jumps discontinuously.
2. **Margin Competitor Switching:**  
   The true margin $m(x) = z_y(x) - \max_{j \ne y} z_j(x)$ has non-differentiable ridge points whenever two competing classes have equal logits:
   $$z_{j_1}(x) = z_{j_2}(x) = \max_{j \ne y} z_j(x)$$
   If the intervention flips the leading competitor from $j_1$ to $j_2$, the effective gradient changes from $\nabla z_y - \nabla z_{j_1}$ to $\nabla z_y - \nabla z_{j_2}$.
3. **Log-Sum-Exp Smoothness Regularization:**  
   Using $m_{\text{LSE}}(x; T)$ replaces the hard maximum with a smooth approximation with analytic gradient:
   $$\nabla m_{\text{LSE}}(x; T) = \nabla z_y(x) - \sum_{j \ne y} p_j(x; T) \nabla z_j(x)$$
   where $p_j(x; T) = \frac{\exp(z_j(x)/T)}{\sum_{k \ne y} \exp(z_k(x)/T)}$ is the softmax over competing classes.

---

### 5. Summary of Mathematical Deliverables & Invariants

| Quantity | Notation | Dimension | Mathematical Formula | Invariant Under Translation? |
| :--- | :--- | :--- | :--- | :---: |
| Clean image | $x$ | $H \times W \times C$ | Input image | Fixed base |
| First sequence | $u = \operatorname{NBJ}(x)$ | $H \times W \times C$ | $\mathcal{J}(\mathcal{B}(\mathcal{N}(x)))$ | N/A |
| Second sequence | $v = \operatorname{BNJ}(x)$ | $H \times W \times C$ | $\mathcal{J}(\mathcal{N}(\mathcal{B}(x)))$ | N/A |
| Swap residual | $d = u - v$ | $H \times W \times C$ | $u - v$ | N/A |
| Shared anchor | $c = (u + v)/2$ | $H \times W \times C$ | $(u + v)/2$ | Invariant |
| Translated residual | $d_\tau$ | $H \times W \times C$ | $\operatorname{roll}(d, \tau)$ | Transformed |
| Residual Fourier power | $P_{d_\tau}$ | $H \times W \times C$ | $\|\mathcal{F}(d_\tau)\|^2$ | **Strictly Invariant** |
| Residual $L_2$ energy | $\|d_\tau\|_2$ | $\mathbb{R}$ | $\sqrt{\sum \|d_\tau\|^2}$ | **Strictly Invariant** |
| Residual pixel histogram | $H(d_\tau)$ | $\mathbb{R}^B$ | Binned counts | **Strictly Invariant** |
| Anchor gradient | $g$ | $H \times W \times C$ | $\nabla_x m(c)$ | Evaluated at anchor |
| FFT response map | $\hat{\mathbf{R}}$ | $H \times W$ | $\alpha \operatorname{Re}(\mathcal{F}^{-1}(\sum G_c \odot \overline{D_c}))$ | Full 50,176 landscape |
| Actual response | $r(\tau, \alpha)$ | $\mathbb{R}$ | $m(x_+) - m(x_-)$ | True neural margin |
| Error term | $e(\tau, \alpha)$ | $\mathbb{R}$ | $r(\tau, \alpha) - \hat{r}(\tau, \alpha)$ | $\mathcal{O}(\alpha^3)$ leading term |
