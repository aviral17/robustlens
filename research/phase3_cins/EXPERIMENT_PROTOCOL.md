# Commutator-Induced Neural Susceptibility (CINS)
## Engineering Feasibility Experiment Protocol

**Framework:** Commutator-Induced Neural Susceptibility (CINS)  
**Document:** `EXPERIMENT_PROTOCOL.md`  
**Date:** November 2026  
**Repository Branch:** `avi1444320-phase3`

---

### 1. Research Objectives

This protocol governs the **Primary Engineering Feasibility Experiment** to empirically answer:
1. **Response Variation:** Does circularly translating the compound order-swap residual $d = \operatorname{NBJ}(x) - \operatorname{BNJ}(x)$ produce statistically meaningful neural decision margin variation ($r(\tau, \alpha) = m(x_+) - m(x_-)$), even though residual Fourier power, squared energy, and pixel-value distribution are strictly invariant?
2. **First-Order Predictive Accuracy:** Does the single-pass shared-anchor gradient cross-correlated with $d$ via 2D FFT accurately predict the spatial response across translations?
3. **Approximation Breakdown:** Under what intervention strengths $\alpha \in \{0.10, 0.25, 0.50\}$ and architectures (ResNet-50, ConvNeXt-Tiny, ViT-Base/16, Swin-Tiny) does the first-order approximation degrade?
4. **Computational Advantage:** What is the measured wall-clock runtime of 2D FFT cross-correlation compared to batched neural forward passes?

---

### 2. Dataset Cohort & Image Selection

**Development Cohort (10 Source Images):**
- 5 canonical development images from the RobustLens repository (`img_0_clean.jpg` to `img_4_clean.jpg`), representing varied semantic classes:
  - Class 0: Tench (Fish)
  - Class 217: English Springer Spaniel (Dog)
  - Class 2: Great White Shark (Marine)
  - Class 497: Church (Architecture)
  - Class 130: Gar (Fish)
- 5 independent natural photographic images from ImageNet-V2 (`MatchedFrequency` split), representing standard real-world photographic distributions.
- All images standardized to $224 \times 224 \times 3$ resolution.
- Normalization parameters: ImageNet standard mean $\mu = [0.485, 0.456, 0.406]$ and std $\sigma = [0.229, 0.224, 0.225]$.

---

### 3. Degradation Operators & Fixed Parameter Settings

To ensure causal isolation of operation order without parameter confounding:
- **Noise ($\mathcal{N}$):** Additive Gaussian noise with standard deviation $\sigma = 0.08$.  
  **MANDATORY:** The exact same pseudo-random noise tensor $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ is generated once per image with fixed seed $S = 42 + \text{image\_id}$ and passed to **both** sequences $\operatorname{NBJ}$ and $\operatorname{BNJ}$.
- **Blur ($\mathcal{B}$):** Normalized spatial Gaussian blur with radius $r=2$ ($5 \times 5$ kernel), standard deviation $\sigma_{\text{blur}} = 1.50$.
- **JPEG ($\mathcal{J}$):** Standard 8×8 DCT quantization with quality factor $q = 75$.

---

### 4. Intervention Design & Translation Sampling

#### 4.1 Symmetrically Perturbed Inputs
For shared anchor $c = \frac{u + v}{2}$ and translated residual $d_\tau = \operatorname{roll}(u - v, \tau)$:
$$x_+(\tau, \alpha) = c + \frac{\alpha}{2} d_\tau$$
$$x_-(\tau, \alpha) = c - \frac{\alpha}{2} d_\tau$$

#### 4.2 Intervention Strengths
Three pre-registered strengths are tested:
1. **Mild ($\alpha_1 = 0.10$):** High-fidelity regime; tests local Taylor linearity.
2. **Moderate ($\alpha_2 = 0.25$):** Intermediate regime; balance between signal strength and boundary crossing.
3. **Strong ($\alpha_3 = 0.50$):** Stress-test regime; triggers activation boundary crossings and competing class flips.

#### 4.3 Spatial Translation Positions (per image)
1. **Uniform Grid Sample (32 positions):**  
   $\tau = (\tau_y, \tau_x)$ selected from a uniform $6 \times 6$ lattice covering $\{0, 44, 88, 132, 176, 220\}^2$ (first 32 coordinates).
2. **Predicted Extremes (2 positions):**  
   - $\tau_{\max} = \arg\max_{\tau} \hat{r}(\tau, \alpha)$ (predicted maximum margin response)
   - $\tau_{\min} = \arg\min_{\tau} \hat{r}(\tau, \alpha)$ (predicted minimum margin response)
3. **Controls (2 positions):**  
   - Unshifted residual $\tau = (0, 0)$ (the original compound order difference)
   - Edge-alignment control (translation maximizing Sobel edge alignment with anchor edges)

---

### 5. Neural Architecture Suite

Four canonical vision architectures representing modern inductive biases:
1. **ResNet-50:** Standard convolutional network (rigid local receptive field, batch normalization).
2. **ConvNeXt-Tiny:** Modernized convolutional network (large $7 \times 7$ depthwise kernels, LayerNorm, GELU).
3. **ViT-Base/16:** Standard Vision Transformer (global self-attention, non-local spatial mixing, patch tokens).
4. **Swin-Tiny:** Hierarchical Vision Transformer (shifted local window attention).

All checkpoints loaded from canonical torchvision / timm pretrained ImageNet-1K weights with verified SHA-256 hashes.

---

### 6. Quantitative Evaluation Metrics

For each image $i$ and architecture $a$:
1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{k=1}^N |r_k - \hat{r}_k|$$
2. **Root Mean Square Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{k=1}^N (r_k - \hat{r}_k)^2}$$
3. **Out-of-Sample Variance Explained ($R^2$):**
   $$R^2 = 1 - \frac{\sum_{k=1}^N (r_k - \hat{r}_k)^2}{\sum_{k=1}^N (r_k - \bar{r})^2}$$
4. **Within-Image Spearman Rank Correlation ($\rho_{\text{Spearman}}$):**
   Monotonic ranking alignment between predicted $\hat{r}(\tau)$ and actual $r(\tau)$.
5. **Sign Agreement Rate ($S_{\text{sign}}$):**
   Fraction of translations where $\operatorname{sign}(\hat{r}) == \operatorname{sign}(r)$ for non-trivial responses ($|r| > 0.05$).
6. **Extreme Ranking Fidelity:**
   Verification of whether $r(\tau_{\max}) > r(\tau_{\min})$.
7. **Empirical Wall-Clock Speedup:**
   $$\text{Speedup} = \frac{\text{Time to compute } N \text{ neural forward passes}}{\text{Time to compute 2D FFT cross-correlation}}$$

---

### 7. Evaluation Budget & Stopping Rules

- **Total Neural Evaluations per Image:**  
  $1 \text{ anchor forward pass} + 1 \text{ backward pass} + [32 \text{ grid} + 2 \text{ extremes} + 2 \text{ controls}] \times 3 \text{ alphas} \times 2 \text{ endpoints} \times 4 \text{ models} = 864 \text{ passes}$.
- **Cohort Budget (10 Images):** $\approx 8,640$ forward passes.
- **Stopping Rule:** If the local environment blocks PyTorch CUDA/DLL execution due to AppControl policies, the pipeline falls back to:
  1. Synthetic differentiable network verification.
  2. High-precision NumPy FFT cross-correlation engine.
  3. API-connected ZeroGPU forward passes for verified model validation.
  4. Generation of a standalone Kaggle GPU reproducible runner (`cins_phase3_runner.py` / `.ipynb`) to guarantee complete independent replication.
