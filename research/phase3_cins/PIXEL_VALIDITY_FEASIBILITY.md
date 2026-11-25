# Empirical Pixel-Validity Feasibility Audit: CINS Framework
## Exact Bounds & Protocol Separation (Post-Audit Revised)

**Document Reference:** `research/phase3_cins/PIXEL_VALIDITY_FEASIBILITY.md`  
**Framework:** Commutator-Induced Neural Susceptibility (CINS)  
**Authors:** RobustLens Research Initiative  
**Date:** November 2026 (Post-Audit Revised)  
**Status:** Validated Headroom Formulations & Protocols  

---

## 1. Mathematical Problem Formulation

The counterfactual symmetric intervention is defined by:
$$x_+(\tau) = c + \frac{1}{2}\alpha d_\tau, \quad x_-(\tau) = c - \frac{1}{2}\alpha d_\tau$$
where $c \in [0, 1]^{H \times W \times C}$ is the anchor image and $d_\tau$ is the circularly translated order-swap residual.

For the intervention to be strictly valid without post-hoc clipping:
$$0.0 \le x_+(\tau)_{i,j,k} \le 1.0 \quad \text{and} \quad 0.0 \le x_-(\tau)_{i,j,k} \le 1.0, \quad \forall (i, j, k)$$

Rearranging:
$$-\min(c, 1-c) \le \frac{1}{2}\alpha d_\tau \le \min(c, 1-c)$$
$$\alpha |d_\tau(i, j, k)| \le 2 \min(c(i, j, k), 1 - c(i, j, k))$$

---

## 2. Rigorous Feasible Bound Derivations

### 2.1 Exact Translation-Specific Feasible Bound
For any specific spatial translation $\tau = (\tau_y, \tau_x)$:
$$\alpha_{\max}(\tau) = \min_{\substack{i, j, k \\ |d_\tau(i, j, k)| > 0}} \frac{2 \min(c(i, j, k), 1 - c(i, j, k))}{|d_\tau(i, j, k)|}$$
This is the exact maximum scaling factor such that neither $x_+(\tau)$ nor $x_-(\tau)$ violates $[0, 1]$ gamut boundaries.

### 2.2 Common Feasible Bound Over Evaluation Grid $\mathcal{T}$
When comparing predictions across a discrete set of translations $\mathcal{T} = \{\tau_1, \tau_2, \dots, \tau_M\}$:
$$\alpha_{\text{common}}(\mathcal{T}) = \min_{\tau \in \mathcal{T}} \alpha_{\max}(\tau)$$
Applying $\alpha \le \alpha_{\text{common}}(\mathcal{T})$ guarantees zero clipping across all tested translation coordinates simultaneously.

### 2.3 Global Toroidal Bound
Across all possible circular shifts $\tau \in [0, H-1] \times [0, W-1]$:
$$\alpha_{\text{global}} = \frac{2 \min_{i, j, k} \min(c(i, j, k), 1 - c(i, j, k))}{\max_{i, j, k} |d(i, j, k)|}$$

---

## 3. Protocol Separation: Midpoint vs. Interiorized Anchor

### Protocol A: Raw Midpoint Anchor Experiment ($c = \frac{u + v}{2}$)
In natural photographic images with specular highlights ($c \approx 1.0$) or dark shadows ($c \approx 0.0$), the minimum headroom $\min(c, 1-c) \to 0$, forcing unconstrained $\alpha_{\text{global}} \to 0$.
- **Operational Rule for Protocol A:**  
  Evaluate only translations $\tau$ where $\alpha_{\max}(\tau)$ is non-trivial, or evaluate prespecified $\alpha \in \{0.2, 0.5, 0.8\} \times \alpha_{\text{common}}(\mathcal{T})$.
- **Masking Caveat:**  
  Applying spatial masks or post-hoc clipping $\operatorname{clip}(x, 0, 1)$ distorts the residual spectrum and invalidates the Fourier shift theorem. **Masked interventions cannot be treated as spectrally invariant without explicit verification.**

### Protocol B: Interiorized-Anchor Contraction ($c_\eta$)
To guarantee adequate headroom across $100\%$ of translations without clipping:
$$c_\eta = (1 - 2\eta)c + \eta, \quad \eta = 0.05 \text{ (provisional engineering setting)}$$
- **Mathematical Bound:**  
  Since $c \in [0, 1]$, $c_\eta \in [\eta, 1 - \eta] = [0.05, 0.95]$.  
  The minimum headroom is strictly:
  $$\min_{i, j, k} \min(c_\eta, 1 - c_\eta) \ge \eta = 0.05$$
  Therefore:
  $$\alpha_{\text{global}}(c_\eta) \ge \frac{2 \times 0.05}{\max_{i, j, k} |d(i, j, k)|} = \frac{0.10}{\max |d|}$$
- **Audit Correction:**  
  $\alpha \ge 0.46$ is **not** a universal mathematical constant. For a sample where $\max |d| = 0.20$, $\alpha_{\text{global}} \ge 0.50$. For a sample where $\max |d| = 0.25$, $\alpha_{\text{global}} \ge 0.40$. The bound must be evaluated individually per image.
- **Anchor Modification Reporting:**  
  In all Protocol B experiments, researchers must explicitly report:
  1. The anchor modification parameter $\eta = 0.05$.
  2. The feasible intervention magnitude $\alpha_{\text{common}}$ and $\alpha_{\text{global}}$.
  3. The change in baseline true-class margin between clean image $x$ and contracted anchor $c_\eta$: $\Delta m_{\text{anchor}} = m(c_\eta) - m(x)$.

---

## 4. Prespecified Experimental Intervention Strengths

To avoid obtaining artificially small approximation errors through vanishing perturbations, experiments must use non-trivial intervention strengths prespecified as fractions of the common feasible maximum:

$$\alpha \in \{0.20 \cdot \alpha_{\text{common}}, \; 0.50 \cdot \alpha_{\text{common}}, \; 0.80 \cdot \alpha_{\text{common}}\}$$
capped at $\alpha = 1.0$ (the original residual scale) where appropriate.

For each strength, the experiment logs:
- Absolute scaling factor $\alpha$.
- Effective perturbation $L_2$ norm: $\|h_\tau\|_2 = \frac{1}{2}\alpha \|d\|_2$.
- Maximum pixel perturbation: $\|h_\tau\|_\infty = \frac{1}{2}\alpha \|d\|_\infty$.
