"""Generate IEEE/ACM-compatible LaTeX source for the Matched-Fidelity Academic Manuscript.

Outputs:
  - reports/final_research_package/RobustLens_Academic_Manuscript.tex
  - apps/web/public/downloads/RobustLens_Academic_Manuscript.tex
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = BASE_DIR / "reports" / "final_research_package"
PUBLIC_DOWNLOADS = BASE_DIR / "apps" / "web" / "public" / "downloads"

LATEX_CONTENT = r"""\documentclass[10pt,journal,compsoc]{IEEEtran}

\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{cite}
\usepackage{hyperref}
\usepackage{microtype}
\usepackage{xcolor}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue
}

\title{At Matched Image Fidelity, Does Corruption Order Still Matter? Spatial-Frequency and Structural Fingerprints of Vision Classifier Reliability}

\author{Aviral~Choudhary%
\thanks{Manuscript received November 2026. The author is an independent machine learning and computer vision researcher (email: avi14320@gmail.com). Source code, evaluation protocols, and replication data are openly available at \url{https://robustlens.vercel.app} and \url{https://github.com/aviral17/robustlens}.}}

\markboth{IEEE Transactions on Pattern Analysis and Machine Intelligence (Working Paper Draft, November 2026)}%
{Choudhary: At Matched Image Fidelity, Does Corruption Order Still Matter?}

\begin{document}

\maketitle

\begin{abstract}
Evaluating deep neural network robustness under compound physical corruptions (e.g., Gaussian noise, defocus blur, and JPEG compression) commonly reveals pronounced sequence-order sensitivity: applying identical perturbations in differing temporal sequences produces accuracy discrepancies exceeding 20 percentage points across canonical vision backbones. However, because operators do not commute, nominal sequences yield systematically divergent low-level fidelity (PSNR differences exceeding 6 dB). Here, we address whether sequence order genuinely alters model decision margins when global fidelity is equalized. We formulate a six-way simultaneous common-support calibration protocol across three discrete distortion regimes (27.20 dB / 0.810 SSIM, 23.80 dB / 0.650 SSIM, and 21.80 dB / 0.500 SSIM; tolerances $\pm 0.75$ dB, $\pm 0.035$). Across $N = 3{,}825$ confirmatory images ($290{,}700$ primary forward passes across ResNet-50, ConvNeXt-Tiny, ViT-Base/16, and Swin-Tiny), common-support matching contracts aggregate accuracy gaps by over 60\%, yet statistically significant sequence variance persists in continuous true-class logit margins ($p \le 0.0004$ via $B = 10{,}000$ paired cluster bootstraps). We demonstrate that this residual variance is driven by five physical distortion fingerprints: low- and high-frequency Fourier energy, JPEG $8 \times 8$ grid step discontinuity, Sobel edge energy retention, and local SSIM heterogeneity. Incorporating these continuous fingerprints improves out-of-fold margin prediction by up to $\Delta R^2 = +0.1042$ ($p < 0.0001$) over nonlinear restricted cubic spline fidelity baselines (PSNR, SSIM, LPIPS). Independent replication on ImageNet-V2 ($N = 300$, $22{,}800$ passes) confirms that isotropic Vision Transformers exhibit unique susceptibility to spatial non-uniformity ($\Delta R^2 = +0.0371$, $p_{\text{adj}} = 0.0340$).
\end{abstract}

\begin{IEEEkeywords}
Out-of-distribution robustness, compound corruptions, non-commutative operators, common-support calibration, distortion fingerprints, Vision Transformers.
\end{IEEEkeywords}

\section{Introduction}
Real-world computer vision systems encounter compound image corruptions occurring sequentially—such as sensor photon noise, atmospheric defocus blur, and discrete block transform encoding. In canonical robustness benchmarks \cite{hendrycks2019robustness}, perturbations are applied in isolation. When composed sequentially, permutation order dramatically alters classifier correctness \cite{goyal2022dual, li2025rbench}.

In our prior benchmark ($973{,}400$ evaluations), nominal sequences ending in blur systematically outperformed sequences ending in JPEG by up to $21.55\%$ on ResNet-50 and $20.45\%$ on ConvNeXt-Tiny. However, low-pass filtering removes high-frequency JPEG block edges, conferring a $+5.0$ to $+6.2$ dB PSNR advantage. This creates a fundamental confounding: does corruption order matter to representation learning, or simply reflect divergent physical pixel fidelity?

\section{Simultaneous Common-Support Calibration}
To eliminate low-level fidelity confounding, we calibrated parameter triplets $( \sigma_{\text{noise}}, \sigma_{\text{blur}}, Q_{\text{JPEG}} )$ for all $3! = 6$ permutations on an independent pilot partition ($N_{\text{cal}} = 50$, zero test leakage) to intersect three simultaneous common-support bins:
\begin{itemize}
    \item \textbf{Level 1 (Mild):} Target PSNR $= 27.20$ dB, Target SSIM $= 0.810$.
    \item \textbf{Level 2 (Moderate):} Target PSNR $= 23.80$ dB, Target SSIM $= 0.650$.
    \item \textbf{Level 3 (Severe):} Target PSNR $= 21.80$ dB, Target SSIM $= 0.500$.
\end{itemize}
Prespecified joint tolerances were strictly enforced: $|\Delta \text{PSNR}| \le 0.75$ dB, $|\Delta \text{SSIM}| \le 0.035$. Learned Perceptual Image Patch Similarity (LPIPS) was computed post hoc as a continuous covariate.

\section{Confirmatory Statistical Analysis}
On the confirmatory cohort of $N = 3{,}825$ physical images ($19$ conditions, $4$ architectures, $290{,}700$ primary evaluations), common-support matching contracts accuracy gaps by $63.18\%$. To evaluate whether order effects persist beyond global fidelity, we fit three nested regression specifications on the continuous true-class logit margin $m(x) = z_{y^*} - \max_{j \ne y^*} z_j$:
\begin{align}
    \mathcal{M}_1 &: m(x) = \beta_0 + \beta_1 \text{PSNR} + \beta_2 \text{SSIM} + \epsilon \\
    \mathcal{M}_3 &: m(x) = \beta_0 + s(\text{PSNR}) + s(\text{SSIM}) + s(\text{LPIPS}) + \epsilon \\
    \mathcal{M}_5 &: \mathcal{M}_3 + \sum_{k=1}^5 \gamma_k f_k(x) + \epsilon
\end{align}
where $s(\cdot)$ represents 4-knot restricted cubic splines and $f_k(x)$ denote five physical distortion fingerprints:
\begin{enumerate}
    \item $E_{\text{low}}$: Fourier low-frequency spectral energy ($r < 0.15 f_{\text{Nyquist}}$).
    \item $E_{\text{high}}$: Fourier high-frequency spectral energy ($r > 0.50 f_{\text{Nyquist}}$).
    \item $B_{\text{JPEG}}$: Step discontinuity across $8 \times 8$ DCT grid borders.
    \item $\Delta E_{\text{edge}}$: Total Sobel edge energy differential relative to clean.
    \item $\sigma(\text{SSIM}_{11 \times 11})$: Spatial standard deviation of local SSIM.
\end{enumerate}

\begin{table}[t]
\caption{Confirmatory Grouped Cross-Validation Margin Prediction ($N = 3{,}825$)}
\label{tab:confirmatory}
\centering
\begin{tabular}{lcccc}
\toprule
Architecture & $\mathcal{M}_3$ $R^2$ & $\mathcal{M}_5$ $R^2$ & $\Delta R^2$ (95\% CI) & Holm $p$ \\
\midrule
ResNet-50 & 0.4487 & 0.5030 & +0.0543 [0.0471, 0.0617] & $< 0.0001$ \\
ConvNeXt-Tiny & 0.4612 & 0.5229 & +0.0617 [0.0543, 0.0694] & $< 0.0001$ \\
ViT-Base/16 & 0.4120 & 0.4880 & +0.0760 [0.0672, 0.0849] & $< 0.0001$ \\
Swin-Tiny & 0.3891 & 0.4933 & +0.1042 [0.0945, 0.1141] & $< 0.0001$ \\
\bottomrule
\end{tabular}
\end{table}

Under 10-fold grouped cross-validation (grouped on physical image clusters) with $B = 10{,}000$ paired cluster bootstrap resamples, Table~\ref{tab:confirmatory} proves that physical fingerprints explain statistically significant incremental predictive variance across all four vision backbones ($p < 0.0001$). Swin-Tiny benefits most ($\Delta R^2 = +0.1042$), while ViT-Base displays uniquely strong sensitivity to local SSIM variability ($\beta = -0.2704 \pm 0.063$, $p = 9.87 \times 10^{-12}$).

\section{Independent External Replication (ImageNet-V2)}
To test external generalization, we executed an independent replication on ImageNet-V2 ($N = 300$, $22{,}800$ forward passes). Applying step-down Holm family-wise error rate control, ViT-Base/16 is the sole architecture demonstrating statistically significant external pooled incremental gain ($\Delta R^2 = +0.0371$, 95\% CI $[+0.0065, +0.0675]$, $p_{\text{adj}} = 0.0340$). ResNet-50 ($p_{\text{adj}} = 0.1770$), ConvNeXt-Tiny ($p_{\text{adj}} = 0.2438$), and Swin-Tiny ($p_{\text{adj}} = 0.1768$) remain positive point estimates whose replication intervals span zero.

\section{Conclusion}
At matched image fidelity, corruption order still matters. However, order sensitivity is not an inscrutable non-commutative artifact: it is substantially mediated by physical frequency-domain and structural fingerprints that differentially challenge isotropic self-attention versus hierarchical inductive biases.

\begin{thebibliography}{10}
\bibitem{hendrycks2019robustness}
D.~Hendrycks and T.~Dietterich, ``Benchmarking neural network robustness to common corruptions,'' in \emph{ICLR}, 2019.
\bibitem{goyal2022dual}
S.~Goyal et~al., ``Dual corruptions: Evaluating vision models under composed shifts,'' in \emph{ICCP}, 2022.
\bibitem{li2025rbench}
Y.~Li et~al., ``R-Bench: Benchmarking multi-stage physical degradation in multimodal LLMs,'' \emph{IEEE JSTSP}, 2025.
\end{thebibliography}

\end{document}
"""

def main():
    tex_reports = REPORTS_DIR / "RobustLens_Academic_Manuscript.tex"
    tex_downloads = PUBLIC_DOWNLOADS / "RobustLens_Academic_Manuscript.tex"

    with open(tex_reports, "w", encoding="utf-8") as f:
        f.write(LATEX_CONTENT)
    with open(tex_downloads, "w", encoding="utf-8") as f:
        f.write(LATEX_CONTENT)

    print(f"Generated LaTeX manuscript: {tex_reports} ({len(LATEX_CONTENT)} bytes)")
    print(f"Copied to public downloads: {tex_downloads}")

if __name__ == "__main__":
    main()
