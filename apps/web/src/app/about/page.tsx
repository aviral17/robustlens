import Link from "next/link";
import {
  Layers,
  Shuffle,
  BarChart3,
  Activity,
  RotateCcw,
  Compass,
  ShieldCheck,
  CheckCircle2,
  Target,
  Sparkles,
  BookOpen,
  FileText,
  ExternalLink,
  Cpu,
  Scale,
  ArrowRight,
  Sliders,
  Check,
} from "lucide-react";

export default function AboutPage() {
  return (
    <main className="min-h-screen pt-28 sm:pt-32 pb-24 px-4 sm:px-6 max-w-5xl mx-auto flex flex-col gap-12 sm:gap-16">
      {/* 1. Hero Section with Tasteful Blue-to-Violet Accent Banner (Light: Refined Blue/Cyan; Dark: Deep Navy/Violet) */}
      <section className="relative overflow-hidden rounded-3xl border border-sky-200/70 bg-gradient-to-br from-sky-50 via-blue-50/40 to-cyan-50/60 p-8 sm:p-12 shadow-sm shadow-blue-500/5 backdrop-blur-sm dark:border-blue-500/25 dark:bg-gradient-to-br dark:from-blue-950/40 dark:via-card dark:to-violet-950/30 dark:shadow-none">
        {/* Decorative Ambient Glow (Light: Sky/Cyan; Dark: Blue/Violet) */}
        <div className="absolute top-0 right-0 -z-10 h-72 w-72 rounded-full bg-sky-400/15 blur-3xl pointer-events-none dark:bg-blue-500/10" />
        <div className="absolute bottom-0 left-0 -z-10 h-72 w-72 rounded-full bg-cyan-400/15 blur-3xl pointer-events-none dark:bg-violet-500/10" />

        <div className="flex flex-col gap-4 max-w-3xl">
          <div className="inline-flex items-center gap-2 rounded-full border border-sky-300/60 bg-sky-100/70 px-3.5 py-1 text-xs sm:text-sm font-semibold text-sky-800 dark:border-blue-500/30 dark:bg-blue-500/10 dark:text-blue-400 w-fit">
            <Sparkles className="h-3.5 w-3.5" />
            <span>Research Overview & Methodology</span>
          </div>

          <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-slate-900 dark:text-foreground font-[family-name:var(--font-poppins)]">
            About RobustLens
          </h1>

          <p className="text-lg sm:text-xl text-slate-600 dark:text-muted-foreground font-medium leading-relaxed">
            An empirical research framework investigating operator-order effects, paired prediction transitions, matched global image fidelity, spatial-frequency/structural fingerprints, and vision-model decision margins across modern computer vision architectures.
          </p>
        </div>

        {/* Quick Metric Highlights */}
        <div className="mt-8 pt-8 border-t border-sky-200/60 dark:border-border/50 grid grid-cols-2 md:grid-cols-4 gap-4 sm:gap-6">
          <div className="flex flex-col gap-1">
            <span className="text-2xl sm:text-3xl font-extrabold text-blue-600 dark:text-blue-500">1,286,900</span>
            <span className="text-xs sm:text-sm text-slate-600 dark:text-muted-foreground font-medium">Total Passes (2 Studies)</span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-2xl sm:text-3xl font-extrabold text-violet-600 dark:text-violet-500">313,500</span>
            <span className="text-xs sm:text-sm text-slate-600 dark:text-muted-foreground font-medium">Matched-Fidelity Passes (Nov 2026)</span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-2xl sm:text-3xl font-extrabold text-emerald-600 dark:text-emerald-500">973,400</span>
            <span className="text-xs sm:text-sm text-slate-600 dark:text-muted-foreground font-medium">Order Benchmark Passes (Oct 2026)</span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-2xl sm:text-3xl font-extrabold text-amber-600 dark:text-amber-500">4</span>
            <span className="text-xs sm:text-sm text-slate-600 dark:text-muted-foreground font-medium">Canonical Architectures</span>
          </div>
        </div>
      </section>

      {/* 2. What RobustLens Does */}
      <section className="flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-500/10 text-blue-500 border border-blue-500/20">
            <Activity className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground font-[family-name:var(--font-poppins)]">
              What RobustLens Does
            </h2>
          </div>
        </div>

        <div className="p-6 sm:p-8 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-4">
          <p className="text-base sm:text-lg text-muted-foreground leading-relaxed">
            RobustLens is an audited evaluation framework and research platform designed to measure how modern vision models respond when image degradations occur in <strong>compound sequences</strong> rather than isolated perturbations.
          </p>
          <p className="text-base sm:text-lg text-muted-foreground leading-relaxed">
            While conventional robustness benchmarks (such as ImageNet-C) treat corruptions as independent, unary operations or unidirectional pipelines, practical sensing applications in robotics, autonomous driving, and medical imaging encounter compounding digital and physical degradations. RobustLens provides the mathematical models, statistical inference protocols, and interactive tools needed to test whether degradation sequence alters model resilience, uncertainty calibration, and instance-level decision boundaries.
          </p>
        </div>
      </section>

      {/* 3. Research Question & Scientific Motivation */}
      <section className="flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet-500/10 text-violet-500 border border-violet-500/20">
            <Shuffle className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground font-[family-name:var(--font-poppins)]">
              Research Question & Scientific Motivation
            </h2>
          </div>
        </div>

        <div className="p-6 sm:p-8 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-6">
          <div className="flex flex-col gap-2">
            <h3 className="text-xl font-bold text-foreground">The Non-Commutativity of Physical & Digital Operators</h3>
            <p className="text-base sm:text-lg text-muted-foreground leading-relaxed">
              In image processing and physical signal acquisition, mathematical operators do not commute:
              <span className="font-mono text-sm sm:text-base text-foreground bg-muted/60 px-2 py-0.5 rounded mx-1.5 border border-border/40">
                T_A ∘ T_B ≠ T_B ∘ T_A
              </span>
              Spatial convolution (blur, <span className="font-semibold text-foreground">B</span>), additive noise (<span className="font-semibold text-foreground">N</span>), and block discrete cosine transform quantization (JPEG, <span className="font-semibold text-foreground">J</span>) produce markedly different pixel distributions and structural representations depending on their order of execution. For example, applying blur after noise smooths high-frequency perturbations, whereas adding noise after blur leaves pixel variance unmitigated.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
            <div className="p-5 rounded-2xl bg-muted/40 border border-border/40 flex flex-col gap-2">
              <span className="text-xs font-bold uppercase tracking-wider text-blue-500">Question 1</span>
              <h4 className="font-bold text-base text-foreground">Total Operational Effect</h4>
              <p className="text-sm text-muted-foreground leading-relaxed">
                How severely does execution order alter top-1 accuracy and calibration when perturbations occur under standard nominal ImageNet-C parameters?
              </p>
            </div>

            <div className="p-5 rounded-2xl bg-muted/40 border border-border/40 flex flex-col gap-2">
              <span className="text-xs font-bold uppercase tracking-wider text-violet-500">Question 2</span>
              <h4 className="font-bold text-base text-foreground">Fidelity-Conditioned Effect</h4>
              <p className="text-sm text-muted-foreground leading-relaxed">
                When output distortion is pre-calibrated to target common (PSNR, SSIM) support, do sequence accuracy gaps persist, and how much do they contract?
              </p>
            </div>

            <div className="p-5 rounded-2xl bg-muted/40 border border-border/40 flex flex-col gap-2">
              <span className="text-xs font-bold uppercase tracking-wider text-amber-500">Question 3</span>
              <h4 className="font-bold text-base text-foreground">Prediction Instability</h4>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Does operator sequence alter individual-sample predictions and failure sets even when aggregate benchmark accuracy appears invariant?
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 4. Models & Experimental Design */}
      <section className="flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/10 text-indigo-500 border border-indigo-500/20">
            <Layers className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground font-[family-name:var(--font-poppins)]">
              Models & Experimental Design
            </h2>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-lg text-foreground">ResNet-50</h3>
              <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-muted text-muted-foreground border border-border/50">
                CNN Baseline
              </span>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Classical deep residual network with standard 3×3 convolutions, residual shortcut connections, and 25.6M parameters. Clean validation top-1 accuracy: 87.36%.
            </p>
          </div>

          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-lg text-foreground">ConvNeXt-Tiny</h3>
              <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-blue-500/10 text-blue-500 border border-blue-500/20">
                Modern CNN
              </span>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Modernized pure convolutional network incorporating 7×7 depthwise convolutions, inverted bottlenecks, and LayerNorm (28.6M parameters). Clean top-1 accuracy: 87.13%.
            </p>
          </div>

          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-lg text-foreground">ViT-Base/16</h3>
              <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-purple-500/10 text-purple-500 border border-purple-500/20">
                Plain Transformer
              </span>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Standard isotropic Vision Transformer using global multi-head self-attention across non-overlapping 16×16 patches (86.6M parameters). Clean top-1 accuracy: 80.25%.
            </p>
          </div>

          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-lg text-foreground">Swin-Tiny</h3>
              <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
                Hierarchical ViT
              </span>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Hierarchical Vision Transformer with shifted local window self-attention and patch merging (28.3M parameters). Clean top-1 accuracy: 86.04%.
            </p>
          </div>
        </div>

        {/* Experimental Scope Detail Card */}
        <div className="p-6 sm:p-8 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-6">
          <div className="flex flex-col gap-1">
            <span className="text-xs font-bold uppercase tracking-wider text-primary">Two Rigorous Research Phases • 1,286,900 Total Passes</span>
            <h3 className="text-xl font-bold text-foreground">Dual Experimental Architecture & Evaluation Regimes</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-muted-foreground leading-relaxed">
            {/* Phase 2: Active November 2026 Study */}
            <div className="p-5 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-2.5">
              <span className="text-xs font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400">
                Phase 2 • Active Canonical Manuscript (November 2026)
              </span>
              <h4 className="font-bold text-base text-foreground">Matched-Fidelity Distortion Fingerprints</h4>
              <p>
                <strong>Cohort:</strong> N = 3,825 held-out Imagenette validation images + N = 300 ImageNet-V2 replication images = <strong>313,500 total forward passes</strong>.
              </p>
              <p>
                <strong>Common-Support Targets:</strong> 3 joint (PSNR, SSIM) target tiers: Mild (27.20 dB / 0.810), Moderate (23.80 dB / 0.650), and Severe (21.80 dB / 0.500) evaluated across all 6 sequence orders (18 corrupted conditions + 1 clean = 19 conditions/image).
              </p>
              <p>
                <strong>Statistical Objective:</strong> Evaluates whether 5 spatial-frequency and structural fingerprints (&Delta;E<sub>edge</sub>, E<sub>low</sub>, E<sub>high</sub>, B<sub>JPEG</sub>, &sigma;<sub>local SSIM</sub>) improve out-of-sample logit-margin prediction (Model 5, 32 cols) beyond nonlinear four-knot restricted cubic splines on PSNR, SSIM, and LPIPS (Model 3, 27 cols).
              </p>
            </div>

            {/* Phase 1: Foundational October 2026 Benchmark */}
            <div className="p-5 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-2.5">
              <span className="text-xs font-bold uppercase tracking-wider text-purple-600 dark:text-purple-400">
                Phase 1 • Foundational Benchmark (October 2026)
              </span>
              <h4 className="font-bold text-base text-foreground">Order Matters Beyond Accuracy</h4>
              <p>
                <strong>Cohort:</strong> N = 3,925 official Imagenette-v2 validation images across 2 complete regimes = <strong>973,400 total forward passes</strong>.
              </p>
              <p>
                <strong>Nominal Regime (486,700 passes):</strong> Fixed standard ImageNet-C parameters (severities 1–5), assessing operational robustness when degradation sequence is unconstrained.
              </p>
              <p>
                <strong>Approximate-Fidelity Regime (486,700 passes):</strong> Parameters targeted shared (PSNR, SSIM) bins on an independent partition (N<sub>cal</sub> = 50), but did not meet prespecified tolerances (PSNR RMSE 0.943 dB vs. &plusmn;0.75 dB tolerance; SSIM RMSE 0.1380 vs. &plusmn;0.035 tolerance). The resulting 63.18% aggregate range contraction remains descriptive. In contrast, the November 2026 calibrated common-support protocol successfully validated all 18/18 configurations within tolerance.
              </p>
              <p>
                <strong>Statistical Objective:</strong> Formulates 5-state joint prediction space separating correctness churn from error migration (E<sub>00_diff</sub>).
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 5. Main Experimental Findings */}
      <section className="flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
            <BarChart3 className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground font-[family-name:var(--font-poppins)]">
              Core Empirical Findings (Both Research Phases)
            </h2>
          </div>
        </div>

        {/* November 2026 Matched-Fidelity Findings */}
        <div className="flex flex-col gap-2 pt-1">
          <span className="text-xs font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400">
            Phase 2 Findings • Spatial-Frequency &amp; Structural Fingerprints (November 2026)
          </span>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-3">
              <span className="text-xs font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400">Finding 1 (Confirmatory)</span>
              <h3 className="font-bold text-lg text-foreground">Fingerprints Add Predictive Value Beyond Global Fidelity</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Adding five spatial-frequency/structural fingerprints improves out-of-sample logit margin R&sup2; by <strong>+0.0094 to +0.0199</strong> across all four architectures (ResNet-50, ConvNeXt-Tiny, ViT-Base/16, Swin-Tiny) and reduces MAE by 0.0334&ndash;0.0535. Paired image-cluster bootstrap tests (B = 10,000) confirm <strong>p<sub>adj</sub> &le; 0.0004</strong> with simultaneous Bonferroni 95% CIs strictly above zero.
              </p>
            </div>

            <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-3">
              <span className="text-xs font-bold uppercase tracking-wider text-blue-500">Finding 2 (Ablation)</span>
              <h3 className="font-bold text-lg text-foreground">Sobel Edge-Energy is the Strongest Contributor</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Feature ablations show that <strong>Sobel edge-energy change (&Delta;E<sub>edge</sub>)</strong> produces the largest single-feature R&sup2; gain in every architecture, with Fourier residual-energy features (E<sub>low</sub>, E<sub>high</sub>) providing a second significant contribution. Full sequence order and continuous fingerprints add distinct, non-redundant predictive detail.
              </p>
            </div>

            <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-3">
              <span className="text-xs font-bold uppercase tracking-wider text-violet-500">Finding 3 (Replication)</span>
              <h3 className="font-bold text-lg text-foreground">ViT-Base/16 Replicates on ImageNet-V2</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                On an independent 300-image ImageNet-V2 cohort (22,800 passes across MatchedFrequency, Threshold0.7, and TopImages), <strong>ViT-Base/16 retains a statistically significant pooled R&sup2; gain</strong> after Holm correction (&Delta;R&sup2; = +0.0371, 95% CI [0.0065, 0.0675], p<sub>adj</sub> = 0.0340); the other architectures show positive point estimates with intervals crossing zero.
              </p>
            </div>
          </div>
        </div>

        {/* October 2026 Foundational Findings */}
        <div className="flex flex-col gap-2 pt-4">
          <span className="text-xs font-bold uppercase tracking-wider text-purple-600 dark:text-purple-400">
            Phase 1 Findings • Compound Corruption Order Effects (October 2026)
          </span>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-3">
              <span className="text-xs font-bold uppercase tracking-wider text-rose-500">Finding 4 (Order Sensitivity)</span>
              <h3 className="font-bold text-lg text-foreground">Terminal Operator Effects &amp; Order Sensitivity</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                In the tested nominal regime, sequences ending in blur had higher average accuracy than sequences ending in noise (conferring a +17.69% boost across architectures), partly by smoothing prior high-frequency artifacts. Crucially, terminal operator identity alone does not explain all sequence variation: substantial accuracy differences persist even among sequences sharing the identical terminal operator (e.g., NBJ at 58.90% vs. BNJ at 42.88% in ConvNeXt-Tiny—a 16.02 percentage point divergence despite both terminating in JPEG). Sequence processing order remains a critical independent determinant of model resilience.
              </p>
            </div>

            <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-3">
              <span className="text-xs font-bold uppercase tracking-wider text-blue-500">Finding 5 (Fidelity Contraction)</span>
              <h3 className="font-bold text-lg text-foreground">Descriptive Gap Contraction Under Approximate Fidelity</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                In the October approximate-fidelity benchmark, equalizing shared (PSNR, SSIM) bins contracted the cross-sequence accuracy range of the four-model mean by a descriptive <strong>63.18%</strong> (from 18.43% down to 6.79%, with architecture-specific contractions averaging 58.08%). Because this exploratory regime did not satisfy prescribed tolerances (PSNR RMSE 0.943 dB, SSIM RMSE 0.1380), the November confirmatory study instituted a rigorously validated common-support protocol (18/18 configurations validated within &plusmn;0.75 dB PSNR, &plusmn;0.035 SSIM), confirming that spatial-frequency and structural fingerprints drive logit margins beyond global fidelity.
              </p>
            </div>

            <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-3">
              <span className="text-xs font-bold uppercase tracking-wider text-amber-500">Finding 6 (Churn Decomposition)</span>
              <h3 className="font-bold text-lg text-foreground">Prediction Churn & Error Migration</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                <strong>Predicted-Label Churn</strong> (40.78% nominal, 34.74% targeted) is more than double <strong>Correctness-State Churn</strong> (16.93% nominal, 12.34% targeted) due to widespread <strong>Error Migration</strong> (E<sub>00_diff</sub> &asymp; 23%, where models predict different incorrect classes across orders).
              </p>
            </div>
          </div>
        </div>

        {/* Churn Definition Callout Card */}
        <div className="p-6 rounded-3xl bg-muted/40 border border-border/50 flex flex-col gap-2 mt-2">
          <h4 className="font-bold text-base text-foreground flex items-center gap-2">
            <RotateCcw className="h-4 w-4 text-amber-500" />
            <span>Understanding Churn: Predicted-Label vs. Correctness-State Churn</span>
          </h4>
          <p className="text-sm text-muted-foreground leading-relaxed">
            In standard machine learning literature, <strong>correctness-state churn</strong> only records instances where a model switches between a correct and an incorrect prediction across two conditions. In contrast, <strong>predicted-label churn</strong> measures whether the actual class label assigned by the model changes—even if both predictions are wrong. Because vision models continuously rotate between distinct incorrect classes under different degradation sequences (Error Migration, E<sub>00_diff</sub>), aggregate accuracy substantially understates model instability under non-commutative distribution shift.
          </p>
        </div>
      </section>

      {/* 6. Statistical Evaluation Methods */}
      <section className="flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-amber-500/10 text-amber-500 border border-amber-500/20">
            <Scale className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground font-[family-name:var(--font-poppins)]">
              Statistical Evaluation Methods
            </h2>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-2">
            <h3 className="font-bold text-lg text-foreground">Image-Level Clustered Bootstrap</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              To account for repeated-measures correlation (each physical image is evaluated across 6 sequences and 5 severities), we perform clustered bootstrap resampling (B = 5,000 resamples) clustered strictly at the physical image level (N = 3,925 clusters). This prevents pseudoreplication and avoids deflating standard errors by √K.
            </p>
          </div>

          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-2">
            <h3 className="font-bold text-lg text-foreground">Full-Rank Saturated Six-Cell Decomposition</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              We specify a full-rank linear decomposition partitioning sequence effects into intercept, entry operators (Start Blur, Start JPEG), exit operators (End Blur, End JPEG), and a balanced parity contrast between even and odd permutations (det(X) = -6). Direct marginal contrasts are tested with asymptotic Wald p-values.
            </p>
          </div>

          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-2">
            <h3 className="font-bold text-lg text-foreground">Multiple Testing Multiplicity Control</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              All pairwise hypothesis tests (McNemar&apos;s paired tests and paired sign-flip permutation tests) are adjusted using Holm-Bonferroni (controlling Family-Wise Error Rate) and Benjamini-Hochberg (controlling False Discovery Rate) to guarantee statistical validity across extensive comparison grids.
            </p>
          </div>

          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-2">
            <h3 className="font-bold text-lg text-foreground">Calibration & Conformal Prediction</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              We compute Expected Calibration Error (ECE), evaluate temperature scaling reliability diagrams, and apply Split Conformal Prediction to produce prediction sets with finite-sample marginal coverage guarantees, tracking set size expansion as distribution shift severity intensifies.
            </p>
          </div>
        </div>
      </section>

      {/* 7. Interactive Website Features & Research Connection */}
      <section className="flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-500/10 text-blue-500 border border-blue-500/20">
            <Compass className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground font-[family-name:var(--font-poppins)]">
              Interactive Platform & Research Alignment
            </h2>
          </div>
        </div>

        <div className="p-6 sm:p-8 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-6">
          <p className="text-base sm:text-lg text-muted-foreground leading-relaxed">
            The RobustLens interactive website allows researchers and reviewers to explore the underlying empirical dataset, inspect single-image failure transitions, and verify statistical models across dedicated modules:
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <Link
              href="/order-effects"
              className="group p-5 rounded-2xl bg-muted/30 border border-border/50 hover:bg-muted/70 transition-all flex flex-col gap-2"
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-base text-foreground group-hover:text-primary transition-colors">
                  Order Effects
                </span>
                <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:translate-x-1 transition-transform" />
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Full 973,400-evaluation compound corruption dashboard, sequence rankings, nominal vs. matched fidelity comparisons, and ANCOVA models.
              </p>
            </Link>

            <Link
              href="/leaderboard"
              className="group p-5 rounded-2xl bg-muted/30 border border-border/50 hover:bg-muted/70 transition-all flex flex-col gap-2"
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-base text-foreground group-hover:text-primary transition-colors">
                  Leaderboard
                </span>
                <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:translate-x-1 transition-transform" />
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Comparative architecture rankings, mean compound accuracy, cross-sequence variance, and calibration scores.
              </p>
            </Link>

            <Link
              href="/explorer"
              className="group p-5 rounded-2xl bg-muted/30 border border-border/50 hover:bg-muted/70 transition-all flex flex-col gap-2"
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-base text-foreground group-hover:text-primary transition-colors">
                  Explorer
                </span>
                <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:translate-x-1 transition-transform" />
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Side-by-side inspection of clean vs. corrupted images with precomputed model predictions, softmax confidences, and severity sliders.
              </p>
            </Link>

            <Link
              href="/significance"
              className="group p-5 rounded-2xl bg-muted/30 border border-border/50 hover:bg-muted/70 transition-all flex flex-col gap-2"
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-base text-foreground group-hover:text-primary transition-colors">
                  Significance
                </span>
                <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:translate-x-1 transition-transform" />
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Paired McNemar tests, paired bootstrap confidence bounds, and multiple-testing adjusted p-values.
              </p>
            </Link>

            <Link
              href="/calibration"
              className="group p-5 rounded-2xl bg-muted/30 border border-border/50 hover:bg-muted/70 transition-all flex flex-col gap-2"
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-base text-foreground group-hover:text-primary transition-colors">
                  Calibration & ECE
                </span>
                <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:translate-x-1 transition-transform" />
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Reliability diagrams, confidence histograms, and post-hoc temperature scaling under progressive corruption.
              </p>
            </Link>

            <Link
              href="/methods"
              className="group p-5 rounded-2xl bg-muted/30 border border-border/50 hover:bg-muted/70 transition-all flex flex-col gap-2"
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-base text-foreground group-hover:text-primary transition-colors">
                  Methods
                </span>
                <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:translate-x-1 transition-transform" />
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Formal mathematical formulations, bootstrap procedures, and statistical rationale for the RobustLens evaluation suite.
              </p>
            </Link>
          </div>

          <div className="p-4 rounded-2xl bg-muted/30 border border-border/40 text-xs text-muted-foreground leading-relaxed">
            <strong className="text-foreground">Delineation Note:</strong> The core scientific research findings presented in the publications and dashboard derive from the verified offline evaluations comprising <strong>1,286,900 forward passes</strong> (973,400 in the foundational benchmark across 3,925 validation images, and 313,500 in the matched-fidelity confirmatory and replication study across 3,825 held-out Imagenette images + 300 ImageNet-V2 images). The interactive live-inference playground (<code className="text-foreground font-mono">/api/inference</code>) is a lightweight demonstration connecting to an external GPU endpoint for single user uploads, completely separated from the audited research data.
          </div>
        </div>
      </section>

      {/* 8. Reproducibility & Data Availability */}
      <section className="flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-teal-500/10 text-teal-500 border border-teal-500/20">
            <ShieldCheck className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground font-[family-name:var(--font-poppins)]">
              Reproducibility & Data Availability
            </h2>
          </div>
        </div>

        <div className="p-6 sm:p-8 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-6">
          <p className="text-base sm:text-lg text-muted-foreground leading-relaxed">
            All code, statistical scripts, data schemas, and research deliverables are designed for complete independent verification:
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-start gap-3 p-4 rounded-2xl bg-muted/30 border border-border/40">
              <Check className="h-5 w-5 text-teal-500 shrink-0 mt-0.5" />
              <div className="flex flex-col gap-1">
                <span className="font-bold text-sm text-foreground">Zero-Fabrication Data Audit</span>
                <span className="text-xs text-muted-foreground leading-relaxed">
                  Every reported finding traces directly to raw evaluation records (973,400 in the foundational benchmark and 313,500 in the matched-fidelity confirmatory study) archived in open compressed CSV formats with cryptographic manifests.
                </span>
              </div>
            </div>

            <div className="flex items-start gap-3 p-4 rounded-2xl bg-muted/30 border border-border/40">
              <Check className="h-5 w-5 text-teal-500 shrink-0 mt-0.5" />
              <div className="flex flex-col gap-1">
                <span className="font-bold text-sm text-foreground">One-Click Numerical Verification</span>
                <span className="text-xs text-muted-foreground leading-relaxed">
                  The script <code className="text-foreground font-mono">verify_all_numbers.py</code> audits every transition cell, accuracy gap, and row count in &lt;5 seconds.
                </span>
              </div>
            </div>

            <div className="flex items-start gap-3 p-4 rounded-2xl bg-muted/30 border border-border/40">
              <Check className="h-5 w-5 text-teal-500 shrink-0 mt-0.5" />
              <div className="flex flex-col gap-1">
                <span className="font-bold text-sm text-foreground">52-Test Automated Pytest Suite</span>
                <span className="text-xs text-muted-foreground leading-relaxed">
                  Automated test suite covers clustered bootstrap, McNemar, Holm/BH corrections, ECE, conformal bounds, and full-rank matrix algebra.
                </span>
              </div>
            </div>

            <div className="flex items-start gap-3 p-4 rounded-2xl bg-muted/30 border border-border/40">
              <Check className="h-5 w-5 text-teal-500 shrink-0 mt-0.5" />
              <div className="flex flex-col gap-1">
                <span className="font-bold text-sm text-foreground">Self-Contained Research Package</span>
                <span className="text-xs text-muted-foreground leading-relaxed">
                  Canonical manuscript Markdown, ReportLab PDF builder, 300 DPI figures, and literature review matrix preserved in <code className="text-foreground font-mono">reports/final_research_package/</code>.
                </span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 9. Scope & Future Directions */}
      <section className="flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-rose-500/10 text-rose-500 border border-rose-500/20">
            <Target className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground font-[family-name:var(--font-poppins)]">
              Scope & Future Research Directions
            </h2>
          </div>
        </div>

        <div className="p-6 sm:p-8 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <h3 className="text-lg font-bold text-foreground">Academic Scope & Methodological Boundaries</h3>
            <p className="text-sm sm:text-base text-muted-foreground leading-relaxed">
              The present benchmark is evaluated on the 10-class FastAI Imagenette-v2 validation partition (3,925 images) to make exhaustive 6-sequence factorial evaluation with seed-matched noise realizations computationally feasible on accessible academic hardware. While Imagenette preserves the diversity and resolution of ImageNet-1k, broader 1,000-class validation remains an exciting avenue for scaled cluster computation.
            </p>
          </div>

          <div className="flex flex-col gap-2 pt-2 border-t border-border/50">
            <h3 className="text-lg font-bold text-foreground">Future Horizons</h3>
            <ul className="text-sm sm:text-base text-muted-foreground leading-relaxed list-disc list-inside space-y-1.5">
              <li>
                <strong>Continuous Common Support Matching:</strong> Exploring continuous generative distortion projection (e.g., neural style transfer or diffusion-based matching) to tighten joint (PSNR, SSIM) support beyond discrete parameter grids.
              </li>
              <li>
                <strong>Dense Prediction & Temporal Video Streams:</strong> Extending sequence permutation analysis to object detection, semantic segmentation, and temporal video perception in autonomous driving stacks.
              </li>
              <li>
                <strong>Order-Aware Robust Regularization:</strong> Formulating multi-stage training objectives that explicitly regularize against sequence non-commutativity and error migration.
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* 10. Publications, Research Deliverables & Citations */}
      <section className="flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-500/10 text-purple-500 border border-purple-500/20">
            <BookOpen className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground font-[family-name:var(--font-poppins)]">
              Publications, Academic Deliverables & Citations
            </h2>
          </div>
        </div>

        <div className="p-6 sm:p-8 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-6">
          {/* New November 2026 Matched-Fidelity Study */}
          <div className="flex flex-col gap-4 pb-6 border-b border-border/50">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div className="flex flex-col gap-1.5">
                <span className="text-xs font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400">
                  New Scientific Manuscript (November 2026) • Matched-Fidelity Distortion Fingerprints
                </span>
                <h3 className="text-lg sm:text-xl font-extrabold text-foreground">
                  Spatial-Frequency and Structural Distortion Fingerprints Beyond Global Image Fidelity: Predicting Vision-Model Margins under Compound Image Corruption
                </h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  <strong>Authors:</strong> Suryakanth V. Gangashetty<sup>1</sup> &amp; Aviral Choudhary<sup>2</sup><br />
                  <span className="text-xs text-muted-foreground">
                    <sup>1</sup>Department of Computer Science and Engineering, Koneru Lakshmaiah Education Foundation, India &bull; <sup>2</sup>Independent Researcher, India
                  </span>
                </p>
                <p className="text-xs font-semibold text-emerald-600 dark:text-emerald-400">
                  Cohort: N = 3,825 Confirmatory Images (290,700 passes) + N = 300 ImageNet-V2 Replication (22,800 passes) = 313,500 Total Forward Passes
                </p>
              </div>
            </div>

            {/* Direct Document Downloads */}
            <div className="flex flex-wrap gap-3 pt-2">
              <a
                href="/downloads/RobustLens_Academic_Manuscript.pdf"
                download="RobustLens_Matched_Fidelity_Manuscript_Final.pdf"
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-full text-xs font-bold bg-primary text-primary-foreground hover:bg-primary/90 transition-all shadow-sm"
              >
                <FileText className="h-4 w-4" />
                <span>Download Canonical Manuscript (8-Page PDF with Vector Figures)</span>
              </a>
              <a
                href="/downloads/RobustLens_Final_Research_Report.pdf"
                download="RobustLens_Final_Research_Report.pdf"
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-full text-xs font-bold bg-muted hover:bg-muted/80 text-foreground transition-all border border-border/60"
              >
                <FileText className="h-4 w-4" />
                <span>Download Comprehensive Research Report (10-Page PDF)</span>
              </a>
              <a
                href="/downloads/RobustLens_Academic_Manuscript.docx"
                download="RobustLens_Academic_Manuscript.docx"
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-full text-xs font-bold bg-muted hover:bg-muted/80 text-foreground transition-all border border-border/60"
              >
                <FileText className="h-4 w-4" />
                <span>Download Academic Manuscript (Editable DOCX)</span>
              </a>
              <a
                href="/downloads/RobustLens_Academic_Manuscript.tex"
                download="RobustLens_Matched_Fidelity_Manuscript_Final.tex"
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-full text-xs font-bold bg-muted hover:bg-muted/80 text-foreground transition-all border border-border/60 font-mono"
              >
                <FileText className="h-4 w-4" />
                <span>Download LaTeX Source (.tex)</span>
              </a>
              <a
                href="/downloads/RobustLens_Matched_Fidelity_Final_Source.zip"
                download="RobustLens_Matched_Fidelity_Final_Source.zip"
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-full text-xs font-bold bg-muted hover:bg-muted/80 text-foreground transition-all border border-border/60"
              >
                <FileText className="h-4 w-4" />
                <span>Download Full LaTeX Source + Vector Figures (.zip)</span>
              </a>
            </div>

            <div className="flex flex-col gap-2 pt-2">
              <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">BibTeX Citation (November 2026 Manuscript)</span>
              <div className="p-4 rounded-2xl bg-muted/50 border border-border/50 font-mono text-xs text-foreground overflow-x-auto leading-relaxed">
                <pre>{`@article{gangashetty2026spatialfrequency,
  title   = {Spatial-Frequency and Structural Distortion Fingerprints Beyond Global Image Fidelity: Predicting Vision-Model Margins under Compound Image Corruption},
  author  = {Gangashetty, Suryakanth V. and Choudhary, Aviral},
  year    = {2026},
  month   = {November},
  journal = {RobustLens Research Initiative Working Paper},
  note    = {Confirmatory study N=3,825 held-out Imagenette-v2 images, 290,700 primary passes + ImageNet-V2 replication}
}`}</pre>
              </div>
            </div>
          </div>

          {/* October 2026 Benchmark Release */}
          <div className="flex flex-col gap-4 pb-4 border-b border-border/50">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div className="flex flex-col gap-1.5">
                <span className="text-xs font-bold uppercase tracking-wider text-purple-500">Benchmark Release (v1.0 • October 2026)</span>
                <h3 className="text-lg sm:text-xl font-bold text-foreground">
                  Order Matters Beyond Accuracy: Paired Prediction Analysis of Compound Image Corruptions
                </h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  <strong>Authors:</strong> Suryakanth V. Gangashetty<sup>1</sup> &amp; Aviral Choudhary<sup>2</sup> &bull; <strong>Date:</strong> October 2026<br />
                  <span className="text-xs text-muted-foreground">
                    <sup>1</sup>Department of Computer Science and Engineering, Koneru Lakshmaiah Education Foundation, India &bull; <sup>2</sup>Independent Researcher, India
                  </span>
                </p>
                <p className="text-xs font-semibold text-purple-600 dark:text-purple-400">
                  Cohort: N = 3,925 Validation Images across 2 Regimes (Nominal &amp; Targeted) = 973,400 Total Forward Passes
                </p>
              </div>
              <span className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-bold bg-purple-500/10 text-purple-600 dark:text-purple-400 border border-purple-500/20 shrink-0 self-start sm:self-auto">
                <FileText className="h-3.5 w-3.5" />
                <span>Foundational Paper &amp; Archive</span>
              </span>
            </div>

            {/* Direct Document Downloads for Historical Release */}
            <div className="flex flex-wrap gap-3 pt-1">
              <a
                href="/downloads/RobustLens_Final_Paper.pdf"
                download="RobustLens_Foundational_Paper.pdf"
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-full text-xs font-bold bg-muted hover:bg-muted/80 text-foreground transition-all border border-border/60"
              >
                <FileText className="h-4 w-4" />
                <span>Download Foundational Paper (PDF, 6 Pages)</span>
              </a>
              <a
                href="/downloads/RobustLens_Foundational_Paper.tex"
                download="RobustLens_Foundational_Paper.tex"
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-full text-xs font-bold bg-muted hover:bg-muted/80 text-foreground transition-all border border-border/60 font-mono"
              >
                <FileText className="h-4 w-4" />
                <span>Download Foundational LaTeX Source (.tex)</span>
              </a>
              <a
                href="/downloads/RobustLens_Research_Package.zip"
                download="RobustLens_Research_Package.zip"
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-full text-xs font-bold bg-muted hover:bg-muted/80 text-foreground transition-all border border-border/60"
              >
                <FileText className="h-4 w-4" />
                <span>Download Complete Research Archive (.zip)</span>
              </a>
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">BibTeX Citation (October 2026 Foundational Paper)</span>
            <div className="p-4 rounded-2xl bg-muted/50 border border-border/50 font-mono text-xs text-foreground overflow-x-auto leading-relaxed">
              <pre>{`@article{gangashetty2026ordermatters,
  title   = {Order Matters Beyond Accuracy: Paired Prediction Analysis of Compound Image Corruptions},
  author  = {Gangashetty, Suryakanth V. and Choudhary, Aviral},
  year    = {2026},
  month   = {October},
  journal = {RobustLens Research Initiative Working Paper},
  version = {1.0},
  note    = {Foundational compound-corruption benchmark: 973,400 forward passes on 3,925 Imagenette-v2 images}
}`}</pre>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
