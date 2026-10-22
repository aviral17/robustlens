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
            An empirical research framework investigating operator non-commutativity, prediction churn, and physical distortion confounding across modern computer vision architectures.
          </p>
        </div>

        {/* Quick Metric Highlights */}
        <div className="mt-8 pt-8 border-t border-sky-200/60 dark:border-border/50 grid grid-cols-2 md:grid-cols-4 gap-4 sm:gap-6">
          <div className="flex flex-col gap-1">
            <span className="text-2xl sm:text-3xl font-extrabold text-blue-600 dark:text-blue-500">973,400</span>
            <span className="text-xs sm:text-sm text-slate-600 dark:text-muted-foreground font-medium">Forward Passes Evaluated</span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-2xl sm:text-3xl font-extrabold text-violet-600 dark:text-violet-500">3,925</span>
            <span className="text-xs sm:text-sm text-slate-600 dark:text-muted-foreground font-medium">Physical Validation Images</span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-2xl sm:text-3xl font-extrabold text-emerald-600 dark:text-emerald-500">63.18%</span>
            <span className="text-xs sm:text-sm text-slate-600 dark:text-muted-foreground font-medium">Range Gap Contraction</span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-2xl sm:text-3xl font-extrabold text-amber-600 dark:text-amber-500">40.78%</span>
            <span className="text-xs sm:text-sm text-slate-600 dark:text-muted-foreground font-medium">Nominal Label Churn</span>
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
        <div className="p-6 sm:p-8 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-4">
          <h3 className="text-lg font-bold text-foreground">Factorial Benchmark Architecture & Regimes</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-muted-foreground leading-relaxed">
            <div className="flex flex-col gap-2">
              <strong className="text-foreground">Dataset Partition & Deterministic Seeds:</strong>
              <p>
                Evaluated on the official FastAI Imagenette-v2 validation split (3,925 physical images across 10 canonical ImageNet classes). Every evaluation uses deterministically seed-matched noise realizations on identical physical images.
              </p>
              <strong className="text-foreground mt-2">Factorial Permutations:</strong>
              <p>
                All 3! = 6 permutations of Gaussian Noise (N), Defocus Blur (B), and JPEG Compression (J)—namely NBJ, NJB, BNJ, BJN, JNB, and JBN—evaluated across 5 severity levels.
              </p>
            </div>
            <div className="flex flex-col gap-2">
              <strong className="text-foreground">Nominal-Order Regime (486,700 passes):</strong>
              <p>
                Evaluates fixed standard parameters (ImageNet-C severities 1–5), assessing operational robustness when degradation parameters are unconstrained.
              </p>
              <strong className="text-foreground mt-2">Approximate-Fidelity Regime (486,700 passes):</strong>
              <p>
                Parameters pre-calibrated on an independent training partition (N_cal = 50) targeting common (PSNR, SSIM) support to reduce low-level physical confounding.
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
              Main Experimental Findings
            </h2>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-3">
            <span className="text-xs font-bold uppercase tracking-wider text-rose-500">Finding 1</span>
            <h3 className="font-bold text-lg text-foreground">Terminal Smoothing Confound</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Nominal evaluations show massive sequence gaps up to <strong>21.55% in ResNet-50</strong> (35.71% on BJN vs. 57.27% on JNB) and <strong>20.45% in ConvNeXt-Tiny</strong> (45.99% on BJN vs. 66.44% on NJB). These gaps are dominated by terminal operator identity: ending in Blur confers a +17.69% boost by filtering prior noise and JPEG artifacts, increasing PSNR by ~6.2 dB.
            </p>
          </div>

          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-3">
            <span className="text-xs font-bold uppercase tracking-wider text-blue-500">Finding 2</span>
            <h3 className="font-bold text-lg text-foreground">Gap Contraction Under Fidelity</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Targeting common (PSNR, SSIM) support contracts the cross-sequence range of the four-model mean accuracies by <strong>63.18%</strong> (from 18.43% down to 6.79%). Architecture-specific gap contractions average <strong>58.08%</strong> (77.15% in ConvNeXt, 72.62% in ResNet, 50.75% in ViT, and 31.80% in Swin).
            </p>
          </div>

          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-3">
            <span className="text-xs font-bold uppercase tracking-wider text-amber-500">Finding 3</span>
            <h3 className="font-bold text-lg text-foreground">Prediction Churn & Error Migration</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              <strong>Predicted-Label Churn</strong> (40.78% nominal, 34.74% targeted) is more than double <strong>Correctness-State Churn</strong> (16.93% nominal, 12.34% targeted) due to widespread <strong>Error Migration</strong> (E00_diff ≈ 23%, where models predict different incorrect classes across orders). Pairs with near-identical accuracy exhibit up to <strong>10.97% excess churn</strong>.
            </p>
          </div>
        </div>

        {/* Churn Definition Callout Card */}
        <div className="p-6 rounded-3xl bg-muted/40 border border-border/50 flex flex-col gap-2">
          <h4 className="font-bold text-base text-foreground flex items-center gap-2">
            <RotateCcw className="h-4 w-4 text-amber-500" />
            <span>Understanding Churn: Predicted-Label vs. Correctness-State Churn</span>
          </h4>
          <p className="text-sm text-muted-foreground leading-relaxed">
            In standard machine learning literature, <strong>correctness-state churn</strong> only records instances where a model switches between a correct and an incorrect prediction across two conditions. In contrast, <strong>predicted-label churn</strong> measures whether the actual class label assigned by the model changes—even if both predictions are wrong. Because vision models continuously rotate between distinct incorrect classes under different degradation sequences (Error Migration, E00_diff), aggregate accuracy substantially understates model instability under non-commutative distribution shift.
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
            <strong className="text-foreground">Delineation Note:</strong> The core scientific research findings presented in the paper and dashboard derive from the verified offline benchmark of 973,400 forward passes on 3,925 physical validation images. The interactive live-inference demo (/api/inference) is a lightweight playground connecting to an external Gradio endpoint for single user uploads, completely separated from the benchmark data.
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
            All code, statistical scripts, data schemas, and research deliverables are designed for complete verification:
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-start gap-3 p-4 rounded-2xl bg-muted/30 border border-border/40">
              <Check className="h-5 w-5 text-teal-500 shrink-0 mt-0.5" />
              <div className="flex flex-col gap-1">
                <span className="font-bold text-sm text-foreground">Zero-Fabrication Data Audit</span>
                <span className="text-xs text-muted-foreground leading-relaxed">
                  Every reported number traces directly to 973,400 raw evaluation records archived in open compressed CSV formats with open schemas.
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
    </main>
  );
}
