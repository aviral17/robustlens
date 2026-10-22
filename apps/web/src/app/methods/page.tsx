import Link from "next/link";
import {
  Activity,
  Shuffle,
  Layers,
  BarChart3,
  Scale,
  ShieldCheck,
  Check,
  Sparkles,
  AlertCircle,
  ArrowRight,
  RotateCcw,
  Sliders,
  Compass,
  Cpu,
  Target,
  ExternalLink,
} from "lucide-react";

export default function MethodsPage() {
  return (
    <main className="min-h-screen pt-28 sm:pt-32 pb-24 px-4 sm:px-6 max-w-5xl mx-auto flex flex-col gap-12 sm:gap-16">
      {/* 1. Page Hero with Violet-to-Periwinkle Accent Treatment (Light: Lavender/Periwinkle; Dark: Deep Navy/Violet) */}
      <section className="relative overflow-hidden rounded-3xl border border-violet-200/70 bg-gradient-to-br from-violet-50 via-purple-50/40 to-indigo-50/60 p-8 sm:p-12 shadow-sm shadow-violet-500/5 backdrop-blur-sm dark:border-blue-500/25 dark:bg-gradient-to-br dark:from-blue-950/40 dark:via-card dark:to-violet-950/30 dark:shadow-none">
        {/* Decorative Ambient Glow (Light: Violet/Indigo; Dark: Blue/Violet) */}
        <div className="absolute top-0 right-0 -z-10 h-72 w-72 rounded-full bg-violet-400/15 blur-3xl pointer-events-none dark:bg-blue-500/10" />
        <div className="absolute bottom-0 left-0 -z-10 h-72 w-72 rounded-full bg-indigo-400/15 blur-3xl pointer-events-none dark:bg-violet-500/10" />

        <div className="flex flex-col gap-4 max-w-3xl">
          <div className="inline-flex items-center gap-2 rounded-full border border-violet-300/60 bg-violet-100/70 px-3.5 py-1 text-xs sm:text-sm font-semibold text-violet-800 dark:border-blue-500/30 dark:bg-blue-500/10 dark:text-blue-400 w-fit">
            <Sparkles className="h-3.5 w-3.5" />
            <span>Research Methodology & Mathematical Formulations</span>
          </div>

          <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-slate-900 dark:text-foreground font-[family-name:var(--font-poppins)]">
            Methodology
          </h1>

          <p className="text-lg sm:text-xl text-slate-600 dark:text-muted-foreground font-medium leading-relaxed">
            Theoretical foundations, statistical models, and executable inference pipelines for evaluating computer vision architectures under non-commutative compound corruptions.
          </p>
        </div>

        {/* Pipeline Overview Pills */}
        <div className="mt-8 pt-8 border-t border-violet-200/60 dark:border-border/50 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          <div className="p-3 rounded-2xl bg-white/70 dark:bg-muted/40 border border-violet-200/60 dark:border-border/40 shadow-xs dark:shadow-none flex flex-col gap-0.5">
            <span className="text-[11px] font-bold uppercase tracking-wider text-blue-600 dark:text-blue-500">Stage 1</span>
            <span className="text-xs font-semibold text-slate-800 dark:text-foreground">Paired Inputs</span>
          </div>
          <div className="p-3 rounded-2xl bg-white/70 dark:bg-muted/40 border border-violet-200/60 dark:border-border/40 shadow-xs dark:shadow-none flex flex-col gap-0.5">
            <span className="text-[11px] font-bold uppercase tracking-wider text-cyan-600 dark:text-cyan-500">Stage 2</span>
            <span className="text-xs font-semibold text-slate-800 dark:text-foreground">Permutation Regimes</span>
          </div>
          <div className="p-3 rounded-2xl bg-white/70 dark:bg-muted/40 border border-violet-200/60 dark:border-border/40 shadow-xs dark:shadow-none flex flex-col gap-0.5">
            <span className="text-[11px] font-bold uppercase tracking-wider text-amber-600 dark:text-amber-500">Stage 3</span>
            <span className="text-xs font-semibold text-slate-800 dark:text-foreground">Instability Space</span>
          </div>
          <div className="p-3 rounded-2xl bg-white/70 dark:bg-muted/40 border border-violet-200/60 dark:border-border/40 shadow-xs dark:shadow-none flex flex-col gap-0.5">
            <span className="text-[11px] font-bold uppercase tracking-wider text-violet-600 dark:text-violet-500">Stage 4</span>
            <span className="text-xs font-semibold text-slate-800 dark:text-foreground">Inference & Tests</span>
          </div>
          <div className="p-3 rounded-2xl bg-white/70 dark:bg-muted/40 border border-violet-200/60 dark:border-border/40 shadow-xs dark:shadow-none flex flex-col gap-0.5 col-span-2 sm:col-span-1">
            <span className="text-[11px] font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-500">Stage 5</span>
            <span className="text-xs font-semibold text-slate-800 dark:text-foreground">Calibration & Sets</span>
          </div>
        </div>
      </section>

      {/* Stage 1: Experimental Setup & Paired Input Structure */}
      <section className="flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-500/10 text-blue-500 border border-blue-500/20">
            <Layers className="h-5 w-5" />
          </div>
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-blue-500">Pipeline Stage 1</span>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground font-[family-name:var(--font-poppins)]">
              Experimental Setup & Paired Input Structure
            </h2>
          </div>
        </div>

        <div className="p-6 sm:p-8 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-6">
          <p className="text-base sm:text-lg text-muted-foreground leading-relaxed">
            Conventional benchmarks often evaluate corruptions on uncoupled samples or randomized splits, conflating sample variance with architectural resilience. RobustLens enforces a strict <strong>paired-input experimental design</strong>: every model is evaluated on identical physical images across all sequence permutations, using deterministically seed-matched noise realizations.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-5 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-2">
              <h3 className="font-bold text-base text-foreground">Dataset Partition & Units of Analysis</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                The canonical benchmark is conducted on the official <strong>FastAI Imagenette-v2 validation split</strong> (<span className="font-mono text-foreground font-semibold">N = 3,925</span> physical images across 10 canonical ImageNet categories). The fundamental unit of statistical analysis is the physical image ID, forming natural clusters across repeated evaluations.
              </p>
            </div>

            <div className="p-5 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-2">
              <h3 className="font-bold text-base text-foreground">Architectural Diversity (4 Paradigms)</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Evaluates four representative vision paradigms: classical convolutional (<strong>ResNet-50</strong>, 25.6M params), modern depthwise convolutional (<strong>ConvNeXt-Tiny</strong>, 28.6M params), isotropic self-attention (<strong>ViT-Base/16</strong>, 86.6M params), and hierarchical shifted-window attention (<strong>Swin-Tiny</strong>, 28.3M params).
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Stage 2: Compound-Corruption Permutations & Regimes */}
      <section className="flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-500/10 text-cyan-500 border border-cyan-500/20">
            <Shuffle className="h-5 w-5" />
          </div>
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-cyan-500">Pipeline Stage 2</span>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground font-[family-name:var(--font-poppins)]">
              Compound Permutations & Experimental Regimes
            </h2>
          </div>
        </div>

        <div className="p-6 sm:p-8 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-6">
          <div className="flex flex-col gap-2">
            <h3 className="text-xl font-bold text-foreground">Symmetric Group Permutation Design</h3>
            <p className="text-base sm:text-lg text-muted-foreground leading-relaxed">
              We evaluate the complete symmetric group <span className="font-serif italic text-foreground">S₃</span> of three canonical image processing operators: Gaussian Noise (<span className="font-semibold text-foreground">N</span>), Defocus Blur (<span className="font-semibold text-foreground">B</span>), and JPEG Compression (<span className="font-semibold text-foreground">J</span>). This yields all <span className="font-mono text-foreground font-semibold">3! = 6</span> execution sequences across 5 severity levels:
            </p>
            <div className="flex flex-wrap gap-2 pt-2">
              {["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"].map((seq, idx) => (
                <span
                  key={seq}
                  className="px-3.5 py-1.5 rounded-xl font-mono text-xs sm:text-sm font-bold bg-muted/80 text-foreground border border-border/60"
                >
                  {seq} {idx % 2 === 0 ? "(Even, A₃)" : "(Odd, Transposition)"}
                </span>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
            <div className="p-6 rounded-2xl bg-muted/30 border border-border/50 flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-cyan-500">Regime 1</span>
                <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                  486,700 passes
                </span>
              </div>
              <h4 className="font-bold text-lg text-foreground">Nominal-Order Regime</h4>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Applies standard ImageNet-C operational parameters across severities 1–5. Captures the <strong>Total Operational Effect</strong> of operator sequencing in unconstrained environments, where downstream distortion shifts form part of the natural operational pipeline.
              </p>
            </div>

            <div className="p-6 rounded-2xl bg-muted/30 border border-border/50 flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-violet-500">Regime 2</span>
                <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded-full bg-violet-500/10 text-violet-400 border border-violet-500/20">
                  486,700 passes
                </span>
              </div>
              <h4 className="font-bold text-lg text-foreground">PSNR/SSIM-Targeted Regime</h4>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Pre-calibrates corruption parameters on an independent training partition (<span className="font-mono text-foreground">N_cal = 50</span>) to target five common (PSNR, SSIM) support bins. Evaluates the <strong>Fidelity-Conditioned Effect</strong> to assess how much cross-sequence variation contracts when low-level distortion is equalized.
              </p>
            </div>
          </div>

          {/* Methodological Qualification Callout */}
          <div className="p-5 rounded-2xl bg-amber-500/10 border border-amber-500/25 flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-amber-500 shrink-0 mt-0.5" />
            <div className="flex flex-col gap-1 text-xs sm:text-sm text-muted-foreground leading-relaxed">
              <strong className="text-foreground">Methodological Qualification on Approximate Fidelity:</strong>
              Because prespecified calibration tolerances (±0.75 dB PSNR, ±0.035 SSIM) were not achieved over the discrete parameter search grid (achieving PSNR RMSE 0.943 dB and SSIM RMSE 0.1380 due to the trade-off between smoothing and edge preservation), this regime is strictly an <strong>approximate-fidelity descriptive comparison</strong>. It demonstrates that low-level distortion equalizing contracts performance gaps by 63.18%, but does not claim complete elimination of confounding or exact causal isolation.
            </div>
          </div>
        </div>
      </section>

      {/* Stage 3: Accuracy, Fidelity, and Instability Measurements */}
      <section className="flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-amber-500/10 text-amber-500 border border-amber-500/20">
            <RotateCcw className="h-5 w-5" />
          </div>
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-amber-500">Pipeline Stage 3</span>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground font-[family-name:var(--font-poppins)]">
              Accuracy, Fidelity & 5-State Instability Partition
            </h2>
          </div>
        </div>

        <div className="p-6 sm:p-8 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-6">
          <p className="text-base sm:text-lg text-muted-foreground leading-relaxed">
            Aggregate top-1 accuracy alone masks significant behavioral shifts across sequence permutations. RobustLens formalizes a complete <strong>5-state joint prediction space decomposition</strong> to separate shifts in correctness from shifts in predicted class identity.
          </p>

          {/* 5-State Transition Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
            <div className="p-4 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-1">
              <span className="font-mono text-sm font-bold text-emerald-500">E₁₁ (Both Correct)</span>
              <p className="text-xs text-muted-foreground">Both sequences predict the ground-truth label correctly.</p>
              <span className="text-xs font-mono text-foreground mt-1">43.4% Nom / 47.9% Match</span>
            </div>
            <div className="p-4 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-1">
              <span className="font-mono text-sm font-bold text-rose-500">E₁₀ (Loss / L)</span>
              <p className="text-xs text-muted-foreground">Sequence A is correct; Sequence B is incorrect.</p>
              <span className="text-xs font-mono text-foreground mt-1">11.0% Nom / 5.8% Match</span>
            </div>
            <div className="p-4 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-1">
              <span className="font-mono text-sm font-bold text-blue-500">E₀₁ (Gain / G)</span>
              <p className="text-xs text-muted-foreground">Sequence A is incorrect; Sequence B is correct.</p>
              <span className="text-xs font-mono text-foreground mt-1">6.0% Nom / 6.6% Match</span>
            </div>
            <div className="p-4 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-1">
              <span className="font-mono text-sm font-bold text-muted-foreground">E₀₀_same (Same Wrong)</span>
              <p className="text-xs text-muted-foreground">Both sequences predict the identical wrong class.</p>
              <span className="text-xs font-mono text-foreground mt-1">15.8% Nom / 17.4% Match</span>
            </div>
            <div className="p-4 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-1">
              <span className="font-mono text-sm font-bold text-amber-500">E₀₀_diff (Error Migration)</span>
              <p className="text-xs text-muted-foreground">Both sequences fail, but predict different incorrect classes.</p>
              <span className="text-xs font-mono text-foreground mt-1">23.8% Nom / 22.4% Match</span>
            </div>
          </div>

          {/* Mathematical Formulations */}
          <div className="flex flex-col gap-4 pt-2">
            <h3 className="text-lg font-bold text-foreground">Formal Theorem of Prediction Instability</h3>

            <div className="p-5 rounded-2xl bg-muted/50 border border-border/50 font-mono text-xs sm:text-sm text-foreground flex flex-col gap-3 overflow-x-auto">
              <div>
                <span className="text-muted-foreground">// Correctness-State Churn (transitions across binary accuracy):</span>
                <p className="font-bold text-blue-400 mt-0.5">
                  Correctness_Churn = P(C_A ≠ C_B) = P(E₁₀) + P(E₀₁) = L + G
                </p>
              </div>

              <div className="pt-2 border-t border-border/40">
                <span className="text-muted-foreground">// Predicted-Label Churn (transitions across predicted class space):</span>
                <p className="font-bold text-amber-400 mt-0.5">
                  Label_Churn = P(Ŷ_A ≠ Ŷ_B) = Correctness_Churn + P(E₀₀_diff)
                </p>
              </div>

              <div className="pt-2 border-t border-border/40">
                <span className="text-muted-foreground">// Excess Churn (instability occurring beyond net accuracy differences):</span>
                <p className="font-bold text-violet-400 mt-0.5">
                  Excess_Label_Churn = Label_Churn - |ΔAcc| = 2 · min(G, L) + P(E₀₀_diff)
                </p>
              </div>
            </div>

            <p className="text-sm text-muted-foreground leading-relaxed">
              <strong>Key Insight:</strong> Predicted-label churn (34.74%–40.78%) is more than double correctness-state churn (12.34%–16.93%) because models frequently rotate among distinct incorrect labels under non-commutative perturbations (Error Migration, <span className="font-mono text-foreground">E₀₀_diff ≈ 23%</span>). Even sequence pairs with near-zero accuracy difference exhibit up to 10.97% excess churn.
            </p>
          </div>
        </div>
      </section>

      {/* Stage 4: Statistical Inference & Uncertainty Estimation */}
      <section className="flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet-500/10 text-violet-500 border border-violet-500/20">
            <Scale className="h-5 w-5" />
          </div>
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-violet-500">Pipeline Stage 4</span>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground font-[family-name:var(--font-poppins)]">
              Statistical Inference & Multiplicity Control
            </h2>
          </div>
        </div>

        <div className="p-6 sm:p-8 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="flex flex-col gap-2">
              <h3 className="font-bold text-lg text-foreground">Image-Level Clustered Bootstrap</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Because each of the 3,925 physical images is evaluated repeatedly across multiple sequences and severities, standard i.i.d. pooling would introduce severe pseudoreplication, deflating standard errors by a factor of approximately √K.
              </p>
              <p className="text-sm text-muted-foreground leading-relaxed">
                RobustLens performs <strong>image-level clustered bootstrap resampling</strong> (<span className="font-mono text-foreground font-semibold">B = 5,000 resamples</span>) clustered strictly at the physical image level (<span className="font-mono text-foreground">N = 3,925 clusters</span>). All condition observations for each resampled image are drawn together as a block, preserving repeated-measures covariance and yielding valid 95% percentile confidence bounds.
              </p>
            </div>

            <div className="flex flex-col gap-2">
              <h3 className="font-bold text-lg text-foreground">Saturated 6-Cell Linear Decomposition</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                To rigorously attribute sequence variation, we formulate a full-rank linear decomposition:
              </p>
              <div className="p-3.5 rounded-xl bg-muted/50 border border-border/40 font-mono text-xs text-foreground">
                μ = X · β  ⟺  β = X⁻¹ · μ,   det(X) = -6 ≠ 0
              </div>
              <p className="text-sm text-muted-foreground leading-relaxed">
                The parameters β partition sequence variance into Intercept (baseline), Start Blur (β₁), Start JPEG (β₂), End Blur (β₃), End JPEG (β₄), and a balanced Parity Contrast (β₅) between even and odd permutations, supplemented by direct marginal contrast tests.
              </p>
            </div>
          </div>

          {/* McNemar and Hypothesis Tests */}
          <div className="pt-4 border-t border-border/50 grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="flex flex-col gap-2">
              <h3 className="font-bold text-base text-foreground">Paired McNemar & Permutation Tests</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                To test pairwise sequence differences on identical physical images, we employ <strong>McNemar&apos;s paired test</strong> on discordant outcomes (<span className="font-mono text-foreground">χ² = (|L - G| - 1)² / (L + G)</span>, df = 1) and non-directional <strong>Monte Carlo sign-flip permutation tests</strong> (<span className="font-mono text-foreground">R = 20,000 permutations</span>) with Rademacher sign randomizations.
              </p>
            </div>

            <div className="flex flex-col gap-2">
              <h3 className="font-bold text-base text-foreground">Multiplicity Adjustments in Log Space</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                To control familywise error rates across large hypothesis grids without numerical truncation, all Wald test p-values are calculated via the Gaussian survival function in log space. Multiplicity is adjusted using:
              </p>
              <ul className="text-xs sm:text-sm text-muted-foreground list-disc list-inside space-y-1">
                <li><strong>Holm-Bonferroni (FWER control):</strong> Applied to primary decomposition coefficients and marginal contrasts.</li>
                <li><strong>Benjamini-Hochberg (FDR control):</strong> Applied across secondary exploratory comparison families.</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Stage 5: Calibration & Conformal Prediction */}
      <section className="flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
            <BarChart3 className="h-5 w-5" />
          </div>
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-emerald-500">Pipeline Stage 5</span>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground font-[family-name:var(--font-poppins)]">
              Uncertainty Calibration & Conformal Sets
            </h2>
          </div>
        </div>

        <div className="p-6 sm:p-8 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="p-5 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-2">
              <h3 className="font-bold text-base text-foreground">Expected Calibration Error (ECE)</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                ECE measures the agreement between predicted softmax confidence and empirical accuracy by partitioning probabilities into <span className="font-mono text-foreground">M = 10</span> equally spaced bins:
              </p>
              <div className="p-3 rounded-xl bg-muted/60 border border-border/50 font-mono text-xs text-foreground">
                ECE = ∑ ( |B_m| / N ) · |acc(B_m) - conf(B_m)|
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Under compound corruption shift, model overconfidence escalates. Post-hoc temperature scaling (<span className="font-mono text-foreground">p̂ = softmax(z / T)</span>) is evaluated to measure recalibration efficacy.
              </p>
            </div>

            <div className="p-5 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-2">
              <h3 className="font-bold text-base text-foreground">Split Conformal Prediction</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Split conformal prediction constructs prediction sets <span className="font-mono text-foreground">C(x) = &#123;y : p̂(y) ≥ 1 - q̂&#125;</span> using non-conformity scores on a calibration split to guarantee user-specified marginal coverage:
              </p>
              <div className="p-3 rounded-xl bg-muted/60 border border-border/50 font-mono text-xs text-foreground">
                q̂ = Quantile( ⌈(n+1)(1-α)⌉ / n, &#123;s_i&#125; )
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                <strong>Exchangeability Boundary:</strong> Formal coverage guarantees require exchangeability between calibration and evaluation data. Under compound distribution shift, exchangeability is violated, causing empirical coverage to contract and average prediction set sizes to inflate.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Delineation of Research vs. Demo Capabilities */}
      <section className="p-6 sm:p-8 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10 text-primary border border-primary/20">
            <Compass className="h-4.5 w-4.5" />
          </div>
          <h3 className="font-bold text-xl text-foreground">
            Research Scope vs. Interactive Demo Delineation
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-muted-foreground leading-relaxed">
          <div className="p-4 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-1.5">
            <strong className="text-foreground">Canonical Compound-Corruption Research Experiment:</strong>
            <ul className="list-disc list-inside space-y-1 text-xs">
              <li>3,925 validation images from FastAI Imagenette-v2 (10 ImageNet classes).</li>
              <li>4 architectures: ResNet-50, ConvNeXt-Tiny, ViT-Base/16, Swin-Tiny.</li>
              <li>Full 3! = 6 permutations of Gaussian Noise, Defocus Blur, and JPEG across 5 severities.</li>
              <li>Two experimental regimes: Nominal and Approximate-Fidelity (973,400 total passes).</li>
              <li>Archived in open compressed CSV formats and verified via automated test scripts.</li>
            </ul>
          </div>

          <div className="p-4 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-1.5">
            <strong className="text-foreground">Interactive Website Capabilities:</strong>
            <ul className="list-disc list-inside space-y-1 text-xs">
              <li><Link href="/order-effects" className="text-primary hover:underline font-semibold">Order Effects:</Link> Interactive visualizer for the full 973,400-evaluation research dataset.</li>
              <li><Link href="/explorer" className="text-primary hover:underline font-semibold">Explorer:</Link> Side-by-side inspection of clean vs. corrupted images with precomputed inferences.</li>
              <li><Link href="/significance" className="text-primary hover:underline font-semibold">Significance:</Link> Interactive McNemar paired tests and Holm/BH adjusted comparison tables.</li>
              <li><span className="font-semibold text-foreground">Live Inference API (/api/inference):</span> Optional Gradio integration for single user uploads, decoupled from the offline benchmark.</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Reproducibility & Open Verification Summary */}
      <section className="p-6 sm:p-8 rounded-3xl bg-muted/40 border border-border/50 flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-5 w-5 text-teal-500" />
            <h3 className="font-bold text-lg text-foreground">Reproducibility & One-Click Verification</h3>
          </div>
          <span className="text-xs font-mono font-semibold px-2.5 py-1 rounded-full bg-teal-500/10 text-teal-400 border border-teal-500/20">
            52 / 52 Pytests Passing
          </span>
        </div>

        <p className="text-sm text-muted-foreground leading-relaxed">
          Every equation, matrix decomposition, transition table, and numerical finding presented on this platform is verifiable via the repository&apos;s executable test suite. Running <code className="text-foreground font-mono bg-muted/70 px-1.5 py-0.5 rounded border border-border/50">python ml/scripts/verify_all_numbers.py</code> verifies all 973,400 records and derived statistics in under five seconds with zero discrepancies.
        </p>

        <div className="flex flex-wrap gap-3 pt-1 items-center">
          <Link
            href="/order-effects"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-xs font-bold bg-primary text-primary-foreground hover:bg-primary/90 transition-colors shadow-sm"
          >
            <span>Explore Order Effects Data</span>
            <ArrowRight className="h-3.5 w-3.5" />
          </Link>
          <Link
            href="/about"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-xs font-bold bg-muted hover:bg-muted/80 text-foreground transition-colors border border-border/60"
          >
            <span>View About & Project Overview</span>
          </Link>
          <a
            href="https://doi.org/10.5281/zenodo.23483939"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-full text-xs font-semibold bg-purple-500/10 hover:bg-purple-500/20 text-purple-600 dark:text-purple-400 transition-colors border border-purple-500/30"
          >
            <span>Zenodo DOI: 10.5281/zenodo.23483939</span>
            <ExternalLink className="h-3 w-3" />
          </a>
        </div>
      </section>
    </main>
  );
}
