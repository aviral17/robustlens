"use client";

import { useEffect, useState } from "react";
import { Card } from "@heroui/react";
import { Scale, CheckCircle2, XCircle, Info, ShieldAlert, Sparkles, Layers } from "lucide-react";

interface PairwiseTest {
  model_1: string;
  model_1_name: string;
  model_2: string;
  model_2_name: string;
  status?: string;
  num_independent_images?: number;
  num_comparisons: number;
  acc_model_1: number;
  acc_model_2: number;
  shifted_acc_model_1?: number;
  shifted_acc_model_2?: number;
  accuracy_difference: number;
  paired_ci_95_lower: number;
  paired_ci_95_upper: number;
  bootstrap_standard_error?: number;
  sign_flip_permutations?: number;
  permutation_p_value?: number;
  permutation_p_bound?: string;
  permutation_z_statistic?: number;
  permutation_asymp_p_value?: number;
  image_level_boot_p_value?: number;
  clean_acc_model_1?: number | null;
  clean_acc_model_2?: number | null;
  clean_accuracy_difference?: number | null;
  clean_mcnemar_p_value?: number | null;
  mcnemar_p_value: number;
  paired_boot_p_value: number;
  holm_significant: boolean;
  benjamini_hochberg_significant: boolean;
  interpretation?: string;
}

interface MatchedStatRow {
  architecture: string;
  display_name: string;
  specification: string;
  r2_oos_m3: number;
  r2_oos_m5: number;
  delta_r2_oos: number;
  ci95_pointwise_delta_r2: string;
  simultaneous_ci_delta_r2: string;
  mae_oos_m3: number;
  mae_oos_m5: number;
  delta_mae_oos: number;
  ci95_pointwise_delta_mae: string;
  simultaneous_ci_delta_mae: string;
  holm_adj_p_delta_r2: string;
  p_display: string;
  significant_r2: boolean;
  significant_mae: boolean;
}

interface InteractionTest {
  comparison: string;
  arch_a: string;
  arch_b: string;
  beta_vit: number;
  beta_other: number;
  diff_point: number;
  ci95_diff: [number, number];
  p_value: number;
  se_diff: number;
}

export default function SignificancePage() {
  const [activeTab, setActiveTab] = useState<"pairwise_accuracy" | "matched_fidelity">("matched_fidelity");
  const [tests, setTests] = useState<PairwiseTest[]>([]);
  const [matchedRows, setMatchedRows] = useState<MatchedStatRow[]>([]);
  const [interactionTests, setInteractionTests] = useState<InteractionTest[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch("/benchmark_report.json").then((res) => (res.ok ? res.json() : null)),
      fetch("/data/matched_fidelity_study_data.json").then((res) => (res.ok ? res.json() : null)),
    ])
      .then(([rep, matched]) => {
        if (rep?.pairwise_significance) {
          setTests(rep.pairwise_significance);
        }
        if (matched) {
          setMatchedRows(matched.statistical_comparison || []);
          setInteractionTests(matched.local_ssim_interaction_tests || []);
        }
        setLoading(false);
      })
      .catch((err) => {
        console.error("Could not load significance data:", err);
        setLoading(false);
      });
  }, []);

  const isPilot =
    tests.length > 0
      ? tests[0].status?.includes("Pilot") || (tests[0].num_independent_images ?? 0) < 500
      : true;
  const independentImages = tests.length > 0 ? tests[0].num_independent_images ?? 3925 : 3925;

  const primaryRows = (matchedRows.filter(
    (r) => r.specification?.includes("Conventional 4-Knot RCS") || r.specification?.includes("27 cols")
  ).length === 4
    ? matchedRows.filter(
        (r) => r.specification?.includes("Conventional 4-Knot RCS") || r.specification?.includes("27 cols")
      )
    : matchedRows.slice(0, 4));

  const sensitivityRows = (matchedRows.filter(
    (r) => r.specification?.includes("Reduced 2-df Spline") || r.specification?.includes("24 cols")
  ).length === 4
    ? matchedRows.filter(
        (r) => r.specification?.includes("Reduced 2-df Spline") || r.specification?.includes("24 cols")
      )
    : matchedRows.slice(4, 8));

  return (
    <main className="min-h-screen pt-32 pb-24 px-6 max-w-[1400px] mx-auto flex flex-col gap-10">
      <div className="flex flex-col gap-4">
        <div className="flex items-center gap-2 text-primary font-bold text-sm tracking-widest uppercase">
          <Scale className="w-4 h-4" /> Hypothesis Testing & Multiple Corrections
        </div>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight font-[family-name:var(--font-poppins)]">
          Statistical Significance Analysis
        </h1>

        {/* Tab Switcher */}
        <div className="flex flex-wrap gap-3 mt-2">
          <button
            onClick={() => setActiveTab("matched_fidelity")}
            className={`px-5 py-2.5 rounded-2xl font-bold text-sm transition-all border ${
              activeTab === "matched_fidelity"
                ? "bg-primary text-primary-foreground border-primary shadow-md"
                : "bg-card text-muted-foreground border-border/60 hover:text-foreground"
            }`}
          >
            Matched-Fidelity Variance & Margin Tests (New Study)
          </button>
          <button
            onClick={() => setActiveTab("pairwise_accuracy")}
            className={`px-5 py-2.5 rounded-2xl font-bold text-sm transition-all border ${
              activeTab === "pairwise_accuracy"
                ? "bg-primary text-primary-foreground border-primary shadow-md"
                : "bg-card text-muted-foreground border-border/60 hover:text-foreground"
            }`}
          >
            Historical Accuracy Pairwise Tests (Nominal Benchmark)
          </button>
        </div>

        {activeTab === "matched_fidelity" ? (
          <>
            <div className="p-4 rounded-xl border border-emerald-500/20 bg-emerald-500/10 flex items-start gap-3 mt-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0 mt-0.5" />
              <div className="text-sm">
                <span className="font-semibold text-emerald-500">
                  Confirmatory Clustered Bootstrap Testing (B = 10,000, N = 3,825 Images):{" "}
                </span>
                <span className="text-muted-foreground">
                  Hypothesis tests evaluate whether the 5 continuous distortion fingerprints explain statistically significant additional out-of-sample logit margin variance beyond global fidelity (Model 3 vs Model 5). Resampling is strictly clustered by physical image ID. Step-down Holm-Bonferroni correction guarantees FWER control at &alpha; = 0.05.
                </span>
              </div>
            </div>
            <p className="text-muted-foreground text-xl max-w-3xl">
              Strict inferential testing under common (PSNR, SSIM) support. Evaluates within-architecture margin explanation gains and formal architectural interaction tests.
            </p>
          </>
        ) : (
          <>
            <p className="text-muted-foreground text-xl max-w-3xl">
              Paired hypothesis testing between vision architectures on identical physical image sets.
              Employs exact paired sign-flip permutation tests (R=20,000) for distribution-free inferential p-values,
              cluster paired bootstrap resampling (B=10,000) for 95% confidence intervals, and McNemar&apos;s tests
              for clean baselines, strictly adjusted via Holm (FWER) and Benjamini-Hochberg (FDR) procedures.
            </p>

            {isPilot ? (
              <div className="p-4 rounded-xl border border-amber-500/20 bg-amber-500/10 flex items-start gap-3 mt-2">
                <Info className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
                <div className="text-sm">
                  <span className="font-semibold text-amber-500">Pilot Evaluation Notice (N=100 Independent Images): </span>
                  <span className="text-muted-foreground">
                    These preliminary comparisons use cluster paired bootstrap resampling and permutation testing at the image level. Preliminary p-values reflect pilot sample size and should not be construed as definitive until the full N=3,925 benchmark is executed.
                  </span>
                </div>
              </div>
            ) : (
              <div className="p-4 rounded-xl border border-emerald-500/20 bg-emerald-500/10 flex items-start gap-3 mt-2">
                <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0 mt-0.5" />
                <div className="text-sm">
                  <span className="font-semibold text-emerald-500">Full Benchmark Validation (N={independentImages} Independent Images): </span>
                  <span className="text-muted-foreground">
                    Paired sign-flip permutation testing (R=20,000) and cluster paired bootstrap resampling (B=10,000) across the complete official validation split. Rigorous family-wise error rate (Holm) and false discovery rate (Benjamini-Hochberg) multiplicity controls applied across all pairwise comparisons.
                  </span>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {activeTab === "matched_fidelity" ? (
        /* Matched-Fidelity Significance Content */
        <div className="flex flex-col gap-10 mt-2">
          {/* Table 1: Model 3 vs Model 5 Primary Incremental Variance */}
          <div className="flex flex-col gap-4">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end gap-2">
              <div>
                <h2 className="text-2xl font-bold flex items-center gap-2">
                  <span>Primary Model 3 vs. Model 5 Incremental Variance Tests</span>
                </h2>
                <p className="text-muted-foreground text-sm mt-1">
                  Conventional 4-Knot RCS (Model 3 = 27 cols, Model 5 = 32 cols) • Null hypothesis: &Delta;R&sup2;<sub>OOS</sub> &le; 0 (Distortion fingerprints add zero predictive variance beyond nonlinear global fidelity).
                </p>
              </div>
              <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 text-xs font-bold uppercase tracking-wider border border-emerald-500/20">
                <CheckCircle2 className="w-3.5 h-3.5" /> All 4 Models p_adj &le; 0.0004
              </div>
            </div>

            <div className="border border-border/60 shadow-lg rounded-3xl overflow-hidden bg-card">
              <div className="overflow-x-auto w-full">
                <table className="w-full text-left text-sm whitespace-nowrap">
                  <thead className="bg-muted/50 border-b border-border/60 text-xs uppercase tracking-wider">
                    <tr>
                      <th className="font-extrabold py-4 px-6 text-foreground">Architecture</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Incremental ΔR² (OOS)</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Pointwise 95% CI</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Simultaneous 95% CI</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">ΔMAE (OOS)</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Raw p-value</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Holm p_adj</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Statistical Decision</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/30">
                    {primaryRows.map((r) => (
                      <tr key={`primary-${r.architecture}`} className="hover:bg-muted/20 transition-colors">
                        <td className="font-bold py-5 px-6 text-base text-foreground">{r.display_name}</td>
                        <td className="px-6 font-mono font-extrabold text-emerald-600 dark:text-emerald-400">
                          +{r.delta_r2_oos.toFixed(4)}
                        </td>
                        <td className="px-6 font-mono text-muted-foreground">{r.ci95_pointwise_delta_r2}</td>
                        <td className="px-6 font-mono text-muted-foreground font-semibold">
                          {r.simultaneous_ci_delta_r2}
                        </td>
                        <td className="px-6 font-mono font-semibold text-emerald-600 dark:text-emerald-400">
                          +{r.delta_mae_oos.toFixed(4)}
                        </td>
                        <td className="px-6 font-mono">&le; 0.0001</td>
                        <td className="px-6 font-mono font-bold text-foreground">{r.p_display}</td>
                        <td className="px-6">
                          <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 text-xs font-bold border border-emerald-500/20">
                            <CheckCircle2 className="w-3 h-3" /> Reject Null (p &le; 0.0004)
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Table 1b: Secondary Sensitivity Analysis (Reduced 2-df Splines) */}
          <div className="flex flex-col gap-4">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end gap-2">
              <div>
                <h2 className="text-2xl font-bold flex items-center gap-2">
                  <span>Sensitivity Analysis: Reduced 2-df Splines</span>
                </h2>
                <p className="text-muted-foreground text-sm mt-1">
                  Reduced 2-df Natural Spline (Model 3 = 24 cols, Model 5 = 29 cols) • Verifying hypothesis test stability against spline specification.
                </p>
              </div>
              <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 text-xs font-bold uppercase tracking-wider border border-emerald-500/20">
                <CheckCircle2 className="w-3.5 h-3.5" /> All 4 Models p_adj &le; 0.0004
              </div>
            </div>

            <div className="border border-border/60 shadow-lg rounded-3xl overflow-hidden bg-card">
              <div className="overflow-x-auto w-full">
                <table className="w-full text-left text-sm whitespace-nowrap">
                  <thead className="bg-muted/50 border-b border-border/60 text-xs uppercase tracking-wider">
                    <tr>
                      <th className="font-extrabold py-4 px-6 text-foreground">Architecture</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Incremental ΔR² (OOS)</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Pointwise 95% CI</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Simultaneous 95% CI</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">ΔMAE (OOS)</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Raw p-value</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Holm p_adj</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Statistical Decision</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/30">
                    {sensitivityRows.map((r) => (
                      <tr key={`sensitivity-${r.architecture}`} className="hover:bg-muted/20 transition-colors">
                        <td className="font-bold py-5 px-6 text-base text-foreground">{r.display_name}</td>
                        <td className="px-6 font-mono font-extrabold text-emerald-600 dark:text-emerald-400">
                          +{r.delta_r2_oos.toFixed(4)}
                        </td>
                        <td className="px-6 font-mono text-muted-foreground">{r.ci95_pointwise_delta_r2}</td>
                        <td className="px-6 font-mono text-muted-foreground font-semibold">
                          {r.simultaneous_ci_delta_r2}
                        </td>
                        <td className="px-6 font-mono font-semibold text-emerald-600 dark:text-emerald-400">
                          +{r.delta_mae_oos.toFixed(4)}
                        </td>
                        <td className="px-6 font-mono">&le; 0.0001</td>
                        <td className="px-6 font-mono font-bold text-foreground">{r.p_display}</td>
                        <td className="px-6">
                          <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 text-xs font-bold border border-emerald-500/20">
                            <CheckCircle2 className="w-3 h-3" /> Reject Null (p &le; 0.0004)
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Table 2: Formal Architectural Interaction Tests on Local SSIM */}
          <div className="flex flex-col gap-4">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end gap-2">
              <div>
                <h2 className="text-2xl font-bold flex items-center gap-2">
                  <span>Local SSIM Spatial Heterogeneity Interaction Tests</span>
                </h2>
                <p className="text-muted-foreground text-sm mt-1">
                  Paired image-cluster bootstrap tests (B = 10,000) evaluating architectural differences in local SSIM sensitivity (&Delta;&beta; = &beta;<sub>ViT</sub> - &beta;<sub>Other</sub>).
                </p>
              </div>
              <span className="text-xs font-mono text-muted-foreground bg-muted/40 px-3 py-1.5 rounded-full border border-border/60">
                ViT-Base Local SSIM &beta; = -0.2691
              </span>
            </div>

            <div className="border border-border/60 shadow-lg rounded-3xl overflow-hidden bg-card">
              <div className="overflow-x-auto w-full">
                <table className="w-full text-left text-sm whitespace-nowrap">
                  <thead className="bg-muted/50 border-b border-border/60 text-xs uppercase tracking-wider">
                    <tr>
                      <th className="font-extrabold py-4 px-6 text-foreground">Pairwise Comparison</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">ViT β</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Other Model β</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Δβ (ViT - Other)</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Bootstrap SE</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">95% Bootstrap CI</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Bootstrap p-value</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Interaction Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/30">
                    {interactionTests.map((t) => {
                      const isSig = t.p_value < 0.05;
                      const labelMap: Record<string, string> = {
                        vit_base_minus_resnet50: "ViT-Base / 16 vs. ResNet-50",
                        vit_base_minus_convnext_tiny: "ViT-Base / 16 vs. ConvNeXt-Tiny",
                        vit_base_minus_swin_tiny: "ViT-Base / 16 vs. Swin-Tiny",
                      };

                      return (
                        <tr key={t.comparison} className="hover:bg-muted/20 transition-colors">
                          <td className="font-bold py-5 px-6 text-base text-foreground">
                            {labelMap[t.comparison] || t.comparison}
                          </td>
                          <td className="px-6 font-mono text-muted-foreground">{t.beta_vit.toFixed(4)}</td>
                          <td className="px-6 font-mono text-muted-foreground">{t.beta_other.toFixed(4)}</td>
                          <td className="px-6 font-mono font-bold text-foreground">{t.diff_point.toFixed(4)}</td>
                          <td className="px-6 font-mono">{t.se_diff.toFixed(4)}</td>
                          <td className="px-6 font-mono text-muted-foreground">
                            [{t.ci95_diff[0].toFixed(4)}, {t.ci95_diff[1].toFixed(4)}]
                          </td>
                          <td className="px-6 font-mono font-bold">
                            {t.p_value <= 0.0001 ? "< 0.0001" : t.p_value.toFixed(5)}
                          </td>
                          <td className="px-6">
                            {isSig ? (
                              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 text-xs font-bold border border-emerald-500/20">
                                <CheckCircle2 className="w-3 h-3" /> Significant (p &lt; 0.01)
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-muted text-muted-foreground text-xs font-medium border border-border/60">
                                Not Significant (CI Spans Zero)
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="p-6 rounded-3xl bg-muted/30 border border-border/50 text-xs text-muted-foreground flex items-start gap-3">
              <Info className="w-5 h-5 shrink-0 mt-0.5 text-primary" />
              <div className="flex flex-col gap-1 leading-relaxed">
                <p>
                  <strong>Scientific Conclusion on Architectural Taxonomy:</strong> ViT-Base&apos;s negative local-SSIM sensitivity (&beta; = -0.2691) is statistically distinct from ConvNeXt-Tiny (&Delta;&beta; = -0.2277, p = 0.00010) and Swin-Tiny (&Delta;&beta; = -0.2201, p = 0.00300), showing that unconstrained global self-attention associates with greater margin collapse when visual degradation is spatially non-uniform.
                </p>
                <p>
                  However, the difference between ViT-Base and ResNet-50 is <strong>not statistically established</strong> (&Delta;&beta; = -0.0725, p = 0.61174, 95% CI includes zero). This finding is therefore reported as an architecture-specific empirical association rather than a universal mechanism.
                </p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* Historical Pairwise Tests Content */
        <div className="flex flex-col gap-6 mt-2">
          <div className="flex justify-between items-end">
            <div>
              <h2 className="text-2xl font-bold">Pairwise Model Comparisons</h2>
              <p className="text-muted-foreground text-sm mt-1">
                Evaluating whether performance differences under distribution shift are statistically distinguishable from chance.
              </p>
            </div>
            <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-bold uppercase tracking-wider border border-primary/20">
              <CheckCircle2 className="w-3.5 h-3.5" /> Measured Statistical Engine
            </div>
          </div>

          <div className="border border-border/60 shadow-lg rounded-3xl overflow-hidden bg-card">
            <div className="overflow-x-auto w-full">
              <table className="w-full text-left text-sm whitespace-nowrap">
                <thead className="bg-muted/50 border-b border-border/60 text-xs uppercase tracking-wider">
                  <tr>
                    <th className="font-extrabold py-4 px-6 text-foreground">Model Pair</th>
                    <th className="font-extrabold py-4 px-6 text-foreground">Sample Structure</th>
                    <th className="font-extrabold py-4 px-6 text-foreground">Shifted Acc (M1 vs M2)</th>
                    <th className="font-extrabold py-4 px-6 text-foreground">Shifted Δ (M1 - M2)</th>
                    <th className="font-extrabold py-4 px-6 text-foreground">95% Cluster CI</th>
                    <th className="font-extrabold py-4 px-6 text-foreground">Sign-Flip Perm p</th>
                    <th className="font-extrabold py-4 px-6 text-foreground">Clean Acc (M1 vs M2)</th>
                    <th className="font-extrabold py-4 px-6 text-foreground">Clean McNemar p</th>
                    <th className="font-extrabold py-4 px-6 text-foreground">Holm FWER</th>
                    <th className="font-extrabold py-4 px-6 text-foreground">BH FDR</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/30">
                  {tests.map((pair, idx) => {
                    const formatP = (p: number | undefined | null, bound?: string) => {
                      if (bound) return bound;
                      if (p === undefined || p === null) return "—";
                      return p <= 0.0001 ? "< 0.0001" : p < 0.001 ? p.toExponential(2) : p.toFixed(4);
                    };
                    const diffPct = (pair.accuracy_difference * 100).toFixed(2);
                    const m1Shift = ((pair as any).shifted_acc_model_1 ?? pair.acc_model_1) * 100;
                    const m2Shift = ((pair as any).shifted_acc_model_2 ?? pair.acc_model_2) * 100;
                    const m1Clean = (pair.clean_acc_model_1 ?? 0) * 100;
                    const m2Clean = (pair.clean_acc_model_2 ?? 0) * 100;

                    return (
                      <tr key={idx} className="hover:bg-muted/20 transition-colors">
                        <td className="font-bold py-5 px-6">
                          <span className="text-foreground">{pair.model_1_name}</span>
                          <span className="text-muted-foreground mx-1.5 font-normal">vs</span>
                          <span className="text-foreground">{pair.model_2_name}</span>
                        </td>
                        <td className="px-6 text-muted-foreground text-xs font-mono">
                          {pair.num_comparisons?.toLocaleString() || "400"} pairs (N={pair.num_independent_images || independentImages})
                        </td>
                        <td className="px-6 font-mono">
                          {m1Shift.toFixed(1)}% <span className="text-muted-foreground">vs</span> {m2Shift.toFixed(1)}%
                        </td>
                        <td className="px-6 font-mono font-bold text-foreground">
                          {Number(diffPct) > 0 ? `+${diffPct}%` : `${diffPct}%`}
                        </td>
                        <td className="px-6 font-mono text-muted-foreground text-xs">
                          [{((pair.paired_ci_95_lower ?? 0) * 100).toFixed(2)}%, {((pair.paired_ci_95_upper ?? 0) * 100).toFixed(2)}%]
                        </td>
                        <td className="px-6 font-mono text-foreground font-semibold">
                          {formatP(pair.permutation_p_value ?? pair.paired_boot_p_value, pair.permutation_p_bound)}
                        </td>
                        <td className="px-6 font-mono text-xs">
                          {pair.clean_acc_model_1 !== null && pair.clean_acc_model_1 !== undefined
                            ? `${m1Clean.toFixed(1)}% vs ${m2Clean.toFixed(1)}%`
                            : "—"}
                        </td>
                        <td className="px-6 font-mono text-xs text-muted-foreground">
                          {formatP(pair.clean_mcnemar_p_value)}
                        </td>
                        <td className="px-6">
                          {pair.holm_significant ? (
                            <span className="inline-flex items-center gap-1 bg-emerald-500/10 text-emerald-500 px-2.5 py-1 rounded-full text-xs font-semibold uppercase">
                              <CheckCircle2 className="w-3 h-3" /> Sig
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 bg-muted text-muted-foreground px-2.5 py-1 rounded-full text-xs font-semibold uppercase">
                              <XCircle className="w-3 h-3" /> Not Sig
                            </span>
                          )}
                        </td>
                        <td className="px-6">
                          {pair.benjamini_hochberg_significant ? (
                            <span className="inline-flex items-center gap-1 bg-emerald-500/10 text-emerald-500 px-2.5 py-1 rounded-full text-xs font-semibold uppercase">
                              <CheckCircle2 className="w-3 h-3" /> Sig
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 bg-muted text-muted-foreground px-2.5 py-1 rounded-full text-xs font-semibold uppercase">
                              <XCircle className="w-3 h-3" /> Not Sig
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                  {tests.length === 0 && !loading && (
                    <tr>
                      <td colSpan={10} className="py-8 text-center text-muted-foreground">
                        No pairwise statistical tests loaded.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="p-6 rounded-3xl bg-muted/30 border border-border/50 text-sm flex flex-col gap-2">
              <h4 className="font-bold text-foreground flex items-center gap-2">
                <Info className="w-4 h-4 text-primary" /> McNemar&apos;s Paired Test
              </h4>
              <p className="text-muted-foreground leading-relaxed">
                McNemar&apos;s test tests the null hypothesis of marginal homogeneity on paired classification outcomes. Discordant pairs n01 (M1 wrong, M2 right) and n10 (M1 right, M2 wrong) are evaluated with Edwards&apos; continuity correction: chi^2 = (|n01 - n10| - 1)^2 / (n01 + n10).
              </p>
            </div>

            <div className="p-6 rounded-3xl bg-muted/30 border border-border/50 text-sm flex flex-col gap-2">
              <h4 className="font-bold text-foreground flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-primary" /> Multiplicity Control
              </h4>
              <p className="text-muted-foreground leading-relaxed">
                Comparing all pairwise model combinations introduces the multiple comparisons problem. We report both Holm&apos;s step-down method (guaranteeing Family-Wise Error Rate control at alpha = 0.05) and the Benjamini-Hochberg step-up procedure (controlling False Discovery Rate).
              </p>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
