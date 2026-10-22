"use client";

import { useEffect, useState } from "react";
import { Card } from "@heroui/react";
import { Scale, CheckCircle2, XCircle, Info, ShieldAlert } from "lucide-react";

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

export default function SignificancePage() {
  const [tests, setTests] = useState<PairwiseTest[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/benchmark_report.json")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load benchmark_report.json");
        return res.json();
      })
      .then((data) => {
        setTests(data.pairwise_significance || []);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Could not load report:", err);
        setLoading(false);
      });
  }, []);

  const isPilot = tests.length > 0 ? (tests[0].status?.includes("Pilot") || (tests[0].num_independent_images ?? 0) < 500) : true;
  const independentImages = tests.length > 0 ? (tests[0].num_independent_images ?? 3925) : 3925;

  return (
    <main className="min-h-screen pt-32 pb-24 px-6 max-w-[1400px] mx-auto flex flex-col gap-10">
      <div className="flex flex-col gap-4">
        <div className="flex items-center gap-2 text-primary font-bold text-sm tracking-widest uppercase">
          <Scale className="w-4 h-4" /> Hypothesis Testing & Multiple Corrections
        </div>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight font-[family-name:var(--font-poppins)]">
          Statistical Significance Analysis
        </h1>
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
      </div>


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

        {/* Pairwise Tests Table */}
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
                        <span className="text-muted-foreground mx-2 font-normal">vs</span>
                        <span className="text-foreground">{pair.model_2_name}</span>
                      </td>
                      <td className="px-6 font-mono text-xs text-muted-foreground">
                        {pair.num_independent_images ?? 3925} images ({pair.num_comparisons} pairs)
                      </td>
                      <td className="px-6 font-semibold">
                        {m1Shift.toFixed(2)}% vs {m2Shift.toFixed(2)}%
                      </td>
                      <td className="px-6 font-mono font-bold text-foreground">
                        {pair.accuracy_difference > 0 ? `+${diffPct}%` : `${diffPct}%`}
                      </td>
                      <td className="px-6 font-mono text-xs">
                        [{(pair.paired_ci_95_lower * 100).toFixed(2)}%, {(pair.paired_ci_95_upper * 100).toFixed(2)}%]
                      </td>
                      <td className="px-6 font-mono font-semibold text-primary">
                        {formatP(pair.permutation_p_value ?? (pair as any).image_level_boot_p_value, pair.permutation_p_bound ?? (pair as any).empirical_bootstrap_bound)}
                      </td>
                      <td className="px-6 font-mono text-xs text-muted-foreground">
                        {pair.clean_acc_model_1 ? `${m1Clean.toFixed(2)}% vs ${m2Clean.toFixed(2)}%` : "—"}
                      </td>
                      <td className="px-6 font-mono text-xs">
                        {formatP(pair.clean_mcnemar_p_value)}
                      </td>
                      <td className="px-6">
                        {pair.holm_significant ? (
                          <span className="inline-flex items-center gap-1 bg-green-500/10 text-green-500 px-2.5 py-1 rounded-full text-xs font-bold uppercase">
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
                          <span className="inline-flex items-center gap-1 bg-green-500/10 text-green-500 px-2.5 py-1 rounded-full text-xs font-bold uppercase">
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
                      No pairwise statistical tests loaded. Run `uv run python ml/scripts/make_report.py` to populate.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Statistical Details Box */}
        <div className="grid md:grid-cols-2 gap-6">
          <div className="p-6 rounded-3xl bg-muted/30 border border-border/50 text-sm flex flex-col gap-2">
            <h4 className="font-bold text-foreground flex items-center gap-2">
              <Info className="w-4 h-4 text-primary" /> McNemar&apos;s Paired Test
            </h4>
            <p className="text-muted-foreground leading-relaxed">
              McNemar&apos;s test tests the null hypothesis of marginal homogeneity on paired classification outcomes.
              Discordant pairs n01 (M1 wrong, M2 right) and n10 (M1 right, M2 wrong) are evaluated with Edwards&apos;
              continuity correction: chi^2 = (|n01 - n10| - 1)^2 / (n01 + n10).
            </p>
          </div>

          <div className="p-6 rounded-3xl bg-muted/30 border border-border/50 text-sm flex flex-col gap-2">
            <h4 className="font-bold text-foreground flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-primary" /> Multiplicity Control
            </h4>
            <p className="text-muted-foreground leading-relaxed">
              Comparing all pairwise model combinations introduces the multiple comparisons problem.
              We report both Holm&apos;s step-down method (guaranteeing Family-Wise Error Rate control at alpha = 0.05)
              and the Benjamini-Hochberg step-up procedure (controlling False Discovery Rate).
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
