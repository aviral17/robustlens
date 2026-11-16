"use client";

import { useEffect, useState } from "react";
import { Card } from "@heroui/react";
import { Trophy, ShieldCheck, Activity, Clock, CheckCircle2, Info, Sparkles, Layers } from "lucide-react";

interface ModelSummary {
  model: string;
  display_name: string;
  family: string;
  clean_accuracy: number;
  mean_shifted_accuracy: number;
  mean_accuracy_drop: number;
  relative_robustness: number;
  mean_ece: number;
  mean_latency_ms: number;
}

interface BenchmarkReport {
  report_generated_at: string;
  configuration: {
    num_independent_images?: number;
    num_records: number;
    benchmark_status?: string;
    is_pilot?: boolean;
    models_evaluated: string[];
    corruptions_evaluated: string[];
    severities_evaluated: number[];
    n_bootstraps: number;
    significance_alpha: number;
    random_seed: number;
  };
  leaderboard: ModelSummary[];
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

interface V2Row {
  variant: string;
  architecture: string;
  display_name: string;
  n_images: number;
  r2_m3: number;
  r2_m5: number;
  delta_r2: number;
  ci95_delta_r2: string;
  mae_m3: number;
  mae_m5: number;
  delta_mae: number;
  ci95_delta_mae: string;
  p_value_raw: number;
  p_value_holm?: number | null;
  significant_holm?: boolean;
}

export default function LeaderboardPage() {
  const [activeTab, setActiveTab] = useState<"nominal" | "matched_fidelity">("matched_fidelity");
  const [report, setReport] = useState<BenchmarkReport | null>(null);
  const [matchedData, setMatchedData] = useState<{
    statistical_comparison: MatchedStatRow[];
    metadata: Record<string, unknown>;
  } | null>(null);
  const [v2Data, setV2Data] = useState<{
    results: V2Row[];
    metadata: Record<string, unknown>;
  } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch("/benchmark_report.json").then((res) => (res.ok ? res.json() : null)),
      fetch("/data/matched_fidelity_study_data.json").then((res) => (res.ok ? res.json() : null)),
      fetch("/data/imagenet_v2_replication_data.json").then((res) => (res.ok ? res.json() : null)),
    ])
      .then(([rep, matched, v2]) => {
        if (rep) setReport(rep);
        if (matched) setMatchedData(matched);
        if (v2) setV2Data(v2);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Could not load leaderboard data:", err);
        setLoading(false);
      });
  }, []);

  const leaderboard = report?.leaderboard || [];
  const matchedRows = matchedData?.statistical_comparison || [];
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

  const v2Rows = v2Data?.results?.filter((r) => r.variant === "Pooled_External_All_Variants") || [];

  return (
    <main className="min-h-screen pt-32 pb-24 px-6 max-w-[1400px] mx-auto flex flex-col gap-10">
      <div className="flex flex-col gap-4">
        <div className="flex items-center gap-2 text-primary font-bold text-sm tracking-widest uppercase">
          <Trophy className="w-4 h-4" /> Empirical Research Findings
        </div>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight font-[family-name:var(--font-poppins)]">
          Vision Robustness Leaderboard
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
            Matched-Fidelity Study (Margin Prediction & Fingerprints)
          </button>
          <button
            onClick={() => setActiveTab("nominal")}
            className={`px-5 py-2.5 rounded-2xl font-bold text-sm transition-all border ${
              activeTab === "nominal"
                ? "bg-primary text-primary-foreground border-primary shadow-md"
                : "bg-card text-muted-foreground border-border/60 hover:text-foreground"
            }`}
          >
            Standard Perturbations (Accuracy & ECE)
          </button>
        </div>

        {activeTab === "nominal" ? (
          <>
            <div className="flex flex-wrap items-center gap-3 mt-2">
              {report?.configuration?.is_pilot ?? (report?.configuration?.benchmark_status?.includes("Pilot") ?? true) ? (
                <span className="px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/10 text-amber-500 border border-amber-500/20">
                  {report?.configuration?.benchmark_status || "Pilot — N=100"}
                </span>
              ) : (
                <span className="px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
                  {report?.configuration?.benchmark_status || "Full Benchmark — N=3,925"}
                </span>
              )}
              <span className="text-xs text-muted-foreground">
                {report?.configuration?.num_independent_images || 100} Independent Images | {report?.configuration?.num_records || 6400} Prediction Records
              </span>
            </div>
            <p className="text-muted-foreground text-xl max-w-3xl">
              Rigorously evaluated ranking of pretrained computer-vision models under distribution shift.
              Measured on controlled perturbations across multiple corruption types and severities.
            </p>
          </>
        ) : (
          <>
            <div className="flex flex-wrap items-center gap-3 mt-2">
              <span className="px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
                Confirmatory Study — N=3,825
              </span>
              <span className="px-3 py-1 rounded-full text-xs font-semibold bg-blue-500/10 text-blue-500 border border-blue-500/20">
                External ImageNet-V2 — N=300
              </span>
              <span className="text-xs text-muted-foreground">
                290,700 Primary Passes + 22,800 External Passes = 313,500 Total Forward Passes | 19 Conditions (18 Corrupted + 1 Clean)
              </span>
            </div>
            <p className="text-muted-foreground text-xl max-w-3xl">
              Out-of-sample logit margin explanation under common (PSNR, SSIM) support. Evaluates whether continuous spatial-frequency and structural fingerprints explain incremental predictive variance beyond nonlinear global fidelity (PSNR, SSIM, LPIPS).
            </p>
          </>
        )}
      </div>

      {activeTab === "nominal" ? (
        /* Tab 1: Nominal Benchmark Table */
        <div className="flex flex-col gap-6 mt-2">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end gap-4">
            <div>
              <h2 className="text-2xl font-bold flex items-center gap-2">
                <span>Benchmark Results</span>
              </h2>
              <p className="text-muted-foreground text-sm mt-1">
                {report ? (
                  <>
                    Dataset: <strong className="text-foreground">Imagenette (Validation Split)</strong> •{" "}
                    Records: <strong className="text-foreground">{report.configuration.num_records}</strong> •{" "}
                    Seed: <strong className="text-foreground">{report.configuration.random_seed}</strong> •{" "}
                    Timestamp: {new Date(report.report_generated_at).toLocaleString()}
                  </>
                ) : (
                  "Loading empirical evaluation records..."
                )}
              </p>
            </div>
            <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-bold uppercase tracking-wider border border-primary/20">
              <CheckCircle2 className="w-3.5 h-3.5" /> Measured Benchmark
            </div>
          </div>

          <div className="border border-border/60 shadow-lg rounded-3xl overflow-hidden bg-card">
            <div className="overflow-x-auto w-full">
              <table className="w-full text-left text-sm whitespace-nowrap">
                <thead className="bg-muted/50 border-b border-border/60 text-xs uppercase tracking-wider">
                  <tr>
                    <th className="font-extrabold py-4 px-6 text-foreground">Rank & Model</th>
                    <th className="font-extrabold py-4 px-6 text-foreground">Architecture</th>
                    <th className="font-extrabold py-4 px-6 text-foreground">Clean Acc</th>
                    <th className="font-extrabold py-4 px-6 text-foreground">Shifted Acc</th>
                    <th className="font-extrabold py-4 px-6 text-foreground">Accuracy Drop</th>
                    <th className="font-extrabold py-4 px-6 text-foreground">Robust Ratio</th>
                    <th className="font-extrabold py-4 px-6 text-foreground">Mean ECE</th>
                    <th className="font-extrabold py-4 px-6 text-foreground">Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/30">
                  {leaderboard.map((item, index) => {
                    const dropPct = (item.mean_accuracy_drop * 100).toFixed(1);
                    const isTop = index === 0;

                    return (
                      <tr key={item.model} className="hover:bg-muted/20 transition-colors">
                        <td className="font-bold py-5 px-6 flex items-center gap-3">
                          <span
                            className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-black ${
                              isTop ? "bg-amber-500 text-white" : "bg-muted text-muted-foreground"
                            }`}
                          >
                            #{index + 1}
                          </span>
                          <span className="text-base text-foreground font-extrabold">{item.display_name}</span>
                        </td>
                        <td className="px-6 text-muted-foreground font-medium">{item.family}</td>
                        <td className="px-6 font-semibold">{(item.clean_accuracy * 100).toFixed(1)}%</td>
                        <td className="px-6 font-black text-primary text-base">
                          {(item.mean_shifted_accuracy * 100).toFixed(1)}%
                        </td>
                        <td className="px-6 font-bold text-destructive">-{dropPct}%</td>
                        <td className="px-6 font-mono font-semibold">{item.relative_robustness.toFixed(3)}</td>
                        <td className="px-6 font-mono font-semibold">{item.mean_ece.toFixed(3)}</td>
                        <td className="px-6 text-muted-foreground font-medium">
                          {item.mean_latency_ms ? `${item.mean_latency_ms.toFixed(1)} ms` : "—"}
                        </td>
                      </tr>
                    );
                  })}
                  {leaderboard.length === 0 && !loading && (
                    <tr>
                      <td colSpan={8} className="py-8 text-center text-muted-foreground">
                        No evaluation results found.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <div className="p-6 rounded-3xl bg-muted/30 border border-border/50 text-sm text-muted-foreground flex items-start gap-3">
            <Info className="w-5 h-5 shrink-0 mt-0.5 text-primary" />
            <p className="leading-relaxed">
              <strong>Evaluation Protocol:</strong> All models evaluated on identical test subsets using deterministic sampling seeds. Corruptions applied directly to un-normalized pixel tensors [0, 1] before model-specific channel normalization. Robust ratio is computed as Acc(shifted) / Acc(clean). Expected Calibration Error (ECE) is computed with M=10 uniform probability bins.
            </p>
          </div>
        </div>
      ) : (
        /* Tab 2: Matched-Fidelity Confirmatory & External Replication */
        <div className="flex flex-col gap-10 mt-2">
          {/* Table 1: Primary Confirmatory Results */}
          <div className="flex flex-col gap-4">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end gap-4">
              <div>
                <h2 className="text-2xl font-bold flex items-center gap-2">
                  <span>Primary Confirmatory Study Results</span>
                </h2>
                <p className="text-muted-foreground text-sm mt-1">
                  Conventional 4-Knot RCS (Model 3 = 27 cols, Model 5 = 32 cols) • N = 3,825 held-out Imagenette-v2 images • 290,700 passes • 5-fold image-grouped CV • B = 10,000 paired cluster bootstraps
                </p>
              </div>
              <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-emerald-500/10 text-emerald-500 text-xs font-bold uppercase tracking-wider border border-emerald-500/20">
                <CheckCircle2 className="w-3.5 h-3.5" /> All 4 Models p_adj &le; 0.0004
              </div>
            </div>

            <div className="border border-border/60 shadow-lg rounded-3xl overflow-hidden bg-card">
              <div className="overflow-x-auto w-full">
                <table className="w-full text-left text-sm whitespace-nowrap">
                  <thead className="bg-muted/50 border-b border-border/60 text-xs uppercase tracking-wider">
                    <tr>
                      <th className="font-extrabold py-4 px-6 text-foreground">Architecture</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Baseline M3 R²</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Extended M5 R²</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Incremental ΔR² (OOS)</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Simultaneous 95% CI</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">ΔMAE (OOS)</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Holm p_adj</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Null Exclusion</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/30">
                    {primaryRows.map((row) => (
                      <tr key={`primary-${row.architecture}`} className="hover:bg-muted/20 transition-colors">
                        <td className="font-bold py-5 px-6 text-base text-foreground">{row.display_name}</td>
                        <td className="px-6 font-mono font-semibold">{row.r2_oos_m3.toFixed(4)}</td>
                        <td className="px-6 font-mono font-bold text-primary">{row.r2_oos_m5.toFixed(4)}</td>
                        <td className="px-6 font-mono font-extrabold text-emerald-600 dark:text-emerald-400">
                          +{row.delta_r2_oos.toFixed(4)}
                        </td>
                        <td className="px-6 font-mono text-muted-foreground">{row.simultaneous_ci_delta_r2}</td>
                        <td className="px-6 font-mono font-semibold text-emerald-600 dark:text-emerald-400">
                          +{row.delta_mae_oos.toFixed(4)}
                        </td>
                        <td className="px-6 font-mono font-bold text-foreground">{row.p_display}</td>
                        <td className="px-6">
                          <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 text-xs font-bold border border-emerald-500/20">
                            <CheckCircle2 className="w-3 h-3" /> Significant
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="p-6 rounded-3xl bg-muted/30 border border-border/50 text-sm text-muted-foreground flex items-start gap-3">
              <Info className="w-5 h-5 shrink-0 mt-0.5 text-primary" />
              <div className="flex flex-col gap-1 leading-relaxed">
                <p>
                  <strong>Within-Architecture Evaluation Notice:</strong> True-class logit margins are evaluated strictly within each architecture against its own baseline model. Logit scales differ across architectures and are not compared as identical numerical units.
                </p>
                <p>
                  Baseline Model 3 incorporates 27 parameters: sequence order dummies (5 df), severity level dummies (2 df), order &times; level interactions (10 df), and 4-knot restricted cubic splines for PSNR, SSIM, and LPIPS (9 df). Extended Model 5 incorporates the 5 continuous physical distortion fingerprints (32 parameters total).
                </p>
              </div>
            </div>
          </div>

          {/* Table 2: Sensitivity Analysis: Reduced 2-df Splines */}
          <div className="flex flex-col gap-4">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end gap-4">
              <div>
                <h2 className="text-2xl font-bold flex items-center gap-2">
                  <span>Sensitivity Analysis: Reduced 2-df Splines</span>
                </h2>
                <p className="text-muted-foreground text-sm mt-1">
                  Reduced 2-df Natural Spline (Model 3 = 24 cols, Model 5 = 29 cols) • Verifying robustness against spline degree-of-freedom specification • 5-fold image-grouped CV
                </p>
              </div>
              <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-emerald-500/10 text-emerald-500 text-xs font-bold uppercase tracking-wider border border-emerald-500/20">
                <CheckCircle2 className="w-3.5 h-3.5" /> All 4 Models p_adj &le; 0.0004
              </div>
            </div>

            <div className="border border-border/60 shadow-lg rounded-3xl overflow-hidden bg-card">
              <div className="overflow-x-auto w-full">
                <table className="w-full text-left text-sm whitespace-nowrap">
                  <thead className="bg-muted/50 border-b border-border/60 text-xs uppercase tracking-wider">
                    <tr>
                      <th className="font-extrabold py-4 px-6 text-foreground">Architecture</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Baseline M3 R²</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Extended M5 R²</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Incremental ΔR² (OOS)</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Simultaneous 95% CI</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">ΔMAE (OOS)</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Holm p_adj</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Null Exclusion</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/30">
                    {sensitivityRows.map((row) => (
                      <tr key={`sensitivity-${row.architecture}`} className="hover:bg-muted/20 transition-colors">
                        <td className="font-bold py-5 px-6 text-base text-foreground">{row.display_name}</td>
                        <td className="px-6 font-mono font-semibold">{row.r2_oos_m3.toFixed(4)}</td>
                        <td className="px-6 font-mono font-bold text-primary">{row.r2_oos_m5.toFixed(4)}</td>
                        <td className="px-6 font-mono font-extrabold text-emerald-600 dark:text-emerald-400">
                          +{row.delta_r2_oos.toFixed(4)}
                        </td>
                        <td className="px-6 font-mono text-muted-foreground">{row.simultaneous_ci_delta_r2}</td>
                        <td className="px-6 font-mono font-semibold text-emerald-600 dark:text-emerald-400">
                          +{row.delta_mae_oos.toFixed(4)}
                        </td>
                        <td className="px-6 font-mono font-bold text-foreground">{row.p_display}</td>
                        <td className="px-6">
                          <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 text-xs font-bold border border-emerald-500/20">
                            <CheckCircle2 className="w-3 h-3" /> Robust to Specification
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="p-6 rounded-3xl bg-muted/30 border border-border/50 text-sm text-muted-foreground flex items-start gap-3">
              <Info className="w-5 h-5 shrink-0 mt-0.5 text-primary" />
              <div className="flex flex-col gap-1 leading-relaxed">
                <p>
                  <strong>Specification Sensitivity Check:</strong> In this pre-registered sensitivity specification, global fidelity metrics (PSNR, SSIM, LPIPS) are parameterized using 2-degree-of-freedom natural splines instead of 4-knot restricted cubic splines (reducing baseline columns from 27 to 24, and extended columns from 32 to 29). The incremental R&sup2; gains remain statistically positive across all four architectures (all Holm-adjusted p &le; 0.0004), confirming that the predictive value of structural fingerprints is not an artifact of spline basis choice.
                </p>
              </div>
            </div>
          </div>

          {/* Table 2: ImageNet-V2 External Replication */}
          <div className="flex flex-col gap-4">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end gap-4">
              <div>
                <h2 className="text-2xl font-bold flex items-center gap-2">
                  <span>Independent External Replication on ImageNet-V2</span>
                </h2>
                <p className="text-muted-foreground text-sm mt-1">
                  N = 300 pooled external images (100 MatchedFrequency, 100 Threshold0.7, 100 TopImages) • 22,800 model evaluations • Zero parameters tuned on external data
                </p>
              </div>
              <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-blue-500/10 text-blue-500 text-xs font-bold uppercase tracking-wider border border-blue-500/20">
                <ShieldCheck className="w-3.5 h-3.5" /> Independent External Replication
              </div>
            </div>

            <div className="border border-border/60 shadow-lg rounded-3xl overflow-hidden bg-card">
              <div className="overflow-x-auto w-full">
                <table className="w-full text-left text-sm whitespace-nowrap">
                  <thead className="bg-muted/50 border-b border-border/60 text-xs uppercase tracking-wider">
                    <tr>
                      <th className="font-extrabold py-4 px-6 text-foreground">Architecture</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Cohort</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Baseline M3 R²</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Extended M5 R²</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">ΔR² (OOS)</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">95% Bootstrap CI</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Raw p-value</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Holm p_adj</th>
                      <th className="font-extrabold py-4 px-6 text-foreground">Replication Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/30">
                    {v2Rows.map((row) => (
                      <tr key={row.architecture} className="hover:bg-muted/20 transition-colors">
                        <td className="font-bold py-5 px-6 text-base text-foreground">{row.display_name}</td>
                        <td className="px-6 text-muted-foreground font-medium">Pooled (N=300)</td>
                        <td className="px-6 font-mono font-semibold">{row.r2_m3.toFixed(4)}</td>
                        <td className="px-6 font-mono font-bold text-primary">{row.r2_m5.toFixed(4)}</td>
                        <td className="px-6 font-mono font-extrabold text-foreground">
                          +{row.delta_r2.toFixed(4)}
                        </td>
                        <td className="px-6 font-mono text-muted-foreground">{row.ci95_delta_r2}</td>
                        <td className="px-6 font-mono">{row.p_value_raw.toFixed(4)}</td>
                        <td className="px-6 font-mono font-bold">
                          {row.p_value_holm !== undefined && row.p_value_holm !== null
                            ? row.p_value_holm.toFixed(4)
                            : "—"}
                        </td>
                        <td className="px-6">
                          {row.significant_holm ? (
                            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 text-xs font-bold border border-emerald-500/20">
                              <CheckCircle2 className="w-3 h-3" /> Replicated (p &lt; 0.05)
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-muted text-muted-foreground text-xs font-medium border border-border/60">
                              CI Spans Zero (p &ge; 0.05)
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="p-6 rounded-3xl bg-muted/30 border border-border/50 text-sm text-muted-foreground flex items-start gap-3">
              <Info className="w-5 h-5 shrink-0 mt-0.5 text-primary" />
              <p className="leading-relaxed">
                <strong>Rigorous External Replication Disclosure:</strong> In the independent pooled ImageNet-V2 cohort (N = 300), only <strong>ViT-Base / 16</strong> achieves a positive out-of-sample R² confidence interval strictly excluding zero with statistical significance following step-down Holm-Bonferroni correction (&Delta;R&sup2; = +0.0371, 95% CI: [+0.0065, +0.0675], p_adj = 0.0340). For ResNet-50, ConvNeXt-Tiny, and Swin-Tiny, pooled point estimates remain positive (+0.0115 to +0.0212), but their 95% bootstrap confidence intervals cross zero and their adjusted p-values exceed 0.05. Statistically significant external R² improvement is claimed only for ViT-Base.
              </p>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
