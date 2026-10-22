"use client";

import { useEffect, useState } from "react";
import { Card } from "@heroui/react";
import { Trophy, ShieldCheck, Activity, Clock, CheckCircle2, Info } from "lucide-react";

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


export default function LeaderboardPage() {
  const [report, setReport] = useState<BenchmarkReport | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/benchmark_report.json")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load benchmark_report.json");
        return res.json();
      })
      .then((data) => {
        setReport(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Could not load report:", err);
        setLoading(false);
      });
  }, []);

  const leaderboard = report?.leaderboard || [];

  return (
    <main className="min-h-screen pt-32 pb-24 px-6 max-w-[1400px] mx-auto flex flex-col gap-10">
      <div className="flex flex-col gap-4">
        <div className="flex items-center gap-2 text-primary font-bold text-sm tracking-widest uppercase">
          <Trophy className="w-4 h-4" /> Empirical Research Findings
        </div>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight font-[family-name:var(--font-poppins)]">
          Vision Robustness Leaderboard
        </h1>
        <div className="flex flex-wrap items-center gap-3">
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
      </div>


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

        {/* Leaderboard Table */}
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
                        <span className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-black ${
                          isTop ? "bg-amber-500 text-white" : "bg-muted text-muted-foreground"
                        }`}>
                          #{index + 1}
                        </span>
                        <span className="text-base text-foreground font-extrabold">{item.display_name}</span>
                      </td>
                      <td className="px-6 text-muted-foreground font-medium">{item.family}</td>
                      <td className="px-6 font-semibold">{(item.clean_accuracy * 100).toFixed(1)}%</td>
                      <td className="px-6 font-black text-primary text-base">
                        {(item.mean_shifted_accuracy * 100).toFixed(1)}%
                      </td>
                      <td className="px-6 font-bold text-destructive">
                        -{dropPct}%
                      </td>
                      <td className="px-6 font-mono font-semibold">
                        {item.relative_robustness.toFixed(3)}
                      </td>
                      <td className="px-6 font-mono font-semibold">
                        {item.mean_ece.toFixed(3)}
                      </td>
                      <td className="px-6 text-muted-foreground font-medium">
                        {item.mean_latency_ms ? `${item.mean_latency_ms.toFixed(1)} ms` : "—"}
                      </td>
                    </tr>
                  );
                })}
                {leaderboard.length === 0 && !loading && (
                  <tr>
                    <td colSpan={8} className="py-8 text-center text-muted-foreground">
                      No evaluation results found. Run `uv run python ml/scripts/run_eval.py` to populate.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Methodology Note */}
        <div className="p-6 rounded-3xl bg-muted/30 border border-border/50 text-sm text-muted-foreground flex items-start gap-3">
          <Info className="w-5 h-5 shrink-0 mt-0.5 text-primary" />
          <p className="leading-relaxed">
            <strong>Evaluation Protocol:</strong> All models evaluated on identical test subsets using deterministic sampling seeds.
            Corruptions applied directly to un-normalized pixel tensors [0, 1] before model-specific channel normalization.
            Robust ratio is computed as Acc(shifted) / Acc(clean).
            Expected Calibration Error (ECE) is computed with M=10 uniform probability bins.
          </p>
        </div>
      </div>
    </main>
  );
}
