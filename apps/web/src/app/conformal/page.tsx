"use client";

import { useEffect, useState } from "react";
import { Card } from "@heroui/react";
import { ShieldCheck, CheckCircle2, AlertTriangle, Info, Layers } from "lucide-react";

interface ConformalModelDiag {
  display_name: string;
  alpha: number;
  nominal_coverage: number;
  clean_empirical_coverage: number;
  shifted_empirical_coverage: number;
  coverage_violation_under_shift: number;
}

interface ConformalCoverageItem {
  model: string;
  display_name: string;
  significance_alpha: number;
  nominal_coverage: number;
  nonconformity_threshold: number;
  clean_coverage: number;
  shifted_coverage: number;
  coverage_gap: number;
}

interface BenchmarkReport {
  conformal_diagnostics?: Record<string, ConformalModelDiag>;
  conformal_coverage?: ConformalCoverageItem[];
}

export default function ConformalPage() {
  const [report, setReport] = useState<BenchmarkReport | null>(null);

  useEffect(() => {
    fetch("/benchmark_report.json")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load benchmark_report.json");
        return res.json();
      })
      .then((data) => setReport(data))
      .catch((err) => console.error("Error loading conformal report:", err));
  }, []);

  // Normalize entries from either conformal_coverage (filtered to alpha=0.05) or conformal_diagnostics
  let entries: [string, ConformalModelDiag][] = [];
  if (report?.conformal_coverage && report.conformal_coverage.length > 0) {
    const targetAlpha = 0.05;
    const items = report.conformal_coverage.filter((c) => Math.abs(c.significance_alpha - targetAlpha) < 1e-4);
    entries = items.map((item) => [
      item.model,
      {
        display_name: item.display_name,
        alpha: item.significance_alpha,
        nominal_coverage: item.nominal_coverage,
        clean_empirical_coverage: item.clean_coverage,
        shifted_empirical_coverage: item.shifted_coverage,
        coverage_violation_under_shift: item.coverage_gap,
      },
    ]);
  } else if (report?.conformal_diagnostics) {
    entries = Object.entries(report.conformal_diagnostics);
  }

  return (
    <main className="min-h-screen pt-32 pb-24 px-6 max-w-[1400px] mx-auto flex flex-col gap-10">
      <div className="flex flex-col gap-4">
        <div className="flex items-center gap-2 text-primary font-bold text-sm tracking-widest uppercase">
          <ShieldCheck className="w-4 h-4" /> Distribution-Free Uncertainty Guarantees
        </div>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight font-[family-name:var(--font-poppins)]">
          Split Conformal Prediction Under Shift
        </h1>
        <p className="text-muted-foreground text-xl max-w-3xl">
          Conformal prediction constructs finite-sample prediction sets with guaranteed marginal coverage $1 - \alpha$.
          Here we empirically evaluate the breakdown of coverage guarantees when the fundamental exchangeability assumption is violated by distribution shift.
        </p>
      </div>

      <div className="flex flex-col gap-6 mt-2">
        <div className="flex justify-between items-end">
          <div>
            <h2 className="text-2xl font-bold">Empirical Coverage vs. Nominal Guarantees</h2>
            <p className="text-muted-foreground text-sm mt-1">
              Nominal error budget $\alpha = 0.05$ (Target coverage: 95.0%). Calibrated on clean validation data.
            </p>
          </div>
          <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-bold uppercase tracking-wider border border-primary/20">
            <CheckCircle2 className="w-3.5 h-3.5" /> Measured Benchmark
          </div>
        </div>

        {/* Conformal Diagnostics Table */}
        <div className="border border-border/60 shadow-lg rounded-3xl overflow-hidden bg-card">
          <div className="overflow-x-auto w-full">
            <table className="w-full text-left text-sm whitespace-nowrap">
              <thead className="bg-muted/50 border-b border-border/60 text-xs uppercase tracking-wider">
                <tr>
                  <th className="font-extrabold py-4 px-6 text-foreground">Model Architecture</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Nominal Guarantee</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Clean Coverage</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Shifted Coverage</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Coverage Violation (Gap)</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Exchangeability Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {entries.map(([key, item]) => {
                  const isViolated = item.coverage_violation_under_shift > 0.05;

                  return (
                    <tr key={key} className="hover:bg-muted/20 transition-colors">
                      <td className="font-bold py-5 px-6 text-foreground text-base">
                        {item.display_name}
                      </td>
                      <td className="px-6 font-mono font-bold text-muted-foreground">
                        {(item.nominal_coverage * 100).toFixed(1)}%
                      </td>
                      <td className="px-6 font-mono font-bold text-green-500">
                        {(item.clean_empirical_coverage * 100).toFixed(1)}%
                      </td>
                      <td className="px-6 font-mono font-bold text-destructive">
                        {(item.shifted_empirical_coverage * 100).toFixed(1)}%
                      </td>
                      <td className="px-6 font-mono font-black text-foreground">
                        {(item.coverage_violation_under_shift * 100).toFixed(1)}%
                      </td>
                      <td className="px-6">
                        {isViolated ? (
                          <span className="inline-flex items-center gap-1 bg-destructive/10 text-destructive px-3 py-1 rounded-full text-xs font-bold uppercase">
                            <AlertTriangle className="w-3.5 h-3.5" /> Violated
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 bg-green-500/10 text-green-500 px-3 py-1 rounded-full text-xs font-bold uppercase">
                            <CheckCircle2 className="w-3.5 h-3.5" /> Preserved
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

        {/* Theoretical Takeaway Card */}
        <div className="p-6 md:p-8 rounded-3xl bg-muted/30 border border-border/50 text-sm flex flex-col gap-4">
          <h4 className="font-bold text-foreground text-base flex items-center gap-2">
            <Info className="w-5 h-5 text-primary" /> Key Research Finding: The Exchangeability Gap
          </h4>
          <p className="text-muted-foreground leading-relaxed">
            Split conformal prediction guarantees that coverage holds at 1 - alpha under the strict assumption
            that calibration and test data points are exchangeable (i.i.d.).
            Under distribution shift, the test distribution P_shift(X, Y) != P_clean(X, Y),
            causing the nominal 95% coverage guarantee to degrade significantly.
            This demonstrates why safety-critical vision systems cannot rely on naive static conformal calibration when deployed into real-world, dynamic environments.
          </p>
        </div>
      </div>
    </main>
  );
}
