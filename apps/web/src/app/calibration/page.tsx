"use client";

import { useEffect, useState } from "react";
import { Card } from "@heroui/react";
import { Target, CheckCircle2, Info, AlertTriangle } from "lucide-react";

interface ConditionBreakdown {
  model: string;
  model_name: string;
  corruption: string;
  severity: number;
  accuracy: number;
  mean_confidence: number;
  ece: number;
  reliability_diagram: {
    accuracy_bins: (number | null)[];
    confidence_bins: (number | null)[];
  };
}

interface BenchmarkReport {
  leaderboard: {
    model: string;
    display_name: string;
    family: string;
    clean_accuracy: number;
    mean_ece: number;
  }[];
  detailed_breakdown: ConditionBreakdown[];
}

export default function CalibrationPage() {
  const [report, setReport] = useState<BenchmarkReport | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("resnet50");

  useEffect(() => {
    fetch("/benchmark_report.json")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load benchmark_report.json");
        return res.json();
      })
      .then((data) => {
        setReport(data);
        if (data.leaderboard && data.leaderboard.length > 0) {
          setSelectedModel(data.leaderboard[0].model);
        }
      })
      .catch((err) => console.error("Error loading calibration data:", err));
  }, []);

  const cleanCondition = report?.detailed_breakdown.find(
    (b) => b.model === selectedModel && b.corruption === "clean"
  );

  const shiftedConditions = report?.detailed_breakdown.filter(
    (b) => b.model === selectedModel && b.corruption !== "clean"
  ) || [];

  const highestShifted = shiftedConditions.reduce<ConditionBreakdown | null>(
    (max, cur) => (!max || cur.severity > max.severity ? cur : max),
    null
  );

  return (
    <main className="min-h-screen pt-32 pb-24 px-6 max-w-[1400px] mx-auto flex flex-col gap-10">
      <div className="flex flex-col gap-4">
        <div className="flex items-center gap-2 text-primary font-bold text-sm tracking-widest uppercase">
          <Target className="w-4 h-4" /> Uncertainty Quantification
        </div>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight font-[family-name:var(--font-poppins)]">
          Model Calibration Analysis
        </h1>
        <p className="text-muted-foreground text-xl max-w-3xl">
          Empirical evaluation of Expected Calibration Error (ECE) and reliability diagrams.
          Measures whether model softmax probabilities reflect true empirical correctness under distribution shift.
        </p>
      </div>

      {/* Model Selector Tabs */}
      {report?.leaderboard && (
        <div className="flex flex-wrap gap-3">
          {report.leaderboard.map((m) => (
            <button
              key={m.model}
              onClick={() => setSelectedModel(m.model)}
              className={`px-5 py-2.5 rounded-2xl font-bold text-sm transition-all border ${
                selectedModel === m.model
                  ? "bg-primary text-primary-foreground border-primary shadow-md"
                  : "bg-card text-muted-foreground border-border/60 hover:text-foreground"
              }`}
            >
              {m.display_name} (Mean ECE: {(m.mean_ece * 100).toFixed(1)}%)
            </button>
          ))}
        </div>
      )}

      {/* Reliability Comparison Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Clean Calibration Card */}
        <Card className="bg-card shadow-lg border border-border/60 rounded-3xl p-8 flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="text-2xl font-extrabold">Clean Distribution Calibration</h3>
                <p className="text-muted-foreground text-sm mt-1">Unperturbed test baseline</p>
              </div>
              <span className="text-xs bg-green-500/10 text-green-500 font-bold px-3 py-1.5 rounded-full uppercase">
                Baseline ECE: {cleanCondition ? `${(cleanCondition.ece * 100).toFixed(1)}%` : "—"}
              </span>
            </div>

            <div className="my-6 p-6 rounded-2xl bg-muted/20 border border-border/40">
              <div className="flex justify-between text-sm font-semibold mb-2">
                <span className="text-muted-foreground">Mean Top-1 Confidence:</span>
                <span className="font-mono text-foreground">
                  {cleanCondition ? `${(cleanCondition.mean_confidence * 100).toFixed(1)}%` : "—"}
                </span>
              </div>
              <div className="flex justify-between text-sm font-semibold">
                <span className="text-muted-foreground">Empirical Accuracy:</span>
                <span className="font-mono text-foreground">
                  {cleanCondition ? `${(cleanCondition.accuracy * 100).toFixed(1)}%` : "—"}
                </span>
              </div>
            </div>
          </div>

          <div className="text-xs text-muted-foreground border-t border-border/40 pt-4">
            Under unshifted inputs, modern neural architectures often exhibit slight overconfidence or well-calibrated confidence in top ranks.
          </div>
        </Card>

        {/* Shifted Calibration Card */}
        <Card className="bg-card shadow-lg border border-border/60 rounded-3xl p-8 flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="text-2xl font-extrabold">Under Severe Distribution Shift</h3>
                <p className="text-muted-foreground text-sm mt-1">
                  {highestShifted
                    ? `${highestShifted.corruption.replace("_", " ")} (Severity ${highestShifted.severity})`
                    : "Shifted test input"}
                </p>
              </div>
              <span className="text-xs bg-destructive/10 text-destructive font-bold px-3 py-1.5 rounded-full uppercase">
                Shifted ECE: {highestShifted ? `${(highestShifted.ece * 100).toFixed(1)}%` : "—"}
              </span>
            </div>

            <div className="my-6 p-6 rounded-2xl bg-muted/20 border border-border/40">
              <div className="flex justify-between text-sm font-semibold mb-2">
                <span className="text-muted-foreground">Mean Top-1 Confidence:</span>
                <span className="font-mono text-foreground">
                  {highestShifted ? `${(highestShifted.mean_confidence * 100).toFixed(1)}%` : "—"}
                </span>
              </div>
              <div className="flex justify-between text-sm font-semibold">
                <span className="text-muted-foreground">Empirical Accuracy:</span>
                <span className="font-mono text-foreground text-destructive">
                  {highestShifted ? `${(highestShifted.accuracy * 100).toFixed(1)}%` : "—"}
                </span>
              </div>
            </div>
          </div>

          <div className="text-xs text-muted-foreground border-t border-border/40 pt-4 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-destructive shrink-0" />
            <span>
              <strong>Calibration Breakdown:</strong> Under distribution shift, accuracy collapses faster than softmax confidence,
              producing high Expected Calibration Error (overconfident errors).
            </span>
          </div>
        </Card>
      </div>

      {/* Mathematical Methodology */}
      <div className="p-6 rounded-3xl bg-muted/30 border border-border/50 text-sm flex flex-col gap-2">
        <h4 className="font-bold text-foreground flex items-center gap-2">
          <Info className="w-4 h-4 text-primary" /> Mathematical Formulation of ECE
        </h4>
        <p className="text-muted-foreground leading-relaxed">
          Samples are partitioned into M=10 equally spaced confidence bins.
          Expected Calibration Error (ECE) is computed as the weighted absolute difference between bin accuracy and bin confidence across all bins: ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|.
        </p>
      </div>

      {/* Critical Distinction Notice: Classifier vs Parameter Calibration */}
      <div className="p-8 rounded-3xl bg-card border border-border/60 shadow-lg flex flex-col gap-4">
        <div className="flex items-center gap-2 text-primary font-bold text-sm tracking-wider uppercase">
          <Target className="w-4 h-4" /> Crucial Methodological Distinction
        </div>
        <h3 className="text-2xl font-bold text-foreground">
          Classifier Probability Calibration vs. Image-Fidelity Parameter Calibration
        </h3>
        <p className="text-sm text-muted-foreground leading-relaxed">
          In RobustLens research, the term <strong>&quot;calibration&quot;</strong> refers to two mathematically distinct operations that must not be conflated:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
          <div className="p-5 rounded-2xl bg-muted/20 border border-border/40 flex flex-col gap-2">
            <span className="text-xs font-bold uppercase tracking-wider text-amber-500">Operation 1 (Evaluated on this page)</span>
            <h4 className="font-bold text-base text-foreground">Classifier Probability Calibration (ECE)</h4>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Measures whether a classifier&apos;s output softmax probabilities correctly match posterior empirical accuracy. Evaluates post-hoc model certainty under distribution shifts.
            </p>
          </div>
          <div className="p-5 rounded-2xl bg-muted/20 border border-border/40 flex flex-col gap-2">
            <span className="text-xs font-bold uppercase tracking-wider text-emerald-500">Operation 2 (Evaluated in Order Effects)</span>
            <h4 className="font-bold text-base text-foreground">Image-Fidelity Parameter Calibration</h4>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Solves parameter optimization on an independent sample (N<sub>cal</sub> = 50) across 6 compound corruption sequences to match low-level fidelity targets on common support (Level 1: 27.20 dB / 0.810; Level 2: 23.80 dB / 0.650; Level 3: 21.80 dB / 0.500; tolerances &plusmn;0.75 dB, &plusmn;0.035).
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
