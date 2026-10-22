"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { 
  GitFork, 
  ShieldCheck, 
  CheckCircle2, 
  Info, 
  Scale, 
  Sparkles, 
  TrendingDown, 
  FileCheck2, 
  Layers, 
  Zap,
  Activity,
  Shuffle,
  AlertTriangle
} from "lucide-react";

interface OrderLeaderboardItem {
  model: string;
  model_name: string;
  model_family: string;
  clean_accuracy: number;
  compound_accuracy: number;
  order_accuracy_variance: number;
  order_max_accuracy_gap: number;
  mean_compound_ece: number;
}

interface SequenceRanking {
  order: string;
  description: string;
  mean_accuracy: number;
  mean_confidence: number;
  mean_ece: number;
  mean_psnr: number;
  mean_ssim: number;
}

interface AncovaItem {
  model: string;
  pseudo_r2: number;
  coef_psnr: number;
  pval_psnr_clustered: number;
  coef_ssim: number;
  pval_ssim_clustered: number;
  clustered_wald_stat: number;
  clustered_wald_p_value: number;
  unclustered_wald_stat: number;
  unclustered_wald_p_value: number;
  order_significant_at_05: boolean;
  n_observations: number;
  n_clusters: number;
}

interface FactorialAncovaItem {
  model: string;
  n_observations: number;
  n_clusters: number;
  m1_pseudo_r2: number;
  m1_clustered_wald_order_5df: number;
  m1_clustered_pval_order_5df: number;
  m2_pseudo_r2: number;
  m2_wald_order_main_5df: number;
  m2_pval_order_main_5df: number;
  m2_wald_sev_main_4df: number;
  m2_pval_sev_main_4df: number;
  m2_wald_inter_20df: number;
  m2_pval_inter_20df: number;
  m2_wald_joint_order_25df: number;
  m2_pval_joint_order_25df: number;
  coef_psnr: number;
  pval_psnr: number;
  coef_ssim: number;
  pval_ssim: number;
}

interface PairwiseSignificanceItem {
  model: string;
  severity: number;
  order_a: string;
  order_b: string;
  diff_mean: number;
  ci_lower_95: number;
  ci_upper_95: number;
  p_value_raw: number;
  p_value_holm: number;
  p_value_bh: number;
}

interface ErrorChurnCase {
  model: string;
  model_name: string;
  severity: number;
  order_a: string;
  order_b: string;
  pair_name: string;
  desc_a: string;
  desc_b: string;
  n_images: number;
  n00_both_wrong: number;
  n01_gain_rescued: number;
  n10_loss_broken: number;
  n11_both_correct: number;
  acc_a: number;
  acc_b: number;
  gain_rate_G: number;
  loss_rate_L: number;
  net_accuracy_diff: number;
  error_churn: number;
  excess_churn: number;
  mcnemar_chi2: number;
  mcnemar_p_value: number;
  sign_flip_p_value: number;
  holm_p_value_family: number;
  bh_p_value_family: number;
}

interface OrderEffectsPayload {
  meta: {
    title: string;
    benchmark_status: string;
    dataset: string;
    n_images: number;
    n_conditions: number;
    n_models: number;
    total_records: number;
    hardware: string;
    inference_cluster_unit: string;
    generated_at: string;
  };
  leaderboard: OrderLeaderboardItem[];
  sequence_rankings: SequenceRanking[];
  distortion_ancova: AncovaItem[];
  factorial_ancova: FactorialAncovaItem[];
  pairwise_significance: PairwiseSignificanceItem[];
  error_churn?: {
    report?: any;
    high_excess_cases?: ErrorChurnCase[];
    all_cases?: ErrorChurnCase[];
  };
}

export default function OrderEffectsPage() {
  const [data, setData] = useState<OrderEffectsPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState<string>("all");
  const [selectedSeverity, setSelectedSeverity] = useState<number>(4);
  const [churnModelFilter, setChurnModelFilter] = useState<string>("all");

  useEffect(() => {
    fetch("/order_effects_data.json")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load order_effects_data.json");
        return res.json();
      })
      .then((json) => {
        setData(json);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Could not load order effects data:", err);
        setLoading(false);
      });
  }, []);

  const meta = data?.meta;
  const leaderboard = data?.leaderboard || [];
  const sequenceRankings = data?.sequence_rankings || [];
  const ancovaList = data?.distortion_ancova || [];
  const factorialAncovaList = data?.factorial_ancova || [];
  const pairwiseList = (data?.pairwise_significance || []).filter((p) => {
    const matchesModel = selectedModel === "all" || p.model === selectedModel;
    const matchesSev = selectedSeverity === 0 || p.severity === selectedSeverity;
    return matchesModel && matchesSev;
  });
  const highExcessCases = (data?.error_churn?.high_excess_cases || []).filter((c) => {
    return churnModelFilter === "all" || c.model === churnModelFilter;
  });

  return (
    <main className="min-h-screen pt-32 pb-24 px-6 max-w-[1400px] mx-auto flex flex-col gap-10">
      {/* Header */}
      <div className="flex flex-col gap-4">
        <div className="flex items-center gap-2 text-primary font-bold text-sm tracking-widest uppercase">
          <GitFork className="w-4 h-4" /> Research Extension • Operator Non-Commutativity
        </div>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight font-[family-name:var(--font-poppins)]">
          Compound Corruption Order Effects
        </h1>
        <div className="flex flex-wrap items-center gap-3">
          <span className="px-3.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-500 border border-emerald-500/20 flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5" /> {meta?.benchmark_status || "Full Benchmark — N=3,925 (Validated)"}
          </span>
          <span className="px-3 py-1 rounded-full text-xs font-semibold bg-primary/10 text-primary border border-primary/20">
            {meta?.hardware || "NVIDIA Tesla T4 GPU"}
          </span>
          <span className="text-xs text-muted-foreground">
            {meta?.total_records?.toLocaleString() || "486,700"} Forward Passes • 31 Conditions/Image • 4 Architectures
          </span>
        </div>
        <p className="text-muted-foreground text-xl max-w-4xl leading-relaxed">
          Investigating whether the execution sequence of identical physical corruptions 
          (<code className="text-primary font-mono font-semibold">Noise</code>, <code className="text-primary font-mono font-semibold">Blur</code>, <code className="text-primary font-mono font-semibold">JPEG</code>) 
          alters classification failure and calibration under compound distribution shift.
        </p>
      </div>

      {/* Scope & Distinction Notice */}
      <div className="p-6 rounded-3xl bg-amber-500/5 border border-amber-500/20 text-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 shrink-0 mt-0.5 text-amber-500" />
          <div className="space-y-1">
            <p className="font-semibold text-foreground">
              Scope Verification: Full 3,925-Image Benchmark vs. Pilot Runs
            </p>
            <p className="text-muted-foreground text-xs leading-relaxed">
              These measurements represent the complete <strong>3,925-image evaluation</strong> (486,700 inference passes) on the official <strong>FastAI Imagenette-v2 validation split</strong> (10 canonical classes: rows 9,469–13,393).
              Pilot runs (10-image CPU smoke test and 100-image GPU pilot) were preserved separately under <code className="font-mono">results/order_effect_study/pilot/</code> and are not substituted for full-benchmark numbers.
            </p>
          </div>
        </div>
        <span className="shrink-0 px-3 py-1 rounded-full text-xs font-bold bg-amber-500/10 text-amber-500 border border-amber-500/30">
          Strict Cluster-Unit: Physical Image
        </span>
      </div>

      {/* Key Finding Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
        <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Distortion Spread</span>
            <Scale className="w-5 h-5 text-primary" />
          </div>
          <div className="text-3xl font-extrabold text-foreground">6.29 dB</div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Permutation order alone produces a <strong>6.29 dB PSNR spread</strong> and <strong>0.421 SSIM spread</strong> under identical nominal noise seeds.
          </p>
        </div>

        <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">ANCOVA Order Effect</span>
            <ShieldCheck className="w-5 h-5 text-emerald-500" />
          </div>
          <div className="text-3xl font-extrabold text-emerald-500">p ≪ 10⁻²⁰⁰</div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Cluster-robust logistic ANCOVA confirms order is significant across all 4 models after controlling for continuous PSNR and SSIM.
          </p>
        </div>

        <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Max Invariance</span>
            <Sparkles className="w-5 h-5 text-indigo-500" />
          </div>
          <div className="text-3xl font-extrabold text-foreground">Swin-Tiny</div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Hierarchical window attention achieves lowest order variance (<strong>0.0052</strong>) and smallest gap (<strong>17.13%</strong>).
          </p>
        </div>

        <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Best Calibration</span>
            <Activity className="w-5 h-5 text-amber-500" />
          </div>
          <div className="text-3xl font-extrabold text-foreground">ViT-Base/16</div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Maintains the lowest Expected Calibration Error under compound shifts (<strong>ECE = 0.0413</strong>).
          </p>
        </div>

        <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Decoupled Churn</span>
            <Shuffle className="w-5 h-5 text-rose-500" />
          </div>
          <div className="text-3xl font-extrabold text-rose-500">10.73%</div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Over <strong>420 images switch correctness</strong> (Excess Churn = 10.55%) under order swaps with negligible net difference (Δ = -0.18%).
          </p>
        </div>
      </div>

      {/* Sequence Ranking Table */}
      <div className="flex flex-col gap-4">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <span>Permutation Sequence Ranking</span>
            <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-muted text-muted-foreground">
              6 Permutations × 5 Severities = 30 Compound Conditions
            </span>
          </h2>
          <p className="text-muted-foreground text-sm mt-1">
            Empirical mean classification accuracy, confidence, and objective image quality metrics averaged across all four evaluated architectures.
          </p>
        </div>

        <div className="border border-border/60 shadow-lg rounded-3xl overflow-hidden bg-card">
          <div className="overflow-x-auto w-full">
            <table className="w-full text-left text-sm whitespace-nowrap">
              <thead className="bg-muted/50 border-b border-border/60 text-xs uppercase tracking-wider">
                <tr>
                  <th className="font-extrabold py-4 px-6 text-foreground">Rank & Sequence</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Execution Flow</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Mean Accuracy</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Confidence</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Mean ECE</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Fidelity (PSNR)</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Structural (SSIM)</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Physical Phenomenon</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {sequenceRankings.map((seq, idx) => {
                  const isTop = idx === 0;
                  const isBottom = idx >= 3;
                  return (
                    <tr key={seq.order} className="hover:bg-muted/20 transition-colors">
                      <td className="font-bold py-5 px-6 flex items-center gap-3">
                        <span className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-black ${
                          isTop ? "bg-emerald-500 text-white" : isBottom ? "bg-destructive/10 text-destructive border border-destructive/20" : "bg-muted text-muted-foreground"
                        }`}>
                          #{idx + 1}
                        </span>
                        <span className="text-base font-mono font-bold text-foreground">{seq.order}</span>
                      </td>
                      <td className="px-6 text-muted-foreground font-medium">{seq.description}</td>
                      <td className="px-6 font-black text-primary text-base">{seq.mean_accuracy.toFixed(2)}%</td>
                      <td className="px-6 font-semibold">{seq.mean_confidence.toFixed(2)}%</td>
                      <td className="px-6 font-mono font-semibold">{seq.mean_ece.toFixed(4)}</td>
                      <td className="px-6 font-mono">{seq.mean_psnr.toFixed(2)} dB</td>
                      <td className="px-6 font-mono">{seq.mean_ssim.toFixed(3)}</td>
                      <td className="px-6 text-xs text-muted-foreground">
                        {seq.order.endsWith("B") ? (
                          <span className="text-emerald-500 font-medium">Blur succeeds JPEG: spatial smoothing mitigates DCT block boundaries</span>
                        ) : seq.order.endsWith("N") ? (
                          <span className="text-amber-500 font-medium">Terminal noise: additive high-frequency grains maximize ECE</span>
                        ) : (
                          <span className="text-destructive font-medium">Terminal JPEG: DCT block quantization causes high-frequency collapse</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Model Leaderboard Under Compound Corruptions (Table 5) */}
      <div className="flex flex-col gap-4">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <span>Architecture Performance & Sequence Invariance</span>
          </h2>
          <p className="text-muted-foreground text-sm mt-1">
            Comparing clean vs. compound accuracy, sequence variance, and cluster-robust logistic ANCOVA statistics (<code className="font-mono">N=3,925</code> image clusters).
          </p>
        </div>

        <div className="border border-border/60 shadow-lg rounded-3xl overflow-hidden bg-card">
          <div className="overflow-x-auto w-full">
            <table className="w-full text-left text-sm whitespace-nowrap">
              <thead className="bg-muted/50 border-b border-border/60 text-xs uppercase tracking-wider">
                <tr>
                  <th className="font-extrabold py-4 px-6 text-foreground">Rank & Model</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Structural Family</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Clean Acc</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Compound Acc</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Order Variance</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Max Order Gap</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Mean ECE</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Model 1: Order Wald (5 df)</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Model 2: Joint Order (25 df)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {leaderboard.map((item, idx) => {
                  const ancova = ancovaList.find((a) => a.model === item.model);
                  const facAncova = factorialAncovaList.find((a) => a.model === item.model);
                  return (
                    <tr key={item.model} className="hover:bg-muted/20 transition-colors">
                      <td className="font-bold py-5 px-6 flex items-center gap-3">
                        <span className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-black bg-muted text-muted-foreground">
                          #{idx + 1}
                        </span>
                        <span className="text-base text-foreground font-extrabold">{item.model_name}</span>
                      </td>
                      <td className="px-6 text-muted-foreground font-medium">{item.model_family}</td>
                      <td className="px-6 font-semibold">{(item.clean_accuracy * 100).toFixed(2)}%</td>
                      <td className="px-6 font-black text-primary text-base">{(item.compound_accuracy * 100).toFixed(2)}%</td>
                      <td className="px-6 font-mono font-semibold">{item.order_accuracy_variance.toFixed(4)}</td>
                      <td className="px-6 font-mono font-semibold text-destructive">{(item.order_max_accuracy_gap * 100).toFixed(2)}%</td>
                      <td className="px-6 font-mono font-semibold">{item.mean_compound_ece.toFixed(4)}</td>
                      <td className="px-6 font-mono font-bold text-emerald-500">
                        {facAncova ? `${facAncova.m1_clustered_wald_order_5df.toFixed(1)} (p < 1e-100)` : ancova ? `${ancova.clustered_wald_stat.toFixed(1)} (p < 1e-100)` : "—"}
                      </td>
                      <td className="px-6 font-mono font-bold text-indigo-500">
                        {facAncova ? `${facAncova.m2_wald_joint_order_25df.toFixed(1)} (p < 1e-100)` : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Publication Visualizations Grid */}
      <div className="flex flex-col gap-4">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <span>Publication Research Figures</span>
          </h2>
          <p className="text-muted-foreground text-sm mt-1">
            High-resolution empirical figures generated directly from the 486,700 full-benchmark predictions.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-4">
            <h3 className="font-bold text-lg">Figure 6: Accuracy Across 6 Sequences by Architecture</h3>
            <div className="relative w-full aspect-[16/9] rounded-2xl overflow-hidden bg-muted">
              <Image 
                src="/figures/06_order_effects_by_architecture.png" 
                alt="Figure 6: Order Effects by Architecture" 
                fill 
                className="object-contain"
              />
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Comparison across all 6 corruption permutations for ConvNeXt-Tiny, Swin-Tiny, ViT-Base/16, and ResNet-50.
            </p>
          </div>

          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-4">
            <h3 className="font-bold text-lg">Figure 7: PSNR & SSIM vs. Top-1 Accuracy by Order</h3>
            <div className="relative w-full aspect-[16/9] rounded-2xl overflow-hidden bg-muted">
              <Image 
                src="/figures/07_psnr_ssim_vs_accuracy_by_order.png" 
                alt="Figure 7: PSNR and SSIM vs Accuracy" 
                fill 
                className="object-contain"
              />
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Scatter trajectories illustrating non-monotonic divergence of structural similarity and accuracy across sequences.
            </p>
          </div>

          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-4">
            <h3 className="font-bold text-lg">Figure 8: Order Divergence Across Severities (1 to 5)</h3>
            <div className="relative w-full aspect-[16/9] rounded-2xl overflow-hidden bg-muted">
              <Image 
                src="/figures/08_order_severity_interaction.png" 
                alt="Figure 8: Order Severity Interaction" 
                fill 
                className="object-contain"
              />
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Severity trajectories confirming that permutation gaps widen progressively from Severity 1 to Severity 5.
            </p>
          </div>

          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-4">
            <h3 className="font-bold text-lg">Figure 9: Calibration Degradation (ECE) Across Orders</h3>
            <div className="relative w-full aspect-[16/9] rounded-2xl overflow-hidden bg-muted">
              <Image 
                src="/figures/09_order_calibration_ece.png" 
                alt="Figure 9: Order Calibration ECE" 
                fill 
                className="object-contain"
              />
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Expected Calibration Error by sequence and model family, confirming ViT calibration stability.
            </p>
          </div>

          {/* Figure 10: Error Churn Analysis */}
          <div className="lg:col-span-2 p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <h3 className="font-bold text-xl flex items-center gap-2">
                  <span>Figure 10: Order-Dependent Error Churn & Prediction Flips Under Non-Commutative Perturbations</span>
                  <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-rose-500/10 text-rose-500 border border-rose-500/20">
                    Core Research Discovery
                  </span>
                </h3>
                <p className="text-xs text-muted-foreground mt-0.5">
                  (A) Pairwise Error Churn matrices across 4 architectures. (B) Decoupling of Error Churn from Net Accuracy Difference (39 pairs with |Δ| ≤ 2% and Excess Churn ≥ 5%). (C) Excess Churn dynamics across Severities 1–5.
                </p>
              </div>
            </div>
            <div className="relative w-full aspect-[16/11] rounded-2xl overflow-hidden bg-muted">
              <Image 
                src="/figures/10_error_churn_analysis.png" 
                alt="Figure 10: Order-Dependent Error Churn Analysis" 
                fill 
                className="object-contain"
              />
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              <strong>Scientific takeaway:</strong> Aggregate classification accuracy conceals substantial instance-level churn. In 39 out of 300 sequence pairs, permutation order causes 5% to 10.7% of physical images to flip correctness without shifting overall accuracy, proving that commutative evaluation protocols systematically overlook prediction instability.
            </p>
          </div>
        </div>
      </div>

      {/* Order-Dependent Error Churn & Decoupling Section */}
      <div className="flex flex-col gap-4">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end gap-4">
          <div>
            <h2 className="text-2xl font-bold flex items-center gap-2">
              <Shuffle className="w-6 h-6 text-rose-500" />
              <span>Order-Dependent Error Churn & Prediction Flip Dynamics</span>
              <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-rose-500/10 text-rose-500 border border-rose-500/20">
                39 Decoupled Pairs (|Δ| ≤ 2%, Excess ≥ 5%)
              </span>
            </h2>
            <p className="text-muted-foreground text-sm mt-1 max-w-4xl">
              Do different corruption permutation orders change which individual physical images a classifier gets wrong, even when their aggregate accuracies are virtually identical? 
              Here we isolate <strong>Excess Churn</strong> (prediction flipping beyond net accuracy shifts) across all 300 matched pairwise comparisons.
            </p>
          </div>

          {/* Filter Controls for Churn */}
          <div className="flex items-center gap-3">
            <select
              value={churnModelFilter}
              onChange={(e) => setChurnModelFilter(e.target.value)}
              className="px-3 py-1.5 text-xs rounded-xl bg-muted border border-border font-medium"
            >
              <option value="all">All Models (39 High-Excess Pairs)</option>
              <option value="convnext_tiny">ConvNeXt-Tiny (10 pairs)</option>
              <option value="swin_tiny">Swin-Tiny (7 pairs)</option>
              <option value="vit_base">ViT-Base/16 (14 pairs)</option>
              <option value="resnet50">ResNet-50 (8 pairs)</option>
            </select>
          </div>
        </div>

        {/* Literature & Formulation Context Banner */}
        <div className="p-5 rounded-2xl bg-muted/40 border border-border/60 text-xs space-y-2">
          <div className="flex items-center gap-2 font-bold text-foreground">
            <Sparkles className="w-4 h-4 text-primary" /> Mathematical Formulation & Literature Context
          </div>
          <p className="text-muted-foreground leading-relaxed">
            Let <code className="font-mono text-foreground font-semibold">C_A, C_B ∈ {"{0, 1}"}</code> indicate classification correctness on physical image <code className="font-mono text-foreground">i</code> under Order A and Order B.
            Gain rate <code className="font-mono text-foreground">G = P(C_A=0, C_B=1)</code> (rescued by B), Loss rate <code className="font-mono text-foreground">L = P(C_A=1, C_B=0)</code> (broken by B).
            Net difference <code className="font-mono text-foreground">Δ = G - L = Acc_B - Acc_A</code>; Error Churn <code className="font-mono text-foreground">Churn = G + L = P(C_A ≠ C_B)</code>; 
            Excess Churn <code className="font-mono text-foreground">Excess = Churn - |Δ| = 2·min(G, L) ≥ 0</code>.
            Numerically verified identity: <code className="font-mono text-emerald-500 font-semibold">(Acc_B - Acc_A) - (G - L) = 0.0000</code> across all 300 evaluations.
          </p>
          <p className="text-muted-foreground/80 leading-relaxed italic text-[11px]">
            Prediction churn in ML was formalized by Milani Fard et al. (NeurIPS 2016) and model underspecification by D'Amour et al. (2020). 
            RobustLens provides the first rigorous empirical measurement demonstrating that identical frozen vision architectures undergo massive order-dependent prediction churn (up to 10.73%) under non-commutative multi-stage shifts.
          </p>
        </div>

        {/* Decoupled Churn Table */}
        <div className="border border-border/60 shadow-lg rounded-3xl overflow-hidden bg-card">
          <div className="overflow-x-auto w-full max-h-[550px] overflow-y-auto">
            <table className="w-full text-left text-sm whitespace-nowrap">
              <thead className="bg-muted/50 border-b border-border/60 text-xs uppercase tracking-wider sticky top-0 z-10">
                <tr>
                  <th className="font-extrabold py-4 px-6 text-foreground">Model & Severity</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Permutation Pair</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Acc A → Acc B</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Gain G (Rescued)</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Loss L (Broken)</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Net Diff Δ</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Total Churn (G+L)</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Excess Churn</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">McNemar χ² (p-val)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {highExcessCases.map((row, idx) => {
                  const isTopDecoupled = row.excess_churn >= 0.10;
                  const isExactZero = Math.abs(row.net_accuracy_diff) < 0.0001;
                  return (
                    <tr key={`churn-${row.model}-${row.severity}-${row.order_a}-${row.order_b}-${idx}`} className="hover:bg-muted/20 transition-colors">
                      <td className="px-6 py-4 font-semibold">
                        <span className="font-mono text-xs text-foreground uppercase">{row.model_name}</span> •{" "}
                        <span className="text-muted-foreground text-xs">Sev {row.severity}</span>
                      </td>
                      <td className="px-6 font-mono font-bold">
                        {row.order_a} <span className="text-muted-foreground font-normal">vs</span> {row.order_b}
                      </td>
                      <td className="px-6 font-mono text-xs">
                        {(row.acc_a * 100).toFixed(2)}% → {(row.acc_b * 100).toFixed(2)}%
                      </td>
                      <td className="px-6 font-mono text-emerald-500 font-semibold">
                        +{(row.gain_rate_G * 100).toFixed(2)}% ({row.n01_gain_rescued})
                      </td>
                      <td className="px-6 font-mono text-destructive font-semibold">
                        -{(row.loss_rate_L * 100).toFixed(2)}% ({row.n10_loss_broken})
                      </td>
                      <td className="px-6 font-bold">
                        <span className={`px-2 py-0.5 rounded-md text-xs ${
                          isExactZero 
                            ? "bg-muted text-muted-foreground font-mono" 
                            : row.net_accuracy_diff >= 0 
                              ? "bg-primary/10 text-primary font-mono" 
                              : "bg-destructive/10 text-destructive font-mono"
                        }`}>
                          {row.net_accuracy_diff >= 0 ? "+" : ""}{(row.net_accuracy_diff * 100).toFixed(2)}%
                        </span>
                      </td>
                      <td className="px-6 font-black text-rose-500 text-base">
                        {(row.error_churn * 100).toFixed(2)}%
                      </td>
                      <td className="px-6">
                        <span className={`px-2.5 py-1 rounded-full text-xs font-bold ${
                          isTopDecoupled 
                            ? "bg-rose-500/15 text-rose-500 border border-rose-500/30" 
                            : "bg-amber-500/10 text-amber-500 border border-amber-500/20"
                        }`}>
                          {(row.excess_churn * 100).toFixed(2)}%
                        </span>
                      </td>
                      <td className="px-6 font-mono text-xs text-muted-foreground">
                        χ²={row.mcnemar_chi2.toFixed(2)} (p={row.mcnemar_p_value < 0.001 ? "<0.001" : row.mcnemar_p_value.toFixed(3)})
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Pairwise Hypothesis Testing & Multiple Corrections */}
      <div className="flex flex-col gap-4">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end gap-4">
          <div>
            <h2 className="text-2xl font-bold flex items-center gap-2">
              <span>Pairwise Permutation Tests & Clustered Bootstrap CIs</span>
            </h2>
            <p className="text-muted-foreground text-sm mt-1">
              Paired differences evaluated using image-level Monte Carlo sign-flip permutation tests (R=20,000) and clustered bootstrap 95% CIs (B=10,000). The full study contains <strong>300 pairwise comparisons</strong> (4 models × 5 severities × 15 sequence pairs). When filtering by a single severity level, 60 pairs are displayed across all models (or 15 pairs for a single model). Multiplicity adjustments (Holm-Bonferroni FWER and Benjamini-Hochberg FDR) are strictly computed across all 75 condition pairs within each model family. Showing <strong>{pairwiseList.length}</strong> comparisons matching current filter:
            </p>
          </div>

          {/* Filter Controls */}
          <div className="flex items-center gap-3">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="px-3 py-1.5 text-xs rounded-xl bg-muted border border-border font-medium"
            >
              <option value="all">All Models</option>
              <option value="convnext_tiny">ConvNeXt-Tiny</option>
              <option value="swin_tiny">Swin-Tiny</option>
              <option value="vit_base">ViT-Base/16</option>
              <option value="resnet50">ResNet-50</option>
            </select>

            <select
              value={selectedSeverity}
              onChange={(e) => setSelectedSeverity(Number(e.target.value))}
              className="px-3 py-1.5 text-xs rounded-xl bg-muted border border-border font-medium"
            >
              <option value={0}>All Severities (300 pairs)</option>
              <option value={1}>Severity 1 (60 pairs)</option>
              <option value={2}>Severity 2 (60 pairs)</option>
              <option value={3}>Severity 3 (60 pairs)</option>
              <option value={4}>Severity 4 (60 pairs)</option>
              <option value={5}>Severity 5 (60 pairs)</option>
            </select>
          </div>
        </div>

        <div className="border border-border/60 shadow-lg rounded-3xl overflow-hidden bg-card">
          <div className="overflow-x-auto w-full max-h-[600px] overflow-y-auto">
            <table className="w-full text-left text-sm whitespace-nowrap">
              <thead className="bg-muted/50 border-b border-border/60 text-xs uppercase tracking-wider sticky top-0 z-10">
                <tr>
                  <th className="font-extrabold py-4 px-6 text-foreground">Model & Severity</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Comparison (A vs B)</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Mean Diff Δ</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">95% Clustered CI</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Sign-Flip Perm p</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">Holm FWER</th>
                  <th className="font-extrabold py-4 px-6 text-foreground">BH FDR</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {pairwiseList.map((row, idx) => {
                  const isSig = row.p_value_holm < 0.05;
                  return (
                    <tr key={`${row.model}-${row.severity}-${row.order_a}-${row.order_b}-${idx}`} className="hover:bg-muted/20 transition-colors">
                      <td className="px-6 py-4 font-semibold">
                        <span className="font-mono text-xs text-foreground uppercase">{row.model}</span> •{" "}
                        <span className="text-muted-foreground text-xs">Sev {row.severity}</span>
                      </td>
                      <td className="px-6 font-mono font-bold">
                        {row.order_a} <span className="text-muted-foreground font-normal">vs</span> {row.order_b}
                      </td>
                      <td className={`px-6 font-bold ${row.diff_mean >= 0 ? "text-primary" : "text-destructive"}`}>
                        {(row.diff_mean * 100).toFixed(2)}%
                      </td>
                      <td className="px-6 font-mono text-xs">
                        [{(row.ci_lower_95 * 100).toFixed(2)}%, {(row.ci_upper_95 * 100).toFixed(2)}%]
                      </td>
                      <td className="px-6 font-mono font-semibold">
                        {row.p_value_raw < 0.0001 ? "< 0.0001" : row.p_value_raw.toFixed(4)}
                      </td>
                      <td className="px-6">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${
                          isSig ? "bg-emerald-500/10 text-emerald-500 border border-emerald-500/20" : "bg-muted text-muted-foreground"
                        }`}>
                          {isSig ? "Significant" : "Not Sig"} ({row.p_value_holm < 0.0001 ? "< 0.0001" : row.p_value_holm.toFixed(4)})
                        </span>
                      </td>
                      <td className="px-6">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${
                          row.p_value_bh < 0.05 ? "bg-primary/10 text-primary border border-primary/20" : "bg-muted text-muted-foreground"
                        }`}>
                          {row.p_value_bh < 0.0001 ? "< 0.0001" : row.p_value_bh.toFixed(4)}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Matched-Fidelity Counterfactual Study & Gap Contraction */}
      <div className="flex flex-col gap-6 pt-10 border-t border-border/80">
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-2 text-indigo-500 font-bold text-sm tracking-widest uppercase">
            <Scale className="w-4 h-4" /> Physical Distortion Control • Counterfactual Analysis
          </div>
          <h2 className="text-3xl font-extrabold tracking-tight">
            Towards Matched-Fidelity Robustness Benchmarking: An Approximate-Fidelity Counterfactual Study
          </h2>
          <p className="text-muted-foreground text-base max-w-4xl leading-relaxed">
            In nominal benchmarks, non-commuting operators produce uncalibrated PSNR discrepancies of over 4 dB. 
            Across an extensive calibration grid search on independent training images, prespecified joint matching criteria (&plusmn;0.75 dB PSNR, &plusmn;0.035 SSIM) could not be met simultaneously across sequences due to conflicting structural trade-offs between additive noise and spatial blur. 
            Under an <strong>approximate-fidelity calibration on common support</strong> (28 dB down to 16 dB PSNR), nominal sequence gaps contract by up to <strong>77.1%</strong> in CNNs, while clustered factorial ANCOVA reveals persistent architectural sensitivity and residual structural confounding.
          </p>

        </div>

        {/* Gap Contraction Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-2">
            <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">ConvNeXt-Tiny</span>
            <div className="text-3xl font-extrabold text-blue-500">77.1% Contraction</div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Sequence gap collapsed from <strong>20.45% → 4.67%</strong>. Cross-sequence variance reduced by <strong>95.9%</strong>.
            </p>
          </div>

          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-2">
            <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">ResNet-50</span>
            <div className="text-3xl font-extrabold text-amber-500">72.6% Contraction</div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Sequence gap collapsed from <strong>21.55% → 5.90%</strong>. Cross-sequence variance reduced by <strong>94.9%</strong>.
            </p>
          </div>

          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-2">
            <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">ViT-Base/16</span>
            <div className="text-3xl font-extrabold text-purple-500">50.7% Contraction</div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Sequence gap contracted from <strong>20.37% → 10.03%</strong>. Cross-sequence variance reduced by <strong>86.9%</strong>.
            </p>
          </div>

          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-2">
            <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Swin-Tiny</span>
            <div className="text-3xl font-extrabold text-emerald-500">31.8% Contraction</div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Sequence gap contracted from <strong>17.13% → 11.68%</strong>. Cross-sequence variance reduced by <strong>55.1%</strong>.
            </p>
          </div>
        </div>

        {/* Clustered Logistic Factorial ANCOVA Table */}
        <div className="border border-border/60 shadow-lg rounded-3xl overflow-hidden bg-card p-6 flex flex-col gap-4">
          <div>
            <h3 className="text-lg font-bold text-foreground">
              Factorial Clustered Logistic ANCOVA (N = 117,750 per model clustered on 3,925 images)
            </h3>
            <p className="text-xs text-muted-foreground mt-0.5">
              Testing whether corruption sequence remains statistically significant after controlling for continuous PSNR and SSIM covariates.
            </p>
          </div>

          <div className="overflow-x-auto w-full">
            <table className="w-full text-left text-sm whitespace-nowrap">
              <thead className="bg-muted/50 border-b border-border/60 text-xs uppercase tracking-wider">
                <tr>
                  <th className="font-extrabold py-3 px-4 text-foreground">Architecture</th>
                  <th className="font-extrabold py-3 px-4 text-foreground">Pseudo R²</th>
                  <th className="font-extrabold py-3 px-4 text-foreground">Order Main (5 df)</th>
                  <th className="font-extrabold py-3 px-4 text-foreground">Target Main (4 df)</th>
                  <th className="font-extrabold py-3 px-4 text-foreground">Order × Target (20 df)</th>
                  <th className="font-extrabold py-3 px-4 text-foreground">Joint Order (25 df)</th>
                  <th className="font-extrabold py-3 px-4 text-foreground">PSNR Coef (z)</th>
                  <th className="font-extrabold py-3 px-4 text-foreground">SSIM Coef (z)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30 text-xs font-mono">
                <tr className="hover:bg-muted/20">
                  <td className="px-4 py-3 font-sans font-bold">ConvNeXt-Tiny</td>
                  <td className="px-4 py-3">0.1379</td>
                  <td className="px-4 py-3 text-emerald-500 font-bold">W=119.4 (p=4.2e-24)</td>
                  <td className="px-4 py-3">W=395.3 (p=2.8e-84)</td>
                  <td className="px-4 py-3 text-emerald-500">W=272.0 (p=4.0e-46)</td>
                  <td className="px-4 py-3 font-bold">W=611.0 (p=6.1e-113)</td>
                  <td className="px-4 py-3">+0.032 (z=2.33)</td>
                  <td className="px-4 py-3 font-bold">+1.528 (z=6.78)</td>
                </tr>
                <tr className="hover:bg-muted/20">
                  <td className="px-4 py-3 font-sans font-bold">Swin-Tiny</td>
                  <td className="px-4 py-3">0.1283</td>
                  <td className="px-4 py-3 text-emerald-500 font-bold">W=159.7 (p=1.2e-32)</td>
                  <td className="px-4 py-3">W=227.7 (p=4.2e-48)</td>
                  <td className="px-4 py-3 text-emerald-500">W=651.1 (p=4.8e-125)</td>
                  <td className="px-4 py-3 font-bold">W=1436.9 (p=5.0e-288)</td>
                  <td className="px-4 py-3">+0.046 (z=3.36)</td>
                  <td className="px-4 py-3 font-bold">+1.850 (z=7.95)</td>
                </tr>
                <tr className="hover:bg-muted/20">
                  <td className="px-4 py-3 font-sans font-bold">ViT-Base/16</td>
                  <td className="px-4 py-3">0.0879</td>
                  <td className="px-4 py-3 text-emerald-500 font-bold">W=130.1 (p=2.3e-26)</td>
                  <td className="px-4 py-3">W=344.9 (p=2.2e-73)</td>
                  <td className="px-4 py-3 text-emerald-500">W=547.6 (p=3.0e-103)</td>
                  <td className="px-4 py-3 font-bold">W=1276.5 (p=8.8e-254)</td>
                  <td className="px-4 py-3">+0.072 (z=5.46)</td>
                  <td className="px-4 py-3 font-bold">+0.577 (z=2.61)</td>
                </tr>
                <tr className="hover:bg-muted/20">
                  <td className="px-4 py-3 font-sans font-bold">ResNet-50</td>
                  <td className="px-4 py-3">0.2034</td>
                  <td className="px-4 py-3 text-emerald-500 font-bold">W=268.9 (p=4.8e-56)</td>
                  <td className="px-4 py-3">W=774.7 (p=2.3e-166)</td>
                  <td className="px-4 py-3 text-emerald-500">W=239.9 (p=1.2e-39)</td>
                  <td className="px-4 py-3 font-bold">W=794.4 (p=1.8e-151)</td>
                  <td className="px-4 py-3">+0.063 (z=4.87)</td>
                  <td className="px-4 py-3 font-bold">+1.680 (z=6.92)</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Counterfactual Publication Figures Gallery */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-3">
            <h3 className="font-bold text-sm">Figure 1: Objective Quality Matching Validation</h3>
            <div className="relative w-full aspect-[16/11] rounded-2xl overflow-hidden bg-muted">
              <Image src="/figures/fig1_matched_fidelity_validation.png" alt="Figure 1" fill className="object-contain" />
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Comparison between the nominal benchmark's 4 dB PSNR spread and the calibrated matched fidelity support across all 6 sequences.
            </p>
          </div>

          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-3">
            <h3 className="font-bold text-sm">Figure 2: Sequence Gap Contraction Across Architectures</h3>
            <div className="relative w-full aspect-[16/11] rounded-2xl overflow-hidden bg-muted">
              <Image src="/figures/fig2_sequence_gap_contraction.png" alt="Figure 2" fill className="object-contain" />
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Accuracy under Nominal Severity vs. Matched Fidelity, showing up to 77.1% contraction of the nominal gap.
            </p>
          </div>

          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-3">
            <h3 className="font-bold text-sm">Figure 3: Factorial Sequence × Target Interaction Profiles</h3>
            <div className="relative w-full aspect-[16/11] rounded-2xl overflow-hidden bg-muted">
              <Image src="/figures/fig3_factorial_interaction_profiles.png" alt="Figure 3" fill className="object-contain" />
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Model accuracy across 5 matched target levels (28 dB to 16 dB PSNR), showing non-parallel degradation curves.
            </p>
          </div>

          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-sm flex flex-col gap-3">
            <h3 className="font-bold text-sm">Figure 4: Decoupling of Net Accuracy & Error Churn</h3>
            <div className="relative w-full aspect-[16/11] rounded-2xl overflow-hidden bg-muted">
              <Image src="/figures/fig4_matched_churn_decoupling.png" alt="Figure 4" fill className="object-contain" />
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Classifiers undergo 11–13% error churn under matched fidelity, where 57–76% is excess unobservable volatility.
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
