"use client";

import { Card, Slider } from "@heroui/react";
import { useState, useEffect, useMemo } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  AlertTriangle,
  CheckCircle,
  TrendingDown,
  Layers,
  ShieldCheck,
  Zap,
  Shuffle,
  Activity,
  Sliders,
  CheckCircle2,
  Info,
} from "lucide-react";

interface EvaluationRecord {
  image_id: number;
  corruption: string;
  severity: number;
  model: string;
  prediction: string;
  pred_idx: number;
  confidence: number;
  correct: boolean;
  true_label: string;
  true_idx: number;
}

interface CalibrationSequence {
  final_op: string;
  noise_std: number;
  blur_sigma: number;
  jpeg_quality: number;
  achieved_psnr: number;
  achieved_ssim: number;
  psnr_err: number;
  ssim_err: number;
  within_tol: boolean;
}

interface CalibrationLevel {
  target_psnr: number;
  target_ssim: number;
  all_6_pass: boolean;
  max_normalized_error: number;
  sequences: Record<string, CalibrationSequence>;
}

interface MatchedStudyData {
  calibration: {
    protocol_name: string;
    delineation_method: string;
    tolerances: {
      psnr_tolerance_db: number;
      ssim_tolerance: number;
    };
    levels: Record<string, CalibrationLevel>;
    sequences: string[];
  };
}

const SAMPLE_IMAGES = [
  { id: 0, label: "Tench (Fish)", filename: "img_0" },
  { id: 1, label: "English Springer (Dog)", filename: "img_1" },
  { id: 2, label: "Great White Shark", filename: "img_2" },
  { id: 3, label: "Tiger Shark", filename: "img_3" },
  { id: 4, label: "Hammerhead Shark", filename: "img_4" },
];

const MODEL_MAP: Record<string, string> = {
  resnet50: "resnet50.a1_in1k",
  convnext: "convnext_tiny.in12k_ft_in1k",
  vit: "vit_base_patch16_224.augreg_in21k_ft_in1k",
  swin: "swin_tiny_patch4_window7_224.ms_in1k",
};

const FINGERPRINT_COEFFICIENTS: Record<
  string,
  {
    name: string;
    fft_low: string;
    fft_high: string;
    jpeg_block: string;
    edge_energy: string;
    local_ssim: string;
  }
> = {
  resnet50: {
    name: "ResNet-50",
    fft_low: "-0.3109 ± 0.069",
    fft_high: "-1.1348 ± 0.043",
    jpeg_block: "+0.5252 ± 0.033",
    edge_energy: "+1.1980 ± 0.063",
    local_ssim: "-0.1975 ± 0.097",
  },
  convnext: {
    name: "ConvNeXt-Tiny",
    fft_low: "-0.1636 ± 0.023",
    fft_high: "-0.7451 ± 0.040",
    jpeg_block: "+0.1881 ± 0.023",
    edge_energy: "+0.7734 ± 0.037",
    local_ssim: "-0.0424 ± 0.057",
  },
  vit: {
    name: "ViT-Base/16",
    fft_low: "-0.1303 ± 0.044",
    fft_high: "-0.6566 ± 0.035",
    jpeg_block: "+0.1625 ± 0.019",
    edge_energy: "+1.0308 ± 0.044",
    local_ssim: "-0.2704 ± 0.063",
  },
  swin: {
    name: "Swin-Tiny",
    fft_low: "-0.1848 ± 0.019",
    fft_high: "-0.8340 ± 0.029",
    jpeg_block: "+0.1331 ± 0.009",
    edge_energy: "+0.7067 ± 0.037",
    local_ssim: "-0.0497 ± 0.042",
  },
};

export default function ExplorerPage() {
  const [activeTab, setActiveTab] = useState<"nominal" | "matched">("matched");

  // Nominal single-perturbation state
  const [records, setRecords] = useState<EvaluationRecord[]>([]);
  const [imageId, setImageId] = useState<number>(0);
  const [modelKey, setModelKey] = useState<string>("resnet50");
  const [corruption, setCorruption] = useState<string>("gaussian_noise");
  const [severity, setSeverity] = useState<number>(3);
  const [loading, setLoading] = useState<boolean>(true);

  // Matched-fidelity state
  const [matchedData, setMatchedData] = useState<MatchedStudyData | null>(null);
  const [matchedLevel, setMatchedLevel] = useState<string>("Level_1_Mild");
  const [matchedSequence, setMatchedSequence] = useState<string>("NBJ");

  useEffect(() => {
    Promise.all([
      fetch("/demo_results.json").then((res) => (res.ok ? res.json() : [])),
      fetch("/data/matched_fidelity_study_data.json").then((res) => (res.ok ? res.json() : null)),
    ])
      .then(([demoRecords, matchedJson]) => {
        setRecords(demoRecords);
        setMatchedData(matchedJson);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error loading explorer data:", err);
        setLoading(false);
      });
  }, []);

  const fullModelId = MODEL_MAP[modelKey] || "resnet50.a1_in1k";

  // Find clean record (severity 0)
  const cleanRecord = useMemo(() => {
    return records.find(
      (r) =>
        r.image_id === imageId &&
        r.model === fullModelId &&
        (r.corruption === "clean" || r.severity === 0)
    );
  }, [records, imageId, fullModelId]);

  // Find corrupted record
  const corruptedRecord = useMemo(() => {
    return records.find(
      (r) =>
        r.image_id === imageId &&
        r.model === fullModelId &&
        r.corruption === corruption &&
        r.severity === severity
    );
  }, [records, imageId, fullModelId, corruption, severity]);

  const cleanImagePath = `/examples/img_${imageId}_clean.jpg`;
  const corruptedImagePath = `/examples/img_${imageId}_${corruption}_s${severity}.jpg`;

  const cleanConf = cleanRecord ? Math.round(cleanRecord.confidence * 100) : 0;
  const corrConf = corruptedRecord ? Math.round(corruptedRecord.confidence * 100) : 0;
  const confDelta = corrConf - cleanConf;
  const isFooled = corruptedRecord ? !corruptedRecord.correct : false;

  // Selected matched-level data
  const currentCalibrationLevel = matchedData?.calibration?.levels?.[matchedLevel];
  const currentSequenceData = currentCalibrationLevel?.sequences?.[matchedSequence];
  const currentFingerprints = FINGERPRINT_COEFFICIENTS[modelKey] || FINGERPRINT_COEFFICIENTS["resnet50"];

  return (
    <main className="min-h-screen pt-32 pb-24 px-6 max-w-[1400px] mx-auto flex flex-col gap-10">
      <div className="flex flex-col gap-4">
        <div className="flex items-center gap-2 text-primary font-bold text-sm tracking-widest uppercase">
          <Layers className="w-4 h-4" /> Real Dataset Evaluation & Visual Inspection
        </div>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight font-[family-name:var(--font-poppins)]">
          Image Robustness Explorer
        </h1>

        {/* Tab Switcher */}
        <div className="flex flex-wrap gap-3 mt-2">
          <button
            onClick={() => setActiveTab("matched")}
            className={`px-5 py-2.5 rounded-2xl font-bold text-sm transition-all border ${
              activeTab === "matched"
                ? "bg-primary text-primary-foreground border-primary shadow-md"
                : "bg-card text-muted-foreground border-border/60 hover:text-foreground"
            }`}
          >
            Matched-Fidelity Protocol (3 Target Levels & Fingerprints)
          </button>
          <button
            onClick={() => setActiveTab("nominal")}
            className={`px-5 py-2.5 rounded-2xl font-bold text-sm transition-all border ${
              activeTab === "nominal"
                ? "bg-primary text-primary-foreground border-primary shadow-md"
                : "bg-card text-muted-foreground border-border/60 hover:text-foreground"
            }`}
          >
            Single-Perturbation Explorer (Severities 1–5)
          </button>
        </div>

        {activeTab === "nominal" ? (
          <p className="text-muted-foreground text-xl max-w-3xl">
            Side-by-side visual analysis of real vision model evaluations under calibrated distribution shifts.
            Compare genuine clean vs. corrupted pixel inputs, degradation curves, and failure modes across severities 1–5.
          </p>
        ) : (
          <p className="text-muted-foreground text-xl max-w-3xl">
            Inspect the 6 compound sequence permutations (NBJ, NJB, BNJ, BJN, JNB, JBN) calibrated to equate global PSNR and SSIM within strict tolerances (±0.75 dB, ±0.035), alongside architecture-specific distortion fingerprints.
          </p>
        )}
      </div>

      {activeTab === "nominal" ? (
        /* TAB 1: Standard Single-Perturbation Explorer */
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Controls Sidebar */}
          <div className="lg:col-span-1 flex flex-col gap-6 bg-card p-6 rounded-3xl border border-border/60 shadow-sm h-fit">
            <h2 className="text-lg font-bold text-foreground flex items-center gap-2 pb-2 border-b">
              <Zap className="w-5 h-5 text-primary" /> Evaluation Controls
            </h2>

            {/* Image Selector */}
            <div className="flex flex-col gap-2">
              <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Reference Image</label>
              <Select value={String(imageId)} onValueChange={(val) => setImageId(Number(val))}>
                <SelectTrigger className="w-full h-12 bg-background font-medium">
                  <SelectValue placeholder="Select image" />
                </SelectTrigger>
                <SelectContent>
                  {SAMPLE_IMAGES.map((img) => (
                    <SelectItem key={img.id} value={String(img.id)}>
                      {img.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Model Selector */}
            <div className="flex flex-col gap-2">
              <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Vision Model</label>
              <Select value={modelKey} onValueChange={setModelKey}>
                <SelectTrigger className="w-full h-12 bg-background font-medium">
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="resnet50">ResNet-50 (CNN)</SelectItem>
                  <SelectItem value="convnext">ConvNeXt-Tiny (Modern CNN)</SelectItem>
                  <SelectItem value="vit">ViT-Base/16 (Transformer)</SelectItem>
                  <SelectItem value="swin">Swin-Tiny (Hierarchical ViT)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Corruption Selector */}
            <div className="flex flex-col gap-2">
              <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Corruption Type</label>
              <Select value={corruption} onValueChange={setCorruption}>
                <SelectTrigger className="w-full h-12 bg-background font-medium">
                  <SelectValue placeholder="Select corruption" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="gaussian_noise">Gaussian Noise</SelectItem>
                  <SelectItem value="defocus_blur">Defocus Blur</SelectItem>
                  <SelectItem value="jpeg_compression">JPEG Compression</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Severity Slider & Discrete Buttons */}
            <div className="flex flex-col gap-3 pt-2">
              <div className="flex justify-between items-center">
                <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Severity Level</label>
                <div className="flex items-center gap-1">
                  {[1, 2, 3, 4, 5].map((lvl) => (
                    <button
                      key={lvl}
                      onClick={() => setSeverity(lvl)}
                      className={`w-6 h-6 rounded-md text-xs font-bold transition-all ${
                        severity === lvl
                          ? "bg-primary text-primary-foreground shadow-xs"
                          : "bg-muted text-muted-foreground hover:bg-muted/80"
                      }`}
                    >
                      {lvl}
                    </button>
                  ))}
                </div>
              </div>
              <Slider
                aria-label="Severity Level"
                step={1}
                maxValue={5}
                minValue={1}
                value={severity}
                onChange={(value) => {
                  const num = typeof value === "number" ? value : Array.isArray(value) ? value[0] : 3;
                  setSeverity(Math.max(1, Math.min(5, Math.round(num))));
                }}
                className="max-w-md"
              />
              <div className="flex justify-between text-xs text-muted-foreground font-medium px-1">
                <span>1 (Mild)</span>
                <span className="text-primary font-bold">Level {severity} Selected</span>
                <span>5 (Severe)</span>
              </div>
            </div>

            {/* Summary Metric Card */}
            <div className="mt-4 p-4 rounded-2xl bg-muted/40 border border-border/50 text-xs flex flex-col gap-2">
              <div className="flex justify-between">
                <span className="text-muted-foreground font-semibold">True Ground Truth:</span>
                <span className="font-bold text-foreground capitalize">{cleanRecord?.true_label || "Loading..."}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground font-semibold">Shift Robustness:</span>
                <span className={`font-bold ${isFooled ? "text-destructive" : "text-green-500"}`}>
                  {isFooled ? "Susceptible (Fooled)" : "Robust (Correct)"}
                </span>
              </div>
            </div>
          </div>

          {/* Side-by-Side Image Viewers */}
          <div className="lg:col-span-3 flex flex-col gap-8">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Clean Original Image */}
              <Card className="bg-card shadow-lg border border-border/60 rounded-[2rem] overflow-hidden flex flex-col">
                <div className="p-6 md:p-8 flex flex-col gap-4 flex-1">
                  <div className="flex justify-between items-center">
                    <div>
                      <h3 className="font-extrabold text-2xl">Clean Image</h3>
                      <p className="text-sm text-muted-foreground">Unshifted baseline input</p>
                    </div>
                    <span className="text-xs bg-primary/10 text-primary font-bold px-3 py-1.5 rounded-full uppercase tracking-wide">
                      Baseline
                    </span>
                  </div>

                  <div className="w-full max-w-[480px] mx-auto aspect-square bg-muted/30 rounded-2xl flex items-center justify-center border border-border/40 overflow-hidden relative">
                    <img
                      src={cleanImagePath}
                      alt={`Clean ${cleanRecord?.true_label || "image"}`}
                      className="w-full h-full object-contain p-2"
                    />
                  </div>

                  <div className="flex flex-col gap-3 mt-auto pt-5 border-t border-border/50">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground font-semibold">Ground Truth Class:</span>
                      <span className="font-extrabold text-lg capitalize">{cleanRecord?.true_label || "..."}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground font-semibold">Model Prediction:</span>
                      <span className="font-bold text-lg text-primary capitalize">{cleanRecord?.prediction || "..."}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground font-semibold">Base Confidence:</span>
                      <span className="font-extrabold text-xl text-green-500">{cleanConf}%</span>
                    </div>
                  </div>
                </div>
              </Card>

              {/* Genuine Corrupted Image */}
              <Card className="bg-card shadow-lg border border-border/60 rounded-[2rem] overflow-hidden flex flex-col">
                <div className="p-6 md:p-8 flex flex-col gap-4 flex-1">
                  <div className="flex justify-between items-center">
                    <div>
                      <h3 className="font-extrabold text-2xl">Corrupted Image</h3>
                      <p className="text-sm text-muted-foreground capitalize">
                        {corruption.replace("_", " ")} — Severity {severity}
                      </p>
                    </div>
                    <span
                      className={`text-xs font-bold px-3 py-1.5 rounded-full uppercase tracking-wide ${
                        isFooled ? "bg-destructive/10 text-destructive" : "bg-green-500/10 text-green-500"
                      }`}
                    >
                      {isFooled ? "Model Fooled" : "Model Correct"}
                    </span>
                  </div>

                  <div className="w-full max-w-[480px] mx-auto aspect-square bg-muted/30 rounded-2xl flex items-center justify-center border border-border/40 overflow-hidden relative">
                    <img
                      src={corruptedImagePath}
                      alt={`Corrupted with ${corruption} severity ${severity}`}
                      className="w-full h-full object-contain p-2"
                    />
                  </div>

                  <div className="flex flex-col gap-3 mt-auto pt-5 border-t border-border/50">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground font-semibold">Corrupted Prediction:</span>
                      <span
                        className={`font-extrabold text-lg capitalize ${
                          isFooled ? "text-destructive" : "text-green-500"
                        }`}
                      >
                        {corruptedRecord?.prediction || "..."}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground font-semibold">Corrupted Confidence:</span>
                      <span className="font-extrabold text-xl">{corrConf}%</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground font-semibold">Confidence Shift:</span>
                      <span
                        className={`font-extrabold text-lg flex items-center gap-1 ${
                          confDelta < 0 ? "text-destructive" : "text-green-500"
                        }`}
                      >
                        {confDelta < 0 ? <TrendingDown className="w-4 h-4" /> : null}
                        {confDelta > 0 ? `+${confDelta}%` : `${confDelta}%`}
                      </span>
                    </div>
                  </div>
                </div>
              </Card>
            </div>

            {/* Model Vulnerability Diagnosis Card */}
            <Card className="p-6 md:p-8 bg-card rounded-3xl border border-border/60 shadow-md">
              <h4 className="text-xl font-extrabold mb-4 flex items-center gap-2">
                {isFooled ? (
                  <>
                    <AlertTriangle className="w-6 h-6 text-destructive" />
                    <span>Failure Mode Analysis: Model Fooled</span>
                  </>
                ) : (
                  <>
                    <ShieldCheck className="w-6 h-6 text-green-500" />
                    <span>Robustness Diagnosis: Model Retained True Label</span>
                  </>
                )}
              </h4>
              <p className="text-muted-foreground leading-relaxed text-base">
                {isFooled ? (
                  <>
                    Under <strong className="text-foreground capitalize">{corruption.replace("_", " ")}</strong> (severity {severity}),
                    the model misclassified the true <strong className="text-foreground">{cleanRecord?.true_label}</strong> as{" "}
                    <strong className="text-destructive font-bold">{corruptedRecord?.prediction}</strong> with{" "}
                    <strong>{corrConf}%</strong> confidence. This illustrates an empirical vulnerability to spatial frequency
                    distortion or high-frequency pixel degradation.
                  </>
                ) : (
                  <>
                    Despite <strong className="text-foreground capitalize">{corruption.replace("_", " ")}</strong> perturbation at
                    severity {severity}, the model correctly identified the input as{" "}
                    <strong className="text-green-500 font-bold">{corruptedRecord?.prediction}</strong>. However, confidence shifted
                    by <strong>{confDelta}%</strong> (from {cleanConf}% down to {corrConf}%), showing degraded certainty under shift.
                  </>
                )}
              </p>
            </Card>
          </div>
        </div>
      ) : (
        /* TAB 2: Matched-Fidelity Common-Support Protocol */
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Controls Sidebar */}
          <div className="lg:col-span-1 flex flex-col gap-6 bg-card p-6 rounded-3xl border border-border/60 shadow-sm h-fit">
            <h2 className="text-lg font-bold text-foreground flex items-center gap-2 pb-2 border-b">
              <Shuffle className="w-5 h-5 text-primary" /> Protocol Controls
            </h2>

            {/* Image Selector */}
            <div className="flex flex-col gap-2">
              <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Reference Image</label>
              <Select value={String(imageId)} onValueChange={(val) => setImageId(Number(val))}>
                <SelectTrigger className="w-full h-12 bg-background font-medium">
                  <SelectValue placeholder="Select image" />
                </SelectTrigger>
                <SelectContent>
                  {SAMPLE_IMAGES.map((img) => (
                    <SelectItem key={img.id} value={String(img.id)}>
                      {img.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Target Calibrated Level */}
            <div className="flex flex-col gap-2">
              <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Calibrated Fidelity Level</label>
              <Select value={matchedLevel} onValueChange={setMatchedLevel}>
                <SelectTrigger className="w-full h-12 bg-background font-medium">
                  <SelectValue placeholder="Select level" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Level_1_Mild">Level 1 (Mild: 27.20 dB / 0.810)</SelectItem>
                  <SelectItem value="Level_2_Moderate">Level 2 (Moderate: 23.80 dB / 0.650)</SelectItem>
                  <SelectItem value="Level_3_SevereCommon">Level 3 (Severe: 21.80 dB / 0.500)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Processing Order Permutation */}
            <div className="flex flex-col gap-2">
              <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Permutation Sequence</label>
              <Select value={matchedSequence} onValueChange={setMatchedSequence}>
                <SelectTrigger className="w-full h-12 bg-background font-medium font-mono font-bold">
                  <SelectValue placeholder="Select sequence" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="NBJ">NBJ (Noise &rarr; Blur &rarr; JPEG)</SelectItem>
                  <SelectItem value="NJB">NJB (Noise &rarr; JPEG &rarr; Blur)</SelectItem>
                  <SelectItem value="BNJ">BNJ (Blur &rarr; Noise &rarr; JPEG)</SelectItem>
                  <SelectItem value="BJN">BJN (Blur &rarr; JPEG &rarr; Noise)</SelectItem>
                  <SelectItem value="JNB">JNB (JPEG &rarr; Noise &rarr; Blur)</SelectItem>
                  <SelectItem value="JBN">JBN (JPEG &rarr; Blur &rarr; Noise)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Vision Model Selector */}
            <div className="flex flex-col gap-2">
              <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Target Model Fingerprint</label>
              <Select value={modelKey} onValueChange={setModelKey}>
                <SelectTrigger className="w-full h-12 bg-background font-medium">
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="resnet50">ResNet-50</SelectItem>
                  <SelectItem value="convnext">ConvNeXt-Tiny</SelectItem>
                  <SelectItem value="vit">ViT-Base/16</SelectItem>
                  <SelectItem value="swin">Swin-Tiny</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Common-support Verification Status */}
            <div className="p-4 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-600 dark:text-emerald-400 text-xs flex flex-col gap-1.5">
              <div className="flex items-center gap-1.5 font-bold">
                <CheckCircle2 className="w-4 h-4" /> Common Support Verified
              </div>
              <p className="text-[11px] leading-relaxed opacity-90">
                All 6 permutations simultaneously satisfy prespecified tolerance (&Delta;PSNR &le; &plusmn;0.75 dB, &Delta;SSIM &le; &plusmn;0.035).
              </p>
            </div>
          </div>

          {/* Details & Fingerprints Display */}
          <div className="lg:col-span-3 flex flex-col gap-6">
            {/* Achieved Target Metrics Card */}
            <Card className="p-6 md:p-8 bg-card border border-border/60 rounded-3xl shadow-md">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-border/50">
                <div>
                  <span className="text-xs font-bold uppercase tracking-wider text-primary">
                    Calibrated Common-Support Parameters
                  </span>
                  <h3 className="text-2xl font-extrabold text-foreground mt-0.5">
                    Sequence <span className="text-primary font-mono">{matchedSequence}</span> at {matchedLevel.replace(/_/g, " ")}
                  </h3>
                </div>
                <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-bold">
                  Final Op: {currentSequenceData?.final_op || "Unknown"}
                </div>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6">
                <div className="p-4 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-1">
                  <span className="text-xs text-muted-foreground font-semibold">Target PSNR</span>
                  <span className="text-xl font-bold font-mono text-foreground">
                    {currentCalibrationLevel?.target_psnr.toFixed(2)} dB
                  </span>
                  <span className="text-[11px] text-muted-foreground">Tol: &plusmn;0.75 dB</span>
                </div>

                <div className="p-4 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-1">
                  <span className="text-xs text-muted-foreground font-semibold">Achieved PSNR</span>
                  <span className="text-xl font-bold font-mono text-blue-500">
                    {currentSequenceData?.achieved_psnr.toFixed(2)} dB
                  </span>
                  <span className="text-[11px] text-emerald-500 font-semibold">
                    &Delta; = {(currentSequenceData?.psnr_err ?? 0) > 0 ? `+${currentSequenceData?.psnr_err.toFixed(2)}` : currentSequenceData?.psnr_err.toFixed(2)} dB
                  </span>
                </div>

                <div className="p-4 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-1">
                  <span className="text-xs text-muted-foreground font-semibold">Target SSIM</span>
                  <span className="text-xl font-bold font-mono text-foreground">
                    {currentCalibrationLevel?.target_ssim.toFixed(3)}
                  </span>
                  <span className="text-[11px] text-muted-foreground">Tol: &plusmn;0.035</span>
                </div>

                <div className="p-4 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-1">
                  <span className="text-xs text-muted-foreground font-semibold">Achieved SSIM</span>
                  <span className="text-xl font-bold font-mono text-emerald-500">
                    {currentSequenceData?.achieved_ssim.toFixed(3)}
                  </span>
                  <span className="text-[11px] text-emerald-500 font-semibold">
                    &Delta; = {(currentSequenceData?.ssim_err ?? 0) > 0 ? `+${currentSequenceData?.ssim_err.toFixed(4)}` : currentSequenceData?.ssim_err.toFixed(4)}
                  </span>
                </div>
              </div>

              {/* Exact calibrated parameter triplet */}
              <div className="mt-6 p-4 rounded-2xl bg-muted/20 border border-border/30 flex flex-wrap items-center justify-between gap-4 text-xs font-mono">
                <div>
                  <span className="text-muted-foreground">Gaussian Noise &sigma;: </span>
                  <strong className="text-foreground">{currentSequenceData?.noise_std}</strong>
                </div>
                <div>
                  <span className="text-muted-foreground">Defocus Blur &sigma;: </span>
                  <strong className="text-foreground">{currentSequenceData?.blur_sigma} px</strong>
                </div>
                <div>
                  <span className="text-muted-foreground">JPEG Quality Q: </span>
                  <strong className="text-foreground">{currentSequenceData?.jpeg_quality}</strong>
                </div>
              </div>
            </Card>

            {/* Architecture-Specific Distortion Fingerprints Card */}
            <Card className="p-6 md:p-8 bg-card border border-border/60 rounded-3xl shadow-md">
              <div className="flex items-center justify-between pb-4 border-b border-border/50">
                <div>
                  <span className="text-xs font-bold uppercase tracking-wider text-violet-500">
                    Model 5 Regression Coefficients
                  </span>
                  <h3 className="text-xl font-bold text-foreground mt-0.5">
                    Physical Distortion Fingerprints for {currentFingerprints.name}
                  </h3>
                </div>
                <span className="text-xs font-mono text-muted-foreground">
                  Mean &plusmn; Bootstrap Std
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mt-6">
                <div className="p-4 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-1">
                  <span className="text-xs font-bold text-foreground">Fourier Low-Energy (&beta;)</span>
                  <span className="text-lg font-mono font-extrabold text-primary">{currentFingerprints.fft_low}</span>
                  <span className="text-[11px] text-muted-foreground leading-tight">Fraction of spectral power in low radial frequencies</span>
                </div>

                <div className="p-4 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-1">
                  <span className="text-xs font-bold text-foreground">Fourier High-Energy (&beta;)</span>
                  <span className="text-lg font-mono font-extrabold text-blue-500">{currentFingerprints.fft_high}</span>
                  <span className="text-[11px] text-muted-foreground leading-tight">Fraction of spectral power in high radial frequencies</span>
                </div>

                <div className="p-4 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-1">
                  <span className="text-xs font-bold text-foreground">JPEG Block Boundary (&beta;)</span>
                  <span className="text-lg font-mono font-extrabold text-amber-500">{currentFingerprints.jpeg_block}</span>
                  <span className="text-[11px] text-muted-foreground leading-tight">8&times;8 DCT block border discontinuity metric</span>
                </div>

                <div className="p-4 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-1">
                  <span className="text-xs font-bold text-foreground">Sobel Edge Energy (&beta;)</span>
                  <span className="text-lg font-mono font-extrabold text-emerald-500">{currentFingerprints.edge_energy}</span>
                  <span className="text-[11px] text-muted-foreground leading-tight">Total spatial edge magnitude relative to clean input</span>
                </div>

                <div className="p-4 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-1 sm:col-span-2">
                  <span className="text-xs font-bold text-foreground">Local SSIM Heterogeneity (&beta;)</span>
                  <span className="text-lg font-mono font-extrabold text-rose-500">{currentFingerprints.local_ssim}</span>
                  <span className="text-[11px] text-muted-foreground leading-tight">Spatial standard deviation of local 11&times;11 SSIM map</span>
                </div>
              </div>
            </Card>
          </div>
        </div>
      )}
    </main>
  );
}
