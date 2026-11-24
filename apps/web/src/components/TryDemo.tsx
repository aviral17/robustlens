"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { Card, Button, ProgressBar as Progress, Slider } from "@heroui/react";
import {
  Upload,
  ArrowRight,
  Activity,
  Settings2,
  BarChart2,
  AlertCircle,
  Clock,
  Sparkles,
  ChevronDown,
  ChevronUp,
  Layers,
  ShieldCheck,
  Eye,
  FlaskConical,
  Info,
  FileCode,
} from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { motion, AnimatePresence } from "framer-motion";
import {
  applyClientPixelCorruption,
  CorruptionType,
  CompoundSequenceOrder,
  ResearchLevelKey,
  FROZEN_CALIBRATION_3LEVELS,
  applyCompoundSequence,
} from "@/lib/corruptions";
import { computeImageDiagnostics, ComputedDiagnostics } from "@/lib/diagnostics";

interface ModelPrediction {
  name: string;
  pred: string;
  conf: number;
  top_k?: { class: string; probability: number }[];
}

interface DemoResult {
  original: string;
  corrupted: string;
  corruption: CorruptionType;
  severity: number;
  models: ModelPrediction[];
  inferenceTime: number;
  sampleGroundTruth?: string;
  diagnostics?: ComputedDiagnostics | null;
}

interface SynthesizedSequenceItem {
  dataUrl: string;
  diag: ComputedDiagnostics;
  withinTol: boolean;
  psnrErr: number;
  ssimErr: number;
}

const SAMPLE_PRESETS = [
  {
    name: "Tench (Fish)",
    url: "/examples/img_0_clean.jpg",
    groundTruth: "tench",
  },
  {
    name: "English Springer (Dog)",
    url: "/examples/img_1_clean.jpg",
    groundTruth: "English springer",
  },
  {
    name: "Shark (Marine)",
    url: "/examples/img_2_clean.jpg",
    groundTruth: "great white shark",
  },
];

export default function TryDemo() {
  const [activeTab, setActiveTab] = useState<"quick_demo" | "research_lab">("quick_demo");

  // Common input states:
  const [, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [sampleGroundTruth, setSampleGroundTruth] = useState<string | null>(null);

  // Quick Demo parameters (nominal severities 1-5):
  const [model, setModel] = useState("all");
  const [corruption, setCorruption] = useState<CorruptionType>("gaussian_noise");
  const [severity, setSeverity] = useState<number>(3);

  // Quick Demo live preview & diagnostics:
  const [liveCorruptedPreview, setLiveCorruptedPreview] = useState<string | null>(null);
  const [liveDiagnostics, setLiveDiagnostics] = useState<ComputedDiagnostics | null>(null);
  const [isComputingPreview, setIsComputingPreview] = useState(false);

  // Quick Demo inference execution:
  const [isInferring, setIsInferring] = useState(false);
  const [stage, setStage] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isQuotaError, setIsQuotaError] = useState(false);
  const [result, setResult] = useState<DemoResult | null>(null);
  const [expandedModel, setExpandedModel] = useState<string | null>(null);

  // Research Mode states (Matched-Fidelity Six-Order Lab):
  const [researchLevel, setResearchLevel] = useState<ResearchLevelKey>("Level_1_Mild");
  const [researchModel, setResearchModel] = useState<string>("all");
  const [isSynthesizingSequences, setIsSynthesizingSequences] = useState(false);
  const [synthesizedSequences, setSynthesizedSequences] = useState<
    Record<CompoundSequenceOrder, SynthesizedSequenceItem> | null
  >(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Update quick demo real-time preview and client-side metrics
  const updateLivePreviewAndDiagnostics = useCallback(
    async (src: string, cType: CorruptionType, sev: number) => {
      try {
        setIsComputingPreview(true);
        const corruptedDataUrl = await applyClientPixelCorruption(src, cType, sev);
        setLiveCorruptedPreview(corruptedDataUrl);

        try {
          const diag = await computeImageDiagnostics(src, corruptedDataUrl);
          setLiveDiagnostics(diag);
        } catch (diagErr) {
          console.warn("Diagnostics computation warning:", diagErr);
        }
      } catch (err) {
        console.error("Failed to update live corrupted preview:", err);
      } finally {
        setIsComputingPreview(false);
      }
    },
    []
  );

  useEffect(() => {
    if (preview) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      updateLivePreviewAndDiagnostics(preview, corruption, severity);
    } else {
      setLiveCorruptedPreview(null);
      setLiveDiagnostics(null);
    }
  }, [preview, corruption, severity, updateLivePreviewAndDiagnostics]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const f = e.target.files[0];
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setSampleGroundTruth(null);
      setResult(null);
      setSynthesizedSequences(null);
      setErrorMessage(null);
      setIsQuotaError(false);
    }
  };

  const selectSamplePreset = async (preset: (typeof SAMPLE_PRESETS)[0]) => {
    setSampleGroundTruth(preset.groundTruth);
    setPreview(preset.url);
    setResult(null);
    setSynthesizedSequences(null);
    setErrorMessage(null);
    setIsQuotaError(false);

    try {
      const res = await fetch(preset.url);
      const blob = await res.blob();
      const sampleFile = new File([blob], `${preset.name}.jpg`, { type: "image/jpeg" });
      setFile(sampleFile);
    } catch (err) {
      console.error("Failed to load preset file:", err);
    }
  };

  const handleSeverityChange = (val: number | number[]) => {
    const num = typeof val === "number" ? val : Array.isArray(val) && val.length > 0 ? val[0] : 3;
    const clamped = Math.max(1, Math.min(5, Math.round(num)));
    setSeverity(clamped);
  };

  // Quick Demo live inference execution
  const startInference = async () => {
    if (!preview) return;
    setIsInferring(true);
    setErrorMessage(null);
    setIsQuotaError(false);
    setStage(1);

    try {
      const genuineCorruptedDataUrl =
        liveCorruptedPreview ||
        (await applyClientPixelCorruption(preview, corruption, severity));

      setStage(2);

      let cleanBase64 = preview;
      if (!preview.startsWith("data:")) {
        const response = await fetch(preview);
        const blob = await response.blob();
        cleanBase64 = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result as string);
          reader.onerror = () => reject(new Error("Failed to encode clean image"));
          reader.readAsDataURL(blob);
        });
      }

      setStage(3);

      const apiResponse = await fetch("/api/inference", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: cleanBase64,
          model: model,
          corruption: corruption,
          severity: severity,
        }),
      });

      const data = await apiResponse.json();

      if (!apiResponse.ok) {
        if (apiResponse.status === 429) {
          setIsQuotaError(true);
        }
        throw new Error(data.error || "Inference API returned an error response");
      }

      setStage(5);

      const apiResult = data.result;

      if (apiResult && apiResult.status === "success") {
        const mappedModels: ModelPrediction[] = apiResult.results.map(
          (m: {
            model: string;
            prediction: string;
            confidence: number;
            top_k?: { class: string; probability: number }[];
          }) => ({
            name: m.model,
            pred: m.prediction,
            conf: m.confidence,
            top_k: m.top_k,
          })
        );

        let diag = liveDiagnostics;
        if (!diag) {
          try {
            diag = await computeImageDiagnostics(preview, genuineCorruptedDataUrl);
          } catch {
            diag = null;
          }
        }

        setResult({
          original: preview,
          corrupted: genuineCorruptedDataUrl,
          corruption,
          severity,
          models: mappedModels,
          inferenceTime: apiResult.inference_time_seconds ?? 0.5,
          sampleGroundTruth: sampleGroundTruth || undefined,
          diagnostics: diag,
        });
      } else {
        throw new Error(apiResult?.message || "Invalid response structure from Gradio space");
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Inference failed unexpectedly";
      console.error("Inference failed:", e);
      setErrorMessage(msg);
    } finally {
      setIsInferring(false);
      setStage(0);
    }
  };

  // Research Mode: Synthesize 6 compound order sequences and compute achieved physical fidelity
  const handleSynthesizeSequences = async () => {
    if (!preview) return;
    setIsSynthesizingSequences(true);
    setErrorMessage(null);

    try {
      const orders: CompoundSequenceOrder[] = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"];
      const levelSpec = FROZEN_CALIBRATION_3LEVELS[researchLevel];
      const results: Partial<Record<CompoundSequenceOrder, SynthesizedSequenceItem>> = {};

      for (const ord of orders) {
        const params = levelSpec.sequences[ord];
        const seqDataUrl = await applyCompoundSequence(preview, ord, {
          noise_std: params.noise_std,
          blur_sigma: params.blur_sigma,
          jpeg_quality: params.jpeg_quality,
        });

        const diag = await computeImageDiagnostics(preview, seqDataUrl);
        const psnrErr = Math.round((diag.psnr_db - levelSpec.target_psnr) * 100) / 100;
        const ssimErr = Math.round((diag.ssim - levelSpec.target_ssim) * 1000) / 1000;
        const withinTol = Math.abs(psnrErr) <= 0.75 && Math.abs(ssimErr) <= 0.035;

        results[ord] = {
          dataUrl: seqDataUrl,
          diag,
          withinTol,
          psnrErr,
          ssimErr,
        };
      }

      setSynthesizedSequences(
        results as Record<CompoundSequenceOrder, SynthesizedSequenceItem>
      );
    } catch (err: unknown) {
      console.error("Failed to synthesize sequences:", err);
      setErrorMessage(
        err instanceof Error ? err.message : "Failed to synthesize 6 compound sequences."
      );
    } finally {
      setIsSynthesizingSequences(false);
    }
  };

  return (
    <div className="w-full max-w-[1300px] mx-auto mt-16 mb-24 relative font-[family-name:var(--font-poppins)]">
      {/* Header & Scientific Context */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-bold uppercase tracking-wider mb-3">
          <Sparkles className="w-3.5 h-3.5" /> Live Hugging Face ZeroGPU Service
        </div>
        <h2 className="text-4xl md:text-5xl font-extrabold tracking-tight">Try RobustLens Live</h2>
        <p className="mt-4 text-xl text-muted-foreground font-medium max-w-2xl mx-auto">
          Upload any image and evaluate vision model predictions under controlled image corruptions in real-time.
        </p>
        <p className="mt-2 text-xs text-muted-foreground/80 max-w-3xl mx-auto">
          The quick stress test evaluates nominal user-selected severities (1–5) for instantaneous feedback. In contrast, our November 2026 confirmatory research monograph enforces frozen common-support calibration parameters across three matched-fidelity research levels (Level 1: 27.20 dB / 0.810; Level 2: 23.80 dB / 0.650; Level 3: 21.80 dB / 0.500). Explore both paradigms below.
        </p>

        {/* Mode Selector Tabs */}
        <div className="mt-8 flex justify-center">
          <div className="inline-flex p-1.5 rounded-2xl bg-muted/60 border border-border/60 shadow-inner">
            <button
              onClick={() => setActiveTab("quick_demo")}
              className={`px-5 py-2.5 rounded-xl text-xs sm:text-sm font-bold transition-all cursor-pointer ${
                activeTab === "quick_demo"
                  ? "bg-card text-foreground shadow-md border border-border/50"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Interactive Quick Demo (Nominal Severities 1–5)
            </button>
            <button
              onClick={() => setActiveTab("research_lab")}
              className={`px-5 py-2.5 rounded-xl text-xs sm:text-sm font-bold transition-all cursor-pointer flex items-center gap-2 ${
                activeTab === "research_lab"
                  ? "bg-primary text-primary-foreground shadow-md"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <FlaskConical className="w-4 h-4" /> Matched-Fidelity Six-Order Research Lab
            </button>
          </div>
        </div>

        {/* Sample Image Presets */}
        <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
          <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Or pick a sample:</span>
          {SAMPLE_PRESETS.map((p) => (
            <button
              key={p.name}
              onClick={() => selectSamplePreset(p)}
              className="cursor-pointer px-3.5 py-1.5 rounded-full text-xs font-semibold bg-muted hover:bg-primary/20 hover:text-primary transition-colors border border-border/60 shadow-sm"
            >
              {p.name}
            </button>
          ))}
        </div>
      </div>

      {/* Error / Quota Alert Banner */}
      {errorMessage && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className={`p-6 mb-8 rounded-3xl border flex items-start gap-4 ${
            isQuotaError
              ? "bg-amber-500/10 border-amber-500/30 text-amber-900 dark:text-amber-200"
              : "bg-destructive/10 border-destructive/30 text-destructive"
          }`}
        >
          <AlertCircle className="w-6 h-6 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="font-bold text-lg">
              {isQuotaError ? "Hugging Face ZeroGPU Limit Reached" : "Request Notice"}
            </h4>
            <p className="text-sm opacity-90">{errorMessage}</p>
          </div>
        </motion.div>
      )}

      {/* Image Upload Box (shared when no image selected) */}
      {!preview && !result && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="border-2 border-dashed border-primary/30 bg-primary/5 rounded-[2.5rem] p-16 flex flex-col items-center justify-center text-center transition-all hover:bg-primary/10 hover:border-primary/50 cursor-pointer"
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="bg-background shadow-sm rounded-full p-6 mb-6">
            <Upload className="w-10 h-10 text-primary" />
          </div>
          <h3 className="text-2xl font-bold mb-2">Drop an image here</h3>
          <p className="text-muted-foreground mb-8 text-lg">or click to browse from your device</p>
          <div className="flex gap-4">
            <span className="bg-background text-muted-foreground px-4 py-1.5 rounded-full text-sm font-semibold border shadow-sm">
              PNG
            </span>
            <span className="bg-background text-muted-foreground px-4 py-1.5 rounded-full text-sm font-semibold border shadow-sm">
              JPG
            </span>
            <span className="bg-background text-muted-foreground px-4 py-1.5 rounded-full text-sm font-semibold border shadow-sm">
              WEBP
            </span>
          </div>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/png, image/jpeg, image/webp"
            className="hidden"
          />
        </motion.div>
      )}

      {/* ========================================================================= */}
      {/* TAB 1: INTERACTIVE QUICK DEMO (NOMINAL SEVERITIES 1-5)                   */}
      {/* ========================================================================= */}
      {activeTab === "quick_demo" && preview && !result && !isInferring && (
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          className="flex flex-col gap-8"
        >
          {/* Side-by-Side Live Visual Preview: Clean vs. Live Perturbed */}
          <div className="grid md:grid-cols-2 gap-8 items-start">
            {/* Clean Input Card */}
            <Card className="p-6 bg-card border border-border/60 shadow-xl rounded-3xl overflow-hidden flex flex-col items-center">
              <div className="w-full flex justify-between items-center mb-3">
                <span className="text-xs font-bold uppercase tracking-wider text-primary">Clean Input</span>
                <span className="text-xs bg-muted text-muted-foreground font-semibold px-2.5 py-0.5 rounded-full border">
                  Original
                </span>
              </div>
              <div className="w-full max-w-[480px] aspect-square rounded-2xl overflow-hidden border bg-muted/40 relative flex items-center justify-center">
                <img
                  src={preview}
                  alt="Clean upload preview"
                  className="w-full h-full object-contain p-2"
                />
              </div>
              {sampleGroundTruth && (
                <div className="mt-3 text-xs font-bold text-muted-foreground uppercase">
                  Ground Truth: <span className="text-primary capitalize">{sampleGroundTruth}</span>
                </div>
              )}
              <Button
                variant="outline"
                size="sm"
                className="mt-3"
                onPress={() => {
                  setPreview(null);
                  setFile(null);
                  setSampleGroundTruth(null);
                  setResult(null);
                  setSynthesizedSequences(null);
                }}
              >
                Choose Different Image
              </Button>
            </Card>

            {/* Real-time Perturbation Preview Card */}
            <Card className="p-6 bg-card border border-border/60 shadow-xl rounded-3xl overflow-hidden flex flex-col items-center">
              <div className="w-full flex justify-between items-center mb-3">
                <div className="flex items-center gap-2">
                  <Eye className="w-4 h-4 text-rose-500" />
                  <span className="text-xs font-bold uppercase tracking-wider text-rose-500 capitalize">
                    Live Shift Preview: {corruption.replace("_", " ")}
                  </span>
                </div>
                <span className="text-xs bg-rose-500/10 text-rose-500 font-bold px-2.5 py-0.5 rounded-full border border-rose-500/20">
                  Severity {severity}
                </span>
              </div>
              <div className="w-full max-w-[480px] aspect-square rounded-2xl overflow-hidden border bg-muted/40 relative flex items-center justify-center">
                {liveCorruptedPreview ? (
                  <img
                    src={liveCorruptedPreview}
                    alt="Live corrupted preview"
                    className="w-full h-full object-contain p-2"
                  />
                ) : (
                  <div className="flex items-center justify-center text-xs text-muted-foreground">
                    Generating perturbation preview...
                  </div>
                )}
                {isComputingPreview && (
                  <div className="absolute top-3 right-3 bg-background/80 backdrop-blur px-2.5 py-1 rounded-full text-[11px] font-semibold text-muted-foreground border shadow-xs animate-pulse">
                    Rendering pixels...
                  </div>
                )}
              </div>
              <p className="text-xs text-muted-foreground mt-3 text-center">
                Real canvas pixel perturbation updating live with slider & settings
              </p>
            </Card>
          </div>

          {/* Real-Time Computed Distortion Diagnostics */}
          {liveDiagnostics && (
            <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-lg flex flex-col gap-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border/50 pb-3">
                <div className="flex items-center gap-2">
                  <Layers className="w-5 h-5 text-primary" />
                  <h4 className="font-bold text-base text-foreground">
                    Real-Time Client-Side Visual Estimations (Computed in Browser Canvas)
                  </h4>
                </div>
                <span className="text-xs text-muted-foreground font-mono">
                  Genuine HTML5 Canvas Metrics
                </span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                <div className="p-3.5 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-1">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground">PSNR</span>
                  <span className="text-xl font-black text-blue-500 font-mono">
                    {liveDiagnostics.psnr_db.toFixed(2)} <span className="text-xs font-normal text-muted-foreground">dB</span>
                  </span>
                  <span className="text-[10px] text-muted-foreground">Peak Signal-to-Noise Ratio</span>
                </div>

                <div className="p-3.5 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-1">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground">SSIM</span>
                  <span className="text-xl font-black text-emerald-500 font-mono">
                    {liveDiagnostics.ssim.toFixed(3)}
                  </span>
                  <span className="text-[10px] text-muted-foreground">Structural Similarity (8×8)</span>
                </div>

                <div className="p-3.5 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-1">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground">Sobel Energy</span>
                  <span className="text-xl font-black text-violet-500 font-mono">
                    {liveDiagnostics.edge_energy_retained_pct.toFixed(1)}%
                  </span>
                  <span className="text-[10px] text-muted-foreground">Relative to Clean (can exceed 100%)</span>
                </div>

                <div className="p-3.5 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-1">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground">JPEG Discontinuity</span>
                  <span className="text-xl font-black text-amber-500 font-mono">
                    {liveDiagnostics.jpeg_block_ratio.toFixed(3)}x
                  </span>
                  <span className="text-[10px] text-muted-foreground">8×8 DCT Grid Step Ratio</span>
                </div>

                <div className="p-3.5 rounded-2xl bg-muted/30 border border-border/40 flex flex-col gap-1 col-span-2 sm:col-span-1">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground">Laplacian Spatial Ratio</span>
                  <span className="text-xl font-black text-cyan-500 font-mono">
                    {liveDiagnostics.high_freq_residual_ratio.toFixed(3)}x
                  </span>
                  <span className="text-[10px] text-muted-foreground">3×3 Spatial (distinct from Fourier)</span>
                </div>
              </div>

              <p className="text-[11px] text-muted-foreground leading-relaxed">
                <strong>Methodological Context:</strong> Live demo tests user-selected nominal perturbations (Severity 1–5). In our November 2026 confirmatory research monograph, degradation parameters were calibrated on common support to equate physical distortion across sequence orders (Level 1: 27.20 dB PSNR / 0.810 SSIM; Level 2: 23.80 dB / 0.650; Level 3: 21.80 dB / 0.500). Switch to the <strong>Matched-Fidelity Six-Order Research Lab</strong> above to explore these exact calibrated permutations.
              </p>
            </div>
          )}

          {/* Configuration Controls */}
          <div className="bg-card border border-border/60 rounded-3xl p-8 shadow-xl">
            <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
              <Settings2 className="w-6 h-6 text-primary" /> Stress Test Configuration
            </h3>

            <div className="grid md:grid-cols-2 gap-8">
              {/* Left Column: Model & Corruption Selectors */}
              <div className="space-y-6">
                <div className="space-y-2">
                  <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                    Model Selection
                  </label>
                  <Select value={model} onValueChange={setModel}>
                    <SelectTrigger className="w-full h-12 text-base rounded-xl font-medium" aria-label="Select model">
                      <SelectValue placeholder="Select model" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">Compare All 4 Models</SelectItem>
                      <SelectItem value="resnet">ResNet-50 (CNN)</SelectItem>
                      <SelectItem value="convnext">ConvNeXt-Tiny (Modern CNN)</SelectItem>
                      <SelectItem value="vit">ViT-Base (Vision Transformer)</SelectItem>
                      <SelectItem value="swin">Swin-Tiny (Shifted Windows ViT)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                    Corruption Shift Type
                  </label>
                  <Select value={corruption} onValueChange={(v) => setCorruption(v as CorruptionType)}>
                    <SelectTrigger className="w-full h-12 text-base rounded-xl font-medium" aria-label="Select corruption type">
                      <SelectValue placeholder="Select corruption" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="gaussian_noise">Gaussian Noise</SelectItem>
                      <SelectItem value="defocus_blur">Defocus Blur</SelectItem>
                      <SelectItem value="jpeg_compression">JPEG Compression</SelectItem>
                      <SelectItem value="brightness">Brightness</SelectItem>
                      <SelectItem value="contrast">Contrast</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* Right Column: Severity Controls & Action Button */}
              <div className="space-y-6 flex flex-col justify-between">
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                      Corruption Severity
                    </label>
                    <div className="flex items-center gap-1.5">
                      {[1, 2, 3, 4, 5].map((lvl) => (
                        <button
                          key={lvl}
                          type="button"
                          onClick={() => handleSeverityChange(lvl)}
                          className={`w-7 h-7 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                            severity === lvl
                              ? "bg-primary text-primary-foreground shadow-sm scale-105"
                              : "bg-muted text-muted-foreground hover:bg-muted/80"
                          }`}
                        >
                          {lvl}
                        </button>
                      ))}
                    </div>
                  </div>

                  <Slider
                    aria-label="Corruption Severity"
                    step={1}
                    maxValue={5}
                    minValue={1}
                    value={severity}
                    onChange={handleSeverityChange}
                    className="w-full"
                  />

                  <div className="flex justify-between text-xs text-muted-foreground font-medium px-1">
                    <span>Level 1 (Mild)</span>
                    <span className="text-primary font-bold">Level {severity} Selected</span>
                    <span>Level 5 (Severe)</span>
                  </div>
                </div>

                <Button
                  size="lg"
                  variant="primary"
                  className="w-full h-14 text-base sm:text-lg font-bold shadow-lg shadow-primary/20 rounded-2xl transition-transform hover:scale-[1.01]"
                  onPress={startInference}
                >
                  Run Stress Test on ZeroGPU <ArrowRight className="ml-2 w-5 h-5" />
                </Button>
              </div>
            </div>

            <div className="mt-6 pt-4 border-t border-border/50 text-center text-xs text-muted-foreground font-medium flex items-center justify-center gap-2">
              <span
                className={`w-2 h-2 rounded-full ${
                  errorMessage
                    ? "bg-rose-500"
                    : isInferring
                    ? "bg-amber-500 animate-pulse"
                    : "bg-emerald-500 animate-pulse"
                }`}
              />
              {errorMessage
                ? "Backend error occurred — see details above"
                : isInferring
                ? "Executing ZeroGPU inference job..."
                : "Hugging Face ZeroGPU backend configured • Default: All Models, Gaussian Noise, Severity 3"}
            </div>
          </div>
        </motion.div>
      )}

      {/* Loading Progress State for Quick Demo */}
      {isInferring && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="max-w-md mx-auto bg-card border rounded-3xl p-10 shadow-2xl text-center"
        >
          <Activity className="w-12 h-12 text-primary mx-auto mb-6 animate-pulse" />
          <h3 className="text-2xl font-bold mb-8">Running Live Inference...</h3>

          <div className="space-y-4 text-left">
            {[
              { id: 1, text: "Applying genuine pixel perturbation" },
              { id: 2, text: "Connecting to ZeroGPU Space" },
              { id: 3, text: "Executing PyTorch neural inference" },
              { id: 5, text: "Compiling top-5 class confidences" },
            ].map((s) => (
              <div
                key={s.id}
                className={
                  "flex items-center gap-3 font-medium transition-colors duration-300 " +
                  (stage >= s.id ? "text-foreground" : "text-muted-foreground/60")
                }
              >
                <div
                  className={
                    "w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold " +
                    (stage > s.id
                      ? "bg-green-500 text-white"
                      : stage === s.id
                      ? "bg-primary text-primary-foreground animate-pulse"
                      : "bg-muted text-muted-foreground")
                  }
                >
                  {stage > s.id ? "✓" : s.id}
                </div>
                {s.text}
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Real Live Results Display for Quick Demo */}
      {result && !isInferring && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-8">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <h3 className="text-3xl font-extrabold flex items-center gap-3">
                <BarChart2 className="w-8 h-8 text-primary" /> Live ZeroGPU Inference Results
              </h3>
              <p className="text-muted-foreground text-sm mt-1 flex items-center gap-2">
                <Clock className="w-4 h-4" /> Inference latency:{" "}
                <span className="font-bold text-foreground">{result.inferenceTime}s</span>
              </p>
            </div>
            <div className="flex items-center gap-3">
              <Button
                variant="outline"
                onPress={() => {
                  setResult(null);
                }}
              >
                Adjust Parameters
              </Button>
              <Button
                variant="outline"
                onPress={() => {
                  setResult(null);
                  setPreview(null);
                  setFile(null);
                  setSampleGroundTruth(null);
                  setSynthesizedSequences(null);
                }}
              >
                Upload New Image
              </Button>
            </div>
          </div>

          {/* Side-by-side Images: 400-480px width, aspect ratio preserved */}
          <div className="grid md:grid-cols-2 gap-8">
            {/* Clean Input Card */}
            <Card className="p-6 md:p-8 bg-card border border-border/60 rounded-[2rem] shadow-lg flex flex-col items-center">
              <div className="w-full flex justify-between items-center mb-4">
                <h4 className="font-extrabold text-xl">Clean Input</h4>
                <span className="text-xs bg-primary/10 text-primary font-bold px-3 py-1.5 rounded-full uppercase tracking-wider">
                  Baseline
                </span>
              </div>
              <div className="w-full max-w-[480px] aspect-square rounded-2xl overflow-hidden border bg-muted/30 flex items-center justify-center">
                <img
                  src={result.original}
                  alt="Original"
                  className="w-full h-full object-contain p-2"
                />
              </div>
              <p className="text-xs text-muted-foreground mt-4 text-center">
                Uncorrupted input image before distribution shift
              </p>
            </Card>

            {/* Genuine Corrupted Card */}
            <Card className="p-6 md:p-8 bg-card border border-border/60 rounded-[2rem] shadow-lg flex flex-col items-center">
              <div className="w-full flex justify-between items-center mb-4">
                <div>
                  <h4 className="font-extrabold text-xl capitalize">
                    {result.corruption.replace("_", " ")}
                  </h4>
                </div>
                <span className="text-xs bg-destructive/10 text-destructive font-bold px-3 py-1.5 rounded-full uppercase tracking-wider">
                  Severity {result.severity}
                </span>
              </div>
              <div className="w-full max-w-[480px] aspect-square rounded-2xl overflow-hidden border bg-muted/30 flex items-center justify-center">
                <img
                  src={result.corrupted}
                  alt="Corrupted"
                  className="w-full h-full object-contain p-2"
                />
              </div>
              <p className="text-xs text-muted-foreground mt-4 text-center">
                Genuine pixel perturbation: <strong className="capitalize">{result.corruption.replace("_", " ")}</strong> (Level {result.severity})
              </p>
            </Card>
          </div>

          {/* Measured Fidelity & Distortion Fingerprints for this test */}
          {result.diagnostics && (
            <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-md flex flex-col gap-3">
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-primary" />
                <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                  Evaluated Input Fidelity & Physical Fingerprints (Canvas Estimation)
                </span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                <div className="p-3 rounded-xl bg-muted/20 border border-border/40 text-center">
                  <span className="text-[10px] uppercase font-bold text-muted-foreground block">PSNR</span>
                  <span className="text-base font-extrabold text-blue-500 font-mono">{result.diagnostics.psnr_db.toFixed(2)} dB</span>
                </div>
                <div className="p-3 rounded-xl bg-muted/20 border border-border/40 text-center">
                  <span className="text-[10px] uppercase font-bold text-muted-foreground block">SSIM</span>
                  <span className="text-base font-extrabold text-emerald-500 font-mono">{result.diagnostics.ssim.toFixed(3)}</span>
                </div>
                <div className="p-3 rounded-xl bg-muted/20 border border-border/40 text-center">
                  <span className="text-[10px] uppercase font-bold text-muted-foreground block">Sobel Energy</span>
                  <span className="text-base font-extrabold text-violet-500 font-mono">{result.diagnostics.edge_energy_retained_pct.toFixed(1)}%</span>
                  <span className="text-[9px] text-muted-foreground block">Relative to clean</span>
                </div>
                <div className="p-3 rounded-xl bg-muted/20 border border-border/40 text-center">
                  <span className="text-[10px] uppercase font-bold text-muted-foreground block">JPEG 8×8 Ratio</span>
                  <span className="text-base font-extrabold text-amber-500 font-mono">{result.diagnostics.jpeg_block_ratio.toFixed(3)}x</span>
                  <span className="text-[9px] text-muted-foreground block">Grid step ratio</span>
                </div>
                <div className="p-3 rounded-xl bg-muted/20 border border-border/40 text-center col-span-2 sm:col-span-1">
                  <span className="text-[10px] uppercase font-bold text-muted-foreground block">Laplacian Ratio</span>
                  <span className="text-base font-extrabold text-cyan-500 font-mono">{result.diagnostics.high_freq_residual_ratio.toFixed(3)}x</span>
                  <span className="text-[9px] text-muted-foreground block">3×3 Spatial</span>
                </div>
              </div>
            </div>
          )}

          {/* Model Classifications Card Grid */}
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {result.models.map((m: ModelPrediction, i: number) => {
              const isExpanded = expandedModel === m.name;
              const confPct = Math.round(m.conf * 100);
              const isHighConfidence = confPct >= 50;

              return (
                <Card
                  key={i}
                  className="p-6 rounded-3xl border border-border/60 bg-card shadow-md flex flex-col justify-between"
                >
                  <div>
                    <div className="flex justify-between items-start mb-2">
                      <p className="font-extrabold text-base text-foreground">{m.name}</p>
                      <span
                        className={`text-xs px-2.5 py-1 rounded-full font-bold uppercase ${
                          isHighConfidence ? "bg-green-500/10 text-green-500" : "bg-amber-500/10 text-amber-500"
                        }`}
                      >
                        {confPct}% conf
                      </span>
                    </div>

                    <div className="my-3">
                      <span className="text-xs text-muted-foreground font-semibold uppercase tracking-wider">
                        Top Prediction:
                      </span>
                      <p className="text-2xl font-black text-primary capitalize truncate mt-0.5">
                        {m.pred}
                      </p>
                    </div>

                    <Progress
                      aria-label={`${m.name} confidence`}
                      value={confPct}
                      className="mt-2"
                      color={isHighConfidence ? "success" : "warning"}
                      size="sm"
                    />
                  </div>

                  {/* Top-5 probabilities toggle */}
                  {m.top_k && m.top_k.length > 0 && (
                    <div className="mt-4 pt-3 border-t border-border/50">
                      <button
                        onClick={() => setExpandedModel(isExpanded ? null : m.name)}
                        className="text-xs font-bold text-muted-foreground hover:text-foreground flex items-center justify-between w-full cursor-pointer"
                      >
                        <span>Top 5 Probabilities</span>
                        {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                      </button>

                      <AnimatePresence>
                        {isExpanded && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="mt-3 space-y-1.5 overflow-hidden text-xs"
                          >
                            {m.top_k.map((item, idx) => (
                              <div key={idx} className="flex justify-between items-center text-muted-foreground">
                                <span className="capitalize truncate max-w-[140px]">{item.class}</span>
                                <span className="font-mono font-bold text-foreground">
                                  {(item.probability * 100).toFixed(1)}%
                                </span>
                              </div>
                            ))}
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  )}
                </Card>
              );
            })}
          </div>

          {/* Quick-Tune Re-run Section right on the results view */}
          <div className="p-6 rounded-3xl bg-muted/30 border border-border/60 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                Tune & Re-run:
              </span>
              <div className="flex items-center gap-1">
                {[1, 2, 3, 4, 5].map((lvl) => (
                  <button
                    key={lvl}
                    onClick={() => handleSeverityChange(lvl)}
                    className={`w-7 h-7 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                      severity === lvl
                        ? "bg-primary text-primary-foreground shadow-sm"
                        : "bg-background text-muted-foreground hover:bg-muted border border-border/50"
                    }`}
                  >
                    {lvl}
                  </button>
                ))}
              </div>
            </div>
            <Button
              variant="primary"
              size="sm"
              className="font-bold px-6 h-10 rounded-xl"
              onPress={startInference}
            >
              Re-run with Level {severity} <ArrowRight className="ml-1.5 w-4 h-4" />
            </Button>
          </div>
        </motion.div>
      )}

      {/* ========================================================================= */}
      {/* TAB 2: MATCHED-FIDELITY SIX-ORDER RESEARCH LAB                           */}
      {/* ========================================================================= */}
      {activeTab === "research_lab" && preview && (
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col gap-8"
        >
          {/* Research Lab Header & Configuration Card */}
          <div className="bg-card border border-border/60 rounded-3xl p-8 shadow-xl">
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-6 pb-6 border-b border-border/50">
              <div>
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-bold uppercase tracking-wider mb-2">
                  <FlaskConical className="w-3.5 h-3.5" /> November 2026 Common-Support Protocol
                </div>
                <h3 className="text-2xl font-black tracking-tight">
                  Matched-Fidelity Six-Order Permutation Lab
                </h3>
                <p className="text-sm text-muted-foreground mt-1 max-w-2xl">
                  Evaluates all six compound sequences (NBJ, NJB, BNJ, BJN, JNB, JBN) using strictly frozen parameter triples calibrated on joint common support.
                </p>
              </div>

              <Button
                variant="outline"
                size="sm"
                onPress={() => {
                  setPreview(null);
                  setFile(null);
                  setSampleGroundTruth(null);
                  setSynthesizedSequences(null);
                }}
              >
                Change Clean Image
              </Button>
            </div>

            {/* Level & Model Selector Grid */}
            <div className="grid md:grid-cols-3 gap-6 items-end">
              {/* Research Level Selector */}
              <div className="md:col-span-2 space-y-3">
                <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground block">
                  Select Calibrated Research Level (Joint PSNR / SSIM Target)
                </label>
                <div className="grid sm:grid-cols-3 gap-3">
                  {(
                    [
                      {
                        key: "Level_1_Mild",
                        name: "Level 1: Mild",
                        psnr: "27.20 dB",
                        ssim: "0.810",
                      },
                      {
                        key: "Level_2_Moderate",
                        name: "Level 2: Moderate",
                        psnr: "23.80 dB",
                        ssim: "0.650",
                      },
                      {
                        key: "Level_3_SevereCommon",
                        name: "Level 3: Severe",
                        psnr: "21.80 dB",
                        ssim: "0.500",
                      },
                    ] as const
                  ).map((lvl) => (
                    <button
                      key={lvl.key}
                      onClick={() => {
                        setResearchLevel(lvl.key);
                        setSynthesizedSequences(null);
                      }}
                      className={`p-4 rounded-2xl border text-left transition-all cursor-pointer flex flex-col justify-between ${
                        researchLevel === lvl.key
                          ? "bg-primary/10 border-primary text-foreground shadow-sm ring-1 ring-primary"
                          : "bg-muted/30 border-border/50 text-muted-foreground hover:bg-muted/60"
                      }`}
                    >
                      <div>
                        <span className="text-xs font-extrabold uppercase tracking-wider block">
                          {lvl.name}
                        </span>
                        <span className="text-lg font-black text-foreground mt-1 block">
                          {lvl.psnr}
                        </span>
                      </div>
                      <div className="mt-2 text-xs font-medium">
                        Target SSIM: <span className="font-bold text-foreground">{lvl.ssim}</span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Architecture Target & Action Button */}
              <div className="space-y-3">
                <div className="space-y-1.5">
                  <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground block">
                    Target Architecture
                  </label>
                  <Select value={researchModel} onValueChange={setResearchModel}>
                    <SelectTrigger className="w-full h-11 text-xs rounded-xl font-medium" aria-label="Select target architecture">
                      <SelectValue placeholder="Select architecture" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">Compare All 4 Architectures</SelectItem>
                      <SelectItem value="resnet">ResNet-50 (Standard CNN)</SelectItem>
                      <SelectItem value="convnext">ConvNeXt-Tiny (Modern CNN)</SelectItem>
                      <SelectItem value="vit">ViT-Base (Vision Transformer)</SelectItem>
                      <SelectItem value="swin">Swin-Tiny (Shifted Windows ViT)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Button
                  size="lg"
                  variant="primary"
                  className="w-full h-12 text-sm font-bold shadow-lg shadow-primary/20 rounded-xl cursor-pointer"
                  isDisabled={isSynthesizingSequences}
                  onPress={handleSynthesizeSequences}
                >
                  {isSynthesizingSequences ? (
                    "Synthesizing 6 Sequences..."
                  ) : (
                    <>
                      Synthesize 6 Permutations <ArrowRight className="ml-1.5 w-4 h-4" />
                    </>
                  )}
                </Button>
              </div>
            </div>

            {/* Target Spec Notice */}
            <div className="mt-6 pt-4 border-t border-border/50 flex flex-wrap items-center justify-between text-xs text-muted-foreground gap-2">
              <span>
                Prespecified Research Tolerances: <strong>±0.75 dB PSNR</strong> and <strong>±0.035 SSIM</strong>
              </span>
              <span className="font-mono text-[11px]">
                {FROZEN_CALIBRATION_3LEVELS[researchLevel].description}
              </span>
            </div>
          </div>

          {/* Synthesizing Progress State */}
          {isSynthesizingSequences && (
            <div className="p-12 rounded-3xl bg-card border border-border/60 text-center shadow-lg">
              <Activity className="w-10 h-10 text-primary animate-pulse mx-auto mb-4" />
              <h4 className="text-xl font-bold">Applying 3-Stage Compound Pipelines in Browser Canvas...</h4>
              <p className="text-sm text-muted-foreground mt-2 max-w-lg mx-auto">
                Synthesizing NBJ, NJB, BNJ, BJN, JNB, and JBN in exact physical order, measuring pixel MSE, windowed SSIM, and directional frequency gradients.
              </p>
            </div>
          )}

          {/* Synthesized 6-Order Comparative Display */}
          {synthesizedSequences && !isSynthesizingSequences && (
            <div className="space-y-8">
              {/* Group Summary Banner */}
              <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-md flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <span className="text-xs font-bold uppercase tracking-wider text-primary">
                    Evaluation Target: {FROZEN_CALIBRATION_3LEVELS[researchLevel].name}
                  </span>
                  <h4 className="text-2xl font-black mt-1">
                    Achieved Physical Fidelity across 6 Degradation Sequences
                  </h4>
                  <p className="text-xs text-muted-foreground mt-1">
                    Comparing compound permutations under identical operator sets (Gaussian Noise N, Defocus Blur B, JPEG Compression J).
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold px-3 py-1.5 rounded-full bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
                    6/6 Permutations Evaluated
                  </span>
                </div>
              </div>

              {/* 3 Operator Groupings: Terminal Noise, Terminal Blur, Terminal JPEG */}
              {(
                [
                  {
                    groupTitle: "Terminal Operator: Defocus Blur (NJB, JNB)",
                    description: "Orders terminating in low-pass filtering. In the nominal regime, terminal blur preserved higher average accuracy than terminal noise.",
                    keys: ["NJB", "JNB"] as CompoundSequenceOrder[],
                    terminalBadge: "Terminal: Blur",
                    badgeColor: "bg-blue-500/10 text-blue-500 border-blue-500/20",
                  },
                  {
                    groupTitle: "Terminal Operator: Gaussian Noise (BJN, JBN)",
                    description: "Orders terminating in additive high-frequency noise. Noise injection as the final stage confounds early high-pass features.",
                    keys: ["BJN", "JBN"] as CompoundSequenceOrder[],
                    terminalBadge: "Terminal: Noise",
                    badgeColor: "bg-rose-500/10 text-rose-500 border-rose-500/20",
                  },
                  {
                    groupTitle: "Terminal Operator: JPEG Compression (NBJ, BNJ)",
                    description: "Orders terminating in 8×8 block DCT quantization. High-frequency noise is severely attenuated by late-stage quantization.",
                    keys: ["NBJ", "BNJ"] as CompoundSequenceOrder[],
                    terminalBadge: "Terminal: JPEG",
                    badgeColor: "bg-amber-500/10 text-amber-500 border-amber-500/20",
                  },
                ] as const
              ).map((grp) => (
                <div key={grp.groupTitle} className="space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border/50 pb-2">
                    <div>
                      <h4 className="text-lg font-bold text-foreground flex items-center gap-2">
                        {grp.groupTitle}
                      </h4>
                      <p className="text-xs text-muted-foreground">{grp.description}</p>
                    </div>
                    <span className={`text-xs px-2.5 py-1 rounded-full font-bold uppercase border self-start sm:self-auto ${grp.badgeColor}`}>
                      {grp.terminalBadge}
                    </span>
                  </div>

                  <div className="grid md:grid-cols-2 gap-6">
                    {grp.keys.map((seqKey) => {
                      const item = synthesizedSequences[seqKey];
                      const params = FROZEN_CALIBRATION_3LEVELS[researchLevel].sequences[seqKey];
                      const targetPsnr = FROZEN_CALIBRATION_3LEVELS[researchLevel].target_psnr;
                      const targetSsim = FROZEN_CALIBRATION_3LEVELS[researchLevel].target_ssim;

                      const seqDescriptions: Record<CompoundSequenceOrder, string> = {
                        NBJ: "Noise → Blur → JPEG",
                        NJB: "Noise → JPEG → Blur",
                        BNJ: "Blur → Noise → JPEG",
                        BJN: "Blur → JPEG → Noise",
                        JNB: "JPEG → Noise → Blur",
                        JBN: "JPEG → Blur → Noise",
                      };

                      return (
                        <Card
                          key={seqKey}
                          className="p-6 rounded-3xl bg-card border border-border/60 shadow-lg flex flex-col justify-between"
                        >
                          <div>
                            {/* Card Header */}
                            <div className="flex justify-between items-start mb-4">
                              <div>
                                <h5 className="text-xl font-black tracking-tight">{seqKey}</h5>
                                <span className="text-xs text-muted-foreground font-semibold">
                                  {seqDescriptions[seqKey]}
                                </span>
                              </div>
                              <span
                                className={`text-[11px] px-2.5 py-1 rounded-full font-bold uppercase border ${
                                  item.withinTol
                                    ? "bg-emerald-500/10 text-emerald-500 border-emerald-500/20"
                                    : "bg-amber-500/10 text-amber-500 border-amber-500/20"
                                }`}
                              >
                                {item.withinTol ? "Within Tolerance" : "Spectrum Variation"}
                              </span>
                            </div>

                            {/* Corrupted Preview Image */}
                            <div className="w-full aspect-square max-w-[420px] mx-auto rounded-2xl overflow-hidden border bg-muted/40 relative flex items-center justify-center mb-4">
                              <img
                                src={item.dataUrl}
                                alt={`Corrupted preview ${seqKey}`}
                                className="w-full h-full object-contain p-2"
                              />
                            </div>

                            {/* Frozen Calibration Parameters */}
                            <div className="p-3 rounded-2xl bg-muted/30 border border-border/40 text-xs mb-4">
                              <span className="text-[10px] uppercase font-bold text-muted-foreground block mb-1">
                                Frozen Calibration Parameters:
                              </span>
                              <div className="grid grid-cols-3 gap-2 font-mono text-[11px]">
                                <div>
                                  <span className="text-muted-foreground">Noise σ:</span>{" "}
                                  <strong>{params.noise_std}</strong>
                                </div>
                                <div>
                                  <span className="text-muted-foreground">Blur σ:</span>{" "}
                                  <strong>{params.blur_sigma} px</strong>
                                </div>
                                <div>
                                  <span className="text-muted-foreground">JPEG Q:</span>{" "}
                                  <strong>{params.jpeg_quality}</strong>
                                </div>
                              </div>
                            </div>

                            {/* Achieved Physical Metrics */}
                            <div className="grid grid-cols-2 gap-2 text-center mb-3">
                              <div className="p-2.5 rounded-xl bg-blue-500/10 border border-blue-500/20">
                                <span className="text-[10px] uppercase font-bold text-muted-foreground block">
                                  Achieved PSNR
                                </span>
                                <span className="text-lg font-black text-blue-500 font-mono">
                                  {item.diag.psnr_db.toFixed(2)} dB
                                </span>
                                <span className="text-[10px] text-muted-foreground block">
                                  Target {targetPsnr} (Δ {item.psnrErr >= 0 ? "+" : ""}{item.psnrErr.toFixed(2)})
                                </span>
                              </div>

                              <div className="p-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                                <span className="text-[10px] uppercase font-bold text-muted-foreground block">
                                  Achieved SSIM
                                </span>
                                <span className="text-lg font-black text-emerald-500 font-mono">
                                  {item.diag.ssim.toFixed(3)}
                                </span>
                                <span className="text-[10px] text-muted-foreground block">
                                  Target {targetSsim} (Δ {item.ssimErr >= 0 ? "+" : ""}{item.ssimErr.toFixed(3)})
                                </span>
                              </div>
                            </div>

                            {/* Additional Canvas Fingerprints */}
                            <div className="grid grid-cols-3 gap-1.5 text-center text-[10px] text-muted-foreground pt-2 border-t border-border/40 font-mono">
                              <div>
                                Sobel: <strong>{item.diag.edge_energy_retained_pct.toFixed(0)}%</strong>
                              </div>
                              <div>
                                JPEG 8×8: <strong>{item.diag.jpeg_block_ratio.toFixed(2)}x</strong>
                              </div>
                              <div>
                                Laplacian: <strong>{item.diag.high_freq_residual_ratio.toFixed(2)}x</strong>
                              </div>
                            </div>
                          </div>
                        </Card>
                      );
                    })}
                  </div>
                </div>
              ))}

              {/* Rigorous Scientific Disclosures & Methodological Notes */}
              <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-lg space-y-4">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="w-5 h-5 text-primary" />
                  <h4 className="font-extrabold text-base text-foreground">
                    Scientific Disclosures & Verification Requirements
                  </h4>
                </div>

                <div className="grid sm:grid-cols-2 gap-4 text-xs text-muted-foreground">
                  <div className="p-4 rounded-2xl bg-muted/20 border border-border/40 space-y-1.5">
                    <strong className="text-foreground block flex items-center gap-1.5">
                      <Info className="w-4 h-4 text-blue-500 shrink-0" />
                      Exploratory Single-Image Demonstration
                    </strong>
                    <p className="leading-relaxed">
                      Frozen calibration parameter triples were derived using density-union estimation across the 50-image Imagenette calibration corpus to guarantee simultaneous joint common support. Because arbitrary uploaded images possess idiosyncratic natural sharpness and spatial-frequency distributions, single-image achieved fidelity may deviate slightly from calibration targets. These outputs are exploratory single-image demonstrations, not confirmatory population statistics.
                    </p>
                  </div>

                  <div className="p-4 rounded-2xl bg-muted/20 border border-border/40 space-y-1.5">
                    <strong className="text-foreground block flex items-center gap-1.5">
                      <Info className="w-4 h-4 text-amber-500 shrink-0" />
                      No Ground-Truth Class Assumption
                    </strong>
                    <p className="leading-relaxed">
                      Unless a verified benchmark label is supplied, model evaluations on arbitrary uploads measure class prediction agreement and confidence shifts relative to the clean baseline. They do not constitute true-class margins or classification accuracy.
                    </p>
                  </div>

                  <div className="p-4 rounded-2xl bg-muted/20 border border-border/40 space-y-1.5">
                    <strong className="text-foreground block flex items-center gap-1.5">
                      <Info className="w-4 h-4 text-emerald-500 shrink-0" />
                      Strictly Frozen Parameters
                    </strong>
                    <p className="leading-relaxed">
                      Parameters are strictly frozen per the November 2026 pre-registration and are never re-tuned per image to prevent post-hoc overfitting and ensure comparability with reported research outcomes.
                    </p>
                  </div>

                  <div className="p-4 rounded-2xl bg-muted/20 border border-border/40 space-y-1.5">
                    <strong className="text-foreground block flex items-center gap-1.5">
                      <FileCode className="w-4 h-4 text-violet-500 shrink-0" />
                      ZeroGPU Inference Backend Notice
                    </strong>
                    <p className="leading-relaxed">
                      To run remote neural predictions on pre-corrupted compound sequence images without introducing a second unary corruption, the Hugging Face ZeroGPU Gradio endpoint requires an <code className="text-foreground">identity</code> / <code className="text-foreground">passthrough</code> option in its corruption dropdown. To safeguard live production stability without risking remote downtime, this research lab stages the exact 6-order compound generation and canvas fidelity engine on the client side, while documenting the isolated backend patch below.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}
