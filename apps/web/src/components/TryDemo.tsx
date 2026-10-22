"use client";

import React, { useState, useRef } from "react";
import { Card, Button, ProgressBar as Progress, Slider } from "@heroui/react";
import {
  Upload,
  ArrowRight,
  Activity,
  Settings2,
  BarChart2,
  CheckCircle2,
  AlertCircle,
  Clock,
  Sparkles,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { motion, AnimatePresence } from "framer-motion";
import { applyClientPixelCorruption, CorruptionType } from "@/lib/corruptions";

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
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [sampleGroundTruth, setSampleGroundTruth] = useState<string | null>(null);

  const [model, setModel] = useState("all");
  const [corruption, setCorruption] = useState<CorruptionType>("gaussian_noise");
  const [severity, setSeverity] = useState(3);

  const [isInferring, setIsInferring] = useState(false);
  const [stage, setStage] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isQuotaError, setIsQuotaError] = useState(false);
  const [result, setResult] = useState<DemoResult | null>(null);
  const [expandedModel, setExpandedModel] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const f = e.target.files[0];
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setSampleGroundTruth(null);
      setResult(null);
      setErrorMessage(null);
      setIsQuotaError(false);
    }
  };

  const selectSamplePreset = async (preset: (typeof SAMPLE_PRESETS)[0]) => {
    setSampleGroundTruth(preset.groundTruth);
    setPreview(preset.url);
    setResult(null);
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

  const startInference = async () => {
    if (!preview) return;
    setIsInferring(true);
    setErrorMessage(null);
    setIsQuotaError(false);
    setStage(1); // Preparing image

    try {
      // 1. Generate genuinely corrupted image pixels for UI display
      // This uses HTML5 canvas pixel transformations matching hf_space/app.py exactly
      const genuineCorruptedDataUrl = await applyClientPixelCorruption(
        preview,
        corruption,
        severity
      );

      setStage(2); // Connecting to ZeroGPU Space

      // 2. Prepare base64 clean image for remote inference
      // IMPORTANT: To prevent double-corruption, we transmit the CLEAN image to /api/inference.
      // The Hugging Face ZeroGPU endpoint is the authoritative evaluation locus.
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

      setStage(3); // Running model inference on ZeroGPU (NVIDIA A10G)

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

      setStage(5); // Parsing classifications

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

        setResult({
          original: preview,
          corrupted: genuineCorruptedDataUrl,
          corruption,
          severity,
          models: mappedModels,
          inferenceTime: apiResult.inference_time_seconds ?? 0.5,
          sampleGroundTruth: sampleGroundTruth || undefined,
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

  return (
    <div className="w-full max-w-[1300px] mx-auto mt-16 mb-24 relative font-[family-name:var(--font-poppins)]">
      <div className="text-center mb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-bold uppercase tracking-wider mb-3">
          <Sparkles className="w-3.5 h-3.5" /> Live Hugging Face ZeroGPU Service
        </div>
        <h2 className="text-4xl md:text-5xl font-extrabold tracking-tight">Try RobustLens Live</h2>
        <p className="mt-4 text-xl text-muted-foreground font-medium max-w-2xl mx-auto">
          Upload any image and evaluate vision model predictions under calibrated distribution shifts in real-time.
        </p>

        {/* Sample Image Presets */}
        <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
          <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Or pick a sample:</span>
          {SAMPLE_PRESETS.map((p) => (
            <button
              key={p.name}
              onClick={() => selectSamplePreset(p)}
              className="px-3.5 py-1.5 rounded-full text-xs font-semibold bg-muted hover:bg-primary/20 hover:text-primary transition-colors border border-border/60 shadow-sm"
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
              {isQuotaError ? "Hugging Face ZeroGPU Limit Reached" : "Inference Request Failed"}
            </h4>
            <p className="text-sm opacity-90">{errorMessage}</p>
          </div>
        </motion.div>
      )}

      {/* Image Upload Box */}
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

      {/* Image Uploaded — Configuration Screen */}
      {preview && !result && !isInferring && (
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          className="grid md:grid-cols-2 gap-10 items-start"
        >
          {/* 400-480px width image card with natural aspect ratio */}
          <Card className="p-6 bg-card border border-border/60 shadow-xl rounded-3xl overflow-hidden flex flex-col items-center">
            <div className="w-full max-w-[480px] aspect-square rounded-2xl overflow-hidden border bg-muted/40 relative flex items-center justify-center">
              <img
                src={preview}
                alt="Upload preview"
                className="w-full h-full object-contain p-2"
              />
              <div className="absolute top-4 left-4 bg-background/90 backdrop-blur px-3 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider shadow-sm border">
                Input Image
              </div>
            </div>
            {sampleGroundTruth && (
              <div className="mt-4 text-xs font-bold text-muted-foreground uppercase">
                Known Ground Truth: <span className="text-primary capitalize">{sampleGroundTruth}</span>
              </div>
            )}
            <Button
              variant="outline"
              size="sm"
              className="mt-4"
              onPress={() => {
                setPreview(null);
                setFile(null);
                setSampleGroundTruth(null);
              }}
            >
              Choose Different Image
            </Button>
          </Card>

          {/* Configuration Controls */}
          <div className="flex flex-col gap-6 py-2">
            <div className="bg-card border border-border/60 rounded-3xl p-8 shadow-xl">
              <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <Settings2 className="w-6 h-6 text-primary" /> Stress Test Configuration
              </h3>

              <div className="space-y-6">
                <div className="space-y-3">
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

                <div className="space-y-3">
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

                <div className="space-y-3 pt-2">
                  <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground flex justify-between">
                    <span>Corruption Severity</span>
                    <span className="text-primary font-bold text-sm">Level {severity}</span>
                  </label>
                  <Slider
                    aria-label="Corruption Severity"
                    step={1}
                    maxValue={5}
                    minValue={1}
                    value={severity}
                    onChange={(v) => setSeverity(v as number)}
                    className="max-w-md"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground font-medium px-1">
                    <span>1 (Mild)</span>
                    <span>5 (Severe)</span>
                  </div>
                </div>
              </div>
            </div>

            <Button
              size="lg"
              variant="primary"
              className="w-full h-16 text-lg font-bold shadow-lg shadow-primary/20 rounded-2xl transition-transform hover:scale-[1.01]"
              onPress={startInference}
            >
              Run Stress Test on ZeroGPU <ArrowRight className="ml-2 w-5 h-5" />
            </Button>

            <div className="text-center text-xs text-muted-foreground font-medium flex items-center justify-center gap-2">
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
                : "Hugging Face ZeroGPU backend configured"}
            </div>
          </div>
        </motion.div>
      )}

      {/* Loading Progress State */}
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

      {/* Real Live Results Display */}
      {result && !isInferring && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-8">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-3xl font-extrabold flex items-center gap-3">
                <BarChart2 className="w-8 h-8 text-primary" /> Live ZeroGPU Inference Results
              </h3>
              <p className="text-muted-foreground text-sm mt-1 flex items-center gap-2">
                <Clock className="w-4 h-4" /> Inference latency:{" "}
                <span className="font-bold text-foreground">{result.inferenceTime}s</span>
              </p>
            </div>
            <Button variant="outline" onPress={() => setResult(null)}>
              Test Another Configuration
            </Button>
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
              {/* Clean rendering of genuine corrupted canvas pixels without fake CSS filters */}
              <div className="w-full max-w-[480px] aspect-square rounded-2xl overflow-hidden border bg-muted/30 flex items-center justify-center">
                <img
                  src={result.corrupted}
                  alt="Corrupted"
                  className="w-full h-full object-contain p-2"
                />
              </div>
              <p className="text-xs text-muted-foreground mt-4 text-center">
                Genuine pixel perturbation applied: <strong className="capitalize">{result.corruption.replace("_", " ")}</strong> (Level {result.severity})
              </p>
            </Card>
          </div>

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
                        className="text-xs font-bold text-muted-foreground hover:text-foreground flex items-center justify-between w-full"
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
        </motion.div>
      )}
    </div>
  );
}
