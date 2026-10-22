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
import { AlertTriangle, CheckCircle, TrendingDown, Layers, ShieldCheck, Zap } from "lucide-react";

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

export default function ExplorerPage() {
  const [records, setRecords] = useState<EvaluationRecord[]>([]);
  const [imageId, setImageId] = useState<number>(0);
  const [modelKey, setModelKey] = useState<string>("resnet50");
  const [corruption, setCorruption] = useState<string>("gaussian_noise");
  const [severity, setSeverity] = useState<number>(3);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetch("/demo_results.json")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load demo_results.json");
        return res.json();
      })
      .then((data: EvaluationRecord[]) => {
        setRecords(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error loading benchmark dataset:", err);
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

  return (
    <main className="min-h-screen pt-32 pb-24 px-6 max-w-[1400px] mx-auto flex flex-col gap-10">
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-2 text-primary font-bold text-sm tracking-widest uppercase">
          <Layers className="w-4 h-4" /> Real Dataset Evaluation
        </div>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight font-[family-name:var(--font-poppins)]">
          Image Robustness Explorer
        </h1>
        <p className="text-muted-foreground text-xl max-w-3xl">
          Side-by-side visual analysis of real vision model evaluations under calibrated distribution shifts.
          Compare genuine clean vs. corrupted pixel inputs, degradation curves, and failure modes.
        </p>
      </div>

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

          {/* Severity Slider */}
          <div className="flex flex-col gap-3 pt-2">
            <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider flex justify-between">
              <span>Severity Level</span>
              <span className="text-primary font-extrabold text-sm">Level {severity}</span>
            </label>
            <Slider
              aria-label="Severity Level"
              step={1}
              maxValue={5}
              minValue={1}
              value={severity}
              onChange={(value) => setSeverity(value as number)}
              className="max-w-md"
            />
            <div className="flex justify-between text-xs text-muted-foreground font-medium px-1">
              <span>1 (Mild)</span>
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

                {/* 400-480px responsive image display with object-contain */}
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

                {/* 400-480px responsive image display with object-contain (genuine corrupted pixels) */}
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
    </main>
  );
}
