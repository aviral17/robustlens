"use client";

import React, { useState } from "react";
import { Button, ProgressBar as Progress } from "@heroui/react";
import {
  FlaskConical,
  Activity,
  ArrowRight,
  ShieldCheck,
  Download,
  Eye,
  Maximize2,
  ChevronDown,
  ChevronUp,
  Cpu,
  BarChart3,
  Sparkles,
} from "lucide-react";
import {
  CompoundSequenceOrder,
  ResearchLevelKey,
  FROZEN_CALIBRATION_3LEVELS,
  applyCompoundSequenceWithStages,
  CompoundSequenceResult,
  SequenceStageInfo,
} from "@/lib/corruptions";
import {
  computeManuscriptFingerprints,
  generateFingerprintVisualMaps,
  DistortionFingerprints,
  FingerprintVisualMaps,
} from "@/lib/structural_fingerprints";
import {
  buildResearchPassportJson,
  buildResearchPassportCsv,
  downloadFile,
  computeSha256,
  ResearchPassport,
  SequenceRunRecord,
} from "@/lib/passport_export";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export interface SourceImageMeta {
  name: string;
  dimensions: string;
  format: string;
  size: string;
}

interface SynthesizedSequenceFull {
  sequence: CompoundSequenceOrder;
  finalDataUrl: string;
  stages: [SequenceStageInfo, SequenceStageInfo, SequenceStageInfo, SequenceStageInfo];
  fingerprints: DistortionFingerprints;
  visualMaps: FingerprintVisualMaps;
  withinTol: boolean;
  psnrErr: number;
  ssimErr: number;
}

interface ModelPredictionResult {
  model: string;
  prediction: string;
  confidence: number;
  top_k?: { class: string; probability: number }[];
}

interface ResearchLabViewProps {
  preview: string;
  sourceMeta: SourceImageMeta | null;
  sampleGroundTruth: string | null;
  onChangeImage: () => void;
  onRemoveImage: () => void;
}

export default function ResearchLabView({
  preview,
  sourceMeta,
  sampleGroundTruth,
  onChangeImage,
  onRemoveImage,
}: ResearchLabViewProps) {
  // Calibration level selection
  const [researchLevel, setResearchLevel] = useState<ResearchLevelKey>("Level_1_Mild");
  const [selectedArchitecture, setSelectedArchitecture] = useState<string>("all");

  // Synthesis state
  const [isSynthesizing, setIsSynthesizing] = useState(false);
  const [synthesizedData, setSynthesizedData] = useState<
    Record<CompoundSequenceOrder, SynthesizedSequenceFull> | null
  >(null);
  const [synthesisError, setSynthesisError] = useState<string | null>(null);

  // Microscope interactive state
  const [microscopeSequence, setMicroscopeSequence] = useState<CompoundSequenceOrder>("NBJ");
  const [activeStageIndex, setActiveStageIndex] = useState<number>(3); // 0..3
  const [comparisonMode, setComparisonMode] = useState<"current" | "clean" | "split">("current");
  const [inspectModalImage, setInspectModalImage] = useState<string | null>(null);

  // Fingerprint visual map tab
  const [activeMapTab, setActiveMapTab] = useState<"fourier" | "sobel" | "ssim" | "jpeg">("fourier");

  // Inference state
  const [isInferringMatrix, setIsInferringMatrix] = useState(false);
  const [inferenceProgress, setInferenceProgress] = useState<{ current: number; total: number; label: string } | null>(null);
  const [inferenceMatrix, setInferenceMatrix] = useState<Record<string, Record<string, ModelPredictionResult>> | null>(null);
  const [expandedPredKeys, setExpandedPredKeys] = useState<Record<string, boolean>>({});

  // Synthesize 6 permutations with intermediate stages and exact fingerprints
  const handleSynthesize = async () => {
    if (!preview) return;
    setIsSynthesizing(true);
    setSynthesisError(null);

    try {
      const orders: CompoundSequenceOrder[] = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"];
      const levelSpec = FROZEN_CALIBRATION_3LEVELS[researchLevel];
      const results: Partial<Record<CompoundSequenceOrder, SynthesizedSequenceFull>> = {};

      for (const ord of orders) {
        const params = levelSpec.sequences[ord];
        const seqResult: CompoundSequenceResult = await applyCompoundSequenceWithStages(preview, ord, {
          noise_std: params.noise_std,
          blur_sigma: params.blur_sigma,
          jpeg_quality: params.jpeg_quality,
        });

        // Compute exact manuscript fingerprints & visual diagnostic maps
        const [fps, visualMaps] = await Promise.all([
          computeManuscriptFingerprints(preview, seqResult.finalDataUrl),
          generateFingerprintVisualMaps(preview, seqResult.finalDataUrl),
        ]);

        const psnrErr = Math.round((fps.psnr_db - levelSpec.target_psnr) * 100) / 100;
        const ssimErr = Math.round((fps.mean_ssim - levelSpec.target_ssim) * 1000) / 1000;
        const withinTol = Math.abs(psnrErr) <= 0.75 && Math.abs(ssimErr) <= 0.035;

        results[ord] = {
          sequence: ord,
          finalDataUrl: seqResult.finalDataUrl,
          stages: seqResult.stages,
          fingerprints: fps,
          visualMaps,
          withinTol,
          psnrErr,
          ssimErr,
        };
      }

      setSynthesizedData(
        results as Record<CompoundSequenceOrder, SynthesizedSequenceFull>
      );
    } catch (err: unknown) {
      console.error("Synthesis error:", err);
      setSynthesisError(err instanceof Error ? err.message : "Failed to synthesize permutations");
    } finally {
      setIsSynthesizing(false);
    }
  };

  // Run genuine Hugging Face ZeroGPU inference across Clean + 6 Permutations
  const handleRunInferenceMatrix = async () => {
    if (!preview || !synthesizedData) return;
    setIsInferringMatrix(true);
    setSynthesisError(null);

    const conditions: { id: string; label: string; dataUrl: string }[] = [
      { id: "Clean", label: "Clean Baseline", dataUrl: preview },
      { id: "NBJ", label: "NBJ (Noise→Blur→JPEG)", dataUrl: synthesizedData.NBJ.finalDataUrl },
      { id: "NJB", label: "NJB (Noise→JPEG→Blur)", dataUrl: synthesizedData.NJB.finalDataUrl },
      { id: "BNJ", label: "BNJ (Blur→Noise→JPEG)", dataUrl: synthesizedData.BNJ.finalDataUrl },
      { id: "BJN", label: "BJN (Blur→JPEG→Noise)", dataUrl: synthesizedData.BJN.finalDataUrl },
      { id: "JNB", label: "JNB (JPEG→Noise→Blur)", dataUrl: synthesizedData.JNB.finalDataUrl },
      { id: "JBN", label: "JBN (JPEG→Blur→Noise)", dataUrl: synthesizedData.JBN.finalDataUrl },
    ];

    const matrix: Record<string, Record<string, ModelPredictionResult>> = inferenceMatrix ? { ...inferenceMatrix } : {};

    try {
      for (let i = 0; i < conditions.length; i++) {
        const cond = conditions[i];
        if (matrix[cond.id]) continue; // Cached

        setInferenceProgress({
          current: i + 1,
          total: conditions.length,
          label: `Evaluating ${cond.label}...`,
        });

        const res = await fetch("/api/inference", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: cond.dataUrl,
            model: selectedArchitecture,
            corruption: "none",
            severity: 1,
          }),
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || `Inference failed on ${cond.label}`);
        }

        const apiResult = data.result;
        if (apiResult && apiResult.status === "success" && Array.isArray(apiResult.results)) {
          const modelMap: Record<string, ModelPredictionResult> = {};
          for (const m of apiResult.results) {
            modelMap[m.model] = {
              model: m.model,
              prediction: m.prediction,
              confidence: m.confidence,
              top_k: m.top_k,
            };
          }
          matrix[cond.id] = modelMap;
          setInferenceMatrix({ ...matrix });
        }
      }
    } catch (err: unknown) {
      console.error("Inference matrix error:", err);
      setSynthesisError(
        err instanceof Error
          ? err.message
          : "Inference queue encountered an error. Partial results are preserved."
      );
    } finally {
      setIsInferringMatrix(false);
      setInferenceProgress(null);
    }
  };

  // Export Reproducible Research Run Passport (JSON)
  const handleExportJson = async () => {
    if (!synthesizedData) return;
    const sha256 = await computeSha256(preview);
    const levelSpec = FROZEN_CALIBRATION_3LEVELS[researchLevel];

    const orders: CompoundSequenceOrder[] = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"];
    const seqRecords: Partial<Record<CompoundSequenceOrder, SequenceRunRecord>> = {};

    for (const ord of orders) {
      const item = synthesizedData[ord];
      const params = levelSpec.sequences[ord];

      const predictionsRecord: Record<string, {
        top1_class: string;
        confidence: number;
        clean_top1_class: string;
        clean_confidence: number;
        prediction_retained: boolean;
        top5: { class: string; probability: number }[];
      }> = {};

      if (inferenceMatrix && inferenceMatrix[ord]) {
        for (const [mName, mPred] of Object.entries(inferenceMatrix[ord])) {
          const cleanPred = inferenceMatrix.Clean ? inferenceMatrix.Clean[mName] : null;
          predictionsRecord[mName] = {
            top1_class: mPred.prediction,
            confidence: mPred.confidence,
            clean_top1_class: cleanPred ? cleanPred.prediction : "N/A",
            clean_confidence: cleanPred ? cleanPred.confidence : 0,
            prediction_retained: cleanPred ? cleanPred.prediction.toLowerCase() === mPred.prediction.toLowerCase() : false,
            top5: mPred.top_k || [],
          };
        }
      }

      seqRecords[ord] = {
        sequence: ord,
        final_operator: params.final_op,
        parameters: {
          noise_std: params.noise_std,
          blur_sigma: params.blur_sigma,
          jpeg_quality: params.jpeg_quality,
        },
        target_psnr: levelSpec.target_psnr,
        target_ssim: levelSpec.target_ssim,
        achieved_psnr: item.fingerprints.psnr_db,
        achieved_ssim: item.fingerprints.mean_ssim,
        psnr_error: item.psnrErr,
        ssim_error: item.ssimErr,
        tolerance_pass: item.withinTol,
        fingerprints: item.fingerprints,
        predictions: Object.keys(predictionsRecord).length > 0 ? predictionsRecord : undefined,
      };
    }

    const passport: ResearchPassport = {
      passport_id: `RL-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
      timestamp_utc: new Date().toISOString(),
      source_image: {
        filename: sourceMeta?.name || "uploaded_sample.jpg",
        dimensions: sourceMeta?.dimensions || "256x256",
        format: sourceMeta?.format || "JPEG",
        sha256,
      },
      research_level: {
        key: researchLevel,
        name: levelSpec.name,
        target_psnr: levelSpec.target_psnr,
        target_ssim: levelSpec.target_ssim,
        prespecified_psnr_tolerance_db: 0.75,
        prespecified_ssim_tolerance: 0.035,
      },
      sequences: seqRecords as Record<CompoundSequenceOrder, SequenceRunRecord>,
      scientific_disclosures: {
        protocol: "RobustLens Matched-Fidelity Confirmatory Protocol (November 2026)",
        exploratory_demonstration: "Parameters were frozen on Imagenette (N=50). Results on user-uploaded images are exploratory single-image demonstrations, not population statistical claims.",
        no_ground_truth_accuracy_claim: "Without verified benchmark labels, model outputs reflect classification agreement and confidence shift relative to the clean baseline, not true accuracy or margin.",
        strictly_frozen_parameters: "Calibration parameter triples are strictly frozen and never retuned per image.",
      },
      system_environment: {
        framework: "Next.js 15 App Router",
        frontend_version: "2.1.0",
        inference_backend: "Hugging Face ZeroGPU (NVIDIA A10G / Gradio Client)",
      },
    };

    const jsonContent = buildResearchPassportJson(passport);
    downloadFile(jsonContent, `robustlens_passport_${researchLevel}_${Date.now()}.json`, "application/json");
  };

  // Export Reproducible Research Run Passport (CSV)
  const handleExportCsv = async () => {
    if (!synthesizedData) return;
    const sha256 = await computeSha256(preview);
    const levelSpec = FROZEN_CALIBRATION_3LEVELS[researchLevel];

    const orders: CompoundSequenceOrder[] = ["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"];
    const seqRecords: Partial<Record<CompoundSequenceOrder, SequenceRunRecord>> = {};

    for (const ord of orders) {
      const item = synthesizedData[ord];
      const params = levelSpec.sequences[ord];

      const predictionsRecord: Record<string, {
        top1_class: string;
        confidence: number;
        clean_top1_class: string;
        clean_confidence: number;
        prediction_retained: boolean;
        top5: { class: string; probability: number }[];
      }> = {};

      if (inferenceMatrix && inferenceMatrix[ord]) {
        for (const [mName, mPred] of Object.entries(inferenceMatrix[ord])) {
          const cleanPred = inferenceMatrix.Clean ? inferenceMatrix.Clean[mName] : null;
          predictionsRecord[mName] = {
            top1_class: mPred.prediction,
            confidence: mPred.confidence,
            clean_top1_class: cleanPred ? cleanPred.prediction : "N/A",
            clean_confidence: cleanPred ? cleanPred.confidence : 0,
            prediction_retained: cleanPred ? cleanPred.prediction.toLowerCase() === mPred.prediction.toLowerCase() : false,
            top5: mPred.top_k || [],
          };
        }
      }

      seqRecords[ord] = {
        sequence: ord,
        final_operator: params.final_op,
        parameters: {
          noise_std: params.noise_std,
          blur_sigma: params.blur_sigma,
          jpeg_quality: params.jpeg_quality,
        },
        target_psnr: levelSpec.target_psnr,
        target_ssim: levelSpec.target_ssim,
        achieved_psnr: item.fingerprints.psnr_db,
        achieved_ssim: item.fingerprints.mean_ssim,
        psnr_error: item.psnrErr,
        ssim_error: item.ssimErr,
        tolerance_pass: item.withinTol,
        fingerprints: item.fingerprints,
        predictions: Object.keys(predictionsRecord).length > 0 ? predictionsRecord : undefined,
      };
    }

    const passport: ResearchPassport = {
      passport_id: `RL-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
      timestamp_utc: new Date().toISOString(),
      source_image: {
        filename: sourceMeta?.name || "uploaded_sample.jpg",
        dimensions: sourceMeta?.dimensions || "256x256",
        format: sourceMeta?.format || "JPEG",
        sha256,
      },
      research_level: {
        key: researchLevel,
        name: levelSpec.name,
        target_psnr: levelSpec.target_psnr,
        target_ssim: levelSpec.target_ssim,
        prespecified_psnr_tolerance_db: 0.75,
        prespecified_ssim_tolerance: 0.035,
      },
      sequences: seqRecords as Record<CompoundSequenceOrder, SequenceRunRecord>,
      scientific_disclosures: {
        protocol: "RobustLens Matched-Fidelity Confirmatory Protocol (November 2026)",
        exploratory_demonstration: "Parameters were frozen on Imagenette (N=50).",
        no_ground_truth_accuracy_claim: "Without verified labels, outputs reflect transitions.",
        strictly_frozen_parameters: "Parameters are strictly frozen.",
      },
      system_environment: {
        framework: "Next.js 15 App Router",
        frontend_version: "2.1.0",
        inference_backend: "Hugging Face ZeroGPU",
      },
    };

    const csvContent = buildResearchPassportCsv(passport);
    downloadFile(csvContent, `robustlens_passport_${researchLevel}_${Date.now()}.csv`, "text/csv;charset=utf-8;");
  };

  const activeSeqData = synthesizedData ? synthesizedData[microscopeSequence] : null;

  return (
    <div className="flex flex-col gap-8">
      {/* ========================================================================= */}
      {/* PRIORITY 0: CLEAN SOURCE IMAGE PANEL (INSTANT & PERSISTENT)               */}
      {/* ========================================================================= */}
      <div className="bg-card border-2 border-primary/30 rounded-3xl p-6 shadow-xl relative overflow-hidden">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4 pb-4 border-b border-border/50">
          <div className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-xs font-black uppercase tracking-widest text-primary">
              CLEAN SOURCE IMAGE
            </span>
            <span className="text-[10px] bg-emerald-500/10 text-emerald-500 font-bold px-2 py-0.5 rounded-full border border-emerald-500/20">
              Loaded & Verified
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              className="text-xs cursor-pointer font-bold"
              onPress={onChangeImage}
            >
              Change Image
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="text-xs cursor-pointer text-destructive hover:bg-destructive/10 border-destructive/30"
              onPress={onRemoveImage}
            >
              Remove Image
            </Button>
          </div>
        </div>

        <div className="grid sm:grid-cols-4 gap-6 items-center">
          {/* Actual Image Thumbnail — IMMEDIATELY RENDERED WITHOUT GPU/SYNTHESIS */}
          <div className="sm:col-span-1 aspect-square max-w-[180px] mx-auto sm:mx-0 w-full rounded-2xl overflow-hidden border bg-muted/40 relative flex items-center justify-center shadow-inner group">
            <img
              src={preview}
              alt="Clean source preview"
              className="w-full h-full object-contain p-2 transition-transform duration-300 group-hover:scale-105"
            />
          </div>

          {/* Metadata Chips & Info */}
          <div className="sm:col-span-3 space-y-3">
            <div>
              <h4 className="text-lg font-black tracking-tight text-foreground truncate">
                {sourceMeta?.name || "Source Image Baseline"}
              </h4>
              <p className="text-xs text-muted-foreground mt-0.5">
                Active source baseline for 6-way compound permutation synthesis and physical distortion analysis.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2 pt-1 font-mono text-xs">
              <span className="px-3 py-1 rounded-xl bg-muted/50 border border-border/60 text-foreground font-semibold flex items-center gap-1.5">
                <span className="text-muted-foreground uppercase text-[10px] font-bold">Dimensions:</span>
                {sourceMeta?.dimensions || "Calculating..."}
              </span>
              <span className="px-3 py-1 rounded-xl bg-muted/50 border border-border/60 text-foreground font-semibold flex items-center gap-1.5">
                <span className="text-muted-foreground uppercase text-[10px] font-bold">Format:</span>
                {sourceMeta?.format || "JPEG"}
              </span>
              <span className="px-3 py-1 rounded-xl bg-muted/50 border border-border/60 text-foreground font-semibold flex items-center gap-1.5">
                <span className="text-muted-foreground uppercase text-[10px] font-bold">Size:</span>
                {sourceMeta?.size || "Cached"}
              </span>
              {sampleGroundTruth && (
                <span className="px-3 py-1 rounded-xl bg-primary/10 border border-primary/30 text-primary font-bold flex items-center gap-1.5">
                  <span className="uppercase text-[10px]">Ground Truth:</span>
                  {sampleGroundTruth}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Synthesis Error Notice */}
      {synthesisError && (
        <div className="p-4 rounded-2xl bg-destructive/10 border border-destructive/30 text-destructive text-sm font-medium">
          {synthesisError}
        </div>
      )}

      {/* Research Settings & Synthesis Trigger */}
      <div className="bg-card border border-border/60 rounded-3xl p-8 shadow-xl">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-6 pb-6 border-b border-border/50">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-bold uppercase tracking-wider mb-2">
              <FlaskConical className="w-3.5 h-3.5" /> November 2026 Protocol
            </div>
            <h3 className="text-2xl font-black tracking-tight">
              Matched-Fidelity Six-Order Research Lab
            </h3>
            <p className="text-sm text-muted-foreground mt-1 max-w-2xl">
              Synthesizes all six compound sequences (NBJ, NJB, BNJ, BJN, JNB, JBN) using strictly frozen parameter triples calibrated on joint common support.
            </p>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-6 items-end">
          {/* Research Level Selector */}
          <div className="md:col-span-2 space-y-3">
            <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground block">
              Calibrated Research Level (Joint Target)
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
                    setSynthesizedData(null);
                    setInferenceMatrix(null);
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
              <Select value={selectedArchitecture} onValueChange={setSelectedArchitecture}>
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
              isDisabled={isSynthesizing}
              onPress={handleSynthesize}
            >
              {isSynthesizing ? (
                "Synthesizing 6 Permutations..."
              ) : (
                <>
                  Synthesize 6 Permutations <ArrowRight className="ml-1.5 w-4 h-4" />
                </>
              )}
            </Button>
          </div>
        </div>

        <div className="mt-6 pt-4 border-t border-border/50 flex flex-wrap items-center justify-between text-xs text-muted-foreground gap-2">
          <span>
            Prespecified Tolerances: <strong>±0.75 dB PSNR</strong> and <strong>±0.035 SSIM</strong>
          </span>
          <span className="font-mono text-[11px]">
            {FROZEN_CALIBRATION_3LEVELS[researchLevel].description}
          </span>
        </div>
      </div>

      {/* Synthesis In Progress */}
      {isSynthesizing && (
        <div className="p-12 rounded-3xl bg-card border border-border/60 text-center shadow-lg">
          <Activity className="w-10 h-10 text-primary animate-pulse mx-auto mb-4" />
          <h4 className="text-xl font-bold">Applying 3-Stage Sequential Pipelines in Canvas...</h4>
          <p className="text-sm text-muted-foreground mt-2 max-w-lg mx-auto">
            Computing intermediate transformations, 2D FFT radial power bands, Sobel gradient changes, 8×8 block discontinuities, and local SSIM maps.
          </p>
        </div>
      )}

      {/* ========================================================================= */}
      {/* SYNTHESIZED RESEARCH SUITE (PRIORITIES 1, 2, 3, 4)                        */}
      {/* ========================================================================= */}
      {synthesizedData && !isSynthesizing && (
        <div className="space-y-12">
          {/* Action Bar: Export Run Passport & Run Inference */}
          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-md flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-primary" />
                <span className="text-xs font-bold uppercase tracking-wider text-primary">
                  Evaluation Ready: {FROZEN_CALIBRATION_3LEVELS[researchLevel].name}
                </span>
              </div>
              <h4 className="text-2xl font-black mt-1">Six-Order Research Suite</h4>
              <p className="text-xs text-muted-foreground mt-0.5">
                Inspect step-by-step transformations, manuscript fingerprints, neural architecture predictions, and export run passport.
              </p>
            </div>

            {/* Passport Export Buttons (Priority 4) */}
            <div className="flex flex-wrap items-center gap-2.5">
              <Button
                variant="outline"
                size="sm"
                className="text-xs font-bold cursor-pointer"
                onPress={handleExportJson}
              >
                <Download className="w-3.5 h-3.5 mr-1" /> Export Passport (JSON)
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="text-xs font-bold cursor-pointer"
                onPress={handleExportCsv}
              >
                <Download className="w-3.5 h-3.5 mr-1" /> Export Passport (CSV)
              </Button>
              <Button
                variant="primary"
                size="sm"
                className="text-xs font-bold cursor-pointer shadow-md"
                isDisabled={isInferringMatrix}
                onPress={handleRunInferenceMatrix}
              >
                <Cpu className="w-3.5 h-3.5 mr-1" />
                {isInferringMatrix ? "Evaluating GPU Models..." : "Run Neural Predictions"}
              </Button>
            </div>
          </div>

          {/* Inference Progress Banner */}
          {isInferringMatrix && inferenceProgress && (
            <div className="p-6 rounded-3xl bg-primary/10 border border-primary/30 flex flex-col gap-2">
              <div className="flex justify-between items-center text-xs font-bold">
                <span className="text-primary flex items-center gap-2">
                  <Activity className="w-4 h-4 animate-pulse" /> {inferenceProgress.label}
                </span>
                <span className="font-mono text-muted-foreground">
                  {inferenceProgress.current} / {inferenceProgress.total} requests
                </span>
              </div>
              <Progress
                aria-label="Inference progress"
                value={(inferenceProgress.current / inferenceProgress.total) * 100}
                className="w-full"
                color="accent"
                size="sm"
              />
            </div>
          )}

          {/* ===================================================================== */}
          {/* PRIORITY 1: SIX-ORDER SEQUENCE MICROSCOPE                             */}
          {/* ===================================================================== */}
          <div className="bg-card border border-border/60 rounded-3xl p-8 shadow-xl space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-border/50">
              <div>
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 text-blue-500 text-xs font-bold uppercase tracking-wider mb-1">
                  <Eye className="w-3.5 h-3.5" /> Priority 1 Inspection
                </div>
                <h4 className="text-2xl font-black tracking-tight">Sequence Transformation Microscope</h4>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Step through individual transformation stages (Clean → Op 1 → Op 2 → Op 3) for each of the six order permutations.
                </p>
              </div>

              {/* Sequence Selector Pills */}
              <div className="flex flex-wrap items-center gap-1.5 p-1 rounded-2xl bg-muted/60 border border-border/60">
                {(["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"] as CompoundSequenceOrder[]).map((ord) => (
                  <button
                    key={ord}
                    onClick={() => {
                      setMicroscopeSequence(ord);
                      setActiveStageIndex(3);
                    }}
                    className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all cursor-pointer ${
                      microscopeSequence === ord
                        ? "bg-primary text-primary-foreground shadow-sm"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {ord}
                  </button>
                ))}
              </div>
            </div>

            {activeSeqData && (
              <div className="grid lg:grid-cols-12 gap-8 items-start">
                {/* Microscope Viewport (Left 7 Cols) */}
                <div className="lg:col-span-7 space-y-4">
                  <div className="relative aspect-square max-w-[500px] mx-auto rounded-3xl overflow-hidden border bg-muted/30 shadow-inner flex items-center justify-center">
                    {/* Active Stage Image */}
                    {comparisonMode === "clean" ? (
                      <img
                        src={preview}
                        alt="Clean source"
                        className="w-full h-full object-contain p-2"
                      />
                    ) : (
                      <img
                        src={activeSeqData.stages[activeStageIndex].dataUrl}
                        alt={activeSeqData.stages[activeStageIndex].label}
                        className="w-full h-full object-contain p-2"
                      />
                    )}

                    {/* Stage Overlay Badge */}
                    <div className="absolute top-4 left-4 bg-background/85 backdrop-blur px-3 py-1 rounded-full text-xs font-bold border shadow-xs flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-primary" />
                      {comparisonMode === "clean"
                        ? "Clean Source Baseline"
                        : activeSeqData.stages[activeStageIndex].label}
                    </div>

                    {/* Zoom / Fullscreen Button */}
                    <button
                      onClick={() =>
                        setInspectModalImage(
                          comparisonMode === "clean"
                            ? preview
                            : activeSeqData.stages[activeStageIndex].dataUrl
                        )
                      }
                      className="absolute top-4 right-4 bg-background/80 hover:bg-background backdrop-blur p-2 rounded-full border shadow-xs cursor-pointer text-muted-foreground hover:text-foreground"
                      title="Inspect Full Resolution"
                    >
                      <Maximize2 className="w-4 h-4" />
                    </button>
                  </div>

                  {/* Comparison Controls */}
                  <div className="flex items-center justify-center gap-2 text-xs">
                    <span className="text-muted-foreground font-semibold">View Mode:</span>
                    <button
                      onClick={() => setComparisonMode("current")}
                      className={`px-3 py-1 rounded-lg font-bold cursor-pointer transition-all ${
                        comparisonMode === "current"
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted text-muted-foreground hover:bg-muted/80"
                      }`}
                    >
                      Active Stage
                    </button>
                    <button
                      onClick={() => setComparisonMode("clean")}
                      className={`px-3 py-1 rounded-lg font-bold cursor-pointer transition-all ${
                        comparisonMode === "clean"
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted text-muted-foreground hover:bg-muted/80"
                      }`}
                    >
                      Clean Baseline
                    </button>
                  </div>
                </div>

                {/* Stage Stepper & Operator Analysis (Right 5 Cols) */}
                <div className="lg:col-span-5 space-y-4">
                  <div className="space-y-2">
                    <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground block">
                      Transformation Sequence Pipeline
                    </span>
                    <h5 className="text-xl font-black">
                      {microscopeSequence}: {activeSeqData.stages.slice(1).map((s) => s.operatorName).join(" → ")}
                    </h5>
                  </div>

                  {/* 4 Interactive Stage Steps */}
                  <div className="space-y-2.5">
                    {activeSeqData.stages.map((stg) => (
                      <div
                        key={stg.stageIndex}
                        onClick={() => {
                          setActiveStageIndex(stg.stageIndex);
                          setComparisonMode("current");
                        }}
                        className={`p-3.5 rounded-2xl border transition-all cursor-pointer flex items-center justify-between ${
                          activeStageIndex === stg.stageIndex && comparisonMode !== "clean"
                            ? "bg-primary/10 border-primary ring-1 ring-primary shadow-sm"
                            : "bg-muted/20 border-border/50 hover:bg-muted/40"
                        }`}
                      >
                        <div className="flex items-center gap-3">
                          <div
                            className={`w-7 h-7 rounded-xl flex items-center justify-center text-xs font-bold ${
                              activeStageIndex === stg.stageIndex
                                ? "bg-primary text-primary-foreground"
                                : "bg-muted text-muted-foreground"
                            }`}
                          >
                            {stg.stageIndex}
                          </div>
                          <div>
                            <span className="text-xs font-bold block text-foreground">
                              {stg.label}
                            </span>
                            <span className="text-[11px] text-muted-foreground font-mono">
                              {stg.parameterDescription}
                            </span>
                          </div>
                        </div>

                        <div className="w-12 h-12 rounded-xl overflow-hidden border bg-background shrink-0">
                          <img
                            src={stg.dataUrl}
                            alt={stg.label}
                            className="w-full h-full object-cover"
                          />
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Physical Fidelity Summary for this Sequence */}
                  <div className="p-4 rounded-2xl bg-muted/30 border border-border/40 text-xs space-y-2">
                    <span className="font-bold text-foreground block uppercase text-[10px] tracking-wider">
                      Measured Physical Fidelity ({microscopeSequence})
                    </span>
                    <div className="grid grid-cols-2 gap-2 font-mono">
                      <div className="p-2 rounded-xl bg-blue-500/10 border border-blue-500/20 text-center">
                        <span className="text-[9px] uppercase font-bold text-muted-foreground block">Achieved PSNR</span>
                        <strong className="text-blue-500 text-sm">{activeSeqData.fingerprints.psnr_db.toFixed(2)} dB</strong>
                      </div>
                      <div className="p-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-center">
                        <span className="text-[9px] uppercase font-bold text-muted-foreground block">Achieved SSIM</span>
                        <strong className="text-emerald-500 text-sm">{activeSeqData.fingerprints.mean_ssim.toFixed(3)}</strong>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* ===================================================================== */}
          {/* PRIORITY 2: STRUCTURAL DISTORTION FINGERPRINT MICROSCOPE             */}
          {/* ===================================================================== */}
          <div className="bg-card border border-border/60 rounded-3xl p-8 shadow-xl space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-border/50">
              <div>
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-violet-500/10 text-violet-500 text-xs font-bold uppercase tracking-wider mb-1">
                  <BarChart3 className="w-3.5 h-3.5" /> Priority 2 Analysis
                </div>
                <h4 className="text-2xl font-black tracking-tight">Structural Distortion Fingerprint Microscope</h4>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Inspect the five manuscript fingerprints computed via canonical 2D radial FFT, Sobel convolution, and DCT block gradients.
                </p>
              </div>

              {/* Fingerprint Visual Map Tabs */}
              <div className="flex flex-wrap items-center gap-1.5 p-1 rounded-2xl bg-muted/60 border border-border/60">
                {(
                  [
                    { id: "fourier", label: "Fourier Power Spectrum" },
                    { id: "sobel", label: "Sobel Edge Map" },
                    { id: "ssim", label: "Local SSIM Heatmap" },
                    { id: "jpeg", label: "8×8 Block Grid" },
                  ] as const
                ).map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveMapTab(tab.id)}
                    className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all cursor-pointer ${
                      activeMapTab === tab.id
                        ? "bg-violet-600 text-white shadow-sm"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>

            {activeSeqData && (
              <div className="grid lg:grid-cols-12 gap-8 items-center">
                {/* Visual Map Viewport */}
                <div className="lg:col-span-5 flex flex-col items-center">
                  <div className="relative aspect-square max-w-[380px] w-full rounded-2xl overflow-hidden border bg-background shadow-inner flex items-center justify-center">
                    <img
                      src={
                        activeMapTab === "fourier"
                          ? activeSeqData.visualMaps.fourierSpectrumUrl
                          : activeMapTab === "sobel"
                          ? activeSeqData.visualMaps.sobelEdgeMapUrl
                          : activeMapTab === "ssim"
                          ? activeSeqData.visualMaps.ssimHeatmapUrl
                          : activeSeqData.visualMaps.jpegOverlayUrl
                      }
                      alt={activeMapTab}
                      className="w-full h-full object-contain"
                    />
                  </div>
                  <span className="text-[11px] text-muted-foreground mt-2 text-center">
                    {activeMapTab === "fourier" && "2D radial FFT spectrum: Cyan ring (r < 0.15) = Low band, Rose ring (r >= 0.45) = High band"}
                    {activeMapTab === "sobel" && "Sobel gradient magnitude map highlighting preserved edges vs noise confounds"}
                    {activeMapTab === "ssim" && "Spatial SSIM heatmap: Red/amber = severe structural distortion, emerald = preserved structure"}
                    {activeMapTab === "jpeg" && "8×8 DCT grid boundary step discontinuity overlay"}
                  </span>
                </div>

                {/* 5 Fingerprint Cards */}
                <div className="lg:col-span-7 grid sm:grid-cols-2 gap-3 text-xs">
                  <div className="p-3.5 rounded-2xl bg-muted/20 border border-border/40 space-y-1">
                    <span className="text-[10px] uppercase font-bold text-muted-foreground block">
                      1. Fourier Low-Freq Energy (E_low)
                    </span>
                    <span className="text-xl font-black text-blue-500 font-mono block">
                      {activeSeqData.fingerprints.fft_energy_low.toFixed(4)}
                    </span>
                    <span className="text-[10px] text-muted-foreground block">
                      Energy in r &lt; 0.15 radial disk
                    </span>
                  </div>

                  <div className="p-3.5 rounded-2xl bg-muted/20 border border-border/40 space-y-1">
                    <span className="text-[10px] uppercase font-bold text-muted-foreground block">
                      2. Fourier High-Freq Energy (E_high)
                    </span>
                    <span className="text-xl font-black text-rose-500 font-mono block">
                      {activeSeqData.fingerprints.fft_energy_high.toFixed(4)}
                    </span>
                    <span className="text-[10px] text-muted-foreground block">
                      Energy in r ≥ 0.45 outer annulus
                    </span>
                  </div>

                  <div className="p-3.5 rounded-2xl bg-muted/20 border border-border/40 space-y-1">
                    <span className="text-[10px] uppercase font-bold text-muted-foreground block">
                      3. JPEG 8×8 Block Discontinuity (B_JPEG)
                    </span>
                    <span className="text-xl font-black text-amber-500 font-mono block">
                      {activeSeqData.fingerprints.jpeg_block_artifact.toFixed(4)}x
                    </span>
                    <span className="text-[10px] text-muted-foreground block">
                      Grid boundary-to-interior step ratio
                    </span>
                  </div>

                  <div className="p-3.5 rounded-2xl bg-muted/20 border border-border/40 space-y-1">
                    <span className="text-[10px] uppercase font-bold text-muted-foreground block">
                      4. Sobel Edge Energy Delta (ΔE_edge)
                    </span>
                    <span className="text-xl font-black text-violet-500 font-mono block">
                      {activeSeqData.fingerprints.delta_edge_energy >= 0 ? "+" : ""}
                      {activeSeqData.fingerprints.delta_edge_energy.toFixed(4)}
                    </span>
                    <span className="text-[10px] text-muted-foreground block">
                      Relative gradient change vs clean
                    </span>
                  </div>

                  <div className="p-3.5 rounded-2xl bg-muted/20 border border-border/40 space-y-1 sm:col-span-2">
                    <span className="text-[10px] uppercase font-bold text-muted-foreground block">
                      5. Local SSIM Spatial Deviation (σ_local_SSIM)
                    </span>
                    <span className="text-xl font-black text-emerald-500 font-mono block">
                      {activeSeqData.fingerprints.std_local_ssim.toFixed(4)}
                    </span>
                    <span className="text-[10px] text-muted-foreground block">
                      Standard deviation of 8×8 windowed SSIM map across image plane
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Compact Metric Comparison Table for All 6 Sequences */}
            <div className="pt-4 border-t border-border/50 space-y-3">
              <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground block">
                Cross-Sequence Fingerprint Comparison Matrix
              </span>
              <div className="overflow-x-auto rounded-2xl border border-border/60">
                <table className="w-full text-xs text-left">
                  <thead className="bg-muted/50 border-b border-border/60 text-muted-foreground font-semibold">
                    <tr>
                      <th className="p-3">Sequence</th>
                      <th className="p-3">PSNR (dB)</th>
                      <th className="p-3">SSIM</th>
                      <th className="p-3">E_low</th>
                      <th className="p-3">E_high</th>
                      <th className="p-3">B_JPEG</th>
                      <th className="p-3">ΔE_edge</th>
                      <th className="p-3">σ_local_SSIM</th>
                      <th className="p-3">Tolerance</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/40 font-mono">
                    {(["NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"] as CompoundSequenceOrder[]).map((ord) => {
                      const item = synthesizedData[ord];
                      const isSelected = ord === microscopeSequence;
                      return (
                        <tr
                          key={ord}
                          onClick={() => setMicroscopeSequence(ord)}
                          className={`cursor-pointer transition-colors ${
                            isSelected ? "bg-primary/10 font-bold" : "hover:bg-muted/30"
                          }`}
                        >
                          <td className="p-3 font-sans font-bold text-foreground flex items-center gap-1.5">
                            {ord}
                            {isSelected && <span className="w-1.5 h-1.5 rounded-full bg-primary" />}
                          </td>
                          <td className="p-3 text-blue-500">{item.fingerprints.psnr_db.toFixed(2)}</td>
                          <td className="p-3 text-emerald-500">{item.fingerprints.mean_ssim.toFixed(3)}</td>
                          <td className="p-3">{item.fingerprints.fft_energy_low.toFixed(4)}</td>
                          <td className="p-3">{item.fingerprints.fft_energy_high.toFixed(4)}</td>
                          <td className="p-3">{item.fingerprints.jpeg_block_artifact.toFixed(3)}x</td>
                          <td className="p-3">
                            {item.fingerprints.delta_edge_energy >= 0 ? "+" : ""}
                            {item.fingerprints.delta_edge_energy.toFixed(3)}
                          </td>
                          <td className="p-3">{item.fingerprints.std_local_ssim.toFixed(4)}</td>
                          <td className="p-3">
                            <span
                              className={`px-2 py-0.5 rounded-full text-[10px] font-bold font-sans ${
                                item.withinTol
                                  ? "bg-emerald-500/10 text-emerald-500"
                                  : "bg-amber-500/10 text-amber-500"
                              }`}
                            >
                              {item.withinTol ? "Pass" : "Δ Tol"}
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

          {/* ===================================================================== */}
          {/* PRIORITY 3: FOUR-ARCHITECTURE PREDICTION COMPARISON MATRIX            */}
          {/* ===================================================================== */}
          <div className="bg-card border border-border/60 rounded-3xl p-8 shadow-xl space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-border/50">
              <div>
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-500 text-xs font-bold uppercase tracking-wider mb-1">
                  <Cpu className="w-3.5 h-3.5" /> Priority 3 Evaluation
                </div>
                <h4 className="text-2xl font-black tracking-tight">Four-Architecture Prediction Comparison Matrix</h4>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Evaluates model prediction churn and softmax score transition relative to the clean baseline across all six compound sequences.
                </p>
              </div>

              {!inferenceMatrix && (
                <Button
                  variant="primary"
                  size="sm"
                  isDisabled={isInferringMatrix}
                  onPress={handleRunInferenceMatrix}
                  className="font-bold cursor-pointer"
                >
                  {isInferringMatrix ? "Running Neural Inference..." : "Evaluate All 4 Architectures"}
                </Button>
              )}
            </div>

            {inferenceMatrix ? (
              <div className="space-y-4">
                <div className="overflow-x-auto rounded-2xl border border-border/60">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-muted/50 border-b border-border/60 text-muted-foreground font-semibold">
                      <tr>
                        <th className="p-3">Condition</th>
                        <th className="p-3">ResNet-50</th>
                        <th className="p-3">ConvNeXt-Tiny</th>
                        <th className="p-3">ViT-Base</th>
                        <th className="p-3">Swin-Tiny</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border/40">
                      {(["Clean", "NBJ", "NJB", "BNJ", "BJN", "JNB", "JBN"] as const).map((condId) => {
                        const rowModels = inferenceMatrix[condId];
                        const cleanRow = inferenceMatrix.Clean;

                        return (
                          <tr key={condId} className={condId === "Clean" ? "bg-muted/20 font-bold" : ""}>
                            <td className="p-3 font-bold text-foreground">
                              {condId === "Clean" ? (
                                <span className="text-primary">Clean Baseline</span>
                              ) : (
                                <span>{condId}</span>
                              )}
                            </td>

                            {["ResNet-50", "ConvNeXt-Tiny", "ViT-Base", "Swin-Tiny"].map((mName) => {
                              const pred = rowModels ? rowModels[mName] : null;
                              const cleanPred = cleanRow ? cleanRow[mName] : null;
                              if (!pred) {
                                return <td key={mName} className="p-3 text-muted-foreground">Pending</td>;
                              }

                              const isRetained = cleanPred
                                ? cleanPred.prediction.toLowerCase() === pred.prediction.toLowerCase()
                                : true;
                              const confPct = Math.round(pred.confidence * 100);
                              const predKey = `${condId}_${mName}`;
                              const isExpanded = expandedPredKeys[predKey];

                              return (
                                <td key={mName} className="p-3 align-top">
                                  <div className="space-y-1">
                                    <div className="flex items-center justify-between gap-1">
                                      <span className="font-bold text-foreground capitalize truncate max-w-[130px]" title={pred.prediction}>
                                        {pred.prediction}
                                      </span>
                                      <span className="font-mono text-[10px] text-muted-foreground">
                                        {confPct}%
                                      </span>
                                    </div>

                                    {condId !== "Clean" && (
                                      <span
                                        className={`px-1.5 py-0.5 rounded text-[10px] font-bold block w-fit ${
                                          isRetained
                                            ? "bg-emerald-500/10 text-emerald-500"
                                            : "bg-rose-500/10 text-rose-500"
                                        }`}
                                      >
                                        {isRetained ? "Retained" : "Shifted"}
                                      </span>
                                    )}

                                    {/* Top-5 toggle */}
                                    {pred.top_k && pred.top_k.length > 0 && (
                                      <div className="pt-1">
                                        <button
                                          onClick={() =>
                                            setExpandedPredKeys((prev) => ({
                                              ...prev,
                                              [predKey]: !prev[predKey],
                                            }))
                                          }
                                          className="text-[10px] text-muted-foreground hover:text-foreground flex items-center gap-0.5 cursor-pointer"
                                        >
                                          <span>Top 5</span>
                                          {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                                        </button>

                                        {isExpanded && (
                                          <div className="mt-1 space-y-0.5 text-[10px] text-muted-foreground font-mono">
                                            {pred.top_k.map((k, kIdx) => (
                                              <div key={kIdx} className="flex justify-between">
                                                <span className="capitalize truncate max-w-[100px]">{k.class}</span>
                                                <span>{(k.probability * 100).toFixed(0)}%</span>
                                              </div>
                                            ))}
                                          </div>
                                        )}
                                      </div>
                                    )}
                                  </div>
                                </td>
                              );
                            })}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <p className="text-[11px] text-muted-foreground leading-relaxed">
                  <strong>Scientific Methodology Note:</strong> Softmax scores represent uncalibrated output distributions and are not true probabilities. Without known ground-truth labels, prediction shifts indicate operational churn and stability changes across permutations, not ground-truth classification error.
                </p>
              </div>
            ) : (
              <div className="p-8 rounded-2xl bg-muted/20 border border-dashed border-border/60 text-center">
                <Cpu className="w-8 h-8 text-muted-foreground mx-auto mb-2 opacity-50" />
                <p className="text-sm font-semibold text-foreground">Neural Predictions Not Yet Evaluated</p>
                <p className="text-xs text-muted-foreground mt-1 max-w-md mx-auto">
                  Click the button above to execute live Hugging Face ZeroGPU inference across all four vision architectures for the clean baseline and six sequence outputs.
                </p>
              </div>
            )}
          </div>

          {/* Scientific Integrity Disclosures */}
          <div className="p-6 rounded-3xl bg-card border border-border/60 shadow-lg space-y-3">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-primary" />
              <h4 className="font-extrabold text-sm text-foreground uppercase tracking-wider">
                Audited Research Integrity Disclosures
              </h4>
            </div>
            <div className="grid sm:grid-cols-2 gap-4 text-xs text-muted-foreground">
              <div className="p-3.5 rounded-2xl bg-muted/20 border border-border/40">
                <strong className="text-foreground block mb-1">Exploratory Single-Image Demonstration</strong>
                <p className="leading-relaxed">
                  Calibration parameters are derived on Imagenette (N=50). Results on arbitrary uploaded images demonstrate physical sequence behavior on individual frequency spectra and do not constitute population confirmatory claims.
                </p>
              </div>
              <div className="p-3.5 rounded-2xl bg-muted/20 border border-border/40">
                <strong className="text-foreground block mb-1">Strict Parameter Invariance</strong>
                <p className="leading-relaxed">
                  Parameters are frozen per pre-registration and never post-hoc adjusted per image. Achieved physical fidelity is measured and reported with its deviation from nominal target.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Fullscreen Image Modal */}
      {inspectModalImage && (
        <div
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 cursor-pointer"
          onClick={() => setInspectModalImage(null)}
        >
          <div className="relative max-w-4xl max-h-[90vh] bg-card rounded-3xl p-4 border border-border/60 shadow-2xl overflow-hidden flex flex-col items-center">
            <img
              src={inspectModalImage}
              alt="Full inspection view"
              className="max-h-[80vh] w-auto object-contain rounded-2xl"
            />
            <span className="text-xs text-muted-foreground mt-2">
              Click anywhere to close full inspection view
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
