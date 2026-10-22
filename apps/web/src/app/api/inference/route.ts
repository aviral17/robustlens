import { NextResponse } from 'next/server';
import { Client } from "@gradio/client";

// The exact 5 model choices expected by the Gradio inference space:
export const GRADIO_MODEL_CHOICES = [
  "ResNet-50",
  "ConvNeXt-Tiny",
  "ViT-Base",
  "Swin-Tiny",
  "Compare All Models",
] as const;

export type GradioModelChoice = (typeof GRADIO_MODEL_CHOICES)[number];

// Explicit mapping from frontend model IDs, dropdown values, and common variants to exact Gradio choices
export const MODEL_NAME_MAP: Record<string, GradioModelChoice> = {
  // Required frontend model IDs
  "resnet": "ResNet-50",
  "convnext": "ConvNeXt-Tiny",
  "vit": "ViT-Base",
  "swin": "Swin-Tiny",
  "all": "Compare All Models",
  "compare-all": "Compare All Models",
  "compare_all": "Compare All Models",
  "compare all": "Compare All Models",
  "compare all models": "Compare All Models",

  // Exact Gradio choices (case-insensitive)
  "resnet-50": "ResNet-50",
  "convnext-tiny": "ConvNeXt-Tiny",
  "vit-base": "ViT-Base",
  "swin-tiny": "Swin-Tiny",

  // Additional variants found across the frontend
  "resnet50": "ResNet-50",
  "convnext_tiny": "ConvNeXt-Tiny",
  "convnext_t": "ConvNeXt-Tiny",
  "convnext-t": "ConvNeXt-Tiny",
  "vit_base": "ViT-Base",
  "vit_b_16": "ViT-Base",
  "vit-b-16": "ViT-Base",
  "vit-b/16": "ViT-Base",
  "swin_tiny": "Swin-Tiny",
  "swin-t": "Swin-Tiny",
  "swin_t": "Swin-Tiny",
};

/**
 * Resolves a model identifier to an exact Gradio choice.
 * Returns null if the identifier is unsupported.
 */
export function resolveModelChoice(modelInput?: unknown): GradioModelChoice | null {
  if (modelInput === undefined || modelInput === null || modelInput === "") {
    return "Compare All Models";
  }

  if (typeof modelInput !== "string") {
    return null;
  }

  const normalizedKey = modelInput.trim().toLowerCase();
  return MODEL_NAME_MAP[normalizedKey] || null;
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { image, model, corruption, severity } = body;
    
    // Validate image format
    if (!image || typeof image !== "string" || !image.includes(",")) {
      return NextResponse.json(
        { error: "Invalid or missing image. A base64 data URL string is required." },
        { status: 400 }
      );
    }

    // Explicitly validate and resolve model ID at API boundary
    const mappedModel = resolveModelChoice(model);
    if (!mappedModel) {
      return NextResponse.json(
        {
          error: `Unsupported model '${model}'. Valid choices are: 'resnet' (ResNet-50), 'convnext' (ConvNeXt-Tiny), 'vit' (ViT-Base), 'swin' (Swin-Tiny), 'all' (Compare All Models).`,
          supported_models: GRADIO_MODEL_CHOICES,
        },
        { status: 400 }
      );
    }

    const hfEndpoint = process.env.ROBUSTLENS_HF_SPACE_URL;
    if (!hfEndpoint) {
      return NextResponse.json(
        { error: "ROBUSTLENS_HF_SPACE_URL not configured in server environment." },
        { status: 500 }
      );
    }

    // Server-side Hugging Face authentication token
    const rawHfToken = process.env.HF_TOKEN?.trim();
    const hfToken = rawHfToken as `hf_${string}` | undefined;

    // Keep corruption and severity arguments unchanged unless missing
    const normalizedSeverity = Number(severity) || 3;
    const normalizedCorruption = corruption || "gaussian_noise";
    
    // Connect to the Gradio space using server-side token for ZeroGPU quota
    const client = await Client.connect(hfEndpoint, {
      ...(hfToken ? { token: hfToken } : {}),
    });
    
    // Convert base64 image to Blob reliably in Node
    const base64Data = image.split(',')[1];
    const mimeType = image.split(',')[0].split(':')[1].split(';')[0];
    const buffer = Buffer.from(base64Data, 'base64');
    const imageBlob = new Blob([buffer], { type: mimeType });
    
    // Send prediction request with mappedModel
    const result = await client.predict("/run_inference", [
      imageBlob,
      mappedModel,
      normalizedCorruption,
      normalizedSeverity,
    ]);
    
    const resultData = (result as { data: unknown[] }).data[0];
    return NextResponse.json({ result: resultData });
    
  } catch (error: unknown) {
    const rawHfToken = process.env.HF_TOKEN?.trim();
    const rawErrorMessage = error instanceof Error ? error.message : "Internal server error";
    // Ensure HF token is never leaked in server logs or client responses
    const errorMessage = rawHfToken && rawHfToken.length > 5
      ? rawErrorMessage.replaceAll(rawHfToken, "[REDACTED]")
      : rawErrorMessage;

    console.error("Gradio Inference Error:", errorMessage);

    // Detect ZeroGPU quota exhaustion and return clear, actionable explanation
    const isZeroGpuQuotaError = /zerogpu|runs limit|quota|exceeded/i.test(errorMessage);
    if (isZeroGpuQuotaError) {
      return NextResponse.json(
        {
          error: "Hugging Face ZeroGPU quota exhausted. The connected Hugging Face account's compute allowance has reached its limit. Please wait for the quota to reset or upgrade your Hugging Face account for additional GPU compute.",
          details: errorMessage,
        },
        { status: 429 }
      );
    }

    return NextResponse.json({ error: errorMessage }, { status: 500 });
  }
}

