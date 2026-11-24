/**
 * Reproducible Research Run Passport Export Engine
 * 
 * Generates audited, reproducible JSON and CSV experiment passports
 * for individual interactive runs under the RobustLens November 2026 protocol.
 */

import { CompoundSequenceOrder, ResearchLevelKey } from './corruptions';
import { DistortionFingerprints } from './structural_fingerprints';

export interface SequenceRunRecord {
  sequence: CompoundSequenceOrder;
  final_operator: 'Noise' | 'Blur' | 'JPEG';
  parameters: {
    noise_std: number;
    blur_sigma: number;
    jpeg_quality: number;
  };
  target_psnr: number;
  target_ssim: number;
  achieved_psnr: number;
  achieved_ssim: number;
  psnr_error: number;
  ssim_error: number;
  tolerance_pass: boolean;
  fingerprints: DistortionFingerprints;
  predictions?: Record<string, {
    top1_class: string;
    confidence: number;
    clean_top1_class: string;
    clean_confidence: number;
    prediction_retained: boolean;
    top5: { class: string; probability: number }[];
  }>;
}

export interface ResearchPassport {
  passport_id: string;
  timestamp_utc: string;
  source_image: {
    filename: string;
    dimensions: string;
    format: string;
    sha256: string;
  };
  research_level: {
    key: ResearchLevelKey;
    name: string;
    target_psnr: number;
    target_ssim: number;
    prespecified_psnr_tolerance_db: number;
    prespecified_ssim_tolerance: number;
  };
  sequences: Record<CompoundSequenceOrder, SequenceRunRecord>;
  scientific_disclosures: {
    protocol: string;
    exploratory_demonstration: string;
    no_ground_truth_accuracy_claim: string;
    strictly_frozen_parameters: string;
  };
  system_environment: {
    framework: string;
    frontend_version: string;
    inference_backend: string;
  };
}

/**
 * Computes SHA-256 hash of a string or dataUrl using Web Crypto API
 */
export async function computeSha256(dataUrlOrText: string): Promise<string> {
  try {
    const encoder = new TextEncoder();
    const data = encoder.encode(dataUrlOrText);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
  } catch {
    return 'unavailable';
  }
}

/**
 * Builds the comprehensive structured JSON passport
 */
export function buildResearchPassportJson(
  passportData: ResearchPassport
): string {
  return JSON.stringify(passportData, null, 2);
}

/**
 * Builds the comprehensive tabular CSV passport
 */
export function buildResearchPassportCsv(
  passportData: ResearchPassport
): string {
  const headers = [
    'experiment_id',
    'timestamp_utc',
    'image_filename',
    'image_dimensions',
    'image_sha256',
    'research_level',
    'sequence',
    'final_operator',
    'noise_std',
    'blur_sigma_px',
    'jpeg_quality',
    'target_psnr_db',
    'achieved_psnr_db',
    'psnr_deviation_db',
    'target_ssim',
    'achieved_ssim',
    'ssim_deviation',
    'tolerance_pass',
    'fft_energy_low',
    'fft_energy_med',
    'fft_energy_high',
    'jpeg_block_artifact',
    'delta_edge_energy',
    'std_local_ssim',
    'evaluated_architecture',
    'clean_top1_class',
    'corrupted_top1_class',
    'corrupted_confidence_pct',
    'prediction_retained',
  ];

  const rows: string[] = [headers.join(',')];

  const orders: CompoundSequenceOrder[] = ['NBJ', 'NJB', 'BNJ', 'BJN', 'JNB', 'JBN'];
  for (const ord of orders) {
    const item = passportData.sequences[ord];
    if (!item) continue;

    // Check predictions if available
    const models = item.predictions ? Object.keys(item.predictions) : ['none'];
    for (const mName of models) {
      const pred = item.predictions ? item.predictions[mName] : null;

      const rowValues = [
        passportData.passport_id,
        passportData.timestamp_utc,
        `"${passportData.source_image.filename.replace(/"/g, '""')}"`,
        `"${passportData.source_image.dimensions}"`,
        passportData.source_image.sha256.slice(0, 16),
        passportData.research_level.key,
        ord,
        item.final_operator,
        item.parameters.noise_std,
        item.parameters.blur_sigma,
        item.parameters.jpeg_quality,
        item.target_psnr,
        item.achieved_psnr,
        item.psnr_error,
        item.target_ssim,
        item.achieved_ssim,
        item.ssim_error,
        item.tolerance_pass ? 'TRUE' : 'FALSE',
        item.fingerprints.fft_energy_low,
        item.fingerprints.fft_energy_med,
        item.fingerprints.fft_energy_high,
        item.fingerprints.jpeg_block_artifact,
        item.fingerprints.delta_edge_energy,
        item.fingerprints.std_local_ssim,
        `"${mName}"`,
        pred ? `"${pred.clean_top1_class.replace(/"/g, '""')}"` : 'N/A',
        pred ? `"${pred.top1_class.replace(/"/g, '""')}"` : 'N/A',
        pred ? Math.round(pred.confidence * 100) : 'N/A',
        pred ? (pred.prediction_retained ? 'TRUE' : 'FALSE') : 'N/A',
      ];

      rows.push(rowValues.join(','));
    }
  }

  return rows.join('\r\n');
}

/**
 * Triggers a browser file download
 */
export function downloadFile(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
