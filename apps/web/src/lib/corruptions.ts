/**
 * Client-side genuine pixel corruption engine.
 * Matches the mathematical corruption formulations in hf_space/app.py and ml/src/robustlens/corruptions.
 *
 * NOTE ON ARCHITECTURE & PREVENTING DOUBLE-CORRUPTION:
 * - When sending an image for remote live inference to Hugging Face ZeroGPU (/api/inference),
 *   we MUST send the original CLEAN image. The remote space applies the corruption authoritatively.
 * - This module is used exclusively for rendering the genuine pixel-level corrupted visual in the UI,
 *   replacing misleading CSS overlays with real pixel modifications.
 */

export type CorruptionType = 'gaussian_noise' | 'defocus_blur' | 'jpeg_compression' | 'brightness' | 'contrast';

export interface CorruptionOptions {
  type: CorruptionType;
  severity: number; // 1 to 5
}

/**
 * Standard Box-Muller transform for generating Gaussian random variables
 */
function gaussianRandom(mean = 0, stdev = 1): number {
  let u = 1 - Math.random(); // Subtraction to get (0, 1]
  let v = Math.random();
  let z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return z * stdev + mean;
}

/**
 * Loads an image src/data URL into an HTMLImageElement
 */
function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = (err) => reject(new Error('Failed to load image for pixel corruption: ' + err));
    img.src = src;
  });
}

/**
 * Applies genuine pixel corruption to an image and returns a base64 Data URL.
 */
export async function applyClientPixelCorruption(
  imageSrc: string,
  type: CorruptionType,
  severity: number
): Promise<string> {
  const img = await loadImage(imageSrc);
  const canvas = document.createElement('canvas');
  const width = img.naturalWidth || img.width;
  const height = img.naturalHeight || img.height;
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Could not get 2D canvas context');

  const sevIndex = Math.min(Math.max(severity, 1), 5) - 1;

  if (type === 'defocus_blur') {
    // Defocus blur matching hf_space: c = [1, 2, 3, 4, 5]
    const radii = [1.5, 3, 4.5, 6, 8];
    const radius = radii[sevIndex];
    ctx.filter = `blur(${radius}px)`;
    ctx.drawImage(img, 0, 0, width, height);
    return canvas.toDataURL('image/png');
  }

  // Draw clean image first for pixel-based transformations
  ctx.drawImage(img, 0, 0, width, height);

  if (type === 'jpeg_compression') {
    // JPEG quality matching hf_space: c = [80, 65, 50, 35, 20]
    const qualities = [0.80, 0.65, 0.50, 0.35, 0.20];
    const quality = qualities[sevIndex];
    return canvas.toDataURL('image/jpeg', quality);
  }

  const imgData = ctx.getImageData(0, 0, width, height);
  const data = imgData.data;

  if (type === 'gaussian_noise') {
    // Gaussian noise stds matching hf_space: c = [0.04, 0.06, 0.08, 0.09, 0.10]
    const stds = [0.04, 0.06, 0.08, 0.09, 0.10];
    const std = stds[sevIndex] * 255;

    for (let i = 0; i < data.length; i += 4) {
      const noise = gaussianRandom(0, std);
      data[i] = Math.min(255, Math.max(0, data[i] + noise));         // R
      data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + noise)); // G
      data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + noise)); // B
      // Alpha untouched
    }
    ctx.putImageData(imgData, 0, 0);
    return canvas.toDataURL('image/png');
  }

  if (type === 'brightness') {
    // Brightness enhance matching hf_space: 1 + c where c = [0.1, 0.2, 0.3, 0.4, 0.5]
    const factor = 1 + [0.1, 0.2, 0.3, 0.4, 0.5][sevIndex];
    for (let i = 0; i < data.length; i += 4) {
      data[i] = Math.min(255, Math.max(0, Math.round(data[i] * factor)));
      data[i + 1] = Math.min(255, Math.max(0, Math.round(data[i + 1] * factor)));
      data[i + 2] = Math.min(255, Math.max(0, Math.round(data[i + 2] * factor)));
    }
    ctx.putImageData(imgData, 0, 0);
    return canvas.toDataURL('image/png');
  }

  if (type === 'contrast') {
    // Contrast enhance matching hf_space: 1 + c where c = [0.1, 0.2, 0.3, 0.4, 0.5]
    const factor = 1 + [0.1, 0.2, 0.3, 0.4, 0.5][sevIndex];
    for (let i = 0; i < data.length; i += 4) {
      data[i] = Math.min(255, Math.max(0, Math.round((data[i] - 128) * factor + 128)));
      data[i + 1] = Math.min(255, Math.max(0, Math.round((data[i + 1] - 128) * factor + 128)));
      data[i + 2] = Math.min(255, Math.max(0, Math.round((data[i + 2] - 128) * factor + 128)));
    }
    ctx.putImageData(imgData, 0, 0);
    return canvas.toDataURL('image/png');
  }

  return canvas.toDataURL('image/png');
}

export type CompoundSequenceOrder = 'NBJ' | 'NJB' | 'BNJ' | 'BJN' | 'JNB' | 'JBN';
export type ResearchLevelKey = 'Level_1_Mild' | 'Level_2_Moderate' | 'Level_3_SevereCommon';

export interface SequenceParameters {
  final_op: 'Noise' | 'Blur' | 'JPEG';
  noise_std: number;
  blur_sigma: number;
  jpeg_quality: number;
  achieved_psnr: number;
  achieved_ssim: number;
}

export interface ResearchLevelSpec {
  name: string;
  target_psnr: number;
  target_ssim: number;
  description: string;
  sequences: Record<CompoundSequenceOrder, SequenceParameters>;
}

export const FROZEN_CALIBRATION_3LEVELS: Record<ResearchLevelKey, ResearchLevelSpec> = {
  Level_1_Mild: {
    name: 'Level 1: Mild Common Support',
    target_psnr: 27.20,
    target_ssim: 0.810,
    description: 'Target: 27.20 dB PSNR / 0.810 SSIM (Tolerance: ±0.75 dB, ±0.035 SSIM)',
    sequences: {
      NBJ: { final_op: 'JPEG', noise_std: 0.070, blur_sigma: 1.0, jpeg_quality: 75, achieved_psnr: 27.22, achieved_ssim: 0.820 },
      NJB: { final_op: 'Blur', noise_std: 0.070, blur_sigma: 1.0, jpeg_quality: 35, achieved_psnr: 27.08, achieved_ssim: 0.820 },
      BNJ: { final_op: 'JPEG', noise_std: 0.035, blur_sigma: 1.0, jpeg_quality: 55, achieved_psnr: 27.10, achieved_ssim: 0.803 },
      BJN: { final_op: 'Noise', noise_std: 0.015, blur_sigma: 1.0, jpeg_quality: 75, achieved_psnr: 27.33, achieved_ssim: 0.801 },
      JNB: { final_op: 'Blur', noise_std: 0.070, blur_sigma: 1.0, jpeg_quality: 90, achieved_psnr: 27.26, achieved_ssim: 0.807 },
      JBN: { final_op: 'Noise', noise_std: 0.015, blur_sigma: 1.0, jpeg_quality: 55, achieved_psnr: 27.42, achieved_ssim: 0.808 },
    },
  },
  Level_2_Moderate: {
    name: 'Level 2: Moderate Common Support',
    target_psnr: 23.80,
    target_ssim: 0.650,
    description: 'Target: 23.80 dB PSNR / 0.650 SSIM (Tolerance: ±0.75 dB, ±0.035 SSIM)',
    sequences: {
      NBJ: { final_op: 'JPEG', noise_std: 0.200, blur_sigma: 1.0, jpeg_quality: 75, achieved_psnr: 23.69, achieved_ssim: 0.647 },
      NJB: { final_op: 'Blur', noise_std: 0.200, blur_sigma: 1.0, jpeg_quality: 35, achieved_psnr: 23.55, achieved_ssim: 0.651 },
      BNJ: { final_op: 'JPEG', noise_std: 0.120, blur_sigma: 0.5, jpeg_quality: 10, achieved_psnr: 23.75, achieved_ssim: 0.646 },
      BJN: { final_op: 'Noise', noise_std: 0.015, blur_sigma: 1.8, jpeg_quality: 20, achieved_psnr: 23.81, achieved_ssim: 0.647 },
      JNB: { final_op: 'Blur', noise_std: 0.120, blur_sigma: 1.0, jpeg_quality: 10, achieved_psnr: 23.96, achieved_ssim: 0.646 },
      JBN: { final_op: 'Noise', noise_std: 0.015, blur_sigma: 1.8, jpeg_quality: 10, achieved_psnr: 23.65, achieved_ssim: 0.656 },
    },
  },
  Level_3_SevereCommon: {
    name: 'Level 3: Severe Common Support',
    target_psnr: 21.80,
    target_ssim: 0.500,
    description: 'Target: 21.80 dB PSNR / 0.500 SSIM (Tolerance: ±0.75 dB, ±0.035 SSIM)',
    sequences: {
      NBJ: { final_op: 'JPEG', noise_std: 0.200, blur_sigma: 0.5, jpeg_quality: 35, achieved_psnr: 21.71, achieved_ssim: 0.515 },
      NJB: { final_op: 'Blur', noise_std: 0.200, blur_sigma: 0.5, jpeg_quality: 20, achieved_psnr: 21.65, achieved_ssim: 0.515 },
      BNJ: { final_op: 'JPEG', noise_std: 0.070, blur_sigma: 2.8, jpeg_quality: 35, achieved_psnr: 22.01, achieved_ssim: 0.496 },
      BJN: { final_op: 'Noise', noise_std: 0.070, blur_sigma: 0.5, jpeg_quality: 20, achieved_psnr: 22.03, achieved_ssim: 0.486 },
      JNB: { final_op: 'Blur', noise_std: 0.120, blur_sigma: 0.5, jpeg_quality: 35, achieved_psnr: 21.96, achieved_ssim: 0.496 },
      JBN: { final_op: 'Noise', noise_std: 0.070, blur_sigma: 0.5, jpeg_quality: 20, achieved_psnr: 22.19, achieved_ssim: 0.496 },
    },
  },
};

/**
 * Applies a 3-stage compound sequential distortion (N, B, J permutations) to an image
 * using exact physical order and parameters.
 */
export async function applyCompoundSequence(
  imageSrc: string,
  sequence: CompoundSequenceOrder,
  params: { noise_std: number; blur_sigma: number; jpeg_quality: number }
): Promise<string> {
  let currentSrc = imageSrc;

  // Split sequence into individual operations: e.g. 'NBJ' -> ['N', 'B', 'J']
  const ops = sequence.split('') as ('N' | 'B' | 'J')[];

  for (const op of ops) {
    const img = await loadImage(currentSrc);
    const canvas = document.createElement('canvas');
    const width = img.naturalWidth || img.width;
    const height = img.naturalHeight || img.height;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get 2D canvas context');

    if (op === 'N') {
      ctx.drawImage(img, 0, 0, width, height);
      const imgData = ctx.getImageData(0, 0, width, height);
      const data = imgData.data;
      const std = params.noise_std * 255;
      for (let i = 0; i < data.length; i += 4) {
        const noise = gaussianRandom(0, std);
        data[i] = Math.min(255, Math.max(0, data[i] + noise));
        data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + noise));
        data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + noise));
      }
      ctx.putImageData(imgData, 0, 0);
      currentSrc = canvas.toDataURL('image/png');
    } else if (op === 'B') {
      ctx.filter = `blur(${params.blur_sigma}px)`;
      ctx.drawImage(img, 0, 0, width, height);
      currentSrc = canvas.toDataURL('image/png');
    } else if (op === 'J') {
      ctx.drawImage(img, 0, 0, width, height);
      const quality = Math.max(0.01, Math.min(1.0, params.jpeg_quality / 100));
      currentSrc = canvas.toDataURL('image/jpeg', quality);
    }
  }

  return currentSrc;
}
