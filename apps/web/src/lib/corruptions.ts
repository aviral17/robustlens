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
