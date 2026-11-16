/**
 * RobustLens Client-Side Distortion & Fidelity Diagnostics Engine
 * 
 * Computes genuine pixel-level fidelity metrics (PSNR, SSIM) and physical
 * distortion fingerprints (Sobel edge energy, JPEG 8x8 block boundary ratio,
 * high-frequency residual energy) directly from HTML5 Canvas ImageData.
 * 
 * All formulas match the mathematical definitions documented in the
 * RobustLens research monograph and IEEE/ACM academic manuscript.
 */

export interface ComputedDiagnostics {
  psnr_db: number;
  ssim: number;
  edge_energy_retained_pct: number;
  jpeg_block_ratio: number;
  high_freq_residual_ratio: number;
  mean_absolute_error: number;
  timestamp: number;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = (err) => reject(new Error('Failed to load image for diagnostics: ' + err));
    img.src = src;
  });
}

/**
 * Computes genuine physical fidelity and distortion metrics between clean and corrupted images.
 */
export async function computeImageDiagnostics(
  cleanSrc: string,
  corruptedSrc: string
): Promise<ComputedDiagnostics> {
  const [cleanImg, corrImg] = await Promise.all([loadImage(cleanSrc), loadImage(corruptedSrc)]);

  const width = Math.min(cleanImg.naturalWidth || cleanImg.width, 480);
  const height = Math.min(cleanImg.naturalHeight || cleanImg.height, 480);

  // Clean canvas
  const cleanCanvas = document.createElement('canvas');
  cleanCanvas.width = width;
  cleanCanvas.height = height;
  const cleanCtx = cleanCanvas.getContext('2d', { willReadFrequently: true });
  if (!cleanCtx) throw new Error('Could not get clean canvas context');
  cleanCtx.drawImage(cleanImg, 0, 0, width, height);
  const cleanData = cleanCtx.getImageData(0, 0, width, height).data;

  // Corrupted canvas
  const corrCanvas = document.createElement('canvas');
  corrCanvas.width = width;
  corrCanvas.height = height;
  const corrCtx = corrCanvas.getContext('2d', { willReadFrequently: true });
  if (!corrCtx) throw new Error('Could not get corrupted canvas context');
  corrCtx.drawImage(corrImg, 0, 0, width, height);
  const corrData = corrCtx.getImageData(0, 0, width, height).data;

  const totalPixels = width * height;
  let sse = 0;
  let sae = 0;

  // Luminance arrays for SSIM and frequency analysis (Rec. 601 coefficients)
  const cleanLum = new Float32Array(totalPixels);
  const corrLum = new Float32Array(totalPixels);

  for (let i = 0; i < totalPixels; i++) {
    const idx = i * 4;
    const r1 = cleanData[idx];
    const g1 = cleanData[idx + 1];
    const b1 = cleanData[idx + 2];

    const r2 = corrData[idx];
    const g2 = corrData[idx + 1];
    const b2 = corrData[idx + 2];

    const dr = r1 - r2;
    const dg = g1 - g2;
    const db = b1 - b2;

    sse += dr * dr + dg * dg + db * db;
    sae += Math.abs(dr) + Math.abs(dg) + Math.abs(db);

    cleanLum[i] = 0.299 * r1 + 0.587 * g1 + 0.114 * b1;
    corrLum[i] = 0.299 * r2 + 0.587 * g2 + 0.114 * b2;
  }

  // 1. MSE and PSNR (dB)
  const mse = sse / (3 * totalPixels);
  const psnr = mse <= 1e-6 ? 99.99 : 10 * Math.log10((255 * 255) / mse);
  const mae = sae / (3 * totalPixels);

  // 2. SSIM (Structural Similarity Index on Luminance)
  const ssim = computeSSIM(cleanLum, corrLum, width, height);

  // 3. Sobel Edge Energy Retained
  const edgeEnergyClean = computeSobelEnergy(cleanLum, width, height);
  const edgeEnergyCorr = computeSobelEnergy(corrLum, width, height);
  const edgeEnergyRetainedPct = edgeEnergyClean > 0
    ? (edgeEnergyCorr / edgeEnergyClean) * 100
    : 100.0;

  // 4. JPEG 8x8 Block Boundary Discontinuity Ratio
  const jpegBlockRatio = computeJpegBlockBoundaryRatio(corrLum, width, height);

  // 5. High-Frequency Residual Energy Ratio
  const highFreqRatio = computeHighFreqResidualRatio(cleanLum, corrLum, width, height);

  return {
    psnr_db: Math.round(psnr * 100) / 100,
    ssim: Math.round(ssim * 1000) / 1000,
    edge_energy_retained_pct: Math.round(edgeEnergyRetainedPct * 10) / 10,
    jpeg_block_ratio: Math.round(jpegBlockRatio * 1000) / 1000,
    high_freq_residual_ratio: Math.round(highFreqRatio * 1000) / 1000,
    mean_absolute_error: Math.round(mae * 100) / 100,
    timestamp: Date.now(),
  };
}

/**
 * Windowed SSIM implementation across 8x8 non-overlapping blocks
 */
function computeSSIM(
  x: Float32Array,
  y: Float32Array,
  width: number,
  height: number
): number {
  const blockSize = 8;
  const C1 = 6.5025;   // (0.01 * 255)^2
  const C2 = 58.5225;  // (0.03 * 255)^2

  let totalSsim = 0;
  let blockCount = 0;

  for (let by = 0; by <= height - blockSize; by += blockSize) {
    for (let bx = 0; bx <= width - blockSize; bx += blockSize) {
      let sumX = 0;
      let sumY = 0;
      let sumX2 = 0;
      let sumY2 = 0;
      let sumXY = 0;
      const N = blockSize * blockSize;

      for (let dy = 0; dy < blockSize; dy++) {
        for (let dx = 0; dx < blockSize; dx++) {
          const idx = (by + dy) * width + (bx + dx);
          const px = x[idx];
          const py = y[idx];
          sumX += px;
          sumY += py;
          sumX2 += px * px;
          sumY2 += py * py;
          sumXY += px * py;
        }
      }

      const muX = sumX / N;
      const muY = sumY / N;
      const varX = (sumX2 / N) - (muX * muX);
      const varY = (sumY2 / N) - (muY * muY);
      const covXY = (sumXY / N) - (muX * muY);

      const num = (2 * muX * muY + C1) * (2 * covXY + C2);
      const den = (muX * muX + muY * muY + C1) * (varX + varY + C2);

      totalSsim += den > 0 ? num / den : 1.0;
      blockCount++;
    }
  }

  return blockCount > 0 ? Math.max(0, Math.min(1, totalSsim / blockCount)) : 1.0;
}

/**
 * Computes mean Sobel gradient magnitude
 */
function computeSobelEnergy(lum: Float32Array, width: number, height: number): number {
  let totalEnergy = 0;
  let count = 0;

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      // Sobel horizontal
      const gx =
        -lum[idx - width - 1] + lum[idx - width + 1] +
        -2 * lum[idx - 1] + 2 * lum[idx + 1] +
        -lum[idx + width - 1] + lum[idx + width + 1];

      // Sobel vertical
      const gy =
        -lum[idx - width - 1] - 2 * lum[idx - width] - lum[idx - width + 1] +
        lum[idx + width - 1] + 2 * lum[idx + width] + lum[idx + width + 1];

      totalEnergy += Math.sqrt(gx * gx + gy * gy);
      count++;
    }
  }

  return count > 0 ? totalEnergy / count : 0;
}

/**
 * Measures step discontinuities at 8x8 DCT grid lines relative to interior steps
 */
function computeJpegBlockBoundaryRatio(lum: Float32Array, width: number, height: number): number {
  let boundaryDiffSum = 0;
  let boundaryCount = 0;
  let interiorDiffSum = 0;
  let interiorCount = 0;

  // Horizontal step differences across columns
  for (let y = 0; y < height; y++) {
    const rowOffset = y * width;
    for (let x = 0; x < width - 1; x++) {
      const diff = Math.abs(lum[rowOffset + x + 1] - lum[rowOffset + x]);
      if ((x + 1) % 8 === 0) {
        boundaryDiffSum += diff;
        boundaryCount++;
      } else {
        interiorDiffSum += diff;
        interiorCount++;
      }
    }
  }

  const boundaryMean = boundaryCount > 0 ? boundaryDiffSum / boundaryCount : 1;
  const interiorMean = interiorCount > 0 ? interiorDiffSum / interiorCount : 1;

  return interiorMean > 0.1 ? boundaryMean / interiorMean : 1.0;
}

/**
 * Computes high-frequency residual ratio using a 3x3 Laplacian filter
 */
function computeHighFreqResidualRatio(
  clean: Float32Array,
  corr: Float32Array,
  width: number,
  height: number
): number {
  let cleanHighEnergy = 0;
  let corrHighEnergy = 0;

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;

      // 3x3 Laplacian: 4 * center - up - down - left - right
      const lapClean =
        4 * clean[idx] -
        clean[idx - width] - clean[idx + width] -
        clean[idx - 1] - clean[idx + 1];

      const lapCorr =
        4 * corr[idx] -
        corr[idx - width] - corr[idx + width] -
        corr[idx - 1] - corr[idx + 1];

      cleanHighEnergy += lapClean * lapClean;
      corrHighEnergy += lapCorr * lapCorr;
    }
  }

  return cleanHighEnergy > 1 ? corrHighEnergy / cleanHighEnergy : 1.0;
}
