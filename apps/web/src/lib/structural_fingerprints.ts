/**
 * RobustLens Canonical Structural Distortion Fingerprint Engine
 * 
 * Implements the exact 5 physical distortion fingerprints defined in the
 * RobustLens confirmatory research monograph (November 2026):
 * 1. Fourier Low-Frequency Residual Energy (E_low, r_norm < 0.15)
 * 2. Fourier High-Frequency Residual Energy (E_high, r_norm >= 0.45)
 * 3. JPEG 8x8 Block-Boundary Discontinuity Ratio (B_JPEG)
 * 4. Sobel Edge Energy Relative Change (Delta E_edge)
 * 5. Local SSIM Spatial Standard Deviation (sigma_local_SSIM)
 * 
 * Also generates diagnostic visual canvas maps for interactive inspection:
 * - 2D Radial Fourier Residual Power Spectrum
 * - Spatial Sobel Gradient Magnitude Map
 * - Local SSIM Spatial Heatmap
 * - 8x8 DCT Block Discontinuity Grid Overlay
 */

export interface DistortionFingerprints {
  fft_energy_low: number;
  fft_energy_med: number;
  fft_energy_high: number;
  jpeg_block_artifact: number;
  delta_edge_energy: number;
  std_local_ssim: number;
  mean_ssim: number;
  psnr_db: number;
}

export interface FingerprintVisualMaps {
  fourierSpectrumUrl: string;
  sobelEdgeMapUrl: string;
  ssimHeatmapUrl: string;
  jpegOverlayUrl: string;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = (err) => reject(new Error('Failed to load image for fingerprint analysis: ' + err));
    img.src = src;
  });
}

/**
 * 1D Radix-2 In-place Cooley-Tukey FFT
 */
function fft1d(real: Float32Array, imag: Float32Array): void {
  const n = real.length;
  if (n <= 1) return;

  // Bit reversal permutation
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      const tr = real[i]; real[i] = real[j]; real[j] = tr;
      const ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
    }
    let k = n >> 1;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }

  // Iterative butterflies
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wStepR = Math.cos(angle);
    const wStepI = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let wr = 1.0;
      let wi = 0.0;
      for (let m = 0; m < half; m++) {
        const uR = real[i + m];
        const uI = imag[i + m];
        const vR = real[i + m + half] * wr - imag[i + m + half] * wi;
        const vI = real[i + m + half] * wi + imag[i + m + half] * wr;

        real[i + m] = uR + vR;
        imag[i + m] = uI + vI;
        real[i + m + half] = uR - vR;
        imag[i + m + half] = uI - vI;

        const nextWr = wr * wStepR - wi * wStepI;
        wi = wr * wStepI + wi * wStepR;
        wr = nextWr;
      }
    }
  }
}

/**
 * 2D FFT on N x N grid with fftshift
 */
function compute2dFftShiftedPower(delta: Float32Array, N: number): Float32Array {
  const real = new Float32Array(delta);
  const imag = new Float32Array(N * N);

  // Row-wise FFT
  const rowR = new Float32Array(N);
  const rowI = new Float32Array(N);
  for (let y = 0; y < N; y++) {
    const off = y * N;
    for (let x = 0; x < N; x++) {
      rowR[x] = real[off + x];
      rowI[x] = 0;
    }
    fft1d(rowR, rowI);
    for (let x = 0; x < N; x++) {
      real[off + x] = rowR[x];
      imag[off + x] = rowI[x];
    }
  }

  // Column-wise FFT
  const colR = new Float32Array(N);
  const colI = new Float32Array(N);
  for (let x = 0; x < N; x++) {
    for (let y = 0; y < N; y++) {
      colR[y] = real[y * N + x];
      colI[y] = imag[y * N + x];
    }
    fft1d(colR, colI);
    for (let y = 0; y < N; y++) {
      real[y * N + x] = colR[y];
      imag[y * N + x] = colI[y];
    }
  }

  // Centered power spectrum (fftshift)
  const power = new Float32Array(N * N);
  const half = N / 2;
  for (let y = 0; y < N; y++) {
    const shiftY = (y + half) % N;
    for (let x = 0; x < N; x++) {
      const shiftX = (x + half) % N;
      const r = real[y * N + x];
      const im = imag[y * N + x];
      power[shiftY * N + shiftX] = r * r + im * im;
    }
  }

  return power;
}

/**
 * Computes exact manuscript physical distortion fingerprints
 */
export async function computeManuscriptFingerprints(
  cleanSrc: string,
  corruptSrc: string
): Promise<DistortionFingerprints> {
  const [cleanImg, corrImg] = await Promise.all([loadImage(cleanSrc), loadImage(corruptSrc)]);

  const N = 256; // Standardized power-of-2 analysis grid
  const canvasClean = document.createElement('canvas');
  canvasClean.width = N;
  canvasClean.height = N;
  const ctxClean = canvasClean.getContext('2d', { willReadFrequently: true });
  if (!ctxClean) throw new Error('Cannot get 2D clean context');
  ctxClean.drawImage(cleanImg, 0, 0, N, N);
  const cleanData = ctxClean.getImageData(0, 0, N, N).data;

  const canvasCorr = document.createElement('canvas');
  canvasCorr.width = N;
  canvasCorr.height = N;
  const ctxCorr = canvasCorr.getContext('2d', { willReadFrequently: true });
  if (!ctxCorr) throw new Error('Cannot get 2D corrupt context');
  ctxCorr.drawImage(corrImg, 0, 0, N, N);
  const corrData = ctxCorr.getImageData(0, 0, N, N).data;

  // 1. Grayscale luminance arrays in [0.0, 1.0]
  const grayClean = new Float32Array(N * N);
  const grayCorr = new Float32Array(N * N);
  const delta = new Float32Array(N * N);

  let sse = 0;
  for (let i = 0; i < N * N; i++) {
    const idx = i * 4;
    // Standard sRGB luminance Rec. 601 coefficients
    const lc = (0.299 * cleanData[idx] + 0.587 * cleanData[idx + 1] + 0.114 * cleanData[idx + 2]) / 255.0;
    const lk = (0.299 * corrData[idx] + 0.587 * corrData[idx + 1] + 0.114 * corrData[idx + 2]) / 255.0;
    grayClean[i] = lc;
    grayCorr[i] = lk;
    delta[i] = lk - lc;

    const diff = lk - lc;
    sse += diff * diff;
  }

  const mse = sse / (N * N);
  const psnr = mse <= 1e-10 ? 99.99 : 10.0 * Math.log10(1.0 / mse);

  // 2. 2D Fourier Residual Power Spectrum (Low, Med, High radial bands)
  const power = compute2dFftShiftedPower(delta, N);
  const halfN = N / 2;
  const maxR = Math.sqrt(halfN * halfN + halfN * halfN);

  let lowPower = 0;
  let medPower = 0;
  let highPower = 0;
  let totalPower = 0;

  for (let y = 0; y < N; y++) {
    const dy = y - halfN;
    for (let x = 0; x < N; x++) {
      const dx = x - halfN;
      const rNorm = Math.sqrt(dx * dx + dy * dy) / maxR;
      const p = power[y * N + x];
      totalPower += p;

      if (rNorm < 0.15) {
        lowPower += p;
      } else if (rNorm < 0.45) {
        medPower += p;
      } else {
        highPower += p;
      }
    }
  }

  const safeTotalPower = totalPower + 1e-12;
  const eLow = lowPower / safeTotalPower;
  const eMed = medPower / safeTotalPower;
  const eHigh = highPower / safeTotalPower;

  // 3. JPEG 8x8 Block-Boundary Discontinuity Ratio
  let bhSum = 0;
  let bhCount = 0;
  let ihSum = 0;
  let ihCount = 0;

  for (let y = 0; y < N; y++) {
    const row = y * N;
    for (let x = 0; x < N - 1; x++) {
      const diff = Math.abs(grayCorr[row + x + 1] - grayCorr[row + x]);
      if ((x + 1) % 8 === 0) {
        bhSum += diff;
        bhCount++;
      } else {
        ihSum += diff;
        ihCount++;
      }
    }
  }

  const bh = bhCount > 0 ? bhSum / bhCount : 0.0;
  const ih = ihCount > 0 ? ihSum / ihCount : 1.0;

  let bvSum = 0;
  let bvCount = 0;
  let ivSum = 0;
  let ivCount = 0;

  for (let y = 0; y < N - 1; y++) {
    const isBoundary = (y + 1) % 8 === 0;
    const row1 = y * N;
    const row2 = (y + 1) * N;
    for (let x = 0; x < N; x++) {
      const diff = Math.abs(grayCorr[row2 + x] - grayCorr[row1 + x]);
      if (isBoundary) {
        bvSum += diff;
        bvCount++;
      } else {
        ivSum += diff;
        ivCount++;
      }
    }
  }

  const bv = bvCount > 0 ? bvSum / bvCount : 0.0;
  const iv = ivCount > 0 ? ivSum / ivCount : 1.0;

  const jpegBlockRatio = ((bh / (ih + 1e-6)) + (bv / (iv + 1e-6))) / 2.0;

  // 4. Sobel Edge Energy Relative Difference
  let cleanEdgeSum = 0;
  let corrEdgeSum = 0;
  let edgeCount = 0;

  for (let y = 1; y < N - 1; y++) {
    for (let x = 1; x < N - 1; x++) {
      const idx = y * N + x;

      // Sobel Clean
      const gxc =
        -grayClean[idx - N - 1] + grayClean[idx - N + 1] +
        -2 * grayClean[idx - 1] + 2 * grayClean[idx + 1] +
        -grayClean[idx + N - 1] + grayClean[idx + N + 1];
      const gyc =
        -grayClean[idx - N - 1] - 2 * grayClean[idx - N] - grayClean[idx - N + 1] +
        grayClean[idx + N - 1] + 2 * grayClean[idx + N] + grayClean[idx + N + 1];
      cleanEdgeSum += Math.sqrt(gxc * gxc + gyc * gyc);

      // Sobel Corrupt
      const gxk =
        -grayCorr[idx - N - 1] + grayCorr[idx - N + 1] +
        -2 * grayCorr[idx - 1] + 2 * grayCorr[idx + 1] +
        -grayCorr[idx + N - 1] + grayCorr[idx + N + 1];
      const gyk =
        -grayCorr[idx - N - 1] - 2 * grayCorr[idx - N] - grayCorr[idx - N + 1] +
        grayCorr[idx + N - 1] + 2 * grayCorr[idx + N] + grayCorr[idx + N + 1];
      corrEdgeSum += Math.sqrt(gxk * gxk + gyk * gyk);

      edgeCount++;
    }
  }

  const edgeCleanMean = edgeCount > 0 ? cleanEdgeSum / edgeCount : 1.0;
  const edgeCorrMean = edgeCount > 0 ? corrEdgeSum / edgeCount : 1.0;
  const deltaEdgeEnergy = (edgeCorrMean - edgeCleanMean) / (edgeCleanMean + 1e-6);

  // 5. Local SSIM Spatial Map and Standard Deviation (8x8 windowed blocks)
  const localSsimValues: number[] = [];
  const C1 = 0.0001; // (0.01 * 1.0)^2
  const C2 = 0.0009; // (0.03 * 1.0)^2
  const blockSize = 8;
  const blockPixels = blockSize * blockSize;

  for (let by = 0; by <= N - blockSize; by += blockSize) {
    for (let bx = 0; bx <= N - blockSize; bx += blockSize) {
      let sumC = 0;
      let sumK = 0;
      let sumC2 = 0;
      let sumK2 = 0;
      let sumCK = 0;

      for (let dy = 0; dy < blockSize; dy++) {
        const row = (by + dy) * N;
        for (let dx = 0; dx < blockSize; dx++) {
          const valC = grayClean[row + (bx + dx)];
          const valK = grayCorr[row + (bx + dx)];
          sumC += valC;
          sumK += valK;
          sumC2 += valC * valC;
          sumK2 += valK * valK;
          sumCK += valC * valK;
        }
      }

      const muC = sumC / blockPixels;
      const muK = sumK / blockPixels;
      const varC = (sumC2 / blockPixels) - (muC * muC);
      const varK = (sumK2 / blockPixels) - (muK * muK);
      const covCK = (sumCK / blockPixels) - (muC * muK);

      const num = (2 * muC * muK + C1) * (2 * covCK + C2);
      const den = (muC * muC + muK * muK + C1) * (varC + varK + C2);
      const ssimVal = den > 0 ? num / den : 1.0;
      localSsimValues.push(Math.max(0, Math.min(1, ssimVal)));
    }
  }

  const meanSsim = localSsimValues.reduce((a, b) => a + b, 0) / localSsimValues.length;
  let varSum = 0;
  for (const v of localSsimValues) {
    const diff = v - meanSsim;
    varSum += diff * diff;
  }
  const stdLocalSsim = Math.sqrt(varSum / localSsimValues.length);

  return {
    fft_energy_low: Math.round(eLow * 10000) / 10000,
    fft_energy_med: Math.round(eMed * 10000) / 10000,
    fft_energy_high: Math.round(eHigh * 10000) / 10000,
    jpeg_block_artifact: Math.round(jpegBlockRatio * 10000) / 10000,
    delta_edge_energy: Math.round(deltaEdgeEnergy * 10000) / 10000,
    std_local_ssim: Math.round(stdLocalSsim * 10000) / 10000,
    mean_ssim: Math.round(meanSsim * 1000) / 1000,
    psnr_db: Math.round(psnr * 100) / 100,
  };
}

/**
 * Generates all 4 visual diagnostic canvas maps
 */
export async function generateFingerprintVisualMaps(
  cleanSrc: string,
  corruptSrc: string
): Promise<FingerprintVisualMaps> {
  const [cleanImg, corrImg] = await Promise.all([loadImage(cleanSrc), loadImage(corruptSrc)]);

  const N = 256;
  const canvasClean = document.createElement('canvas');
  canvasClean.width = N;
  canvasClean.height = N;
  const ctxClean = canvasClean.getContext('2d', { willReadFrequently: true });
  if (!ctxClean) throw new Error('Cannot get 2D clean context');
  ctxClean.drawImage(cleanImg, 0, 0, N, N);
  const cleanData = ctxClean.getImageData(0, 0, N, N).data;

  const canvasCorr = document.createElement('canvas');
  canvasCorr.width = N;
  canvasCorr.height = N;
  const ctxCorr = canvasCorr.getContext('2d', { willReadFrequently: true });
  if (!ctxCorr) throw new Error('Cannot get 2D corrupt context');
  ctxCorr.drawImage(corrImg, 0, 0, N, N);
  const corrData = ctxCorr.getImageData(0, 0, N, N).data;

  const grayClean = new Float32Array(N * N);
  const grayCorr = new Float32Array(N * N);
  const delta = new Float32Array(N * N);

  for (let i = 0; i < N * N; i++) {
    const idx = i * 4;
    const lc = (0.299 * cleanData[idx] + 0.587 * cleanData[idx + 1] + 0.114 * cleanData[idx + 2]) / 255.0;
    const lk = (0.299 * corrData[idx] + 0.587 * corrData[idx + 1] + 0.114 * corrData[idx + 2]) / 255.0;
    grayClean[i] = lc;
    grayCorr[i] = lk;
    delta[i] = lk - lc;
  }

  // 1. Fourier 2D Log-Power Spectrum Map with Circular Overlays
  const power = compute2dFftShiftedPower(delta, N);
  const fourierCanvas = document.createElement('canvas');
  fourierCanvas.width = N;
  fourierCanvas.height = N;
  const fourierCtx = fourierCanvas.getContext('2d');
  if (!fourierCtx) throw new Error('Cannot get fourier context');

  let maxLogPower = 0;
  const logPower = new Float32Array(N * N);
  for (let i = 0; i < N * N; i++) {
    const lp = Math.log1p(power[i]);
    logPower[i] = lp;
    if (lp > maxLogPower) maxLogPower = lp;
  }

  const fourierImgData = fourierCtx.createImageData(N, N);
  const fd = fourierImgData.data;
  const scale = maxLogPower > 0 ? 255 / maxLogPower : 1;

  for (let i = 0; i < N * N; i++) {
    const intensity = Math.min(255, Math.round(logPower[i] * scale));
    const pIdx = i * 4;
    // Plasma/turbo color gradient
    fd[pIdx] = intensity;                     // R
    fd[pIdx + 1] = Math.round(intensity * 0.7); // G
    fd[pIdx + 2] = Math.round(255 - intensity * 0.8); // B
    fd[pIdx + 3] = 255;
  }
  fourierCtx.putImageData(fourierImgData, 0, 0);

  // Overlay Low-pass and High-pass boundary circles
  const halfN = N / 2;
  const maxR = Math.sqrt(halfN * halfN + halfN * halfN);
  const rLow = 0.15 * maxR;
  const rHigh = 0.45 * maxR;

  fourierCtx.lineWidth = 1.5;
  // Low-frequency band circle (Cyan)
  fourierCtx.strokeStyle = 'rgba(6, 182, 212, 0.85)';
  fourierCtx.beginPath();
  fourierCtx.arc(halfN, halfN, rLow, 0, 2 * Math.PI);
  fourierCtx.stroke();

  // High-frequency band circle (Rose)
  fourierCtx.strokeStyle = 'rgba(244, 63, 94, 0.85)';
  fourierCtx.beginPath();
  fourierCtx.arc(halfN, halfN, rHigh, 0, 2 * Math.PI);
  fourierCtx.stroke();

  const fourierSpectrumUrl = fourierCanvas.toDataURL('image/png');

  // 2. Sobel Edge Energy Gradient Map
  const sobelCanvas = document.createElement('canvas');
  sobelCanvas.width = N;
  sobelCanvas.height = N;
  const sobelCtx = sobelCanvas.getContext('2d');
  if (!sobelCtx) throw new Error('Cannot get sobel context');
  const sobelImgData = sobelCtx.createImageData(N, N);
  const sd = sobelImgData.data;

  for (let y = 1; y < N - 1; y++) {
    for (let x = 1; x < N - 1; x++) {
      const idx = y * N + x;
      const gx =
        -grayCorr[idx - N - 1] + grayCorr[idx - N + 1] +
        -2 * grayCorr[idx - 1] + 2 * grayCorr[idx + 1] +
        -grayCorr[idx + N - 1] + grayCorr[idx + N + 1];
      const gy =
        -grayCorr[idx - N - 1] - 2 * grayCorr[idx - N] - grayCorr[idx - N + 1] +
        grayCorr[idx + N - 1] + 2 * grayCorr[idx + N] + grayCorr[idx + N + 1];
      const mag = Math.min(255, Math.round(Math.sqrt(gx * gx + gy * gy) * 128));

      const pIdx = idx * 4;
      sd[pIdx] = mag;
      sd[pIdx + 1] = Math.round(mag * 0.9);
      sd[pIdx + 2] = Math.round(mag * 1.1);
      sd[pIdx + 3] = 255;
    }
  }
  sobelCtx.putImageData(sobelImgData, 0, 0);
  const sobelEdgeMapUrl = sobelCanvas.toDataURL('image/png');

  // 3. Local SSIM Heatmap Map (8x8 Block Grid)
  const ssimCanvas = document.createElement('canvas');
  ssimCanvas.width = N;
  ssimCanvas.height = N;
  const ssimCtx = ssimCanvas.getContext('2d');
  if (!ssimCtx) throw new Error('Cannot get ssim context');

  const C1 = 0.0001;
  const C2 = 0.0009;
  const blockSize = 8;
  const blockPixels = blockSize * blockSize;

  for (let by = 0; by <= N - blockSize; by += blockSize) {
    for (let bx = 0; bx <= N - blockSize; bx += blockSize) {
      let sumC = 0, sumK = 0, sumC2 = 0, sumK2 = 0, sumCK = 0;
      for (let dy = 0; dy < blockSize; dy++) {
        const row = (by + dy) * N;
        for (let dx = 0; dx < blockSize; dx++) {
          const valC = grayClean[row + (bx + dx)];
          const valK = grayCorr[row + (bx + dx)];
          sumC += valC;
          sumK += valK;
          sumC2 += valC * valC;
          sumK2 += valK * valK;
          sumCK += valC * valK;
        }
      }
      const muC = sumC / blockPixels;
      const muK = sumK / blockPixels;
      const varC = (sumC2 / blockPixels) - (muC * muC);
      const varK = (sumK2 / blockPixels) - (muK * muK);
      const covCK = (sumCK / blockPixels) - (muC * muK);

      const num = (2 * muC * muK + C1) * (2 * covCK + C2);
      const den = (muC * muC + muK * muK + C1) * (varC + varK + C2);
      const ssimVal = den > 0 ? Math.max(0, Math.min(1, num / den)) : 1.0;

      // Colormap: Low SSIM (<0.6) = Red/Amber, High SSIM (>0.85) = Emerald
      let r = 0, g = 0, b = 0;
      if (ssimVal < 0.5) {
        r = 239; g = 68; b = 68; // Rose / Red
      } else if (ssimVal < 0.75) {
        r = 245; g = 158; b = 11; // Amber
      } else if (ssimVal < 0.9) {
        r = 59; g = 130; b = 246; // Blue
      } else {
        r = 16; g = 185; b = 129; // Emerald
      }

      ssimCtx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ssimCtx.fillRect(bx, by, blockSize, blockSize);
    }
  }
  const ssimHeatmapUrl = ssimCanvas.toDataURL('image/png');

  // 4. JPEG 8x8 Block Discontinuity Grid Overlay
  const jpegCanvas = document.createElement('canvas');
  jpegCanvas.width = N;
  jpegCanvas.height = N;
  const jpegCtx = jpegCanvas.getContext('2d');
  if (!jpegCtx) throw new Error('Cannot get jpeg context');

  // Draw corrupted image as base
  jpegCtx.drawImage(corrImg, 0, 0, N, N);

  // Draw 8x8 DCT grid lines
  jpegCtx.strokeStyle = 'rgba(234, 179, 8, 0.45)';
  jpegCtx.lineWidth = 1;
  for (let p = 8; p < N; p += 8) {
    // Vertical line
    jpegCtx.beginPath();
    jpegCtx.moveTo(p, 0);
    jpegCtx.lineTo(p, N);
    jpegCtx.stroke();

    // Horizontal line
    jpegCtx.beginPath();
    jpegCtx.moveTo(0, p);
    jpegCtx.lineTo(N, p);
    jpegCtx.stroke();
  }

  const jpegOverlayUrl = jpegCanvas.toDataURL('image/png');

  return {
    fourierSpectrumUrl,
    sobelEdgeMapUrl,
    ssimHeatmapUrl,
    jpegOverlayUrl,
  };
}
