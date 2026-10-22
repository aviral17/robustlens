# Reproducibility Guide: RobustLens Benchmark

This document provides exact, step-by-step instructions for reproducing the complete RobustLens robustness evaluation pipeline, statistical analysis, and dashboard integration from scratch.

---

## 1. System Requirements & Environment

### Software Prerequisites
- **Python**: Version 3.11 or newer (tested with Python 3.13 / 3.14)
- **Node.js**: Version 18.18+ or 20+
- **pnpm**: Version 9+
- **uv**: Fast Python package manager ([Astral uv](https://github.com/astral-sh/uv))
- **GPU (for full inference)**: NVIDIA Tesla T4 (16 GB VRAM) or equivalent CUDA GPU. Evaluation on CPU is supported for unit testing and small subsets.

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/aviral199/robustlens.git
cd robustlens

# 1. Setup Python ML environment using uv
cd ml
uv sync --extra dev
cd ..

# 2. Setup Next.js Web Application
cd apps/web
pnpm install
cd ../..
```

---

## 2. Dataset Provenance & Split Verification

RobustLens uses the official **FastAI Imagenette-v2** dataset, a 10-class subset of ImageNet-1K with verified WordNet synsets:
- **Mirror Source**: `johnowhitaker/imagenette2-320` on Hugging Face (13,394 total rows in parquet format).
- **Split Structure**:
  - **Training partition**: Rows 0 to 9,468 ($N = 9{,}469$).
  - **Validation partition**: Rows 9,469 to 13,393 ($N = 3{,}925$).
  - **Per-Class Validation Counts**:
    - Class 0 (tench, `n01440764`): 387
    - Class 1 (English springer, `n02102040`): 395
    - Class 2 (cassette player, `n02979186`): 357
    - Class 3 (chain saw, `n03000684`): 386
    - Class 4 (church, `n03028079`): 409
    - Class 5 (French horn, `n03394916`): 394
    - Class 6 (garbage truck, `n03417042`): 389
    - Class 7 (gas pump, `n03425413`): 419
    - Class 8 (golf ball, `n03445777`): 399
    - Class 9 (parachute, `n03888257`): 390
    - Total: 3,925 images (arithmetic mean: 392.5).

### Run Split Integrity Verification Test
```bash
uv run --project ml pytest ml/tests/test_dataset.py -v
```

---

## 3. Running the Evaluation Pipeline

### A. Dedicated Kaggle GPU Run (Full 3,925-Image Benchmark)
The full benchmark was executed on Kaggle GPU using kernel `aviral199/robustlens-robustness-evaluation` (Version 4):

```powershell
# 1. Prepare evaluation notebook
uv run --project ml python ml/scripts/prepare_full_eval_notebook.py

# 2. Push kernel to Kaggle
powershell -Command "& 'c:\STATISTICS & CV\robustlens\ml\.venv\Scripts\kaggle.exe' kernels push -p 'ml/notebooks/eval'"

# 3. Monitor status until COMPLETE
powershell -Command "& 'c:\STATISTICS & CV\robustlens\ml\.venv\Scripts\kaggle.exe' kernels status aviral199/robustlens-robustness-evaluation"

# 4. Download full output artifacts into results/full_eval/
powershell -Command "& 'c:\STATISTICS & CV\robustlens\ml\.venv\Scripts\kaggle.exe' kernels output aviral199/robustlens-robustness-evaluation -p 'results/full_eval' --force"
```

### B. Local Evaluation (Subset or Pilot)
To run locally on CPU or local GPU:
```bash
# Pilot 100-image evaluation:
uv run --project ml python ml/scripts/run_eval.py --subset-size 100 --output results/pilot_eval/outputs/

# Full local run (requires ~16 GB RAM and CUDA GPU recommended):
uv run --project ml python ml/scripts/run_eval.py --subset-size 3925 --output results/full_eval/outputs/
```

---

## 4. Statistical Report & Synthesis Generation

To regenerate the statistical synthesis (clustered bootstrap CIs with B=10,000, paired sign-flip permutation tests with R=20,000, clean McNemar tests, Holm/BH corrections, ECE, conformal coverage):

```bash
uv run --project ml python ml/scripts/make_report.py \
    --input results/full_eval/outputs/ \
    --output results/full_eval/outputs/ \
    --n-bootstraps 10000 \
    --alpha 0.05 \
    --seed 42

# Sync report to Next.js dashboard public directory
copy results\full_eval\outputs\benchmark_report.json apps\web\public\benchmark_report.json
```

---

## 5. Generating Publication Figures & Compiling PDF Manuscript

To generate high-resolution figures (300 DPI) and compile the publication-grade PDF manuscript:

```bash
# Generate all 5 research figures
uv run --project ml python ml/scripts/generate_research_figures.py

# Compile publication-grade academic PDF manuscript
uv run --project ml python ml/scripts/generate_manuscript_pdf.py
```
Output artifacts:
1. `reports/figures/01_clean_vs_shifted_accuracy.png`
2. `reports/figures/02_corruption_severity_degradation.png`
3. `reports/figures/03_per_corruption_breakdown.png`
4. `reports/figures/04_calibration_ece_shift.png`
5. `reports/figures/05_pairwise_clustered_confidence_intervals.png`
6. `reports/RobustLens_Research_Manuscript.pdf`

---

## 6. Running Tests & Building the Web Dashboard

```bash
# Run all 33 ML unit tests
uv run --project ml --extra dev pytest ml/tests

# Run Next.js production build
pnpm --dir apps/web build

# Launch local dashboard server
pnpm --dir apps/web start -p 3000
```
Open `http://localhost:3000` to interact with the full benchmark dashboard.

---

## 7. Reproducing Compound Corruption Order Effects Study

### A. Download & Validate Benchmark Outputs
The canonical prediction artifact is `results/order_effect_study/full/outputs/all_order_predictions.csv.gz` (486,700 rows, 8.62 MB gzipped).

```bash
# Validate full 3,925-image benchmark outputs:
uv run --project ml python ml/scripts/validate_order_full.py --dir results/order_effect_study/full/outputs/ --images 3925
```

### B. Fit Factorial Clustered ANCOVA Models
```bash
# Fit Model 1 (main effects: Order 5 df + PSNR + SSIM) and Model 2 (factorial: Order 5 df + Severity 4 df + Order x Severity 20 df + Joint Order 25 df)
uv run --project ml python ml/scripts/fit_factorial_clustered_ancova.py
```
Output artifact: `results/order_effect_study/full/outputs/factorial_ancova_results.csv`

### C. Generate Figures 6–9
```bash
uv run --project ml python ml/scripts/generate_order_figures.py
```
Output artifacts:
1. `reports/figures/06_order_effects_by_architecture.png`
2. `reports/figures/07_psnr_ssim_vs_accuracy_by_order.png`
3. `reports/figures/08_order_severity_interaction.png`
4. `reports/figures/09_order_calibration_ece.png`

### D. Compile Updated Manuscript PDF
```bash
uv run --project ml python ml/scripts/generate_manuscript_pdf.py
```
Output:
- `reports/RobustLens_Research_Manuscript.pdf` (9 pages, includes complete Tables 1–5 & Figures 1–9)

### E. Local Execution via CLI
```bash
# Pilot (100 images) or full local run:
uv run --project ml python ml/scripts/run_order_eval.py --subset-size 100 --output results/order_effect_study/pilot/
```

---

## 8. Reproducing Matched-Fidelity Counterfactual Benchmark

### A. Zero-Leakage Parameter Calibration
Calibrate corruption parameters across all 6 sequences on 50 training split images (indices 0..9468) to match 5 target physical fidelity levels (PSNR 28.0, 24.5, 21.5, 18.5, 16.0 dB):

```bash
uv run --project ml python ml/scripts/calibrate_matched_fidelity.py
```
Outputs:
- `results/order_effect_study/matched_fidelity/calibration_config.json`
- `results/order_effect_study/matched_fidelity/matching_residuals.csv`

### B. Remote Kaggle Execution & Artifact Download
Executed under kernel `aviral199/robustlens-matched-fidelity-full-benchmark` (Version 1, Tesla T4 GPU, runtime ~2.1 hours).
```bash
# Download all 9 output files:
kaggle kernels output aviral199/robustlens-matched-fidelity-full-benchmark -p results/order_effect_study/matched_fidelity/outputs --force
```

### C. Validation of Matched-Fidelity Artifacts
```bash
uv run --project ml python ml/scripts/validate_matched_fidelity.py
```
Validates:
- Exactly 486,700 rows, 3,925 images, 4 models, 31 conditions per image.
- Zero null or missing values across all prediction columns.

### D. Fit Matched Factorial Clustered ANCOVA & Nominal Comparison
```bash
uv run --project ml python ml/scripts/fit_matched_factorial_ancova.py
uv run --project ml python ml/scripts/compare_nominal_vs_matched.py
```
Outputs:
- `results/order_effect_study/matched_fidelity/outputs/matched_factorial_ancova.csv`
- `results/order_effect_study/matched_fidelity/outputs/nominal_vs_matched_summary.csv`
- `results/order_effect_study/matched_fidelity/outputs/nominal_vs_matched_comparison.json`

### E. Generate Publication Figures 1–5
```bash
uv run --project ml python ml/scripts/generate_matched_fidelity_figures.py
```
Outputs in `reports/figures/` (and synced to `apps/web/public/figures/`):
1. `fig1_matched_fidelity_validation.png` (PSNR/SSIM calibration validation)
2. `fig2_sequence_gap_contraction.png` (Nominal vs. matched gap contraction)
3. `fig3_factorial_interaction_profiles.png` (Target-level degradation profiles)
4. `fig4_matched_churn_decoupling.png` (Error churn vs. net accuracy decoupling)
5. `fig5_matched_calibration.png` (Expected Calibration Error by architecture)

### F. Compile New Research Manuscript & Run Unit Tests
```bash
# Compile publication PDF:
uv run --project ml python ml/scripts/generate_new_manuscript_pdf.py

# Run matched-fidelity test suite:
uv run --project ml pytest ml/tests/test_matched_fidelity.py -v
```
Output:
- `reports/RobustLens_New_Research_Manuscript.pdf` (6 pages, ReportLab letter)

