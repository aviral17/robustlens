# Reproducibility Guide: Order Matters Beyond Accuracy: Paired Prediction Analysis of Compound Image Corruptions

**Title:** Order Matters Beyond Accuracy: Paired Prediction Analysis of Compound Image Corruptions  
**Artifact:** Complete Environment Specification, Data Provenance, & Reproduction Protocol  
**Author:** Aviral Choudhary (RobustLens Research Project)  
**Date:** October 2026  
**Repository Path:** `reports/final_research_package/REPRODUCIBILITY.md`  

---

## 1. System Requirements & Environment Specifications

### 1.1 Software & Hardware Architecture
- **Operating System:** Windows 11 / Linux (Ubuntu 22.04 LTS or newer)
- **Python Environment Manager:** Astral `uv` (tested on Python 3.11–3.14)
- **Node.js Environment Manager:** `pnpm` 9.0+ (Node.js v20+)
- **Core Python Dependencies:**
  - `torch >= 2.0` & `torchvision >= 0.15`
  - `timm >= 1.0` (PyTorch Image Models)
  - `pandas >= 2.1`, `numpy >= 1.26`, `scipy >= 1.12`, `statsmodels >= 0.14`
  - `matplotlib >= 3.8`, `reportlab >= 4.0`, `pypdf >= 4.0`
- **Compute:** Local CPU is fully sufficient for replicating the statistical analysis, decomposition, bootstrapping, and figure/PDF compilation. The initial $973{,}400$ GPU inferences were executed on NVIDIA Tesla T4 GPUs via Kaggle Kernels.

### 1.2 Quickstart Setup
```bash
# Clone the repository
git clone https://github.com/aviral199/robustlens.git
cd robustlens

# Setup Python ML environment using uv
cd ml
uv sync --extra dev
cd ..

# Setup Next.js Web Dashboard
cd apps/web
pnpm install
cd ../..
```

---

## 2. Dataset Provenance & Split Integrity

The benchmark evaluates the official **FastAI Imagenette-v2** dataset (10 natural WordNet synsets derived from ImageNet-1k):
- **Hugging Face Mirror:** `johnowhitaker/imagenette2-320` (13,394 total rows in parquet format).
- **Partition Structure:**
  - **Training Partition (Rows 0 to 9,468; $N = 9{,}469$ images):** Used strictly for zero-leakage calibration of sequence parameters.
  - **Validation Partition (Rows 9,469 to 13,393; $N = 3{,}925$ images):** Used exclusively for model evaluation and reporting.
- **Per-Class Validation Counts ($N = 3{,}925$):**
  - Class 0 (`tench`, `n01440764`): 387
  - Class 1 (`English springer`, `n02102040`): 395
  - Class 2 (`cassette player`, `n02979186`): 357
  - Class 3 (`chain saw`, `n03000684`): 386
  - Class 4 (`church`, `n03028079`): 409
  - Class 5 (`French horn`, `n03394916`): 394
  - Class 6 (`garbage truck`, `n03417042`): 389
  - Class 7 (`gas pump`, `n03425413`): 419
  - Class 8 (`golf ball`, `n03445777`): 399
  - Class 9 (`parachute`, `n03888257`): 390

---

## 3. Step-by-Step Reproduction Pipeline

All primary artifacts can be regenerated deterministically from the validated raw prediction files:
1. `results/order_effect_study/full/outputs/all_order_predictions.csv.gz` ($486{,}700$ records, MD5 verified).
2. `results/order_effect_study/matched_fidelity/outputs/all_order_predictions.csv.gz` ($486{,}700$ records, MD5 verified).

### Step 3.1: Execute Full-Rank Saturated Six-Cell Decomposition & Clustered Bootstrap
Computes the full-rank linear decomposition ($\beta_0 \dots \beta_5$) with balanced parity contrast between even and odd permutations, 5,000 cluster bootstrap resamples on physical images, Holm-adjusted p-values in log space for primary top-1 accuracy without literal 0.0 underflows, and FDR-adjusted p-values for secondary metrics:
```bash
cd ml
uv run python scripts/run_final_order_decomposition_analysis.py
```
**Outputs generated:**
- `reports/final_research_package/analysis/decomposition_coefficients.csv`
- `reports/final_research_package/analysis/order_cell_means.csv`
- `reports/final_research_package/analysis/marginal_contrasts.csv`
- `reports/final_research_package/analysis/error_churn_matrix.csv`
- `reports/final_research_package/analysis/prediction_instability_transitions.csv`
- `reports/final_research_package/analysis/per_class_instability_summary.csv`
- `reports/final_research_package/analysis/nominal_vs_matched_gap_contraction.csv`
- `reports/final_research_package/analysis/decomposition_summary.json`

### Step 3.2: Regenerate Publication Figures
Generates Figures 1–5 in high resolution (300 DPI):
```bash
cd ml
uv run python scripts/generate_final_package_figures.py
```
**Outputs generated:**
- `reports/final_research_package/figures/fig1_calibration_residuals_and_tolerances.png`
- `reports/final_research_package/figures/fig2_nominal_vs_matched_gap_contraction.png`
- `reports/final_research_package/figures/fig3_saturated_decomposition_coefficients.png`
- `reports/final_research_package/figures/fig4_paired_error_churn_decomposition.png`
- `reports/final_research_package/figures/fig5_calibration_ece_and_confidence.png`

### Step 3.3: Compile Primary Research Manuscript PDF
Builds the publication-grade 5-page PDF using ReportLab:
```bash
cd ml
uv run python scripts/generate_final_paper_pdf.py
uv run python scripts/generate_phd_working_paper_pdf.py
```
**Outputs generated:**
- `reports/final_research_package/RobustLens_Final_Paper.pdf`
- `reports/final_research_package/RobustLens_PhD_Working_Paper.pdf`

### Step 3.4: Run Automated Statistical & Unit Tests
Verifies that sequence labels, design matrix rank ($\det(X) = -6$), 5-state transitions, label churn identities, calibration tolerances, and full ML modules hold (52/52 tests passing):
```bash
cd ml
uv run pytest -v
```

---

## 4. Documentation of External Confirmation Blocker (ImageNet-V2)

In compliance with our non-negotiable rule forbidding data fabrication or speculative claims:
- **Status:** An independent external confirmation run on the 10-synset subset of ImageNet-V2 was investigated.
- **Audit Finding:** The local Hugging Face cache (`~/.cache/huggingface/hub`) contains only the FastAI Imagenette splits (`datasets--frgfm--imagenette` and `datasets--johnowhitaker--imagenette2-320`). ImageNet-V2 is not cached locally.
- **Scientific Decision:** Downloading ImageNet-V2 and running 486,700 inferences on GPU was deliberately excluded in this final audit pass to preserve bounded execution and avoid unsanctioned GPU compute. 
- **Reproduction for External Researchers:** For researchers with CUDA GPU access wishing to evaluate ImageNet-V2 on these models, the evaluation pipeline can be run as follows:
  ```bash
  cd ml
  uv run python scripts/run_eval.py \
      --dataset imagenet_v2_matched_synsets \
      --models resnet50.a1_in1k convnext_tiny.in12k_ft_in1k vit_base_patch16_224.augreg_in21k_ft_in1k swin_tiny_patch4_window7_224.ms_in1k \
      --config results/order_effect_study/matched_fidelity/calibration_config.json \
      --output results/order_effect_study/external_imagenet_v2/
  ```

---

## 5. Summary of Experimental Regimes and Non-Negotiable Demarcations

| Dimension | Regime 1: Nominal-Order Study | Regime 2: Approximate-Fidelity Study |
| :--- | :--- | :--- |
| **Data Artifact** | `results/order_effect_study/full/outputs/` | `results/order_effect_study/matched_fidelity/outputs/` |
| **Parameter Protocol** | Canonical ImageNet-C fixed parameters across sequences | Sequence-specific parameters pre-calibrated to target (PSNR, SSIM) bins |
| **Distortion Balance** | Severely imbalanced (terminal blur smooths JPEG blocks; PSNR gap $\sim 5\text{ dB}$) | Calibrated to common support (cross-sequence gap contracted by $63.18\%$) |
| **Tolerance Status** | N/A (uncontrolled) | **Failed** prespecified tolerances ($\pm 0.75\text{ dB}$, $\pm 0.035$ SSIM); reported honestly |
| **Primary Paper Role** | Documents standard benchmark confound | Investigates residual non-commutativity and gap contraction |
