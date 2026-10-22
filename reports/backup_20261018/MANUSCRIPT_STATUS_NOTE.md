# Manuscript Status & Portfolio Distinction Note

**Project:** RobustLens: Evaluating Vision Architecture Robustness and Uncertainty Under Controlled Distribution Shifts  
**Document:** `reports/RobustLens_Research_Manuscript.pdf` & `reports/RobustLens_Research_Manuscript.md`  
**Date:** October 2026  
**Author:** Aviral (Avi)  

---

### Purpose of this Note
This document explicitly clarifies the status of the RobustLens research manuscript for faculty admissions committees, prospective advisors, and reviewers for Fall 2027 US Computer Science / Machine Learning PhD applications.

---

### 1. Distinction from Peer-Reviewed Conference / Journal Publications
- **Current Status: Independent Research Manuscript & Working Paper.**
- The manuscript represents an independent research project conducted by the author to demonstrate research formulation, experimental rigor, statistical error control, and end-to-end technical execution.
- **It has NOT yet been submitted to, peer-reviewed by, or accepted for publication at any conference or journal venue** (such as CVPR, ICCV, ECCV, NeurIPS, ICML, or ICLR).
- It should be cited and reviewed as an **independent research manuscript and working paper draft**, accompanied by open-source code, reproducible Kaggle GPU artifacts, and an interactive dashboard.

---

### 2. Scope of Scientific Claims
To prevent overstatement during critical faculty review:
1. **Scope is Focused on Imagenette-v2 (10 classes):** Sourced from the Hugging Face mirror (`johnowhitaker/imagenette2-320`), the validation split contains exactly rows 9,469 through 13,393 ($N = 3{,}925$ images; per-class counts 357 to 419, sum=3,925), matching the official FastAI validation partition size and classes. While individual image file hashes against the upstream FastAI tarball were not separately computed, the partition size, synset labels, and class distribution correspond strictly to the official validation split. It does not evaluate all 1,000 ImageNet classes or the complete 19-corruption ImageNet-C suite.
2. **Evaluated Architectures:** Results are established specifically for the four open-source checkpoints evaluated: ResNet-50 (`resnet50.a1_in1k`), ConvNeXt-Tiny (`convnext_tiny.in12k_ft_in1k`), ViT-Base/16 (`vit_base_patch16_224.augreg_in21k_ft_in1k`), and Swin-Tiny (`swin_tiny_patch4_window7_224.ms_in1k`). Pretraining dataset scales differ across checkpoints and are acknowledged as confounding factors.
3. **Core Contribution:** The primary scientific merit of the project is its **methodological rigor**: eliminating pseudoreplication in repeated-measures corruption benchmarks via physical image clustering ($N_{\text{clusters}}=3{,}925$), decoupling bootstrap estimation from exact paired sign-flip permutation hypothesis testing ($R=20{,}000$), demonstrating empirical conformal exchangeability breakdown, and evaluating non-commutative compound sequence permutations.

---

### 3. Verification & Artifact Integrity
All empirical numbers, confidence intervals, p-values, calibration errors, and figures are 100% reproducible from immutable prediction records stored in:
- **Original Benchmark (251,200 forward passes, 16 conditions):**
  - Raw Predictions: `results/full_eval/outputs/all_predictions.json` (251,200 records, 169.5 MB)
  - Statistical Synthesis: `results/full_eval/outputs/benchmark_report.json`
  - Pairwise Statistical Table: `results/full_eval/outputs/pairwise_significance.csv`
  - Leaderboard Table: `results/full_eval/outputs/leaderboard_summary.csv`
  - Execution: Kaggle Kernel `aviral199/robustlens-robustness-evaluation`, Version 4 (Tesla T4 GPU).
- **Compound Corruption Order Effects Study (486,700 forward passes, 31 conditions):**
  - Canonical Raw Predictions: `results/order_effect_study/full/outputs/all_order_predictions.csv.gz` (486,700 records, 8.62 MB gzipped)
  - Factorial Clustered ANCOVA: `results/order_effect_study/full/outputs/factorial_ancova_results.csv` ($N_{\text{clusters}} = 3{,}925$, Models 1 & 2)
  - Clustered Main-Effects ANCOVA: `results/order_effect_study/full/outputs/distortion_ancova.csv` ($N_{\text{clusters}} = 3{,}925$)
  - Monte Carlo Sign-Flip Permutation Tests: `results/order_effect_study/full/outputs/order_pairwise_significance.csv` ($R = 20{,}000$)
  - Omnibus Tests: `results/order_effect_study/full/outputs/order_omnibus_tests.json`
  - Sequence Leaderboard: `results/order_effect_study/full/outputs/order_effects_leaderboard.csv` & `order_effects_summary.csv`
  - Statistical Manifest & Synthesis: `results/order_effect_study/full/outputs/manifest.json` & `benchmark_report.json`
  - Execution: Kaggle Kernel `aviral199/robustlens-order-effects-full-benchmark`, Version 2.
