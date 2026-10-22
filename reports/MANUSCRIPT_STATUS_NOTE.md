# Manuscript Status & Portfolio Distinction Note

**Project:** RobustLens: Non-Commutative Compound-Corruption Benchmarking in Vision Classifiers  
**Primary Working Paper:** `reports/final_research_package/RobustLens_PhD_Working_Paper.pdf` & `RobustLens_PhD_Working_Paper.md` (*"Order Has Consequences: Total Effects and Fidelity-Conditioned Prediction Instability under Compound Image Corruptions"*)  
**Companion 5-Page Paper:** `reports/final_research_package/RobustLens_Final_Paper.pdf` & `RobustLens_Final_Paper.md`  
**Date:** October 2026  
**Author:** Avi  
**Target Venue:** Fall 2027 US Computer Science / Machine Learning PhD Applications & Future Peer Review  

---

### Purpose of this Note
This document explicitly clarifies the academic status, scope of claims, and data provenance of the RobustLens research package for faculty admissions committees, prospective advisors, and reviewers.

---

### 1. Distinction from Peer-Reviewed Conference / Journal Publications
- **Current Status: Independent Research Working Paper & Complete Research Package.**
- This package represents an independent research project conducted by the author to demonstrate research formulation, experimental rigor, statistical error control, and end-to-end technical execution.
- **It has NOT yet been submitted to, peer-reviewed by, or accepted for publication at any conference or journal venue** (such as CVPR, ICCV, ECCV, NeurIPS, ICML, or ICLR).
- It should be cited and reviewed as an **independent research manuscript and working paper draft**, accompanied by open-source code, reproducible GPU artifacts, verified mathematical supplements, and an interactive web dashboard.

---

### 2. Scope of Scientific Claims & Honest Disclosures
To ensure complete transparency and prevent overstatement during critical faculty review:
1. **Two Distinct Regimes:** The benchmark evaluates two separate experimental regimes:
   - **Nominal-Order Regime:** Evaluates standard ImageNet-C fixed parameters across sequences ($486{,}700$ records). Documents massive gaps (up to $21.55\%$), but demonstrates that these gaps are heavily confounded by terminal low-pass filter smoothing.
   - **Approximate-Fidelity Regime:** Evaluates sequence-specific parameters pre-calibrated to common (PSNR, SSIM) support ($486{,}700$ records). Transparently reports that prespecified matching tolerances ($\pm 0.75\text{ dB}$, $\pm 0.035$ SSIM) were **not met** over the tested discrete grid. Quantifies a $63.18\%$ overall gap contraction.
2. **Dataset Scope:** Experiments were conducted on the FastAI Imagenette-v2 benchmark (10 natural WordNet synsets, $N = 3{,}925$ validation images, $N = 9{,}469$ training images). Evaluations on full 1,000-class ImageNet-1k remain necessary before generalizing benchmark conclusions.
3. **No Claim of Universal Impossibility:** The empirical failure of our discrete grid search to achieve joint PSNR/SSIM matching does not prove universal physical impossibility across all continuous parameter spaces; it is framed strictly as failure within the tested discrete grid and parameter families.
4. **No Speculative External Claims:** External validation on ImageNet-V2 is explicitly documented as a computational blocker (not cached locally, avoiding unsanctioned GPU compute) rather than fabricated.

---

### 3. Verification & Artifact Integrity
All empirical numbers, confidence intervals, p-values, calibration errors, and figures are 100% reproducible from immutable prediction records stored in:
- **Final Research Package (`reports/final_research_package/`):**
  - Primary PhD Working Paper: `RobustLens_PhD_Working_Paper.pdf` & `RobustLens_PhD_Working_Paper.md`
  - Companion 5-Page Paper: `RobustLens_Final_Paper.pdf` & `RobustLens_Final_Paper.md`
  - Technical Supplement: `Technical_Supplement.md`
  - Literature Review & Prior-Art Matrix: `LITERATURE_REVIEW_MATRIX.md`
  - Scientific Audit & Issue Resolution Log: `FINAL_REVIEW_ISSUES.md`
  - Originality & Scope Audit: `ORIGINALITY_AND_LIMITATIONS_AUDIT.md`
  - Reproducibility Guide: `REPRODUCIBILITY.md`
  - Figures: `figures/` (Figures 1–5 in 300 DPI)
  - Analysis Tables: `analysis/` (`decomposition_coefficients.csv`, `order_cell_means.csv`, `error_churn_matrix.csv`, `marginal_contrasts.csv`, `nominal_vs_matched_gap_contraction.csv`)
- **Full Nominal-Order Study Artifacts:**
  - `results/order_effect_study/full/outputs/all_order_predictions.csv.gz` ($486{,}700$ records, MD5 verified)
  - `results/order_effect_study/full/outputs/manifest.json`
- **Approximate-Fidelity Study Artifacts:**
  - `results/order_effect_study/matched_fidelity/outputs/all_order_predictions.csv.gz` ($486{,}700$ records, MD5 verified)
  - `results/order_effect_study/matched_fidelity/calibration_config.json` ($N_{\text{cal}} = 50$ training images)
  - `results/order_effect_study/matched_fidelity/matching_residuals.csv`
- **Foundational Benchmark Artifacts:**
  - `results/full_eval/outputs/all_predictions.json` ($251{,}200$ records, 169.5 MB)
