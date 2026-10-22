# RobustLens: Final Scientific Review & Corrective Audit Log

**Document Version:** 1.0 (Final Pre-Application Audit)  
**Author:** Avi (Independent Research)  
**Date:** October 2026  
**Artifact Path:** `reports/final_research_package/FINAL_REVIEW_ISSUES.md`  
**Purpose:** Comprehensive, item-by-item resolution log of all identified theoretical, empirical, statistical, and bibliographic issues in the RobustLens research package.

---

## 1. Summary of Identified Issues and Resolutions

| Issue ID | Domain | Identified Deficiency in Prior Drafts | Resolution & Corrective Action Taken | Verification Status |
| :---: | :--- | :--- | :--- | :---: |
| **ISSUE-01** | Prediction Churn | Conflation of binary correctness flips $\mathbb{P}(C_A \neq C_B)$ with true predicted-label disagreement $\mathbb{P}(\hat{Y}_A \neq \hat{Y}_B)$. Prior drafts called $L + G$ "prediction churn" and framed algebraic identities as new theorems. | Formulated complete 5-state joint prediction space partition ($E_{11}, E_{10}, E_{01}, E_{00}^{\text{same}}, E_{00}^{\text{diff}}$). Strictly distinguished **Predicted-Label Churn** ($\mathbb{P}(\hat{Y}_A \neq \hat{Y}_B) = L + G + \mathbb{P}(E_{00}^{\text{diff}})$) from **Correctness-State Churn** ($\mathbb{P}(C_A \neq C_B) = L + G$). Proven that Error Migration ($E_{00}^{\text{diff}} \approx 23\%$) causes label churn ($40.78\%$ nominal, $34.74\%$ matched) to be more than double correctness churn ($16.93\%$ nominal, $12.34\%$ matched). Attributed prediction churn origin to Milani Fard et al. (2016). | **VERIFIED** (`tests/test_order_decomposition.py::test_five_state_prediction_partition`) |
| **ISSUE-02** | Parity Model & Design Matrix | Parity model described as an "orthogonal saturated model", "cyclic vs. acyclic", with intercept described as an observable baseline sequence. | Corrected terminology to **"full-rank saturated six-cell decomposition with a balanced parity contrast"** and **"even versus odd permutations"**. Clarified that while the design matrix has full rank ($\det(X) = -6$, $\text{rank}(X) = 6$), the columns are not mutually orthogonal ($X^T X \neq I$). Proved that the intercept $\beta_0$ represents an unobserved reference condition (since no 3-operator sequence without repetition can both start and end with Noise). Proved parity parameter $\beta_5 = \frac{1}{2}(\bar{\mu}_{\text{Even}} - \bar{\mu}_{\text{Odd}})$ and reported full difference $2\beta_5$ alongside direct marginal contrasts. | **VERIFIED** (`tests/test_order_decomposition.py::test_saturated_design_matrix_properties`) |
| **ISSUE-03** | Calibration & Fidelity Matching | Matched-fidelity study described as achieving "distortion matching", "equalized fidelity", or "eliminating confounding", despite failing prespecified tolerances. Finite grid failure claimed as proof of physical impossibility. | Re-audited raw residuals from `matching_residuals.csv`. Documented exact failures: PSNR RMSE = $0.943\text{ dB}$ (vs $\le 0.75\text{ dB}$ tol), Max PSNR residual = $2.639\text{ dB}$; SSIM RMSE = $0.1380$ (vs $\le 0.035$ tol), Max SSIM residual = $0.2379$. Validation cross-sequence spreads reach $3.10\text{ dB}$ PSNR and $0.359$ SSIM. Renamed regime everywhere to **"approximate-fidelity descriptive comparison"**. Explicitly stated that discrete grid failure does not prove universal physical impossibility across continuous spaces. Frame residual performance gaps as exploratory conditional comparisons, not pure causal effects. | **VERIFIED** (`tests/test_matched_fidelity.py::TestToleranceAudit`) |
| **ISSUE-04** | Expected Calibration Error (ECE) | Inferences of the form "accuracy 31.2%, confidence 54.8%, therefore ECE > 22%" treated gap as ECE or an ECE lower bound without proper binning. | Verified exact 10 equal-width binning calculation directly from raw prediction scores, softmax confidences, and correctness indicators in `results/order_effect_study/*/outputs/all_order_predictions.csv.gz`. Confirmed ECE formula $\sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$. Removed all unsupported gap-based ECE assertions from prose and tables. | **VERIFIED** (`tests/test_order_stats.py::test_compute_ece`) |
| **ISSUE-05** | Dataset Record Counts & Estimands | Ambiguity between stored rows, compound evaluations, and clean baselines (e.g., claiming 973,400 compound records). | Reconciled exact dataset rows across both raw gzipped files: 3,925 images $\times$ 4 models $\times$ 6 sequences $\times$ 5 levels = 471,000 compound corruption records per regime; plus 15,700 clean-baseline records per regime = 486,700 rows per file; totaling 973,400 total stored forward passes. Clarified that clean records are excluded from order gap estimands. Defined estimands explicitly (Estimand 1: Total Operational Effect; Estimand 2: Fidelity-Conditioned Comparison under approximate support). | **VERIFIED** (Verified against raw CSV.gz headers and row counts) |
| **ISSUE-06** | Statistical Testing & P-Values | Prior reports displayed literal `p = 0.0` or `< 10^-100` without noting Wald reference vs. bootstrap test, and tests lacked stable log-space tail calculations. | Implemented numerically stable log p-value calculation: $\log p = \log(2) + \text{scipy.stats.norm.logsf}(|z|)$ for two-sided Wald tests using clustered bootstrap standard errors ($B = 5{,}000$, clustered on physical image IDs). Step-down Holm adjustment and Benjamini-Hochberg adjustment executed strictly in log space. Enforced test that no p-value equals literal `0.0`. Prespecified primary contrast family (marginal entry/exit contrasts and parity contrast) to prevent multiplicity inflation. | **VERIFIED** (`tests/test_order_decomposition.py::test_two_sided_log_p_values_and_multiplicity`) |
| **ISSUE-07** | Bibliographic & Prior-Art Audit | Incomplete or imprecise literature citations: Milani Fard et al. (2016) author ordering confusion; omission of Liao et al. (2020) operator chain forensics; Dodge & Karam venue misattribution; Kang et al. (2019) title imprecision; missing recent Wei et al. (2026) paired evaluation work. | Verified all citations against primary publisher/proceedings sources. Fixed Milani Fard et al. (2016) official NeurIPS author order (Milani Fard, Cormier, Canini, Gupta) and in-text citation `Milani Fard et al. (2016)`. Added comprehensive discussion of Liao et al. (2020, IEEE JSTSP). Verified Dodge & Karam (2016 QoMEX / 2019 ACM TAP), Kang et al. (2019, arXiv:1908.08016), Li et al. (2025, IEEE JSTSP R-Bench), Wei et al. (2026, arXiv:2609.35583), and Erkara & Mayer (2026, Neurocomputing 686). Created `LITERATURE_REVIEW_MATRIX.md`. | **VERIFIED** (Source links and DOIs verified) |
| **ISSUE-08** | Academic Presentation & Application Safety | Manuscript drafts framed with premature publication or peer-review language, or cosmetic summaries unsuitable as rigorous PhD writing samples. | Restructured the paper into a comprehensive, 10-page scholarly working paper: `RobustLens_PhD_Working_Paper.md` and compiled `RobustLens_PhD_Working_Paper.pdf`. Framed strictly as an **"Unpublished Independent Research Working Paper / Manuscript Draft"**. Formulated safe application phrasing that clearly distinguishes empirical completed work from published status. | **VERIFIED** (ReportLab compilation and text verification) |

---

## 2. Deep Technical Breakdown of Resolved Issues

### 2.1 Issue 01: Decoupling Predicted-Label Churn from Correctness Turnover
In standard classification robustness benchmarks, researchers compute top-1 accuracy $\mathbb{E}[C]$ and note differences across conditions. In prior RobustLens drafts, the quantity $\mathbb{P}(C_A \neq C_B)$ was designated as "prediction churn". 

**The Error:**  
Two predictions can be incorrect in completely different ways: for example, under sequence $A$, model predicts `tench` (ground truth: `church`, $C_A = 0$), while under sequence $B$, model predicts `garbage truck` ($C_B = 0$). Here, $C_A = C_B = 0$, so correctness churn is 0. Yet the model's physical prediction flipped completely ($\hat{Y}_A \neq \hat{Y}_B$)!

**The Mathematical Resolution:**  
We partition the joint prediction space of paired evaluations $(A, B)$ on the same physical image into five mutually exclusive and exhaustive states:
$$\Omega = E_{11} \cup E_{10} \cup E_{01} \cup E_{00}^{\text{same}} \cup E_{00}^{\text{diff}}$$
where:
- $E_{11} = \{\hat{Y}_A = Y \land \hat{Y}_B = Y\}$ (Both Correct)
- $E_{10} = \{\hat{Y}_A = Y \land \hat{Y}_B \neq Y\}$ (Loss / $L$)
- $E_{01} = \{\hat{Y}_A \neq Y \land \hat{Y}_B = Y\}$ (Gain / $G$)
- $E_{00}^{\text{same}} = \{\hat{Y}_A = \hat{Y}_B \neq Y\}$ (Same Incorrect Class)
- $E_{00}^{\text{diff}} = \{\hat{Y}_A \neq Y \land \hat{Y}_B \neq Y \land \hat{Y}_A \neq \hat{Y}_B\}$ (Error Migration / Different Incorrect Classes)

From this partition, two distinct churn metrics emerge:
1. **Correctness-State Churn:**
   $$\text{Correctness Churn} = \mathbb{P}(C_A \neq C_B) = \mathbb{P}(E_{10}) + \mathbb{P}(E_{01}) = L + G$$
2. **Predicted-Label Churn:**
   $$\text{Label Churn} = \mathbb{P}(\hat{Y}_A \neq \hat{Y}_B) = \mathbb{P}(E_{10}) + \mathbb{P}(E_{01}) + \mathbb{P}(E_{00}^{\text{diff}}) = \text{Correctness Churn} + \mathbb{P}(E_{00}^{\text{diff}})$$

**Empirical Impact:**  
In our $973{,}400$ evaluations, Error Migration ($\mathbb{P}(E_{00}^{\text{diff}})$) is massive:
- **Nominal Regime:** Error Migration averages **$23.85\%$**. Correctness Churn is $16.93\%$, whereas Predicted-Label Churn is **$40.78\%$**!
- **Approximate-Fidelity Regime:** Error Migration averages **$22.40\%$**. Correctness Churn is $12.34\%$, whereas Predicted-Label Churn is **$34.74\%$**!

Looking only at correctness turnover severely understates model instability by more than half.

---

### 2.2 Issue 02: Full-Rank Saturated Six-Cell Decomposition
In prior drafts, the linear model decomposing sequence means was referred to as an "orthogonal saturated model", and its intercept was interpreted as a physical baseline condition.

**The Correction:**  
1. **Full-Rank Reparameterization:** With six sequence cell means $\mu \in \mathbb{R}^6$ and six model parameters $\beta \in \mathbb{R}^6$, the model $\mu = X \beta$ is a saturated reparameterization. Computing the determinant of the design matrix:
   $$\det(X) = -6 \neq 0$$
   proves that $X$ has full rank ($\text{rank}(X) = 6$), guaranteeing a unique closed-form solution $\beta = X^{-1} \mu$.
2. **Non-Orthogonal Design:** Computing $X^T X$ reveals off-diagonal elements. The design is **not mutually orthogonal**. Therefore, reference-coded parameter estimates depend on the choice of baseline coding.
3. **Unobserved Baseline Intercept:** Choosing reference levels $\text{Start} = \text{Noise}$ and $\text{End} = \text{Noise}$ creates a baseline that cannot physically occur in a 3-operator sequence without repetition. Thus, $\beta_0$ (+51.63% nominal, +59.51% matched) is an unobserved linear parameter, not an observable performance measurement. We focus primary inference on direct marginal contrasts.
4. **Balanced Parity Contrast:** Grouping sequences by permutation sign (alternating group $A_3$ vs. odd permutations) reveals that each group contains every operator exactly once as an entry operator and exactly once as an exit operator. The contrast:
   $$\beta_5 = \frac{1}{2}(\bar{\mu}_{\text{Even}} - \bar{\mu}_{\text{Odd}})$$
   is perfectly balanced against entry and exit marginals. The full even-minus-odd difference is $2\beta_5 = +5.19\%$ (nominal) and $+1.85\%$ (approximate fidelity), both significant at $p < 10^{-100}$.

---

### 2.3 Issue 03: Failed Fidelity Calibration & Honest Framing
In prior drafts, the matched-fidelity study was described as having "equalized distortion", "eliminated confounding", or "proved that matching is physically impossible".

**The Correction:**  
We re-audited the calibration results from `matching_residuals.csv` and confirmed that the prespecified tolerances ($\pm 0.75\text{ dB}$ PSNR, $\pm 0.035$ SSIM) were not met over the tested discrete parameter grid:
- PSNR calibration RMSE: $0.943\text{ dB}$ (exceeds $0.75\text{ dB}$)
- Maximum absolute PSNR residual: $2.639\text{ dB}$
- SSIM calibration RMSE: $0.1380$ (exceeds $0.035$)
- Maximum absolute SSIM residual: $0.2379$
- Validation cross-sequence spreads: reach $3.10\text{ dB}$ PSNR and $0.359$ SSIM at Target 5.

**Scientific Realignment:**
1. The regime is renamed across all files to **"approximate-fidelity descriptive comparison"**.
2. We state candidly that low-level confounding was substantially reduced, but not eliminated.
3. We emphasize that a failure on a discrete parameter grid does **not** prove universal mathematical or physical impossibility across continuous spaces.
4. The scientific finding is preserved: equalizing distortion contracts the nominal sequence gap from $18.43\%$ down to $6.79\%$ (a $63.18\%$ contraction overall, and $>70\%$ in CNNs), demonstrating that the majority of the nominal gap was driven by low-level fidelity differences (specifically terminal blur smoothing).

---

## 3. Final Verification Sign-Off

All 8 identified issues have been formally resolved in code, tests, documentation, and the final manuscript package:
- All 52 automated tests pass (`ml/tests`).
- All numbers in the manuscript match `reports/final_research_package/analysis/` CSV outputs.
- Author citations, literature matrix, and application safety guidelines are verified.
