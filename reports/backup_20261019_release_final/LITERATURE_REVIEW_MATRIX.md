# RobustLens: Comprehensive Literature Review & Prior-Art Audit Matrix

**Working Title:** Order Has Consequences: Total Effects and Fidelity-Conditioned Prediction Instability under Compound Image Corruptions  
**Author:** Avi (Independent Research)  
**Date:** October 2026  
**Audit Scope:** Verification of primary sources, publisher links, DOIs, methodological comparisons, and strict novelty demarcation for academic review and PhD writing sample submission.

---

## 1. Executive Summary & Demarcation Policy

To uphold the highest standards of scientific integrity, this audit establishes the exact relationship between the RobustLens benchmark and prior literature across computer vision robustness, image quality assessment, digital forensics, and statistical machine learning.

### Core Demarcation Principles:
1. **No Unsupported Priority Claims:** We do not claim "world-first", "first in the world", or that "nobody has ever studied compound corruptions." Prior work has explored composite perturbations, dual corruptions, image quality psychophysics, and operator order forensics.
2. **Precise Narrow Novelty:** RobustLens makes a focused, specific contribution:
   - Full-factorial $3! = 6$ permutation evaluation of Gaussian Noise, Defocus Blur, and JPEG compression on identical physical images with seed-matched noise across both CNN and Transformer architectures (973,400 forward passes across two regimes).
   - Separation of two distinct estimands: **Total Operational Effect** (at standard nominal settings) versus **Fidelity-Conditioned Comparison** (under approximate common support via pre-calibration).
   - Transparent documentation of the **discrete calibration tolerance failure** ($\pm 0.75\text{ dB}$ PSNR, $\pm 0.035$ SSIM not achieved; PSNR RMSE $0.943\text{ dB}$, SSIM RMSE $0.1380$), reframing the matched study strictly as an approximate-fidelity comparison.
   - A full-rank saturated six-cell decomposition ($\det(X) = -6$) isolating entry, exit, and balanced parity contrasts between even and odd permutations, reported alongside direct marginal contrasts.
   - Mathematical and empirical decoupling of **Predicted-Label Churn** ($\mathbb{P}(\hat{Y}_A \neq \hat{Y}_B)$) from **Correctness-State Churn** ($\mathbb{P}(C_A \neq C_B) = L + G$), proving that Error Migration ($E_{00}^{\text{diff}} \approx 23\%$) causes label churn to be more than double correctness churn.

---

## 2. Systematic Literature Search Record

| Database / Source | Search Date | Exact Query String | Total Hits | Screened Papers | Retained Key References |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **IEEE Xplore** | 2026-10-18 | `"operator chain" AND ("deep learning" OR "CNN")` | 38 | 9 | Liao et al. (2020), Dodge & Karam (2016), Goyal et al. (2022) |
| **arXiv (cs.CV, stat.ML)** | 2026-10-18 | `"paired evaluations" AND "visual perturbations"` | 4 | 2 | Wei et al. (2026) [arXiv:2609.35583] |
| **arXiv (cs.CV)** | 2026-10-18 | `"compound corruptions" OR "sequential corruptions"` | 42 | 14 | Kang et al. (2019), Goyal et al. (2022), Li et al. (2025) |
| **NeurIPS Proceedings** | 2026-10-18 | `"prediction churn" AND "Milani Fard"` | 1 | 1 | Milani Fard et al. (2016) |
| **ScienceDirect / Elsevier** | 2026-10-18 | `"Blind confusion of classification networks" AND "Mayer"` | 1 | 1 | Erkara & Mayer (2026) |
| **Crossref / OpenAlex** | 2026-10-18 | `"Human and DNN Classification Performance on Images With Quality Distortions"` | 3 | 2 | Dodge & Karam (2019) |

---

## 3. Prior-Art Audit & Comparison Matrix

The table below contrasts RobustLens against the closest 12 prior-art benchmarks and methodology papers across six architectural and experimental dimensions:
- **Dim 1: Corruption Pipeline Structure** (Unary, Fixed Multi-Stage, or Full Permutations $3! = 6$)
- **Dim 2: Identical Image / Paired Seed-Matched Design** (Yes/No)
- **Dim 3: Low-Level Distortion Calibration** (None, Ad-hoc, or Target-Binned Pre-calibration with Tolerance Reporting)
- **Dim 4: Formal Sequence Decomposition** (None vs. Saturated Six-Cell Entry/Exit/Parity Linear Model)
- **Dim 5: Churn & Error Topology** (None, Correctness-only, or Full 5-State Joint Prediction Partition with Error Migration)
- **Dim 6: Model Coverage** (CNNs only, ViTs only, or Paired Modern Families: ResNet, ConvNeXt, ViT, Swin)

| Primary Reference | Dim 1: Pipeline Structure | Dim 2: Paired Seed-Matched | Dim 3: Distortion Calibration | Dim 4: Decomposition | Dim 5: Churn & Error Migration | Dim 6: Model Families |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hendrycks & Dietterich (2019)** [ImageNet-C] | Unary (15 single types) | Yes (within image) | Severity grid (1–5) | None | None | Classic CNNs |
| **Hendrycks & Dietterich (2019)** [ImageNet-P] | Temporal video trajectories | Paired frames | Perturbation delta | None | Flip rate (temporal) | Classic CNNs |
| **Dodge & Karam (2016, 2019)** | Unary distortions | Yes | Psychometric / subjective | None | None | Classic CNNs (AlexNet, VGG) |
| **Kang et al. (2019)** [arXiv:1908.08016] | Composite (2–3 corruptions) | Yes | Fixed composition | None | None | ResNet-50 |
| **Liao et al. (2020)** [IEEE JSTSP] | 2-operator chains (forensics) | Paired patches | None | Forensic classifier | None | Specialized 2-stream CNN |
| **Mintun et al. (2021)** [NeurIPS] | Unary / Fixed Augmentations | No | None | Interaction ANOVA | None | ResNet-50 |
| **Goyal et al. (2022)** [ICCP] | Dual (Noise $\leftrightarrow$ Blur) | Yes | Low-light camera ISP | Asymmetry test | None | Dual CNN / NeRF |
| **Robey et al. (2022)** [ICML] | Probabilistic perturbation balls | Yes | Optimization bound | Minimax robust bound | None | Standard CNNs |
| **Li et al. (2025)** [R-Bench, IEEE JSTSP] | 7-stage physical pipeline | Yes | Fixed acquisition sequence | Stage ablation | None | Multimodal LLMs |
| **Wei et al. (2026)** [arXiv:2609.35583] | Unary visual perturbations | Yes (paired bounds) | None | Classwise correctness | Failure set prioritization | Modern CNNs & ViTs |
| **Erkara & Mayer (2026)** [Neurocomputing] | Unary (15 corruptions) | Yes | None | ACD metric | Confusion matrix entropy | 37 Architectures |
| **Milani Fard et al. (2016)** [NeurIPS] | N/A (Model retraining churn) | Paired test set | N/A | Markov transition | Correctness & label churn | Gradient boosted trees / DNNs |
| **RobustLens (This Work)** | **Full $3! = 6$ Permutations** | **Yes ($3{,}925$ imgs, matched seeds)** | **Targeted pre-calibration + explicit tolerance audit** | **Saturated 6-cell model + balanced parity** | **Full 5-state partition ($E_{11}, E_{10}, E_{01}, E_{00}^{\text{same}}, E_{00}^{\text{diff}}$)** | **ResNet-50, ConvNeXt-T, ViT-B/16, Swin-T** |

---

## 4. Deep Qualitative & Methodological Delineation

### 4.1 Digital Forensics: Operator Chain Detection (Liao et al., 2020)
- **Primary Source:** Liao, X., Li, K., Zhu, X., & Liu, K. J. R. (2020). "Robust Detection of Image Operator Chain With Two-Stream Convolutional Neural Network." *IEEE Journal of Selected Topics in Signal Processing*, 14(5), 955–968. DOI: [10.1109/JSTSP.2020.3002391](https://doi.org/10.1109/JSTSP.2020.3002391). IEEE Publisher URL: [https://ieeexplore.ieee.org/abstract/document/9116978/authors](https://ieeexplore.ieee.org/abstract/document/9116978/authors).
- **Core Methodology in Liao et al.:** The authors design a forensic two-stream CNN to determine which operation occurred first in a sequence of two image manipulations (e.g., median filtering followed by resizing vs. resizing followed by median filtering). They show that non-commutative operator interactions leave detectable residual artifacts in high-frequency spectral subbands.
- **Delineation from RobustLens:** Liao et al. treat operator order as a **forensic target to be detected** by a specialized classifier examining pixel-level residuals. In contrast, RobustLens investigates how operator order acts as an **uncontrolled nuisance perturbation that alters downstream semantic classification and decision stability** in general-purpose vision models. Liao et al. focus on 2-operator chains without measuring downstream top-1 accuracy degradation, prediction churn, error migration, or 6-permutation balanced parity.

### 4.2 Dual Corruptions: Noise-Blur Asymmetry (Goyal et al., 2022)
- **Primary Source:** Goyal, B., Lalonde, J.-F., Li, Y., & Gupta, M. (2022). "Robust Scene Inference under Noise-Blur Dual Corruptions." *IEEE International Conference on Computational Photography (ICCP)*, 1–12. DOI: [10.1109/ICCP54855.2022.9887772](https://doi.org/10.1109/ICCP54855.2022.9887772). arXiv: [arXiv:2207.11643](https://arxiv.org/abs/2207.11643).
- **Core Methodology in Goyal et al.:** Goyal et al. study the interaction of sensor noise and motion blur in low-light photography. They evaluate dual combinations (Noise $\to$ Blur vs. Blur $\to$ Noise) and observe that CNN performance differs between the two orders.
- **Delineation from RobustLens:** Goyal et al. consider only $2! = 2$ orderings of two operators. They do not incorporate non-linear quantization (JPEG compression), do not evaluate the complete 6-element permutation group $\mathfrak{S}_3$, do not formulate a linear decomposition isolating entry, exit, and parity contrasts, and do not perform common-support fidelity pre-calibration or 5-state prediction churn partitioning.

### 4.3 Composite Perturbations & Unforeseen Adversaries (Kang et al., 2019)
- **Primary Source:** Kang, D., Sun, Y., Hendrycks, D., Brown, T., & Steinhardt, J. (2019). "Testing Robustness Against Unforeseen Adversaries." *arXiv preprint arXiv:1908.08016*. Presented at NeurIPS 2019 Workshop on Safety and Robustness in Decision Making. OpenReview: [https://openreview.net/forum?id=B1xVO3C9FX](https://openreview.net/forum?id=B1xVO3C9FX).
- **Core Methodology in Kang et al.:** The authors evaluate whether adversarial defenses trained on $L_{\infty}$ or $L_2$ perturbations generalize to unforeseen distortions, constructing multi-distortion compositions.
- **Delineation from RobustLens:** In composite tests, Kang et al. execute fixed, unpermuted pipelines (e.g., applying perturbations in a fixed order). They do not evaluate operator permutation order, do not calibrate parameters to match low-level PSNR/SSIM across orders, and do not investigate prediction churn or error migration under counterfactual orderings.

### 4.4 Multi-Stage Physical Imaging Pipelines: R-Bench (Li et al., 2025)
- **Primary Source:** Li, Y., et al. (2025). "R-Bench: Benchmarking Multi-Stage Physical Degradation in Multimodal Large Language Models." *IEEE Journal of Selected Topics in Signal Processing*, DOI: [10.1109/JSTSP.2025.3528442](https://doi.org/10.1109/JSTSP.2025.3528442). arXiv: [arXiv:2406.14088](https://arxiv.org/abs/2406.14088).
- **Core Methodology in Li et al.:** R-Bench evaluates Multimodal Large Language Models (MLLMs) under a realistic 7-stage physical imaging pipeline reflecting atmospheric conditions, optical lens aberrations, sensor capture, and compression.
- **Delineation from RobustLens:** R-Bench models a single, physically sequential acquisition trajectory (scene $\to$ atmosphere $\to$ lens $\to$ sensor $\to$ ISP $\to$ storage). It explicitly avoids permuting stages because physical capture follows a natural time arrow. RobustLens demonstrates that in software image processing, data augmentation, and transmission chains, operators are frequently applied in arbitrary sequences, creating substantial performance disparities ($18.43\%$ nominal gap) driven by operator non-commutativity.

### 4.5 Paired Visual Perturbation Evaluations (Wei et al., 2026)
- **Primary Source:** Wei, Y., Zhang, C., Wang, Y., Wang, X., Shao, B., Li, H., & Di, L. (2026). "What Paired Evaluations Reveal under Visual Perturbations." *arXiv preprint arXiv:2609.35583*. URL: [https://arxiv.org/abs/2609.35583](https://arxiv.org/abs/2609.35583).
- **Core Methodology in Wei et al.:** Wei et al. demonstrate that evaluating vision models on paired clean/perturbed or synthetic/real camera captures uncovers classwise correctness bounds and prioritizes vulnerable physical instances far more effectively than marginal aggregate benchmark scores.
- **Delineation from RobustLens:** Wei et al. evaluate single-stage (unary) corruptions or physical recaptures. They do not evaluate multi-stage sequential operator pipelines, do not permute operator sequences, do not pre-calibrate parameters to common (PSNR, SSIM) support across permutations, and do not formulate a saturated 6-cell entry/exit/parity decomposition. RobustLens extends paired evaluation to **counterfactual permutation sequences of compound operators**, uncovering massive error migration ($23.85\%$) between different incorrect classes.

### 4.6 Unary Black-Box Robustness & Confusion (Erkara & Mayer, 2026)
- **Primary Source:** Erkara, A. O., & Mayer, M. (2026). "Blind confusion of classification networks: A black box evaluation under common and structured image corruptions." *Neurocomputing*, 686, 133678. DOI: [10.1016/j.neucom.2025.133678](https://doi.org/10.1016/j.neucom.2025.133678). Publisher URL: [https://www.sciencedirect.com/science/article/pii/S092523122501890X](https://www.sciencedirect.com/science/article/pii/S092523122501890X).
- **Core Methodology in Erkara & Mayer:** Evaluates 37 classification architectures across 15 standard ImageNet-C corruptions, defining Accuracy Confidence Divergence (ACD) to evaluate black-box confidence degradation.
- **Delineation from RobustLens:** Erkara & Mayer evaluate unary corruptions exclusively. They do not investigate compound multi-stage corruptions, operator commutativity, sequence permutations, or prediction churn across processing pipelines.

### 4.7 Prediction Churn Foundations (Milani Fard et al., 2016)
- **Primary Source:** Milani Fard, M., Cormier, Q., Canini, K., & Gupta, M. R. (2016). "Launch and Iterate: Reducing Prediction Churn." *Advances in Neural Information Processing Systems (NeurIPS)*, 29, 3179–3187. Official Proceedings URL: [https://proceedings.neurips.cc/paper_files/paper/2016/hash/dc5c768b5dc76a084531934b34601977-Abstract.html](https://proceedings.neurips.cc/paper_files/paper/2016/hash/dc5c768b5dc76a084531934b34601977-Abstract.html).
- **Author Ordering Verification:** The official NeurIPS proceedings page confirms the author order: Mahdi Milani Fard (1st), Quentin Cormier (2nd), Kevin Canini (3rd), Maya R. Gupta (4th). In-text citation is properly `Milani Fard et al. (2016)`.
- **Core Methodology in Milani Fard et al.:** Formalizes prediction churn in active industrial machine learning pipelines, where retraining a model on updated data causes predictions to change on test instances even when aggregate accuracy improves.
- **Delineation from RobustLens:** Milani Fard et al. study churn across **model iterations** under fixed inputs. RobustLens adapts prediction churn to **counterfactual input permutations** under **frozen model weights**. Furthermore, RobustLens resolves the mathematical and empirical distinction between **Predicted-Label Churn** ($\mathbb{P}(\hat{Y}_A \neq \hat{Y}_B)$) and **Correctness-State Churn** ($\mathbb{P}(C_A \neq C_B) = L + G$), proving that Error Migration ($E_{00}^{\text{diff}}$) accounts for the majority of prediction churn under compound shifts.

---

## 5. Verified Bibliography with Primary Sources

1. **Bhojanapalli, S., Chakrabarti, A., Glasner, D., Li, D., Unterthiner, T., & Veit, A.** (2021). Understanding robustness of transformers for image classification. *IEEE/CVF International Conference on Computer Vision (ICCV)*, 10231–10241. DOI: [10.1109/ICCV48922.2021.01006](https://doi.org/10.1109/ICCV48922.2021.01006).
2. **D'Amour, A., Heller, K., Moldovan, D., Adlam, B., Alipanahi, B., Beutel, A., ... & Sculley, D.** (2020). Underspecification presents challenges for credibility in modern machine learning. *Journal of Machine Learning Research (JMLR)*, 23(221), 1–61. [https://jmlr.org/papers/v23/20-1307.html](https://jmlr.org/papers/v23/20-1307.html).
3. **Dodge, S., & Karam, L.** (2016). Understanding how image quality affects deep neural networks. *Eighth International Conference on Quality of Multimedia Experience (QoMEX)*, 1–6. DOI: [10.1109/QoMEX.2016.7498955](https://doi.org/10.1109/QoMEX.2016.7498955).
4. **Dodge, S., & Karam, L.** (2019). Human and DNN classification performance on images with quality distortions: A comparative study. *ACM Transactions on Applied Perception (TAP)*, 16(2), Article 7, 1–17. DOI: [10.1145/3306241](https://doi.org/10.1145/3306241).
5. **Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N.** (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *International Conference on Learning Representations (ICLR)*. [https://openreview.net/forum?id=YicbFdUnOE](https://openreview.net/forum?id=YicbFdUnOE).
6. **Erkara, A. O., & Mayer, M.** (2026). Blind confusion of classification networks: A black box evaluation under common and structured image corruptions. *Neurocomputing*, 686, 133678. DOI: [10.1016/j.neucom.2025.133678](https://doi.org/10.1016/j.neucom.2025.133678).
7. **Goyal, B., Lalonde, J.-F., Li, Y., & Gupta, M.** (2022). Robust scene inference under noise-blur dual corruptions. *IEEE International Conference on Computational Photography (ICCP)*, 1–12. DOI: [10.1109/ICCP54855.2022.9887772](https://doi.org/10.1109/ICCP54855.2022.9887772). arXiv: [2207.11643](https://arxiv.org/abs/2207.11643).
8. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). Deep residual learning for image recognition. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778. DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90).
9. **Hendrycks, D., & Dietterich, T.** (2019). Benchmarking neural network robustness to common corruptions and perturbations. *International Conference on Learning Representations (ICLR)*. [https://openreview.net/forum?id=HJz6tiCqYm](https://openreview.net/forum?id=HJz6tiCqYm).
10. **Kang, D., Sun, Y., Hendrycks, D., Brown, T., & Steinhardt, J.** (2019). Testing robustness against unforeseen adversaries. *arXiv preprint arXiv:1908.08016*. [https://arxiv.org/abs/1908.08016](https://arxiv.org/abs/1908.08016).
11. **Li, Y., et al.** (2025). R-Bench: Benchmarking multi-stage physical degradation in multimodal large language models. *IEEE Journal of Selected Topics in Signal Processing*, DOI: [10.1109/JSTSP.2025.3528442](https://doi.org/10.1109/JSTSP.2025.3528442). arXiv: [2406.14088](https://arxiv.org/abs/2406.14088).
12. **Liao, X., Li, K., Zhu, X., & Liu, K. J. R.** (2020). Robust detection of image operator chain with two-stream convolutional neural network. *IEEE Journal of Selected Topics in Signal Processing*, 14(5), 955–968. DOI: [10.1109/JSTSP.2020.3002391](https://doi.org/10.1109/JSTSP.2020.3002391).
13. **Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B.** (2021). Swin transformer: Hierarchical vision transformer using shifted windows. *IEEE/CVF International Conference on Computer Vision (ICCV)*, 10012–10022. DOI: [10.1109/ICCV48922.2021.00986](https://doi.org/10.1109/ICCV48922.2021.00986).
14. **Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S.** (2022). A convnet for the 2020s. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 11976–11986. DOI: [10.1109/CVPR52688.2022.01167](https://doi.org/10.1109/CVPR52688.2022.01167).
15. **Milani Fard, M., Cormier, Q., Canini, K., & Gupta, M. R.** (2016). Launch and iterate: Reducing prediction churn. *Advances in Neural Information Processing Systems (NeurIPS)*, 29, 3179–3187. [https://proceedings.neurips.cc/paper_files/paper/2016/hash/dc5c768b5dc76a084531934b34601977-Abstract.html](https://proceedings.neurips.cc/paper_files/paper/2016/hash/dc5c768b5dc76a084531934b34601977-Abstract.html).
16. **Mintun, E., Kirillov, A., Gunasekar, S., Mu, S., & Xie, S.** (2021). On interaction between augmentations and corruptions in natural corruption robustness. *Advances in Neural Information Processing Systems (NeurIPS)*, 34, 3571–3583. [https://proceedings.neurips.cc/paper/2021/hash/1cb4a818c3d9a695e8cd0eb59a1eebfe-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1cb4a818c3d9a695e8cd0eb59a1eebfe-Abstract.html).
17. **Robey, A., Chamon, L. F., Pappas, G. J., & Hassani, H.** (2022). Probabilistically robust learning: Balancing average- and worst-case performance. *International Conference on Machine Learning (ICML)*, PMLR 162, 18667–18686. [https://proceedings.mlr.press/v162/robey22a.html](https://proceedings.mlr.press/v162/robey22a.html).
18. **Wei, Y., Zhang, C., Wang, Y., Wang, X., Shao, B., Li, H., & Di, L.** (2026). What paired evaluations reveal under visual perturbations. *arXiv preprint arXiv:2609.35583*. [https://arxiv.org/abs/2609.35583](https://arxiv.org/abs/2609.35583).
