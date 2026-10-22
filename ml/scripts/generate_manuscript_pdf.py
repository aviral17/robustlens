"""Generate publication-grade PDF manuscript for RobustLens with ReportLab.

Includes all sections, audited results tables (Tables 1-5), and Figures 1-9
derived from verified full-benchmark artifacts (N=3,925 images, 737,900 predictions).
Clean ASCII and HTML typographic formatting; zero missing glyphs; verified metadata.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        if self._pageNumber > 1:
            self.drawString(54, 750, "RobustLens: Statistical Evaluation of Vision Models Under Distribution Shift")
            self.drawRightString(612 - 54, 750, "Academic Research Manuscript")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 612 - 54, 744)

        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 36, page_str)
        self.drawString(54, 36, "RobustLens Working Paper • Independent Research Draft")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 46, 612 - 54, 46)
        self.restoreState()


def build_manuscript_pdf(output_pdf_path=None):
    if output_pdf_path is None:
        output_pdf_path = os.path.join(REPO_ROOT, "reports", "RobustLens_Research_Manuscript.pdf")

    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
        title="RobustLens: Statistical Evaluation of Vision Models Under Distribution Shift",
        author="Aviral (Avi)",
        subject="Empirical Robustness, Uncertainty Quantification, and Distribution Shift in Computer Vision"
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_CENTER,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        alignment=TA_CENTER,
        spaceAfter=8
    )

    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748b"),
        alignment=TA_CENTER,
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12.5,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#1e293b"),
        alignment=TA_JUSTIFY,
        spaceAfter=5
    )

    abstract_style = ParagraphStyle(
        "AbstractText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#1e293b"),
        alignment=TA_JUSTIFY,
        leftIndent=14,
        rightIndent=14,
        spaceAfter=8
    )

    caption_style = ParagraphStyle(
        "CaptionStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor("#475569"),
        alignment=TA_CENTER,
        spaceBefore=3,
        spaceAfter=6,
        keepWithNext=True
    )

    table_header_style = ParagraphStyle(
        "TH",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=6.8,
        leading=8.5,
        textColor=colors.white,
        alignment=TA_CENTER
    )

    table_cell_style = ParagraphStyle(
        "TC",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.5,
        leading=8,
        textColor=colors.HexColor("#1e293b"),
        alignment=TA_CENTER
    )

    table_cell_left = ParagraphStyle(
        "TCL",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.5,
        leading=8,
        textColor=colors.HexColor("#1e293b"),
        alignment=TA_LEFT
    )

    ref_style = ParagraphStyle(
        "RefStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.5,
        leading=8.5,
        textColor=colors.HexColor("#334155"),
        leftIndent=12,
        firstLineIndent=-12,
        spaceAfter=2.5
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("RobustLens: Clustered Statistical Inference and Operator Non-Commutativity in Vision Robustness", title_style))
    story.append(Paragraph("A Controlled Evaluation of Inductive Biases, Predictive Uncertainty, and Corruption Permutations", subtitle_style))
    story.append(Paragraph(
        "<b>Aviral (Avi)</b> &bull; Independent Research Working Paper<br/>"
        "Artifact Scope: FastAI Imagenette-v2 Validation Split (N=3,925 images) &bull; 737,900 Total Forward Passes",
        meta_style
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f172a"), spaceAfter=8))

    # Abstract
    story.append(Paragraph("<b>Abstract</b>", h2_style))
    abstract_text = (
        "Standard computer vision benchmarks assess out-of-distribution (OOD) robustness by pooling corrupted image-condition "
        "observations, introducing pseudoreplication that deflates standard errors by a factor of approximately &radic;K. Furthermore, conventional benchmarks "
        "evaluate image corruptions primarily as isolated unary operations or fixed pipelines, neglecting the non-commutative nature of physical image degradation cascades. "
        "In this study, we present <b>RobustLens</b>, an audited robustness evaluation framework deployed on the FastAI Imagenette-v2 validation partition (N=3,925 images). "
        "We decouple statistical estimation from hypothesis testing by pairing <b>image-level clustered bootstrap confidence intervals</b> (B=10,000) "
        "with <b>Monte Carlo paired sign-flip permutation tests</b> (R=20,000) under Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR) multiplicity control. "
        "Evaluating four canonical architectures (ResNet-50, ConvNeXt-Tiny, ViT-Base/16, and Swin-Tiny across 251,200 forward passes) demonstrates that clean accuracy "
        "fails to predict shifted resilience: ResNet-50 and ConvNeXt-Tiny achieve indistinguishable clean performance (87.36% vs. 87.13%, McNemar p=0.6402), "
        "yet ConvNeXt-Tiny establishes a +5.03 percentage-point advantage under distribution shift (95% Clustered CI: [+4.47%, +5.57%], permutation p &lt; 0.0001, z=17.34). "
        "Extending this to <b>compound sequential corruptions</b> across 486,700 forward passes (6 permutations &times; 5 severities), we evaluate how digital degradation operators "
        "interact: varying execution order of identical perturbations (Noise, Blur, JPEG) induces an empirical PSNR spread of 6.29 dB and SSIM spread of 0.421, "
        "with maximum sequence accuracy gaps of 17.13% to 21.55% across models. Multi-variable cluster-robust logistic regression (N_clusters=3,925) confirms that permutation sequence "
        "remains statistically associated with classification correctness (Wald p &lt; 1e-100) after conditioning on continuous PSNR and SSIM."
    )
    story.append(Paragraph(abstract_text, abstract_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=8))

    # Section 1: Introduction
    story.append(Paragraph("1. Introduction & Research Motivation", h1_style))
    story.append(Paragraph(
        "Deploying deep vision models into safety-critical environments requires rigorous guarantees of out-of-distribution robustness and calibrated uncertainty. "
        "Standard validation sets evaluate models on independent and identically distributed (i.i.d.) test splits, often masking vulnerability to common environmental shifts. "
        "To diagnose these failure modes, synthetic corruption benchmarks such as ImageNet-C (Hendrycks & Dietterich, 2019) apply controlled perturbations to test images. "
        "However, prevailing benchmark practices face two methodological challenges: (1) <i>pseudoreplication</i>, in which repeated corrupted versions of the same "
        "physical image are pooled without cluster adjustments, deflating standard errors, and (2) <i>unary transformation bias</i>, which assesses perturbations in isolation rather than compound sequences.",
        body_style
    ))

    # Section 2: Related Work & Novelty Scope
    story.append(Paragraph("2. Related Work & Actual Scientific Novelty", h1_style))
    story.append(Paragraph(
        "<b>Common Corruptions & Benchmarks:</b> Hendrycks & Dietterich (ICLR 2019) standardized 15 corruptions across 5 severities. Subsequent studies investigated natural distribution shifts "
        "(ImageNet-A, ImageNet-R; Hendrycks et al., 2021) and the interaction between data augmentations and synthetic corruptions (Mintun et al., NeurIPS 2021; Michaelis et al., 2019). "
        "Recently, R-Bench (Li et al., IEEE JSTSP 2025) modeled multi-stage corruption pipelines for large multimodal models, while Goyal et al. (ICCP 2022) explored noise-blur dual corruptions. "
        "CNS-Bench (D&uuml;nkel et al., ICCV 2025) and LAION-C (Li et al., ICML 2025) extended corruption benchmarks to continuous nuisance shifts and web-scale foundation models.<br/>"
        "<b>CNNs vs. Transformers:</b> Bhojanapalli et al. (ICCV 2021) and Bai et al. (NeurIPS 2021) investigated self-attention robustness versus CNNs, noting pretraining scale confounders. "
        "Liu et al. (CVPR 2022) proposed ConvNeXt, modernizing ResNet with 7&times;7 depthwise convolutions and inverted bottlenecks, matching or exceeding Swin Transformers (Liu et al., ICCV 2021).<br/>"
        "<b>Calibration Under Shift:</b> Guo et al. (ICML 2017) highlighted modern deep net overconfidence, while Ovadia et al. (NeurIPS 2019) and Minderer et al. (NeurIPS 2021) established calibration degradation under shift.<br/>"
        "<b>Safe Statement of Exact Novelty:</b> We provide a controlled empirical evaluation of the six application orders of Gaussian noise, defocus blur, and JPEG compression for four pretrained image classifiers on the 10-class Imagenette validation partition. "
        "By pairing every condition across the same source images and accounting for image-level dependence in uncertainty estimates, we characterize how operator order and nominal severity relate to top-1 accuracy, image-fidelity metrics, and calibration. "
        "The results are limited to the tested operators, severity definitions, checkpoints, and dataset; broader generalization and causal interpretation of distortion-adjusted effects remain open.",
        body_style
    ))

    # Section 3: Dataset Scope & Verification
    story.append(Paragraph("3. Dataset Scope & Provenance Verification", h1_style))
    story.append(Paragraph(
        "To ensure full experimental control and computational reproducibility, RobustLens evaluates on the <b>FastAI Imagenette-v2</b> validation partition, "
        "a standardized 10-class subset of ImageNet-1K with canonical WordNet synsets: tench (`n01440764`), English springer spaniel (`n02102040`), "
        "cassette player (`n02979186`), chain saw (`n03000684`), church (`n03028079`), French horn (`n03394916`), garbage truck (`n03417042`), "
        "gas pump (`n03425413`), golf ball (`n03445777`), and parachute (`n03888257`). Sourced from the Hugging Face mirror (`johnowhitaker/imagenette2-320`), "
        "the validation split contains exactly rows 9,469 through 13,393 (N=3,925 images; per-class counts 357 to 419, sum=3,925), matching the official FastAI validation partition.",
        body_style
    ))

    story.append(PageBreak())

    # Section 4: Models & Protocol
    story.append(Paragraph("4. Models & Perturbation Protocol", h1_style))
    story.append(Paragraph(
        "Four canonical open-source checkpoints were evaluated: <b>ResNet-50</b> (`resnet50.a1_in1k`, 25.6M params), <b>ConvNeXt-Tiny</b> (`convnext_tiny.in12k_ft_in1k`, 28.6M params), "
        "<b>ViT-Base/16</b> (`vit_base_patch16_224.augreg_in21k_ft_in1k`, 86.6M params), and <b>Swin-Tiny</b> (`swin_tiny_patch4_window7_224.ms_in1k`, 28.3M params). "
        "Images were normalized to [0, 1] RGB float32 tensors (224&times;224) and perturbed across 3 corruption families at 5 severities: "
        "<b>Gaussian Noise</b> (&sigma; &isin; [0.04, 0.26]), <b>Defocus Blur</b> (kernel k &isin; [3, 11], &sigma; &isin; [1.0, 5.0]), and <b>JPEG Compression</b> (quality Q &isin; [80, 20]), "
        "yielding 16 conditions per image and <b>251,200 forward passes</b> in the primary benchmark.",
        body_style
    ))

    # Section 5: Statistical Methodology
    story.append(Paragraph("5. Statistical Methodology", h1_style))
    story.append(Paragraph(
        "<b>Clustered Paired Bootstrap for Confidence Intervals:</b> Let d_i = x̄_1,i - x̄_2,i denote the paired difference in mean shifted accuracy "
        "on physical image i &isin; {1, ..., N} across all K=15 shifted conditions. We generate B = 10,000 bootstrap resamples of the N=3,925 image clusters to construct "
        "the 95% Clustered Percentile Bootstrap Confidence Interval and estimate bootstrap standard error SE_boot.<br/>"
        "<b>Monte Carlo Paired Sign-Flip Permutation Test:</b> Under the null hypothesis H_0: P(d_i = &delta;) = P(d_i = -&delta;) (model exchangeability per physical image), "
        "swapping model identities corresponds to multiplying d_i by a random Rademacher sign s_i &isin; {-1, +1}. Because exhaustive enumeration across 3,925 images requires evaluating 2<sup>3925</sup> "
        "combinations, we draw R = 20,000 independent Monte Carlo sign vectors. The two-sided permutation p-value with finite-sample pseudocount correction (Phipson & Smyth, 2010) is "
        "p_perm = (1 + &sum; I(|T<sup>(r)</sup>| &ge; |T_obs|)) / (R + 1), bounded at p &lt; 0.0001 when no permutations exceed |T_obs|.<br/>"
        "<b>Clean Baseline McNemar Test & Multiplicity Control:</b> Clean unperturbed images (N=3,925 independent Bernoulli trials) are evaluated using the asymptotic McNemar test "
        "with Edwards' continuity correction (Chi2 = (|n_01 - n_10| - 1)<sup>2</sup> / (n_01 + n_10), df=1). "
        "Multiplicity across all 6 pairwise comparisons is controlled via Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR) at &alpha; = 0.05.",
        body_style
    ))

    # Section 6: Results
    story.append(Paragraph("6. Empirical Findings & Benchmarking Results", h1_style))
    story.append(Paragraph("<b>Table 1: Architecture Leaderboard Summary (N=3,925 Images, 251,200 Predictions)</b>", h2_style))

    table1_data = [
        [Paragraph(h, table_header_style) for h in ["Rank", "Architecture", "Family", "Clean Acc", "Shifted Acc", "Accuracy Drop", "Rel. Robustness", "Mean ECE", "Latency"]],
        [Paragraph(c, table_cell_left if i==1 else table_cell_style) for i, c in enumerate(["#1", "ConvNeXt-Tiny", "Modern CNN", "87.13%", "76.96%", "-10.17%", "0.8833", "0.1538", "421 ms"])],
        [Paragraph(c, table_cell_left if i==1 else table_cell_style) for i, c in enumerate(["#2", "Swin-Tiny", "Hierarchical ViT", "86.04%", "74.97%", "-11.07%", "0.8713", "0.2126", "444 ms"])],
        [Paragraph(c, table_cell_left if i==1 else table_cell_style) for i, c in enumerate(["#3", "ResNet-50", "Classical CNN", "87.36%", "71.93%", "-15.43%", "0.8234", "0.1487", "342 ms"])],
        [Paragraph(c, table_cell_left if i==1 else table_cell_style) for i, c in enumerate(["#4", "ViT-Base/16", "Plain ViT", "80.25%", "68.54%", "-11.72%", "0.8540", "0.1384", "921 ms"])],
    ]
    t1 = Table(table1_data, colWidths=[32, 85, 80, 52, 55, 60, 65, 48, 45])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 3.0),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(t1)
    story.append(Spacer(1, 4))

    # Figure 1
    f1_path = os.path.join(REPO_ROOT, "reports", "figures", "01_clean_vs_shifted_accuracy.png")
    if os.path.exists(f1_path):
        story.append(Image(f1_path, width=6.2*inch, height=2.3*inch))
        story.append(Paragraph("Figure 1: Clean baseline top-1 accuracy vs. mean shifted top-1 accuracy across 15 corruption conditions (N=3,925).", caption_style))

    # Table 2: Pairwise Significance
    story.append(Paragraph("<b>Table 2: Pairwise Statistical Hypothesis Testing (N=3,925 Clusters, R=20,000 Sign-Flip Permutations)</b>", h2_style))
    table2_data = [
        [Paragraph(h, table_header_style) for h in ["Model Pair", "Shifted Acc", "Diff (Delta)", "95% Clustered CI", "Sign-Flip p", "Perm z (p_asymp)", "Clean McNemar p", "Holm / BH"]],
        [Paragraph(c, table_cell_left if i==0 else table_cell_style) for i, c in enumerate(["ConvNeXt vs. ResNet", "76.96% vs 71.93%", "+5.03%", "[+4.47%, +5.57%]", "< 0.0001", "17.34 (2.5e-67)", "0.6402", "Sig / Sig"])],
        [Paragraph(c, table_cell_left if i==0 else table_cell_style) for i, c in enumerate(["ConvNeXt vs. Swin", "76.96% vs 74.97%", "+1.99%", "[+1.42%, +2.54%]", "< 0.0001", "6.94 (3.8e-12)", "0.0145", "Sig / Sig"])],
        [Paragraph(c, table_cell_left if i==0 else table_cell_style) for i, c in enumerate(["ConvNeXt vs. ViT", "76.96% vs 68.54%", "+8.42%", "[+7.67%, +9.19%]", "< 0.0001", "20.83 (2.2e-96)", "< 0.0001", "Sig / Sig"])],
        [Paragraph(c, table_cell_left if i==0 else table_cell_style) for i, c in enumerate(["Swin vs. ResNet", "74.97% vs 71.93%", "+3.04%", "[+2.47%, +3.60%]", "< 0.0001", "10.38 (3.0e-25)", "0.0038", "Sig / Sig"])],
        [Paragraph(c, table_cell_left if i==0 else table_cell_style) for i, c in enumerate(["Swin vs. ViT", "74.97% vs 68.54%", "+6.43%", "[+5.67%, +7.22%]", "< 0.0001", "15.80 (2.9e-56)", "< 0.0001", "Sig / Sig"])],
        [Paragraph(c, table_cell_left if i==0 else table_cell_style) for i, c in enumerate(["ResNet vs. ViT", "71.93% vs 68.54%", "+3.40%", "[+2.64%, +4.17%]", "< 0.0001", "8.73 (2.6e-18)", "< 0.0001", "Sig / Sig"])],
    ]
    t2 = Table(table2_data, colWidths=[90, 75, 45, 80, 52, 75, 58, 47])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 2.8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(t2)
    story.append(Spacer(1, 4))

    # Figure 5: Pairwise Clustered CIs
    f5_path = os.path.join(REPO_ROOT, "reports", "figures", "05_pairwise_clustered_confidence_intervals.png")
    if os.path.exists(f5_path):
        story.append(Image(f5_path, width=6.2*inch, height=2.2*inch))
        story.append(Paragraph("Figure 5: Pairwise mean shifted top-1 accuracy difference Delta (%) with 95% Clustered Paired Bootstrap Confidence Intervals (N=3,925 Image Clusters).", caption_style))

    story.append(PageBreak())

    # Section 7: Degradation Profiles & Calibration
    story.append(Paragraph("7. Perturbation Degradation Profiles & Calibration Analysis", h1_style))
    story.append(Paragraph(
        "<b>Severity Sensitivity:</b> Gaussian noise induces the steepest accuracy drop: ResNet-50 collapses by -48.69 percentage points (87.36% clean to 38.67% at severity 5), "
        "whereas ConvNeXt-Tiny retains 58.42% (+19.75 point margin over ResNet-50). Defocus blur degrades smoothly across severities, and JPEG compression exhibits the least degradation.",
        body_style
    ))

    # Figure 2
    f2_path = os.path.join(REPO_ROOT, "reports", "figures", "02_corruption_severity_degradation.png")
    if os.path.exists(f2_path):
        story.append(Image(f2_path, width=6.2*inch, height=1.9*inch))
        story.append(Paragraph("Figure 2: Top-1 accuracy degradation across severity levels (1 to 5) for Gaussian Noise, Defocus Blur, and JPEG Compression.", caption_style))

    # Table 3: Top-1 Accuracy across Corruption Families
    story.append(Paragraph("<b>Table 3: Top-1 Accuracy (%) Across Corruption Families and Severity Levels (N=3,925 Images per Cell)</b>", h2_style))
    table3_data = [
        [Paragraph(h, table_header_style) for h in ["Architecture", "Perturbation Family", "Clean (s0)", "Sev 1", "Sev 2", "Sev 3", "Sev 4", "Sev 5", "Mean Shifted"]],
        [Paragraph(c, table_cell_left if i<2 else table_cell_style) for i, c in enumerate(["ConvNeXt-Tiny", "Gaussian Noise", "87.13%", "85.20%", "80.97%", "75.85%", "68.99%", "58.42%", "73.89%"])],
        [Paragraph(c, table_cell_left if i<2 else table_cell_style) for i, c in enumerate(["", "Defocus Blur", "87.13%", "84.18%", "80.74%", "77.02%", "73.20%", "69.17%", "76.86%"])],
        [Paragraph(c, table_cell_left if i<2 else table_cell_style) for i, c in enumerate(["", "JPEG Compression", "87.13%", "84.82%", "82.55%", "80.99%", "78.47%", "73.86%", "80.14%"])],
        [Paragraph(c, table_cell_left if i<2 else table_cell_style) for i, c in enumerate(["Swin-Tiny", "Gaussian Noise", "86.04%", "83.18%", "77.86%", "71.97%", "63.85%", "51.72%", "69.72%"])],
        [Paragraph(c, table_cell_left if i<2 else table_cell_style) for i, c in enumerate(["", "Defocus Blur", "86.04%", "83.08%", "79.52%", "75.80%", "71.69%", "67.29%", "75.48%"])],
        [Paragraph(c, table_cell_left if i<2 else table_cell_style) for i, c in enumerate(["", "JPEG Compression", "86.04%", "83.77%", "81.63%", "80.10%", "77.89%", "75.21%", "79.72%"])],
        [Paragraph(c, table_cell_left if i<2 else table_cell_style) for i, c in enumerate(["ResNet-50", "Gaussian Noise", "87.36%", "82.42%", "74.34%", "65.58%", "54.34%", "38.67%", "63.07%"])],
        [Paragraph(c, table_cell_left if i<2 else table_cell_style) for i, c in enumerate(["", "Defocus Blur", "87.36%", "82.93%", "78.60%", "74.22%", "69.83%", "64.92%", "74.10%"])],
        [Paragraph(c, table_cell_left if i<2 else table_cell_style) for i, c in enumerate(["", "JPEG Compression", "87.36%", "83.57%", "81.04%", "79.44%", "76.69%", "72.41%", "78.63%"])],
        [Paragraph(c, table_cell_left if i<2 else table_cell_style) for i, c in enumerate(["ViT-Base/16", "Gaussian Noise", "80.25%", "75.52%", "70.32%", "64.69%", "57.86%", "46.88%", "63.05%"])],
        [Paragraph(c, table_cell_left if i<2 else table_cell_style) for i, c in enumerate(["", "Defocus Blur", "80.25%", "77.53%", "74.45%", "71.01%", "67.24%", "62.78%", "70.60%"])],
        [Paragraph(c, table_cell_left if i<2 else table_cell_style) for i, c in enumerate(["", "JPEG Compression", "80.25%", "76.54%", "74.29%", "72.82%", "70.47%", "65.71%", "71.97%"])],
    ]
    t3 = Table(table3_data, colWidths=[72, 86, 44, 42, 42, 42, 42, 42, 52])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 1.8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(t3)
    story.append(Spacer(1, 4))

    # Figure 3
    f3_path = os.path.join(REPO_ROOT, "reports", "figures", "03_per_corruption_breakdown.png")
    if os.path.exists(f3_path):
        story.append(Image(f3_path, width=6.2*inch, height=1.9*inch))
        story.append(Paragraph("Figure 3: Detailed per-corruption accuracy breakdown across perturbation families.", caption_style))

    story.append(PageBreak())

    # Calibration Analysis
    story.append(Paragraph(
        "<b>Calibration Analysis (10 Confidence Bins):</b> Evaluated with M=10 equal-width confidence bins. On clean validation images, ViT-Base (4.69%) and ResNet-50 (4.89%) "
        "exhibit the lowest calibration error. Under distribution shift, ViT-Base preserves superior probability calibration (mean shifted ECE = 14.45%). "
        "Conversely, Swin-Tiny exhibits severe overconfidence (mean shifted ECE = 21.98%), assigning >85% softmax confidence to incorrect predictions. "
        "Under Split Conformal Prediction (nominal 95% coverage on clean data), distribution shift induces a 13.8%–17.8% empirical coverage deficit due to exchangeability violation.",
        body_style
    ))

    # Table 4: ECE Degradation (Directly matching detailed_breakdown.csv)
    story.append(Paragraph("<b>Table 4: Expected Calibration Error (ECE) Degradation Under Distribution Shift (M=10 Bins)</b>", h2_style))
    table4_data = [
        [Paragraph(h, table_header_style) for h in ["Architecture", "Clean ECE (s0)", "Shifted ECE (Mean)", "ECE Increase (Delta)", "Calibration Profile Under Shift"]],
        [Paragraph(c, table_cell_left if i==0 or i==4 else table_cell_style) for i, c in enumerate(["ViT-Base/16", "4.69%", "14.45%", "+9.76% (+0.0976)", "Most calibrated under distribution shift; low overconfidence"])],
        [Paragraph(c, table_cell_left if i==0 or i==4 else table_cell_style) for i, c in enumerate(["ResNet-50", "4.89%", "15.53%", "+10.64% (+0.1064)", "Well-calibrated on clean; degrades moderately under shift"])],
        [Paragraph(c, table_cell_left if i==0 or i==4 else table_cell_style) for i, c in enumerate(["ConvNeXt-Tiny", "10.02%", "15.74%", "+5.72% (+0.0572)", "Stable across mild/moderate shifts; moderate noise overconfidence"])],
        [Paragraph(c, table_cell_left if i==0 or i==4 else table_cell_style) for i, c in enumerate(["Swin-Tiny", "10.47%", "21.98%", "+11.51% (+0.1151)", "Severe overconfidence under shift; assigns >85% confidence to errors"])],
    ]
    t4 = Table(table4_data, colWidths=[78, 65, 75, 75, 211])
    t4.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 2.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(t4)
    story.append(Spacer(1, 4))

    # Figure 4
    f4_path = os.path.join(REPO_ROOT, "reports", "figures", "04_calibration_ece_shift.png")
    if os.path.exists(f4_path):
        story.append(Image(f4_path, width=6.2*inch, height=2.0*inch))
        story.append(Paragraph("Figure 4: Expected Calibration Error (ECE, %) on clean baseline vs. mean shifted conditions (zero baseline).", caption_style))

    story.append(PageBreak())

    # Section 8: Order Effects Research Extension
    story.append(Paragraph("8. Research Extension: Compound Corruption Permutation Order Effects", h1_style))
    story.append(Paragraph(
        "<b>Theoretical Motivation (Operator Non-Commutativity):</b> In physical camera systems, degradation occurs sequentially: "
        "optical defocus (spatial low-pass filter B), sensor photon shot noise (additive Gaussian noise N), and lossy compression (non-linear DCT quantization J). "
        "Because these operations do not commute, JPEG(Blur(x) + Noise) &ne; Blur(JPEG(x)) + Noise. "
        "Across all N=3,925 validation images with seed-matched noise realizations, permutation sequences induce an empirical PSNR spread of 6.29 dB and SSIM spread of 0.421.",
        body_style
    ))

    # Table 5: Order Effects Leaderboard & Factorial Clustered ANCOVA
    story.append(Paragraph("<b>Table 5: Compound Corruption Order Effects Leaderboard & Clustered Logistic Models (N=3,925 Clusters, 486,700 Predictions)</b>", h2_style))
    table5_data = [
        [Paragraph(h, table_header_style) for h in ["Architecture", "Clean Acc", "Compound Acc", "Order Var", "Max Gap", "Model 1: Order Wald (5 df)", "Model 2: Order Main (5 df)", "Model 2: Inter (20 df)", "Joint Order (25 df)"]],
        [Paragraph(c, table_cell_left if i==0 else table_cell_style) for i, c in enumerate(["ConvNeXt-Tiny", "87.13%", "56.14%", "0.0102", "20.45%", "922.42 (p < 1e-100)", "134.60 (p < 1e-26)", "395.55 (p < 1e-70)", "935.07 (p < 1e-100)"])],
        [Paragraph(c, table_cell_left if i==0 else table_cell_style) for i, c in enumerate(["Swin-Tiny", "86.04%", "55.20%", "0.0052", "17.13%", "1810.83 (p < 1e-100)", "217.01 (p < 1e-44)", "323.63 (p < 1e-55)", "1358.62 (p < 1e-100)"])],
        [Paragraph(c, table_cell_left if i==0 else table_cell_style) for i, c in enumerate(["ViT-Base/16", "80.25%", "50.23%", "0.0098", "20.37%", "1276.88 (p < 1e-100)", "90.05 (p < 1e-17)", "856.81 (p < 1e-100)", "1968.16 (p < 1e-100)"])],
        [Paragraph(c, table_cell_left if i==0 else table_cell_style) for i, c in enumerate(["ResNet-50", "87.36%", "46.06%", "0.0117", "21.55%", "1366.36 (p < 1e-100)", "232.57 (p < 1e-47)", "954.37 (p < 1e-100)", "1835.66 (p < 1e-100)"])],
    ]
    t5 = Table(table5_data, colWidths=[70, 44, 52, 44, 44, 76, 74, 75, 75])
    t5.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 2.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(t5)
    story.append(Spacer(1, 4))

    # Figure 6: Order effects by architecture
    f6_path = os.path.join(REPO_ROOT, "reports", "figures", "06_order_effects_by_architecture.png")
    if os.path.exists(f6_path):
        story.append(Image(f6_path, width=6.2*inch, height=2.3*inch))
        story.append(Paragraph("Figure 6: Classification accuracy across all 6 compound corruption permutation sequences by architecture.", caption_style))

    # Figure 7: PSNR/SSIM vs Accuracy
    f7_path = os.path.join(REPO_ROOT, "reports", "figures", "07_psnr_ssim_vs_accuracy_by_order.png")
    if os.path.exists(f7_path):
        story.append(Image(f7_path, width=6.2*inch, height=2.2*inch))
        story.append(Paragraph("Figure 7: Objective image fidelity (PSNR and SSIM) versus Top-1 accuracy across permutation sequences.", caption_style))

    story.append(PageBreak())

    # Figure 8: Order Severity Interaction
    f8_path = os.path.join(REPO_ROOT, "reports", "figures", "08_order_severity_interaction.png")
    if os.path.exists(f8_path):
        story.append(Image(f8_path, width=6.2*inch, height=2.3*inch))
        story.append(Paragraph("Figure 8: Compound corruption sequence divergence across severity levels (1 to 5) for all four architectures.", caption_style))

    # Figure 9: Order Calibration ECE
    f9_path = os.path.join(REPO_ROOT, "reports", "figures", "09_order_calibration_ece.png")
    if os.path.exists(f9_path):
        story.append(Image(f9_path, width=6.2*inch, height=2.2*inch))
        story.append(Paragraph("Figure 9: Expected Calibration Error (ECE, %) across permutation orders (legend above data, zero baseline).", caption_style))

    story.append(Paragraph(
        "<b>Audited Empirical Insights from Order Effects Study:</b><br/>"
        "1. <b>Statistical Association Beyond PSNR and SSIM:</b> Multi-variable logistic models clustered at the physical image level (N_clusters=3,925) confirm that "
        "permutation order remains strongly associated with classification correctness (Wald p &lt; 1e-100) after conditioning on continuous PSNR and SSIM. "
        "PSNR and SSIM are post-transformation summaries rather than pre-treatment confounders; conditioning on them evaluates whether order predicts correctness "
        "beyond low-level fidelity, without establishing an independent causal mechanism.<br/>"
        "2. <b>Vulnerability Dynamics & Operator Order:</b> Sequences ending in JPEG (<code>NBJ</code>, <code>BNJ</code>) trigger severe high-frequency block boundaries, "
        "causing the lowest overall accuracy (e.g., <code>BNJ</code> mean accuracy is 42.88%). Conversely, sequences where Blur succeeds JPEG (<code>JNB</code> at 56.67% and "
        "<code>NJB</code> at 61.31%) smooth 8&times;8 DCT grid discontinuities, partially preserving spatial structure. Sequences ending in noise (<code>JBN</code> at 43.75% "
        "and <code>BJN</code> at 48.77%) suffer heavy high-frequency noise corruption.<br/>"
        "3. <b>Calibration by Permutation Order:</b> Empirical per-order ECE shows that sequences ending in Gaussian noise (<code>JBN</code>, <code>BJN</code>) produce the highest "
        "calibration error (Swin ECE reaches 18.58% on JBN and 18.42% on BJN; ResNet reaches 11.67%). Conversely, sequences ending in JPEG (<code>NBJ</code>, <code>BNJ</code>) "
        "or Blur (<code>JNB</code>, <code>NJB</code>) exhibit lower ECE (Swin ECE drops to 5.71% on NBJ). The initial hypothesis that terminal JPEG causes the highest ECE was not supported.<br/>"
        "4. <b>Architectural Invariance Hierarchy:</b> Swin-Tiny exhibits the lowest sequence variance (variance 0.0052, gap 17.13%), whereas classical ResNet-50 displays "
        "the greatest vulnerability (variance 0.0117, gap 21.55%). ConvNeXt-Tiny achieves the highest overall compound accuracy (56.14%), while ViT-Base preserves the lowest overall compound ECE (0.0413).",
        body_style
    ))

    story.append(PageBreak())

    # Section 8.4: Order-Dependent Error Churn & Decoupling Dynamics
    story.append(Paragraph("8.4 Order-Dependent Error Churn & Prediction Flip Dynamics", h1_style))
    story.append(Paragraph(
        "<b>Decoupling Instance-Level Prediction Instability from Aggregate Accuracy:</b> "
        "A critical limitation of aggregate accuracy is that invariant group performance may conceal massive instance-level prediction churn (Milani Fard et al., NeurIPS 2016; D'Amour et al., 2020). "
        "Let C<sub>A</sub>, C<sub>B</sub> &isin; {0, 1} indicate top-1 correctness for physical image i under permutation sequences A and B. "
        "We define Gain rate G = P(C<sub>A</sub>=0, C<sub>B</sub>=1) (Order B rescues an image failed by A), "
        "Loss rate L = P(C<sub>A</sub>=1, C<sub>B</sub>=0) (Order B breaks an image rescued by A), "
        "Net difference &Delta;<sub>net</sub> = G - L = Acc<sub>B</sub> - Acc<sub>A</sub>, "
        "Error Churn = G + L = P(C<sub>A</sub> &ne; C<sub>B</sub>), and "
        "Excess Churn = Churn - |&Delta;<sub>net</sub>| = 2&middot;min(G, L) &ge; 0. "
        "Across all 300 matched pairwise evaluations, the identity (Acc<sub>B</sub> - Acc<sub>A</sub>) - (G - L) = 0.0000 was verified with zero numerical discrepancies.",
        body_style
    ))

    # Table 6: Error Churn & Decoupled Pairs
    story.append(Paragraph("<b>Table 6: Order-Dependent Error Churn & Decoupled Sequence Pairs (|&Delta;<sub>net</sub>| &le; 2.0%, Excess Churn &ge; 5.0%)</b>", h2_style))
    table6_data = [
        [Paragraph(h, table_header_style) for h in ["Architecture", "Sev", "Pair", "Acc A &rarr; B", "Gain G", "Loss L", "Net Diff (95% CI)", "Churn (95% CI)", "Excess", "McNemar p"]],
        [Paragraph(c, table_cell_left if i==0 else table_cell_style) for i, c in enumerate(["ViT-Base/16", "1", "NBJ vs BJN", "73.1% &rarr; 72.9%", "+5.27% (207)", "-5.45% (214)", "-0.18% [-1.20, +0.84]", "10.73% [9.76, 11.69]", "10.55%", "0.770"])],
        [Paragraph(c, table_cell_left if i==0 else table_cell_style) for i, c in enumerate(["ConvNeXt-Tiny", "4", "NBJ vs JNB", "56.1% &rarr; 56.3%", "+5.45% (214)", "-5.25% (206)", "+0.20% [-0.79, +1.22]", "10.70% [9.73, 11.67]", "10.50%", "0.733"])],
        [Paragraph(c, table_cell_left if i==0 else table_cell_style) for i, c in enumerate(["ViT-Base/16", "4", "NBJ vs JNB", "51.9% &rarr; 51.8%", "+5.22% (205)", "-5.32% (209)", "-0.10% [-1.15, +0.92]", "10.55% [9.61, 11.49]", "10.45%", "0.883"])],
        [Paragraph(c, table_cell_left if i==0 else table_cell_style) for i, c in enumerate(["ViT-Base/16", "1", "NBJ vs JBN", "73.1% &rarr; 73.2%", "+5.27% (207)", "-5.20% (204)", "+0.08% [-0.92, +1.12]", "10.47% [9.53, 11.44]", "10.39%", "0.921"])],
        [Paragraph(c, table_cell_left if i==0 else table_cell_style) for i, c in enumerate(["ViT-Base/16", "1", "NJB vs JBN", "73.2% &rarr; 73.2%", "+4.99% (196)", "-4.99% (196)", "0.00% [-0.99, +0.99]", "9.99% [9.04, 10.93]", "9.99%", "1.000"])],
        [Paragraph(c, table_cell_left if i==0 else table_cell_style) for i, c in enumerate(["ConvNeXt-Tiny", "2", "BNJ vs JBN", "66.8% &rarr; 65.3%", "+4.48% (176)", "-6.04% (237)", "-1.55% [-2.57, -0.54]", "10.52% [9.58, 11.49]", "8.97%", "0.003"])],
        [Paragraph(c, table_cell_left if i==0 else table_cell_style) for i, c in enumerate(["ResNet-50", "1", "NBJ vs JBN", "78.5% &rarr; 78.5%", "+4.48% (176)", "-4.43% (174)", "+0.05% [-0.89, +0.97]", "8.92% [8.03, 9.83]", "8.87%", "0.957"])],
        [Paragraph(c, table_cell_left if i==0 else table_cell_style) for i, c in enumerate(["ConvNeXt-Tiny", "2", "BNJ vs BJN", "66.8% &rarr; 65.0%", "+4.33% (170)", "-6.11% (240)", "-1.78% [-2.80, -0.82]", "10.45% [9.50, 11.44]", "8.66%", "<0.001"])],
        [Paragraph(c, table_cell_left if i==0 else table_cell_style) for i, c in enumerate(["ResNet-50", "1", "NBJ vs BJN", "78.5% &rarr; 78.6%", "+4.43% (174)", "-4.28% (168)", "+0.15% [-0.76, +1.07]", "8.71% [7.85, 9.61]", "8.56%", "0.787"])],
        [Paragraph(c, table_cell_left if i==0 else table_cell_style) for i, c in enumerate(["ResNet-50", "3", "BNJ vs BJN", "51.5% &rarr; 47.4%", "+9.18% (360)", "-13.34% (524)", "-4.15% [-5.43, -2.85]", "22.52% [21.20, 23.82]", "14.22%", "<0.001"])],
    ]
    t6 = Table(table6_data, colWidths=[66, 20, 50, 60, 48, 48, 76, 76, 32, 36])
    t6.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 2.0),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(t6)
    story.append(Spacer(1, 4))

    # Figure 10: Error Churn Analysis
    f10_path = os.path.join(REPO_ROOT, "reports", "figures", "10_error_churn_analysis.png")
    if os.path.exists(f10_path):
        story.append(Image(f10_path, width=6.2*inch, height=3.5*inch))
        story.append(Paragraph("Figure 10: Order-Dependent Error Churn Analysis: (A) Pairwise churn rate matrices by architecture. (B) Decoupling of Error Churn from Net Difference (39 high-excess pairs). (C) Severity dynamics showing excess churn peak at intermediate levels.", caption_style))

    story.append(Paragraph(
        "<b>Audited Insights on Prediction Churn:</b> "
        "1. In 39 of 300 sequence pairs, classifiers exhibit near-zero net accuracy shift (|&Delta;<sub>net</sub>| &le; 2.0%) accompanied by large excess churn (&ge; 5.0%). "
        "In the extreme case of ViT-Base at Severity 1 (NJB vs JBN), aggregate accuracy is identically 73.17% (&Delta; = 0.00%, p = 1.0), yet 392 physical images flip correctness (Churn = 9.99%). "
        "2. ViT-Base exhibits the highest mean excess churn (6.46%), reflecting global attention weight re-allocation under spatial phase shifts. "
        "3. ResNet-50 exhibits peak excess churn of 14.22% (Sev 3, BNJ vs BJN, 360 rescues vs 524 breaks). "
        "4. Confidence increases by ~+26% on Gain transitions and drops by ~-22% on Loss transitions.",
        body_style
    ))

    story.append(PageBreak())

    # Limitations & Honest Scope Assessment
    story.append(Paragraph("9. Limitations & Honest Scope Assessment", h1_style))
    story.append(Paragraph(
        "To maintain scientific integrity for graduate admissions review, we explicitly articulate our scope limitations:<br/>"
        "1. <b>10-Class Benchmark (Imagenette-v2):</b> Evaluations were conducted on 10 WordNet synsets (N=3,925 validation images) rather than the complete "
        "1,000-class ImageNet-1K benchmark (N=50,000). Findings should not be extrapolated to fine-grained 1,000-class discrimination.<br/>"
        "2. <b>Perturbation Diversity:</b> We evaluated 3 canonical perturbation families (15 shifted conditions in primary benchmark, 30 compound conditions in order study). "
        "We did not evaluate the full 19 corruption types of ImageNet-C or natural adversarial shifts (ImageNet-A, ImageNet-R).<br/>"
        "3. <b>Pretraining Disparities:</b> Pretrained weights originate from standard open-source checkpoints: ConvNeXt-Tiny was pretrained on ImageNet-12K and fine-tuned on 1K; "
        "ViT-Base was pretrained on ImageNet-21K with AugReg and fine-tuned on 1K; ResNet-50 and Swin-Tiny were pretrained directly on ImageNet-1K. Pretraining scale is an unavoidable confounder.<br/>"
        "4. <b>Task Specificity:</b> Limited strictly to 2D image classification; findings do not generalize directly to dense prediction (detection/segmentation).",
        body_style
    ))

    # Section 10: Conclusion
    story.append(Paragraph("10. Conclusion", h1_style))
    story.append(Paragraph(
        "RobustLens delivers an audited, reproducible benchmark investigating the robustness, uncertainty, and operator permutation sensitivity of modern vision architectures. "
        "By replacing naive un-clustered pooling with image-level clustered bootstrap confidence intervals and Monte Carlo paired sign-flip permutation tests, we resolve pseudoreplication. "
        "Our empirical results show that clean accuracy fails to predict out-of-distribution performance, that compound corruption order significantly alters accuracy and calibration, "
        "and that architectural inductive biases strongly mediate vulnerability to cascading degradation.",
        body_style
    ))

    # References
    story.append(Paragraph("11. References", h1_style))
    refs = [
        "1. Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction. <i>arXiv:2107.07511</i>.",
        "2. Bai, Y., et al. (2021). Are transformers more robust than CNNs? <i>NeurIPS</i>, 34, 26831–26843.",
        "3. Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. <i>JRSS-B</i>, 57(1), 289–300.",
        "4. Bhojanapalli, S., et al. (2021). Understanding robustness of transformers for image classification. <i>ICCV</i>, 10231–10241.",
        "5. D'Amour, A., et al. (2020). Underspecification presents challenges for credibility in modern machine learning. <i>JMLR</i>, 23(189), 1–61.",
        "6. Davison, A. C., & Hinkley, D. V. (1997). <i>Bootstrap Methods and their Application</i>. Cambridge University Press.",
        "7. Deng, J., et al. (2009). ImageNet: A large-scale hierarchical image database. <i>CVPR</i>, 248–255.",
        "8. Dosovitskiy, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. <i>ICLR</i>.",
        "9. D&uuml;nkel, P., et al. (2025). CNS-Bench: Benchmarking image classifier robustness under continuous nuisance shifts. <i>ICCV</i>.",
        "10. Goyal, S., et al. (2022). Robust scene inference under noise-blur dual corruptions. <i>ICCP</i>, 1–11.",
        "11. Guo, C., et al. (2017). On calibration of modern neural networks. <i>ICML</i>, PMLR, 1321–1330.",
        "12. Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions. <i>ICLR</i>.",
        "13. Hendrycks, D., et al. (2021). Natural adversarial examples. <i>CVPR</i>, 15262–15271.",
        "14. Holm, S. (1979). A simple sequentially rejective multiple test procedure. <i>Scand. J. Statist.</i>, 6(2), 65–70.",
        "15. Howard, J. (2019). Imagenette: A smaller subset of 10 easily classified classes from ImageNet. <i>fastai/imagenette</i>.",
        "16. Hurlbert, S. H. (1984). Pseudoreplication and the design of ecological field experiments. <i>Ecol. Monogr.</i>, 54(2), 187–211.",
        "17. Lehmann, E. L., & Romano, J. P. (2005). <i>Testing Statistical Hypotheses</i> (3rd ed.). Springer.",
        "18. Li, B., et al. (2025). LAION-C: An out-of-distribution benchmark for web-scale vision models. <i>ICML</i>.",
        "19. Li, Y., et al. (2025). R-Bench: Are your large multimodal models robust to real-world corruptions? <i>IEEE JSTSP</i>.",
        "20. Liu, Z., et al. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. <i>ICCV</i>, 10012–10022.",
        "21. Liu, Z., et al. (2022). A ConvNet for the 2020s. <i>CVPR</i>, 11976–11986.",
        "22. Michaelis, C., et al. (2019). Benchmarking robustness in object detection: Autonomous driving when winter is coming. <i>arXiv:1907.07484</i>.",
        "23. Milani Fard, M., et al. (2016). Launch and iterate: Reducing prediction churn. <i>NeurIPS</i>, 29, 3179–3187.",
        "24. Minderer, M., et al. (2021). Revisiting the calibration of modern neural networks. <i>NeurIPS</i>, 34, 15682–15694.",
        "25. Mintun, E., et al. (2021). On interaction between augmentations and corruptions in ImageNet-C. <i>NeurIPS</i>, 34, 3571–3583.",
        "26. Ovadia, Y., et al. (2019). Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. <i>NeurIPS</i>, 32.",
        "27. Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never be zero. <i>Stat. Appl. Genet. Mol. Biol.</i>, 9(1).",
        "28. Recht, B., et al. (2019). Do ImageNet classifiers generalize to ImageNet? <i>ICML</i>, PMLR, 5389–5400.",
        "29. Steiner, A., et al. (2022). How to train your ViT? Data, augmentation, and regularization in vision transformers. <i>TMLR</i>.",
        "30. Vovk, V., Gammerman, A., & Shafer, G. (2005). <i>Algorithmic Learning in a Random World</i>. Springer.",
    ]
    for r in refs:
        story.append(Paragraph(r, ref_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Publication-grade PDF successfully built at: {output_pdf_path}")

if __name__ == "__main__":
    build_manuscript_pdf()
