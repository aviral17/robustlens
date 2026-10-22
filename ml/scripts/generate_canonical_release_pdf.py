#!/usr/bin/env python3
"""Generate the canonical publication-grade PDF for RobustLens.

Target: reports/final_research_package/RobustLens_Compound_Corruption_Manuscript.pdf
Title: "Order Matters Beyond Accuracy: Paired Prediction Analysis of Compound Image Corruptions"
Author: Aviral Choudhary
Affiliation: RobustLens Research Project
Status: Unpublished Working Paper / PhD Research Writing Sample
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPO_ROOT / "reports/final_research_package/analysis"
FIG_DIR = REPO_ROOT / "reports/final_research_package/figures"
OUT_PDF = REPO_ROOT / "reports/final_research_package/RobustLens_Compound_Corruption_Manuscript.pdf"


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
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Running header on pages 2+
        if self._pageNumber > 1:
            self.drawString(54, 752, "Order Matters Beyond Accuracy: Paired Prediction Analysis of Compound Image Corruptions")
            self.drawRightString(612 - 54, 752, "RobustLens Research Working Paper")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 746, 612 - 54, 746)

        # Running footer on all pages
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 32, page_str)
        self.drawString(54, 32, "RobustLens Research Working Paper • Aviral Choudhary • October 2026")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 42, 612 - 54, 42)
        self.restoreState()


def build_canonical_manuscript_pdf():
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=46,
        bottomMargin=48,
        title="Order Matters Beyond Accuracy: Paired Prediction Analysis of Compound Image Corruptions",
        author="Aviral Choudhary",
        subject="Computer Vision Robustness, Multi-Stage Corruptions, Statistical Decomposition, PhD Writing Sample"
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_CENTER,
        spaceAfter=3
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#334155"),
        alignment=TA_CENTER,
        spaceAfter=6
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12.5,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.2,
        leading=10.5,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor("#1e293b"),
        alignment=TA_JUSTIFY,
        spaceAfter=3.5
    )

    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.5,
        leading=8.2,
        textColor=colors.HexColor("#1e293b"),
        alignment=TA_CENTER
    )

    table_cell_left = ParagraphStyle(
        "TableCellLeft",
        parent=table_cell,
        alignment=TA_LEFT
    )

    table_header = ParagraphStyle(
        "TableHeader",
        parent=table_cell,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0f172a")
    )

    panel_title_style = ParagraphStyle(
        "PanelTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.2,
        leading=10.5,
        textColor=colors.HexColor("#38bdf8"),
        spaceAfter=3
    )

    panel_body_style = ParagraphStyle(
        "PanelBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.8,
        leading=8.8,
        textColor=colors.HexColor("#f8fafc"),
        alignment=TA_JUSTIFY,
        spaceAfter=2.5
    )

    story = []

    # ================= PAGE 1 =================
    # Title & Metadata
    story.append(Paragraph("Order Matters Beyond Accuracy: Paired Prediction Analysis of Compound Image Corruptions", title_style))
    story.append(Paragraph(
        "<b>Aviral Choudhary</b> • RobustLens Research Project • October 2026<br/>"
        "<b>Document Status:</b> Unpublished Working Paper / PhD Research Writing Sample<br/>"
        "<b>Repository & Data Artifacts:</b> 973,400 forward passes on 3,925 Imagenette validation images across 2 regimes",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#cbd5e1"), spaceBefore=1, spaceAfter=4))

    # WHAT THIS STUDY FINDS (Dark Charcoal Panel on Page 1)
    panel_content = [
        Paragraph("WHAT THIS STUDY FINDS", panel_title_style),
        Paragraph(
            "<b>1. Nominal-Order Benchmarking Conflates Sequence with Image Quality:</b> Under standard ImageNet-C parameters, changing execution order across Gaussian Noise, Defocus Blur, and JPEG Compression induces top-1 accuracy disparities of up to 21.55% (ResNet-50) and 20.45% (ConvNeXt-Tiny) on identical physical images. These nominal differences are heavily driven by terminal operator smoothing: sequences ending in blur boost PSNR by ~6.2 dB and preserve accuracy, whereas sequences ending in noise or JPEG induce severe degradation.",
            panel_body_style
        ),
        Paragraph(
            "<b>2. Fidelity Calibration Contracts Aggregate Gaps but Leaves Residual Differences:</b> Pre-calibrating parameters to target common (PSNR, SSIM) support contracts the cross-sequence range of the four-model mean by <b>63.18%</b> (from 18.43% down to 6.79%; architecture-specific gap contractions average <b>58.08%</b>, contracting by 77.15% in ConvNeXt, 72.62% in ResNet, 50.75% in ViT, and 31.80% in Swin). Prespecified tolerances (±0.75 dB PSNR, ±0.035 SSIM) were not achieved over the tested grid (PSNR RMSE 0.943 dB, SSIM RMSE 0.1380), establishing this strictly as an approximate-fidelity descriptive comparison rather than an exact causal effect at matched fidelity.",
            panel_body_style
        ),
        Paragraph(
            "<b>3. Order Matters Beyond Accuracy (Severe Error Migration):</b> In a complete 5-state joint prediction partition (<i>E</i><sub>11</sub>, <i>E</i><sub>10</sub>, <i>E</i><sub>01</sub>, <i>E</i><sub>00</sub><sup>same</sup>, <i>E</i><sub>00</sub><sup>diff</sup>), <b>Predicted-Label Churn</b> (34.74%–40.78%) is more than double <b>Correctness Churn</b> (12.34%–16.93%) because Error Migration (<i>E</i><sub>00</sub><sup>diff</sup>) accounts for 22.40%–23.85% of all evaluations. Sequence pairs with near-identical aggregate accuracy (|ΔAcc| < 2.5%) exhibit up to 10.97% excess churn, proving that sequence order actively reshuffles decision boundaries across identical visual instances.",
            panel_body_style
        ),
    ]
    t_panel = Table([[panel_content]], colWidths=[504])
    t_panel.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1e293b")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#0f172a")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_panel)
    story.append(Spacer(1, 4))

    # Abstract Box
    abstract_text = (
        "<b>Abstract—</b> Deep neural networks for computer vision are widely evaluated under synthetic distribution shifts, but standard benchmarks "
        "treat corruptions primarily as isolated unary operations (e.g., ImageNet-C) or along fixed unidirectional pipelines (e.g., R-Bench). In real-world imaging pipelines, "
        "degradations occur sequentially through multi-stage physical and digital processes governed by non-commutative operators "
        "(<i>T<sub>A</sub> ∘ T<sub>B</sub> ≠ T<sub>B</sub> ∘ T<sub>A</sub></i>). We conduct a systematic paired evaluation of the complete symmetric group "
        "of three canonical degradations—Gaussian Noise (<i>N</i>), Defocus Blur (<i>B</i>), and JPEG Compression (<i>J</i>)—yielding all 3! = 6 "
        "permutation sequences across 5 severity levels evaluated on four prominent vision architectures (ResNet-50, ConvNeXt-Tiny, ViT-Base/16, Swin-Tiny), "
        "comprising 973,400 total forward-pass evaluations across two distinct experimental regimes (486,700 records per regime on 3,925 FastAI Imagenette-v2 images "
        "with deterministically seed-matched noise realizations). We formalize two distinct estimands: (1) <b>Total Operational Effect</b> at standard ImageNet-C parameters, "
        "where operator non-commutativity drives cross-sequence top-1 accuracy gaps of up to 21.55% (ResNet-50) and 20.45% (ConvNeXt-Tiny) via downstream "
        "terminal smoothing; and (2) <b>Fidelity-Conditioned Comparison</b>, where pre-calibration targeting common (PSNR, SSIM) support contracts the cross-sequence "
        "range of the four-model mean accuracies by 63.18% (dropping from 18.43% down to 6.79%, while the average of the four individual architecture gap contractions is 58.08%, "
        "contracting by 77.15% in ConvNeXt and 72.62% in ResNet). Crucially, prespecified tolerances (±0.75 dB PSNR, ±0.035 SSIM) were not achieved "
        "(PSNR RMSE = 0.943 dB, SSIM RMSE = 0.1380), establishing this regime strictly as an approximate-fidelity descriptive comparison. "
        "Beyond aggregate accuracy, we formulate a complete 5-state joint prediction space partition (<i>E</i><sub>11</sub>, <i>E</i><sub>10</sub>, <i>E</i><sub>01</sub>, "
        "<i>E</i><sub>00</sub><sup>same</sup>, <i>E</i><sub>00</sub><sup>diff</sup>) and prove that <b>Predicted-Label Churn</b> (P(<i>Ŷ<sub>A</sub> ≠ Ŷ<sub>B</sub></i>)) "
        "is more than double <b>Correctness-State Churn</b> (P(<i>C<sub>A</sub> ≠ C<sub>B</sub></i>) = <i>L + G</i>), because Error Migration (<i>E</i><sub>00</sub><sup>diff</sup>) "
        "averages 23.85% in nominal pipelines and 22.40% under approximate fidelity. Pairs with near-identical accuracy (|ΔAcc| < 2.5%) exhibit up to 10.97% excess churn "
        "and over 40% label churn, proving that sequence order actively reshuffles classifier decision boundaries across identical visual instances."
    )
    t_abs = Table([[Paragraph(abstract_text, ParagraphStyle("Abs", parent=body_style, fontSize=6.5, leading=8.6, textColor=colors.HexColor("#1e293b")))]], colWidths=[504])
    t_abs.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_abs)
    story.append(Spacer(1, 4))

    # Section 1: Introduction & Two Distinct Estimands
    story.append(Paragraph("1. Introduction & The Two Distinct Estimands", h1_style))
    story.append(Paragraph(
        "Benchmarking neural network robustness to distribution shift is essential for safety-critical deployment. While ImageNet-C (Hendrycks & Dietterich, 2019) "
        "established standard protocols for evaluating 15 isolated degradations, physical imaging pipelines subject images to multi-stage degradations. "
        "Because operators governing image formation (spatial convolution, stochastic noise, and block DCT quantization) are <b>non-commutative</b> "
        "(<i>T<sub>A</sub> ∘ T<sub>B</sub> ≠ T<sub>B</sub> ∘ T<sub>A</sub></i>), the sequence order in which operators act creates distinct physical signals and feature representations (Liao et al., 2020).<br/>"
        "We explicitly formalize two distinct estimands:<br/>"
        "• <b>Estimand 1 (Total Operational Effect at Fixed Parameters):</b> Evaluates the net impact of sequence order when operators run at identical standard nominal settings. "
        "Here, low-level fidelity disparities (e.g., higher PSNR when blur follows JPEG) are direct physical consequences of operator non-commutativity in real pipelines.<br/>"
        "• <b>Estimand 2 (Fidelity-Conditioned Descriptive Comparison):</b> Evaluates whether cross-sequence accuracy disparities persist when operator parameters are calibrated "
        "to target common (PSNR, SSIM) support. Because discrete calibration did not satisfy prespecified tolerances (±0.75 dB PSNR, ±0.035 SSIM), this estimand provides a "
        "<i>descriptive comparison under approximate common support</i>, rather than an exact causal effect at matched fidelity. Confounding was reduced, but not eliminated.", body_style
    ))

    # ================= PAGE 2 =================
    story.append(PageBreak())

    # Section 2: Related Work & Novelty Demarcation
    story.append(Paragraph("2. Related Work & Novelty Demarcation", h1_style))
    story.append(Paragraph(
        "Prior benchmarks predominantly evaluate unary corruptions (ImageNet-C) or temporal video sequences (ImageNet-P). Dodge & Karam (2016, 2019) pioneered psychometric distortion "
        "matching on single corruptions. Goyal et al. (2022) studied dual corruptions (noise and motion blur in low-light imaging), noting order asymmetry; however, they evaluated only 2! = 2 "
        "orderings without calibrating parameters to common support across the full permutation group. Li et al. (2025) introduced R-Bench, evaluating a 7-stage physical pipeline on MLLMs, "
        "but followed a single unidirectional acquisition sequence without permuting stages. Liao et al. (2020) demonstrated in digital forensics that 2-operator chains leave detectable forensic traces, "
        "confirming operator non-commutativity. Wei et al. (2026) showed that paired single-image evaluations uncover classwise correctness bounds under unary visual perturbations. "
        "Prediction churn was formalized across model retraining iterations by Milani Fard, Cormier, Canini, & Gupta (2016). RobustLens provides the first full-factorial 3! = 6 permutation study "
        "on identical physical images with seed-matched noise, formulating a 5-state prediction space partition that decouples predicted-label churn from correctness churn, a full-rank 6-cell decomposition, "
        "and an empirical fidelity-calibration audit documenting gap contraction.", body_style
    ))

    # Section 3: Mathematical Formulation & 5-State Joint Prediction Partition
    story.append(Paragraph("3. Mathematical Framework & Full 5-State Prediction Partition", h1_style))
    story.append(Paragraph(
        "Let <i>x</i> be a clean RGB image. Operators are Gaussian Noise (<i>N<sub>σ</sub></i>), Defocus Blur (<i>B<sub>κ,γ</sub></i>), and JPEG Compression (<i>J<sub>q</sub></i>). "
        "The symmetric group S<sub>3</sub> contains 6 sequences: NBJ, NJB, BNJ, BJN, JNB, JBN (applied left to right; e.g., NBJ ends in JPEG, NJB ends in Blur, BJN ends in Noise).<br/>"
        "Let <i>Ŷ<sub>A</sub>, Ŷ<sub>B</sub></i> be predicted classes, <i>Y</i> ground truth, and <i>C<sub>A</sub>, C<sub>B</sub> ∈ {0, 1}</i> correctness indicators. "
        "The joint prediction space partitions into five mutually exclusive, exhaustive transitions:<br/>"
        "1. <i>E</i><sub>11</sub> (Both Correct): <i>Ŷ<sub>A</sub> = Y ∧ Ŷ<sub>B</sub> = Y</i>; 2. <i>E</i><sub>10</sub> (Loss / <i>L</i>): <i>Ŷ<sub>A</sub> = Y ∧ Ŷ<sub>B</sub> ≠ Y</i>; "
        "3. <i>E</i><sub>01</sub> (Gain / <i>G</i>): <i>Ŷ<sub>A</sub> ≠ Y ∧ Ŷ<sub>B</sub> = Y</i>; 4. <i>E</i><sub>00</sub><sup>same</sup> (Same Wrong Class): <i>Ŷ<sub>A</sub> = Ŷ<sub>B</sub> ≠ Y</i>; "
        "5. <i>E</i><sub>00</sub><sup>diff</sup> (Error Migration / Different Wrong Classes): <i>Ŷ<sub>A</sub> ≠ Y ∧ Ŷ<sub>B</sub> ≠ Y ∧ Ŷ<sub>A</sub> ≠ Ŷ<sub>B</sub></i>.", body_style
    ))
    story.append(Paragraph(
        "Conservation of probability holds: <i>P(E<sub>11</sub>) + P(E<sub>10</sub>) + P(E<sub>01</sub>) + P(E<sub>00</sub><sup>same</sup>) + P(E<sub>00</sub><sup>diff</sup>) = 1</i>.<br/>"
        "We define and distinguish two churn metrics:<br/>"
        "• <b>Correctness-State Churn:</b> <i>P(C<sub>A</sub> ≠ C<sub>B</sub>) = P(E<sub>10</sub>) + P(E<sub>01</sub>) = L + G</i><br/>"
        "• <b>Predicted-Label Churn:</b> <i>P(Ŷ<sub>A</sub> ≠ Ŷ<sub>B</sub>) = L + G + P(E<sub>00</sub><sup>diff</sup>) = </i> Correctness Churn + Error Migration.<br/>"
        "• <b>Agreement Rate:</b> <i>P(Ŷ<sub>A</sub> = Ŷ<sub>B</sub>) = P(E<sub>11</sub>) + P(E<sub>00</sub><sup>same</sup>) = 1 - </i> Label Churn.<br/>"
        "• <b>Excess Churn:</b> Excess Correctness Churn = 2·min(<i>G, L</i>) = (<i>G+L</i>) - |ΔAcc|; Excess Label Churn = 2·min(<i>G, L</i>) + <i>P(E<sub>00</sub><sup>diff</sup>)</i>.<br/>"
        "<b>Theorem:</b> Whenever models make distinct incorrect predictions (<i>P(E<sub>00</sub><sup>diff</sup>) > 0</i>), Correctness Churn strictly understates classifier instability. "
        "In our experiments, Error Migration is ~23%, causing Predicted-Label Churn to be more than double Correctness Churn.", body_style
    ))

    # Section 4: Saturated Six-Cell Decomposition
    story.append(Paragraph("4. Full-Rank Saturated Six-Cell Linear Model & Statistical Inference", h1_style))
    story.append(Paragraph(
        "To decompose sequence variance across all six cell means <i>μ = [μ<sub>NBJ</sub>, μ<sub>NJB</sub>, μ<sub>BNJ</sub>, μ<sub>BJN</sub>, μ<sub>JNB</sub>, μ<sub>JBN</sub>]<sup>T</sup></i>, "
        "we specify the linear model <i>μ = X β</i> with reference levels Start Noise and End Noise:<br/>"
        "Columns of <i>X</i>: [Intercept, Start Blur, Start JPEG, End Blur, End JPEG, Parity Contrast (Even: +1, Odd: -1)]. "
        "Because <i>det(X) = -6 ≠ 0</i>, <i>X</i> has full rank 6, uniquely determining <i>β = X<sup>-1</sup> μ</i>.<br/>"
        "<b>Critical Design Properties:</b><br/>"
        "1. <b>Non-Orthogonal Design:</b> <i>X<sup>T</sup> X ≠ I</i>; reference-coded parameters must be interpreted alongside direct marginal contrasts.<br/>"
        "2. <b>Unobserved Reference Condition:</b> No 3-operator sequence without repetition can both start and end with Noise. Hence, Intercept <i>β</i><sub>0</sub> is an extrapolated linear parameter, not an observable baseline.<br/>"
        "3. <b>Balanced Parity Contrast:</b> Even permutations (<i>A</i><sub>3</sub>: NBJ, BJN, JNB) and Odd permutations (NJB, BNJ, JBN) both contain every operator exactly once as an entry operator and exactly once as an exit operator. "
        "The coefficient <i>β</i><sub>5</sub> = (μ̄<sub>Even</sub> - μ̄<sub>Odd</sub>)/2 is perfectly balanced against entry and exit marginals, with full contrast 2<i>β</i><sub>5</sub>.<br/>"
        "4. <b>Clustered Bootstrap & Multiplicity:</b> We perform clustered bootstrap resampling (<i>B = 5,000</i> clusters on 3,925 physical image IDs). Asymptotic two-sided Wald p-values are computed as <i>log p = log(2) + logsf(|z|)</i> using cluster-bootstrap standard errors. "
        "Holm step-down adjustment controls FWER (<i>α = 0.05</i>) on primary contrasts; Benjamini-Hochberg controls FDR (<i>q = 0.05</i>) on secondary families.", body_style
    ))

    # ================= PAGE 3 =================
    story.append(PageBreak())

    # Section 5: Regime 1: Nominal-Order Benchmark
    story.append(Paragraph("5. Regime 1: Nominal-Order Benchmark (Estimand 1)", h1_style))
    story.append(Paragraph(
        "Under fixed nominal parameters (ImageNet-C severities 1–5), models evaluated on identical physical images with matched noise seeds exhibit massive performance gaps.", body_style
    ))

    # Figure 1 Image
    fig1_path = FIG_DIR / "figure1_order_accuracy.png"
    if fig1_path.exists():
        story.append(Image(str(fig1_path), width=6.8*inch, height=2.85*inch))
        story.append(Spacer(1, 2))

    # Table 1: Nominal Model Accuracy & Brier Scores
    t1_data = [
        [Paragraph("<b>Model</b>", table_header), Paragraph("<b>Arch</b>", table_header), Paragraph("<b>NBJ</b>", table_header), Paragraph("<b>NJB</b>", table_header), Paragraph("<b>BNJ</b>", table_header), Paragraph("<b>BJN</b>", table_header), Paragraph("<b>JNB</b>", table_header), Paragraph("<b>JBN</b>", table_header), Paragraph("<b>Mean</b>", table_header), Paragraph("<b>Min</b>", table_header), Paragraph("<b>Max</b>", table_header), Paragraph("<b>Gap (Δ)</b>", table_header), Paragraph("<b>Brier</b>", table_header)],
        [Paragraph("<b>ResNet-50</b>", table_cell_left), Paragraph("CNN", table_cell), Paragraph("53.44%", table_cell), Paragraph("56.88%", table_cell), Paragraph("37.30%", table_cell), Paragraph("35.71%", table_cell), Paragraph("57.27%", table_cell), Paragraph("35.79%", table_cell), Paragraph("46.06%", table_cell), Paragraph("35.71%", table_cell), Paragraph("57.27%", table_cell), Paragraph("<b>21.55%</b>", table_cell), Paragraph("0.1587", table_cell)],
        [Paragraph("<b>ConvNeXt-T</b>", table_cell_left), Paragraph("Modern CNN", table_cell), Paragraph("63.98%", table_cell), Paragraph("66.44%", table_cell), Paragraph("48.63%", table_cell), Paragraph("45.99%", table_cell), Paragraph("65.44%", table_cell), Paragraph("46.33%", table_cell), Paragraph("56.14%", table_cell), Paragraph("45.99%", table_cell), Paragraph("66.44%", table_cell), Paragraph("<b>20.45%</b>", table_cell), Paragraph("0.1265", table_cell)],
        [Paragraph("<b>ViT-B/16</b>", table_cell_left), Paragraph("Transformer", table_cell), Paragraph("58.39%", table_cell), Paragraph("60.14%", table_cell), Paragraph("39.77%", table_cell), Paragraph("41.64%", table_cell), Paragraph("59.19%", table_cell), Paragraph("42.23%", table_cell), Paragraph("50.23%", table_cell), Paragraph("39.77%", table_cell), Paragraph("60.14%", table_cell), Paragraph("<b>20.37%</b>", table_cell), Paragraph("0.1481", table_cell)],
        [Paragraph("<b>Swin-Tiny</b>", table_cell_left), Paragraph("Hier. ViT", table_cell), Paragraph("59.81%", table_cell), Paragraph("61.78%", table_cell), Paragraph("45.81%", table_cell), Paragraph("50.24%", table_cell), Paragraph("62.94%", table_cell), Paragraph("50.63%", table_cell), Paragraph("55.20%", table_cell), Paragraph("45.81%", table_cell), Paragraph("62.94%", table_cell), Paragraph("<b>17.13%</b>", table_cell), Paragraph("0.1286", table_cell)],
        [Paragraph("<b>Overall Mean</b>", table_cell_left), Paragraph("—", table_cell), Paragraph("58.90%", table_cell), Paragraph("61.31%", table_cell), Paragraph("42.88%", table_cell), Paragraph("43.40%", table_cell), Paragraph("61.21%", table_cell), Paragraph("43.75%", table_cell), Paragraph("51.91%", table_cell), Paragraph("42.88%", table_cell), Paragraph("61.31%", table_cell), Paragraph("<b>18.43%</b>", table_cell), Paragraph("0.1405", table_cell)],
    ]
    t1 = Table(t1_data, colWidths=[68, 50, 31, 31, 31, 31, 31, 31, 36, 31, 31, 46, 36])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    story.append(t1)
    story.append(Spacer(1, 3))

    story.append(Paragraph(
        "<b>Table 1: Nominal-Order Top-1 Accuracy & Brier Scores.</b> Evaluated across 3,925 images (19,625 evaluations per sequence per model; 471,000 compound records).<br/>"
        "<b>Direct Marginal Contrasts:</b> Start Blur vs. Noise: -16.97% (<i>p < 10<sup>-100</sup></i>); Start JPEG vs. Noise: -7.63% (<i>p < 10<sup>-100</sup></i>). "
        "End Blur vs. Noise: <b>+17.69%</b> (<i>p < 10<sup>-100</sup></i>); End JPEG vs. Noise: +7.32% (<i>p < 10<sup>-100</sup></i>). Balanced Parity Contrast: <b>+5.19%</b> (2<i>β</i><sub>5</sub>, <i>p < 10<sup>-100</sup></i>).<br/>"
        "<b>Finding:</b> In nominal evaluations, the massive 18.43% sequence gap is dominated by terminal operator identity: ending in Blur confers a +17.69% marginal boost over terminal noise "
        "because spatial convolution attenuates prior high-frequency noise and DCT block artifacts, boosting PSNR by ~6.2 dB (mean PSNR 24.6 dB for terminal blur vs. 18.3 dB for terminal noise).", body_style
    ))

    # Figure 2 Image
    fig2_path = FIG_DIR / "figure2_accuracy_fidelity.png"
    if fig2_path.exists():
        story.append(Image(str(fig2_path), width=6.8*inch, height=2.5*inch))
        story.append(Spacer(1, 2))

    # ================= PAGE 4 =================
    story.append(PageBreak())

    # Section 6: Regime 2: PSNR/SSIM-Targeted Descriptive Comparison
    story.append(Paragraph("6. Regime 2: PSNR/SSIM-Targeted Descriptive Comparison (Estimand 2)", h1_style))
    story.append(Paragraph(
        "To test whether order sensitivity persists when output distortion is constrained, operator parameters were pre-calibrated on an independent training partition (<i>N</i><sub>cal</sub> = 50) "
        "targeting five common (PSNR, SSIM) support bins across sequences.", body_style
    ))

    # Table 2: Measured Calibration Diagnostics and Tolerance Violations
    t2_data = [
        [Paragraph("<b>Target</b>", table_header), Paragraph("<b>Target PSNR</b>", table_header), Paragraph("<b>Target SSIM</b>", table_header), Paragraph("<b>Max Abs PSNR Resid</b>", table_header), Paragraph("<b>PSNR Tol (≤ 0.75 dB)</b>", table_header), Paragraph("<b>Max Abs SSIM Resid</b>", table_header), Paragraph("<b>SSIM Tol (≤ 0.035)</b>", table_header)],
        [Paragraph("Target 1", table_cell), Paragraph("28.0 dB", table_cell), Paragraph("0.850", table_cell), Paragraph("1.117 dB (NJB)", table_cell), Paragraph("<font color='#b91c1c'><b>Failed</b></font>", table_cell), Paragraph("0.0686 (BJN)", table_cell), Paragraph("<font color='#b91c1c'><b>Failed</b></font>", table_cell)],
        [Paragraph("Target 2", table_cell), Paragraph("24.5 dB", table_cell), Paragraph("0.740", table_cell), Paragraph("0.936 dB (BJN)", table_cell), Paragraph("<font color='#b91c1c'><b>Failed</b></font>", table_cell), Paragraph("0.1758 (BJN)", table_cell), Paragraph("<font color='#b91c1c'><b>Failed</b></font>", table_cell)],
        [Paragraph("Target 3", table_cell), Paragraph("21.5 dB", table_cell), Paragraph("0.620", table_cell), Paragraph("1.260 dB (NBJ)", table_cell), Paragraph("<font color='#b91c1c'><b>Failed</b></font>", table_cell), Paragraph("0.2379 (BJN)", table_cell), Paragraph("<font color='#b91c1c'><b>Failed</b></font>", table_cell)],
        [Paragraph("Target 4", table_cell), Paragraph("18.5 dB", table_cell), Paragraph("0.480", table_cell), Paragraph("1.989 dB (NBJ)", table_cell), Paragraph("<font color='#b91c1c'><b>Failed</b></font>", table_cell), Paragraph("0.2373 (BJN)", table_cell), Paragraph("<font color='#b91c1c'><b>Failed</b></font>", table_cell)],
        [Paragraph("Target 5", table_cell), Paragraph("16.0 dB", table_cell), Paragraph("0.360", table_cell), Paragraph("2.639 dB (NBJ)", table_cell), Paragraph("<font color='#b91c1c'><b>Failed</b></font>", table_cell), Paragraph("0.2067 (BJN)", table_cell), Paragraph("<font color='#b91c1c'><b>Failed</b></font>", table_cell)],
        [Paragraph("<b>Overall</b>", table_header), Paragraph("—", table_cell), Paragraph("—", table_cell), Paragraph("<b>2.639 dB (Max)</b>", table_cell), Paragraph("<b>RMSE: 0.943 dB (Failed)</b>", table_cell), Paragraph("<b>0.2379 (Max)</b>", table_cell), Paragraph("<b>RMSE: 0.1380 (Failed)</b>", table_cell)],
    ]
    t2 = Table(t2_data, colWidths=[55, 65, 65, 95, 80, 85, 59])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#fee2e2")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
    ]))
    story.append(t2)
    story.append(Spacer(1, 2))

    # Table 3: Approximate-Fidelity Accuracy & Gap Contraction
    t3_data = [
        [Paragraph("<b>Model</b>", table_header), Paragraph("<b>NBJ</b>", table_header), Paragraph("<b>NJB</b>", table_header), Paragraph("<b>BNJ</b>", table_header), Paragraph("<b>BJN</b>", table_header), Paragraph("<b>JNB</b>", table_header), Paragraph("<b>JBN</b>", table_header), Paragraph("<b>Matched Mean</b>", table_header), Paragraph("<b>Matched Gap</b>", table_header), Paragraph("<b>Nominal Gap</b>", table_header), Paragraph("<b>Contraction</b>", table_header), Paragraph("<b>Mean Brier</b>", table_header)],
        [Paragraph("<b>ResNet-50</b>", table_cell_left), Paragraph("50.15%", table_cell), Paragraph("46.76%", table_cell), Paragraph("45.18%", table_cell), Paragraph("50.63%", table_cell), Paragraph("47.05%", table_cell), Paragraph("51.08%", table_cell), Paragraph("48.47%", table_cell), Paragraph("<b>5.90%</b>", table_cell), Paragraph("21.55%", table_cell), Paragraph("<b>72.62%</b>", table_cell), Paragraph("0.1558", table_cell)],
        [Paragraph("<b>ConvNeXt-T</b>", table_cell_left), Paragraph("60.65%", table_cell), Paragraph("56.46%", table_cell), Paragraph("55.97%", table_cell), Paragraph("59.22%", table_cell), Paragraph("56.47%", table_cell), Paragraph("59.86%", table_cell), Paragraph("58.11%", table_cell), Paragraph("<b>4.67%</b>", table_cell), Paragraph("20.45%", table_cell), Paragraph("<b>77.15%</b>", table_cell), Paragraph("0.1245", table_cell)],
        [Paragraph("<b>ViT-B/16</b>", table_cell_left), Paragraph("56.54%", table_cell), Paragraph("53.39%", table_cell), Paragraph("46.51%", table_cell), Paragraph("54.72%", table_cell), Paragraph("53.01%", table_cell), Paragraph("55.68%", table_cell), Paragraph("53.31%", table_cell), Paragraph("<b>10.03%</b>", table_cell), Paragraph("20.37%", table_cell), Paragraph("<b>50.75%</b>", table_cell), Paragraph("0.1444", table_cell)],
        [Paragraph("<b>Swin-Tiny</b>", table_cell_left), Paragraph("57.61%", table_cell), Paragraph("50.44%", table_cell), Paragraph("53.93%", table_cell), Paragraph("61.23%", table_cell), Paragraph("52.31%", table_cell), Paragraph("62.12%", table_cell), Paragraph("56.27%", table_cell), Paragraph("<b>11.68%</b>", table_cell), Paragraph("17.13%", table_cell), Paragraph("<b>31.80%</b>", table_cell), Paragraph("0.1433", table_cell)],
        [Paragraph("<b>Overall Mean</b>", table_cell_left), Paragraph("56.23%", table_cell), Paragraph("51.76%", table_cell), Paragraph("50.40%", table_cell), Paragraph("56.45%", table_cell), Paragraph("52.21%", table_cell), Paragraph("57.18%", table_cell), Paragraph("54.04%", table_cell), Paragraph("<b>6.79%</b>", table_cell), Paragraph("18.43%", table_cell), Paragraph("<b>63.18%</b>", table_cell), Paragraph("0.1420", table_cell)],
    ]
    t3 = Table(t3_data, colWidths=[70, 34, 34, 34, 34, 34, 34, 46, 46, 46, 46, 46])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    story.append(t3)
    story.append(Spacer(1, 2))

    # Figure 3 Image
    fig3_path = FIG_DIR / "figure3_range_contraction.png"
    if fig3_path.exists():
        story.append(Image(str(fig3_path), width=6.8*inch, height=2.6*inch))
        story.append(Spacer(1, 2))

    # Table 4: Saturated Six-Cell Decomposition
    t4_data = [
        [Paragraph("<b>Parameter</b>", table_header), Paragraph("<b>Nominal Estimate (%)</b>", table_header), Paragraph("<b>Nominal 95% Clustered CI</b>", table_header), Paragraph("<b>Nominal <i>p</i><sub>Holm</sub></b>", table_header), Paragraph("<b>Matched Estimate (%)</b>", table_header), Paragraph("<b>Matched 95% Clustered CI</b>", table_header), Paragraph("<b>Matched <i>p</i><sub>Holm</sub></b>", table_header)],
        [Paragraph("<b><i>β</i><sub>0</sub> (Intercept)</b>", table_cell_left), Paragraph("+51.63", table_cell), Paragraph("[+50.71, +52.52]", table_cell), Paragraph("< 10<sup>-100</sup>", table_cell), Paragraph("+59.51", table_cell), Paragraph("[+58.51, +60.51]", table_cell), Paragraph("< 10<sup>-100</sup>", table_cell)],
        [Paragraph("<b><i>β</i><sub>1</sub> (Start Blur)</b>", table_cell_left), Paragraph("-10.83", table_cell), Paragraph("[-11.14, -10.53]", table_cell), Paragraph("< 10<sup>-100</sup>", table_cell), Paragraph("-3.99", table_cell), Paragraph("[-4.22, -3.75]", table_cell), Paragraph("< 10<sup>-100</sup>", table_cell)],
        [Paragraph("<b><i>β</i><sub>2</sub> (Start JPEG)</b>", table_cell_left), Paragraph("-5.29", table_cell), Paragraph("[-5.50, -5.08]", table_cell), Paragraph("< 10<sup>-100</sup>", table_cell), Paragraph("-1.40", table_cell), Paragraph("[-1.58, -1.23]", table_cell), Paragraph("< 10<sup>-50</sup>", table_cell)],
        [Paragraph("<b><i>β</i><sub>3</sub> (End Blur)</b>", table_cell_left), Paragraph("<b>+12.27</b>", table_cell), Paragraph("[+11.88, +12.67]", table_cell), Paragraph("< 10<sup>-100</sup>", table_cell), Paragraph("<b>-6.82</b>", table_cell), Paragraph("[-7.18, -6.46]", table_cell), Paragraph("< 10<sup>-100</sup>", table_cell)],
        [Paragraph("<b><i>β</i><sub>4</sub> (End JPEG)</b>", table_cell_left), Paragraph("<b>+4.67</b>", table_cell), Paragraph("[+4.35, +5.00]", table_cell), Paragraph("< 10<sup>-100</sup>", table_cell), Paragraph("<b>-4.20</b>", table_cell), Paragraph("[-4.50, -3.91]", table_cell), Paragraph("< 10<sup>-100</sup>", table_cell)],
        [Paragraph("<b><i>β</i><sub>5</sub> (Parity Contrast)</b>", table_cell_left), Paragraph("<b>+2.60</b>", table_cell), Paragraph("[+2.52, +2.67]", table_cell), Paragraph("< 10<sup>-100</sup>", table_cell), Paragraph("<b>+0.93</b>", table_cell), Paragraph("[+0.86, +0.99]", table_cell), Paragraph("< 10<sup>-100</sup>", table_cell)],
    ]
    t4 = Table(t4_data, colWidths=[94, 65, 80, 55, 65, 80, 65])
    t4.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
    ]))
    story.append(t4)
    story.append(Spacer(1, 2))

    story.append(Paragraph(
        "<b>Table 4: Saturated Six-Cell Decomposition Parameters.</b> Even after conditioning on entry and exit operators, the balanced parity contrast <i>β</i><sub>5</sub> remains statistically significant in both regimes (+2.60% nominal, +0.93% matched, <i>p < 10<sup>-100</sup></i>).", body_style
    ))

    # ================= PAGE 5 =================
    story.append(PageBreak())

    # Section 7: Prediction Instability, Error Migration & Per-Class Breakdown
    story.append(Paragraph("7. Prediction Instability, Error Migration & Per-Class Breakdown", h1_style))
    story.append(Paragraph(
        "A cornerstone empirical finding is that <b>aggregate accuracy obscures massive per-image prediction churn and error migration.</b>", body_style
    ))

    # Figure 4 Image
    fig4_path = FIG_DIR / "figure4_prediction_churn.png"
    if fig4_path.exists():
        story.append(Image(str(fig4_path), width=6.8*inch, height=2.4*inch))
        story.append(Spacer(1, 2))

    # Table 5: 5-State Prediction Space Transitions
    t5_data = [
        [Paragraph("<b>Regime</b>", table_header), Paragraph("<b>Model</b>", table_header), Paragraph("<b>Both Corr (<i>E</i><sub>11</sub>)</b>", table_header), Paragraph("<b>Loss (<i>E</i><sub>10</sub>)</b>", table_header), Paragraph("<b>Gain (<i>E</i><sub>01</sub>)</b>", table_header), Paragraph("<b>Same Wrong (<i>E</i><sub>00</sub><sup>same</sup>)</b>", table_header), Paragraph("<b>Error Mig (<i>E</i><sub>00</sub><sup>diff</sup>)</b>", table_header), Paragraph("<b>Corr Churn</b>", table_header), Paragraph("<b>Excess Churn</b>", table_header), Paragraph("<b>Label Churn</b>", table_header), Paragraph("<b>Agree Rate</b>", table_header), Paragraph("<b>Mean |ΔConf|</b>", table_header)],
        [Paragraph("Nominal", table_cell), Paragraph("ConvNeXt", table_cell_left), Paragraph("47.63%", table_cell), Paragraph("11.64%", table_cell), Paragraph("5.38%", table_cell), Paragraph("14.99%", table_cell), Paragraph("<b>20.36%</b>", table_cell), Paragraph("17.02%", table_cell), Paragraph("5.36%", table_cell), Paragraph("<b>37.38%</b>", table_cell), Paragraph("62.62%", table_cell), Paragraph("0.1662", table_cell)],
        [Paragraph("Nominal", table_cell), Paragraph("ResNet-50", table_cell_left), Paragraph("36.96%", table_cell), Paragraph("12.06%", table_cell), Paragraph("6.15%", table_cell), Paragraph("14.90%", table_cell), Paragraph("<b>29.93%</b>", table_cell), Paragraph("18.20%", table_cell), Paragraph("5.72%", table_cell), Paragraph("<b>48.14%</b>", table_cell), Paragraph("51.86%", table_cell), Paragraph("0.1850", table_cell)],
        [Paragraph("Nominal", table_cell), Paragraph("Swin-Tiny", table_cell_left), Paragraph("47.98%", table_cell), Paragraph("8.49%", table_cell), Paragraph("5.96%", table_cell), Paragraph("16.06%", table_cell), Paragraph("<b>21.51%</b>", table_cell), Paragraph("14.45%", table_cell), Paragraph("5.82%", table_cell), Paragraph("<b>35.96%</b>", table_cell), Paragraph("64.04%", table_cell), Paragraph("0.1835", table_cell)],
        [Paragraph("Nominal", table_cell), Paragraph("ViT-Base", table_cell_left), Paragraph("41.20%", table_cell), Paragraph("11.76%", table_cell), Paragraph("6.31%", table_cell), Paragraph("17.17%", table_cell), Paragraph("<b>23.57%</b>", table_cell), Paragraph("18.06%", table_cell), Paragraph("6.68%", table_cell), Paragraph("<b>41.63%</b>", table_cell), Paragraph("58.37%", table_cell), Paragraph("0.1732", table_cell)],
        [Paragraph("<b>Nominal</b>", table_cell), Paragraph("<b>Overall</b>", table_cell_left), Paragraph("<b>43.44%</b>", table_cell), Paragraph("<b>10.99%</b>", table_cell), Paragraph("<b>5.95%</b>", table_cell), Paragraph("<b>15.78%</b>", table_cell), Paragraph("<b>23.85%</b>", table_cell), Paragraph("<b>16.93%</b>", table_cell), Paragraph("<b>5.90%</b>", table_cell), Paragraph("<b>40.78%</b>", table_cell), Paragraph("<b>59.22%</b>", table_cell), Paragraph("<b>0.1770</b>", table_cell)],
        [Paragraph("Matched", table_cell), Paragraph("ConvNeXt", table_cell_left), Paragraph("52.64%", table_cell), Paragraph("5.49%", table_cell), Paragraph("5.44%", table_cell), Paragraph("17.44%", table_cell), Paragraph("<b>19.00%</b>", table_cell), Paragraph("10.93%", table_cell), Paragraph("8.51%", table_cell), Paragraph("<b>29.92%</b>", table_cell), Paragraph("70.08%", table_cell), Paragraph("0.1163", table_cell)],
        [Paragraph("Matched", table_cell), Paragraph("ResNet-50", table_cell_left), Paragraph("41.92%", table_cell), Paragraph("6.18%", table_cell), Paragraph("6.92%", table_cell), Paragraph("15.54%", table_cell), Paragraph("<b>29.44%</b>", table_cell), Paragraph("13.10%", table_cell), Paragraph("10.15%", table_cell), Paragraph("<b>42.54%</b>", table_cell), Paragraph("57.46%", table_cell), Paragraph("0.1481", table_cell)],
        [Paragraph("Matched", table_cell), Paragraph("Swin-Tiny", table_cell_left), Paragraph("50.31%", table_cell), Paragraph("4.78%", table_cell), Paragraph("7.14%", table_cell), Paragraph("18.03%", table_cell), Paragraph("<b>19.74%</b>", table_cell), Paragraph("11.92%", table_cell), Paragraph("5.99%", table_cell), Paragraph("<b>31.66%</b>", table_cell), Paragraph("68.34%", table_cell), Paragraph("0.1307", table_cell)],
        [Paragraph("Matched", table_cell), Paragraph("ViT-Base", table_cell_left), Paragraph("46.61%", table_cell), Paragraph("6.61%", table_cell), Paragraph("6.79%", table_cell), Paragraph("18.55%", table_cell), Paragraph("<b>21.44%</b>", table_cell), Paragraph("13.40%", table_cell), Paragraph("9.44%", table_cell), Paragraph("<b>34.85%</b>", table_cell), Paragraph("65.15%", table_cell), Paragraph("0.1314", table_cell)],
        [Paragraph("<b>Matched</b>", table_cell), Paragraph("<b>Overall</b>", table_cell_left), Paragraph("<b>47.87%</b>", table_cell), Paragraph("<b>5.76%</b>", table_cell), Paragraph("<b>6.57%</b>", table_cell), Paragraph("<b>17.39%</b>", table_cell), Paragraph("<b>22.40%</b>", table_cell), Paragraph("<b>12.34%</b>", table_cell), Paragraph("<b>8.52%</b>", table_cell), Paragraph("<b>34.74%</b>", table_cell), Paragraph("<b>65.26%</b>", table_cell), Paragraph("<b>0.1316</b>", table_cell)],
    ]
    t5 = Table(t5_data, colWidths=[40, 52, 44, 38, 38, 48, 48, 38, 38, 42, 42, 36])
    t5.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("BACKGROUND", (0, 5), (-1, 5), colors.HexColor("#f8fafc")),
        ("BACKGROUND", (0, 10), (-1, 10), colors.HexColor("#f8fafc")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.0),
    ]))
    story.append(t5)
    story.append(Spacer(1, 2))

    story.append(Paragraph(
        "<b>Table 5: Complete 5-State Joint Prediction Space Transitions.</b> Error Migration (<i>E</i><sub>00</sub><sup>diff</sup>) accounts for 22.40%–23.85% of all evaluations, pushing Predicted-Label Churn to more than double Correctness Churn.", body_style
    ))

    # Table 6: Per-Class Breakdown Across 10 Imagenette Classes
    t6_data = [
        [Paragraph("<b>Class Name</b>", table_header), Paragraph("<b>Nom Acc</b>", table_header), Paragraph("<b>Nom Conf</b>", table_header), Paragraph("<b>Nom Brier</b>", table_header), Paragraph("<b>Nom Label Churn</b>", table_header), Paragraph("<b>Nom Corr Churn</b>", table_header), Paragraph("<b>Nom Error Mig</b>", table_header), Paragraph("<b>Matched Acc</b>", table_header), Paragraph("<b>Matched Label Churn</b>", table_header), Paragraph("<b>Matched Error Mig</b>", table_header)],
        [Paragraph("<b>tench</b>", table_cell_left), Paragraph("73.57%", table_cell), Paragraph("66.25%", table_cell), Paragraph("0.0921", table_cell), Paragraph("29.09%", table_cell), Paragraph("17.56%", table_cell), Paragraph("11.54%", table_cell), Paragraph("77.22%", table_cell), Paragraph("21.54%", table_cell), Paragraph("11.91%", table_cell)],
        [Paragraph("<b>golf ball</b>", table_cell_left), Paragraph("72.45%", table_cell), Paragraph("66.65%", table_cell), Paragraph("0.1088", table_cell), Paragraph("28.86%", table_cell), Paragraph("17.89%", table_cell), Paragraph("10.97%", table_cell), Paragraph("75.07%", table_cell), Paragraph("21.88%", table_cell), Paragraph("10.83%", table_cell)],
        [Paragraph("<b>parachute</b>", table_cell_left), Paragraph("68.01%", table_cell), Paragraph("61.06%", table_cell), Paragraph("0.1224", table_cell), Paragraph("31.19%", table_cell), Paragraph("16.22%", table_cell), Paragraph("14.97%", table_cell), Paragraph("68.97%", table_cell), Paragraph("28.00%", table_cell), Paragraph("15.75%", table_cell)],
        [Paragraph("<b>chain saw</b>", table_cell_left), Paragraph("55.56%", table_cell), Paragraph("51.44%", table_cell), Paragraph("0.1170", table_cell), Paragraph("41.93%", table_cell), Paragraph("17.42%", table_cell), Paragraph("24.51%", table_cell), Paragraph("59.32%", table_cell), Paragraph("34.59%", table_cell), Paragraph("22.64%", table_cell)],
        [Paragraph("<b>French horn</b>", table_cell_left), Paragraph("51.98%", table_cell), Paragraph("52.58%", table_cell), Paragraph("0.1053", table_cell), Paragraph("41.77%", table_cell), Paragraph("14.21%", table_cell), Paragraph("27.56%", table_cell), Paragraph("53.81%", table_cell), Paragraph("36.00%", table_cell), Paragraph("27.08%", table_cell)],
        [Paragraph("<b>garbage truck</b>", table_cell_left), Paragraph("49.61%", table_cell), Paragraph("50.66%", table_cell), Paragraph("0.1528", table_cell), Paragraph("43.23%", table_cell), Paragraph("19.56%", table_cell), Paragraph("23.67%", table_cell), Paragraph("51.25%", table_cell), Paragraph("36.10%", table_cell), Paragraph("21.24%", table_cell)],
        [Paragraph("<b>gas pump</b>", table_cell_left), Paragraph("48.57%", table_cell), Paragraph("48.14%", table_cell), Paragraph("0.1253", table_cell), Paragraph("46.52%", table_cell), Paragraph("17.51%", table_cell), Paragraph("29.01%", table_cell), Paragraph("50.56%", table_cell), Paragraph("40.01%", table_cell), Paragraph("28.27%", table_cell)],
        [Paragraph("<b>cassette player</b>", table_cell_left), Paragraph("41.35%", table_cell), Paragraph("42.38%", table_cell), Paragraph("0.1988", table_cell), Paragraph("45.62%", table_cell), Paragraph("20.64%", table_cell), Paragraph("24.98%", table_cell), Paragraph("44.40%", table_cell), Paragraph("39.95%", table_cell), Paragraph("22.40%", table_cell)],
        [Paragraph("<b>English springer</b>", table_cell_left), Paragraph("40.48%", table_cell), Paragraph("40.22%", table_cell), Paragraph("0.1786", table_cell), Paragraph("48.72%", table_cell), Paragraph("17.85%", table_cell), Paragraph("30.87%", table_cell), Paragraph("41.61%", table_cell), Paragraph("42.85%", table_cell), Paragraph("28.25%", table_cell)],
        [Paragraph("<b>church</b>", table_cell_left), Paragraph("<b>18.35%</b>", table_cell), Paragraph("38.21%", table_cell), Paragraph("0.2065", table_cell), Paragraph("<b>50.45%</b>", table_cell), Paragraph("<b>11.04%</b>", table_cell), Paragraph("<b>39.41%</b>", table_cell), Paragraph("19.22%", table_cell), Paragraph("<b>46.07%</b>", table_cell), Paragraph("<b>34.70%</b>", table_cell)],
    ]
    t6 = Table(t6_data, colWidths=[74, 46, 46, 46, 52, 48, 50, 46, 50, 46])
    t6.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#fef2f2")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.0),
    ]))
    story.append(t6)
    story.append(Spacer(1, 2))

    # Section 8: Scope Boundaries & Limitations
    story.append(Paragraph("8. Scope Boundaries, Limitations & Reproducibility", h1_style))
    story.append(Paragraph(
        "1. <b>Approximate Fidelity, Not Exact Matching:</b> Prespecified tolerances (±0.75 dB PSNR, ±0.035 SSIM) were not achieved over the tested discrete grid. Residual PSNR spread reached 3.10 dB at Target 5. Residual differences cannot be interpreted as pure architectural effects without acknowledging residual distortion imbalance.<br/>"
        "2. <b>Finite Grid Boundary:</b> The failure of our discrete parameter search does not prove universal physical impossibility across continuous spaces or alternate corruption formulations.<br/>"
        "3. <b>Dataset & Task Scope:</b> Evaluated on FastAI Imagenette-v2 (10 classes, 3,925 images). Evaluations on full ImageNet-1k and dense prediction tasks remain open for future work.<br/>"
        "4. <b>Data Availability:</b> All 973,400 raw prediction records and reproducible scripts are archived in the repository.", body_style
    ))

    # References in two columns
    story.append(Paragraph("References", h1_style))
    ref_style = ParagraphStyle("Ref", parent=body_style, alignment=TA_LEFT, fontSize=6.0, leading=8.0, spaceAfter=1.5)
    refs_col1 = [
        "1. Bhojanapalli, S., et al. (2021). Understanding robustness of transformers. <i>ICCV</i>.",
        "2. D'Amour, A., et al. (2020). Underspecification presents challenges for credibility in ML. <i>JMLR</i>.",
        "3. Dodge, S., & Karam, L. (2016). Understanding how image quality affects DNNs. <i>QoMEX</i>.",
        "4. Dodge, S., & Karam, L. (2019). Human and DNN performance on images with quality distortions. <i>ACM TAP</i>.",
        "5. Dosovitskiy, A., et al. (2021). An image is worth 16x16 words: Transformers at scale. <i>ICLR</i>.",
        "6. Erkara, A. O., & Mayer, M. (2026). Blind confusion of classification networks. <i>Neurocomputing</i>, 686:133678.",
        "7. Goyal, B., et al. (2022). Robust scene inference under noise-blur dual corruptions. <i>ICCP</i>.",
        "8. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. <i>CVPR</i>.",
        "9. Hendrycks, D., & Dietterich, T. (2019). Benchmarking robustness to common corruptions. <i>ICLR</i>.",
    ]
    refs_col2 = [
        "10. Kang, D., et al. (2019). Testing robustness against unforeseen adversaries. <i>arXiv:1908.08016</i>.",
        "11. Li, Y., et al. (2025). R-Bench: Benchmarking multi-stage physical degradation in MLLMs. <i>IEEE JSTSP</i>.",
        "12. Liao, X., Li, K., Zhu, X., & Liu, K. J. R. (2020). Robust detection of image operator chain. <i>IEEE JSTSP</i>.",
        "13. Liu, Z., et al. (2021). Swin transformer: Hierarchical ViT using shifted windows. <i>ICCV</i>.",
        "14. Liu, Z., et al. (2022). A convnet for the 2020s. <i>CVPR</i>.",
        "15. Milani Fard, M., Cormier, Q., Canini, K., & Gupta, M. R. (2016). Launch and iterate: Reducing prediction churn. <i>NeurIPS</i>, 29:3179-3187.",
        "16. Mintun, E., et al. (2021). On interaction between augmentations and corruptions. <i>NeurIPS</i>.",
        "17. RichardWebster, B., et al. (2018). Visual psychophysics for deep neural networks. <i>CVPR</i>.",
        "18. Wei, Y., et al. (2026). What paired evaluations reveal under visual perturbations. <i>arXiv:2609.35583</i>.",
    ]
    ref_table_data = []
    for r1, r2 in zip(refs_col1, refs_col2):
        p1 = Paragraph(r1, ref_style) if r1 else Paragraph("", ref_style)
        p2 = Paragraph(r2, ref_style) if r2 else Paragraph("", ref_style)
        ref_table_data.append([p1, p2])

    t_ref = Table(ref_table_data, colWidths=[250, 250])
    t_ref.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 0.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_ref)

    # Academic Writing Sample Notice
    story.append(Spacer(1, 3))
    note_box_data = [[Paragraph(
        "<b>Academic Writing Sample Notice:</b> This document represents an unpublished working paper prepared by the author as an academic research writing sample for Fall 2027 U.S. PhD applications in Computer Science / Statistics. It does not claim formal peer-reviewed conference or journal publication. All statistical estimates, confidence intervals, and hypothesis tests were derived from validated raw data artifacts without synthetic interpolation or simulated outcomes. In accordance with open-science principles, all findings distinguish measured empirical facts from statistical interpretation.",
        ParagraphStyle("Note", parent=body_style, fontSize=6.2, leading=8.0, textColor=colors.HexColor("#334155"))
    )]]
    t_note = Table(note_box_data, colWidths=[504])
    t_note.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_note)

    # Build Document
    print(f"Compiling Canonical Manuscript PDF to {OUT_PDF}...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated: {OUT_PDF}")


if __name__ == "__main__":
    build_canonical_manuscript_pdf()
