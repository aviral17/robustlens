"""Generate the publication-grade PDF for RobustLens Final Research Paper using ReportLab.

Target: reports/final_research_package/RobustLens_Final_Paper.pdf
Title: "Order Has Consequences: Total Effects and Fidelity-Conditioned Prediction Instability under Compound Image Corruptions"
Author: Avi
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

REPO_ROOT = Path("c:/STATISTICS & CV/robustlens")
ANALYSIS_DIR = REPO_ROOT / "reports/final_research_package/analysis"
FIG_DIR = REPO_ROOT / "reports/final_research_package/figures"
OUT_PDF = REPO_ROOT / "reports/final_research_package/RobustLens_Final_Paper.pdf"


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
            self.drawString(54, 752, "Order Has Consequences: Total Effects and Fidelity-Conditioned Prediction Instability")
            self.drawRightString(612 - 54, 752, "RobustLens Research Package")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 746, 612 - 54, 746)

        # Running footer on all pages
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 32, page_str)
        self.drawString(54, 32, "RobustLens Research Benchmark • Avi (Independent Research) • October 2026")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 42, 612 - 54, 42)
        self.restoreState()


def build_final_paper_pdf():
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=46,
        bottomMargin=48,
        title="Order Has Consequences: Total Effects and Fidelity-Conditioned Prediction Instability under Compound Image Corruptions",
        author="Avi",
        subject="Computer Vision Robustness, Multi-Stage Corruptions, Statistical Decomposition"
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
        leading=12,
        textColor=colors.HexColor("#334155"),
        alignment=TA_CENTER,
        spaceAfter=3
    )

    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#64748b"),
        alignment=TA_CENTER,
        spaceAfter=5
    )

    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.8,
        leading=12.5,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=6,
        spaceAfter=2,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.4,
        leading=10.2,
        textColor=colors.HexColor("#1e293b"),
        alignment=TA_JUSTIFY,
        spaceAfter=3.5
    )

    abstract_style = ParagraphStyle(
        "Abstract",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9.8,
        textColor=colors.HexColor("#1e293b"),
        alignment=TA_JUSTIFY,
        leftIndent=12,
        rightIndent=12,
        spaceAfter=5
    )

    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=6.7,
        leading=8.6,
        textColor=colors.HexColor("#475569"),
        alignment=TA_CENTER,
        spaceBefore=2,
        spaceAfter=3.5,
        keepWithNext=True
    )

    eq_style = ParagraphStyle(
        "Equation",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.0,
        leading=9.0,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_CENTER,
        spaceBefore=2,
        spaceAfter=2.5,
        keepWithNext=True
    )

    th_style = ParagraphStyle("TH", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=6.0, leading=7.5, textColor=colors.white, alignment=TA_CENTER)
    tc_style = ParagraphStyle("TC", parent=styles["Normal"], fontName="Helvetica", fontSize=5.8, leading=7.2, textColor=colors.HexColor("#1e293b"), alignment=TA_CENTER)
    tc_left = ParagraphStyle("TCL", parent=styles["Normal"], fontName="Helvetica", fontSize=5.8, leading=7.2, textColor=colors.HexColor("#1e293b"), alignment=TA_LEFT)
    tc_bold = ParagraphStyle("TCB", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=5.8, leading=7.2, textColor=colors.HexColor("#1e293b"), alignment=TA_CENTER)

    story = []

    # Title Block
    story.append(Paragraph("Order Has Consequences: Total Effects and Fidelity-Conditioned Prediction Instability under Compound Image Corruptions", title_style))
    story.append(Paragraph("A Paired Counterfactual Study of Non-Commutative Image Degradation in Vision Classifiers", subtitle_style))
    story.append(Paragraph("<b>Avi</b> &nbsp;|&nbsp; Independent Research &nbsp;|&nbsp; October 2026 &nbsp;|&nbsp; Code: github.com/aviral199/robustlens", meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0284c7"), spaceBefore=1, spaceAfter=4))

    # Abstract
    story.append(Paragraph("<b>Abstract</b>—Deep vision models are widely evaluated under synthetic corruptions, but standard benchmarks treat degradations as isolated unary operations or along fixed, unidirectional pipelines. In real-world imaging, sequential degradations involve non-commutative mathematical operators (<i>T<sub>A</sub> ∘ T<sub>B</sub> ≠ T<sub>B</sub> ∘ T<sub>A</sub></i>). We conduct an extensive paired evaluation of the full symmetric group of Gaussian Noise, Defocus Blur, and JPEG Compression (3! = 6 sequences × 5 severities) across four architectures (ResNet-50, ConvNeXt-Tiny, ViT-Base, Swin-Tiny; 973,400 total records across two regimes). We separate two distinct estimands: (1) Total operational effect under nominal parameters, where output fidelity varies downstream (up to 5.0 dB PSNR between terminal blur and terminal JPEG) and top-1 accuracy disparities reach 21.55% (ResNet) and 20.45% (ConvNeXt); and (2) Order differences at comparable fidelity (pre-calibrated common support), which contracts sequence gaps by 63.18% overall (down to 4.67% in ConvNeXt), while transparently documenting that prespecified tolerances were not achieved (PSNR RMSE = 0.943 dB, SSIM RMSE = 0.1380). Crucially, a 5-state joint prediction space decomposition proves that Predicted-Label Churn (34.74%–40.78%) is more than double Correctness Churn (12.34%–16.93%) because Error Migration (different wrong classes) averages ~23%. Permuting corruption order alters single-image failure topologies even when aggregate accuracy is invariant.", abstract_style))

    # Section 1: Introduction
    story.append(Paragraph("1. Introduction & Two Distinct Estimands", h1_style))
    story.append(Paragraph("Evaluating vision robustness under distribution shift is essential for reliable deployment. However, standard benchmarks treat corruptions as isolated, unary operations (ImageNet-C) or fixed pipelines (R-Bench). In real-world systems, degradations occur sequentially through non-commutative operators: spatial blur (linear low-pass filter), Gaussian noise (additive stochastic field), and JPEG (non-linear DCT quantization). When applying fixed nominal parameters across permutations, accuracy varies by up to 21.55% on identical physical images. Crucially, we formalize two distinct research estimands: (1) <i>Total Operational Effect:</i> Low-level fidelity changes are part of the downstream treatment pathway of operator non-commutativity, reflecting total real-world impact; and (2) <i>Fidelity-Conditioned Comparison:</i> Evaluating order sensitivity under approximate common-support fidelity. We investigate how much of the nominal gap contracts, whether residual order effects persist, and how sequence order reshuffles single-image predicted labels.", body_style))

    # Section 2: Mathematical Framework & Saturated Decomposition
    story.append(Paragraph("2. Mathematical Framework & Statistical Methodology", h1_style))
    story.append(Paragraph("We evaluate all six permutations <i>S</i> = {NBJ, NJB, BNJ, BJN, JNB, JBN} applied left-to-right. We specify a full-rank saturated 6-cell linear model <i>μ = X β</i> with reference Start Noise, End Noise. Design matrix <i>X</i> has columns [Intercept, Start Blur, Start JPEG, End Blur, End JPEG, Parity Contrast]. Because det(<i>X</i>) = -6, <i>X</i> is invertible (rank 6). The reference cell (Start Noise + End Noise) is physically unobserved in 3-operator permutations without repetition, so the intercept is an extrapolated baseline. Columns are non-orthogonal, so we report direct marginal contrasts alongside reference-coded parameters. The parity contrast <i>β</i><sub>5</sub> compares even (NBJ, BJN, JNB) versus odd (NJB, BNJ, JBN) permutations, balancing all three primitives equally at entry and exit.", body_style))
    story.append(Paragraph("<b>5-State Prediction Space & Churn Decomposition:</b> For image <i>i</i> with true class <i>Y</i> and predictions <i>Ŷ<sub>A</sub>, Ŷ<sub>B</sub></i>, we partition the joint space into: Both Correct (<i>E</i><sub>11</sub>), Loss (<i>E</i><sub>10</sub>), Gain (<i>E</i><sub>01</sub>), Same Wrong Class (<i>E</i><sub>00</sub><sup>same</sup>), and Error Migration (<i>E</i><sub>00</sub><sup>diff</sup>). Total probability is conserved: ∑ P(<i>E<sub>s</sub></i>) = 1. We establish: <i>Correctness Churn</i> = P(<i>E</i><sub>10</sub>) + P(<i>E</i><sub>01</sub>) = <i>L + G</i>; <i>Predicted-Label Churn</i> = P(<i>Ŷ<sub>A</sub> ≠ Ŷ<sub>B</sub></i>) = <i>Correctness Churn</i> + P(<i>E</i><sub>00</sub><sup>diff</sup>); and <i>Agreement Rate</i> = P(<i>E</i><sub>11</sub>) + P(<i>E</i><sub>00</sub><sup>same</sup>) = 1 - <i>Label Churn</i>. Inference uses clustered bootstrap (<i>B</i> = 5,000) on physical images (<i>N</i> = 3,925) with Holm-Bonferroni FWER control for primary accuracy tests and Benjamini-Hochberg FDR control for secondary metrics.", body_style))

    # Section 3: Regime 1 Nominal Order
    story.append(Paragraph("3. Regime 1: Nominal-Order Benchmark & The Distortion Confound", h1_style))
    story.append(Paragraph("Table 1 summarizes model accuracy under standard nominal ImageNet-C parameters across 471,000 compound records plus 15,700 clean baselines.", body_style))

    # Table 1: Nominal Accuracies
    t1_data = [
        [Paragraph("<b>Model</b>", th_style), Paragraph("<b>Arch</b>", th_style), Paragraph("<b>NBJ</b>", th_style), Paragraph("<b>NJB</b>", th_style), Paragraph("<b>BNJ</b>", th_style), Paragraph("<b>BJN</b>", th_style), Paragraph("<b>JNB</b>", th_style), Paragraph("<b>JBN</b>", th_style), Paragraph("<b>Mean</b>", th_style), Paragraph("<b>Gap (Δ)</b>", th_style)],
        [Paragraph("ResNet-50", tc_left), Paragraph("CNN", tc_style), Paragraph("53.44%", tc_style), Paragraph("56.88%", tc_style), Paragraph("37.30%", tc_style), Paragraph("35.71%", tc_style), Paragraph("<b>57.27%</b>", tc_style), Paragraph("35.79%", tc_style), Paragraph("46.06%", tc_style), Paragraph("<b>21.55%</b>", tc_bold)],
        [Paragraph("ConvNeXt-Tiny", tc_left), Paragraph("Modern CNN", tc_style), Paragraph("63.98%", tc_style), Paragraph("<b>66.44%</b>", tc_style), Paragraph("48.63%", tc_style), Paragraph("45.99%", tc_style), Paragraph("65.44%", tc_style), Paragraph("46.33%", tc_style), Paragraph("56.14%", tc_style), Paragraph("<b>20.45%</b>", tc_bold)],
        [Paragraph("ViT-Base/16", tc_left), Paragraph("Transformer", tc_style), Paragraph("58.39%", tc_style), Paragraph("<b>60.14%</b>", tc_style), Paragraph("39.77%", tc_style), Paragraph("41.64%", tc_style), Paragraph("59.19%", tc_style), Paragraph("42.23%", tc_style), Paragraph("50.23%", tc_style), Paragraph("<b>20.37%</b>", tc_bold)],
        [Paragraph("Swin-Tiny", tc_left), Paragraph("Hier. ViT", tc_style), Paragraph("59.81%", tc_style), Paragraph("61.78%", tc_style), Paragraph("45.81%", tc_style), Paragraph("50.24%", tc_style), Paragraph("<b>62.94%</b>", tc_style), Paragraph("50.63%", tc_style), Paragraph("55.20%", tc_style), Paragraph("<b>17.13%</b>", tc_bold)],
        [Paragraph("<b>Overall Mean</b>", tc_left), Paragraph("—", tc_style), Paragraph("58.90%", tc_bold), Paragraph("<b>61.31%</b>", tc_bold), Paragraph("42.88%", tc_bold), Paragraph("43.40%", tc_bold), Paragraph("61.21%", tc_bold), Paragraph("43.75%", tc_bold), Paragraph("51.91%", tc_bold), Paragraph("<b>18.43%</b>", tc_bold)],
    ]
    t1 = Table(t1_data, colWidths=[65, 52, 42, 42, 42, 42, 42, 42, 44, 49])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
    ]))
    story.append(KeepTogether([
        t1,
        Paragraph("<b>Table 1:</b> Model top-1 accuracy under nominal ImageNet-C parameters (<i>N</i> = 3,925 validation images, 471,000 compound records). Maximum cross-sequence gap reaches 21.55% in ResNet and 20.45% in ConvNeXt.", caption_style)
    ]))

    # Figure 1: Sequence Gap Contraction
    fig2_path = FIG_DIR / "fig2_nominal_vs_matched_gap_contraction.png"
    if fig2_path.exists():
        story.append(KeepTogether([
            Image(str(fig2_path), width=6.8 * inch, height=2.35 * inch),
            Paragraph("<b>Figure 1:</b> Cross-sequence accuracy gap contraction from Nominal-Order (Panel A, gaps 17.1%–21.6%) to Approximate-Fidelity (Panel B, gaps 4.7%–11.7%), demonstrating a 63.18% average reduction across architectures.", caption_style)
        ]))

    # Section 4: Regime 2 Approximate Fidelity
    story.append(Paragraph("4. Regime 2: Approximate-Fidelity Calibration & Gap Contraction", h1_style))
    story.append(Paragraph("Parameters were pre-calibrated on <i>N</i><sub>cal</sub> = 50 training images targeting common (PSNR, SSIM) support across 5 distortion bins. As documented in Table 2, prespecified tolerances (±0.75 dB PSNR, ±0.035 SSIM) were <b>not met</b> over the tested discrete grid (PSNR RMSE = 0.943 dB, SSIM RMSE = 0.1380). This establishes the regime as an approximate-fidelity descriptive comparison, not exact matching.", body_style))

    # Table 2: Calibration Diagnostics
    t2_data = [
        [Paragraph("<b>Target Bin</b>", th_style), Paragraph("<b>Target PSNR</b>", th_style), Paragraph("<b>Target SSIM</b>", th_style), Paragraph("<b>Max |PSNR Res|</b>", th_style), Paragraph("<b>PSNR Tol (≤0.75 dB)</b>", th_style), Paragraph("<b>Max |SSIM Res|</b>", th_style), Paragraph("<b>SSIM Tol (≤0.035)</b>", th_style)],
        [Paragraph("Target 1 (Mild)", tc_left), Paragraph("28.0 dB", tc_style), Paragraph("0.850", tc_style), Paragraph("1.117 dB (`NJB`)", tc_style), Paragraph("Failed", tc_bold), Paragraph("0.0686 (`BJN`)", tc_style), Paragraph("Failed", tc_bold)],
        [Paragraph("Target 2 (Mod-Mild)", tc_left), Paragraph("24.5 dB", tc_style), Paragraph("0.740", tc_style), Paragraph("0.936 dB (`BJN`)", tc_style), Paragraph("Failed", tc_bold), Paragraph("0.1758 (`BJN`)", tc_style), Paragraph("Failed", tc_bold)],
        [Paragraph("Target 3 (Moderate)", tc_left), Paragraph("21.5 dB", tc_style), Paragraph("0.620", tc_style), Paragraph("1.260 dB (`NBJ`)", tc_style), Paragraph("Failed", tc_bold), Paragraph("0.2379 (`BJN`)", tc_style), Paragraph("Failed", tc_bold)],
        [Paragraph("Target 4 (Mod-Severe)", tc_left), Paragraph("18.5 dB", tc_style), Paragraph("0.480", tc_style), Paragraph("1.989 dB (`NBJ`)", tc_style), Paragraph("Failed", tc_bold), Paragraph("0.2373 (`BJN`)", tc_style), Paragraph("Failed", tc_bold)],
        [Paragraph("Target 5 (Severe)", tc_left), Paragraph("16.0 dB", tc_style), Paragraph("0.360", tc_style), Paragraph("2.639 dB (`NBJ`)", tc_style), Paragraph("Failed", tc_bold), Paragraph("0.2067 (`BJN`)", tc_style), Paragraph("Failed", tc_bold)],
        [Paragraph("<b>Overall Fit</b>", tc_left), Paragraph("—", tc_style), Paragraph("—", tc_style), Paragraph("<b>RMSE: 0.943 dB</b>", tc_bold), Paragraph("<b>Failed</b>", tc_bold), Paragraph("<b>RMSE: 0.1380</b>", tc_bold), Paragraph("<b>Failed</b>", tc_bold)],
    ]
    t2 = Table(t2_data, colWidths=[90, 65, 60, 85, 80, 85, 75])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fef2f2")]),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    story.append(KeepTogether([
        t2,
        Paragraph("<b>Table 2:</b> Measured calibration residuals on <i>N</i><sub>cal</sub> = 50 training images. Both prespecified tolerances were exceeded, establishing this study as an approximate-fidelity comparison rather than exact matching.", caption_style)
    ]))

    # Figure 2: Calibration Residuals
    fig1_path = FIG_DIR / "fig1_calibration_residuals_and_tolerances.png"
    if fig1_path.exists():
        story.append(KeepTogether([
            Image(str(fig1_path), width=6.8 * inch, height=2.35 * inch),
            Paragraph("<b>Figure 2:</b> Calibration residual diagnostics across target distortion bins. Shaded red bands denote prespecified tolerances (±0.75 dB PSNR, ±0.035 SSIM). All targets exhibit points outside tolerance.", caption_style)
        ]))

    # Table 3: Gap Contraction
    story.append(Paragraph("Despite residual calibration spreads, low-level distortion balancing contracts cross-sequence accuracy gaps dramatically (Table 3): ConvNeXt contracts by 77.15% (gap drops to 4.67%), ResNet by 72.62% (5.90%), ViT by 50.75% (10.03%), and Swin by 31.80% (11.68%). Overall gap contracts from 18.43% down to 6.79% (63.18% reduction).", body_style))

    t3_data = [
        [Paragraph("<b>Model</b>", th_style), Paragraph("<b>NBJ</b>", th_style), Paragraph("<b>NJB</b>", th_style), Paragraph("<b>BNJ</b>", th_style), Paragraph("<b>BJN</b>", th_style), Paragraph("<b>JNB</b>", th_style), Paragraph("<b>JBN</b>", th_style), Paragraph("<b>Matched Mean</b>", th_style), Paragraph("<b>Matched Gap</b>", th_style), Paragraph("<b>Nominal Gap</b>", th_style), Paragraph("<b>Contraction</b>", th_style)],
        [Paragraph("ResNet-50", tc_left), Paragraph("50.15%", tc_style), Paragraph("46.76%", tc_style), Paragraph("45.18%", tc_style), Paragraph("50.63%", tc_style), Paragraph("47.05%", tc_style), Paragraph("<b>51.08%</b>", tc_style), Paragraph("48.47%", tc_style), Paragraph("<b>5.90%</b>", tc_bold), Paragraph("21.55%", tc_style), Paragraph("<b>72.62%</b>", tc_bold)],
        [Paragraph("ConvNeXt-Tiny", tc_left), Paragraph("<b>60.65%</b>", tc_style), Paragraph("56.46%", tc_style), Paragraph("55.97%", tc_style), Paragraph("59.22%", tc_style), Paragraph("56.47%", tc_style), Paragraph("59.86%", tc_style), Paragraph("58.11%", tc_style), Paragraph("<b>4.67%</b>", tc_bold), Paragraph("20.45%", tc_style), Paragraph("<b>77.15%</b>", tc_bold)],
        [Paragraph("ViT-Base/16", tc_left), Paragraph("<b>56.54%</b>", tc_style), Paragraph("53.39%", tc_style), Paragraph("46.51%", tc_style), Paragraph("54.72%", tc_style), Paragraph("53.01%", tc_style), Paragraph("55.68%", tc_style), Paragraph("53.31%", tc_style), Paragraph("<b>10.03%</b>", tc_bold), Paragraph("20.37%", tc_style), Paragraph("<b>50.75%</b>", tc_bold)],
        [Paragraph("Swin-Tiny", tc_left), Paragraph("57.61%", tc_style), Paragraph("50.44%", tc_style), Paragraph("53.93%", tc_style), Paragraph("61.23%", tc_style), Paragraph("52.31%", tc_style), Paragraph("<b>62.12%</b>", tc_style), Paragraph("56.27%", tc_style), Paragraph("<b>11.68%</b>", tc_bold), Paragraph("17.13%", tc_style), Paragraph("<b>31.80%</b>", tc_bold)],
        [Paragraph("<b>Overall Mean</b>", tc_left), Paragraph("56.23%", tc_bold), Paragraph("51.76%", tc_bold), Paragraph("50.40%", tc_bold), Paragraph("56.45%", tc_bold), Paragraph("52.21%", tc_bold), Paragraph("<b>57.18%</b>", tc_bold), Paragraph("54.04%", tc_bold), Paragraph("<b>6.79%</b>", tc_bold), Paragraph("18.43%", tc_bold), Paragraph("<b>63.18%</b>", tc_bold)],
    ]
    t3 = Table(t3_data, colWidths=[65, 42, 42, 42, 42, 42, 42, 52, 50, 48, 51])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#065f46")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0fdf4")]),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
    ]))
    story.append(KeepTogether([
        t3,
        Paragraph("<b>Table 3:</b> Model accuracy under approximate-fidelity calibration. Equalizing low-level distortion contracts sequence gaps by 72.62% to 77.15% in CNNs and 31.80% to 50.75% in Vision Transformers.", caption_style)
    ]))

    # Section 5: Saturated Linear Decomposition
    story.append(Paragraph("5. Full-Rank Saturated Linear Decomposition Analysis", h1_style))
    story.append(Paragraph("Table 4 compares fitted parameters across regimes (<i>B</i> = 5,000 bootstrap resamples, Holm FWER-controlled). In nominal evaluations, terminal blur confers a +12.27% boost due to artifact attenuation. Under calibrated fidelity, the End Blur effect reverses to <b>-6.82%</b>: when PSNR is balanced, terminal blur strips high-frequency discriminative features. The balanced parity contrast remains significant (+0.93%, <i>p</i> &lt; 10<sup>-100</sup>), confirming that permutation parity independently influences classification accuracy.", body_style))

    t4_data = [
        [Paragraph("<b>Parameter</b>", th_style), Paragraph("<b>Nominal Estimate</b>", th_style), Paragraph("<b>Nominal 95% Clustered CI</b>", th_style), Paragraph("<b>Nominal <i>p</i><sub>Holm</sub></b>", th_style), Paragraph("<b>Matched Estimate</b>", th_style), Paragraph("<b>Matched 95% Clustered CI</b>", th_style), Paragraph("<b>Matched <i>p</i><sub>Holm</sub></b>", th_style)],
        [Paragraph("Intercept (β<sub>0</sub>)", tc_left), Paragraph("+51.63%", tc_style), Paragraph("[+50.71%, +52.52%]", tc_style), Paragraph("&lt; 10<sup>-100</sup>", tc_bold), Paragraph("+59.51%", tc_style), Paragraph("[+58.51%, +60.51%]", tc_style), Paragraph("&lt; 10<sup>-100</sup>", tc_bold)],
        [Paragraph("Start Blur (β<sub>1</sub>)", tc_left), Paragraph("-10.83%", tc_style), Paragraph("[-11.14%, -10.53%]", tc_style), Paragraph("&lt; 10<sup>-100</sup>", tc_bold), Paragraph("-3.99%", tc_style), Paragraph("[-4.22%, -3.75%]", tc_style), Paragraph("&lt; 10<sup>-100</sup>", tc_bold)],
        [Paragraph("Start JPEG (β<sub>2</sub>)", tc_left), Paragraph("-5.29%", tc_style), Paragraph("[-5.50%, -5.08%]", tc_style), Paragraph("&lt; 10<sup>-100</sup>", tc_bold), Paragraph("-1.40%", tc_style), Paragraph("[-1.58%, -1.23%]", tc_style), Paragraph("&lt; 10<sup>-50</sup>", tc_bold)],
        [Paragraph("End Blur (β<sub>3</sub>)", tc_left), Paragraph("<b>+12.27%</b>", tc_bold), Paragraph("[+11.88%, +12.67%]", tc_style), Paragraph("&lt; 10<sup>-100</sup>", tc_bold), Paragraph("<b>-6.82%</b>", tc_bold), Paragraph("[-7.18%, -6.46%]", tc_style), Paragraph("&lt; 10<sup>-100</sup>", tc_bold)],
        [Paragraph("End JPEG (β<sub>4</sub>)", tc_left), Paragraph("<b>+4.67%</b>", tc_bold), Paragraph("[+4.35%, +5.00%]", tc_style), Paragraph("&lt; 10<sup>-100</sup>", tc_bold), Paragraph("<b>-4.20%</b>", tc_bold), Paragraph("[-4.50%, -3.91%]", tc_style), Paragraph("&lt; 10<sup>-100</sup>", tc_bold)],
        [Paragraph("Parity Contrast (β<sub>5</sub>)", tc_left), Paragraph("<b>+2.60%</b>", tc_bold), Paragraph("[+2.52%, +2.67%]", tc_style), Paragraph("&lt; 10<sup>-100</sup>", tc_bold), Paragraph("<b>+0.93%</b>", tc_bold), Paragraph("[+0.86%, +0.99%]", tc_style), Paragraph("&lt; 10<sup>-100</sup>", tc_bold)],
    ]
    t4 = Table(t4_data, colWidths=[95, 65, 88, 55, 65, 88, 55])
    t4.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e1b4b")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f3ff")]),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
    ]))
    story.append(KeepTogether([
        t4,
        Paragraph("<b>Table 4:</b> Saturated linear decomposition coefficients for Across-Models Overall mean (<i>B</i> = 5,000 bootstrap replicates, Holm-Bonferroni FWER-adjusted). End operator effect reverses upon distortion calibration.", caption_style)
    ]))

    # Figure 3: Decomposition Forest Plot
    fig3_path = FIG_DIR / "fig3_saturated_decomposition_coefficients.png"
    if fig3_path.exists():
        story.append(KeepTogether([
            Image(str(fig3_path), width=6.8 * inch, height=2.35 * inch),
            Paragraph("<b>Figure 3:</b> Forest plot of decomposition parameters across models (95% clustered bootstrap CIs). Panel A shows nominal dominance of terminal blur, while Panel B reveals negative terminal effects under common support.", caption_style)
        ]))

    # Section 6: Prediction Instability & Error Migration
    story.append(Paragraph("6. Prediction Instability, Error Migration & Per-Class Failure Topologies", h1_style))
    story.append(Paragraph("A central finding of RobustLens is that <b>aggregate accuracy obscures massive per-image prediction churn and error migration.</b> Table 5 reports the complete 5-state joint prediction space decomposition across models. Error Migration (<i>E</i><sub>00</sub><sup>diff</sup>, where models predict different wrong classes) averages 23.85% in nominal and 22.40% under approximate fidelity. Consequently, Predicted-Label Churn (34.74%–40.78%) is more than double Correctness Churn (12.34%–16.93%). Permuting early operators alters specific class failure sets even when aggregate benchmark scores appear invariant.", body_style))

    # Table 5: 5-State Transitions Table
    t5_data = [
        [Paragraph("<b>Regime</b>", th_style), Paragraph("<b>Model</b>", th_style), Paragraph("<b>Both Correct</b>", th_style), Paragraph("<b>Loss (E10)</b>", th_style), Paragraph("<b>Gain (E01)</b>", th_style), Paragraph("<b>Same Wrong</b>", th_style), Paragraph("<b>Error Mig</b>", th_style), Paragraph("<b>Corr Churn</b>", th_style), Paragraph("<b>Label Churn</b>", th_style), Paragraph("<b>Agreement</b>", th_style)],
        [Paragraph("Nominal", tc_style), Paragraph("ConvNeXt", tc_left), Paragraph("47.63%", tc_style), Paragraph("11.64%", tc_style), Paragraph("5.38%", tc_style), Paragraph("14.99%", tc_style), Paragraph("<b>20.36%</b>", tc_bold), Paragraph("17.02%", tc_style), Paragraph("<b>37.38%</b>", tc_bold), Paragraph("62.62%", tc_style)],
        [Paragraph("Nominal", tc_style), Paragraph("ResNet-50", tc_left), Paragraph("36.96%", tc_style), Paragraph("12.06%", tc_style), Paragraph("6.15%", tc_style), Paragraph("14.90%", tc_style), Paragraph("<b>29.93%</b>", tc_bold), Paragraph("18.20%", tc_style), Paragraph("<b>48.14%</b>", tc_bold), Paragraph("51.86%", tc_style)],
        [Paragraph("Nominal", tc_style), Paragraph("Swin-Tiny", tc_left), Paragraph("47.98%", tc_style), Paragraph("8.49%", tc_style), Paragraph("5.96%", tc_style), Paragraph("16.06%", tc_style), Paragraph("<b>21.51%</b>", tc_bold), Paragraph("14.45%", tc_style), Paragraph("<b>35.96%</b>", tc_bold), Paragraph("64.04%", tc_style)],
        [Paragraph("Nominal", tc_style), Paragraph("ViT-Base", tc_left), Paragraph("41.20%", tc_style), Paragraph("11.76%", tc_style), Paragraph("6.31%", tc_style), Paragraph("17.17%", tc_style), Paragraph("<b>23.57%</b>", tc_bold), Paragraph("18.06%", tc_style), Paragraph("<b>41.63%</b>", tc_bold), Paragraph("58.37%", tc_style)],
        [Paragraph("Nominal", tc_style), Paragraph("<b>Overall</b>", tc_left), Paragraph("43.44%", tc_bold), Paragraph("10.99%", tc_bold), Paragraph("5.95%", tc_bold), Paragraph("15.78%", tc_bold), Paragraph("<b>23.85%</b>", tc_bold), Paragraph("16.93%", tc_bold), Paragraph("<b>40.78%</b>", tc_bold), Paragraph("59.22%", tc_bold)],
        [Paragraph("Matched", tc_style), Paragraph("ConvNeXt", tc_left), Paragraph("52.64%", tc_style), Paragraph("5.49%", tc_style), Paragraph("5.44%", tc_style), Paragraph("17.44%", tc_style), Paragraph("<b>19.00%</b>", tc_bold), Paragraph("10.93%", tc_style), Paragraph("<b>29.92%</b>", tc_bold), Paragraph("70.08%", tc_style)],
        [Paragraph("Matched", tc_style), Paragraph("ResNet-50", tc_left), Paragraph("41.92%", tc_style), Paragraph("6.18%", tc_style), Paragraph("6.92%", tc_style), Paragraph("15.54%", tc_style), Paragraph("<b>29.44%</b>", tc_bold), Paragraph("13.10%", tc_style), Paragraph("<b>42.54%</b>", tc_bold), Paragraph("57.46%", tc_style)],
        [Paragraph("Matched", tc_style), Paragraph("Swin-Tiny", tc_left), Paragraph("50.31%", tc_style), Paragraph("4.78%", tc_style), Paragraph("7.14%", tc_style), Paragraph("18.03%", tc_style), Paragraph("<b>19.74%</b>", tc_bold), Paragraph("11.92%", tc_style), Paragraph("<b>31.66%</b>", tc_bold), Paragraph("68.34%", tc_style)],
        [Paragraph("Matched", tc_style), Paragraph("ViT-Base", tc_left), Paragraph("46.61%", tc_style), Paragraph("6.61%", tc_style), Paragraph("6.79%", tc_style), Paragraph("18.55%", tc_style), Paragraph("<b>21.44%</b>", tc_bold), Paragraph("13.40%", tc_style), Paragraph("<b>34.85%</b>", tc_bold), Paragraph("65.15%", tc_style)],
        [Paragraph("Matched", tc_style), Paragraph("<b>Overall</b>", tc_left), Paragraph("47.87%", tc_bold), Paragraph("5.76%", tc_bold), Paragraph("6.57%", tc_bold), Paragraph("17.39%", tc_bold), Paragraph("<b>22.40%</b>", tc_bold), Paragraph("12.34%", tc_bold), Paragraph("<b>34.74%</b>", tc_bold), Paragraph("65.26%", tc_bold)],
    ]
    t5 = Table(t5_data, colWidths=[42, 52, 48, 44, 44, 48, 52, 48, 54, 52])
    t5.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7c2d12")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, 5), [colors.white, colors.HexColor("#fff7ed")]),
        ("ROWBACKGROUNDS", (0, 6), (-1, -1), [colors.white, colors.HexColor("#fef2f2")]),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
    ]))
    story.append(KeepTogether([
        t5,
        Paragraph("<b>Table 5:</b> 5-State joint prediction space transitions (<i>N</i> = 19,625 image-severity pairs per comparison). Error Migration (<i>E</i><sub>00</sub><sup>diff</sup>) averages 22.4%–23.8%, causing Predicted-Label Churn to more than double Correctness Churn.", caption_style)
    ]))

    # Figure 4: Churn Decoupling
    fig4_path = FIG_DIR / "fig4_paired_error_churn_decomposition.png"
    if fig4_path.exists():
        story.append(KeepTogether([
            Image(str(fig4_path), width=6.8 * inch, height=2.10 * inch),
            Paragraph("<b>Figure 4:</b> Decoupling of net accuracy difference |ΔAcc| from excess error churn across regimes. Points along the vertical axis represent sequence pairs with near-zero accuracy gaps but high churn.", caption_style)
        ]))

    # Section 7: Calibration & Uncertainty
    story.append(Paragraph("7. Calibration & Per-Class Failure Topologies", h1_style))
    story.append(Paragraph("Expected Calibration Error (ECE, 10 equal-width bins) reveals systematic overconfidence under compound shifts. Under Target 5, ResNet-50 accuracy collapses to 31.2%, yet confidence remains at 54.8% (ECE &gt; 22%). Per-class breakdown across the 10 Imagenette classes exposes stark vulnerabilities: for 'church', accuracy drops to 18.35% with 50.45% label churn and 39.41% error migration (nearly 4× its correctness churn of 11.04%), whereas 'tench' and 'golf ball' achieve &gt;72% accuracy with only ~11% error migration.", body_style))

    fig5_path = FIG_DIR / "fig5_calibration_ece_and_confidence.png"
    if fig5_path.exists():
        story.append(KeepTogether([
            Image(str(fig5_path), width=6.8 * inch, height=2.10 * inch),
            Paragraph("<b>Figure 5:</b> Confidence vs. accuracy calibration trajectories across sequences. All architectures demonstrate systematic overconfidence under severe distortion, with Swin-Tiny exhibiting heightened calibration dispersion.", caption_style)
        ]))

    # Section 8: Limitations & Scope Boundaries
    story.append(Paragraph("8. Limitations & Scope Boundaries", h1_style))
    story.append(Paragraph("We explicitly emphasize the following boundaries: (1) <i>Approximate Fidelity:</i> Prespecified tolerances (±0.75 dB, ±0.035 SSIM) were not achieved over the tested grid. Residual PSNR spread reaches 3.10 dB at Target 5. (2) <i>Finite Grid Limitation:</i> Failure of our discrete search does not prove universal physical impossibility across continuous parameter spaces. (3) <i>Dataset Scope:</i> Experiments utilized FastAI Imagenette-v2 (10 classes, 3,925 validation images); evaluations on full ImageNet-1k remain necessary for global benchmark validation. (4) <i>Task Scope:</i> The study evaluates classification models only; dense prediction tasks were not evaluated. (5) <i>External ImageNet-V2 Validation:</i> ImageNet-V2 was not pre-cached locally; re-running full GPU inferences was excluded to maintain bounded execution and audit integrity.", body_style))

    # Section 9: Conclusion
    story.append(Paragraph("9. Conclusion", h1_style))
    story.append(Paragraph("RobustLens demonstrates that nominal compound-corruption benchmarks confound operator sequence order with physical distortion imbalance. Equalizing distortion on common support contracts cross-sequence gaps by 63.18% overall (and up to 77.15% in ConvNeXt). However, paired single-image analysis proves that sequence order reshuffles classifier failure sets, generating excess error churn of up to 10.97% and label churn of 34%–40% driven by error migration (~23%). Robustness benchmarks must account for both operator non-commutativity and physical distortion balancing.", body_style))

    # Clean PageBreak for Page 5 (Intentional Final Page)
    story.append(PageBreak())

    # Section 10: Research Artifacts & Open Science
    story.append(Paragraph("10. Research Artifacts, Open Science & Code Availability", h1_style))
    story.append(Paragraph("The complete research package, raw prediction artifacts, analysis scripts, and Next.js interactive tools are maintained in the local repository under <code>c:\\STATISTICS & CV\\robustlens</code> (prepared for public release at <font color='#0284c7'>https://github.com/aviral199/robustlens</font>). All primary data tables and reproducible scripts are packaged locally within <code>reports/final_research_package/</code>:", body_style))
    bullet_style = ParagraphStyle("Bullet", parent=body_style, alignment=TA_LEFT, leftIndent=12, firstLineIndent=-8, spaceAfter=2.5)
    story.append(Paragraph("• <b>Primary Codebase & Pipeline:</b> <font color='#0284c7'>https://github.com/aviral199/robustlens</font> (Local Astral <code>uv</code> repository).", bullet_style))
    story.append(Paragraph("• <b>Interactive Next.js Dashboard:</b> Features single-image shift exploration, side-by-side original vs. corrupted views, and live error churn matrices under both nominal and approximate-fidelity regimes.", bullet_style))
    story.append(Paragraph("• <b>Validated Raw Data Artifacts:</b> Full nominal evaluation records (486,700 rows, <code>results/order_effect_study/full/outputs/all_order_predictions.csv.gz</code>) and approximate-fidelity records (486,700 rows, <code>results/order_effect_study/matched_fidelity/outputs/all_order_predictions.csv.gz</code>) with verified SHA-256 manifests.", bullet_style))
    story.append(Paragraph("• <b>Technical Supplement:</b> Mathematical proofs of full-rank 6-cell model, 5-state prediction space, analytic inverse matrices, calibration equations, and vectorized NumPy bootstrap algorithms are cataloged in <code>reports/final_research_package/Technical_Supplement.md</code>.", bullet_style))

    # Section 11: Acknowledgments & Computational Infrastructure
    story.append(Paragraph("11. Acknowledgments & Computational Infrastructure", h1_style))
    story.append(Paragraph("The author gratefully acknowledges the open-source maintenance teams behind PyTorch, timm (PyTorch Image Models by Ross Wightman), ReportLab, and FastAI Imagenette. Computational evaluations comprising 973,400 total forward passes were conducted using NVIDIA Tesla T4 GPU instances on Kaggle Kernels (Kernel: <code>aviral199/robustlens-order-effects-full-benchmark</code>, Versions 2 & 4). Clustered statistical inference, saturated linear decompositions, and figure generation were executed locally via Astral <code>uv</code>.", body_style))

    # References as clean two-column table
    story.append(Paragraph("References", h1_style))
    ref_style = ParagraphStyle("Ref", parent=body_style, alignment=TA_LEFT, fontSize=6.2, leading=8.4, spaceAfter=2)
    refs_col1 = [
        "1. Bhojanapalli, S., et al. (2021). Understanding robustness of transformers. <i>ICCV</i>.",
        "2. Cormier, Q., Milani Fard, M., Canini, K., & Gupta, M. R. (2016). Launch and iterate: Reducing prediction churn. <i>NeurIPS</i>, 29:3179-3187.",
        "3. D'Amour, A., et al. (2020). Underspecification presents challenges for credibility in ML. <i>JMLR</i>.",
        "4. Dodge, S., & Karam, L. (2016). Understanding how image quality affects DNNs. <i>QoMEX</i>.",
        "5. Dosovitskiy, A., et al. (2021). An image is worth 16x16 words: Transformers at scale. <i>ICLR</i>.",
        "6. Erkara, A. O., & Mayer, M. (2026). Blind confusion of classification networks. <i>Neurocomputing</i>, 686:133678.",
        "7. Goyal, B., et al. (2022). Robust scene inference under noise-blur dual corruptions. <i>ICCP</i>.",
        "8. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. <i>CVPR</i>.",
        "9. Hendrycks, D., & Dietterich, T. (2019). Benchmarking robustness to common corruptions. <i>ICLR</i>.",
    ]
    refs_col2 = [
        "10. Hendrycks, D., et al. (2019). Natural adversarial examples. <i>CVPR</i>.",
        "11. Kang, D., et al. (2019). Testing robustness to compound corruptions. <i>NeurIPS Workshop</i>.",
        "12. Li, Y., et al. (2025). R-Bench: Benchmarking multi-stage physical degradation in MLLMs. <i>IEEE JSTSP</i>.",
        "13. Liao, X., Li, K., Zhu, X., & Liu, K. J. R. (2020). Robust detection of image operator chain. <i>IEEE JSTSP</i>, 14(5):955-968.",
        "14. Liu, Z., et al. (2021). Swin transformer: Hierarchical ViT using shifted windows. <i>ICCV</i>.",
        "15. Liu, Z., et al. (2022). A convnet for the 2020s. <i>CVPR</i>.",
        "16. Mintun, E., et al. (2021). On interaction between augmentations and corruptions. <i>NeurIPS</i>, 34:3571-3583.",
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
        ("TOPPADDING", (0, 0), (-1, -1), 1.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_ref)

    # Academic & Research Positioning Statement
    story.append(Spacer(1, 8))
    note_box_data = [[Paragraph("<b>Academic Audit & Research Scope Note:</b> This manuscript represents an independent research investigation prepared for Fall 2027 PhD applications in Computer Science / Statistics and subsequent external peer review. All statistical estimates, confidence intervals, and hypothesis tests were derived from validated raw data artifacts without synthetic interpolation or simulated outcomes. In accordance with open-science principles, all findings distinguish measured empirical facts from statistical interpretation.", ParagraphStyle("Note", parent=body_style, fontSize=6.5, leading=8.5, textColor=colors.HexColor("#334155")))]]
    t_note = Table(note_box_data, colWidths=[504])
    t_note.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_note)

    # Build Document
    print(f"Compiling PDF to {OUT_PDF}...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated: {OUT_PDF}")


if __name__ == "__main__":
    build_final_paper_pdf()
