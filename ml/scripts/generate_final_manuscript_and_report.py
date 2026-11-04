"""Generate polished final research report (PDF) and academic manuscript (DOCX).

Incorporates all 8 confirmatory audit corrections:
1. Explicit pretraining disclosure & sample quarantine distinction.
2. Corrected ImageNet-V2 external replication (only ViT-Base significant; raw & Holm-adjusted p-values).
3. Calibration matched PSNR/SSIM; LPIPS included as regression predictor.
4. Sobel edge energy rephrased as strongest predictive contributor with collinearity caveats.
5. Formal bootstrap interaction test for ViT vs CNN/Swin local-SSIM coefficients.
6. 3-archetype architectural taxonomy (CNNs vs Hierarchical Swin vs Global ViT).
7. Upper-bound notation for bootstrap resolution limits (p <= 0.0004).
8. Academic references, limitations, authors, and reproducibility registry.
"""

from __future__ import annotations

import json
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = BASE_DIR / "reports" / "final_research_package"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PDF_OUT = REPORTS_DIR / "RobustLens_Final_Research_Report.pdf"
DOCX_OUT = REPORTS_DIR / "RobustLens_Academic_Manuscript.docx"


# ==============================================================================
# 1. DOCX GENERATOR
# ==============================================================================

def set_cell_background(cell, fill_hex: str):
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def add_callout(doc: Document, title: str, text: str, border_color="0284C7", bg_color="F0F9FF"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.5)
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    # Left border only
    borders_xml = f'''
    <w:tcBorders {nsdecls("w")}>
        <w:top w:val="none"/>
        <w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>
        <w:bottom w:val="none"/>
        <w:right w:val="none"/>
    </w:tcBorders>
    '''
    cell._tc.get_or_add_tcPr().append(parse_xml(borders_xml))

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    run_t = p.add_run(title)
    run_t.bold = True
    run_t.font.name = "Calibri"
    run_t.font.size = Pt(11)
    run_t.font.color.rgb = RGBColor(2, 132, 199)

    p2 = cell.add_paragraph()
    p2.paragraph_format.space_before = Pt(2)
    p2.paragraph_format.space_after = Pt(0)
    p2.paragraph_format.line_spacing = 1.15
    run_body = p2.add_run(text)
    run_body.font.name = "Calibri"
    run_body.font.size = Pt(10)
    run_body.font.color.rgb = RGBColor(30, 41, 59)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)


def format_table(table, col_widths, headers, data, align_right_cols=None):
    if align_right_cols is None:
        align_right_cols = []
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    # Header Row
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].width = col_widths[i]
        set_cell_background(hdr_cells[i], "1E293B")
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=120, right=120)
        p = hdr_cells[i].paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if i in align_right_cols else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h)
        r.bold = True
        r.font.name = "Calibri"
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    # Data Rows
    for row_idx, row_data in enumerate(data):
        row_cells = table.add_row().cells
        bg_col = "F8FAFC" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, cell_value in enumerate(row_data):
            row_cells[col_idx].width = col_widths[col_idx]
            set_cell_background(row_cells[col_idx], bg_col)
            set_cell_margins(row_cells[col_idx], top=80, bottom=80, left=120, right=120)
            p = row_cells[col_idx].paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if col_idx in align_right_cols else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(str(cell_value))
            r.font.name = "Calibri"
            r.font.size = Pt(9)
            r.font.color.rgb = RGBColor(51, 65, 85)

    # Table Borders
    tblBorders_xml = f'''
    <w:tblBorders {nsdecls("w")}>
        <w:top w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
        <w:left w:val="none"/>
        <w:bottom w:val="single" w:sz="12" w:space="0" w:color="94A3B8"/>
        <w:right w:val="none"/>
        <w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>
        <w:insideV w:val="none"/>
    </w:tblBorders>
    '''
    table._tbl.tblPr.append(parse_xml(tblBorders_xml))


def build_academic_docx():
    print("Building Editable Academic Manuscript DOCX...")
    doc = Document()

    # Set Margins
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    # Title
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(6)
    p_title.paragraph_format.line_spacing = 1.15
    run_t = p_title.add_run("At Matched Image Fidelity, Does Corruption Order Still Matter?\nSpatial-Frequency and Structural Fingerprints of Vision Classifier Reliability")
    run_t.font.name = "Calibri"
    run_t.font.size = Pt(22)
    run_t.bold = True
    run_t.font.color.rgb = RGBColor(15, 23, 42)

    # Authors
    p_auth = doc.add_paragraph()
    p_auth.paragraph_format.space_before = Pt(4)
    p_auth.paragraph_format.space_after = Pt(2)
    r_a = p_auth.add_run("Aviral Choudhary")
    r_a.font.name = "Calibri"
    r_a.font.size = Pt(13)
    r_a.bold = True
    r_a.font.color.rgb = RGBColor(30, 41, 59)

    p_aff = doc.add_paragraph()
    p_aff.paragraph_format.space_before = Pt(0)
    p_aff.paragraph_format.space_after = Pt(16)
    r_aff = p_aff.add_run("RobustLens Research Initiative\nManuscript Draft • November 2026")
    r_aff.font.name = "Calibri"
    r_aff.font.size = Pt(10)
    r_aff.font.color.rgb = RGBColor(100, 116, 139)

    # Abstract Box
    add_callout(
        doc,
        "Abstract",
        "Deep neural networks for computer vision are known to suffer severe performance degradation under synthetic corruptions, but standard benchmarks (e.g., ImageNet-C) evaluate corruptions strictly as isolated, unary operations. In real-world imaging pipelines, corruptions occur sequentially through non-commutative operations. In this study, we investigate whether the order of multi-stage corruptions affects classifier decision margins when low-level physical fidelity is matched. Evaluating 3,825 confirmatory images across four canonical vision architectures (ResNet-50, ConvNeXt-Tiny, ViT-Base/16, Swin-Tiny) through 290,700 model forward passes, we demonstrate that spatial-frequency and structural distortion fingerprints explain statistically significant additional out-of-sample logit margin variance beyond global fidelity metrics (PSNR, SSIM, LPIPS) across all four architectures (p_adj <= 0.0004 under Holm-Bonferroni control). In an independent external replication on ImageNet-V2 (N=300, 22,800 passes), ViT-Base demonstrates statistically significant external variance explanation (Delta R2 = +0.0371, 95% CI: [+0.0065, +0.0675], p_adj = 0.0340), whereas for CNNs and Swin-Tiny, external gains remain positive but do not reach significance after multiplicity adjustment. Feature ablations demonstrate that Sobel edge energy is the strongest predictive contributor in the model, while formal bootstrap interaction tests reveal that local SSIM spatial heterogeneity associates significantly with margin degradation in ViT-Base compared to ConvNeXt-Tiny (p=0.00010) and Swin-Tiny (p=0.00300), but not ResNet-50 (p=0.61174)."
    )

    # 1. Introduction
    p_h1 = doc.add_heading("1. Introduction & Theoretical Motivation", level=1)
    p_h1.paragraph_format.space_before = Pt(14)
    p_h1.paragraph_format.space_after = Pt(6)

    doc.add_paragraph(
        "Evaluating the robustness of computer vision architectures under distribution shift is essential for reliable deployment in safety-critical applications. Since the introduction of ImageNet-C (Hendrycks & Dietterich, 2019), benchmarks have focused primarily on isolated, single-step perturbations. However, realistic imaging workflows encompass compound physical and digital corruptions: optical scattering (defocus blur B), sensor readout degradation (Gaussian noise N), and compression artifacts (JPEG compression J). Crucially, the mathematical operators modeling these degradations are non-commutative (T_A o T_B != T_B o T_A)."
    )
    doc.add_paragraph(
        "In naive evaluations with fixed nominal severity parameters, altering sequence order produces dramatic top-1 accuracy variations (up to 21.55% in ResNet-50). However, our prior work revealed that nominal order differences are heavily confounded by terminal smoothing: sequences ending in blur attenuate prior high-frequency noise and DCT block edges, conferring an uncalibrated low-level fidelity advantage of up to 6.2 dB PSNR over sequences ending in JPEG. This raises a fundamental research question: When global physical fidelity is matched on common support, does corruption order still matter, and what physical mechanisms explain residual vulnerability?"
    )

    # 2. Experimental Design
    p_h2 = doc.add_heading("2. Experimental Methodology & Calibration Protocol", level=1)
    doc.add_paragraph(
        "To isolate structural degradation from low-level fidelity discrepancies, we calibrated an 18-condition experimental protocol (3 severity levels x 6 sequence permutations) across an independent calibration partition (N_cal = 50). Parameter optimization was conducted strictly to match joint targets in Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) within prespecified tolerance boundaries (Delta PSNR <= 0.75 dB, Delta SSIM <= 0.030):"
    )
    doc.add_paragraph(
        "• Level 1 (Mild Common Support): Target PSNR 28.0 dB, SSIM 0.84\n"
        "• Level 2 (Moderate Common Support): Target PSNR 25.0 dB, SSIM 0.76\n"
        "• Level 3 (Severe Common Support): Target PSNR 23.5 dB, SSIM 0.69"
    )

    add_callout(
        doc,
        "Methodological Clarification: Calibration Targets vs. LPIPS Covariate",
        "The common-support calibration matched PSNR and SSIM targets only. Learned Perceptual Image Patch Similarity (LPIPS; Zhang et al., 2018) was not a calibration target. Instead, LPIPS was computed post hoc as a continuous perceptual covariate and incorporated directly into baseline and extended regression models via restricted cubic splines to control for perceptual distance alongside fidelity metrics."
    )

    doc.add_paragraph(
        "To rigorously quantify model behavior, we evaluated continuous true-class logit margins: z_margin = z_true - max_{j != true} z_j. Prediction models were estimated using 5-fold grouped cross-validation strictly partitioned on physical image clusters, ensuring zero image-level data leakage between training and validation folds. Continuous predictors (PSNR, SSIM, LPIPS) were parameterized using conventional 4-knot restricted cubic splines (knots at 5th, 35th, 65th, and 95th training percentiles; 3 basis degrees of freedom each). Hypothesis testing used paired image-cluster bootstrap resampling with B = 10,000 replicates, controlling family-wise error rate via step-down Holm-Bonferroni correction (alpha = 0.05)."
    )

    # 3. Confirmatory Results
    p_h3 = doc.add_heading("3. Primary Confirmatory Findings (N = 3,825)", level=1)
    doc.add_paragraph(
        "The confirmatory experiment evaluated 3,825 independent physical images from the Imagenette validation split, generating 290,700 model forward passes (15,300 clean reference passes + 275,400 corrupted passes). Model 3 represents the nonlinear global fidelity baseline (27 parameters: Intercept + 5 Order Dummies + 2 Tier Dummies + 10 Order x Tier Interactions + 9 RCS Spline Columns for PSNR, SSIM, LPIPS). Model 5 incorporates the five continuous distortion fingerprints (32 parameters: Model 3 + E_low + E_high + B_JPEG + Delta_Edge + sigma_local_SSIM)."
    )

    # Primary Results Table
    t1 = doc.add_table(rows=1, cols=6)
    w1 = [Inches(1.4), Inches(0.9), Inches(0.9), Inches(1.1), Inches(1.2), Inches(1.0)]
    h1 = ["Architecture", "M3 R²", "M5 R²", "ΔR² (OOS)", "Simultaneous 95% CI", "Holm p_adj"]
    d1 = [
        ["ViT-Base / 16", "0.0591", "0.0790", "+0.0199", "[+0.0115, +0.0278]", "<= 0.0004"],
        ["ConvNeXt-Tiny", "0.0728", "0.0905", "+0.0177", "[+0.0105, +0.0248]", "<= 0.0004"],
        ["Swin-Tiny", "0.0893", "0.1057", "+0.0164", "[+0.0098, +0.0230]", "<= 0.0004"],
        ["ResNet-50", "0.1224", "0.1318", "+0.0094", "[+0.0048, +0.0138]", "<= 0.0004"],
    ]
    format_table(t1, w1, h1, d1, align_right_cols=[1, 2, 3, 4, 5])

    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    doc.add_paragraph(
        "Across all four vision architectures, Model 5 achieves a statistically significant increase in explained out-of-sample logit margin variance (Delta R2 > 0) and a statistically significant reduction in prediction error (Delta MAE: ViT +0.0535, ConvNeXt +0.0385, Swin +0.0334, ResNet +0.0483; all p_adj <= 0.0004). Because zero of the 10,000 bootstrap resamples crossed the null, primary p-values are reported using exact upper-bound resolution notation (p_adj <= 0.0004). Sensitivity analysis using a reduced 2-df spline basis (24 vs 29 parameters) yielded identical conclusions (ViT Delta R2 = +0.0201, ConvNeXt +0.0180, Swin +0.0167, ResNet +0.0096; all p_adj <= 0.0004)."
    )

    # 4. External Replication
    p_h4 = doc.add_heading("4. Independent External Replication on ImageNet-V2 (N = 300)", level=1)
    doc.add_paragraph(
        "To evaluate whether predictive gains generalize to genuinely independent photographic samples collected under distinct crowd-sourced protocols, we executed the frozen 18-condition protocol across the 10 target WordNet synsets in all three ImageNet-V2 sampling regimes (MatchedFrequency, Threshold0.7, TopImages; N=100 each, N=300 pooled; 22,800 model evaluations). Zero parameters or feature selections were tuned on ImageNet-V2."
    )

    # ImageNet-V2 Table
    t2 = doc.add_table(rows=1, cols=7)
    w2 = [Inches(1.3), Inches(0.8), Inches(0.8), Inches(1.0), Inches(1.2), Inches(0.7), Inches(0.7)]
    h2 = ["Architecture", "M3 R²", "M5 R²", "ΔR² (OOS)", "95% Bootstrap CI", "Raw p", "Holm p"]
    d2 = [
        ["ViT-Base / 16", "0.0746", "0.1117", "+0.0371", "[+0.0065, +0.0675]", "0.0085", "0.0340*"],
        ["ResNet-50", "0.0931", "0.1143", "+0.0212", "[-0.0070, +0.0475]", "0.0590", "0.1770"],
        ["ConvNeXt-Tiny", "0.0397", "0.0541", "+0.0144", "[-0.0099, +0.0383]", "0.1219", "0.2438"],
        ["Swin-Tiny", "0.0554", "0.0669", "+0.0115", "[-0.0126, +0.0345]", "0.1768", "0.1768"],
    ]
    format_table(t2, w2, h2, d2, align_right_cols=[1, 2, 3, 4, 5, 6])

    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    add_callout(
        doc,
        "Rigorous Assessment of External Replication Evidence",
        "In the pooled ImageNet-V2 cohort (N=300), only ViT-Base / 16 achieves a positive out-of-sample R² confidence interval strictly excluding zero and retains statistical significance following step-down Holm-Bonferroni correction (Delta R2 = +0.0371, 95% CI: [+0.0065, +0.0675], p_adj = 0.0340). For ResNet-50, ConvNeXt-Tiny, and Swin-Tiny, pooled Delta R² point estimates remain positive (+0.0115 to +0.0212), but their 95% bootstrap confidence intervals cross zero and their adjusted p-values exceed 0.05. Statistically significant external R² improvement cannot be claimed across all four architectures; it is established rigorously on ViT-Base."
    )

    # 5. Feature Ablations
    p_h5 = doc.add_heading("5. Feature Contributions & Collinear Structure Analysis", level=1)
    doc.add_paragraph(
        "To inspect the predictive contribution of the five structural fingerprints, 10 systematic ablation conditions were analyzed across all architectures on the primary confirmatory dataset (N = 3,825):"
    )

    t3 = doc.add_table(rows=1, cols=6)
    w3 = [Inches(1.3), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.1), Inches(1.1)]
    h3 = ["Architecture", "Total M5 Gain", "+Edge Energy", "+Fourier (L/H)", "+JPEG Blocks", "+Local SSIM"]
    d3 = [
        ["ViT-Base / 16", "+0.0199", "+0.0176", "+0.0041", "+0.0016", "+0.0006"],
        ["ConvNeXt-Tiny", "+0.0177", "+0.0135", "+0.0067", "+0.0018", "-0.0003"],
        ["Swin-Tiny", "+0.0164", "+0.0111", "+0.0076", "+0.0014", "-0.0001"],
        ["ResNet-50", "+0.0094", "+0.0074", "+0.0030", "+0.0015", "-0.0001"],
    ]
    format_table(t3, w3, h3, d3, align_right_cols=[1, 2, 3, 4, 5])

    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    add_callout(
        doc,
        "Statistical Interpretation: Collinear Feature Overlap",
        "Sobel edge energy difference (Delta_Edge) is the single strongest individual predictor across all architectures, and dropping it produces the steepest decrease in model R² (ViT drops by -0.0136, ConvNeXt by -0.0101, Swin by -0.0081, ResNet by -0.0053). However, because spatial edges and Fourier frequency spectra are physically correlated, individual feature group contributions share variance. Consequently, add-one gains sum to more than Model 5's total gain and must not be interpreted as mutually exclusive, additive variance percentages. Sobel edge energy is the strongest predictive contributor in the model, rather than a proven causal physical driver."
    )

    # 6. Terminal Operator Disentanglement
    p_h6 = doc.add_heading("6. Terminal Operator Disentanglement", level=1)
    doc.add_paragraph(
        "To establish whether sequence order effects simply reflect the nominal category of the final operator (Final JPEG, Final Blur, Final Noise), four nested models were evaluated on the confirmatory cohort:"
    )
    doc.add_paragraph(
        "• Model A (Terminal Operator Dummies Only, 3 params): R²_OOS in [0.0052, 0.0204]\n"
        "• Model B (Global Fidelity + Terminal Operator Dummies, 23 params): R²_OOS in [0.0525, 0.1125]\n"
        "• Model C (Global Fidelity + Full 6-Sequence Order Dummies, Model 3: 27 params): R²_OOS in [0.0591, 0.1224]\n"
        "• Model D (Full Model: Model 3 + 5 Physical Fingerprints, Model 5: 32 params): R²_OOS in [0.0790, 0.1318]"
    )
    doc.add_paragraph(
        "Terminal operator identity alone explains virtually no logit margin variance (< 2.0%). Controlling for full nominal sequence permutation (Model C) significantly outperforms terminal operator grouping alone (Model B), adding +0.0066 to +0.0145 in R²_OOS. Furthermore, incorporating continuous distortion fingerprints (Model D) provides an additional +0.0094 to +0.0199 in R²_OOS (p < 0.0004), demonstrating that structural degradation operates within nominal operator categories."
    )

    # 7. Architectural Taxonomy & Interaction
    p_h7 = doc.add_heading("7. Architectural Taxonomy & Local-SSIM Analysis", level=1)
    doc.add_paragraph(
        "To prevent imprecise generalizations, vision models must be distinguished into three distinct structural archetypes:"
    )
    doc.add_paragraph(
        "1. Standard Convolutional Neural Networks (CNNs): ResNet-50 (residual bottleneck blocks with 3x3 convolutions) and ConvNeXt-Tiny (inverted bottleneck blocks with 7x7 depthwise convolutions). Characterized by strict local receptive fields and translation equivariance.\n"
        "2. Hierarchical Windowed Vision Transformer: Swin-Tiny. Computes self-attention strictly within localized 7x7 windows with shifted-window cross-patch connections across layers. Combines transformer expressivity with local inductive bias.\n"
        "3. Global-Attention Vision Transformer: ViT-Base/16. Computes unconstrained pairwise self-attention across all 196 image patches globally, possessing no local inductive bias."
    )

    doc.add_paragraph(
        "In Model 5, local SSIM spatial heterogeneity (sigma_local_SSIM) exhibits a negative standardized coefficient for ViT-Base (beta = -0.2691). To test whether ViT-Base differs significantly from the other architectures, we executed a formal paired image-cluster bootstrap interaction test (B = 10,000):"
    )

    t4 = doc.add_table(rows=1, cols=5)
    w4 = [Inches(1.8), Inches(1.1), Inches(1.1), Inches(1.3), Inches(1.2)]
    h4 = ["Comparison", "Δβ (ViT - Other)", "Bootstrap SE", "95% Bootstrap CI", "Bootstrap p-value"]
    d4 = [
        ["ViT-Base vs ConvNeXt-Tiny", "-0.2277", "0.0604", "[-0.3446, -0.1108]", "0.00010*"],
        ["ViT-Base vs Swin-Tiny", "-0.2201", "0.0743", "[-0.3617, -0.0726]", "0.00300*"],
        ["ViT-Base vs ResNet-50", "-0.0725", "0.1432", "[-0.3532, +0.2148]", "0.61174 (NS)"],
    ]
    format_table(t4, w4, h4, d4, align_right_cols=[1, 2, 3, 4])

    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    add_callout(
        doc,
        "Scientific Conclusion on Local-SSIM Sensitivity",
        "ViT-Base's negative local-SSIM sensitivity is statistically distinct from ConvNeXt-Tiny (p = 0.00010) and Swin-Tiny (p = 0.00300), showing that unconstrained global self-attention associates with greater margin collapse when visual degradation is spatially non-uniform. However, the difference between ViT-Base and ResNet-50 is not established (p = 0.61174, 95% CI includes zero). This result is therefore established as an architecture-specific empirical association rather than a demonstrated universal architectural mechanism."
    )

    # 8. Limitations & Pretraining Disclosure
    p_h8 = doc.add_heading("8. Limitations, Pretraining Disclosure & Boundaries", level=1)
    doc.add_paragraph(
        "1. Pretraining Exposure: Confirmatory images are drawn from the FastAI Imagenette partition of ImageNet-1K, which formed part of the pretraining data for all four vision backbones. While confirmatory evaluation enforces strict sample-level quarantine and independence from the calibration (N=50) and pilot (N=100) datasets, true out-of-distribution independence from ImageNet pretraining is evaluated solely via the independent ImageNet-V2 replication.\n"
        "2. Collinearity in Feature Attribution: Spatial edges and frequency spectra are physically coupled. Add-one and drop-one gains describe predictive importance within the model, not isolated physical intervention effects.\n"
        "3. Common-Support Distortion Bounds: Results are established for 3-operator compound degradations within the calibrated common-support envelope (PSNR in [22.8, 28.5] dB, SSIM in [0.65, 0.86]). Extrapolation to other noise regimes (e.g., salt-and-pepper, motion blur) remains unverified.\n"
        "4. External Replication Sample Constraints: In disaggregated 100-image ImageNet-V2 subsets, confidence intervals span zero due to small sample variance. Statistical significance on external data was established only for ViT-Base in the pooled cohort."
    )

    # 9. Reproducibility Registry
    p_h9 = doc.add_heading("9. Cryptographic Reproducibility Registry", level=1)
    doc.add_paragraph("All confirmatory and replication data artifacts are cryptographically verified:")

    t5 = doc.add_table(rows=1, cols=3)
    w5 = [Inches(2.8), Inches(1.0), Inches(2.7)]
    h5 = ["Artifact File Name", "Size (Bytes)", "SHA-256 Checksum"]
    d5 = [
        ["full_distortion_fingerprints_and_margins.csv.gz", "6,405,062", "22de9a8c74804e424ad7e34dcff3388fed34..."],
        ["confirmatory_grouped_cv_predictions.csv.gz", "12,616,116", "01485291af62a7c3e920856d3d75bf0a634c..."],
        ["confirmatory_statistical_comparison.csv", "2,388", "9fb8dbdffeb24116562e1cea93301e9a8b85..."],
        ["confirmatory_local_ssim_interaction_test.json", "1,058", "28e7eb8868e64c2ecbb2f98622c833aa4b61..."],
        ["imagenet_v2_all_evaluations.csv.gz", "570,801", "d8b0e8c894d0c9527cf63a79d380e5b722d7..."],
        ["imagenet_v2_statistical_comparison.csv", "2,042", "f0ae7d812406f527c243bc50085a6a5789fa..."],
        ["imagenet_v2_execution_manifest.json", "228", "9451994191c7811efda82f9d8a572c212c41..."],
    ]
    format_table(t5, w5, h5, d5, align_right_cols=[1])

    # 10. References
    p_h10 = doc.add_heading("10. Academic References", level=1)
    refs = [
        "Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. International Conference on Learning Representations (ICLR).",
        "Harrell, F. E. (2015). Regression Modeling Strategies: With Applications to Linear Models, Logistic and Ordinal Regression, and Survival Analysis (2nd ed.). Springer Series in Statistics.",
        "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778.",
        "Hendrycks, D., & Dietterich, T. (2019). Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. International Conference on Learning Representations (ICLR).",
        "Howard, J. (2019). FastAI Imagenette: A Smaller Subset of 10 Easily Classified Classes from ImageNet. GitHub repository: https://github.com/fastai/imagenette.",
        "Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. IEEE/CVF International Conference on Computer Vision (ICCV), 10012–10022.",
        "Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 11976–11986.",
        "Recht, B., Roelofs, R., Schmidt, L., & Shankar, V. (2019). Do ImageNet Classifiers Generalize to ImageNet? International Conference on Machine Learning (ICML), 5389–5400.",
        "Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE Transactions on Image Processing, 13(4), 600–612.",
        "Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 586–595.",
    ]
    for r in refs:
        p_ref = doc.add_paragraph(r)
        p_ref.paragraph_format.left_indent = Inches(0.4)
        p_ref.paragraph_format.first_line_indent = Inches(-0.4)
        p_ref.paragraph_format.space_after = Pt(4)

    doc.save(DOCX_OUT)
    print(f"Successfully generated academic manuscript: {DOCX_OUT} ({DOCX_OUT.stat().st_size:,} bytes)")


# ==============================================================================
# 2. PDF GENERATOR (REPORTLAB)
# ==============================================================================

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
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "RobustLens Confirmatory Study: Distortion Fingerprints Under Matched Fidelity")
            self.drawRightString(letter[0] - 54, letter[1] - 36, "Final Scientific Report • November 2026")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)

        # Footer (All pages)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawString(54, 36, "Confidential • RobustLens Research Project • Aviral Choudhary")
        self.drawRightString(letter[0] - 54, 36, page_str)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 46, letter[0] - 54, 46)

        self.restoreState()


def build_pdf_report():
    print("Building Polished Final Research Report PDF...")
    doc = SimpleDocTemplate(
        str(PDF_OUT),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom typography
    styles.add(ParagraphStyle(
        name="DocTitle",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="DocSubtitle",
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="SecHead",
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="BodyTextCustom",
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="CalloutTitle",
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#0369A1"),
        spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="CalloutBody",
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
    ))
    styles.add(ParagraphStyle(
        name="TableHeader",
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=0,
    ))
    styles.add(ParagraphStyle(
        name="TableHeaderR",
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=2,
    ))
    styles.add(ParagraphStyle(
        name="TableCell",
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#1E293B"),
        alignment=0,
    ))
    styles.add(ParagraphStyle(
        name="TableCellR",
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#1E293B"),
        alignment=2,
    ))
    styles.add(ParagraphStyle(
        name="RefText",
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#334155"),
        spaceAfter=4,
    ))

    story = []

    # Title & Metadata
    story.append(Paragraph("Spatial-Frequency and Structural Distortion Fingerprints Under Matched Global Image Fidelity", styles["DocTitle"]))
    story.append(Paragraph("<b>Author:</b> Aviral Choudhary &nbsp;|&nbsp; <b>Affiliation:</b> RobustLens Research Project &nbsp;|&nbsp; <b>Date:</b> November 2026<br/><b>Evaluations:</b> 313,500 model passes (3,825 confirmatory images + 300 ImageNet-V2 replication images) &nbsp;|&nbsp; <b>Status:</b> Confirmatory Complete", styles["DocSubtitle"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceAfter=10))

    # Executive Summary Box
    def make_callout(title_txt, body_txt, border_col="#0284C7", bg_col="#F0F9FF"):
        p_t = Paragraph(f"<b>{title_txt}</b>", styles["CalloutTitle"])
        p_b = Paragraph(body_txt, styles["CalloutBody"])
        tbl = Table([[p_t], [p_b]], colWidths=[504])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg_col)),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#BAE6FD")),
            ("LINEBEFORE", (0, 0), (0, -1), 3.0, colors.HexColor(border_col)),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        return tbl

    story.append(make_callout(
        "Executive Summary & Confirmatory Outcome",
        "<b>1. Confirmatory Hypotheses Supported:</b> Incorporating five prespecified distortion fingerprints yields a statistically significant increase in out-of-sample logit margin explained variance (ΔR²_OOS > 0) across all four architectures (ViT-Base +0.0199, ConvNeXt +0.0177, Swin +0.0164, ResNet +0.0094; all p_adj <= 0.0004 under Holm-Bonferroni control).<br/>"
        "<b>2. External Replication on ImageNet-V2 (N=300):</b> Confirmed with statistical significance solely on ViT-Base (Delta R2 = +0.0371, 95% CI: [+0.0065, +0.0675], p_adj = 0.0340). For CNNs and Swin-Tiny, pooled gains remain positive but do not reach significance (p_adj in [0.1768, 0.2438]).<br/>"
        "<b>3. Dominant Physical Predictor:</b> Sobel edge energy difference is the strongest individual contributor in the model, though collinear feature overlap caveats apply.<br/>"
        "<b>4. Architecture-Specific Association:</b> Local SSIM spatial variance selectively associates with margin collapse in ViT-Base compared to ConvNeXt (p=0.00010) and Swin-Tiny (p=0.00300), but not ResNet-50 (p=0.61174)."
    ))
    story.append(Spacer(1, 10))

    # Section 1
    story.append(Paragraph("1. Pretraining Exposure & Calibration Protocol Disclosures", styles["SecHead"]))
    story.append(Paragraph(
        "<b>Pretraining Exposure:</b> The 3,825 confirmatory images belong to the 10 WordNet synsets of FastAI Imagenette-v2, which originate from ImageNet-1K. All four backbones (ResNet-50, ConvNeXt-Tiny, ViT-Base, Swin-Tiny) were pretrained on ImageNet-1K. The confirmatory cohort enforces strict sample-level quarantine from the calibration (N=50) and pilot (N=100) datasets (0 byte overlap via SHA-256). True out-of-distribution independence from ImageNet pretraining is tested exclusively via the ImageNet-V2 replication.<br/>"
        "<b>Calibration vs. LPIPS:</b> Parameter calibration across the 18 conditions matched PSNR and SSIM targets only (Level 1: 28 dB / 0.84; Level 2: 25 dB / 0.76; Level 3: 23.5 dB / 0.69). LPIPS was not a calibration target; it was evaluated post hoc and included as a continuous regression predictor via splines.",
        styles["BodyTextCustom"]
    ))
    story.append(Spacer(1, 6))

    # Section 2: Primary Results
    story.append(Paragraph("2. Primary Confirmatory Results (N = 3,825 Images, 290,700 Passes)", styles["SecHead"]))
    story.append(Paragraph(
        "5-fold grouped cross-validation partitioned strictly on 3,825 image clusters. B=10,000 paired cluster bootstrap resamples. Model 3 (Global Fidelity Baseline: 27 params, 4-knot RCS on PSNR/SSIM/LPIPS) vs Model 5 (Full Fingerprints: 32 params).",
        styles["BodyTextCustom"]
    ))

    t_conf_data = [
        [Paragraph("Architecture", styles["TableHeader"]),
         Paragraph("M3 R²", styles["TableHeaderR"]),
         Paragraph("M5 R²", styles["TableHeaderR"]),
         Paragraph("ΔR² (OOS)", styles["TableHeaderR"]),
         Paragraph("Simultaneous 95% CI", styles["TableHeaderR"]),
         Paragraph("1-Sided 95% LB", styles["TableHeaderR"]),
         Paragraph("ΔMAE (OOS)", styles["TableHeaderR"]),
         Paragraph("Holm p_adj", styles["TableHeaderR"])]
    ]
    raw_conf = [
        ["ViT-Base / 16", "0.0591", "0.0790", "+0.0199", "[+0.0115, +0.0278]", "+0.0123", "+0.0535", "<= 0.0004"],
        ["ConvNeXt-Tiny", "0.0728", "0.0905", "+0.0177", "[+0.0105, +0.0248]", "+0.0113", "+0.0385", "<= 0.0004"],
        ["Swin-Tiny", "0.0893", "0.1057", "+0.0164", "[+0.0098, +0.0230]", "+0.0105", "+0.0334", "<= 0.0004"],
        ["ResNet-50", "0.1224", "0.1318", "+0.0094", "[+0.0048, +0.0138]", "+0.0053", "+0.0483", "<= 0.0004"],
    ]
    for r in raw_conf:
        t_conf_data.append([
            Paragraph(r[0], styles["TableCell"]),
            Paragraph(r[1], styles["TableCellR"]),
            Paragraph(r[2], styles["TableCellR"]),
            Paragraph(f"<b>{r[3]}</b>", styles["TableCellR"]),
            Paragraph(r[4], styles["TableCellR"]),
            Paragraph(r[5], styles["TableCellR"]),
            Paragraph(r[6], styles["TableCellR"]),
            Paragraph(f"<b>{r[7]}</b>", styles["TableCellR"]),
        ])
    tbl_conf = Table(t_conf_data, colWidths=[90, 45, 45, 60, 95, 60, 55, 54])
    tbl_conf.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(tbl_conf)
    story.append(Spacer(1, 8))

    # Section 3: External Replication
    story.append(Paragraph("3. Independent External Replication on ImageNet-V2 (N = 300 Images, 22,800 Passes)", styles["SecHead"]))
    story.append(Paragraph(
        "Evaluated on 10 matching synsets across MatchedFrequency, Threshold0.7, and TopImages with frozen calibration parameters. Zero parameters tuned on ImageNet-V2.",
        styles["BodyTextCustom"]
    ))

    t_v2_data = [
        [Paragraph("Architecture", styles["TableHeader"]),
         Paragraph("M3 R²", styles["TableHeaderR"]),
         Paragraph("M5 R²", styles["TableHeaderR"]),
         Paragraph("ΔR² (OOS)", styles["TableHeaderR"]),
         Paragraph("95% Bootstrap CI", styles["TableHeaderR"]),
         Paragraph("Raw p", styles["TableHeaderR"]),
         Paragraph("Holm p_adj", styles["TableHeaderR"]),
         Paragraph("External Statistical Decision", styles["TableHeader"])]
    ]
    raw_v2 = [
        ["ViT-Base / 16", "0.0746", "0.1117", "+0.0371", "[+0.0065, +0.0675]", "0.0085", "0.0340*", "REJECT NULL (Confirmed)"],
        ["ResNet-50", "0.0931", "0.1143", "+0.0212", "[-0.0070, +0.0475]", "0.0590", "0.1770", "Fail to Reject (p > 0.05)"],
        ["ConvNeXt-Tiny", "0.0397", "0.0541", "+0.0144", "[-0.0099, +0.0383]", "0.1219", "0.2438", "Fail to Reject (p > 0.05)"],
        ["Swin-Tiny", "0.0554", "0.0669", "+0.0115", "[-0.0126, +0.0345]", "0.1768", "0.1768", "Fail to Reject (p > 0.05)"],
    ]
    for r in raw_v2:
        t_v2_data.append([
            Paragraph(r[0], styles["TableCell"]),
            Paragraph(r[1], styles["TableCellR"]),
            Paragraph(r[2], styles["TableCellR"]),
            Paragraph(f"<b>{r[3]}</b>", styles["TableCellR"]),
            Paragraph(r[4], styles["TableCellR"]),
            Paragraph(r[5], styles["TableCellR"]),
            Paragraph(f"<b>{r[6]}</b>", styles["TableCellR"]),
            Paragraph(r[7], styles["TableCell"]),
        ])
    tbl_v2 = Table(t_v2_data, colWidths=[85, 45, 45, 55, 95, 45, 50, 84])
    tbl_v2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(tbl_v2)
    story.append(Spacer(1, 8))

    # Section 4: Feature Contributions & Collinear Structure
    story.append(Paragraph("4. Feature Contributions, Collinear Overlap & Terminal Operator Disentanglement", styles["SecHead"]))
    story.append(Paragraph(
        "<b>Feature Contributions:</b> Sobel edge energy (ΔEdge) is the single strongest individual predictor across all models (ViT +0.0176, ConvNeXt +0.0135, Swin +0.0111, ResNet +0.0074). However, because spatial gradients and Fourier frequencies are physically correlated, individual add-one gains overlap and cannot be interpreted as mutually exclusive variance percentages.<br/>"
        "<b>Terminal Operator Disentanglement:</b> Modeling terminal operator identity alone explains only 0.5% - 2.0% of variance. Full sequence permutation (Model 3) adds +0.0066 to +0.0145. Physical distortion fingerprints (Model 5) add an additional +0.0094 to +0.0199 (p < 0.0004), demonstrating that structural degradation operates within nominal operator categories.",
        styles["BodyTextCustom"]
    ))
    story.append(Spacer(1, 6))

    # Section 5: Architectural Taxonomy & Local SSIM Test
    story.append(Paragraph("5. Architectural Taxonomy & Formal Local-SSIM Interaction Test", styles["SecHead"]))
    story.append(Paragraph(
        "Architectures are categorized into three paradigms: (1) Standard CNNs (ResNet-50, ConvNeXt-Tiny) with local receptive fields; (2) Hierarchical Windowed Transformer (Swin-Tiny) with shifted 7x7 windows; (3) Global-Attention Transformer (ViT-Base/16) with unconstrained all-to-all patch attention.<br/>"
        "To test whether ViT-Base's negative local-SSIM coefficient (β = -0.2691) is statistically distinct, a formal paired cluster bootstrap test (B=10,000) was performed:",
        styles["BodyTextCustom"]
    ))

    t_inter_data = [
        [Paragraph("Comparison", styles["TableHeader"]),
         Paragraph("Δβ (ViT - Other)", styles["TableHeaderR"]),
         Paragraph("Bootstrap SE", styles["TableHeaderR"]),
         Paragraph("95% Bootstrap CI", styles["TableHeaderR"]),
         Paragraph("Two-Sided p", styles["TableHeaderR"]),
         Paragraph("Statistical Conclusion", styles["TableHeader"])]
    ]
    raw_inter = [
        ["ViT-Base vs ConvNeXt-Tiny", "-0.2277", "0.0604", "[-0.3446, -0.1108]", "0.00010*", "Statistically Significant"],
        ["ViT-Base vs Swin-Tiny", "-0.2201", "0.0743", "[-0.3617, -0.0726]", "0.00300*", "Statistically Significant"],
        ["ViT-Base vs ResNet-50", "-0.0725", "0.1432", "[-0.3532, +0.2148]", "0.61174", "Not Significant (p > 0.05)"],
    ]
    for r in raw_inter:
        t_inter_data.append([
            Paragraph(r[0], styles["TableCell"]),
            Paragraph(f"<b>{r[1]}</b>", styles["TableCellR"]),
            Paragraph(r[2], styles["TableCellR"]),
            Paragraph(r[3], styles["TableCellR"]),
            Paragraph(f"<b>{r[4]}</b>", styles["TableCellR"]),
            Paragraph(r[5], styles["TableCell"]),
        ])
    tbl_inter = Table(t_inter_data, colWidths=[120, 75, 65, 95, 65, 84])
    tbl_inter.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(tbl_inter)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Interpretation:</b> ViT-Base's local-SSIM sensitivity differs significantly from ConvNeXt-Tiny and Swin-Tiny, but its difference from ResNet-50 is not established. This is framed as an architecture-specific empirical association rather than a proven universal architectural mechanism.",
        styles["BodyTextCustom"]
    ))
    story.append(Spacer(1, 6))

    # Section 6: Reproducibility & References
    story.append(Paragraph("6. Cryptographic Integrity Registry & Academic References", styles["SecHead"]))
    story.append(Paragraph(
        "<b>Verified Checksums:</b><br/>"
        "• full_distortion_fingerprints_and_margins.csv.gz (6,405,062 B): 22de9a8c74804e424ad7e34dcff3388fed34261e33454e53efdb47adcef58eb6<br/>"
        "• confirmatory_statistical_comparison.csv (2,388 B): 9fb8dbdffeb24116562e1cea93301e9a8b855198acdf28bd7011b3219fe6fa15<br/>"
        "• imagenet_v2_all_evaluations.csv.gz (570,801 B): d8b0e8c894d0c9527cf63a79d380e5b722d7cb760971b87db8c0b5c907106fe6<br/>"
        "• imagenet_v2_statistical_comparison.csv (2,042 B): f0ae7d812406f527c243bc50085a6a5789fa1eec8b71d60b54070a969f62c644",
        styles["RefText"]
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Key References:</b><br/>"
        "1. Hendrycks & Dietterich (2019) ICLR (ImageNet-C) &nbsp;|&nbsp; 2. Recht et al. (2019) ICML (ImageNet-V2) &nbsp;|&nbsp; 3. Zhang et al. (2018) CVPR (LPIPS)<br/>"
        "4. Wang et al. (2004) IEEE TIP (SSIM) &nbsp;|&nbsp; 5. Harrell (2015) Springer (RCS Splines) &nbsp;|&nbsp; 6. Dosovitskiy et al. (2020) ICLR (ViT)<br/>"
        "7. Liu et al. (2021) ICCV (Swin Transformer) &nbsp;|&nbsp; 8. Liu et al. (2022) CVPR (ConvNeXt) &nbsp;|&nbsp; 9. He et al. (2016) CVPR (ResNet)",
        styles["RefText"]
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF report: {PDF_OUT} ({PDF_OUT.stat().st_size:,} bytes)")


def main():
    build_academic_docx()
    build_pdf_report()
    print("\nBoth deliverables successfully generated!")


if __name__ == "__main__":
    main()
