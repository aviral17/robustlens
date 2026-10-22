"""RobustLens Project Guide High-Resolution PDF Generator.

Embeds TrueType fonts (Segoe UI / Arial / Consolas) for razor-sharp typography,
renders all diagrams and research plots directly into the PDF, and applies
ultra-high-contrast executive styling.
"""

from __future__ import annotations

import re
import html
import os
from pathlib import Path
from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
    Preformatted,
    Image as RLImage,
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def register_fonts() -> tuple[str, str, str]:
    """Registers crisp TrueType vector fonts from Windows system font directory."""
    font_dir = "C:/Windows/Fonts"
    body_font = "Helvetica"
    bold_font = "Helvetica-Bold"
    mono_font = "Courier"

    try:
        # Segoe UI
        segoe_path = os.path.join(font_dir, "segoeui.ttf")
        segoe_b_path = os.path.join(font_dir, "segoeuib.ttf")
        consola_path = os.path.join(font_dir, "consola.ttf")

        if os.path.exists(segoe_path) and os.path.exists(segoe_b_path):
            pdfmetrics.registerFont(TTFont("SegoeUI", segoe_path))
            pdfmetrics.registerFont(TTFont("SegoeUI-Bold", segoe_b_path))
            body_font = "SegoeUI"
            bold_font = "SegoeUI-Bold"

        if os.path.exists(consola_path):
            pdfmetrics.registerFont(TTFont("Consolas", consola_path))
            mono_font = "Consolas"
    except Exception as e:
        print(f"Font registration notice: {e}, using default vector font fallbacks.")

    return body_font, bold_font, mono_font


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page counts."""

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
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages: int):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#0f172a"))

        # Header (Pages 2+)
        if self._pageNumber > 1:
            self.drawString(36, 756, "RobustLens: The Complete Project Guide & Platform Walkthrough")
            self.drawRightString(576, 756, "Aviral Choudhary | Zenodo DOI: 10.5281/zenodo.23483939")
            self.setStrokeColor(colors.HexColor("#94a3b8"))
            self.setLineWidth(0.6)
            self.line(36, 750, 576, 750)

        # Footer (All Pages)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#334155"))
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(576, 25, page_str)
        self.drawString(36, 25, "https://robustlens.vercel.app  •  https://github.com/aviral17/robustlens")
        self.setStrokeColor(colors.HexColor("#94a3b8"))
        self.setLineWidth(0.6)
        self.line(36, 35, 576, 35)

        self.restoreState()


def clean_markdown_inline(text: str) -> str:
    """Converts common markdown inline formatting to ReportLab XML tags with high contrast."""
    text = html.escape(text, quote=False)

    # Clean LaTeX math expressions
    text = text.replace(r"{,}", ",")
    text = text.replace(r"\times", "×")
    text = text.replace(r"\circ", "○").replace(r"\to", "→").replace(r"\neq", "≠")
    text = text.replace(r"\Delta", "Δ").replace(r"\chi^2", "χ²")
    text = text.replace(r"\equiv", "≡").replace(r"\pm", "±").replace(r"\approx", "≈")
    text = text.replace(r"\le", "≤").replace(r"\ge", "≥")
    text = text.replace(r"\mathcal{N}", "Noise").replace(r"\mathcal{B}", "Blur").replace(r"\mathcal{J}", "JPEG")
    text = text.replace(r"\mathcal{T}", "T")
    text = text.replace(r"\alpha", "α").replace(r"\det", "det")
    text = re.sub(r"\\text\{([^}]+)\}", r"\1", text)
    text = text.replace(r"_{\text{cal}}", "_cal").replace(r"_{\text{clean}}", "_clean").replace(r"_{\text{shifted}}", "_shifted")
    text = text.replace(r"_{00}^{\text{diff}}", "₀₀(diff)").replace(r"_{00}^{\text{same}}", "₀₀(same)")
    text = text.replace(r"_{11}", "₁₁").replace(r"_{10}", "₁₀").replace(r"_{01}", "₀₁").replace(r"_{00}", "₀₀")

    # Delimiters $...$ -> clean italic
    text = re.sub(r"\$([^\$]+)\$", r"<i>\1</i>", text)

    # Links: [label](url)
    def link_repl(match):
        label, url = match.group(1), match.group(2)
        if url.startswith("http://") or url.startswith("https://"):
            return f'<link href="{url}"><font color="#1d4ed8"><b><u>{label}</u></b></font></link>'
        else:
            return f'<font color="#0f172a"><b>{label}</b></font>'
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link_repl, text)

    # Bold: **text** -> <b>text</b>
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)

    # Italic: *text* -> <i>text</i>
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)

    # Inline code: `text` -> bold dark font
    text = re.sub(r"`([^`]+)`", r'<font name="Consolas" size="8.5" color="#09090b"><b>\1</b></font>', text)

    return text


def build_pdf(md_path: Path, output_pdf_path: Path):
    body_font, bold_font, mono_font = register_fonts()

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=46,
        bottomMargin=46,
    )

    styles = getSampleStyleSheet()

    # Ultra-high-contrast typography
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName=bold_font,
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#020617"),
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName=bold_font,
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1d4ed8"),
        spaceAfter=12,
    )

    h1_style = ParagraphStyle(
        "DocH1",
        parent=styles["Normal"],
        fontName=bold_font,
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#020617"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Normal"],
        fontName=bold_font,
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True,
    )

    h3_style = ParagraphStyle(
        "DocH3",
        parent=styles["Normal"],
        fontName=bold_font,
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName=body_font,
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor("#09090b"),  # Deep crisp charcoal
        spaceAfter=5,
    )

    bullet_style = ParagraphStyle(
        "DocBullet",
        parent=styles["Normal"],
        fontName=body_font,
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor("#09090b"),
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3,
    )

    callout_style = ParagraphStyle(
        "DocCallout",
        parent=styles["Normal"],
        fontName=body_font,
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#082f49"),  # Dark cyan-slate
        spaceBefore=4,
        spaceAfter=4,
    )

    caption_style = ParagraphStyle(
        "DocCaption",
        parent=styles["Normal"],
        fontName=body_font,
        fontSize=8,
        leading=10,
        alignment=1,  # Centered
        textColor=colors.HexColor("#475569"),
        spaceBefore=4,
        spaceAfter=8,
    )

    code_style = ParagraphStyle(
        "DocCode",
        parent=styles["Normal"],
        fontName=mono_font,
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0f172a"),
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName=bold_font,
        fontSize=8,
        leading=10,
        textColor=colors.white,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName=body_font,
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#09090b"),
    )

    story = []
    lines = md_text.splitlines()
    i = 0
    in_code_block = False
    code_lines = []
    code_lang = ""

    docs_dir = md_path.parent

    while i < len(lines):
        line = lines[i]

        # Check for image: ![alt](path)
        img_match = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", line.strip())
        if img_match:
            alt_text = img_match.group(1)
            img_rel_path = img_match.group(2)
            img_full_path = (docs_dir / img_rel_path).resolve()

            if img_full_path.exists():
                try:
                    with PILImage.open(img_full_path) as pil_im:
                        orig_w, orig_h = pil_im.size

                    max_w = 530.0
                    max_h = 300.0
                    scale = min(max_w / orig_w, max_h / orig_h, 1.0)
                    render_w = orig_w * scale
                    render_h = orig_h * scale

                    img_flowable = RLImage(str(img_full_path), width=render_w, height=render_h)

                    # Wrap image in a table for clean visual borders & centering
                    img_table = Table([[img_flowable]], colWidths=[render_w])
                    img_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                        ('TOPPADDING', (0, 0), (-1, -1), 2),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                        ('LEFTPADDING', (0, 0), (-1, -1), 2),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 2),
                    ]))

                    caption = Paragraph(f"<b>Figure:</b> {alt_text}", caption_style)
                    story.append(Spacer(1, 6))
                    story.append(KeepTogether([img_table, caption]))
                    story.append(Spacer(1, 6))
                except Exception as err:
                    print(f"Error loading image {img_full_path}: {err}")
            i += 1
            continue

        # Code block handling
        if line.startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_lang = line[3:].strip()
                code_lines = []
                i += 1
                continue
            else:
                in_code_block = False
                code_content = "\n".join(code_lines)
                pre = Preformatted(code_content[:2500], code_style)
                code_table = Table([[pre]], colWidths=[540])
                code_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                    ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor("#94a3b8")),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ]))
                story.append(code_table)
                story.append(Spacer(1, 6))
                i += 1
                continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Skip empty lines
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        # Horizontal rules
        if stripped in ("---", "___", "***"):
            story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#94a3b8"), spaceBefore=8, spaceAfter=8))
            i += 1
            continue

        # Markdown Tables: | cell | cell |
        if stripped.startswith("|") and stripped.endswith("|"):
            table_raw_rows = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                row_line = lines[i].strip()
                if not re.match(r"^\|(\s*:?-+:?\s*\|)+$", row_line):
                    cells = [c.strip() for c in row_line[1:-1].split("|")]
                    table_raw_rows.append(cells)
                i += 1

            if table_raw_rows:
                num_cols = max(len(r) for r in table_raw_rows)
                formatted_data = []
                for r_idx, row in enumerate(table_raw_rows):
                    row_cells = []
                    while len(row) < num_cols:
                        row.append("")
                    for c_idx, cell in enumerate(row):
                        cleaned = clean_markdown_inline(cell)
                        if r_idx == 0:
                            row_cells.append(Paragraph(cleaned, table_header_style))
                        else:
                            row_cells.append(Paragraph(cleaned, table_cell_style))
                    formatted_data.append(row_cells)

                avail_w = 540
                col_w = avail_w / num_cols
                t = Table(formatted_data, colWidths=[col_w] * num_cols)
                t_style = [
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('LEFTPADDING', (0, 0), (-1, -1), 5),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ]
                for r_idx in range(1, len(formatted_data)):
                    if r_idx % 2 == 1:
                        t_style.append(('BACKGROUND', (0, r_idx), (-1, r_idx), colors.HexColor("#f8fafc")))
                    else:
                        t_style.append(('BACKGROUND', (0, r_idx), (-1, r_idx), colors.white))

                t.setStyle(TableStyle(t_style))
                story.append(t)
                story.append(Spacer(1, 8))
            continue

        # Blockquote: > text
        if stripped.startswith(">"):
            quote_text = stripped[1:].strip()
            while i + 1 < len(lines) and lines[i+1].strip().startswith(">"):
                i += 1
                quote_text += " " + lines[i].strip()[1:].strip()
            p = Paragraph(clean_markdown_inline(quote_text), callout_style)
            quote_table = Table([[p]], colWidths=[540])
            quote_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f0f9ff")),
                ('LINELEFT', (0, 0), (0, -1), 3.5, colors.HexColor("#0284c7")),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#bae6fd")),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(quote_table)
            story.append(Spacer(1, 6))
            i += 1
            continue

        # Headers
        if stripped.startswith("# "):
            title_text = stripped[2:].strip()
            story.append(Paragraph(clean_markdown_inline(title_text), title_style))
            story.append(Spacer(1, 4))
            i += 1
            continue
        elif stripped.startswith("## "):
            story.append(Paragraph(clean_markdown_inline(stripped[3:].strip()), h1_style))
            i += 1
            continue
        elif stripped.startswith("### "):
            story.append(Paragraph(clean_markdown_inline(stripped[4:].strip()), h2_style))
            i += 1
            continue
        elif stripped.startswith("#### "):
            story.append(Paragraph(clean_markdown_inline(stripped[5:].strip()), h3_style))
            i += 1
            continue

        # Bullet lists (*, -, 1., 2.)
        bullet_match = re.match(r"^(\*|-|\d+\.)\s+(.+)$", stripped)
        if bullet_match:
            bullet_sym = bullet_match.group(1)
            bullet_content = bullet_match.group(2)
            prefix = "• " if bullet_sym in ("*", "-") else f"{bullet_sym} "
            story.append(Paragraph(f"{prefix}{clean_markdown_inline(bullet_content)}", bullet_style))
            i += 1
            continue

        # Regular paragraph
        story.append(Paragraph(clean_markdown_inline(stripped), body_style))
        i += 1

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated high-contrast PDF with embedded TrueType fonts & images at {output_pdf_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    md_file = base_dir / "docs" / "ROBUSTLENS_COMPLETE_PROJECT_GUIDE.md"
    pdf_file = base_dir / "docs" / "ROBUSTLENS_COMPLETE_PROJECT_GUIDE.pdf"
    build_pdf(md_file, pdf_file)
