"""RobustLens Project Guide PDF Generator.

Converts docs/ROBUSTLENS_COMPLETE_PROJECT_GUIDE.md into a high-quality,
professionally styled PDF using ReportLab 5.0.1.
"""

from __future__ import annotations

import re
import html
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
    Preformatted,
)
from reportlab.pdfgen import canvas


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
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Do not draw header on the cover page (page 1)
        if self._pageNumber > 1:
            # Header
            self.drawString(36, 756, "RobustLens: The Complete Project Guide & Platform Walkthrough")
            self.drawRightString(576, 756, "Aviral Choudhary | Zenodo DOI: 10.5281/zenodo.23483939")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(36, 750, 576, 750)

        # Footer on all pages
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(576, 25, page_str)
        self.drawString(36, 25, "https://robustlens.vercel.app  •  https://github.com/aviral17/robustlens")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(36, 35, 576, 35)

        self.restoreState()


def clean_markdown_inline(text: str) -> str:
    """Converts common markdown inline formatting to reportlab XML tags."""
    # Escape HTML special chars first
    text = html.escape(text, quote=False)

    # Convert LaTeX math delimiters to italicized or styled representation
    # e.g., $...$ -> <i>...</i>
    text = re.sub(r"\$([^\$]+)\$", r"<i>\1</i>", text)
    text = text.replace(r"\circ", "○").replace(r"\to", "→").replace(r"\neq", "≠")
    text = text.replace(r"\Delta", "Δ").replace(r"\chi^2", "χ²")
    text = text.replace(r"\mathcal{N}", "N(Noise)").replace(r"\mathcal{B}", "B(Blur)").replace(r"\mathcal{J}", "J(JPEG)")
    text = text.replace(r"\alpha", "α").replace(r"\det", "det")

    # Links: [label](url) -> <link href="url"><u>label</u></link> only for http/https URLs
    def link_repl(match):
        label, url = match.group(1), match.group(2)
        if url.startswith("http://") or url.startswith("https://"):
            return f'<link href="{url}"><font color="#2563eb"><u>{label}</u></font></link>'
        else:
            return f'<font color="#1e3a8a"><b>{label}</b></font>'
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link_repl, text)

    # Bold: **text** -> <b>text</b>
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)

    # Italic: *text* -> <i>text</i>
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)

    # Inline code: `text` -> <font name="Courier" color="#0f172a" backcolor="#e2e8f0">&nbsp;\1&nbsp;</font>
    text = re.sub(r"`([^`]+)`", r'<font name="Courier" size="8.5" color="#0f172a">\1</font>', text)

    return text


def build_pdf(md_path: Path, output_pdf_path: Path):
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

    # Custom styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#2563eb"),
        spaceAfter=12,
    )

    h1_style = ParagraphStyle(
        "DocH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )

    h3_style = ParagraphStyle(
        "DocH3",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "DocBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3,
    )

    callout_style = ParagraphStyle(
        "DocCallout",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=4,
        spaceAfter=4,
    )

    code_style = ParagraphStyle(
        "DocCode",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0f172a"),
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1e293b"),
    )

    story = []

    lines = md_text.splitlines()
    i = 0
    in_code_block = False
    code_lines = []
    code_lang = ""

    while i < len(lines):
        line = lines[i]

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
                # Render code box
                pre = Preformatted(code_content[:2000], code_style)
                code_table = Table([[pre]], colWidths=[540])
                code_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                    ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
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
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceBefore=8, spaceAfter=8))
            i += 1
            continue

        # Markdown Tables: | cell | cell |
        if stripped.startswith("|") and stripped.endswith("|"):
            table_raw_rows = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                row_line = lines[i].strip()
                # Check if it's separator row like | :--- | :--- |
                if not re.match(r"^\|(\s*:?-+:?\s*\|)+$", row_line):
                    # It's actual content
                    cells = [c.strip() for c in row_line[1:-1].split("|")]
                    table_raw_rows.append(cells)
                i += 1

            if table_raw_rows:
                # Format into ReportLab Table
                num_cols = max(len(r) for r in table_raw_rows)
                formatted_data = []
                for r_idx, row in enumerate(table_raw_rows):
                    row_cells = []
                    # pad if missing
                    while len(row) < num_cols:
                        row.append("")
                    for c_idx, cell in enumerate(row):
                        cleaned = clean_markdown_inline(cell)
                        if r_idx == 0:
                            row_cells.append(Paragraph(cleaned, table_header_style))
                        else:
                            row_cells.append(Paragraph(cleaned, table_cell_style))
                    formatted_data.append(row_cells)

                # Column widths allocation
                avail_w = 540
                col_w = avail_w / num_cols
                t = Table(formatted_data, colWidths=[col_w] * num_cols)
                t_style = [
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
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
            # collect continuation lines
            while i + 1 < len(lines) and lines[i+1].strip().startswith(">"):
                i += 1
                quote_text += " " + lines[i].strip()[1:].strip()
            p = Paragraph(clean_markdown_inline(quote_text), callout_style)
            quote_table = Table([[p]], colWidths=[540])
            quote_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
                ('LINELEFT', (0, 0), (0, -1), 3, colors.HexColor("#2563eb")),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#bfdbfe")),
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
            # If line 1, use title style
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
    print(f"Successfully generated PDF at {output_pdf_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    md_file = base_dir / "docs" / "ROBUSTLENS_COMPLETE_PROJECT_GUIDE.md"
    pdf_file = base_dir / "docs" / "ROBUSTLENS_COMPLETE_PROJECT_GUIDE.pdf"
    build_pdf(md_file, pdf_file)
