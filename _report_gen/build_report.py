"""Generate the full MicroACT + MicroVLA code-explanation PDF."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    PageBreak,
    KeepTogether,
)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

styles = getSampleStyleSheet()

TITLE = ParagraphStyle(
    "Title",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=22,
    spaceAfter=8,
    alignment=1,
)
SUBTITLE = ParagraphStyle(
    "Subtitle",
    parent=styles["Normal"],
    fontSize=10,
    textColor=colors.grey,
    spaceAfter=18,
    alignment=1,
)
H1 = ParagraphStyle(
    "H1",
    parent=styles["Heading1"],
    fontName="Helvetica-Bold",
    fontSize=18,
    spaceBefore=18,
    spaceAfter=8,
    textColor=colors.HexColor("#222222"),
)
H2 = ParagraphStyle(
    "H2",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=12,
    spaceBefore=10,
    spaceAfter=4,
    textColor=colors.HexColor("#222222"),
)
H3 = ParagraphStyle(
    "H3",
    parent=styles["Heading3"],
    fontName="Helvetica-Bold",
    fontSize=10,
    spaceBefore=8,
    spaceAfter=2,
    textColor=colors.HexColor("#444444"),
)
BODY = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=9.5,
    leading=12,
    spaceAfter=4,
)
BULLET = ParagraphStyle(
    "Bullet",
    parent=BODY,
    leftIndent=14,
    bulletIndent=4,
    spaceAfter=2,
)
CODE_LABEL = ParagraphStyle(
    "CodeLabel",
    parent=BODY,
    fontName="Helvetica-Oblique",
    fontSize=8.5,
    textColor=colors.HexColor("#1a4f8a"),
    spaceBefore=4,
    spaceAfter=2,
)
CODE = ParagraphStyle(
    "Code",
    parent=styles["Code"],
    fontName="Courier",
    fontSize=7.5,
    leading=9.5,
    leftIndent=10,
    backColor=colors.HexColor("#f4f4f4"),
    borderColor=colors.HexColor("#dddddd"),
    borderWidth=0.5,
    borderPadding=4,
    spaceBefore=2,
    spaceAfter=4,
)


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------


def title_block(story, title, subtitle):
    story.append(Paragraph(title, TITLE))
    story.append(Paragraph(subtitle, SUBTITLE))


def h1(story, text):
    story.append(Paragraph(text, H1))


def h2(story, text):
    story.append(Paragraph(text, H2))


def h3(story, text):
    story.append(Paragraph(text, H3))


def body(story, text):
    story.append(Paragraph(text, BODY))


def bullets(story, items):
    for it in items:
        story.append(Paragraph(it, BULLET, bulletText="•"))


def code_block(story, label, code_text):
    story.append(Paragraph(label, CODE_LABEL))
    story.append(Preformatted(code_text, CODE))


# ---------------------------------------------------------------------------
# Page footer
# ---------------------------------------------------------------------------


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(
        0.75 * inch, 0.5 * inch,
        "MicroACT + MicroVLA full code explanation report",
    )
    canvas.drawRightString(
        letter[0] - 0.75 * inch, 0.5 * inch, f"Page {doc.page}"
    )
    canvas.restoreState()


def build(out_path: Path, story):
    doc = BaseDocTemplate(
        str(out_path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="MicroACT + MicroVLA Full Code Explanation Report",
    )
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id="normal",
    )
    template = PageTemplate(id="main", frames=[frame], onPage=_footer)
    doc.addPageTemplates([template])
    doc.build(story)


# ---------------------------------------------------------------------------
# Content modules (filled in section files)
# ---------------------------------------------------------------------------

from sections import preamble, top_level, config_files, data_files
from sections import model_act, model_vla, finetune, rollout_files, viz_files, markers


def main():
    story = []

    title_block(
        story,
        "MicroACT + MicroVLA Full Code Explanation Report",
        "Line-level walkthrough of every source file in the repo, with explicit "
        "tensor shapes at every step. README.md and the existing PDFs are skipped by "
        "request; everything else is covered.",
    )

    preamble.add(story)
    top_level.add(story)
    config_files.add(story)
    data_files.add(story)
    model_act.add(story)
    model_vla.add(story)
    finetune.add(story)
    rollout_files.add(story)
    viz_files.add(story)
    markers.add(story)

    out = Path("/home/raianlaptop/MicroACT/microact_full_code_report.pdf")
    build(out, story)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
