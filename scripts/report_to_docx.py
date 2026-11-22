"""Convert docs/*.md (report, model card) to plain, editable Word files.

    python scripts/report_to_docx.py

Black text only, simple headings, Calibri 11, tables with a plain grid. Easy to edit in Word.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
BLACK = RGBColor(0, 0, 0)


def add_runs(par, text: str):
    """Supports **bold**, *italic* / _italic_ and `code` inline."""
    # Underscores are NOT italics here: class names such as no_boots / no_vest must stay as written.
    for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)", text):
        if not part:
            continue
        if part.startswith("**"):
            r = par.add_run(part[2:-2]); r.bold = True
        elif part.startswith("`"):
            r = par.add_run(part[1:-1]); r.font.name = "Consolas"
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            r = par.add_run(part[1:-1]); r.italic = True
        else:
            r = par.add_run(part)
        r.font.color.rgb = BLACK


def convert(md: Path, out: Path, compact: bool = False):
    """compact=True: narrow margins and 9.5 pt text, for the one-page model card."""
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(9.5 if compact else 11)
    if compact:
        from docx.shared import Cm
        for sec in doc.sections:
            sec.top_margin = sec.bottom_margin = Cm(1.3); sec.left_margin = sec.right_margin = Cm(1.6)
        st.paragraph_format.space_after = Pt(3)
    st.font.color.rgb = BLACK
    for lvl in (1, 2, 3):
        h = doc.styles[f"Heading {lvl}"]
        h.font.color.rgb = BLACK
        h.font.name = "Calibri"
    lines = md.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1]):
            rows = [ln]
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i]); i += 1
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
            header = not all(not c for c in cells[0])
            if not header:  # "| | |" header used only to make a two-column list
                cells = cells[1:]
            t = doc.add_table(rows=len(cells), cols=len(cells[0]))
            t.style = "Table Grid"
            for r, row in enumerate(cells):
                for c, val in enumerate(row[: len(cells[0])]):
                    p = t.cell(r, c).paragraphs[0]
                    add_runs(p, val)
                    if r == 0 and header:
                        for run in p.runs:
                            run.bold = True
            doc.add_paragraph()
            continue
        img = re.match(r"^!\[(.*)\]\((.*)\)\s*$", ln)
        if img:  # a figure: picture plus a short caption (captions are not part of the body word count)
            path = ROOT / img.group(2)
            if path.exists():
                doc.add_picture(str(path), width=Inches(6.0))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                cap = doc.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = cap.add_run(img.group(1)); r.italic = True; r.font.size = Pt(9); r.font.color.rgb = BLACK
            else:
                print("missing figure", path)
            i += 1
            continue
        m = re.match(r"^(#{1,3})\s+(.*)", ln)
        if m:
            level = len(m.group(1))
            if level == 1:
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(m.group(2)); r.bold = True; r.font.size = Pt(14 if compact else 18); r.font.color.rgb = BLACK
            else:
                doc.add_heading(m.group(2), level=level - 1)
        elif re.match(r"^\s*[-*]\s+", ln):
            add_runs(doc.add_paragraph(style="List Bullet"), re.sub(r"^\s*[-*]\s+", "", ln))
        elif re.match(r"^\s*\d+\.\s+", ln):
            add_runs(doc.add_paragraph(style="List Number"), re.sub(r"^\s*\d+\.\s+", "", ln))
        elif ln.strip():
            para = [ln.strip()]
            while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#|\||!\[|\s*[-*]\s|\s*\d+\.\s)", lines[i + 1]):
                i += 1; para.append(lines[i].strip())
            add_runs(doc.add_paragraph(), " ".join(para))
        i += 1
    doc.save(out)
    text = md.read_text(encoding="utf-8")
    if "## References" in text and "## 1." in text:
        text = text[text.index("## 1."):text.index("## References")]
    text = "\n".join(l for l in text.splitlines() if not l.startswith("!["))
    words = len(re.findall(r"\S+", text))
    print(f"{out.name}: {words} words (body, without captions and references)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 2 and args[0].endswith(".md"):  # explicit: input.md output.docx [--compact]
        convert(ROOT / args[0], ROOT / args[1], compact="--compact" in args)
    else:
        for name in args or ["report", "model_card", "week9_proposal"]:
            convert(ROOT / "docs" / f"{name}.md", ROOT / "docs" / f"{name}.docx")
