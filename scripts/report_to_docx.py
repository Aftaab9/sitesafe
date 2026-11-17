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
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
BLACK = RGBColor(0, 0, 0)


def add_runs(par, text: str):
    """Supports **bold**, *italic* / _italic_ and `code` inline."""
    for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`|_[^_]+_|\*[^*]+\*)", text):
        if not part:
            continue
        if part.startswith("**"):
            r = par.add_run(part[2:-2]); r.bold = True
        elif part.startswith("`"):
            r = par.add_run(part[1:-1]); r.font.name = "Consolas"
        elif (part.startswith("_") and part.endswith("_")) or (part.startswith("*") and part.endswith("*")):
            r = par.add_run(part[1:-1]); r.italic = True
        else:
            r = par.add_run(part)
        r.font.color.rgb = BLACK


def convert(md: Path, out: Path):
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(11)
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
            t = doc.add_table(rows=len(cells), cols=len(cells[0]))
            t.style = "Table Grid"
            for r, row in enumerate(cells):
                for c, val in enumerate(row[: len(cells[0])]):
                    p = t.cell(r, c).paragraphs[0]
                    add_runs(p, val)
                    if r == 0:
                        for run in p.runs:
                            run.bold = True
            doc.add_paragraph()
            continue
        m = re.match(r"^(#{1,3})\s+(.*)", ln)
        if m:
            level = len(m.group(1))
            if level == 1:
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(m.group(2)); r.bold = True; r.font.size = Pt(18); r.font.color.rgb = BLACK
            else:
                doc.add_heading(m.group(2), level=level - 1)
        elif re.match(r"^\s*[-*]\s+", ln):
            add_runs(doc.add_paragraph(style="List Bullet"), re.sub(r"^\s*[-*]\s+", "", ln))
        elif re.match(r"^\s*\d+\.\s+", ln):
            add_runs(doc.add_paragraph(style="List Number"), re.sub(r"^\s*\d+\.\s+", "", ln))
        elif ln.strip():
            para = [ln.strip()]
            while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#|\||\s*[-*]\s|\s*\d+\.\s)", lines[i + 1]):
                i += 1; para.append(lines[i].strip())
            add_runs(doc.add_paragraph(), " ".join(para))
        i += 1
    doc.save(out)
    words = len(re.findall(r"\w+", md.read_text(encoding="utf-8")))
    print(f"{out.name}: about {words} words")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["report", "model_card", "week9_proposal"]:
        convert(ROOT / "docs" / f"{name}.md", ROOT / "docs" / f"{name}.docx")
