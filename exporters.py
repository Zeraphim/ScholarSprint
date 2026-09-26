"""Summary export helpers.

A single Markdown source is parsed once into lightweight blocks, then rendered
into each supported download format. Keeping one parser in front of every
renderer means the exported Markdown, PDF, LaTeX, DOCX, and plain text all
describe the same document structure.

This module is intentionally free of Streamlit imports so it can be unit tested
without a running app.
"""

from __future__ import annotations

import importlib.util
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

MARKDOWN = "Markdown (.md)"
PDF = "PDF (.pdf)"
TEXT = "Plain Text (.txt)"
LATEX = "LaTeX (.tex)"
DOCX = "Word (.docx)"


@dataclass(frozen=True)
class ExportFormat:
    """Download metadata for one export target."""

    label: str
    extension: str
    mime: str
    requires: tuple[str, ...] = ()
    package_hint: str = ""


EXPORT_FORMATS: dict[str, ExportFormat] = {
    MARKDOWN: ExportFormat(MARKDOWN, ".md", "text/markdown"),
    PDF: ExportFormat(PDF, ".pdf", "application/pdf", ("fpdf",), "fpdf2"),
    TEXT: ExportFormat(TEXT, ".txt", "text/plain"),
    LATEX: ExportFormat(LATEX, ".tex", "application/x-tex"),
    DOCX: ExportFormat(
        DOCX,
        ".docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ("docx",),
        "python-docx",
    ),
}


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

HEADING = "heading"
PARAGRAPH = "paragraph"
BULLETS = "bullets"
NUMBERS = "numbers"
RULE = "rule"
CODE = "code"


@dataclass(frozen=True)
class Span:
    """An inline run of text with its formatting."""

    text: str
    bold: bool = False
    italic: bool = False
    code: bool = False
    link: str = ""


@dataclass
class Block:
    """A block-level element of the document."""

    kind: str
    level: int = 0
    spans: list[Span] = field(default_factory=list)
    items: list[list[Span]] = field(default_factory=list)
    text: str = ""

    def plain_text(self) -> str:
        if self.kind == CODE:
            return self.text
        return "".join(span.text for span in self.spans)


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_RULE_RE = re.compile(r"^(-{3,}|\*{3,}|_{3,})$")
_BULLET_RE = re.compile(r"^[-*+]\s+(.*)$")
_ORDERED_RE = re.compile(r"^\d+[.)]\s+(.*)$")
_QUOTE_RE = re.compile(r"^>\s?(.*)$")

_INLINE_RE = re.compile(
    r"(?P<code>`+[^`]+?`+)"
    r"|(?P<link>\[(?P<link_text>[^\]]*)\]\((?P<link_url>[^)\s]+)\))"
    r"|(?P<bolditalic>\*\*\*[^*]+?\*\*\*|___[^_]+?___)"
    r"|(?P<bold>\*\*[^*]+?\*\*|__[^_]+?__)"
    r"|(?P<italic>\*[^*]+?\*|_[^_]+?_)"
)


def parse_inline(text: str) -> list[Span]:
    """Split one line of Markdown into formatted spans.

    Nesting beyond bold-italic is not supported; unmatched markers are kept as
    literal characters so nothing is silently dropped.
    """
    spans: list[Span] = []
    cursor = 0

    for match in _INLINE_RE.finditer(text):
        if match.start() > cursor:
            spans.append(Span(text[cursor : match.start()]))

        if match.group("code"):
            spans.append(Span(match.group("code").strip("`"), code=True))
        elif match.group("link"):
            spans.append(
                Span(match.group("link_text"), link=match.group("link_url"))
            )
        elif match.group("bolditalic"):
            spans.append(Span(match.group("bolditalic").strip("*_"), bold=True, italic=True))
        elif match.group("bold"):
            spans.append(Span(match.group("bold").strip("*_"), bold=True))
        else:
            spans.append(Span(match.group("italic").strip("*_"), italic=True))

        cursor = match.end()

    if cursor < len(text):
        spans.append(Span(text[cursor:]))

    return [span for span in spans if span.text]


def parse_markdown(markdown_text: str) -> list[Block]:
    """Parse Markdown into the block list shared by every renderer."""
    lines = (markdown_text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[Block] = []
    index = 0

    while index < len(lines):
        line = lines[index].strip()

        if not line:
            index += 1
            continue

        if line.startswith("```"):
            index += 1
            body: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                body.append(lines[index])
                index += 1
            index += 1  # closing fence
            blocks.append(Block(CODE, text="\n".join(body)))
            continue

        if _RULE_RE.match(line):
            blocks.append(Block(RULE))
            index += 1
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            blocks.append(
                Block(HEADING, level=len(heading.group(1)), spans=parse_inline(heading.group(2).strip()))
            )
            index += 1
            continue

        if _BULLET_RE.match(line) or _ORDERED_RE.match(line):
            ordered = bool(_ORDERED_RE.match(line))
            pattern = _ORDERED_RE if ordered else _BULLET_RE
            items: list[list[Span]] = []
            while index < len(lines):
                candidate = lines[index].strip()
                item = pattern.match(candidate)
                if not item:
                    break
                items.append(parse_inline(item.group(1).strip()))
                index += 1
            blocks.append(Block(NUMBERS if ordered else BULLETS, items=items))
            continue

        paragraph: list[str] = []
        while index < len(lines):
            candidate = lines[index].strip()
            if (
                not candidate
                or candidate.startswith("```")
                or _RULE_RE.match(candidate)
                or _HEADING_RE.match(candidate)
                or _BULLET_RE.match(candidate)
                or _ORDERED_RE.match(candidate)
            ):
                break
            quote = _QUOTE_RE.match(candidate)
            paragraph.append(quote.group(1).strip() if quote else candidate)
            index += 1

        joined = " ".join(part for part in paragraph if part)
        if joined:
            blocks.append(Block(PARAGRAPH, spans=parse_inline(joined)))

    return _normalize_heading_levels(blocks)


def _normalize_heading_levels(blocks: list[Block]) -> list[Block]:
    """Shift heading levels so the shallowest heading becomes level 1."""
    levels = [block.level for block in blocks if block.kind == HEADING]
    if not levels:
        return blocks

    offset = min(levels) - 1
    if offset <= 0:
        return blocks

    for block in blocks:
        if block.kind == HEADING:
            block.level = max(1, block.level - offset)
    return blocks


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #


def render_markdown(markdown_text: str, title: str) -> str:
    """Return the Markdown document with a level-1 title."""
    body = (markdown_text or "").strip()
    heading = f"# {title}".strip()
    return f"{heading}\n\n{body}\n" if body else f"{heading}\n"


def render_text(blocks: list[Block], title: str) -> str:
    """Flatten the document into readable plain text without Markdown syntax."""
    lines: list[str] = []
    if title:
        lines.extend([title, "=" * len(title), ""])

    for block in blocks:
        if block.kind == HEADING:
            text = block.plain_text()
            lines.extend([text, ("-" if block.level > 1 else "=") * max(len(text), 3), ""])
        elif block.kind == PARAGRAPH:
            lines.extend([block.plain_text(), ""])
        elif block.kind == BULLETS:
            lines.extend(f"- {''.join(span.text for span in item)}" for item in block.items)
            lines.append("")
        elif block.kind == NUMBERS:
            lines.extend(
                f"{number}. {''.join(span.text for span in item)}"
                for number, item in enumerate(block.items, start=1)
            )
            lines.append("")
        elif block.kind == RULE:
            lines.extend(["-" * 60, ""])
        elif block.kind == CODE:
            lines.extend([*block.text.splitlines(), ""])

    return "\n".join(lines).strip() + "\n"


_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    # Symbols pdflatex cannot typeset from raw UTF-8 input.
    "\u2013": "--",
    "\u2014": "---",
    "\u2018": "`",
    "\u2019": "'",
    "\u201c": "``",
    "\u201d": "''",
    "\u2026": r"\ldots{}",
    "\u2022": r"$\bullet$",
    "\u2192": r"$\rightarrow$",
    "\u2190": r"$\leftarrow$",
    "\u2264": r"$\leq$",
    "\u2265": r"$\geq$",
    "\u2248": r"$\approx$",
    "\u2260": r"$\neq$",
    "\u00b1": r"$\pm$",
    "\u00d7": r"$\times$",
    "\u00f7": r"$\div$",
    "\u2212": "-",
    "\u221e": r"$\infty$",
    "\u00b0": r"\textdegree{}",
    "\u2032": r"$'$",
    "\u00a0": "~",
    "\u200b": "",
    "\u03b1": r"$\alpha$",
    "\u03b2": r"$\beta$",
    "\u03b3": r"$\gamma$",
    "\u03b4": r"$\delta$",
    "\u03b5": r"$\epsilon$",
    "\u03b8": r"$\theta$",
    "\u03bb": r"$\lambda$",
    "\u03bc": r"$\mu$",
    "\u00b5": r"$\mu$",
    "\u03c0": r"$\pi$",
    "\u03c1": r"$\rho$",
    "\u03c3": r"$\sigma$",
    "\u03c4": r"$\tau$",
    "\u03c6": r"$\phi$",
    "\u03c7": r"$\chi$",
    "\u03a9": r"$\Omega$",
    "\u0394": r"$\Delta$",
    "\u03a3": r"$\Sigma$",
}

_LATEX_SECTIONS = ("section", "subsection", "subsubsection", "paragraph", "subparagraph")


def escape_latex(text: str) -> str:
    """Escape the characters LaTeX treats as markup."""
    return "".join(_LATEX_ESCAPES.get(character, character) for character in text)


def _latex_spans(spans: list[Span]) -> str:
    parts: list[str] = []
    for span in spans:
        if span.code:
            parts.append(rf"\texttt{{{escape_latex(span.text)}}}")
            continue

        text = escape_latex(span.text)
        if span.bold and span.italic:
            text = rf"\textbf{{\textit{{{text}}}}}"
        elif span.bold:
            text = rf"\textbf{{{text}}}"
        elif span.italic:
            text = rf"\textit{{{text}}}"
        if span.link:
            text = rf"\href{{{span.link}}}{{{text}}}"
        parts.append(text)
    return "".join(parts)


def render_latex(blocks: list[Block], title: str) -> str:
    """Render a standalone LaTeX article."""
    body: list[str] = []

    for block in blocks:
        if block.kind == HEADING:
            command = _LATEX_SECTIONS[min(block.level, len(_LATEX_SECTIONS)) - 1]
            body.append(f"\\{command}{{{_latex_spans(block.spans)}}}")
        elif block.kind == PARAGRAPH:
            body.append(_latex_spans(block.spans))
        elif block.kind in (BULLETS, NUMBERS):
            environment = "itemize" if block.kind == BULLETS else "enumerate"
            items = "\n".join(f"  \\item {_latex_spans(item)}" for item in block.items)
            body.append(f"\\begin{{{environment}}}\n{items}\n\\end{{{environment}}}")
        elif block.kind == RULE:
            body.append(r"\begin{center}\rule{0.9\linewidth}{0.4pt}\end{center}")
        elif block.kind == CODE:
            body.append("\\begin{verbatim}\n" + block.text + "\n\\end{verbatim}")

    preamble = "\n".join(
        [
            r"\documentclass[11pt,a4paper]{article}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{hyperref}",
            r"\setlength{\parskip}{0.6em}",
            r"\setlength{\parindent}{0pt}",
            f"\\title{{{escape_latex(title)}}}",
            r"\author{}",
            r"\date{}",
            r"\begin{document}",
            r"\maketitle",
        ]
    )
    return f"{preamble}\n\n" + "\n\n".join(body) + "\n\n" + r"\end{document}" + "\n"


# Core PDF fonts are Latin-1 only, so normalize the punctuation models emit.
_PDF_REPLACEMENTS = {
    "\u2013": "-",
    "\u2014": "--",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u2022": "-",
    "\u2192": "->",
    "\u2190": "<-",
    "\u2264": "<=",
    "\u2265": ">=",
    "\u2248": "~",
    "\u2260": "!=",
    "\u221e": "inf",
    "\u2211": "sum",
    "\u221a": "sqrt",
    "\u00d7": "x",
    "\u2212": "-",
    "\u2032": "'",
    "\u00a0": " ",
    "\u2009": " ",
    "\u202f": " ",
    "\u200b": "",
    "\u2011": "-",
    "\u2122": "(TM)",
    "\u03b1": "alpha",
    "\u03b2": "beta",
    "\u03b3": "gamma",
    "\u03b4": "delta",
    "\u03b5": "epsilon",
    "\u03b8": "theta",
    "\u03bb": "lambda",
    "\u03bc": "mu",
    "\u03c0": "pi",
    "\u03c1": "rho",
    "\u03c3": "sigma",
    "\u03c4": "tau",
    "\u03c6": "phi",
    "\u03c7": "chi",
    "\u0394": "Delta",
    "\u03a3": "Sigma",
    "\u03a9": "Omega",
}


def pdf_safe_text(text: str) -> str:
    """Make text representable by the Latin-1 core PDF fonts."""
    for source, target in _PDF_REPLACEMENTS.items():
        text = text.replace(source, target)
    return text.encode("latin-1", "replace").decode("latin-1")


_PDF_HEADING_SIZES = {1: 15.0, 2: 13.0, 3: 11.5, 4: 11.0}
_PDF_BODY_SIZE = 10.5


def render_pdf(blocks: list[Block], title: str) -> bytes:
    """Render the document to PDF using the pure-Python fpdf2 backend."""
    from fpdf import FPDF

    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(20, 18, 20)
    pdf.set_title(pdf_safe_text(title))
    pdf.add_page()

    left_margin = pdf.l_margin

    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 9, pdf_safe_text(title))
    pdf.ln(4)

    for block in blocks:
        if block.kind == HEADING:
            size = _PDF_HEADING_SIZES.get(block.level, 10.5)
            pdf.ln(2)
            _pdf_write_spans(pdf, block.spans, size, base_style="B")
            pdf.ln(size * 0.62)
        elif block.kind == PARAGRAPH:
            _pdf_write_spans(pdf, block.spans, _PDF_BODY_SIZE)
            pdf.ln(_PDF_BODY_SIZE * 0.62)
        elif block.kind in (BULLETS, NUMBERS):
            pdf.set_left_margin(left_margin + 6)
            for number, item in enumerate(block.items, start=1):
                pdf.set_x(left_margin + 6)
                marker = f"{number}. " if block.kind == NUMBERS else "- "
                _pdf_write_spans(pdf, [Span(marker), *item], _PDF_BODY_SIZE)
                pdf.ln(_PDF_BODY_SIZE * 0.52)
            pdf.set_left_margin(left_margin)
            pdf.ln(2)
        elif block.kind == RULE:
            pdf.ln(2)
            pdf.line(left_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(4)
        elif block.kind == CODE:
            pdf.set_font("Courier", "", 9.5)
            pdf.multi_cell(0, 4.6, pdf_safe_text(block.text))
            pdf.ln(2)

    return bytes(pdf.output())


def _pdf_write_spans(pdf, spans: list[Span], size: float, base_style: str = "") -> None:
    line_height = size * 0.52
    for span in spans:
        style = set(base_style)
        if span.bold:
            style.add("B")
        if span.italic:
            style.add("I")
        pdf.set_font("Courier" if span.code else "Helvetica", "".join(sorted(style)), size)
        pdf.write(line_height, pdf_safe_text(span.text), span.link or "")
    pdf.ln(line_height)


def render_docx(blocks: list[Block], title: str) -> bytes:
    """Render the document to DOCX using python-docx."""
    from docx import Document
    from docx.shared import Pt

    document = Document()
    document.core_properties.title = title
    document.add_heading(title, level=0)

    for block in blocks:
        if block.kind == HEADING:
            paragraph = document.add_heading("", level=min(block.level, 4))
            _docx_add_spans(paragraph, block.spans)
        elif block.kind == PARAGRAPH:
            _docx_add_spans(document.add_paragraph(), block.spans)
        elif block.kind in (BULLETS, NUMBERS):
            style = "List Bullet" if block.kind == BULLETS else "List Number"
            for item in block.items:
                _docx_add_spans(_docx_paragraph(document, style), item)
        elif block.kind == RULE:
            _docx_horizontal_rule(document.add_paragraph())
        elif block.kind == CODE:
            paragraph = document.add_paragraph()
            run = paragraph.add_run(block.text)
            run.font.name = "Courier New"
            run.font.size = Pt(9.5)

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _docx_paragraph(document, style: str):
    """Add a styled paragraph, falling back when a template lacks the style."""
    try:
        return document.add_paragraph(style=style)
    except KeyError:
        return document.add_paragraph()


def _docx_add_spans(paragraph, spans: list[Span]) -> None:
    for span in spans:
        text = f"{span.text} ({span.link})" if span.link else span.text
        run = paragraph.add_run(text)
        run.bold = span.bold or None
        run.italic = span.italic or None
        if span.code:
            run.font.name = "Courier New"


def _docx_horizontal_rule(paragraph) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "auto")
    borders.append(bottom)
    paragraph._p.get_or_add_pPr().append(borders)


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def missing_dependency(format_label: str) -> str | None:
    """Return the pip package needed for a format, or None when it is ready."""
    export_format = EXPORT_FORMATS[format_label]
    for module in export_format.requires:
        if importlib.util.find_spec(module) is None:
            return export_format.package_hint or module
    return None


def available_formats() -> list[str]:
    """Return the formats whose optional dependencies are installed."""
    return [label for label in EXPORT_FORMATS if missing_dependency(label) is None]


def export_file_name(source_name: str, format_label: str) -> str:
    """Build a download name from the source file name and target format."""
    stem = Path(source_name or "summary").stem or "summary"
    stem = re.sub(r"[^\w.\-]+", "_", stem).strip("_") or "summary"
    return f"{stem}{EXPORT_FORMATS[format_label].extension}"


def build_export(markdown_text: str, title: str, format_label: str) -> bytes:
    """Render the summary Markdown into the requested format's bytes."""
    if format_label not in EXPORT_FORMATS:
        raise ValueError(f"Unsupported export format: {format_label}")

    package = missing_dependency(format_label)
    if package:
        raise ModuleNotFoundError(f"{format_label} export requires the '{package}' package")

    if format_label == MARKDOWN:
        return render_markdown(markdown_text, title).encode("utf-8")

    blocks = parse_markdown(markdown_text)

    if format_label == TEXT:
        return render_text(blocks, title).encode("utf-8")
    if format_label == LATEX:
        return render_latex(blocks, title).encode("utf-8")
    if format_label == PDF:
        return render_pdf(blocks, title)
    return render_docx(blocks, title)
