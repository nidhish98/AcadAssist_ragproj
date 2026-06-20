"""
loader.py
─────────
PDF loading layer for the RAKA pipeline.
Supports multiple document types: examination regulations + course syllabi.

Changes from v1
───────────────
• Added syllabus-specific boilerplate patterns (SPIT CSE curriculum headers).
• Added syllabus cover/TOC signal detection.
• Configurable boilerplate per doc_type — pass doc_type="auto" to auto-detect from filename.
• Syllabus footer "SPIT/UG Curriculum/.../pg.N" stripped automatically.
• doc_type stored on PageRecord for downstream use.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Boilerplate: examination regulations ─────────────────────────────────────
_EXAM_BOILERPLATE: list[str] = [
    r"Bharatiya\s+Vidya\s+Bhavan['\u2019]s",
    r"Sardar\s+Patel\s+Institute\s+of\s+Technology",
    r"Bhavan['\u2019]s\s+Campus,?\s*Munshi\s+Nagar",
    r"Andheri\s*\(West\),\s*Mumbai[- ]\d+",
    r"\(An\s+Autonomous\s+Institute\s+Affiliated\s+to\s+the\s+University\s+of\s+Mumbai\)",
    r"\(Autonomous\s+Institute\s+Affiliated\s+to\s+University\s+of\s+Mumbai\)",
    r"An\s+Autonomous\s+Institute\s+Affiliated\s+to",
    r"Empowered\s+Autonomous\s+Institute",
    r"\[Knowledge\s+is\s+Nectar\]",
    r"India\s*$",
]

# ── Boilerplate: CSE syllabus PDF ────────────────────────────────────────────
_SYLLABUS_BOILERPLATE: list[str] = [
    r"Bharatiya\s+Vidya\s+Bhavan['\u2019]s",
    r"Sardar\s+Patel\s+Institute\s+of\s+Technology",
    r"Bhavan['\u2019]s\s+Campus,?",
    r"Munshi\s+Nagar",
    r"Andheri\s*\(West\)",
    r"Mumbai[- ]\d+",
    r"\(Autonomous\s+Institute\s+Affiliated\s+to\s+University\s+of\s+Mumbai\)",
    r"\(An\s+Autonomous\s+Institute\s+Affiliated\to",
    r"\[Knowledge\s+is\s+Nectar\]",
    r"Liberal,\s*Pi-Model\s+of\s+Engineering\s+Education\s*@\s*SPIT",
    r"\(Department\s+of\s+Computer\s+Science\s+and\s+Engineering\)",
    r"SPIT/UG\s+Curriculum/\d{4}\s+Iteration/\w+/pg\.\d+",
]

_ALL_BOILERPLATE: list[str] = list(dict.fromkeys(_EXAM_BOILERPLATE + _SYLLABUS_BOILERPLATE))


def _build_re(patterns: list[str]) -> re.Pattern:
    return re.compile("|".join(patterns), re.IGNORECASE | re.MULTILINE)


_RE_EXAM     = _build_re(_EXAM_BOILERPLATE)
_RE_SYLLABUS = _build_re(_SYLLABUS_BOILERPLATE)
_RE_ALL      = _build_re(_ALL_BOILERPLATE)

# ── Section heading pattern ──────────────────────────────────────────────────
_SECTION_RE = re.compile(
    r"^(\d{1,2}(?:\.\d{1,2})*)[.\s]{1,4}"
    r"([A-Z][A-Za-z ,/\-&:()']{3,60})"
    r"(?:\s+\d{1,3}(?:-\d{1,3})?)?$",
    re.MULTILINE,
)

# ── Page type signals ────────────────────────────────────────────────────────
_TOC_EXAM     = {"page no", "particulars", "section", "preamble", "glossary"}
_TOC_SYLLABUS = {"curriculum structure", "table of contents", "semester-wise",
                 "sem i", "sem ii", "sem iii", "sem iv", "sem v", "sem vi"}
_COVER_EXAM   = {"examination regulations", "knowledge is nectar", "w.e.f."}
_COVER_SYLLABUS = {
    "liberal, pi-model", "department of computer science",
    "2023-2027 batch", "undergraduate academic programs",
    "effective from academic year",
}


# ── Data structures ──────────────────────────────────────────────────────────
@dataclass
class PageRecord:
    """One cleaned page extracted from a PDF."""
    page_num:   int
    raw_text:   str
    page_type:  str   # cover | toc | table | content
    section:    str
    source:     str   # filename stem
    doc_id:     str   # uuid shared across pages of same document
    doc_type:   str   # examination | syllabus | general
    has_table:  bool  = False
    table_text: str   = ""
    word_count: int   = field(init=False)

    def __post_init__(self) -> None:
        self.word_count = len(self.raw_text.split())


# ── Helpers ──────────────────────────────────────────────────────────────────
def _strip_boilerplate(text: str, boilerplate_re: re.Pattern) -> str:
    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            cleaned.append(line)
            continue
        if boilerplate_re.search(s):
            continue
        if re.fullmatch(r"\d{1,3}", s):
            continue
        if re.fullmatch(r"pg\.\s*\d+", s, re.IGNORECASE):
            continue
        if re.match(r"SPIT/UG\s+Curriculum", s, re.IGNORECASE):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _detect_page_type(text: str, doc_type: str) -> str:
    lower = text.lower()
    if not text or len(text.split()) < 10:
        return "cover"
    cover_sig = _COVER_SYLLABUS if doc_type == "syllabus" else _COVER_EXAM
    toc_sig   = _TOC_SYLLABUS   if doc_type == "syllabus" else _TOC_EXAM
    if any(sig in lower for sig in cover_sig) and len(text) < 700:
        return "cover"
    if any(sig in lower for sig in toc_sig) and len(text) < 2000:
        return "toc"
    if "table" in lower and ("|" in text or "\t" in text):
        return "table"
    return "content"


def _extract_tables_pdfplumber(page) -> tuple[bool, str]:
    try:
        tables = page.extract_tables()
        if not tables:
            return False, ""
        rows: list[str] = []
        for table in tables:
            for row in table:
                cells = [str(c).strip() if c else "" for c in row]
                non_empty = [c for c in cells if c and len(c) > 1]
                if non_empty:
                    rows.append(" | ".join(non_empty))
        table_text = "\n".join(rows)
        return bool(rows), table_text
    except Exception as exc:
        logger.debug("Table extraction failed: %s", exc)
        return False, ""


def _find_active_section(text: str, current_section: str) -> str:
    matches = _SECTION_RE.findall(text)
    if matches:
        num, title = matches[-1]
        return f"{num} {title.strip()}"
    return current_section


def _detect_doc_type(pdf_path: Path) -> str:
    name = pdf_path.stem.lower()
    if any(k in name for k in ("syll", "curriculum", "cse", "course", "subject")):
        return "syllabus"
    if any(k in name for k in ("exam", "regulation", "rule")):
        return "examination"
    return "general"


def _pdfplumber_load(
    pdf_path: Path, doc_id: str, doc_type: str, bp_re: re.Pattern
) -> list[PageRecord]:
    import pdfplumber
    pages: list[PageRecord] = []
    current_section = ""
    with pdfplumber.open(str(pdf_path)) as pdf:
        logger.info("pdfplumber: %d pages in '%s'", len(pdf.pages), pdf_path.name)
        for i, page in enumerate(pdf.pages, start=1):
            try:
                raw = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
                has_table, table_text = _extract_tables_pdfplumber(page)
                cleaned = _strip_boilerplate(raw, bp_re)
                if has_table and table_text:
                    cleaned = cleaned + "\n\n[TABLE]\n" + table_text + "\n[/TABLE]"
                ptype = _detect_page_type(cleaned, doc_type)
                current_section = _find_active_section(cleaned, current_section)
                pages.append(PageRecord(
                    page_num=i, raw_text=cleaned, page_type=ptype,
                    section=current_section, source=pdf_path.stem,
                    doc_id=doc_id, doc_type=doc_type,
                    has_table=has_table, table_text=table_text,
                ))
            except Exception as exc:
                logger.error("pdfplumber failed on page %d: %s", i, exc)
                pages.append(PageRecord(
                    page_num=i, raw_text="", page_type="cover",
                    section=current_section, source=pdf_path.stem,
                    doc_id=doc_id, doc_type=doc_type,
                ))
    return pages


def _pypdf_fallback(
    pdf_path: Path, doc_id: str, doc_type: str, bp_re: re.Pattern
) -> list[PageRecord]:
    from pypdf import PdfReader
    pages: list[PageRecord] = []
    current_section = ""
    reader = PdfReader(str(pdf_path))
    logger.info("pypdf fallback: %d pages in '%s'", len(reader.pages), pdf_path.name)
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
            cleaned = _strip_boilerplate(raw, bp_re)
            ptype = _detect_page_type(cleaned, doc_type)
            current_section = _find_active_section(cleaned, current_section)
            pages.append(PageRecord(
                page_num=i, raw_text=cleaned, page_type=ptype,
                section=current_section, source=pdf_path.stem,
                doc_id=doc_id, doc_type=doc_type,
            ))
        except Exception as exc:
            logger.error("pypdf failed on page %d: %s", i, exc)
    return pages


# ── Public API ───────────────────────────────────────────────────────────────
def load_pdf(pdf_path: str | Path, doc_type: str = "auto") -> list[PageRecord]:
    """
    Load and clean all pages from a PDF.

    Parameters
    ──────────
    pdf_path : path to the PDF.
    doc_type : "examination" | "syllabus" | "general" | "auto"
               "auto" detects from filename — recommended for most uses.

    Returns
    ───────
    List of PageRecord objects (cover/toc/empty pages excluded).

    Raises
    ──────
    FileNotFoundError  if path does not exist.
    RuntimeError       if no PDF library is available.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if doc_type == "auto":
        doc_type = _detect_doc_type(pdf_path)
        logger.info("Auto-detected doc_type='%s' for '%s'", doc_type, pdf_path.name)

    bp_re = _RE_SYLLABUS if doc_type == "syllabus" else (
            _RE_EXAM     if doc_type == "examination" else _RE_ALL)

    doc_id = str(uuid.uuid4())
    logger.info("Loading PDF: %s  [doc_id=%s  type=%s]", pdf_path.name, doc_id, doc_type)

    try:
        import pdfplumber  # noqa: F401
        pages = _pdfplumber_load(pdf_path, doc_id, doc_type, bp_re)
    except ImportError:
        logger.warning("pdfplumber not found — using pypdf fallback")
        try:
            pages = _pypdf_fallback(pdf_path, doc_id, doc_type, bp_re)
        except ImportError as exc:
            raise RuntimeError("Install pdfplumber: pip install pdfplumber") from exc

    content_pages = [
        p for p in pages
        if p.page_type not in ("cover", "toc") and p.word_count >= 15
    ]
    logger.info(
        "Loaded %d content pages (skipped %d cover/toc/empty) from '%s'",
        len(content_pages), len(pages) - len(content_pages), pdf_path.name,
    )
    return content_pages