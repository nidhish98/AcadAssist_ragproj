"""
chunker.py
──────────
Text chunking layer for the RAKA pipeline.

Responsibilities
────────────────
• Accept a list of PageRecord objects from loader.py.
• Split page text into semantically coherent chunks of 300–500 words
  (PRD §4.1 — Document Processing Module).
• Respect sentence boundaries — never split mid-sentence.
• Maintain a 50-word sliding overlap between consecutive chunks to
  preserve cross-boundary context for retrieval.
• Carry rich metadata on every chunk: source, section, page range,
  word count, chunk index.
• Skip chunks that are too short (< MIN_WORDS) or appear to be
  pure boilerplate / table-of-contents fragments.
• Deduplicate chunks by content hash (handles duplicate pages).
• Tables embedded by loader.py ([TABLE]...[/TABLE]) are kept intact
  inside a single chunk and never split.

Design decisions (aligned with PRD)
─────────────────────────────────────
• Sentence splitting uses a simple regex tokeniser — no NLTK dependency
  required, making the module self-contained.
• Section headings found mid-page reset the "active section" for
  subsequent chunks on that page.
• Every chunk gets a stable UUID derived from (doc_id + chunk_index)
  for FAISS metadata mapping.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from loader import PageRecord

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
CHUNK_TARGET_WORDS = 400    # ideal chunk size (words)
CHUNK_MAX_WORDS    = 480    # hard ceiling before forced split
CHUNK_MIN_WORDS    = 50     # chunks shorter than this are discarded
OVERLAP_WORDS      = 50     # overlap between consecutive chunks
SENTENCE_DELIMITERS = re.compile(r'(?<=[.!?])\s+(?=[A-Z\d])')  # basic sentence splitter

# Regex that identifies lines which are almost certainly structural noise
_NOISE_LINE_RE = re.compile(
    r"^\s*("
    r"\d{1,3}\s*$"                          # standalone page number
    r"|[ivxlcdmIVXLCDM]+\s*$"              # roman numeral page
    r"|\.{4,}"                              # dot leaders (TOC)
    r"|[-─═]{4,}"                           # horizontal rule
    r"|\s*$"                                # blank line
    r")",
    re.MULTILINE,
)

# Section heading pattern — numbered headings like "2.3 Attendance Requirements"
# Must be on its own line, start with a digit, and the title must be >= 4 chars.
# Explicitly excludes lines that look like table rows (contain | or tabs).
_SECTION_HEADING_RE = re.compile(
    r"^(\d{1,2}(?:\.\d{1,2})*)\s{1,4}"
    r"([A-Z][A-Za-z ,/\-&:()']{3,60})"
    r"(?:\s+\d{1,3}(?:-\d{1,3})?)?$",   # optional trailing page-number
    re.MULTILINE,
)


# ── Data structures ────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    """
    A single retrievable text unit.

    All fields populated before the chunk is stored in FAISS metadata.
    """
    chunk_id:   str          # UUID (stable: derived from doc_id + index)
    doc_id:     str          # UUID shared by all chunks from one document
    chunk_index: int         # 0-based position in the document
    text:       str          # clean, complete chunk text
    source:     str          # filename stem (e.g. "examination")
    section:    str          # active section heading at this chunk
    page_start: int          # first page number contributing to this chunk
    page_end:   int          # last page number contributing to this chunk
    word_count: int          # word count of chunk text
    category:   str = "examination"  # document category tag
    content_hash: str = field(init=False)  # for deduplication

    def __post_init__(self) -> None:
        self.content_hash = hashlib.md5(self.text.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "chunk_id":    self.chunk_index,   # integer index (matches PRD schema)
            "uuid":        self.chunk_id,
            "doc_id":      self.doc_id,
            "text":        self.text,
            "page_start":  self.page_start,
            "page_end":    self.page_end,
            "source":      self.source + ".pdf",
            "section":     self.section,
            "category":    self.category,
            "word_count":  self.word_count,
        }


# ── Sentence tokeniser ─────────────────────────────────────────────────────────
def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using regex.  Handles:
    • Abbreviations (Rs., B.Tech., M.Tech., U.G.) — avoids false splits.
    • Numbered list items (a) b) c)).
    • Sentence-ending punctuation followed by capital letter or digit.
    """
    # Protect known abbreviations from being treated as sentence ends
    abbrevs = [
        r"\bRs\.",
        r"\bB\.Tech\.",
        r"\bM\.Tech\.",
        r"\bMCA\b",
        r"\bU\.G\.",
        r"\bP\.G\.",
        r"\bviz\.",
        r"\betc\.",
        r"\bi\.e\.",
        r"\be\.g\.",
        r"\bw\.e\.f\.",
        r"\bA\.Y\.",
        r"\bH\.O\.D\.",
        r"\bNo\.",
        r"\bSr\.",
        r"\bFig\.",
        r"\bDr\.",
        r"\bProf\.",
        r"\bArt\.",
    ]
    placeholder_map: dict[str, str] = {}
    protected = text
    for pattern in abbrevs:
        for match in re.finditer(pattern, protected):
            original = match.group(0)
            token = original.replace(".", "<<<DOT>>>")
            placeholder_map[token] = original
            protected = protected.replace(original, token, 1)

    # Split on sentence boundaries
    parts = SENTENCE_DELIMITERS.split(protected)

    # Restore placeholders and clean up
    sentences: list[str] = []
    for part in parts:
        for token, original in placeholder_map.items():
            part = part.replace(token, original)
        part = part.strip()
        if part:
            sentences.append(part)

    return sentences


# ── Pre-processing helpers ─────────────────────────────────────────────────────
def _clean_page_text(text: str) -> str:
    """
    Remove noise lines from a single page's text.
    Preserves section headings and table blocks.
    """
    # Collapse multiple blank lines to one
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are pure noise
    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        if _NOISE_LINE_RE.match(line):
            continue
        # Remove Unicode private-use bullets (e.g. \uf06c list bullets)
        line = re.sub(r"[\uf000-\uf8ff]", "•", line)
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _is_junk_chunk(text: str) -> bool:
    """Return True if the chunk is too noisy to be useful for retrieval."""
    words = text.split()
    if len(words) < CHUNK_MIN_WORDS:
        return True
    # Mostly non-alphabetic (e.g. pure table noise)
    alpha_ratio = sum(1 for w in words if re.search(r"[a-zA-Z]", w)) / len(words)
    if alpha_ratio < 0.40:
        return True
    return False


# ── Table block handling ───────────────────────────────────────────────────────
def _extract_table_blocks(text: str) -> tuple[str, list[tuple[str, str]]]:
    """
    Pull [TABLE]...[/TABLE] blocks out of text.
    Returns (text_without_tables, [(placeholder, table_text), ...]).
    Allows tables to be re-injected into the chunk that is active at
    the placeholder position.
    """
    pattern = re.compile(r"\[TABLE\](.*?)\[/TABLE\]", re.DOTALL)
    tables: list[tuple[str, str]] = []
    counter = [0]

    def replacer(m: re.Match) -> str:
        ph = f"<<<TABLE_{counter[0]}>>>"
        tables.append((ph, m.group(1).strip()))
        counter[0] += 1
        return f"\n{ph}\n"

    cleaned = pattern.sub(replacer, text)
    return cleaned, tables


# ── Core chunking logic ────────────────────────────────────────────────────────
def _chunk_sentences(
    sentences: list[str],
    page_start: int,
    page_end: int,
    doc_id: str,
    source: str,
    section: str,
    chunk_index_offset: int,
    table_map: dict[str, str],
) -> list[Chunk]:
    """
    Group sentences into chunks respecting CHUNK_TARGET_WORDS / CHUNK_MAX_WORDS.
    Consecutive chunks share OVERLAP_WORDS of context.
    """
    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_words = 0
    overlap_buffer: list[str] = []

    def _flush(sent_list: list[str], idx: int) -> Optional[Chunk]:
        raw_text = " ".join(sent_list).strip()
        # Re-inject any table placeholders
        for ph, table_content in table_map.items():
            raw_text = raw_text.replace(ph, f"\n{table_content}\n")
        raw_text = raw_text.strip()
        if _is_junk_chunk(raw_text):
            logger.debug("Discarded junk chunk (index offset %d)", idx)
            return None
        chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}:{idx}"))
        return Chunk(
            chunk_id=chunk_uuid,
            doc_id=doc_id,
            chunk_index=chunk_index_offset + idx,
            text=raw_text,
            source=source,
            section=section,
            page_start=page_start,
            page_end=page_end,
            word_count=len(raw_text.split()),
        )

    idx = 0
    for sentence in sentences:
        word_count = len(sentence.split())

        # If a single sentence exceeds CHUNK_MAX_WORDS, split it at commas/semicolons
        if word_count > CHUNK_MAX_WORDS:
            sub_parts = re.split(r"[,;]\s+", sentence)
            for sub in sub_parts:
                sub_wc = len(sub.split())
                if current_words + sub_wc > CHUNK_MAX_WORDS and current_sentences:
                    chunk = _flush(current_sentences, idx)
                    if chunk:
                        chunks.append(chunk)
                        idx += 1
                    # Build overlap from end of current buffer
                    overlap_buffer = []
                    ow = 0
                    for s in reversed(current_sentences):
                        sw = len(s.split())
                        if ow + sw <= OVERLAP_WORDS:
                            overlap_buffer.insert(0, s)
                            ow += sw
                        else:
                            break
                    current_sentences = overlap_buffer.copy()
                    current_words = sum(len(s.split()) for s in current_sentences)
                current_sentences.append(sub)
                current_words += sub_wc
            continue

        # Normal sentence: add to buffer
        if current_words + word_count > CHUNK_MAX_WORDS and current_sentences:
            chunk = _flush(current_sentences, idx)
            if chunk:
                chunks.append(chunk)
                idx += 1
            # Build overlap
            overlap_buffer = []
            ow = 0
            for s in reversed(current_sentences):
                sw = len(s.split())
                if ow + sw <= OVERLAP_WORDS:
                    overlap_buffer.insert(0, s)
                    ow += sw
                else:
                    break
            current_sentences = overlap_buffer.copy()
            current_words = sum(len(s.split()) for s in current_sentences)

        current_sentences.append(sentence)
        current_words += word_count

    # Flush the last buffer
    if current_sentences:
        chunk = _flush(current_sentences, idx)
        if chunk:
            chunks.append(chunk)

    return chunks


# ── Public API ─────────────────────────────────────────────────────────────────
def chunk_pages(
    pages: list[PageRecord],
    category: str = "examination",
) -> list[Chunk]:
    """
    Convert a list of PageRecord objects into retrieval-ready Chunk objects.

    Strategy
    ────────
    1. All pages are concatenated in order with page-break markers so that
       the chunker can span chunks across page boundaries (fixing the
       "The teacher willing to change the" broken-chunk problem).
    2. Section headings found in the text update the active section tag
       for all subsequent chunks.
    3. Tables are extracted, replaced with placeholders, then re-injected
       into whichever chunk spans their position.
    4. Junk and duplicate chunks are discarded.

    Parameters
    ──────────
    pages    : output of loader.load_pdf()
    category : document category label for all chunks

    Returns
    ───────
    List of Chunk objects, deduplicated, indexed from 0.
    """
    if not pages:
        logger.warning("chunk_pages: received empty page list")
        return []

    doc_id = pages[0].doc_id
    source = pages[0].source

    # ── Step 1: Build a full document text with page markers ──────────────────
    # This is the key fix: we merge pages before chunking so sentences that
    # span page boundaries (like "The teacher willing to change the\n<next page>
    # evaluation weightages will apply to HoD...") get properly joined.
    page_spans: list[tuple[str, int]] = []  # (text_segment, page_num)
    full_segments: list[str] = []

    current_section = pages[0].section
    all_table_maps: dict[str, str] = {}
    segment_page_map: list[int] = []  # which page each char range starts on

    merged_text_parts: list[str] = []
    for page in pages:
        cleaned = _clean_page_text(page.raw_text)
        if not cleaned:
            continue
        # Update section from page metadata (set by loader)
        if page.section:
            current_section = page.section
        merged_text_parts.append(cleaned)
        page_spans.append((cleaned, page.page_num))

    full_text = "\n\n".join(merged_text_parts)

    # ── Step 2: Extract table blocks ──────────────────────────────────────────
    full_text_no_tables, table_list = _extract_table_blocks(full_text)
    table_map = {ph: tbl for ph, tbl in table_list}

    # ── Step 3: Sentence-split the full document text ─────────────────────────
    sentences = _split_sentences(full_text_no_tables)
    logger.debug("Total sentences after merge: %d", len(sentences))

    # ── Step 4: Group sentences into chunks ───────────────────────────────────
    # We track which page each sentence likely came from by scanning for
    # the page boundary markers. Since we don't have character offsets, we
    # use a heuristic: divide page numbers across sentences proportionally.
    n_pages = len(pages)
    n_sentences = len(sentences)
    page_nums = [p.page_num for p in pages]

    def _sentence_page(sent_idx: int) -> int:
        """Approximate page number for sentence at sent_idx."""
        if n_sentences == 0 or n_pages == 0:
            return page_nums[0]
        ratio = sent_idx / max(n_sentences - 1, 1)
        page_idx = min(int(ratio * n_pages), n_pages - 1)
        return page_nums[page_idx]

    # Build chunks from sliding window over sentences
    raw_chunks = _chunk_sentences(
        sentences=sentences,
        page_start=pages[0].page_num,
        page_end=pages[-1].page_num,
        doc_id=doc_id,
        source=source,
        section=current_section,
        chunk_index_offset=0,
        table_map=table_map,
    )

    # Assign finer-grained page ranges based on sentence position mapping
    # Re-chunk with accurate page_start / page_end and per-chunk section tags
    refined_chunks: list[Chunk] = []
    sentence_idx = 0
    active_section = pages[0].section if pages else ""

    for chunk in raw_chunks:
        chunk_sentences = _split_sentences(chunk.text)
        end_idx = min(sentence_idx + len(chunk_sentences) - 1, n_sentences - 1)
        chunk.page_start = _sentence_page(sentence_idx)
        chunk.page_end = _sentence_page(end_idx)
        chunk.category = category

        # Detect section heading inside this chunk's own text
        section_matches = _SECTION_HEADING_RE.findall(chunk.text)
        if section_matches:
            # Use the FIRST heading found in this chunk (most relevant)
            num, title = section_matches[0]
            active_section = f"{num} {title.strip()}"
        chunk.section = active_section

        sentence_idx = end_idx + 1
        refined_chunks.append(chunk)

    # ── Step 5: Deduplicate by content hash ───────────────────────────────────
    seen_hashes: set[str] = set()
    deduped: list[Chunk] = []
    for chunk in refined_chunks:
        if chunk.content_hash in seen_hashes:
            logger.debug("Duplicate chunk dropped: %s", chunk.chunk_id)
            continue
        seen_hashes.add(chunk.content_hash)
        deduped.append(chunk)

    # Re-index after deduplication
    for i, chunk in enumerate(deduped):
        chunk.chunk_index = i
        chunk.chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}:{i}"))

    logger.info(
        "Chunking complete: %d chunks from %d pages (min %d words, max %d words)",
        len(deduped),
        len(pages),
        min((c.word_count for c in deduped), default=0),
        max((c.word_count for c in deduped), default=0),
    )
    return deduped