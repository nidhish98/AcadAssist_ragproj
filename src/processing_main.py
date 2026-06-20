"""
processing_main.py
──────────────────
Document processing entry point for the RAKA pipeline.

v2 Changes
──────────
• Supports processing MULTIPLE PDFs in one run (--pdf can repeat).
• Auto-detects document category from filename.
• --merge flag combines all chunks into a single chunks.json for joint indexing.
• Quality report now shows per-document stats.
• --out defaults to data/processed/chunks.json when --merge is used.

Usage
─────
    # Single PDF
    python processing_main.py --pdf data/raw/examination.pdf

    # Multiple PDFs, merged into one index
    python processing_main.py \\
        --pdf data/raw/examination.pdf \\
        --pdf data/raw/cse-syll-2023-27.pdf \\
        --merge \\
        --out data/processed/chunks.json

    # Multiple PDFs, separate chunk files
    python processing_main.py \\
        --pdf data/raw/examination.pdf \\
        --pdf data/raw/cse-syll-2023-27.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from loader import load_pdf, PageRecord
from chunker import chunk_pages, Chunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("raka.processing")


# ── Quality report ────────────────────────────────────────────────────────────
def _print_report(
    pages: list[PageRecord],
    chunks: list[Chunk],
    label: str = "",
) -> None:
    tag = f" [{label}]" if label else ""
    print("\n" + "═" * 62)
    print(f"  RAKA — Document Processing Report{tag}")
    print("═" * 62)

    print(f"\n📄 Pages loaded        : {len(pages)}")
    page_types: dict[str, int] = {}
    for p in pages:
        page_types[p.page_type] = page_types.get(p.page_type, 0) + 1
    for ptype, count in sorted(page_types.items()):
        print(f"   ├─ {ptype:<14}  : {count}")

    if not chunks:
        print("\n⚠  No chunks produced.")
        return

    wc = [c.word_count for c in chunks]
    print(f"\n📦 Total chunks        : {len(chunks)}")
    print(f"   ├─ Min words        : {min(wc)}")
    print(f"   ├─ Max words        : {max(wc)}")
    print(f"   ├─ Avg words        : {sum(wc) // len(wc)}")
    print(f"   └─ Target range     : 300–500 words")

    under = [c for c in chunks if c.word_count < 300]
    over  = [c for c in chunks if c.word_count > 500]
    if under:
        print(f"\n⚠  {len(under)} chunk(s) below 300 words:")
        for c in under[:3]:
            print(f"   chunk {c.chunk_index:>3} | {c.word_count}w | pg {c.page_start}-{c.page_end}")
    if over:
        print(f"\n⚠  {len(over)} chunk(s) above 500 words:")
        for c in over[:3]:
            print(f"   chunk {c.chunk_index:>3} | {c.word_count}w | pg {c.page_start}-{c.page_end}")

    sections = sorted({c.section for c in chunks if c.section})
    print(f"\n📑 Unique sections     : {len(sections)}")
    for sec in sections[:8]:
        print(f"   • {sec[:70]}")
    if len(sections) > 8:
        print(f"   … and {len(sections) - 8} more")

    table_chunks = [c for c in chunks if " | " in c.text]
    print(f"\n📊 Chunks with tables  : {len(table_chunks)}")

    print("\n" + "─" * 62)
    print("  Sample chunks (first 3):")
    print("─" * 62)
    for c in chunks[:3]:
        print(f"\n[Chunk {c.chunk_index}]")
        print(f"  Source  : {c.source}")
        print(f"  Section : {c.section[:60]}")
        print(f"  Pages   : {c.page_start}–{c.page_end}")
        print(f"  Words   : {c.word_count}")
        print(f"  Preview : {c.text[:200].replace(chr(10), ' ')}…")
    print("\n" + "═" * 62 + "\n")


# ── JSON save ─────────────────────────────────────────────────────────────────
def _save_json(chunks: list[Chunk], output_path: Path) -> None:
    data = [c.to_dict() for c in chunks]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d chunks → %s", len(chunks), output_path)


# ── Single PDF pipeline ───────────────────────────────────────────────────────
def run_pipeline(
    pdf_path: str,
    output_path: str | None = None,
    category: str = "auto",
    print_report: bool = True,
) -> list[Chunk]:
    """
    Process a single PDF → chunks.

    Parameters
    ──────────
    pdf_path    : path to PDF.
    output_path : where to write chunks.json (None → <pdf_stem>_chunks.json).
    category    : document category label or "auto" to detect from filename.
    print_report: whether to print quality report.

    Returns
    ───────
    List of Chunk objects.
    """
    pdf  = Path(pdf_path)
    out  = Path(output_path) if output_path else pdf.parent / f"{pdf.stem}_chunks.json"

    # Auto-detect category from filename
    if category == "auto":
        name = pdf.stem.lower()
        if any(k in name for k in ("syll", "curriculum", "cse", "course")):
            category = "syllabus"
        elif any(k in name for k in ("exam", "regulation", "rule")):
            category = "examination"
        else:
            category = "general"
        logger.info("Auto-detected category='%s' for '%s'", category, pdf.name)

    logger.info("Step 1/3 — Loading: %s", pdf.name)
    pages = load_pdf(pdf, doc_type=category)
    if not pages:
        logger.error("No content pages extracted from %s. Aborting.", pdf.name)
        return []

    logger.info("Step 2/3 — Chunking %d pages from %s …", len(pages), pdf.name)
    chunks = chunk_pages(pages, category=category)
    if not chunks:
        logger.error("No chunks produced from %s.", pdf.name)
        return []

    logger.info("Step 3/3 — Saving %d chunks to %s …", len(chunks), out)
    _save_json(chunks, out)

    if print_report:
        _print_report(pages, chunks, label=pdf.name)

    return chunks


# ── Multi-PDF pipeline ────────────────────────────────────────────────────────
def run_multi_pipeline(
    pdf_paths: list[str],
    output_path: str,
    print_report: bool = True,
) -> list[Chunk]:
    """
    Process multiple PDFs and merge all chunks into a single JSON.
    Chunk indices are re-numbered globally after merging.

    Parameters
    ──────────
    pdf_paths   : list of PDF paths.
    output_path : path for merged chunks.json.
    print_report: whether to print per-document and summary reports.

    Returns
    ───────
    List of all merged Chunk objects.
    """
    all_chunks: list[Chunk] = []

    for pdf_path in pdf_paths:
        chunks = run_pipeline(
            pdf_path=pdf_path,
            output_path=None,   # don't save individual files
            category="auto",
            print_report=print_report,
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.error("No chunks produced from any PDF.")
        return []

    # Re-number chunk indices globally
    for i, chunk in enumerate(all_chunks):
        chunk.chunk_index = i

    # Save merged file
    out = Path(output_path)
    _save_json(all_chunks, out)

    print(f"\n{'═'*62}")
    print(f"  RAKA — Merge Summary")
    print(f"{'═'*62}")
    print(f"  PDFs processed  : {len(pdf_paths)}")
    print(f"  Total chunks    : {len(all_chunks)}")

    # Per-document breakdown
    doc_counts: dict[str, int] = {}
    for c in all_chunks:
        doc_counts[c.source] = doc_counts.get(c.source, 0) + 1
    for source, count in doc_counts.items():
        print(f"  ├─ {source:<40} : {count} chunks")

    print(f"  Output file     : {out}")
    print(f"{'═'*62}\n")

    return all_chunks


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAKA Document Processing — PDF → Chunks → JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pdf", action="append", required=True, metavar="PATH",
        help="Path to a PDF file. Repeat for multiple PDFs.",
    )
    parser.add_argument(
        "--out", default=None,
        help=(
            "Output JSON path. "
            "For single PDFs defaults to <pdf_stem>_chunks.json. "
            "Required when --merge is used."
        ),
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge all PDFs into a single chunks.json (required for joint indexing).",
    )
    parser.add_argument(
        "--category", default="auto",
        choices=["auto", "examination", "syllabus", "general"],
        help="Document category label (auto = detect from filename).",
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Suppress quality report output.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print_report = not args.no_report

    if args.merge:
        if not args.out:
            logger.error("--out is required when using --merge.")
            sys.exit(1)
        chunks = run_multi_pipeline(
            pdf_paths=args.pdf,
            output_path=args.out,
            print_report=print_report,
        )
    else:
        # Process each PDF individually
        total = 0
        for pdf_path in args.pdf:
            out = args.out  # if single PDF, use --out; else auto-name
            chunks = run_pipeline(
                pdf_path=pdf_path,
                output_path=out if len(args.pdf) == 1 else None,
                category=args.category,
                print_report=print_report,
            )
            total += len(chunks)
        print(f"✅  Done — {total} total chunks written.")
        return

    if not chunks:
        sys.exit(1)

    print(f"✅  Done — {len(chunks)} total chunks written.")


if __name__ == "__main__":
    main()