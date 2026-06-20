"""
main.py
───────
End-to-end RAKA RAG pipeline entry point.

v2 Changes
──────────
• Demo queries updated to cover BOTH documents (examination + syllabus).
• Shows active backend (groq / openai / mock) prominently.
• run_query() returns backend info in result dict.
• Added --backend flag to query mode for manual override.

Modes
─────
    build   — embed chunks.json → build index → save
    query   — load index → retrieve → generate → print
    demo    — run test queries covering both documents

Usage
─────
    python main.py build --chunks data/processed/chunks.json \\
                         --index  data/processed/

    python main.py query --index data/processed/ \\
                         --question "What courses are in Semester 3?"

    python main.py demo --index data/processed/

Environment
─────────────
    GROQ_API_KEY    = gsk_...   ← primary (recommended)
    OPENAI_API_KEY  = sk-...    ← fallback
    RAKA_MOCK_LLM   = 1         ← force mock (no API needed)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from embedder    import Embedder
from vector_store import VectorStore
from retriever   import Retriever
from generator   import Generator
from utils       import load_chunks_json, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("raka.pipeline")

# ── Demo queries — covers BOTH documents ─────────────────────────────────────
DEMO_QUERIES = [
    # Examination regulations
    "What is the minimum attendance requirement to appear for the end semester examination?",
    "What are the examination fees for detained students with 50 to 65 percent attendance?",
    "What happens if a student is caught using unfair means during an exam?",
    "How many credits are required to be eligible for the B.Tech degree?",
    "How is CGPA calculated?",
    # CSE Syllabus
    "What courses are offered in Semester 3 for CSE students?",
    "What is the course code and credits for Discrete Structures?",
    "What are the NPTEL credit equivalence rules?",
    "What is the teaching scheme for Mathematics II?",
    "What elective courses are available in the CSE curriculum?",
]


# ── Pipeline functions ────────────────────────────────────────────────────────
def run_build(chunks_path: str, index_dir: str) -> None:
    chunks_path = Path(chunks_path)
    index_dir   = Path(index_dir)

    if not chunks_path.exists():
        logger.error("chunks.json not found: %s", chunks_path)
        sys.exit(1)

    ensure_dir(index_dir)

    print(f"\n{'═'*62}")
    print("  RAKA — Build Phase")
    print(f"{'═'*62}")
    print(f"  Chunks file : {chunks_path}")
    print(f"  Index dir   : {index_dir}")
    print(f"{'─'*62}\n")

    chunks = load_chunks_json(chunks_path)
    print(f"✓ Loaded {len(chunks)} chunks")

    # Show per-document breakdown
    doc_counts: dict[str, int] = {}
    for c in chunks:
        doc_counts[c.get("source", "unknown")] = doc_counts.get(c.get("source", "unknown"), 0) + 1
    for source, count in doc_counts.items():
        print(f"  ├─ {source:<45} : {count} chunks")

    retriever = Retriever.build(chunks_path=chunks_path, save_dir=index_dir)

    print(f"\n✅ Build complete.")
    print(f"   Vectors in index : {retriever.index_size}")
    print(f"   Embedding dim    : {retriever.embedding_dim}")
    print(f"   Backend          : {retriever._embedder.backend_name}")
    print(f"   Files written to : {index_dir}\n")


def run_query(
    index_dir: str,
    question: str,
    k: int = 5,
    verbose: bool = False,
) -> dict:
    index_dir = Path(index_dir)
    retriever = Retriever.load(index_dir)
    results   = retriever.retrieve(question, k=k)

    if verbose and results:
        print(f"\n{'─'*62}")
        print(f"  Retrieved {len(results)} chunks:")
        print(f"{'─'*62}")
        for i, r in enumerate(results, 1):
            print(f"  [{i}] score={r.score:.4f}  pg {r.page_start}–{r.page_end}  {r.source}")
            print(f"      {r.text[:130].replace(chr(10), ' ')}…")
        print()

    generator = Generator()
    response  = generator.generate(question, results)

    print(f"\n{'═'*62}")
    print(f"  Question : {question}")
    print(f"{'─'*62}")
    print(f"  Answer:\n")
    # Wrap answer text at 60 chars
    for line in response.answer.split("\n"):
        print(f"  {line}")
    print()
    print(f"{'─'*62}")
    print(f"  Sources ({len(response.sources)}):")
    for src in response.sources:
        print(f"    • {src['source']}  pp.{src['page_start']}–{src['page_end']}"
              f"  score={src['score']:.3f}  [{src['section'][:50]}]")
    print(f"\n  Backend : {response.backend}  |  Model : {response.model}")
    print(f"  Time    : {response.response_time_ms}ms")
    if response.mock:
        print("  ⚠  MockBackend active — add GROQ_API_KEY to .env for real responses")
    print(f"{'═'*62}\n")

    return response.to_dict()


def run_demo(index_dir: str, k: int = 5) -> None:
    print(f"\n{'═'*62}")
    print("  RAKA — Demo Mode (examination + syllabus)")
    print(f"{'═'*62}\n")

    retriever = Retriever.load(index_dir)
    generator = Generator()

    print(f"  Backend : {generator.backend}  |  Model : {generator.model}\n")

    for i, question in enumerate(DEMO_QUERIES, 1):
        print(f"[{i:>2}/{len(DEMO_QUERIES)}] {question}")
        chunks  = retriever.retrieve(question, k=k)
        resp    = generator.generate(question, chunks)
        answer  = resp.answer[:280] + ("…" if len(resp.answer) > 280 else "")
        mock_tag = " [MOCK]" if resp.mock else f" [{resp.backend}/{resp.response_time_ms}ms]"
        print(f"  ↳ {answer}{mock_tag}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raka",
        description="RAKA RAG Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # build
    pb = sub.add_parser("build", help="Build FAISS index from chunks.json")
    pb.add_argument("--chunks", default="data/processed/chunks.json")
    pb.add_argument("--index",  default="data/processed/")

    # query
    pq = sub.add_parser("query", help="Answer a question")
    pq.add_argument("--index",    default="data/processed/")
    pq.add_argument("--question", required=True)
    pq.add_argument("--k",        type=int, default=5)
    pq.add_argument("--verbose",  action="store_true")

    # demo
    pd = sub.add_parser("demo", help="Run demo questions")
    pd.add_argument("--index", default="data/processed/")
    pd.add_argument("--k",     type=int, default=5)

    return parser


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.mode == "build":
        run_build(chunks_path=args.chunks, index_dir=args.index)
    elif args.mode == "query":
        run_query(
            index_dir=args.index,
            question=args.question,
            k=args.k,
            verbose=args.verbose,
        )
    elif args.mode == "demo":
        run_demo(index_dir=args.index, k=args.k)


if __name__ == "__main__":
    main()