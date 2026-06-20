from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generator import Generator, GeneratorResponse
from retriever import Retriever, RetrievalResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("raka.api")

INDEX_DIR = Path(__file__).parent.parent / "data" / "processed"
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = FastAPI(title="RAKA API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_retriever: Retriever | None = None
_generator: Generator | None = None


class QueryRequest(BaseModel):
    question: str
    k: int = 7


class SourceItem(BaseModel):
    source: str
    page_start: int
    page_end: int
    section: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    backend: str
    model: str
    response_time_ms: int


class DocumentInfo(BaseModel):
    source: str
    chunk_count: int


class StatusResponse(BaseModel):
    status: str
    indexed_documents: list[DocumentInfo]
    total_chunks: int
    embedding_dim: int
    backend: str
    llm_backend: str
    llm_model: str


def _load_pipeline() -> None:
    global _retriever, _generator
    if _retriever is None:
        logger.info("Loading retriever from %s", INDEX_DIR)
        _retriever = Retriever.load(INDEX_DIR)
    if _generator is None:
        _generator = Generator()


@app.on_event("startup")
def startup() -> None:
    _load_pipeline()
    logger.info("RAKA API ready")


@app.get("/api/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    if _retriever is None:
        return JSONResponse(status_code=503, content={"status": "not ready"})
    source_counts: dict[str, int] = {}
    for chunk in _retriever._chunks:
        src = chunk.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
    return StatusResponse(
        status="ok",
        indexed_documents=[
            DocumentInfo(source=src, chunk_count=cnt)
            for src, cnt in sorted(source_counts.items())
        ],
        total_chunks=_retriever.index_size,
        embedding_dim=_retriever.embedding_dim,
        backend=_retriever._embedder.backend_name,
        llm_backend=_generator.backend if _generator else "unknown",
        llm_model=_generator.model if _generator else "unknown",
    )


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    t0 = time.time()
    _load_pipeline()
    chunks: list[RetrievalResult] = _retriever.retrieve(req.question, k=req.k)
    response: GeneratorResponse = _generator.generate(req.question, chunks)
    elapsed = int((time.time() - t0) * 1000)
    return QueryResponse(
        answer=response.answer,
        sources=[
            SourceItem(
                source=s["source"],
                page_start=s["page_start"],
                page_end=s["page_end"],
                section=s["section"],
                score=s["score"],
            )
            for s in response.sources
        ],
        backend=response.backend,
        model=response.model,
        response_time_ms=elapsed,
    )


@app.post("/api/upload")
def upload_pdf(file: UploadFile) -> dict:
    pdf_dir = INDEX_DIR.parent.parent / "documents"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    dest = pdf_dir / file.filename
    with open(dest, "wb") as f:
        f.write(file.file.read())
    logger.info("PDF saved to %s", dest)
    from processing_main import run_multi_pipeline
    from main import run_build
    chunks = run_multi_pipeline(
        pdf_paths=[str(dest)],
        output_path=str(INDEX_DIR / "chunks.json"),
        print_report=False,
    )
    run_build(chunks_path=str(INDEX_DIR / "chunks.json"), index_dir=str(INDEX_DIR))
    global _retriever
    _retriever = Retriever.load(INDEX_DIR)
    return {"status": "ok", "chunks_added": len(chunks)}


@app.get("/{path:path}")
def serve_frontend(path: str = "") -> FileResponse:
    file = FRONTEND_DIR / (path or "index.html")
    if file.exists() and file.is_file():
        return FileResponse(file)
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse(status_code=404, content={"detail": "Not found"})


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
