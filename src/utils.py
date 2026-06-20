"""
utils.py
────────
Shared utility functions for the RAKA RAG pipeline.

Contains
────────
• cosine_similarity()     — numpy-based, handles batches
• normalize_vector()      — L2 normalisation for FAISS IP search
• clean_text()            — light text normalisation before embedding
• chunk_text_preview()    — truncate chunk text for logging
• ensure_dir()            — mkdir -p helper
• load_chunks_json()      — typed loader for chunks.json output of chunker.py
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Vector math ────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between two arrays of vectors.

    Parameters
    ──────────
    a : np.ndarray  shape (n, d) or (d,)
    b : np.ndarray  shape (m, d) or (d,)

    Returns
    ───────
    np.ndarray  shape (n, m) — similarity matrix, or scalar if both are 1-D.

    Notes
    ─────
    Equivalent to inner product on L2-normalised vectors. If you already
    normalise before storing in FAISS / numpy store, use inner product directly
    (it's faster). This function works on un-normalised input too.
    """
    a = np.atleast_2d(a).astype(np.float32)
    b = np.atleast_2d(b).astype(np.float32)

    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)

    # Avoid division by zero for zero vectors
    a_norm = np.where(a_norm == 0, 1e-10, a_norm)
    b_norm = np.where(b_norm == 0, 1e-10, b_norm)

    a_unit = a / a_norm
    b_unit = b / b_norm

    result = a_unit @ b_unit.T
    # Squeeze to scalar/1-D if inputs were 1-D
    if result.shape == (1, 1):
        return float(result[0, 0])
    return result


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    L2-normalise a vector or batch of vectors in-place (returns a copy).

    Parameters
    ──────────
    v : np.ndarray  shape (d,) or (n, d)

    Returns
    ───────
    np.ndarray — same shape, unit L2 norm on last axis.
    """
    v = np.array(v, dtype=np.float32)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / (norm if norm > 0 else 1e-10)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    return v / norms


# ── Text utilities ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Light normalisation applied to chunk text before embedding.

    Performs
    ────────
    • Collapse multiple whitespace/newlines to single space.
    • Strip leading/trailing whitespace.
    • Remove control characters (except newline).
    • Normalise unicode bullets to ASCII dash.

    Does NOT do stemming, stopword removal, or lowercasing —
    sentence-transformers / TF-IDF handle that internally.
    """
    if not text:
        return ""
    # Replace newlines / carriage returns with a space
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    # Remove non-printable control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalise unicode bullets / special chars
    text = re.sub(r"[•·●◆▪▶►]", "-", text)
    # Collapse runs of whitespace
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def chunk_text_preview(text: str, max_chars: int = 120) -> str:
    """Return a single-line preview of chunk text for logging."""
    preview = clean_text(text)[:max_chars]
    return preview + ("…" if len(text) > max_chars else "")


# ── File system helpers ────────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist. Return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Chunk loading ──────────────────────────────────────────────────────────────

def load_chunks_json(path: str | Path) -> list[dict[str, Any]]:
    """
    Load chunks.json produced by chunker.py / main.py.

    Parameters
    ──────────
    path : path to chunks.json

    Returns
    ───────
    List of chunk dicts, each containing at minimum:
        chunk_id, text, source, section, page_start, page_end, word_count

    Raises
    ──────
    FileNotFoundError  if file does not exist.
    ValueError         if file is empty or not a JSON array.
    """
    import json

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"chunks.json not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"chunks.json must be a non-empty JSON array: {path}")

    required_keys = {"text", "source"}
    for i, chunk in enumerate(data):
        missing = required_keys - chunk.keys()
        if missing:
            raise ValueError(f"Chunk {i} missing required keys: {missing}")

    logger.info("Loaded %d chunks from %s", len(data), path)
    return data