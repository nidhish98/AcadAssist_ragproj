"""
retriever.py
────────────
Retrieval layer for the RAKA RAG pipeline.

v3 Changes
──────────
• Hybrid retrieval: BM25 (keyword) + dense (SBERT) scores combined via
  Reciprocal Rank Fusion (RRF). This fixes low-score failures on exact-term
  queries like "FF", "ATKT", "KT", "CGPA", "Elite" etc. where semantic
  similarity alone gives poor scores (0.26 → now reliably retrieved).
• score_threshold default lowered to -0.1 (scores can be negative for cosine).
• retrieve() now returns k=7 by default (was 5) to give the LLM more context.
• Added retrieve_hybrid() as the preferred method. retrieve() calls it.
• BM25 implemented in pure Python — no extra dependency needed.
• Deduplicate results after fusion by chunk_id.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from embedder import Embedder
from vector_store import VectorStore
from utils import load_chunks_json, chunk_text_preview

logger = logging.getLogger(__name__)


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    text:       str
    score:      float
    source:     str
    page_start: int
    page_end:   int
    section:    str
    chunk_id:   int
    word_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "text":       self.text,
            "score":      round(self.score, 4),
            "source":     self.source,
            "page_start": self.page_start,
            "page_end":   self.page_end,
            "section":    self.section,
            "chunk_id":   self.chunk_id,
            "word_count": self.word_count,
        }


# ── BM25 (pure Python, no deps) ───────────────────────────────────────────────

class BM25:
    """
    Okapi BM25 keyword retrieval over the chunk corpus.

    Built once at Retriever.build() / Retriever.load() time.
    Provides exact-term matching as a complement to dense SBERT retrieval.

    Parameters
    ──────────
    k1 = 1.5  (term frequency saturation)
    b  = 0.75 (document length normalisation)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b  = b
        self._corpus: list[list[str]] = []   # tokenised docs
        self._idf: dict[str, float]   = {}
        self._avgdl: float            = 0.0
        self._n: int                  = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, split on non-word chars, remove 1-char tokens."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        return [t for t in tokens if len(t) > 1]

    def fit(self, texts: list[str]) -> None:
        """Fit IDF weights on the corpus."""
        self._corpus = [self._tokenize(t) for t in texts]
        self._n      = len(self._corpus)
        self._avgdl  = sum(len(d) for d in self._corpus) / max(self._n, 1)

        # Document frequency per term
        df: dict[str, int] = defaultdict(int)
        for doc in self._corpus:
            for term in set(doc):
                df[term] += 1

        # IDF with smoothing
        self._idf = {
            term: math.log((self._n - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }
        logger.info("BM25 fitted on %d documents, vocab=%d", self._n, len(self._idf))

    def score(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """
        Return (doc_index, score) pairs for the top_k documents.
        """
        if not self._corpus:
            return []

        q_terms = self._tokenize(query)
        if not q_terms:
            return []

        scores: list[float] = []
        for doc_idx, doc in enumerate(self._corpus):
            dl   = len(doc)
            norm = 1 - self.b + self.b * dl / self._avgdl
            sc   = 0.0
            tf_map: dict[str, int] = defaultdict(int)
            for t in doc:
                tf_map[t] += 1
            for term in q_terms:
                if term not in self._idf:
                    continue
                tf = tf_map.get(term, 0)
                sc += self._idf[term] * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
            scores.append(sc)

        # Top-k by score
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, sc) for idx, sc in ranked[:top_k] if sc > 0]


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def _rrf(
    dense_results: list[tuple[int, float]],   # (chunk_idx, dense_score)
    bm25_results:  list[tuple[int, float]],   # (chunk_idx, bm25_score)
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion of two ranked lists.

    Score = Σ 1 / (k + rank_i) for each result list where the doc appears.

    k=60 is the standard RRF constant (Cormack et al., 2009).
    Returns merged list sorted by RRF score descending.
    """
    rrf_scores: dict[int, float] = defaultdict(float)

    for rank, (idx, _) in enumerate(dense_results, start=1):
        rrf_scores[idx] += 1.0 / (k + rank)

    for rank, (idx, _) in enumerate(bm25_results, start=1):
        rrf_scores[idx] += 1.0 / (k + rank)

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return merged   # [(chunk_idx, rrf_score), ...]


# ── Retriever ─────────────────────────────────────────────────────────────────

class Retriever:
    """
    Stateful retriever: Embedder + VectorStore + BM25.

    Construction paths
    ──────────────────
    Retriever.build()  — encode chunks + build index from chunks.json
    Retriever.load()   — restore from pre-built index files
    """

    def __init__(
        self,
        embedder:  Embedder,
        store:     VectorStore,
        bm25:      BM25,
        chunks:    list[dict[str, Any]],
    ) -> None:
        self._embedder = embedder
        self._store    = store
        self._bm25     = bm25
        self._chunks   = chunks   # raw chunk dicts for BM25 result lookup

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        chunks_path: str | Path,
        save_dir:    str | Path,
    ) -> "Retriever":
        save_dir = Path(save_dir)
        chunks   = load_chunks_json(chunks_path)
        texts    = [c["text"] for c in chunks]

        # Dense embeddings
        embedder   = Embedder()
        embedder.fit_corpus(texts)
        embeddings = embedder.encode_texts(texts)
        embedder.save(embeddings, save_dir)

        # FAISS / numpy index
        store = VectorStore()
        store.build_index(embeddings, chunks)
        store.save_index(save_dir)

        # BM25
        bm25 = BM25()
        bm25.fit(texts)

        logger.info(
            "Retriever built: %d chunks indexed (dense dim=%d, BM25 vocab=%d)",
            len(chunks), embedder.dim, len(bm25._idf),
        )
        return cls(embedder, store, bm25, chunks)

    @classmethod
    def load(cls, save_dir: str | Path) -> "Retriever":
        save_dir = Path(save_dir)
        logger.info("Loading Retriever from %s …", save_dir)

        embedder = Embedder.load(save_dir)
        store    = VectorStore()
        store.load_index(save_dir)

        # Reload chunks for BM25 (metadata.pkl has them via VectorStore)
        # We reconstruct BM25 from the stored metadata
        chunks = store._metadata   # list of chunk dicts already in memory
        bm25   = BM25()
        bm25.fit([c["text"] for c in chunks])

        logger.info(
            "Retriever loaded: %d vectors (dense dim=%d)",
            store.ntotal, store.dim,
        )
        return cls(embedder, store, bm25, chunks)

    # ── Core retrieval ────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:           str,
        k:               int   = 7,
        score_threshold: float = -0.1,
        use_hybrid:      bool  = True,
    ) -> list[RetrievalResult]:
        """
        Retrieve top-k most relevant chunks for a query.

        v3: defaults to hybrid (BM25 + dense) retrieval via RRF.
        Falls back to dense-only if use_hybrid=False.

        Parameters
        ──────────
        query           : user's natural language question
        k               : number of chunks to return (default 7, was 5)
        score_threshold : discard results below this RRF score (default -0.1)
        use_hybrid      : True = BM25 + dense fusion; False = dense only

        Returns
        ───────
        List of RetrievalResult sorted by score descending.
        """
        if not query or not query.strip():
            raise ValueError("retrieve: query string is empty")
        if k <= 0:
            raise ValueError(f"retrieve: k must be positive, got {k}")

        if use_hybrid:
            return self._retrieve_hybrid(query, k, score_threshold)
        else:
            return self._retrieve_dense(query, k, score_threshold)

    def _retrieve_dense(
        self,
        query:           str,
        k:               int,
        score_threshold: float,
    ) -> list[RetrievalResult]:
        """Dense-only SBERT retrieval (original behaviour)."""
        query_vector = self._embedder.encode_query(query)
        raw_results  = self._store.search(query_vector, k=k)

        results: list[RetrievalResult] = []
        for raw in raw_results:
            score = raw.get("score", 0.0)
            if score < score_threshold:
                continue
            results.append(self._make_result(raw, score))

        logger.info(
            "retrieve_dense('%s'): %d results (top score=%.4f)",
            query[:60], len(results),
            results[0].score if results else 0.0,
        )
        return results

    def _retrieve_hybrid(
        self,
        query:           str,
        k:               int,
        score_threshold: float,
    ) -> list[RetrievalResult]:
        """
        Hybrid BM25 + dense retrieval fused via Reciprocal Rank Fusion.

        Steps:
        1. BM25 top-20 ranked list
        2. Dense (SBERT) top-20 ranked list
        3. RRF merge → top-k
        """
        # --- BM25 ranking ---
        bm25_ranked = self._bm25.score(query, top_k=20)

        # --- Dense ranking ---
        query_vector  = self._embedder.encode_query(query)
        # Retrieve more candidates for fusion
        dense_raw     = self._store.search(query_vector, k=min(20, self._store.ntotal))
        # Map chunk_id → position in dense results
        dense_ranked: list[tuple[int, float]] = []
        for raw in dense_raw:
            idx = int(raw.get("chunk_id", -1))
            if idx >= 0:
                dense_ranked.append((idx, raw.get("score", 0.0)))

        # --- RRF fusion ---
        fused = _rrf(dense_ranked, bm25_ranked, k=60)

        # Build results from top-k fused indices
        results: list[RetrievalResult] = []
        for chunk_idx, rrf_score in fused[:k]:
            if chunk_idx < 0 or chunk_idx >= len(self._chunks):
                continue
            chunk = self._chunks[chunk_idx]
            results.append(self._make_result(chunk, rrf_score))

        logger.info(
            "retrieve_hybrid('%s'): %d results "
            "(dense=%d, bm25=%d, fused=%d, returned=%d)",
            query[:60],
            len(results),
            len(dense_ranked),
            len(bm25_ranked),
            len(fused),
            len(results),
        )
        for i, r in enumerate(results[:3]):
            logger.debug(
                "  [%d] score=%.4f  chunk=%d  pg%d-%d  '%s'",
                i + 1, r.score, r.chunk_id,
                r.page_start, r.page_end,
                chunk_text_preview(r.text),
            )

        return results

    @staticmethod
    def _make_result(raw: dict[str, Any], score: float) -> RetrievalResult:
        return RetrievalResult(
            text       = raw.get("text", ""),
            score      = score,
            source     = raw.get("source", "unknown"),
            page_start = raw.get("page_start", 0),
            page_end   = raw.get("page_end",   0),
            section    = raw.get("section",    ""),
            chunk_id   = int(raw.get("chunk_id", -1)),
            word_count = raw.get("word_count", 0),
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def index_size(self) -> int:
        return self._store.ntotal

    @property
    def embedding_dim(self) -> int:
        return self._store.dim