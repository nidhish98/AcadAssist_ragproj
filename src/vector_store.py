"""
vector_store.py
───────────────
Vector store layer for the RAKA RAG pipeline.

Implements a FAISS-compatible inner-product index using pure numpy.
When real FAISS is installed, swap NumpyIndexFlatIP for faiss.IndexFlatIP —
the VectorStore interface does not change.

Design
──────
• IndexFlatIP on L2-normalised vectors = cosine similarity search.
• Metadata (chunk dicts) stored in a parallel list keyed by FAISS row index.
• Index + metadata persisted together as faiss.index + metadata.pkl.
• All public methods mirror the FAISS Python API naming where possible.

File layout (under data/processed/)
────────────────────────────────────
    faiss.index    — serialised numpy array (or real FAISS index binary)
    metadata.pkl   — list[dict], index i → chunk metadata for FAISS row i
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np

from utils import normalize_vector, ensure_dir

logger = logging.getLogger(__name__)


# ── Pure-numpy inner-product index (FAISS drop-in) ────────────────────────────

class NumpyIndexFlatIP:
    """
    Exact inner-product search over a static set of L2-normalised vectors.

    For L2-normalised vectors, inner product == cosine similarity.

    Interface mirrors faiss.IndexFlatIP:
        add(xb)          — add (n, d) float32 array
        search(xq, k)    — return (distances, indices) for top-k
        ntotal           — number of stored vectors
        d                — vector dimension
    """

    def __init__(self, d: int) -> None:
        self.d = d
        self._store: Optional[np.ndarray] = None  # (n, d) float32

    @property
    def ntotal(self) -> int:
        return 0 if self._store is None else self._store.shape[0]

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.

        Parameters
        ──────────
        xb : np.ndarray  shape (n, d), dtype float32
        """
        xb = np.asarray(xb, dtype=np.float32)
        if xb.ndim == 1:
            xb = xb[None, :]
        if xb.shape[1] != self.d:
            raise ValueError(
                f"Vector dimension mismatch: index expects {self.d}, got {xb.shape[1]}"
            )
        if self._store is None:
            self._store = xb
        else:
            self._store = np.vstack([self._store, xb])
        logger.debug("Index now contains %d vectors", self.ntotal)

    def search(
        self, xq: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find top-k nearest vectors by inner product.

        Parameters
        ──────────
        xq : np.ndarray  shape (nq, d) or (d,) — query vector(s)
        k  : int — number of results per query

        Returns
        ───────
        (distances, indices)
            distances : np.ndarray  shape (nq, k)  — inner-product scores
            indices   : np.ndarray  shape (nq, k)  — row indices in the index
            Returns -1 indices for padded results when k > ntotal.
        """
        if self._store is None or self.ntotal == 0:
            nq = 1 if xq.ndim == 1 else xq.shape[0]
            return (
                np.full((nq, k), -1.0, dtype=np.float32),
                np.full((nq, k), -1,   dtype=np.int64),
            )

        xq = np.atleast_2d(np.asarray(xq, dtype=np.float32))
        effective_k = min(k, self.ntotal)

        # Inner product: (nq, d) @ (d, n) → (nq, n)
        scores = xq @ self._store.T              # (nq, ntotal)

        # argsort descending — top-k
        if effective_k == self.ntotal:
            top_indices = np.argsort(scores, axis=1)[:, ::-1]
        else:
            # argpartition is O(n) for large n; argsort for small corpora
            top_indices = np.argsort(scores, axis=1)[:, ::-1][:, :effective_k]

        top_scores = np.take_along_axis(scores, top_indices, axis=1)

        # Pad to shape (nq, k) with -1 if effective_k < k
        if effective_k < k:
            pad_width = k - effective_k
            top_scores  = np.pad(top_scores,  ((0, 0), (0, pad_width)), constant_values=-1.0)
            top_indices = np.pad(top_indices, ((0, 0), (0, pad_width)), constant_values=-1)

        return top_scores.astype(np.float32), top_indices.astype(np.int64)

    def save(self, path: Path) -> None:
        """Serialise index vectors to disk."""
        if self._store is None:
            raise RuntimeError("Cannot save an empty index")
        np.save(str(path), self._store)
        logger.info("NumpyIndex saved → %s  (%d vectors, dim=%d)", path, self.ntotal, self.d)

    @classmethod
    def load(cls, path: Path, d: int) -> "NumpyIndexFlatIP":
        """Restore index from disk."""
        store = np.load(str(path)).astype(np.float32)
        idx = cls(d=store.shape[1])
        idx._store = store
        logger.info("NumpyIndex loaded ← %s  (%d vectors, dim=%d)", path, idx.ntotal, idx.d)
        return idx


# ── FAISS shim: replace NumpyIndexFlatIP with real FAISS when available ────────
def _make_index(d: int) -> NumpyIndexFlatIP:
    """
    Return the best available inner-product index for dimension d.

    Drop-in FAISS swap:  when faiss-cpu is installed this function can be
    changed to:
        import faiss
        return faiss.IndexFlatIP(d)
    and the rest of VectorStore requires zero changes.
    """
    try:
        import faiss  # type: ignore
        idx = faiss.IndexFlatIP(d)
        logger.info("Using real FAISS IndexFlatIP (dim=%d)", d)
        return idx  # type: ignore[return-value]
    except ImportError:
        logger.info("FAISS not installed — using NumpyIndexFlatIP (dim=%d)", d)
        return NumpyIndexFlatIP(d)


# ── VectorStore ────────────────────────────────────────────────────────────────

class VectorStore:
    """
    Manages the FAISS/numpy index and its chunk metadata mapping.

    Responsibilities
    ────────────────
    • build_index()   — populate index from embeddings + chunk list
    • save_index()    — persist index + metadata to disk
    • load_index()    — restore from disk
    • search()        — return top-k chunks with scores

    Usage
    ─────
    Build (offline, after embedder.py runs):
        store = VectorStore()
        store.build_index(embeddings, chunks)
        store.save_index("data/processed")

    Retrieve (online, inside retriever.py):
        store = VectorStore()
        store.load_index("data/processed")
        results = store.search(query_vector, k=5)
    """

    _INDEX_FILE   = "faiss.index.npy"     # our numpy version
    _FAISS_FILE   = "faiss.index"          # real FAISS binary
    _METADATA_FILE = "metadata.pkl"

    def __init__(self) -> None:
        self._index: Optional[NumpyIndexFlatIP] = None
        self._metadata: list[dict[str, Any]] = []   # index i → chunk dict
        self._dim: int = 0

    # ── Build ─────────────────────────────────────────────────────────────────

    def build_index(
        self,
        embeddings: np.ndarray,
        chunks: list[dict[str, Any]],
    ) -> None:
        """
        Populate the index from a (n, d) embedding matrix and matching chunk list.

        Parameters
        ──────────
        embeddings : np.ndarray  shape (n, d), float32, L2-normalised
        chunks     : list of chunk dicts from chunks.json — must be same length

        Raises
        ──────
        ValueError  if len(embeddings) != len(chunks) or embeddings is empty.
        """
        if len(embeddings) == 0:
            raise ValueError("build_index: embeddings array is empty")
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"build_index: embeddings ({len(embeddings)}) and chunks "
                f"({len(chunks)}) must have the same length"
            )

        embeddings = np.asarray(embeddings, dtype=np.float32)
        # Re-normalise defensively
        embeddings = normalize_vector(embeddings)

        self._dim = embeddings.shape[1]
        self._index = _make_index(self._dim)
        self._index.add(embeddings)
        self._metadata = list(chunks)   # shallow copy

        logger.info(
            "VectorStore built: %d vectors, dim=%d",
            self._index.ntotal, self._dim,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_index(self, save_dir: str | Path) -> None:
        """
        Persist index + metadata to disk.

        Writes
        ──────
        <save_dir>/faiss.index.npy   — vector matrix (numpy backend)
                or faiss.index        — binary (real FAISS backend)
        <save_dir>/metadata.pkl      — list of chunk dicts
        """
        if self._index is None:
            raise RuntimeError("Cannot save: index has not been built. Call build_index() first.")

        save_dir = ensure_dir(save_dir)

        # Save index
        if isinstance(self._index, NumpyIndexFlatIP):
            self._index.save(save_dir / self._INDEX_FILE)
        else:
            # Real FAISS
            import faiss  # type: ignore
            faiss.write_index(self._index, str(save_dir / self._FAISS_FILE))
            logger.info("FAISS index saved → %s", save_dir / self._FAISS_FILE)

        # Save metadata
        meta_path = save_dir / self._METADATA_FILE
        with open(meta_path, "wb") as f:
            pickle.dump(
                {"metadata": self._metadata, "dim": self._dim},
                f,
            )
        logger.info(
            "Metadata saved → %s  (%d entries)", meta_path, len(self._metadata)
        )

    def load_index(self, save_dir: str | Path) -> None:
        """
        Restore index + metadata from disk.

        Reads
        ─────
        <save_dir>/faiss.index.npy   (numpy) or faiss.index (real FAISS)
        <save_dir>/metadata.pkl

        Raises
        ──────
        FileNotFoundError  if neither index file exists.
        """
        save_dir = Path(save_dir)

        # Load metadata first (needed for dim)
        meta_path = save_dir / self._METADATA_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.pkl not found in {save_dir}")
        with open(meta_path, "rb") as f:
            saved = pickle.load(f)
        self._metadata = saved["metadata"]
        self._dim = saved["dim"]

        # Load index — try numpy file first, then real FAISS binary
        numpy_path = save_dir / self._INDEX_FILE
        faiss_path = save_dir / self._FAISS_FILE

        if numpy_path.exists():
            self._index = NumpyIndexFlatIP.load(numpy_path, self._dim)
        elif faiss_path.exists():
            import faiss  # type: ignore
            self._index = faiss.read_index(str(faiss_path))
            logger.info("FAISS index loaded ← %s", faiss_path)
        else:
            raise FileNotFoundError(
                f"No index file found in {save_dir}. "
                f"Expected '{self._INDEX_FILE}' or '{self._FAISS_FILE}'."
            )

        logger.info(
            "VectorStore loaded: %d vectors, dim=%d",
            self._index.ntotal, self._dim,
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-k most similar chunks for a query vector.

        Parameters
        ──────────
        query_vector : np.ndarray  shape (1, d) or (d,) — L2-normalised
        k            : int — number of results to return

        Returns
        ───────
        list of dicts, each containing all keys from the original chunk dict
        plus:
            "score" : float — cosine similarity score in [-1, 1]

        Returns [] if the index is empty or k=0.

        Raises
        ──────
        RuntimeError  if load_index() / build_index() has not been called.
        """
        if self._index is None:
            raise RuntimeError(
                "VectorStore is empty. Call build_index() or load_index() first."
            )
        if self._index.ntotal == 0:
            logger.warning("search: index is empty, returning no results")
            return []
        if k <= 0:
            return []

        query_vector = np.atleast_2d(
            normalize_vector(np.asarray(query_vector, dtype=np.float32))
        )

        effective_k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query_vector, effective_k)

        scores   = scores[0]    # (k,)
        indices  = indices[0]   # (k,)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores, indices):
            if idx == -1:       # padded result
                continue
            chunk = dict(self._metadata[int(idx)])   # shallow copy
            chunk["score"] = float(score)
            results.append(chunk)

        logger.debug(
            "search: returned %d results for k=%d (top score=%.4f)",
            len(results), k, results[0]["score"] if results else 0.0,
        )
        return results

    @property
    def ntotal(self) -> int:
        """Number of vectors currently in the index."""
        return 0 if self._index is None else self._index.ntotal

    @property
    def dim(self) -> int:
        """Embedding dimension of the index."""
        return self._dim