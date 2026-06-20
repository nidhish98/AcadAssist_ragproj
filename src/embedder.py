"""
embedder.py
───────────
Embedding layer for the RAKA RAG pipeline.

Produces 384-dimensional L2-normalised vectors suitable for cosine-similarity
search via inner product (FAISS IndexFlatIP).

Two backends — selected automatically at import time:

    1. SentenceTransformerBackend  (preferred)
       Uses: sentence-transformers all-MiniLM-L6-v2
       Requires: pip install sentence-transformers
       Output dim: 384 (fixed by model)

    2. LSABackend  (fallback when sentence-transformers is absent)
       Uses: TF-IDF (sklearn) + TruncatedSVD
       Requires: numpy, scikit-learn  (always available)
       Output dim: min(384, corpus_size-1, vocab_size-1)
       Note: Must call fit_corpus() before encoding queries.
             The fitted vectoriser and SVD are saved alongside embeddings.npy
             so they can be reloaded for query encoding at inference time.

Both backends expose the same public interface:
    encode_texts(texts)  → np.ndarray  shape (n, d)
    encode_query(text)   → np.ndarray  shape (1, d)

Usage (indexing / offline):
    embedder = Embedder()
    embedder.fit_corpus(texts)          # no-op for SentenceTransformer backend
    embeddings = embedder.encode_texts(texts)
    embedder.save(path)                 # saves .npy + backend state

Usage (inference / online):
    embedder = Embedder.load(path)      # restores backend state
    qvec = embedder.encode_query("What is the attendance policy?")
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from utils import normalize_vector, clean_text, ensure_dir

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM_SBERT = 384


# ── Backend: SentenceTransformer ───────────────────────────────────────────────
class SentenceTransformerBackend:
    """Thin wrapper around sentence-transformers for consistent interface."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        logger.info("Loading SentenceTransformer model: %s", SBERT_MODEL_NAME)
        self._model = SentenceTransformer(SBERT_MODEL_NAME)
        self.dim: int = self._model.get_sentence_embedding_dimension()
        logger.info("SentenceTransformer ready — embedding dim: %d", self.dim)

    def fit_corpus(self, texts: list[str]) -> None:
        """No-op — SentenceTransformer is pre-trained."""
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts → (n, dim) float32 L2-normalised array."""
        vectors = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,   # L2 normalise inside the model
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def save_state(self, path: Path) -> None:
        """Nothing to save — model weights are in the installed package."""
        pass

    @classmethod
    def load_state(cls, path: Path) -> "SentenceTransformerBackend":
        return cls()

    @property
    def backend_name(self) -> str:
        return "sentence-transformers"


# ── Backend: TF-IDF + TruncatedSVD (LSA) ─────────────────────────────────────
class LSABackend:
    """
    TF-IDF + Latent Semantic Analysis as a drop-in embedding backend.

    Produces dense vectors via TruncatedSVD on a TF-IDF matrix.
    The fitted vectoriser + SVD are persisted alongside embeddings.npy
    so queries at inference time are projected into the same space.

    Limitations vs SBERT
    ────────────────────
    • Not semantic — relies on lexical overlap.
    • Dimension capped at (corpus_size - 1), so tiny corpora yield small dims.
    • Must be retrained when the corpus changes.

    These are acceptable for development without internet access; swap the
    backend to SentenceTransformerBackend in production.
    """

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self._tfidf = TfidfVectorizer(
            min_df=1,
            max_features=10_000,
            ngram_range=(1, 2),
            sublinear_tf=True,          # log(1+tf) for better weighting
            strip_accents="unicode",
        )
        self._svd: Optional[TruncatedSVD] = None
        self.dim: int = 0
        self._fitted: bool = False

    def fit_corpus(self, texts: list[str]) -> None:
        """
        Fit TF-IDF + SVD on the full document corpus.

        Must be called once before encode().
        Sets self.dim to the actual SVD component count.
        """
        from sklearn.decomposition import TruncatedSVD

        logger.info("Fitting LSA backend on %d texts …", len(texts))
        tfidf_matrix = self._tfidf.fit_transform(texts)
        vocab_size = tfidf_matrix.shape[1]

        # TruncatedSVD requires n_components < min(n_samples, n_features)
        max_components = min(EMBEDDING_DIM_SBERT, len(texts) - 1, vocab_size - 1)
        if max_components < 1:
            raise ValueError(
                f"Corpus too small to fit LSA — need at least 2 texts, got {len(texts)}"
            )
        self._svd = TruncatedSVD(n_components=max_components, random_state=42)
        self._svd.fit(tfidf_matrix)
        self.dim = max_components
        self._fitted = True
        explained = self._svd.explained_variance_ratio_.sum()
        logger.info(
            "LSA fitted: %d components, %.1f%% variance explained",
            self.dim, explained * 100,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        """Project texts into LSA space → (n, dim) float32 L2-normalised."""
        if not self._fitted or self._svd is None:
            raise RuntimeError(
                "LSABackend: call fit_corpus() before encode(). "
                "Or load a pre-fitted backend with LSABackend.load_state()."
            )
        tfidf_matrix = self._tfidf.transform(texts)
        vectors = self._svd.transform(tfidf_matrix).astype(np.float32)
        return normalize_vector(vectors)

    def save_state(self, path: Path) -> None:
        """Persist fitted TF-IDF + SVD to disk."""
        state = {"tfidf": self._tfidf, "svd": self._svd, "dim": self.dim}
        with open(path / "lsa_state.pkl", "wb") as f:
            pickle.dump(state, f)
        logger.info("LSA state saved → %s/lsa_state.pkl", path)

    @classmethod
    def load_state(cls, path: Path) -> "LSABackend":
        """Restore a fitted LSABackend from disk."""
        state_file = path / "lsa_state.pkl"
        if not state_file.exists():
            raise FileNotFoundError(f"LSA state file not found: {state_file}")
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        backend = cls()
        backend._tfidf = state["tfidf"]
        backend._svd = state["svd"]
        backend.dim = state["dim"]
        backend._fitted = True
        logger.info("LSA state loaded from %s (dim=%d)", path, backend.dim)
        return backend

    @property
    def backend_name(self) -> str:
        return "lsa-tfidf-svd"


# ── Backend factory ────────────────────────────────────────────────────────────
def _make_backend() -> SentenceTransformerBackend | LSABackend:
    """Return the best available embedding backend."""
    try:
        import sentence_transformers  # noqa: F401
        return SentenceTransformerBackend()
    except ImportError:
        logger.warning(
            "sentence-transformers not installed — using LSA (TF-IDF + SVD) backend. "
            "Install sentence-transformers for production-quality embeddings."
        )
        return LSABackend()


# ── Public Embedder class ──────────────────────────────────────────────────────
class Embedder:
    """
    Unified embedder — exposes encode_texts() and encode_query().

    Automatically selects SentenceTransformer or LSA backend.
    Call fit_corpus() before encoding when using the LSA backend.

    Example
    ───────
    >>> embedder = Embedder()
    >>> embedder.fit_corpus(texts)          # no-op for SBERT
    >>> embeddings = embedder.encode_texts(texts)
    >>> embedder.save("data/processed")
    >>>
    >>> # At inference time:
    >>> embedder = Embedder.load("data/processed")
    >>> qvec = embedder.encode_query("attendance requirements")
    """

    def __init__(self) -> None:
        self._backend = _make_backend()

    # ── Corpus fitting ────────────────────────────────────────────────────────

    def fit_corpus(self, texts: list[str]) -> None:
        """
        Fit the embedding model on the corpus.

        Required for LSA backend; no-op for SentenceTransformer.

        Parameters
        ──────────
        texts : list of raw chunk texts (not cleaned — cleaning done here).
        """
        cleaned = [clean_text(t) for t in texts]
        self._backend.fit_corpus(cleaned)

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of document chunks.

        Parameters
        ──────────
        texts : list[str]  — raw chunk texts

        Returns
        ───────
        np.ndarray  shape (n, dim), dtype float32, L2-normalised.

        Raises
        ──────
        ValueError   if texts is empty.
        RuntimeError if LSA backend has not been fitted.
        """
        if not texts:
            raise ValueError("encode_texts: received empty text list")

        cleaned = [clean_text(t) for t in texts]
        logger.info(
            "Encoding %d texts with %s backend …",
            len(cleaned), self._backend.backend_name,
        )
        embeddings = self._backend.encode(cleaned)

        # Guarantee L2 normalisation (some backends may not enforce this)
        embeddings = normalize_vector(embeddings)

        logger.info("Embeddings shape: %s  dtype: %s", embeddings.shape, embeddings.dtype)
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single user query string.

        Parameters
        ──────────
        query : str — raw query text

        Returns
        ───────
        np.ndarray  shape (1, dim), dtype float32, L2-normalised.

        Notes
        ─────
        Returns shape (1, dim) (not (dim,)) so it can be passed directly
        to vector_store.search() without reshaping.
        """
        if not query or not query.strip():
            raise ValueError("encode_query: query string is empty")

        cleaned = clean_text(query)
        vector = self._backend.encode([cleaned])          # (1, dim)
        vector = normalize_vector(vector)
        return vector.astype(np.float32)

    @property
    def dim(self) -> int:
        """Embedding dimension of the current backend."""
        return self._backend.dim

    @property
    def backend_name(self) -> str:
        return self._backend.backend_name

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(
        self,
        embeddings: np.ndarray,
        save_dir: str | Path,
    ) -> None:
        """
        Persist embeddings and backend state to disk.

        Writes
        ──────
        <save_dir>/embeddings.npy       — (n, dim) float32 array
        <save_dir>/lsa_state.pkl        — only for LSA backend
        <save_dir>/embedder_meta.pkl    — backend name + dim

        Parameters
        ──────────
        embeddings : the array returned by encode_texts()
        save_dir   : directory path (created if absent)
        """
        save_dir = ensure_dir(save_dir)

        # Save embeddings
        emb_path = save_dir / "embeddings.npy"
        np.save(str(emb_path), embeddings)
        logger.info("Embeddings saved → %s  shape=%s", emb_path, embeddings.shape)

        # Save backend state (LSA only; SBERT is stateless)
        self._backend.save_state(save_dir)

        # Save metadata
        meta = {"backend": self._backend.backend_name, "dim": self.dim}
        with open(save_dir / "embedder_meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, save_dir: str | Path) -> "Embedder":
        """
        Restore an Embedder from a previously saved directory.

        Reads
        ─────
        <save_dir>/embedder_meta.pkl    — to determine which backend to restore
        <save_dir>/lsa_state.pkl        — if backend was LSA

        Returns
        ───────
        Embedder instance ready to call encode_query().
        """
        save_dir = Path(save_dir)
        meta_path = save_dir / "embedder_meta.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(f"embedder_meta.pkl not found in {save_dir}")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        instance = cls.__new__(cls)
        backend_name = meta["backend"]

        if backend_name == "sentence-transformers":
            instance._backend = SentenceTransformerBackend.load_state(save_dir)
        else:
            instance._backend = LSABackend.load_state(save_dir)

        logger.info(
            "Embedder loaded: backend=%s  dim=%d",
            backend_name, instance.dim,
        )
        return instance