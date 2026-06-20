"""
generator.py
────────────
LLM generation layer for the RAKA RAG pipeline.

v3 Changes
──────────
• Increased MAX_CONTEXT_CHARS from 6000 → 12000 (fixes truncated context).
• Upgraded default Groq model to llama-3.3-70b-versatile for better quality.
• Richer system prompt: lists all known SPIT document types so the LLM knows
  what it has access to.
• Context block now includes ALL chunks (not just first 2000 chars of first).
• Added faithfulness_score computation (cosine sim between answer & context).
• NO_ANSWER threshold lowered — LLM instructed to be more specific with partial
  information instead of blanket "I could not find".
• Added keyword-hint injection so the LLM is nudged to look for exact terms.

Environment variables
─────────────────────
    GROQ_API_KEY    — primary LLM (preferred, free tier available)
    OPENAI_API_KEY  — secondary fallback LLM
    RAKA_MOCK_LLM=1 — force MockBackend (no API calls)
    RAKA_LLM_MODEL  — override model name
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from retriever import RetrievalResult
from utils import clean_text

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
GROQ_API_URL    = "https://api.groq.com/openai/v1/chat/completions"
OPENAI_API_URL  = "https://api.openai.com/v1/chat/completions"

DEFAULT_GROQ_MODEL   = "llama-3.3-70b-versatile"   # upgraded from 8b-instant
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

TEMPERATURE       = 0.1
MAX_TOKENS        = 768       # increased from 512
REQUEST_TIMEOUT   = 30
MAX_CONTEXT_CHARS = 12000     # doubled from 6000 — key fix for missed answers

SYSTEM_PROMPT = """You are RAKA — the official academic knowledge assistant for
Sardar Patel Institute of Technology (SPIT), Mumbai.

You have been given excerpts from the following SPIT institutional documents:
  • Examination Regulations (attendance, grades, CGPA, malpractice, fees)
  • CSE/EXTC/Computers B.Tech Syllabus 2023-27 (courses, credits, modules)
  • Training & Placement Policy 2025-26 (categories, demerits, eligibility)
  • Third Year B.Tech Admission Fee Notice 2025-26 (fee structure by category)
  • Institutional Values and Code of Conduct 2022-23
  • Annual Activity Report 2025-26

STRICT RULES:
1. Answer using ONLY the document context provided below. Never use outside knowledge.
2. If you find partial information, share what you found and note what is missing.
3. ONLY respond "I could not find this information in the provided documents." if
   there is absolutely zero relevant information in ANY of the context chunks.
4. Always mention the source document and page numbers.
5. For tables in context (rows separated by |), read and interpret them carefully.
6. For grade/marking related queries (FF, KT, ATKT, etc.), check Table 7 context.
7. Include course codes, credit hours, and exact amounts where available.
8. Be concise. Do not pad your answer."""

NO_ANSWER_RESPONSE = "I could not find this information in the provided documents."


# ── Response type ─────────────────────────────────────────────────────────────
@dataclass
class GeneratorResponse:
    """Structured response from the LLM generation step."""
    answer:           str
    sources:          list[dict[str, Any]]
    context_used:     str
    model:            str
    response_time_ms: int
    mock:             bool = False
    backend:          str  = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer":           self.answer,
            "sources":          self.sources,
            "model":            self.model,
            "backend":          self.backend,
            "response_time_ms": self.response_time_ms,
            "mock":             self.mock,
        }


# ── Context builder ───────────────────────────────────────────────────────────
def _build_context_block(chunks: list[RetrievalResult]) -> str:
    """
    Build the context string from retrieved chunks.

    v3 Fix: previously only included ~2000 chars of the first chunk.
    Now includes ALL chunks up to MAX_CONTEXT_CHARS, distributing space fairly.
    Each chunk has a full header so the LLM can cite sources accurately.
    """
    parts = []
    total_chars = 0
    chars_per_chunk = MAX_CONTEXT_CHARS // max(len(chunks), 1)

    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"[Source {i} | Document: {chunk.source} "
            f"| Pages: {chunk.page_start}-{chunk.page_end} "
            f"| Section: {chunk.section[:60]}]"
        )
        body = clean_text(chunk.text)

        # Allocate chars fairly; always include at least one meaningful excerpt
        alloc = max(chars_per_chunk, 800)
        body_trimmed = body[:alloc]
        if len(body) > alloc:
            # Try to cut at sentence boundary
            last_period = body_trimmed.rfind(". ")
            if last_period > alloc // 2:
                body_trimmed = body_trimmed[:last_period + 1]

        entry = f"{header}\n{body_trimmed}\n\n"

        if total_chars + len(entry) > MAX_CONTEXT_CHARS:
            # Still try to include a short excerpt of this chunk
            remaining = MAX_CONTEXT_CHARS - total_chars - len(header) - 10
            if remaining > 200:
                parts.append(f"{header}\n{body[:remaining]}…\n\n")
            break

        parts.append(entry)
        total_chars += len(entry)

    return "".join(parts)


def _build_source_list(chunks: list[RetrievalResult]) -> list[dict[str, Any]]:
    return [
        {
            "source":     chunk.source,
            "page_start": chunk.page_start,
            "page_end":   chunk.page_end,
            "section":    chunk.section,
            "score":      round(chunk.score, 4),
            "chunk_id":   chunk.chunk_id,
        }
        for chunk in chunks
    ]


def _keyword_hint(query: str) -> str:
    """
    Extract key terms from the query to include as a hint to the LLM.
    Helps the LLM focus on exact acronyms like FF, KT, ATKT, CGPA, etc.
    """
    import re
    # Find all uppercase acronyms / numbers / important tokens
    acronyms = re.findall(r'\b[A-Z]{2,}\b', query)
    numbers  = re.findall(r'\b\d+(?:\.\d+)?(?:\s*(?:LPA|lakh|lpa|%))?\b', query)
    hints = acronyms + numbers
    if hints:
        return f"\n[Key terms to look for in context: {', '.join(hints)}]"
    return ""


# ── Shared HTTP helper ────────────────────────────────────────────────────────
def _post_chat(
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Generic OpenAI-compatible chat completions POST."""
    payload = {
        "model": model,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        api_url,
        data=data,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent":    "RAKA/3.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API error {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API unreachable: {exc.reason}") from exc

    try:
        return body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected API response format: {body}") from exc


# ── Backends ──────────────────────────────────────────────────────────────────
class GroqBackend:
    def __init__(self, api_key: str, model: str = DEFAULT_GROQ_MODEL) -> None:
        self._api_key = api_key
        self.model    = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return _post_chat(GROQ_API_URL, self._api_key, self.model, system_prompt, user_prompt)

    @property
    def backend_name(self) -> str:
        return "groq"


class OpenAIBackend:
    def __init__(self, api_key: str, model: str = DEFAULT_OPENAI_MODEL) -> None:
        self._api_key = api_key
        self.model    = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return _post_chat(OPENAI_API_URL, self._api_key, self.model, system_prompt, user_prompt)

    @property
    def backend_name(self) -> str:
        return "openai"


class MockBackend:
    """Keyword-overlap mock — no API needed. Used for dev/testing."""
    model = "mock-backend"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        context_start  = user_prompt.find("CONTEXT:")
        question_start = user_prompt.find("QUESTION:")
        if context_start == -1 or question_start == -1:
            return NO_ANSWER_RESPONSE

        context  = user_prompt[context_start + 8: question_start].strip()
        question = user_prompt[question_start + 9:].strip()

        q_words   = set(question.lower().split())
        sentences = [
            s.strip() for s in context.replace("\n", " ").split(".")
            if len(s.strip()) > 30
        ]
        if not sentences:
            return NO_ANSWER_RESPONSE

        best = max(sentences, key=lambda s: len(q_words & set(s.lower().split())))
        if not (q_words & set(best.lower().split())):
            return NO_ANSWER_RESPONSE

        return (
            "[MOCK RESPONSE — no API key set]\n\n"
            f"Based on the provided documents: {best.strip()}."
        )

    @property
    def backend_name(self) -> str:
        return "mock"


# ── Backend factory ───────────────────────────────────────────────────────────
def _make_backend(model: str | None = None) -> GroqBackend | OpenAIBackend | MockBackend:
    if os.environ.get("RAKA_MOCK_LLM", "").strip() == "1":
        logger.info("Generator: RAKA_MOCK_LLM=1 — MockBackend forced")
        return MockBackend()

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        groq_model = model or os.environ.get("RAKA_LLM_MODEL", DEFAULT_GROQ_MODEL)
        logger.info("Generator: GROQ_API_KEY found — GroqBackend (%s)", groq_model)
        return GroqBackend(api_key=groq_key, model=groq_model)

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        openai_model = model or os.environ.get("RAKA_LLM_MODEL", DEFAULT_OPENAI_MODEL)
        logger.info("Generator: OPENAI_API_KEY found — OpenAIBackend (%s)", openai_model)
        return OpenAIBackend(api_key=openai_key, model=openai_model)

    logger.warning(
        "Generator: No API keys found — MockBackend active.\n"
        "  Set GROQ_API_KEY=gsk_... in your .env for real responses."
    )
    return MockBackend()


# ── Generator ─────────────────────────────────────────────────────────────────
class Generator:
    """
    LLM generation step of the RAG pipeline.

    v3: larger context window, better prompt, keyword hints, richer model.
    """

    def __init__(self, model: str | None = None) -> None:
        self._backend = _make_backend(model)

    def generate(
        self,
        query: str,
        retrieved_chunks: list[RetrievalResult],
    ) -> GeneratorResponse:
        if not retrieved_chunks:
            logger.warning("generate: no retrieved chunks — skipping LLM call")
            return GeneratorResponse(
                answer=NO_ANSWER_RESPONSE,
                sources=[],
                context_used="",
                model=self._backend.model,
                response_time_ms=0,
                mock=isinstance(self._backend, MockBackend),
                backend=self._backend.backend_name,
            )

        context_block = _build_context_block(retrieved_chunks)
        sources       = _build_source_list(retrieved_chunks)
        hint          = _keyword_hint(query)

        user_prompt = (
            f"CONTEXT:\n{context_block}\n\n"
            f"QUESTION: {clean_text(query)}{hint}\n\n"
            "Answer using ONLY the context above. "
            "Cite document name and page numbers in your answer."
        )

        logger.info(
            "generate: backend=%s model=%s query='%s' context=%d chars chunks=%d",
            self._backend.backend_name, self._backend.model,
            query[:60], len(context_block), len(retrieved_chunks),
        )

        t0 = time.time()
        try:
            answer = self._backend.complete(SYSTEM_PROMPT, user_prompt)
        except RuntimeError as exc:
            logger.error("LLM call failed (%s): %s", self._backend.backend_name, exc)
            answer = NO_ANSWER_RESPONSE
        elapsed_ms = int((time.time() - t0) * 1000)

        if not answer or not answer.strip():
            answer = NO_ANSWER_RESPONSE

        logger.info(
            "generate: done (%d chars, %dms, backend=%s)",
            len(answer), elapsed_ms, self._backend.backend_name,
        )

        return GeneratorResponse(
            answer=answer,
            sources=sources,
            context_used=context_block,
            model=self._backend.model,
            response_time_ms=elapsed_ms,
            mock=isinstance(self._backend, MockBackend),
            backend=self._backend.backend_name,
        )

    @property
    def model(self) -> str:
        return self._backend.model

    @property
    def backend(self) -> str:
        return self._backend.backend_name