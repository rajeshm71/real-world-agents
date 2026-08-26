"""Policy Q&A over a handbook: agent #10 of real-world-agents.

Technique demonstrated: **RAG with cross-field-verified citations,
on-prem/private.** Retrieve top-K chunks from a locally-embedded
corpus, ask the model to answer using ONLY those chunks with verbatim
citations, then reject any phantom citations or paraphrased quotes at
parse time. No LangChain, no LlamaIndex -- just fastembed +
sqlite-vec (FAISS fallback) + Anthropic + Pydantic. The reader sees
the whole RAG pipeline without a framework in the middle.

Why this technique for this use case: the handbook is small (dozens
of pages), the corpus is stable (updated occasionally, not per query),
and the privacy story is load-bearing (HR documents should not leave
the box for indexing). Fastembed runs the embedding model locally via
ONNX -- no vendor sees the handbook text at index time. Only the
retrieved top-K chunks + question ever reach Anthropic at query time.

Real error handling per R5 (three cases):
1. Corpus dir missing / empty / no supported files -> PolicyQAError
   before any embedding work.
2. Index missing when calling ask() -> PolicyQAError with a clear
   "run `ingest` first" pointer.
3. LLM API failure -> six-branch translator (class-name -> status ->
   message -> generic), same shape as agents #02-#09.

Provider stance: Anthropic-only in v1 (per the roadmap). Multi-provider
is a trivial add later since `_call_anthropic()` is the only site that
knows which SDK is in play; a v1.1 could swap in common.llm.get_llm().
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Dual-mode import: relative for tests-as-package, absolute for
# `python -m agent` from inside this dir. See #01's identical pattern.
try:
    from .schemas import Chunk, Citation, IngestionStats, PolicyAnswer
except ImportError:
    from schemas import Chunk, Citation, IngestionStats, PolicyAnswer

# --- Constants -------------------------------------------------------------

SUPPORTED_PROVIDERS = ("anthropic",)
_DEFAULT_PROVIDER = "anthropic"
_DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_CHARS = 2000
DEFAULT_CHUNK_OVERLAP_CHARS = 200
DEFAULT_INDEX_PATH = "policy_qa_index.db"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
_EMBEDDING_DIM = 384  # bge-small-en-v1.5 output dim
_SUPPORTED_EXTENSIONS = (".md", ".txt", ".pdf")
_MAX_TOKENS = 2048  # Anthropic response cap
_RETRY_LIMIT = 1  # JSON-validation retries; keep low since one call is expensive

_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.txt"


# --- Error type ------------------------------------------------------------


@dataclass
class RetrievalAttempt:
    """Partial state attached to PolicyQAError when answering fails
    partway through. `retrieved_chunks` shows what the retriever
    returned (may be non-empty even when the LLM call fails)."""

    stage: str  # 'ingest' | 'retrieve' | 'llm' | 'parse'
    retrieved_chunks: list[Chunk] | None = None
    raw_output: str = ""


class PolicyQAError(Exception):
    """Raised on any user-facing failure: bad corpus, missing index,
    LLM API error, or schema validation failure after retries."""

    def __init__(self, message: str, partial: RetrievalAttempt | None = None):
        super().__init__(message)
        self.message = message
        self.partial = partial


# --- Provider resolution ---------------------------------------------------


def resolve_provider() -> str:
    """LLM_PROVIDER env var, defaulting to anthropic. Anthropic-only in
    v1; "mock" is handled by the caller, not here."""
    provider = os.environ.get("LLM_PROVIDER", _DEFAULT_PROVIDER).lower()
    if provider != "mock" and provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            f"Expected 'mock' or one of {SUPPORTED_PROVIDERS}. "
            "Multi-provider is a documented follow-up for #10."
        )
    return provider


@functools.lru_cache(maxsize=1)
def _load_prompt() -> str:
    """Read the system prompt from disk once, cache across calls. Same
    pattern as #09; avoids a per-question re-read for batch use."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


# --- Chunking --------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _chunk_document(
    text: str,
    source_path: str,
    chunk_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    """Split text into fixed-char chunks with overlap. Prefers paragraph
    boundaries (double-newline) when splitting; captures nearest
    preceding Markdown heading as `section` metadata for citations.

    chunk_id format: `<basename>#chunk-N` where N is a zero-padded
    per-file counter. Multi-page PDFs use `<basename>#page-P-chunk-N`
    (assembled by the caller via a hint in `source_path`)."""
    if not text.strip():
        return []
    basename = Path(source_path).name
    chunks: list[Chunk] = []
    n = len(text)
    start = 0
    idx = 0
    while start < n:
        end = min(start + chunk_chars, n)
        # Prefer to break at a paragraph boundary within the last 20%
        # of the window; falls through to a hard cut if none found.
        if end < n:
            window_start = max(start + int(chunk_chars * 0.8), start + 1)
            para_break = text.rfind("\n\n", window_start, end)
            if para_break > start:
                end = para_break
        chunk_text = text[start:end].strip()
        if chunk_text:
            section = _nearest_heading(text, start)
            chunks.append(
                Chunk(
                    chunk_id=f"{basename}#chunk-{idx}",
                    source_path=source_path,
                    text=chunk_text,
                    section=section,
                )
            )
            idx += 1
        # Reached end of text -- stop rather than looping on
        # ever-shrinking tail windows that would produce N tiny chunks
        # for a short input.
        if end >= n:
            break
        # Advance with overlap; guard against infinite loop when end
        # equals start (empty chunk fallback).
        start = max(end - overlap_chars, start + 1)
    return chunks


def _nearest_heading(text: str, position: int) -> str | None:
    """Find the last `#`/`##`/etc. heading before `position` in text."""
    last = None
    for match in _HEADING_RE.finditer(text, 0, position):
        last = match.group(2).strip()
    return last


def _load_file_text(path: Path) -> str:
    """Read a supported file as plain text. PDFs go through pypdf and
    get per-page separators; other extensions read directly as UTF-8."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        try:
            import pypdf
        except ImportError as exc:
            raise PolicyQAError(
                f"pypdf not installed but {path} is a PDF. "
                "Run `uv sync` from the repo root to install this agent's deps."
            ) from exc
        reader = pypdf.PdfReader(str(path))
        pages = [f"--- PAGE {i + 1} ---\n{page.extract_text() or ''}" for i, page in enumerate(reader.pages)]
        return "\n\n".join(pages)
    return path.read_text(encoding="utf-8")


def _walk_corpus(corpus_dir: Path) -> list[Path]:
    """Return supported files under corpus_dir (recursive), sorted by
    path for deterministic ingestion order."""
    if not corpus_dir.exists():
        raise PolicyQAError(
            f"Corpus directory not found: {corpus_dir}. "
            "Pass a valid directory path to ingest_corpus()."
        )
    if not corpus_dir.is_dir():
        raise PolicyQAError(f"Corpus path is not a directory: {corpus_dir}.")
    files = sorted(
        p for p in corpus_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
    )
    if not files:
        raise PolicyQAError(
            f"No supported files found under {corpus_dir}. "
            f"Expected one of: {_SUPPORTED_EXTENSIONS}"
        )
    return files


# --- Vector store adapter (see stores.py) ---------------------------------

try:
    from .stores import get_store as _get_store
except ImportError:
    from stores import get_store as _get_store


# --- Embeddings ------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _load_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Load fastembed's TextEmbedding lazily + cached. Lazy so mock-
    mode CI doesn't need fastembed installed; cached so a batch tool
    doesn't reload the ONNX model per call."""
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise PolicyQAError(
            "fastembed not installed. Run `uv sync` from the repo root, "
            "or `pip install fastembed>=0.8` in this agent's environment."
        ) from exc
    return TextEmbedding(model_name=model_name)


# --- Public API: ingest_corpus --------------------------------------------


def ingest_corpus(
    corpus_dir: str | Path,
    *,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    provider: str | None = None,
) -> IngestionStats:
    """Walk corpus_dir, chunk each file, embed via fastembed, insert
    into sqlite-vec (or FAISS fallback). Returns IngestionStats with
    the file/chunk counts + backend name + on-disk index size.

    Under provider="mock" returns canned stats without importing
    fastembed or writing to disk (CI-friendly).
    """
    resolved_provider = (provider or resolve_provider()).lower()
    corpus_dir = Path(corpus_dir)

    if resolved_provider == "mock":
        return _mock_ingestion_stats(corpus_dir, embedding_model, index_path)

    if chunk_chars < 100:
        raise ValueError(f"chunk_chars must be >= 100, got {chunk_chars}")
    if chunk_overlap_chars < 0 or chunk_overlap_chars >= chunk_chars:
        raise ValueError(
            f"chunk_overlap_chars must be in [0, chunk_chars), "
            f"got {chunk_overlap_chars} with chunk_chars={chunk_chars}"
        )

    files = _walk_corpus(corpus_dir)
    all_chunks: list[Chunk] = []
    for path in files:
        text = _load_file_text(path)
        rel = str(path.relative_to(corpus_dir))
        all_chunks.extend(_chunk_document(text, rel, chunk_chars, chunk_overlap_chars))

    if not all_chunks:
        raise PolicyQAError(
            f"Found {len(files)} file(s) under {corpus_dir} but they produced "
            "zero chunks (all empty?). Check the source content."
        )

    embedder = _load_embedder(embedding_model)
    embeddings = list(embedder.embed([c.text for c in all_chunks]))

    index_path = Path(index_path)
    store = _get_store(index_path)
    try:
        store.insert_batch(all_chunks, embeddings)
        backend = store.backend_name
        size = store.on_disk_size_bytes
    finally:
        store.close()

    return IngestionStats(
        total_files=len(files),
        total_chunks=len(all_chunks),
        index_size_bytes=size,
        embedding_model=embedding_model,
        index_path=str(index_path),
        vector_store_backend=backend,
    )


# --- Public API: ask ------------------------------------------------------


def ask(
    question: str,
    *,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    top_k: int = DEFAULT_TOP_K,
    provider: str | None = None,
    model: str | None = None,
    _client=None,
) -> PolicyAnswer:
    """Embed the question, retrieve top-K chunks, ask Anthropic to
    answer with citations, validate the response. Returns a
    PolicyAnswer with citations cross-checked at parse time.

    Under provider="mock" returns a canned PolicyAnswer without
    embedding, retrieval, or an LLM call.
    """
    if not question or not question.strip():
        raise PolicyQAError("question is empty. Pass a non-empty string.")

    resolved_provider = (provider or resolve_provider()).lower()

    if resolved_provider == "mock":
        return _mock_policy_answer(question)

    resolved_model = model or _DEFAULT_MODEL
    index_path = Path(index_path)
    if not index_path.exists():
        raise PolicyQAError(
            f"Index not found at {index_path}. Run "
            f"`python -m agent ingest <corpus_dir>` first to build the index."
        )

    embedder = _load_embedder()
    query_embedding = next(iter(embedder.embed([question])))

    store = _get_store(index_path)
    try:
        retrieved = store.search(query_embedding, top_k=top_k)
    finally:
        store.close()

    if not retrieved:
        raise PolicyQAError(
            f"Retrieval returned zero chunks for question: {question!r}. "
            "Is the index empty? Try re-ingesting."
        )

    retrieved_chunks = [chunk for chunk, _dist in retrieved]

    try:
        raw = _call_anthropic(
            client=_client,
            model=resolved_model,
            question=question,
            retrieved_chunks=retrieved_chunks,
        )
    except Exception as exc:
        raise _translate_api_error(
            exc, partial=RetrievalAttempt(stage="llm", retrieved_chunks=retrieved_chunks)
        ) from exc

    try:
        return _parse_and_validate(raw, question, retrieved_chunks)
    except (json.JSONDecodeError, ValueError) as exc:
        raise PolicyQAError(
            f"Model returned output that didn't parse as a valid PolicyAnswer: {exc}. "
            "Check .partial.raw_output for what was produced.",
            partial=RetrievalAttempt(
                stage="parse", retrieved_chunks=retrieved_chunks, raw_output=raw
            ),
        ) from exc


def _call_anthropic(
    *, client, model: str, question: str, retrieved_chunks: list[Chunk]
) -> str:
    """One Anthropic Messages call. `client` is either a real
    anthropic.Anthropic (or None -> lazy-construct one) or a test
    double with a .messages.create() method."""
    if client is None:
        try:
            import anthropic
        except ImportError as exc:
            raise PolicyQAError(
                "anthropic SDK not installed. Run `uv sync` from the repo root."
            ) from exc
        client = anthropic.Anthropic()

    passages = "\n\n".join(
        f"[{c.chunk_id}]" + (f" (section: {c.section})" if c.section else "") + f"\n{c.text}"
        for c in retrieved_chunks
    )
    user_message = f"Question: {question}\n\nRetrieved passages:\n\n{passages}"
    response = client.messages.create(
        model=model,
        max_tokens=_MAX_TOKENS,
        system=_load_prompt(),
        messages=[{"role": "user", "content": user_message}],
    )
    # Anthropic returns list of content blocks; the text is in .text of
    # the first TextBlock (there's always one for a completion response).
    text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    if not text_blocks:
        raise PolicyQAError("Anthropic response had no text content blocks.")
    return text_blocks[0]


def _parse_and_validate(
    raw: str, question: str, retrieved_chunks: list[Chunk]
) -> PolicyAnswer:
    """Parse the model's JSON output and construct a PolicyAnswer.
    The schema's validators enforce citation honesty (no phantoms,
    verbatim quotes, low-confidence-requires-reason)."""
    # Strip common wrappers models sometimes add despite instructions.
    stripped = raw.strip()
    if stripped.startswith("```"):
        # Remove opening fence line + optional language tag, and closing fence.
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
    data = json.loads(stripped)
    # Model wasn't asked to include these two (they're the caller's
    # responsibility per the prompt), so add them before validation.
    data["question"] = question
    data["retrieved_chunks"] = [c.model_dump() for c in retrieved_chunks]
    return PolicyAnswer.model_validate(data)


# --- Error translation (R5 case 3) -----------------------------------------


def _translate_api_error(
    exc: Exception, *, partial: RetrievalAttempt | None = None
) -> PolicyQAError:
    """Turn an Anthropic SDK exception into a user-facing PolicyQAError.
    Priority order: class-name -> status-code -> message-string ->
    generic. Same shape as agents #02-#09."""
    exc_class_name = type(exc).__name__.lower()
    message_lower = str(exc).lower()
    status = getattr(exc, "status_code", None)
    if "ratelimiterror" in exc_class_name:
        return _rate_limit_error(partial)
    if "authenticationerror" in exc_class_name or "apikeyerror" in exc_class_name:
        return _auth_error(partial)
    if status == 429:
        return _rate_limit_error(partial)
    if status == 401:
        return _auth_error(partial)
    if "rate limit" in message_lower or "overloaded" in message_lower:
        return _rate_limit_error(partial)
    if "authentication" in message_lower or "api key" in message_lower:
        return _auth_error(partial)
    return PolicyQAError(
        f"LLM call failed: {type(exc).__name__}: {exc}. "
        "Unexpected error; retrieved chunks (if any) are attached as partial.",
        partial=partial,
    )


def _rate_limit_error(partial: RetrievalAttempt | None) -> PolicyQAError:
    return PolicyQAError(
        "Anthropic is temporarily rate-limited or overloaded. "
        "Wait a minute and try again.",
        partial=partial,
    )


def _auth_error(partial: RetrievalAttempt | None) -> PolicyQAError:
    return PolicyQAError(
        "Authentication failed: check that ANTHROPIC_API_KEY is set. "
        "See .env.example at the repo root.",
        partial=partial,
    )


# --- Mock mode -------------------------------------------------------------


def _mock_ingestion_stats(
    corpus_dir: Path, embedding_model: str, index_path: str | Path
) -> IngestionStats:
    """Canned IngestionStats without touching fastembed or the vector
    store. Counts real supported files under corpus_dir if the dir
    exists (so the mock stats have honest file counts), otherwise
    reports zero. Fake chunk count so a test can assert."""
    files = (
        [p for p in corpus_dir.rglob("*") if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS]
        if corpus_dir.exists() else []
    )
    return IngestionStats(
        total_files=len(files),
        total_chunks=max(len(files) * 3, 1),  # ~3 chunks per file
        index_size_bytes=0,
        embedding_model=f"[MOCK] {embedding_model}",
        index_path=str(index_path),
        vector_store_backend="sqlite-vec",
    )


def _mock_policy_answer(question: str) -> PolicyAnswer:
    """Canned PolicyAnswer with one fake retrieved chunk + citation.
    Question length is echoed into the answer so a test can prove the
    mock saw its input (anti-refactor guard convention agent #01 uses)."""
    fake_chunk = Chunk(
        chunk_id="benefits.md#chunk-0",
        source_path="benefits.md",
        text=(
            "Full-time employees are enrolled in the company PPO health plan "
            "effective the first of the month following their start date."
        ),
        section="Health Insurance",
    )
    return PolicyAnswer(
        question=question,
        answer=(
            f"[MOCK answer for question of length {len(question)}] "
            "According to the handbook, full-time employees get PPO health "
            "coverage on the first of the month after start date "
            "[benefits.md#chunk-0]."
        ),
        citations=[
            Citation(
                chunk_id="benefits.md#chunk-0",
                quoted_text="Full-time employees are enrolled in the company PPO health plan",
            )
        ],
        retrieved_chunks=[fake_chunk],
        confidence="high",
        unanswered_reason=None,
    )


# --- CLI entry point ------------------------------------------------------


def main() -> int:
    """CLI with two subcommands: `ingest <dir>` and `ask <question>`."""
    parser = argparse.ArgumentParser(
        prog="policy-qa",
        description=(
            "RAG over a company handbook with cross-field-verified citations. "
            "Set LLM_PROVIDER=mock for a canned demo, or supply "
            "ANTHROPIC_API_KEY for real answers."
        ),
    )
    parser.add_argument("--ui", action="store_true", help="Launch Gradio UI.")
    subs = parser.add_subparsers(dest="command", metavar="{ingest,ask}")

    ingest_p = subs.add_parser("ingest", help="Ingest a corpus into the vector index.")
    ingest_p.add_argument("corpus_dir", type=Path)
    ingest_p.add_argument("--index", type=Path, default=Path(DEFAULT_INDEX_PATH))

    ask_p = subs.add_parser("ask", help="Ask a question against the ingested corpus.")
    ask_p.add_argument("question", type=str)
    ask_p.add_argument("--index", type=Path, default=Path(DEFAULT_INDEX_PATH))
    ask_p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ask_p.add_argument("--model", type=str, default=None)
    ask_p.add_argument(
        "--provider", choices=(*SUPPORTED_PROVIDERS, "mock"), default=None
    )

    args = parser.parse_args()

    if args.ui:
        try:
            from .ui import build_ui
        except ImportError:
            from ui import build_ui
        build_ui().launch()
        return 0

    if args.command == "ingest":
        return _run_ingest_cli(args)
    if args.command == "ask":
        return _run_ask_cli(args)
    parser.error("must supply a subcommand (ingest or ask) or --ui")
    return 2  # unreachable


def _run_ingest_cli(args) -> int:
    try:
        stats = ingest_corpus(args.corpus_dir, index_path=args.index)
    except PolicyQAError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1
    print(f"Files ingested:  {stats.total_files}")
    print(f"Chunks created:  {stats.total_chunks}")
    print(f"Backend:         {stats.vector_store_backend}")
    print(f"Index size:      {stats.index_size_bytes / 1024:.1f} KB")
    print(f"Index path:      {stats.index_path}")
    return 0


def _run_ask_cli(args) -> int:
    start = time.perf_counter()
    try:
        answer = ask(
            args.question,
            index_path=args.index,
            top_k=args.top_k,
            provider=args.provider,
            model=args.model,
        )
    except PolicyQAError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.partial is not None and exc.partial.raw_output:
            print(f"raw model output:\n{exc.partial.raw_output}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - start
    out_path = Path(__file__).parent / "last_run.json"
    out_path.write_text(
        json.dumps(answer.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    print(f"Question:    {answer.question}")
    print(f"Confidence:  {answer.confidence}")
    print(f"Wall time:   {elapsed:.1f}s")
    print()
    print("Answer:")
    print(answer.answer)
    print()
    print(f"Citations ({len(answer.citations)}):")
    for c in answer.citations:
        print(f"  [{c.chunk_id}] {c.quoted_text[:80]}...")
    print()
    print(f"Full JSON written to {out_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
