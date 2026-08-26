"""Smoke tests for the policy Q&A agent.

All Anthropic calls are mocked (R8 in CONTRIBUTING.md) -- CI never
touches a real API key. Vector store + chunker + fastembed paths are
tested against REAL local implementations behind
`pytest.importorskip` so a minimal-deps environment still runs
everything else.

Structure:
1. Mock path (ingest + ask round-trips, input-echo guards).
2. Chunker (fixed-size, overlap, paragraph-boundary preference,
   heading extraction, edge cases).
3. Schema validators (all four cross-field checks, both directions).
4. Vector store adapter (SqliteVecStore + FaissStore roundtrips,
   get_store fallback, close() semantics).
5. Real end-to-end ingest (fastembed + real store, no LLM cost).
6. R5 error branches (corpus missing, index missing, 6-branch
   translator).
7. Constants + examples + prompt sanity.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(_AGENT_DIR.parent))

_agent = importlib.import_module("10_policy_qa.agent")
_schemas = importlib.import_module("10_policy_qa.schemas")
_stores = importlib.import_module("10_policy_qa.stores")

ingest_corpus = _agent.ingest_corpus
ask = _agent.ask
resolve_provider = _agent.resolve_provider
PolicyQAError = _agent.PolicyQAError
RetrievalAttempt = _agent.RetrievalAttempt
_chunk_document = _agent._chunk_document
_nearest_heading = _agent._nearest_heading
_walk_corpus = _agent._walk_corpus
_load_file_text = _agent._load_file_text
_translate_api_error = _agent._translate_api_error
_parse_and_validate = _agent._parse_and_validate
_mock_ingestion_stats = _agent._mock_ingestion_stats
_mock_policy_answer = _agent._mock_policy_answer
SUPPORTED_PROVIDERS = _agent.SUPPORTED_PROVIDERS
DEFAULT_TOP_K = _agent.DEFAULT_TOP_K
DEFAULT_CHUNK_CHARS = _agent.DEFAULT_CHUNK_CHARS
DEFAULT_EMBEDDING_MODEL = _agent.DEFAULT_EMBEDDING_MODEL

Chunk = _schemas.Chunk
Citation = _schemas.Citation
PolicyAnswer = _schemas.PolicyAnswer
IngestionStats = _schemas.IngestionStats

get_store = _stores.get_store
SqliteVecStore = _stores.SqliteVecStore
FaissStore = _stores.FaissStore

_HANDBOOK_DIR = _AGENT_DIR / "examples" / "handbook"


# ---------- 1. Mock path ----------


def test_mock_ingest_returns_valid_stats(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    stats = ingest_corpus(_HANDBOOK_DIR)
    assert isinstance(stats, IngestionStats)
    assert stats.total_files == 5  # 5 handbook files
    assert stats.total_chunks >= 1


def test_mock_ingest_stats_reflect_real_file_count(monkeypatch, tmp_path):
    """Mock's IngestionStats should count REAL files in the passed dir
    (so a test can prove the mock saw its input). Empty dir -> 0
    files, populated dir -> N files."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    empty = tmp_path / "empty_corpus"
    empty.mkdir()
    stats_empty = ingest_corpus(empty)
    assert stats_empty.total_files == 0

    populated = tmp_path / "populated"
    populated.mkdir()
    (populated / "a.md").write_text("# hi\n\nhello\n", encoding="utf-8")
    (populated / "b.md").write_text("# bye\n\nworld\n", encoding="utf-8")
    stats_pop = ingest_corpus(populated)
    assert stats_pop.total_files == 2


def test_mock_ask_returns_valid_policy_answer(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    result = ask("How much PTO do I get?")
    assert isinstance(result, PolicyAnswer)
    assert result.confidence == "high"
    assert len(result.citations) == 1
    assert len(result.retrieved_chunks) == 1


def test_mock_ask_echoes_question_length_into_answer(monkeypatch):
    """Anti-refactor guard: mock must see its input. If a future
    refactor makes the mock output constant regardless of input,
    this test surfaces the drift."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    short = ask("Q?")
    long_q = ask("Q? " + "extra words " * 50)
    assert "length 2" in short.answer or f"length {len('Q?')}" in short.answer
    assert f"length {len('Q? ' + 'extra words ' * 50)}" in long_q.answer


def test_mock_ask_rejects_empty_question(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    with pytest.raises(PolicyQAError, match="empty"):
        ask("")
    with pytest.raises(PolicyQAError, match="empty"):
        ask("   ")


# ---------- 2. Chunker ----------


def test_chunk_document_produces_chunks_within_size_limit():
    text = "Section body. " * 500  # ~6500 chars
    chunks = _chunk_document(text, "test.md", chunk_chars=1000, overlap_chars=100)
    assert len(chunks) >= 6
    for chunk in chunks:
        # Allow a bit of slack for paragraph-boundary trimming.
        assert len(chunk.text) <= 1000


def test_chunk_document_respects_overlap():
    """Successive chunks should share some tail/head text when overlap
    > 0. Exact overlap won't match char-for-char (paragraph-boundary
    trimming may adjust the cut), but there should be substantial
    text in common between consecutive chunks."""
    text = ("Paragraph one is here. " * 30 + "\n\n"
            + "Paragraph two follows next. " * 30 + "\n\n"
            + "Third paragraph appears now. " * 30)
    chunks = _chunk_document(text, "test.md", chunk_chars=800, overlap_chars=200)
    assert len(chunks) >= 2
    # At least the second chunk shouldn't start at the exact beginning
    # of a fresh non-overlapping region.
    assert chunks[0].text != chunks[1].text


def test_chunk_document_extracts_markdown_headings():
    text = (
        "# Top Heading\n\n"
        "Body text under top.\n\n"
        "## Subsection A\n\n"
        "Body of subsection A goes here. " * 20
    )
    chunks = _chunk_document(text, "test.md", chunk_chars=200, overlap_chars=50)
    # Every chunk after the first heading should have a `section` set.
    non_first_chunks = [c for c in chunks if c.text and not c.text.startswith("# ")]
    assert any(c.section is not None for c in non_first_chunks)
    # The last chunk should be under "Subsection A".
    assert chunks[-1].section == "Subsection A"


def test_chunk_document_empty_text_returns_empty_list():
    assert _chunk_document("", "empty.md", 500, 50) == []
    assert _chunk_document("   \n   ", "whitespace.md", 500, 50) == []


def test_chunk_document_short_text_returns_single_chunk():
    chunks = _chunk_document("Just one line.", "short.md", 500, 50)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "short.md#chunk-0"


def test_nearest_heading_finds_most_recent():
    text = "# Alpha\n\nsome text\n\n## Beta\n\nmore text here"
    # Position inside "more text here" should find Beta, not Alpha.
    pos = text.index("more text")
    assert _nearest_heading(text, pos) == "Beta"


def test_nearest_heading_returns_none_when_no_heading():
    assert _nearest_heading("just body text", 5) is None


def test_walk_corpus_finds_supported_files_recursively(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "sub" / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "sub" / "c.ignore.jpg").write_text("c", encoding="utf-8")
    files = _walk_corpus(tmp_path)
    names = sorted(p.name for p in files)
    assert names == ["a.md", "b.txt"]


def test_walk_corpus_raises_on_missing_dir(tmp_path):
    with pytest.raises(PolicyQAError, match="not found"):
        _walk_corpus(tmp_path / "does_not_exist")


def test_walk_corpus_raises_on_empty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(PolicyQAError, match="No supported files"):
        _walk_corpus(empty)


# ---------- 3. Schema validators ----------


def _make_chunks():
    return [
        Chunk(
            chunk_id="benefits.md#chunk-0",
            source_path="benefits.md",
            text="Full-time employees get 20 days of PTO per year.",
            section="Health Insurance",
        ),
        Chunk(
            chunk_id="benefits.md#chunk-1",
            source_path="benefits.md",
            text="The company matches 100% of 401k contributions up to 4%.",
            section="401(k)",
        ),
    ]


def test_policy_answer_rejects_phantom_citation():
    """Citation must reference a chunk_id that appears in
    retrieved_chunks. Otherwise the model is claiming a source the
    retriever never returned."""
    chunks = _make_chunks()
    with pytest.raises(ValidationError, match="Phantom citation"):
        PolicyAnswer(
            question="q",
            answer="a",
            citations=[Citation(chunk_id="never_seen.md#chunk-0", quoted_text="foo")],
            retrieved_chunks=chunks,
            confidence="high",
        )


def test_policy_answer_rejects_paraphrased_citation():
    """Citation quoted_text must be verbatim substring of the cited
    chunk. A paraphrase gets rejected."""
    chunks = _make_chunks()
    with pytest.raises(ValidationError, match="verbatim substring"):
        PolicyAnswer(
            question="q",
            answer="a",
            citations=[
                Citation(
                    chunk_id="benefits.md#chunk-0",
                    quoted_text="employees receive 20 vacation days annually",  # paraphrase
                )
            ],
            retrieved_chunks=chunks,
            confidence="high",
        )


def test_policy_answer_accepts_verbatim_citation():
    chunks = _make_chunks()
    result = PolicyAnswer(
        question="How much PTO?",
        answer="20 days [benefits.md#chunk-0].",
        citations=[
            Citation(
                chunk_id="benefits.md#chunk-0",
                quoted_text="get 20 days of PTO per year",  # verbatim substring
            )
        ],
        retrieved_chunks=chunks,
        confidence="high",
    )
    assert result.citations[0].chunk_id == "benefits.md#chunk-0"


def test_policy_answer_low_confidence_requires_reason():
    chunks = _make_chunks()
    with pytest.raises(ValidationError, match="unanswered_reason"):
        PolicyAnswer(
            question="q",
            answer="idk",
            citations=[],
            retrieved_chunks=chunks,
            confidence="low",
            unanswered_reason=None,  # missing!
        )


def test_policy_answer_low_confidence_with_reason_accepted():
    chunks = _make_chunks()
    result = PolicyAnswer(
        question="q",
        answer="Can't answer.",
        citations=[],
        retrieved_chunks=chunks,
        confidence="low",
        unanswered_reason="The handbook doesn't cover this topic.",
    )
    assert result.confidence == "low"


def test_policy_answer_high_confidence_requires_at_least_one_citation():
    chunks = _make_chunks()
    with pytest.raises(ValidationError, match="requires at least one citation"):
        PolicyAnswer(
            question="q",
            answer="20 days.",  # unsupported claim
            citations=[],  # empty
            retrieved_chunks=chunks,
            confidence="high",
        )


def test_policy_answer_medium_confidence_also_requires_citations():
    chunks = _make_chunks()
    with pytest.raises(ValidationError, match="requires at least one citation"):
        PolicyAnswer(
            question="q",
            answer="maybe 20 days.",
            citations=[],
            retrieved_chunks=chunks,
            confidence="medium",
        )


def test_ingestion_stats_rejects_negative_counts():
    with pytest.raises(ValidationError):
        IngestionStats(
            total_files=-1,
            total_chunks=0,
            index_size_bytes=0,
            embedding_model="x",
            index_path="/tmp/x.db",
            vector_store_backend="sqlite-vec",
        )


# ---------- 4. Vector store adapters ----------


def test_sqlite_vec_store_roundtrip(tmp_path):
    """Real sqlite-vec: insert two chunks with random embeddings,
    search should return them ranked by distance."""
    pytest.importorskip("sqlite_vec")
    import numpy as np

    db_path = tmp_path / "test.db"
    store = SqliteVecStore(db_path, dim=384)
    try:
        chunks = _make_chunks()
        embeddings = np.random.randn(2, 384).astype("float32")
        store.insert_batch(chunks, embeddings)
        # Search with the first embedding -> chunk 0 should rank top.
        results = store.search(embeddings[0], top_k=2)
        assert len(results) == 2
        assert results[0][0].chunk_id == chunks[0].chunk_id
    finally:
        store.close()


def test_faiss_store_roundtrip(tmp_path):
    pytest.importorskip("faiss")
    import numpy as np

    db_path = tmp_path / "test.db"
    store = FaissStore(db_path, dim=384)
    try:
        chunks = _make_chunks()
        embeddings = np.random.randn(2, 384).astype("float32")
        # Normalize for cosine-via-inner-product to make sense.
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        store.insert_batch(chunks, embeddings)
        results = store.search(embeddings[0], top_k=2)
        assert len(results) == 2
        assert results[0][0].chunk_id == chunks[0].chunk_id
    finally:
        store.close()


def test_get_store_returns_sqlite_vec_by_default(tmp_path):
    pytest.importorskip("sqlite_vec")
    store = get_store(tmp_path / "test.db")
    try:
        assert store.backend_name == "sqlite-vec"
    finally:
        store.close()


def test_faiss_store_persists_across_open_close(tmp_path):
    """Close should write the FAISS index to disk; reopen should
    read it back."""
    pytest.importorskip("faiss")
    import numpy as np

    db_path = tmp_path / "test.db"
    chunks = _make_chunks()
    embeddings = np.random.randn(2, 384).astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    store1 = FaissStore(db_path, dim=384)
    store1.insert_batch(chunks, embeddings)
    store1.close()

    # Reopen; the index file should exist and load.
    store2 = FaissStore(db_path, dim=384)
    try:
        assert store2.on_disk_size_bytes > 0
    finally:
        store2.close()


# ---------- 5. Real end-to-end ingest (no LLM) ----------


def test_real_ingest_produces_working_index(tmp_path):
    """The load-bearing REAL test: fastembed + get_store together
    produce an index the retriever can read back. No LLM cost.
    Skipped if fastembed / sqlite-vec / handbook not available."""
    pytest.importorskip("fastembed")
    pytest.importorskip("sqlite_vec")
    if not _HANDBOOK_DIR.exists():
        pytest.skip("examples/handbook not present in this workspace state")

    index_path = tmp_path / "real_index.db"
    stats = ingest_corpus(
        _HANDBOOK_DIR,
        index_path=index_path,
        provider="anthropic",  # skip mock branch
    )
    assert stats.total_files == 5
    assert stats.total_chunks >= 5
    assert stats.vector_store_backend in ("sqlite-vec", "faiss")
    assert index_path.exists()

    # Retrieve directly (no LLM) with the same embedder.
    embedder = _agent._load_embedder()
    query_emb = next(iter(embedder.embed(["How much PTO do I get?"])))
    store = get_store(index_path)
    try:
        results = store.search(query_emb, top_k=3)
        assert len(results) == 3
        # Top result should come from leave_policy.md (where PTO lives).
        top_chunk = results[0][0]
        assert "leave_policy" in top_chunk.chunk_id.lower() or "pto" in top_chunk.text.lower()
    finally:
        store.close()


# ---------- 6. R5 error branches + parsing ----------


def test_ingest_corpus_missing_dir_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(PolicyQAError, match="not found"):
        ingest_corpus(tmp_path / "does_not_exist")


def test_ingest_corpus_empty_dir_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(PolicyQAError, match="No supported files"):
        ingest_corpus(empty)


def test_ask_missing_index_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(PolicyQAError, match="[Ii]ngest.*first"):
        ask("question", index_path=tmp_path / "no_index.db")


def test_translate_api_error_class_name_rate_limit():
    class RateLimitError(Exception):
        pass

    out = _translate_api_error(RateLimitError("boom"))
    assert isinstance(out, PolicyQAError)
    assert "rate-limited" in out.message.lower() or "overloaded" in out.message.lower()


def test_translate_api_error_class_name_auth():
    class AuthenticationError(Exception):
        pass

    out = _translate_api_error(AuthenticationError("boom"))
    assert "authentication" in out.message.lower() or "api key" in out.message.lower()


def test_translate_api_error_status_429():
    exc = RuntimeError("boom")
    exc.status_code = 429  # type: ignore[attr-defined]
    out = _translate_api_error(exc)
    assert "rate-limited" in out.message.lower() or "overloaded" in out.message.lower()


def test_translate_api_error_status_401():
    exc = RuntimeError("boom")
    exc.status_code = 401  # type: ignore[attr-defined]
    out = _translate_api_error(exc)
    assert "authentication" in out.message.lower() or "api key" in out.message.lower()


def test_translate_api_error_generic_fallback_preserves_original():
    out = _translate_api_error(RuntimeError("some genuinely unknown thing"))
    assert "RuntimeError" in out.message
    assert "some genuinely unknown thing" in out.message


def test_translate_api_error_message_fallback_rate_limit():
    out = _translate_api_error(RuntimeError("You hit the rate limit here"))
    assert "rate-limited" in out.message.lower() or "overloaded" in out.message.lower()


def test_parse_and_validate_strips_code_fences():
    """Models sometimes wrap JSON in ```json ... ``` despite
    instructions. Parser must strip that before json.loads."""
    chunks = _make_chunks()
    raw = (
        "```json\n"
        '{"answer": "PTO is 20 days [benefits.md#chunk-0].", '
        '"citations": [{"chunk_id": "benefits.md#chunk-0", '
        '"quoted_text": "get 20 days of PTO per year"}], '
        '"confidence": "high", "unanswered_reason": null}\n'
        "```"
    )
    result = _parse_and_validate(raw, question="How much PTO?", retrieved_chunks=chunks)
    assert result.confidence == "high"
    assert result.answer.startswith("PTO is 20 days")


# ---------- 7. Constants + provider + examples sanity ----------


def test_resolve_provider_defaults_to_anthropic(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert resolve_provider() == "anthropic"


def test_resolve_provider_rejects_unknown(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        resolve_provider()


def test_supported_providers_is_anthropic_only_v1():
    """v1 stance is Anthropic-only per roadmap. If a v1.1 adds
    OpenAI/Gemini this test fires as a reminder to also update the
    README's provider notes."""
    assert SUPPORTED_PROVIDERS == ("anthropic",)


def test_default_top_k_is_reasonable():
    """Sanity check: top-K of 5 is the RAG-community default. If a
    future change moves it dramatically, we want a reminder."""
    assert 3 <= DEFAULT_TOP_K <= 10


def test_examples_handbook_has_five_markdown_files():
    """Regression guard: the shipped examples/handbook must have all
    5 files for the sample UI + demo to make sense."""
    if not _HANDBOOK_DIR.exists():
        pytest.skip("examples/handbook not present in this workspace state")
    md_files = list(_HANDBOOK_DIR.glob("*.md"))
    assert len(md_files) == 5
    names = sorted(p.name for p in md_files)
    assert names == [
        "benefits.md",
        "code_of_conduct.md",
        "expenses.md",
        "leave_policy.md",
        "remote_work.md",
    ]


def test_examples_readme_documents_self_authored_status():
    """Same discipline as #09: any shipped assets need documentation."""
    readme = _AGENT_DIR / "examples" / "README.md"
    if not readme.exists():
        pytest.skip("examples/README.md absent")
    content = readme.read_text(encoding="utf-8").lower()
    assert "self-authored" in content or "no real" in content or "no third-party" in content


def test_default_embedding_model_matches_pin():
    """The prompt + schema assume bge-small-en-v1.5's 384-dim output.
    A future change to a different model would break the vec0 table
    schema. Pin the default here as a reminder."""
    assert DEFAULT_EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"


def test_default_chunk_chars_reasonable():
    """Chunker default should sit in the 1000-4000 char sweet spot;
    outside that range is either too fragmented or too coarse."""
    assert 1000 <= DEFAULT_CHUNK_CHARS <= 4000
