# Policy Q&A over a handbook (RAG with citations, on-prem/private)

Point the agent at a directory of `.md` / `.txt` / `.pdf` handbook files. Ask a question. Get an answer with verbatim citations that link every claim back to a specific chunk of the source. Handbook contents are embedded locally (no vendor sees them at index time) via fastembed + ONNX; only the retrieved top-K chunks + your question ever reach the LLM at query time. A deployable OSS RAG demo for anyone who wants the "retrieve then generate with verified citations" pattern without dragging LangChain or LlamaIndex into their codebase.

## Verification status

| Path | Status |
|---|---|
| Mock mode (`LLM_PROVIDER=mock`) | Fully covered by 46-test suite |
| Chunker (fixed-size, overlap, heading extraction, edge cases) | Fully covered |
| Schema validators (phantom citations, verbatim quotes, low-confidence + reason, high/medium + citations) | Fully covered, both directions |
| Vector store adapters (SqliteVecStore + FaissStore roundtrip) | Fully covered, real subprocess tests behind `pytest.importorskip` |
| **Real end-to-end ingestion** (fastembed + sqlite-vec on the shipped handbook + semantic retrieval) | **Verified** by `test_real_ingest_produces_working_index` -- pointing at `examples/handbook/` produces a working index and the retriever correctly returns the leave_policy.md chunk for a PTO question. No LLM cost. |
| Real Anthropic answering (`LLM_PROVIDER=anthropic` + key) | **Not yet verified against a live API call.** Structural correctness proven via the mock + parse tests. Same open-item status as every other agent's first ship. |

## Technique demonstrated

**Retrieval-augmented generation (RAG) with cross-field-verified citations, on-prem/private, no framework.** Three moving parts:

1. **Ingestion**: walk a corpus directory, chunk each document (fixed-char with overlap, paragraph-boundary aware, Markdown heading captured as metadata), embed each chunk locally via fastembed's ONNX-based `BAAI/bge-small-en-v1.5` (~30 MB model, no torch), insert into a vector store.
2. **Vector store adapter**: sqlite-vec primary (single `.db` file, inspectable with the sqlite3 CLI), FAISS fallback when sqlite-vec's extension load fails. `stores.py` isolates the backend-specific SQL and index plumbing behind a shared `VectorStore` Protocol.
3. **Answering**: embed the question, retrieve top-K chunks, prompt Anthropic to answer using ONLY those chunks with `[chunk_id]` inline markers and verbatim `quoted_text` snippets. Parse the JSON response. Validate at parse time via three Pydantic model validators that reject phantom citations (citing chunks the retriever never returned), paraphrased quotes (quoted_text that isn't a verbatim substring of the cited chunk), and unexplained low-confidence answers.

Distinct from every earlier agent: this is the first RAG pipeline in the catalog. #01 through #09 all operate on user-supplied input as the sole context; #10 introduces a persistent corpus + retrieval step.

## Why this technique for this use case

Policy Q&A over a handbook is exactly the shape RAG is best at: small corpus (dozens of pages), stable content (updated occasionally, not per query), answers should be verifiable by pointing the user at the exact source passage, and the whole thing should work without sending the handbook to a vendor for indexing.

**Why local embeddings**: privacy is load-bearing for HR content. fastembed's ONNX runtime keeps everything on the box; no vendor sees the handbook text at index time. Only the retrieved chunks + question reach Anthropic at query time (~few hundred tokens of handbook per query, not the whole doc).

**Why no framework**: LangChain and LlamaIndex give you RAG in one line but hide all the interesting decisions. This agent shows you the whole pipeline in 640 LOC across two files (`agent.py` + `stores.py`). Copy the shape into your own project when you need retrieval-with-citations somewhere; you'll know exactly what each piece does.

Where this technique is NOT the right fit: (a) very large corpora (millions of chunks -- sqlite-vec starts to hurt above ~100K, switch to a real vector DB); (b) reasoning-heavy questions where the answer requires synthesizing across many chunks (RAG's flat top-K retrieval misses cross-chunk relationships); (c) rapidly changing corpora where re-ingesting per query would dominate cost.

## What it does

**Ingestion** (`ingest_corpus`): walks a corpus directory, chunks each supported file, embeds via fastembed, writes to a sqlite-vec or FAISS index. Returns an `IngestionStats` with file/chunk counts + backend + on-disk size.

**Answering** (`ask`): embeds the question, retrieves top-K chunks from the index, calls Anthropic with a structured prompt, parses the JSON response, cross-validates every citation. Returns a `PolicyAnswer` with:

- `question` (echoed back)
- `answer` (plain-English text with inline `[chunk_id]` markers)
- `citations` (list of `{chunk_id, quoted_text}`, each cross-checked at parse time)
- `retrieved_chunks` (the top-K raw retrieval as an audit trail)
- `confidence` (`"high"` | `"medium"` | `"low"`)
- `unanswered_reason` (required when confidence=low, so the model must explain what's missing rather than silently giving up)

Under `LLM_PROVIDER=mock` both entry points return canned results without touching fastembed, the vector store, or Anthropic.

## How to run locally

Four commands from a fresh clone (`python -m agent` must run from inside the agent's own directory: `agent` is a submodule of the digit-prefixed `10_policy_qa` package):

```bash
git clone https://github.com/rajeshm71/real-world-agents.git
cd real-world-agents
cp .env.example .env    # set ANTHROPIC_API_KEY (or LLM_PROVIDER=mock)
cd agents/10_policy_qa
```

Mock demo (no API key, no fastembed, canned result):

```bash
LLM_PROVIDER=mock uv run python -m agent ingest examples/handbook
LLM_PROVIDER=mock uv run python -m agent ask "How much PTO do I get?"
```

Real ingest against the shipped example handbook (needs fastembed installed; first run downloads the ~30 MB ONNX model):

```bash
uv run python -m agent ingest examples/handbook
uv run python -m agent ask "How much PTO do I get?"
```

Rough cost at Anthropic Claude Sonnet-5 pricing: **~$0.01 per question** (5 chunks × ~500 tokens each + ~200 token answer). Ingestion is free (local embeddings). Switch the retrieval depth with `--top-k`:

```bash
uv run python -m agent ask "How does parental leave work?" --top-k 3
```

Point at your own directory of `.md` / `.txt` / `.pdf` files:

```bash
uv run python -m agent ingest /path/to/your/handbook --index my_handbook.db
uv run python -m agent ask "your question" --index my_handbook.db
```

Gradio UI (two tabs: Ingest + Ask):

```bash
uv run python -m agent --ui
```

Every real CLI ask writes the full `PolicyAnswer` JSON to `last_run.json` next to `agent.py` (gitignored via `agents/*/last_run.json`).

## Code walkthrough

Under 700 LOC across `agent.py` (~640) + `stores.py` (~210). Read in order:

1. **`schemas.py`**: `Chunk`, `Citation`, `IngestionStats`, `PolicyAnswer`. Four model validators on `PolicyAnswer` enforce citation honesty: (a) every citation's `chunk_id` must appear in `retrieved_chunks` (no phantom citations); (b) `quoted_text` must be a verbatim substring of the cited chunk after whitespace normalization (no paraphrases); (c) `confidence="low"` requires `unanswered_reason`; (d) `high`/`medium` confidence requires at least one citation. Reject at parse time so a hallucinated citation never reaches the caller.
2. **`prompts/system.txt`**: instructions for the model. Explicit rules: use ONLY retrieved passages, every claim needs a `[chunk_id]` inline marker + a `Citation` entry, `quoted_text` must be verbatim (caller cross-checks), set `confidence="low"` + `unanswered_reason` if the passages don't answer.
3. **`agent.py::_chunk_document()`**: fixed-char chunks with overlap. Prefers to break at paragraph boundaries within the last 20% of the window (so a chunk doesn't cut mid-sentence). Captures nearest preceding Markdown heading as `section` metadata for citations. `chunk_id` format is `<filename>#chunk-<N>`.
4. **`agent.py::_load_file_text()`**: `.md` / `.txt` read directly as UTF-8; `.pdf` goes through pypdf and gets `--- PAGE N ---` separators between pages. Raises `PolicyQAError` with a clear "install pypdf" pointer if the extension is missing.
5. **`stores.py::SqliteVecStore`**: preferred backend. Two tables in one `.db` file: `chunks` (metadata rows) and `vecs` (vec0 virtual table with `float[384]` embeddings). Search uses a subquery structure (`SELECT ... FROM (SELECT rowid FROM vecs WHERE embedding MATCH ? LIMIT N) v JOIN chunks c ON ...`) because sqlite-vec's KNN constraint check requires the LIMIT on the vec table directly, not on a joined result.
6. **`stores.py::FaissStore`**: fallback backend. FAISS `IndexFlatIP` for the vectors + a sibling sqlite metadata file. Uses cosine similarity via normalized inner product (fastembed's embeddings are already unit-normalized). Inverts the score to `1.0 - inner_product` so the return shape matches SqliteVecStore's lower-is-better distance.
7. **`stores.py::get_store()`**: tries `SqliteVecStore` first; falls back to `FaissStore` on `sqlite3.OperationalError` / `OSError` / `RuntimeError` / `ImportError`. Callers never care which one they got.
8. **`agent.py::ingest_corpus()`**: public API. Under `provider="mock"` short-circuits to `_mock_ingestion_stats` without importing fastembed. Real path: walk corpus → load each file → chunk → embed via `_load_embedder()` (fastembed's `TextEmbedding`) → insert into store. Returns `IngestionStats`.
9. **`agent.py::ask()`**: public API. Under `provider="mock"` short-circuits. Real path: R5 case 1 (index missing) → embed question → retrieve top-K → `_call_anthropic()` → `_parse_and_validate()`. Every failure surfaces as `PolicyQAError` with a `RetrievalAttempt` partial attached so a caller can inspect what the retriever returned before the LLM call failed.
10. **`agent.py::_call_anthropic()`**: raw Anthropic Messages API call, no Instructor. Same "no framework" shape as #02 -- lets a reader see what the raw request/response looks like. Handles the `content` list → text extraction; raises PolicyQAError if no text blocks come back.
11. **`agent.py::_parse_and_validate()`**: strips common `” ```json ... ``` "` code fences models add despite instructions, json.loads, injects `question` + `retrieved_chunks` (which the caller adds; the model isn't asked to echo them), then Pydantic-validates. Any citation-honesty violation raises `ValidationError`.
12. **`agent.py::_translate_api_error()`**: R5 case 3. Six-branch priority order (class-name → status → message → generic), same shape as agents #02-#09. Attaches the `RetrievalAttempt` partial so a caller inspecting a rate-limit error can still see what was retrieved.
13. **`ui.py::build_ui()`**: Gradio Blocks with two tabs -- Ingest (corpus picker → stats card) and Ask (question box + top-K slider → answer with citation cards). Each citation card shows quoted_text + source_path + section for eyeball-verification.
14. **`tests/test_smoke.py`**: 46 tests including a **real end-to-end ingest test** (behind `pytest.importorskip("fastembed")` + `pytest.importorskip("sqlite_vec")`) that proves the full pipeline works: 5 handbook files → real fastembed embeddings → real sqlite-vec index → real semantic retrieval correctly returns the leave_policy.md chunk when asked about PTO. No LLM cost.

## When to use / When NOT to use

**Use when:**
- You have a small-to-medium corpus (dozens to a few thousand pages) and need Q&A over it
- Verifiable citations matter more than answer generation speed (this pattern always shows the reader which passage backs each claim)
- Privacy of the corpus is important (handbook contents never leave the box at index time)
- You want to see the RAG pattern in ~700 LOC without a framework abstracting it away

**Do NOT use when:**
- Your corpus is huge (>100K chunks): switch to a purpose-built vector DB (Qdrant, Weaviate, pgvector at scale)
- You need answers that synthesize across many chunks (flat top-K retrieval misses cross-chunk relationships; consider a graph or hybrid retrieval approach)
- Your questions are conversational / follow-ups: each `ask()` is stateless; no chat memory in v1
- You need multilingual (bge-small-en-v1.5 is English-only; swap to a multilingual embedding model + re-index)

## Where this fails

Specific, honest failure modes:

- **The model paraphrases a citation and gets rejected**: this is the schema working correctly. `PolicyAnswer.model_validate()` raises `ValidationError`, `_parse_and_validate()` wraps it in `PolicyQAError` with `.partial.raw_output` set. Read the raw output to see what the model tried; usually a whitespace mismatch or a synonym substitution. Occasional; retry (or bump the retry limit if it becomes chronic on your corpus).
- **Retrieval misses the actual answer**: dense embeddings sometimes return semantically-similar-but-topically-wrong chunks. Symptom: `confidence="low"` + `unanswered_reason` says the passages don't cover the topic even when they do. Workaround: increase `--top-k`, or add a keyword-search fallback (BM25 hybrid retrieval) as a v1.1 improvement.
- **Answer is confidently wrong because the corpus itself is wrong**: RAG grounds answers in the retrieved chunks, not in ground truth. If your handbook says "employees get 30 days of PTO" when the actual policy is 20 days, the agent will confidently cite the wrong number. This is fine (the citation lets a reader spot-check) but users must not treat the answer as authoritative without checking the cited source.
- **Very long PDFs slow ingestion**: pypdf's per-page extraction is O(pages) and can take 5-10 minutes for a 500-page PDF. First-run embedding is O(chunks) at ~10 chunks/second on CPU. Consider ingesting once and reusing the index (`--index my_index.db`).
- **sqlite-vec extension load fails on some platforms**: pre-built wheels don't cover every OS/architecture. When it fails, `get_store()` catches the failure and falls back to FAISS transparently. `IngestionStats.vector_store_backend` reports which backend actually ended up in use; check it if performance is unexpectedly bad (FAISS is slower for small corpora due to per-batch numpy overhead).
- **Code-fenced JSON responses**: models sometimes wrap the JSON in ``` ```json ... ``` ``` despite the prompt saying not to. `_parse_and_validate()` strips the fences; if a model wraps it in something weirder (`<answer>...</answer>`, HTML, etc.) the parse fails and you see the raw output in `PolicyQAError.partial.raw_output`.
- **Rate limit / API failure**: no auto-retry with backoff (explicit design decision -- same stance as agents #02-#09). One failed API call gets translated to "temporarily rate-limited"; you re-run manually.
- **Real Anthropic answering not verified against a live API call yet**: the mock + parsing tests cover the code paths, but no end-to-end run with a real Anthropic key has been billed at ship time. Same open-item status as every prior agent's first ship. Set `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` and run `python -m agent ask "..."` to shake out any live-call quirks.

## Roadmap (post-v1 improvements documented)

- **Reranking**: add a `bge-reranker-base` cross-encoder pass on the top-K to reorder before sending to the LLM. ~2x latency at query time but improves precision on ambiguous queries.
- **Multi-provider**: OpenAI and Gemini via `common.llm.get_llm()` -- trivial add since `_call_anthropic` is the only Anthropic-specific site.
- **Streaming answers**: v1 returns the full `PolicyAnswer` at once; a streaming variant could return the answer + citations incrementally as the model generates.
- **Query expansion / HyDE**: transform the query into a hypothetical answer before embedding, retrieve against that -- helps when the user's question uses different vocabulary than the handbook.
- **Conversation memory**: each `ask()` is currently stateless; a follow-up "and what about parental leave?" wouldn't remember the previous PTO question. A v1.1 could add a `conversation_id` parameter and thread state.
