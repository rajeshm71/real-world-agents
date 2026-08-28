# Fact-checker (speech / video / text vs live web search)

Give the agent a piece of content -- raw text, a `.md` / `.txt`
transcript, a local audio file, or a YouTube URL -- and get back a
per-claim fact-check report. Uses a LOCAL Ollama LLM (default
`llama3.1:8b`) for claim extraction and verdict adjudication, and a
multi-provider search chain (Tavily -> Brave -> DuckDuckGo, with
auto-failover on rate-limit or missing key) for the "what does the
web say" step. First agent in the catalog that combines transcript
processing with live retrieval-plus-verification.

## Verification status

| Path | Status |
|---|---|
| Mock mode (`LLM_PROVIDER=mock`) | Fully covered by the shipped test suite |
| Loader (text / .md / .txt passthrough; audio + YouTube via a lazy-imported #14 transcriber stub) | Fully covered |
| Search chain (Tavily first, Brave fallback, DDG last-resort, rate-limit failover, exhausted-chain error) | Fully covered against stub providers |
| Extract stage (claim substring drop, claim_id assignment, JSON parse guard) | Fully covered against Ollama stub |
| Verify stage (verdict shape, non-verbatim evidence drop, no-search-hits path, exhausted-search error) | Fully covered against Ollama + search stubs |
| Schema validators (InputSource one-of + kind consistency, verdict/evidence/confidence rules, report parity by claim_id, claim substring guarantee) | Fully covered, both directions |
| Real Ollama + real Tavily/Brave/DDG (`LLM_PROVIDER=ollama` + `ollama pull llama3.1:8b`) | **Not yet verified against a live run.** Structural correctness proven via mock + stub tests. Same open-item status as every other agent's first ship. |

## Technique demonstrated

**Multi-stage local-LLM + live-web-search pipeline with cross-field-verified evidence.**

Four stages, each isolated behind a single injectable function:

1. `_load_source` -- text / `.md` / `.txt` / audio / YouTube. Audio and YouTube reuse agent #14's transcriber via `importlib.import_module`, so #14 is NOT a workspace dep and #16 stays installable in isolation.
2. `_extract_claims` -- local Ollama with a structured-output schema returns `list[FactualClaim]`. Every claim's text is cross-checked as a whitespace-normalized substring of the source; hallucinated claims are dropped before they can pollute the report.
3. `_verify_claim` -- for each claim, build a search query (entities + word-boundary-truncated claim text), query the multi-provider chain, feed top-K snippets back to Ollama for a verdict + confidence + verbatim-quoted evidence. Non-verbatim evidence is dropped in Python before `ClaimVerdict` is constructed; a `supported` verdict whose evidence was all dropped downgrades to `unclear` instead of raising.
4. `_assemble_report` -- `FactCheckReport` with per-claim verdicts, summary counts, and `run_meta` including which search providers were actually hit.

Distinct from every earlier agent:
- First to combine speech/audio ingestion with live web retrieval + verification.
- First to depend on a LOCAL LLM (Ollama) by default rather than a cloud API.
- First to run a rate-limit-aware provider fallback chain.

## Why this technique for this use case

An orator wants to spot-check factual claims before publishing. The natural shape is:

- Full-text-in, structured-report-out (batch, not streaming).
- Grounding requires TODAY'S information, not the LLM's training-time knowledge -- so live web search is load-bearing.
- Verdict text alone is not enough; each verdict needs verbatim quoted evidence with a URL so the human can double-check.

Where this pattern is NOT the right fit: real-time (sub-second) fact-check during live speech; adversarial contexts where search results are actively poisoned; short-lived viral claims where Google / Bing / Brave / DDG have not yet indexed authoritative pushback.

## What it does

**`fact_check(source, ...)`** returns a `FactCheckReport`:

- `source` -- `InputSource(kind, path/url/raw_text)`.
- `source_text` -- the full text the pipeline saw (raw text, transcript file contents, or Whisper output).
- `claims` -- extracted verifiable claims, each with `claim_id`, `text` (verbatim substring of source), `claim_type`, `entities`.
- `verdicts` -- parallel to claims by `claim_id`. Each carries `verdict` in `{supported, contradicted, unclear, unverifiable}`, `confidence` in `{high, medium, low}`, `explanation`, and `evidence` (verbatim quotes + URLs, cross-checked against the search snippets).
- `summary` -- verdict-class counts.
- `run_meta` -- Ollama model, search chain configuration, per-stage elapsed seconds, which providers were actually hit, and any evidence / claim drops recorded during the run.

Under `LLM_PROVIDER=mock`, the agent returns a scripted 3-claim report (1 supported, 1 contradicted, 1 unverifiable) without touching Ollama, the search providers, or #14. Source-size is echoed into `run_meta['mock_source_size']` as the anti-refactor guard.

## How to run locally

Four commands from a fresh clone, plus one-time Ollama setup:

```bash
git clone https://github.com/rajeshm71/real-world-agents.git
cd real-world-agents
cp .env.example .env    # optional: set TAVILY_API_KEY / BRAVE_API_KEY
cd agents/16_fact_checker
```

Install Ollama and pull the default model (one-time):

```bash
# from https://ollama.com/download
ollama serve &            # keep this running
ollama pull llama3.1:8b
```

Mock demo (no Ollama needed, no network, canned 3-claim result):

```bash
LLM_PROVIDER=mock uv run python -m agent examples/sample_transcript.md
```

Real fact-check against the shipped example transcript:

```bash
uv run python -m agent examples/sample_transcript.md
```

Force a specific search provider:

```bash
uv run python -m agent examples/sample_transcript.md --search ddg
```

Cap claims (protects against runaway search-provider spend on a long transcript):

```bash
uv run python -m agent examples/sample_transcript.md --top-k 5
```

Real fact-check against a local audio file or a YouTube URL (requires agent #14's transcriber to be installed via `uv sync --all-packages`):

```bash
uv run python -m agent ~/speeches/talk.mp3
uv run python -m agent https://youtube.com/watch?v=...
```

Estimated cost: **$0 per run** in the default setup. Ollama is local; DuckDuckGo requires no key; Tavily's free tier is 1000 queries/month and Brave's is 2000/month. The 5 claims * 5 search results/claim default fits comfortably in either free tier for hundreds of transcripts per month.

## Code walkthrough

- `schemas.py`: five models. `InputSource` (kind + exactly-one-payload invariant), `FactualClaim`, `EvidenceSnippet`, `ClaimVerdict` (strong-verdict-needs-evidence, unverifiable-has-no-evidence, low-confidence-caps-verdict rules), `FactCheckReport` (verdicts parallel to claims by claim_id, every claim text is a substring of source_text).
- `search.py`: `SearchClient` Protocol; `TavilyProvider` / `BraveProvider` / `DDGProvider` each translate their native rate-limit error to a shared `SearchRateLimit`. `ChainProvider` orchestrates ordered failover; `build_search_client("auto")` composes the chain from whichever env vars are set.
- `agent.py` (`_load_source`, `_transcribe_via_podcast_processor`): stage A. URL detection routes to the #14 lazy-import path; unsupported file suffix raises.
- `agent.py` (`_extract_claims`): stage B. Ollama structured-output call with `_CLAIMS_SCHEMA`; substring cross-check drops invented claims and records the drops in `run_meta`.
- `agent.py` (`_verify_claim`, `_build_search_query`, `_format_verify_prompt`): stage C. Per-claim search + LLM adjudication. Non-verbatim evidence is dropped in Python before `ClaimVerdict` construction; a stripped-empty strong verdict downgrades to `unclear` to satisfy the schema instead of raising.
- `agent.py` (`_translate_llm_error`): R5 -- dedicated branches for connection-refused (`ollama serve` hint), model-not-pulled (`ollama pull` hint), 429, 401, generic.
- `prompts/extract.txt` + `prompts/verify.txt`: the two rubrics driving the Ollama calls. Verify rubric includes the "quoted_text MUST be verbatim from the provided snippets" contract the caller then enforces.

## When to use / When NOT to use

**Use this pattern when:**
- You need to sanity-check speech / video / text for factual accuracy before publishing.
- You want the LLM step to stay local (no per-claim cloud cost, no transcript leaving the box).
- You are OK with batch-mode (full transcript in, report out) latency of seconds-to-minutes.

**Do NOT use this pattern when:**
- You need sub-second per-claim latency (real-time fact-check overlay during a live stream).
- You need attribution across multiple speakers in a debate (requires diarization; not in v1).
- Your claims are so niche that top-K search results reliably miss the authoritative source. Consider a corpus-specific RAG pipeline (see #10) instead.

## Where this fails

- **Model spirals on ambiguous claims.** `llama3.1:8b` is a small model; it will sometimes rate a plainly-true claim as `unclear` because the snippets don't quote it verbatim. Larger local models (`llama3.1:70b` if you have the GPU) or a cloud fallback would help. v1 sticks with the small model for accessibility.
- **Snippet-only view of a source.** The Ollama adjudicator sees only the search-result snippets, not the full pages. A snippet can be misleading even when the full page confirms the opposite. v1 does not follow links; a follow-up could fetch top-K pages.
- **English-centric.** Both the prompts and `llama3.1:8b`'s bias are English. Cross-language claims produce weaker verdicts.
- **Search-result freshness.** Providers index at different cadences; a claim contradicted in a news article from six hours ago may still be `supported` by an older top-K result.
- **Adversarial-search fragility.** Nothing here defends against poisoned search results engineered to confirm a false claim. Treat verdicts as first-pass, not authoritative.
- **Ollama server dependency.** No cloud-LLM fallback in v1 (deliberate: the user asked for a local default). If `ollama serve` is not running, the CLI raises with a `ollama pull llama3.1:8b` hint but does not fall back to OpenAI/Anthropic.

## Roadmap (post-v1 improvements)

- Fetch top-K pages in full (not just snippets) before adjudication.
- Cloud-LLM fallback flag (`--llm openai`) for users who want higher-quality verdicts without a GPU.
- Speaker diarization (route each claim to the panelist who said it).
- Claim clustering (near-duplicate claims across paragraphs collapsed into one verdict).
- Automatic correction generation (verdict + "here is what the current truth is").
- Streaming-mode: incremental fact-check as a live transcript grows.
- Longitudinal tracking: "true when said" vs "true today" distinction.
