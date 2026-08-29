"""Smoke tests for the fact-checker.

Zero real Ollama calls, zero real network. Every non-mock path uses
either a monkey-patched provider stub or an injected `_ollama_client`
/ `_search` / `_whisper_model`. Only local file I/O touches disk.

Sections:
1. Mock path (6).
2. Loader (6).
3. Search chain (7).
4. Extract stage (5).
5. Verify stage (5).
6. Schema validators (8).
7. R5 branches (5).
8. Constants + example sanity (3).
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(_AGENT_DIR.parent))

_agent = importlib.import_module("16_fact_checker.agent")
_schemas = importlib.import_module("16_fact_checker.schemas")
_search = importlib.import_module("16_fact_checker.search")

fact_check = _agent.fact_check
FactCheckError = _agent.FactCheckError
FactCheckAttempt = _agent.FactCheckAttempt
_load_source = _agent._load_source
_extract_claims = _agent._extract_claims
_verify_claim = _agent._verify_claim
_translate_llm_error = _agent._translate_llm_error
_build_search_query = _agent._build_search_query
SUPPORTED_PROVIDERS = _agent.SUPPORTED_PROVIDERS

InputSource = _schemas.InputSource
FactualClaim = _schemas.FactualClaim
EvidenceSnippet = _schemas.EvidenceSnippet
ClaimVerdict = _schemas.ClaimVerdict
FactCheckReport = _schemas.FactCheckReport

SearchHit = _search.SearchHit
SearchRateLimit = _search.SearchRateLimit
SearchUnavailable = _search.SearchUnavailable
SearchAllUnavailable = _search.SearchAllUnavailable
ChainProvider = _search.ChainProvider
build_search_client = _search.build_search_client

_TRANSCRIPT = _AGENT_DIR / "examples" / "sample_transcript.md"


# --- Helpers ---------------------------------------------------------------


_TRIAGE_DEFER_TO_SEARCH = json.dumps({
    "needs_search": True, "verdict": None, "confidence": None,
    "explanation": "Deferring to web search for stub-based test.",
})
_TRIAGE_FAST_PATH_SUPPORTED = json.dumps({
    "needs_search": False, "verdict": "supported", "confidence": "high",
    "explanation": "Well-established fact.",
})


class _StubOllama:
    """Duck-typed Ollama client. `responses` is a list of JSON strings
    the stub returns in order. Each call consumes one.

    Extract calls consume 1 response. Verify calls consume 2 (triage
    + adjudicate) OR 1 (triage says needs_search=false, fast path).
    The `auto_triage` flag (default True) prepends a
    'defer to search' triage response before every verify-shaped
    response so existing tests keep working without churn."""

    def __init__(self, responses, auto_triage: bool = True):
        self._responses = list(responses)
        self._auto_triage = auto_triage
        self.call_count = 0

    def chat(self, **kwargs):
        self.call_count += 1
        text = self._responses.pop(0)
        return {"message": {"content": text}}


class _StubSearch:
    """Duck-typed search client. `hits_by_query` maps query prefixes
    to hit lists (matched by `startswith`); a missing query returns
    an empty list."""

    provider_name = "ddg"
    last_used_provider = "ddg"

    def __init__(self, hits_by_query=None, default_hits=None):
        self._map = hits_by_query or {}
        self._default = default_hits or []
        self.calls: list[str] = []

    def search(self, query, *, max_results):
        self.calls.append(query)
        for prefix, hits in self._map.items():
            if query.startswith(prefix):
                return hits[:max_results]
        return self._default[:max_results]


class _RateLimitedProvider:
    provider_name = "tavily"

    def search(self, query, *, max_results):
        raise SearchRateLimit("simulated rate limit")


class _AlwaysWorkingProvider:
    def __init__(self, name):
        self.provider_name = name

    def search(self, query, *, max_results):
        return [SearchHit(title="ok", url=f"https://{self.provider_name}.example",
                          snippet="stubbed", provider=self.provider_name)]


# --- 1. Mock path ----------------------------------------------------------


def test_mock_returns_valid_report(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    r = fact_check(str(_TRANSCRIPT))
    assert isinstance(r, FactCheckReport)


def test_mock_source_size_echoed(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    r = fact_check(str(_TRANSCRIPT))
    assert r.run_meta["mock_source_size"] == _TRANSCRIPT.stat().st_size


def test_mock_three_scripted_claims(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    r = fact_check(str(_TRANSCRIPT))
    assert len(r.claims) == 3
    assert len(r.verdicts) == 3


def test_mock_all_three_verdict_classes(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    r = fact_check(str(_TRANSCRIPT))
    verdicts = {v.verdict for v in r.verdicts}
    assert "supported" in verdicts
    assert "contradicted" in verdicts
    assert "unverifiable" in verdicts


def test_mock_claim_texts_are_substrings_of_source(monkeypatch):
    """The mock's scripted claims must respect the substring guarantee."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    r = fact_check(str(_TRANSCRIPT))
    # Just constructing the report already validates this; if we got
    # here, the check passed.
    assert r.source_text


def test_mock_skips_ollama_and_search(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setitem(sys.modules, "ollama", None)
    monkeypatch.setitem(sys.modules, "tavily", None)
    monkeypatch.setitem(sys.modules, "ddgs", None)
    r = fact_check(str(_TRANSCRIPT))
    assert r.run_meta["provider"] == "mock"


# --- 2. Loader -------------------------------------------------------------


def test_load_text_passthrough():
    src, text = _load_source("The Eiffel Tower is in Paris.")
    assert src.kind == "text"
    assert text == "The Eiffel Tower is in Paris."


def test_load_md_file(tmp_path):
    p = tmp_path / "transcript.md"
    p.write_text("# Speech\n\nHello world.", encoding="utf-8")
    src, text = _load_source(p)
    assert src.kind == "transcript_file"
    assert "Hello world" in text


def test_load_txt_file(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("plain notes", encoding="utf-8")
    src, text = _load_source(p)
    assert src.kind == "transcript_file"
    assert text == "plain notes"


def test_load_youtube_url_lazy_imports_pp(monkeypatch, tmp_path):
    """URLs must route through the #14 transcriber lazy-import."""
    # Fake the podcast_processor module for the lazy import.
    fake_pp = types.ModuleType("14_podcast_processor.agent")

    def _fake_fetch(source, *, tmpdir):
        return types.SimpleNamespace(path=tmp_path / "fake.wav",
                                     origin="youtube", url=source)

    def _fake_transcribe(audio_src, **kwargs):
        return types.SimpleNamespace(full_text="youtube transcript body")

    fake_pp._fetch_audio = _fake_fetch
    fake_pp._transcribe = _fake_transcribe
    monkeypatch.setitem(sys.modules, "14_podcast_processor.agent", fake_pp)
    src, text = _load_source("https://youtube.com/watch?v=abc")
    assert src.kind == "youtube"
    assert text == "youtube transcript body"


def test_load_audio_file_lazy_imports_pp(monkeypatch, tmp_path):
    fake_pp = types.ModuleType("14_podcast_processor.agent")
    fake_pp._fetch_audio = lambda source, *, tmpdir: types.SimpleNamespace(
        path=Path(source), origin="local", url=None
    )
    fake_pp._transcribe = lambda audio_src, **kwargs: types.SimpleNamespace(
        full_text="audio transcript body"
    )
    monkeypatch.setitem(sys.modules, "14_podcast_processor.agent", fake_pp)
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"ID3")
    src, text = _load_source(audio)
    assert src.kind == "audio"
    assert text == "audio transcript body"


def test_load_unsupported_file_suffix_rejected(tmp_path):
    p = tmp_path / "notes.pdf"
    p.write_bytes(b"%PDF")
    with pytest.raises(FactCheckError, match="unsupported file suffix"):
        _load_source(p)


# --- 3. Search chain ------------------------------------------------------


def test_chain_tavily_first_when_key_set(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake")
    tavily = _AlwaysWorkingProvider("tavily")
    chain = ChainProvider([tavily])
    hits = chain.search("q", max_results=3)
    assert hits[0].provider == "tavily"


def test_chain_falls_back_on_rate_limit():
    tavily = _RateLimitedProvider()
    ddg = _AlwaysWorkingProvider("ddg")
    chain = ChainProvider([tavily, ddg])
    hits = chain.search("q", max_results=3)
    assert chain.last_used_provider == "ddg"
    assert hits[0].provider == "ddg"


def test_chain_three_way_fallback():
    a = _RateLimitedProvider()
    b = _RateLimitedProvider()
    c = _AlwaysWorkingProvider("ddg")
    chain = ChainProvider([a, b, c])
    chain.search("q", max_results=3)
    assert chain.last_used_provider == "ddg"


def test_chain_exhausted_raises():
    chain = ChainProvider([_RateLimitedProvider(), _RateLimitedProvider()])
    with pytest.raises(SearchAllUnavailable):
        chain.search("q", max_results=3)


def test_chain_empty_construction_raises():
    with pytest.raises(SearchAllUnavailable):
        ChainProvider([])


def test_build_client_auto_with_no_keys_includes_ddg_only(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    client = build_search_client("auto")
    # Auto returns a ChainProvider even for a single available provider.
    assert isinstance(client, ChainProvider)


def test_build_client_forced_kind_returns_single_provider():
    from importlib import import_module
    s = import_module("16_fact_checker.search")
    # DDG is always constructible (no key required).
    client = s.build_search_client("ddg")
    assert client.provider_name == "ddg"


# --- 4. Extract stage -----------------------------------------------------


def test_extract_returns_valid_claims():
    source = "The Eiffel Tower is located in Paris and stands 330 meters tall."
    resp = json.dumps({
        "claims": [
            {"text": "The Eiffel Tower is located in Paris",
             "claim_type": "event", "entities": ["Eiffel Tower", "Paris"]},
        ]
    })
    stub = _StubOllama([resp])
    run_meta: dict = {}
    claims = _extract_claims(source, ollama_client=stub,
                             ollama_model="test", run_meta=run_meta)
    assert len(claims) == 1
    assert claims[0].claim_id == "c000"
    assert claims[0].claim_type == "event"


def test_extract_drops_invented_claims():
    """Claims whose text isn't in the source get dropped."""
    source = "The Eiffel Tower is located in Paris."
    resp = json.dumps({
        "claims": [
            {"text": "The Eiffel Tower is located in Paris",
             "claim_type": "event"},
            # Invented claim - not in source.
            {"text": "The moon is made of cheese, obviously",
             "claim_type": "other"},
        ]
    })
    run_meta: dict = {}
    claims = _extract_claims(source, ollama_client=_StubOllama([resp]),
                             ollama_model="t", run_meta=run_meta)
    assert len(claims) == 1
    assert len(run_meta["claim_extraction_drops"]) == 1


def test_extract_assigns_sequential_claim_ids():
    source = "Alpha claim here today. Beta claim also here today."
    resp = json.dumps({
        "claims": [
            {"text": "Alpha claim here today", "claim_type": "other"},
            {"text": "Beta claim also here today", "claim_type": "other"},
        ]
    })
    run_meta: dict = {}
    claims = _extract_claims(source, ollama_client=_StubOllama([resp]),
                             ollama_model="t", run_meta=run_meta)
    assert [c.claim_id for c in claims] == ["c000", "c001"]


def test_extract_empty_claims_ok():
    source = "This paragraph has no verifiable facts, only opinions."
    resp = json.dumps({"claims": []})
    run_meta: dict = {}
    claims = _extract_claims(source, ollama_client=_StubOllama([resp]),
                             ollama_model="t", run_meta=run_meta)
    assert claims == []


def test_extract_bad_json_raises():
    """Two consecutive bad-JSON responses (retry-once exhausted) raise."""
    run_meta: dict = {}
    with pytest.raises(FactCheckError, match="non-JSON output"):
        _extract_claims("source", ollama_client=_StubOllama(["nope", "nope"]),
                        ollama_model="t", run_meta=run_meta)


# --- 5. Verify stage ------------------------------------------------------


def _make_claim(text="The Eiffel Tower is in Paris always."):
    return FactualClaim(
        claim_id="c000", text=text,
        claim_type="event", entities=["Eiffel Tower"],
    )


def test_verify_returns_supported_verdict():
    claim = _make_claim()
    hits = [SearchHit(title="Eiffel", url="https://ex.com/e",
                     snippet="The Eiffel Tower is in Paris.", provider="ddg")]
    search = _StubSearch(default_hits=hits)
    resp = json.dumps({
        "verdict": "supported", "confidence": "high",
        "explanation": "Multiple sources agree.",
        "evidence": [{"quoted_text": "The Eiffel Tower is in Paris.",
                      "source_url": "https://ex.com/e"}],
    })
    run_meta: dict = {}
    # verify now calls triage FIRST; feed a defer-to-search response
    # before the actual verify response.
    v = _verify_claim(claim,
                     ollama_client=_StubOllama([_TRIAGE_DEFER_TO_SEARCH, resp]),
                     ollama_model="t", search_client=search,
                     max_search_results=5, run_meta=run_meta)
    assert v.verdict == "supported"
    assert len(v.evidence) == 1
    assert v.verification_method == "web_search"


def test_verify_drops_non_verbatim_evidence():
    claim = _make_claim()
    hits = [SearchHit(title="Eiffel", url="https://ex.com/e",
                     snippet="The Eiffel Tower is in Paris.", provider="ddg")]
    search = _StubSearch(default_hits=hits)
    resp = json.dumps({
        "verdict": "supported", "confidence": "medium",
        "explanation": "One source cited.",
        "evidence": [
            {"quoted_text": "Paraphrase not in snippet",
             "source_url": "https://ex.com/e"},
        ],
    })
    run_meta: dict = {}
    v = _verify_claim(claim,
                     ollama_client=_StubOllama([_TRIAGE_DEFER_TO_SEARCH, resp]),
                     ollama_model="t", search_client=search,
                     max_search_results=5, run_meta=run_meta)
    # Non-verbatim evidence dropped -> supported downgrades to unclear.
    assert v.verdict == "unclear"
    assert v.evidence == []


def test_verify_no_search_hits_produces_unverifiable():
    claim = _make_claim()
    search = _StubSearch(default_hits=[])
    stub = _StubOllama([_TRIAGE_DEFER_TO_SEARCH])
    run_meta: dict = {}
    v = _verify_claim(claim, ollama_client=stub, ollama_model="t",
                     search_client=search, max_search_results=5,
                     run_meta=run_meta)
    assert v.verdict == "unverifiable"
    assert v.evidence == []
    # 1 call for triage; verify-adjudicate is skipped because search
    # returned nothing.
    assert stub.call_count == 1


def test_verify_search_exhausted_raises():
    claim = _make_claim()
    chain = ChainProvider([_RateLimitedProvider(), _RateLimitedProvider()])
    stub = _StubOllama([_TRIAGE_DEFER_TO_SEARCH])
    run_meta: dict = {}
    with pytest.raises(FactCheckError, match="search chain exhausted"):
        _verify_claim(claim, ollama_client=stub, ollama_model="t",
                     search_client=chain, max_search_results=5,
                     run_meta=run_meta)


def test_verify_fast_path_skips_search_on_well_known_fact():
    """Agentic fast path: triage returns high-confidence verdict with
    needs_search=false, so no web search or adjudicate call fires."""
    claim = _make_claim(text="Two plus two equals four in basic arithmetic.")
    search = _StubSearch(default_hits=[])
    stub = _StubOllama([_TRIAGE_FAST_PATH_SUPPORTED])
    run_meta: dict = {}
    v = _verify_claim(claim, ollama_client=stub, ollama_model="t",
                     search_client=search, max_search_results=5,
                     run_meta=run_meta)
    assert v.verdict == "supported"
    assert v.verification_method == "model_knowledge"
    assert v.evidence == []
    # Only triage consumed one call; no adjudicate.
    assert stub.call_count == 1
    # Search client was never invoked.
    assert search.calls == []
    # Fast-path count recorded.
    assert run_meta["triage_decisions"]["no_search"] == 1


def test_verify_low_confidence_triage_falls_through_to_search():
    """Fast path requires confidence=high; medium/low must trigger search."""
    claim = _make_claim()
    hits = [SearchHit(title="x", url="https://ex.com/e",
                     snippet="The Eiffel Tower is in Paris.", provider="ddg")]
    search = _StubSearch(default_hits=hits)
    triage_medium = json.dumps({
        "needs_search": False, "verdict": "supported", "confidence": "medium",
        "explanation": "I think so but I'd like to verify.",
    })
    adjudicate = json.dumps({
        "verdict": "supported", "confidence": "high", "explanation": "yes",
        "evidence": [{"quoted_text": "The Eiffel Tower is in Paris.",
                      "source_url": "https://ex.com/e"}],
    })
    stub = _StubOllama([triage_medium, adjudicate])
    run_meta: dict = {}
    v = _verify_claim(claim, ollama_client=stub, ollama_model="t",
                     search_client=search, max_search_results=5,
                     run_meta=run_meta)
    # Search WAS invoked (fast path rejected because confidence!=high).
    assert search.calls != []
    assert v.verification_method == "web_search"
    assert stub.call_count == 2


def test_verify_bad_json_raises():
    """Two consecutive bad-JSON responses (retry-once exhausted) raise."""
    claim = _make_claim()
    hits = [SearchHit(title="x", url="https://ex.com/e", snippet="text",
                     provider="ddg")]
    search = _StubSearch(default_hits=hits)
    # Triage OK, then two consecutive bad JSON on adjudicate.
    stub = _StubOllama([_TRIAGE_DEFER_TO_SEARCH, "not json 1", "not json 2"])
    run_meta: dict = {}
    with pytest.raises(FactCheckError, match="non-JSON output"):
        _verify_claim(claim, ollama_client=stub, ollama_model="t",
                     search_client=search, max_search_results=5,
                     run_meta=run_meta)


# --- 6. Schema validators -------------------------------------------------


def test_input_source_exactly_one_payload():
    with pytest.raises(ValidationError, match="exactly one of"):
        InputSource(kind="text", path=Path("/tmp/x"), raw_text="hello")


def test_input_source_kind_matches_payload():
    with pytest.raises(ValidationError, match="expects .* to be set"):
        InputSource(kind="youtube", raw_text="not a url")


def test_factual_claim_text_min_length():
    with pytest.raises(ValidationError):
        FactualClaim(claim_id="c1", text="short", claim_type="other")


def test_verdict_supported_requires_evidence():
    with pytest.raises(ValidationError, match="strong verdict without evidence"):
        ClaimVerdict(
            claim=_make_claim(), verdict="supported", confidence="high",
            explanation="ok", evidence=[],
        )


def test_verdict_supported_plus_low_confidence_rejected():
    ev = EvidenceSnippet(source_url="https://x", source_title="t",
                         quoted_text="q", search_provider="ddg")
    with pytest.raises(ValidationError, match="contradictory"):
        ClaimVerdict(
            claim=_make_claim(), verdict="supported", confidence="low",
            explanation="ok", evidence=[ev],
        )


def test_verdict_unverifiable_with_evidence_rejected():
    ev = EvidenceSnippet(source_url="https://x", source_title="t",
                         quoted_text="q", search_provider="ddg")
    with pytest.raises(ValidationError, match="nothing was found"):
        ClaimVerdict(
            claim=_make_claim(), verdict="unverifiable", confidence="low",
            explanation="ok", evidence=[ev],
        )


def test_report_verdicts_parallel_to_claims_by_id():
    source_text = "The Eiffel Tower is located in Paris always today."
    c0 = FactualClaim(claim_id="c000", text="The Eiffel Tower is located in Paris",
                    claim_type="event")
    c1 = FactualClaim(claim_id="c001", text="Paris always today",
                    claim_type="other")
    ev = EvidenceSnippet(source_url="https://x", source_title="t",
                         quoted_text="q", search_provider="ddg")
    v0 = ClaimVerdict(claim=c0, verdict="supported", confidence="high",
                     explanation="ok", evidence=[ev])
    # Verdicts in wrong order.
    v1 = ClaimVerdict(claim=c1, verdict="unclear", confidence="medium",
                     explanation="ok")
    with pytest.raises(ValidationError, match="parallel to claims by claim_id"):
        FactCheckReport(
            source=InputSource(kind="text", raw_text=source_text),
            source_text=source_text, claims=[c0, c1], verdicts=[v1, v0],
        )


def test_report_claims_must_be_substrings_of_source():
    source_text = "Only these exact words are here today."
    bogus = FactualClaim(claim_id="c000", text="Words not in source at all",
                        claim_type="other")
    v = ClaimVerdict(claim=bogus, verdict="unclear", confidence="low",
                    explanation="ok")
    with pytest.raises(ValidationError, match="not a verbatim substring"):
        FactCheckReport(
            source=InputSource(kind="text", raw_text=source_text),
            source_text=source_text, claims=[bogus], verdicts=[v],
        )


# --- 7. R5 branches -------------------------------------------------------


def test_translate_ollama_connection_refused():
    err = _translate_llm_error(
        RuntimeError("Connection refused to localhost:11434"), stage="extract"
    )
    assert "ollama serve" in err.message


def test_translate_ollama_model_not_found():
    err = _translate_llm_error(
        RuntimeError("model 'llama3.1:8b' not found"), stage="verify"
    )
    assert "ollama pull" in err.message


def test_translate_ollama_rate_limit():
    err = _translate_llm_error(RuntimeError("Rate limit hit"), stage="extract")
    assert "rate-limited" in err.message


def test_translate_ollama_401():
    exc = RuntimeError("Unauthorized")
    exc.status_code = 401
    err = _translate_llm_error(exc, stage="verify")
    assert "authentication failed" in err.message.lower()


def test_translate_generic_fallthrough():
    err = _translate_llm_error(RuntimeError("some weird thing"), stage="extract")
    assert "Ollama call failed" in err.message


# --- 8. Constants + example sanity ----------------------------------------


def test_supported_providers_ollama_only():
    assert SUPPORTED_PROVIDERS == ("ollama",)


def test_sample_transcript_exists_with_expected_shape():
    text = _TRANSCRIPT.read_text(encoding="utf-8")
    assert "Eiffel Tower" in text
    assert "Great Wall" in text


def test_build_search_query_uses_entities_and_truncates():
    claim = FactualClaim(
        claim_id="c1",
        text="A very long claim " * 50,  # >150 chars
        claim_type="other",
        entities=["Alice", "Bob"],
    )
    q = _build_search_query(claim)
    assert "Alice" in q
    assert "Bob" in q
    # No mid-word truncation (word-boundary respected).
    assert not q.rstrip().endswith("A very long claim A very long clai")


# --- 9. Post-review hardening (H1, M1, L1, L5) ----------------------------


def test_extract_retries_once_on_bad_json_then_succeeds():
    """H1: bad JSON on first attempt, valid JSON on second attempt."""
    source = "The Eiffel Tower is located in Paris today always."
    good = json.dumps({
        "claims": [
            {"text": "The Eiffel Tower is located in Paris",
             "claim_type": "event", "entities": []}
        ]
    })
    stub = _StubOllama(["not json at all", good])
    run_meta: dict = {}
    claims = _extract_claims(source, ollama_client=stub, ollama_model="t",
                             run_meta=run_meta)
    assert len(claims) == 1
    assert stub.call_count == 2


def test_extract_gives_up_after_retry():
    """H1: two consecutive bad-JSON responses still raise."""
    stub = _StubOllama(["nope 1", "nope 2"])
    run_meta: dict = {}
    with pytest.raises(FactCheckError, match="after 2 attempt"):
        _extract_claims("source", ollama_client=stub, ollama_model="t",
                        run_meta=run_meta)
    assert stub.call_count == 2


def test_verify_retries_once_on_bad_json_then_succeeds():
    """H1: verify stage also retries once."""
    claim = _make_claim()
    hits = [SearchHit(title="Eiffel", url="https://ex.com/e",
                     snippet="The Eiffel Tower is in Paris.", provider="ddg")]
    search = _StubSearch(default_hits=hits)
    good = json.dumps({
        "verdict": "supported", "confidence": "high",
        "explanation": "ok",
        "evidence": [{"quoted_text": "The Eiffel Tower is in Paris.",
                      "source_url": "https://ex.com/e"}],
    })
    # Triage OK, then bad-json + good on adjudicate retry.
    stub = _StubOllama([_TRIAGE_DEFER_TO_SEARCH, "not json", good])
    run_meta: dict = {}
    v = _verify_claim(claim, ollama_client=stub, ollama_model="t",
                     search_client=search, max_search_results=5,
                     run_meta=run_meta)
    assert v.verdict == "supported"
    # 3 calls: 1 triage + 2 adjudicate (bad-json then retry).
    assert stub.call_count == 3


def test_load_nonexistent_path_instance_raises(tmp_path):
    """M1: a Path instance pointing at a nonexistent file must raise,
    not silently downgrade to raw text."""
    ghost = tmp_path / "does_not_exist.mp3"
    with pytest.raises(FactCheckError, match="does not exist"):
        _load_source(ghost)


def test_load_str_that_is_not_a_path_still_becomes_text():
    """M1 regression guard: a plain string that happens not to name a
    file should still become text (not raise)."""
    src, text = _load_source("The Eiffel Tower is in Paris.")
    assert src.kind == "text"
    assert text == "The Eiffel Tower is in Paris."


def test_ollama_response_text_none_content_raises():
    """L1: an object-shaped response with .message.content == None
    must raise FactCheckError, not return None."""
    resp = types.SimpleNamespace(
        message=types.SimpleNamespace(content=None)
    )
    with pytest.raises(FactCheckError, match="missing message.content"):
        _agent._ollama_response_text(resp)


def test_ollama_response_text_empty_content_raises():
    """L1 corollary: empty string is treated the same as missing."""
    resp = {"message": {"content": ""}}
    with pytest.raises(FactCheckError, match="missing message.content"):
        _agent._ollama_response_text(resp)


def test_search_query_capped_at_400_chars():
    """L5: even a claim with many long entities must produce a query
    under the shared _MAX_SEARCH_QUERY_LEN cap."""
    claim = FactualClaim(
        claim_id="c1",
        text="A short claim about a specific topic and more content here.",
        claim_type="other",
        entities=["Entity Name " * 50] * 3,  # each entity ~600 chars
    )
    q = _build_search_query(claim)
    assert len(q) <= 400
    # Word-boundary truncation.
    assert not q.endswith("Entity Nam")
