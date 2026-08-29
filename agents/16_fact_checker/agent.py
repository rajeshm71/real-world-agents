"""Fact-checker: agent #16 of real-world-agents.

Technique demonstrated: **speech / video / text fact-checked against
live web search with local-LLM adjudication.** Real deployable use
case: an orator wants to catch outdated numbers, misattributed
quotes, or untruths before publishing.

Four-stage pipeline:
1. _load_source  -- text / .md / .txt / audio / YouTube (audio and
                    YouTube paths lazy-import #14's transcriber).
2. _extract_claims -- local Ollama returns list[FactualClaim];
                      claims whose text is not a verbatim substring
                      of the source are dropped (no hallucinated
                      claims allowed into the report).
3. _verify_claim  -- web search (Tavily -> Brave -> DDG chain with
                     auto-failover) + local Ollama adjudicator.
                     Evidence quoted_text is substring-checked
                     against the search snippet it came from before
                     the ClaimVerdict is constructed; a supported
                     verdict whose evidence was all dropped raises.
4. _assemble_report -- FactCheckReport with per-claim verdicts + a
                       summary count per verdict class.

Provider stance: LOCAL Ollama by default (matches the user's
"local models as default" preference). No cloud-LLM fallback in v1;
that would violate the local-default story. Multi-provider search
chain via search.py.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

try:
    from .schemas import (
        ClaimVerdict,
        EvidenceSnippet,
        FactCheckReport,
        FactualClaim,
        InputSource,
        _substring_of_normalized,
    )
    from .search import (
        SearchAllUnavailable,
        SearchClient,
        SearchHit,
        build_search_client,
    )
except ImportError:
    from schemas import (  # type: ignore[no-redef]
        ClaimVerdict,
        EvidenceSnippet,
        FactCheckReport,
        FactualClaim,
        InputSource,
        _substring_of_normalized,
    )
    from search import (  # type: ignore[no-redef]
        SearchAllUnavailable,
        SearchClient,
        SearchHit,
        build_search_client,
    )

# --- Constants -------------------------------------------------------------

SUPPORTED_PROVIDERS = ("ollama",)
_DEFAULT_PROVIDER = "ollama"
_DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
_DEFAULT_OLLAMA_HOST = "http://localhost:11434"
_DEFAULT_MAX_SEARCH_RESULTS = 5
_MAX_ENTITIES = 6
_AUDIO_SUFFIXES = (".mp3", ".m4a", ".wav", ".flac", ".ogg")
_TEXT_SUFFIXES = (".md", ".txt")

_EXTRACT_PROMPT_PATH = Path(__file__).parent / "prompts" / "extract.txt"
_VERIFY_PROMPT_PATH = Path(__file__).parent / "prompts" / "verify.txt"
_TRIAGE_PROMPT_PATH = Path(__file__).parent / "prompts" / "triage.txt"

_MAX_SEARCH_QUERY_LEN = 400  # Tavily / Brave both cap around this.
_JSON_PARSE_RETRIES = 1
# Ollama's default num_ctx is 2048 tokens -- silently TRUNCATES any
# input longer than that. A 15-minute transcript is ~3-5k tokens; the
# extract-stage prompt plus transcript can easily exceed the default
# and the model sees only the head. 8192 fits a ~30-minute transcript
# comfortably; users with longer content can pass their own via the
# _ollama_client hook.
_OLLAMA_NUM_CTX = 8192


# --- Error type ------------------------------------------------------------


@dataclass
class FactCheckAttempt:
    stage: Literal["load", "extract", "verify", "assemble"]
    claim_id: str | None = None
    raw_output: str = ""


class FactCheckError(Exception):
    def __init__(self, message: str, partial: FactCheckAttempt | None = None):
        super().__init__(message)
        self.message = message
        self.partial = partial


# --- Provider resolution ---------------------------------------------------


def resolve_provider() -> str:
    provider = os.environ.get("LLM_PROVIDER", _DEFAULT_PROVIDER).lower()
    if provider != "mock" and provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            f"Expected 'mock' or one of {SUPPORTED_PROVIDERS}."
        )
    return provider


@functools.lru_cache(maxsize=1)
def _load_extract_prompt() -> str:
    return _EXTRACT_PROMPT_PATH.read_text(encoding="utf-8")


@functools.lru_cache(maxsize=1)
def _load_verify_prompt() -> str:
    return _VERIFY_PROMPT_PATH.read_text(encoding="utf-8")


@functools.lru_cache(maxsize=1)
def _load_triage_prompt() -> str:
    return _TRIAGE_PROMPT_PATH.read_text(encoding="utf-8")


_TRIAGE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "needs_search": {"type": "boolean"},
        "verdict": {
            "type": ["string", "null"],
            "enum": ["supported", "contradicted", None],
        },
        "confidence": {
            "type": ["string", "null"],
            "enum": ["high", "medium", "low", None],
        },
        "explanation": {"type": "string"},
    },
    "required": ["needs_search", "explanation"],
}


def _triage_claim(
    claim: FactualClaim,
    *,
    ollama_client: Any,
    ollama_model: str,
) -> dict:
    """Fast-path decision: can this claim be adjudicated from the
    model's own knowledge, or does it need web search? Returns the
    parsed triage response with keys: needs_search, verdict,
    confidence, explanation."""
    system_prompt = _load_triage_prompt()
    user_prompt = (
        f"Claim: {claim.text}\n"
        f"Entities: {', '.join(claim.entities) if claim.entities else '(none)'}\n"
        f"Claim type: {claim.claim_type}"
    )
    data, _raw = _ollama_json_call(
        ollama_client=ollama_client,
        ollama_model=ollama_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        schema=_TRIAGE_SCHEMA,
        stage="verify",
        claim_id=claim.claim_id,
    )
    return data


# --- Stage A: load ---------------------------------------------------------


_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def _load_source(
    source: str | Path,
    *,
    _whisper_model: Any = None,
) -> tuple[InputSource, str]:
    source_str = str(source)
    if _URL_RE.match(source_str):
        text = _transcribe_via_podcast_processor(source_str, is_url=True,
                                                 _whisper_model=_whisper_model)
        return InputSource(kind="youtube", url=source_str), text

    # A caller who passed a Path instance clearly expected a file, not
    # text. Refuse to silently downgrade a nonexistent Path to raw text
    # (a real UX bug: typo'd paths would become 30-char "transcripts").
    if isinstance(source, Path):
        if not source.exists() or not source.is_file():
            raise FactCheckError(
                f"file {source_str!r} does not exist or is not a regular file.",
                partial=FactCheckAttempt(stage="load"),
            )
        return _load_existing_path(source, _whisper_model=_whisper_model)

    # `source` is a str: could be a real path OR raw text. Only treat
    # as a path if the string actually resolves to a file on disk.
    path = Path(source_str)
    if path.exists() and path.is_file():
        return _load_existing_path(path, _whisper_model=_whisper_model)
    # Fall through: treat as raw text.
    if not source_str.strip():
        raise FactCheckError(
            "source is empty. Pass a URL, a path, or non-empty text.",
            partial=FactCheckAttempt(stage="load"),
        )
    return InputSource(kind="text", raw_text=source_str), source_str


def _load_existing_path(
    path: Path, *, _whisper_model: Any = None
) -> tuple[InputSource, str]:
    suffix = path.suffix.lower()
    if suffix in _AUDIO_SUFFIXES:
        text = _transcribe_via_podcast_processor(str(path), is_url=False,
                                                 _whisper_model=_whisper_model)
        return InputSource(kind="audio", path=path.resolve()), text
    if suffix in _TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace")
        return InputSource(kind="transcript_file", path=path.resolve()), text
    raise FactCheckError(
        f"unsupported file suffix {suffix!r}; use audio "
        f"{list(_AUDIO_SUFFIXES)}, text {list(_TEXT_SUFFIXES)}, or a "
        "raw string.",
        partial=FactCheckAttempt(stage="load"),
    )


def _transcribe_via_podcast_processor(
    source: str, *, is_url: bool, _whisper_model: Any = None
) -> str:
    """Lazy-import #14's transcriber. Cross-agent import via importlib
    keeps agent #16's pyproject.toml from forcing faster-whisper /
    yt-dlp on a minimal install (they're picked up if #14 is already
    installed via `uv sync --all-packages`)."""
    try:
        pp = importlib.import_module("14_podcast_processor.agent")
    except ImportError as exc:
        raise FactCheckError(
            "audio and YouTube inputs require agent #14 "
            "(podcast_processor). Run `uv sync --all-packages` from the "
            "repo root to install it, or pass a .md/.txt/raw-text source.",
            partial=FactCheckAttempt(stage="load"),
        ) from exc
    import tempfile

    with tempfile.TemporaryDirectory(prefix="fact_checker_") as tmp:
        tmp_path = Path(tmp)
        try:
            audio_src = pp._fetch_audio(source, tmpdir=tmp_path)
            transcript = pp._transcribe(
                audio_src,
                whisper_size="small",
                compute_type="int8",
                device="cpu",
                _whisper_model=_whisper_model,
            )
        except Exception as exc:
            raise FactCheckError(
                f"transcription via #14 failed: {type(exc).__name__}: {exc}",
                partial=FactCheckAttempt(stage="load"),
            ) from exc
    return transcript.full_text


# --- Stage B: extract ------------------------------------------------------


_CLAIMS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "claim_type": {
                        "type": "string",
                        "enum": [
                            "statistic",
                            "date",
                            "quote",
                            "event",
                            "attribution",
                            "other",
                        ],
                    },
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["text", "claim_type"],
            },
        }
    },
    "required": ["claims"],
}


def _ollama_json_call(
    *,
    ollama_client: Any,
    ollama_model: str,
    messages: list[dict],
    schema: dict,
    stage: str,
    claim_id: str | None = None,
) -> tuple[dict, str]:
    """Ollama structured-output call with retry-once-on-bad-JSON.

    Returns (parsed_data, last_raw_text). Small local models
    occasionally emit malformed JSON even under a schema constraint;
    the second attempt appends the parse error to the last message so
    the model can correct itself. Mirrors #10's / #14's retry-once
    pattern."""
    last_raw = ""
    last_error = ""
    working_messages = list(messages)
    for attempt in range(_JSON_PARSE_RETRIES + 1):
        try:
            # `think=False` disables reasoning-mode output on models
            # that support it (qwen3.x, deepseek-r1, etc.). Without it,
            # reasoning tokens crowd out the actual structured response
            # and .message.content comes back empty. Non-reasoning
            # models silently ignore the flag.
            resp = ollama_client.chat(
                model=ollama_model,
                messages=working_messages,
                format=schema,
                options={"temperature": 0.0, "num_ctx": _OLLAMA_NUM_CTX},
                think=False,
            )
        except Exception as exc:
            raise _translate_llm_error(exc, stage=stage, claim_id=claim_id) from exc
        last_raw = _ollama_response_text(resp)
        try:
            return json.loads(last_raw), last_raw
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            if attempt >= _JSON_PARSE_RETRIES:
                raise FactCheckError(
                    f"Ollama returned non-JSON output at stage {stage!r} "
                    f"after {attempt + 1} attempt(s): {last_error}",
                    partial=FactCheckAttempt(
                        stage=stage, claim_id=claim_id, raw_output=last_raw  # type: ignore[arg-type]
                    ),
                ) from exc
            working_messages = list(messages) + [
                {
                    "role": "user",
                    "content": (
                        f"Your previous response could not be parsed as JSON: "
                        f"{last_error}. Return valid JSON matching the schema."
                    ),
                }
            ]
    raise FactCheckError(  # unreachable
        f"retry loop at stage {stage!r} exited without a result.",
        partial=FactCheckAttempt(stage=stage, claim_id=claim_id),  # type: ignore[arg-type]
    )


def _extract_claims(
    source_text: str,
    *,
    ollama_client: Any,
    ollama_model: str,
    run_meta: dict[str, Any],
) -> list[FactualClaim]:
    system_prompt = _load_extract_prompt()
    user_prompt = f"Source text:\n\n{source_text}"
    data, _raw = _ollama_json_call(
        ollama_client=ollama_client,
        ollama_model=ollama_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        schema=_CLAIMS_SCHEMA,
        stage="extract",
    )
    raw_claims = data.get("claims", [])
    claims: list[FactualClaim] = []
    drops: list[str] = []
    for i, item in enumerate(raw_claims):
        text = (item.get("text") or "").strip()
        if not text or not _substring_of_normalized(text, source_text):
            drops.append(text[:80] if text else "(empty)")
            continue
        entities = [e for e in (item.get("entities") or []) if e][:_MAX_ENTITIES]
        try:
            claims.append(
                FactualClaim(
                    claim_id=f"c{len(claims):03d}",
                    text=text,
                    claim_type=item.get("claim_type", "other"),
                    entities=entities,
                    approx_char_offset=_normalized_find(source_text, text),
                )
            )
        except ValidationError as exc:
            drops.append(f"validation failed on {text[:60]!r}: {exc}")
            continue
    run_meta["claim_extraction_drops"] = drops
    return claims


def _normalized_find(haystack: str, needle: str) -> int | None:
    """Best-effort char offset of `needle` in `haystack`. Returns the
    first exact-match position; if only the normalized form matches,
    returns None (offset is approximate at best)."""
    pos = haystack.find(needle)
    return pos if pos >= 0 else None


# --- Stage C: verify -------------------------------------------------------


_VERDICT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["supported", "contradicted", "unclear", "unverifiable"],
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "explanation": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quoted_text": {"type": "string"},
                    "source_url": {"type": "string"},
                },
                "required": ["quoted_text", "source_url"],
            },
        },
    },
    "required": ["verdict", "confidence", "explanation", "evidence"],
}


def _verify_claim(
    claim: FactualClaim,
    *,
    ollama_client: Any,
    ollama_model: str,
    search_client: SearchClient,
    max_search_results: int,
    run_meta: dict[str, Any],
) -> ClaimVerdict:
    # Agentic fast path: ask the model whether this claim is
    # answerable from well-established knowledge (2+2=4, capital of
    # France, etc.). If YES with high confidence, skip the web
    # searches entirely -- saves ~15-30s and 2 search calls per
    # claim on the fast path.
    triage = _triage_claim(
        claim, ollama_client=ollama_client, ollama_model=ollama_model
    )
    decision = "no_search" if not triage.get("needs_search", True) else "web_search"
    counts = run_meta.setdefault("triage_decisions", {"no_search": 0, "web_search": 0})
    counts[decision] = counts.get(decision, 0) + 1
    if (
        not triage.get("needs_search", True)
        and triage.get("verdict") in ("supported", "contradicted")
        and triage.get("confidence") == "high"
    ):
        # Fast path: model self-attests high confidence on a
        # universally-known fact. Skip search entirely.
        try:
            return ClaimVerdict(
                claim=claim,
                verdict=triage["verdict"],
                confidence="high",
                explanation=(
                    (triage.get("explanation") or "").strip()
                    + " [Adjudicated from model knowledge; no web search "
                    "performed because this is a well-established fact.]"
                ),
                evidence=[],
                verification_method="model_knowledge",
            )
        except ValidationError:
            # Model's self-triage failed schema; fall through to
            # search rather than raising.
            pass

    # Slow path: two-pass grounded web search + LLM adjudication.
    grounding_query = _build_grounding_query(claim)
    specific_query = _build_search_query(claim)
    all_hits: list[SearchHit] = []
    try:
        if grounding_query and grounding_query != specific_query:
            grounding_hits = search_client.search(
                grounding_query, max_results=max_search_results
            )
            all_hits.extend(grounding_hits)
        specific_hits = search_client.search(
            specific_query, max_results=max_search_results
        )
        all_hits.extend(specific_hits)
    except SearchAllUnavailable as exc:
        raise FactCheckError(
            f"search chain exhausted while verifying {claim.claim_id!r}: {exc}",
            partial=FactCheckAttempt(stage="verify", claim_id=claim.claim_id),
        ) from exc
    provider_used = getattr(search_client, "last_used_provider", None) or getattr(
        search_client, "provider_name", "ddg"
    )
    # Dedupe by URL, keeping the FIRST hit per URL. Grounding hits
    # come first so if the same URL surfaces in both passes, the
    # grounding version wins (typically the more entity-focused
    # snippet).
    hits_by_url_first: dict[str, SearchHit] = {}
    for h in all_hits:
        if h.url and h.url not in hits_by_url_first:
            hits_by_url_first[h.url] = h
    hits = list(hits_by_url_first.values())

    if not hits:
        return ClaimVerdict(
            claim=claim,
            verdict="unverifiable",
            confidence="low",
            explanation="No search results returned for this claim.",
            evidence=[],
        )

    system_prompt = _load_verify_prompt()
    user_prompt = _format_verify_prompt(claim, hits)
    data, raw_text = _ollama_json_call(
        ollama_client=ollama_client,
        ollama_model=ollama_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        schema=_VERDICT_SCHEMA,
        stage="verify",
        claim_id=claim.claim_id,
    )

    evidence_out: list[EvidenceSnippet] = []
    evidence_drops: list[str] = []
    for ev in data.get("evidence", []):
        quoted = (ev.get("quoted_text") or "").strip()
        url = (ev.get("source_url") or "").strip()
        hit = hits_by_url_first.get(url)
        if hit is None or not quoted:
            evidence_drops.append(f"orphan or empty: {url!r}")
            continue
        if not _substring_of_normalized(quoted, hit.snippet):
            evidence_drops.append(
                f"non-verbatim from {hit.url!r}: {quoted[:60]!r}"
            )
            continue
        evidence_out.append(
            EvidenceSnippet(
                source_url=url,
                source_title=hit.title or url,
                quoted_text=quoted,
                search_provider=hit.provider,
            )
        )

    if evidence_drops:
        run_meta.setdefault("evidence_drops", {})[claim.claim_id] = evidence_drops
    run_meta.setdefault("search_providers_used", set()).add(provider_used)

    verdict_str = data.get("verdict", "unclear")
    confidence_str = data.get("confidence", "low")
    explanation = (data.get("explanation") or "").strip()
    if not explanation:
        explanation = "Model did not provide an explanation."

    # If the model claimed a strong verdict but every piece of evidence
    # was dropped as non-verbatim, degrade to 'unclear' to satisfy the
    # ClaimVerdict validator instead of raising.
    if verdict_str in ("supported", "contradicted") and not evidence_out:
        verdict_str = "unclear"
        confidence_str = "low"

    try:
        return ClaimVerdict(
            claim=claim,
            verdict=verdict_str,  # type: ignore[arg-type]
            confidence=confidence_str,  # type: ignore[arg-type]
            explanation=explanation,
            evidence=evidence_out
            if verdict_str != "unverifiable"
            else [],
        )
    except ValidationError as exc:
        raise FactCheckError(
            f"verdict for {claim.claim_id!r} failed validation: {exc}",
            partial=FactCheckAttempt(
                stage="verify", claim_id=claim.claim_id, raw_output=raw_text
            ),
        ) from exc


def _build_grounding_query(claim: FactualClaim) -> str:
    """Build a query that grounds the model on what the claim's
    key entities / terms ARE, so a post-training-cutoff concept
    (like a new tech term the model doesn't recognize) isn't judged
    incorrectly for lack of context. Returns empty string when the
    claim has no entities worth grounding."""
    if not claim.entities:
        return ""
    # Take the two most-prominent entities and pair with a definitional
    # cue so the search surfaces "what is X" style results, not just
    # incidental mentions.
    entities = claim.entities[:2]
    return f"what is {' '.join(entities)} definition explanation"


def _build_search_query(claim: FactualClaim) -> str:
    text = claim.text.strip()
    if len(text) > 150:
        # Word-boundary truncate.
        truncated = text[:150].rsplit(" ", 1)[0]
    else:
        truncated = text
    if claim.entities:
        query = " ".join(claim.entities) + " " + truncated
    else:
        query = truncated
    # Cap total query length -- Tavily / Brave both reject around 400+
    # chars and a spuriously long entity list would otherwise bomb the
    # search call.
    if len(query) > _MAX_SEARCH_QUERY_LEN:
        query = query[:_MAX_SEARCH_QUERY_LEN].rsplit(" ", 1)[0]
    return query


def _format_verify_prompt(claim: FactualClaim, hits: list[SearchHit]) -> str:
    lines = [
        f"Claim: {claim.text}",
        f"Entities: {', '.join(claim.entities) if claim.entities else '(none)'}",
        "",
        "Search results:",
    ]
    for i, h in enumerate(hits, start=1):
        lines.append(f"[{i}] {h.title}")
        lines.append(f"    URL: {h.url}")
        lines.append(f"    Snippet: {h.snippet}")
        lines.append("")
    lines.append(
        "Return a JSON verdict per the rubric. Every quoted_text in evidence "
        "MUST be a verbatim substring of one of the snippets above; the "
        "caller drops any non-verbatim quotes."
    )
    return "\n".join(lines)


def _ollama_response_text(resp: Any) -> str:
    """Extract the message text from an Ollama chat response. Handles
    both dict-shaped (ollama<0.4) and object-shaped (ollama>=0.4)
    responses defensively; raises FactCheckError when `content` is
    missing, None, or empty (some server error paths return an
    object with .message.content == None instead of raising)."""
    msg = getattr(resp, "message", None)
    if msg is not None:
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content:
            return content
    if isinstance(resp, dict):
        message = resp.get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content:
            return content
    raise FactCheckError(
        f"Ollama response missing message.content: {type(resp).__name__}"
    )


# --- Stage D: assemble ----------------------------------------------------


def _assemble_report(
    source: InputSource,
    source_text: str,
    claims: list[FactualClaim],
    verdicts: list[ClaimVerdict],
    run_meta: dict[str, Any],
) -> FactCheckReport:
    summary = {
        v: sum(1 for x in verdicts if x.verdict == v)
        for v in ("supported", "contradicted", "unclear", "unverifiable")
    }
    # `set` values in run_meta don't JSON-serialize; convert here so
    # the dump in the CLI works.
    if isinstance(run_meta.get("search_providers_used"), set):
        run_meta["search_providers_used"] = sorted(run_meta["search_providers_used"])
    return FactCheckReport(
        source=source,
        source_text=source_text,
        claims=claims,
        verdicts=verdicts,
        summary=summary,
        run_meta=run_meta,
    )


# --- Public entry point ----------------------------------------------------


def fact_check(
    source: str | Path,
    *,
    ollama_model: str = _DEFAULT_OLLAMA_MODEL,
    ollama_host: str = _DEFAULT_OLLAMA_HOST,
    search_provider: Literal["auto", "tavily", "brave", "ddg"] = "auto",
    max_search_results: int = _DEFAULT_MAX_SEARCH_RESULTS,
    top_k_claims: int | None = None,
    provider: str | None = None,
    _ollama_client: Any = None,
    _search: Any = None,
    _whisper_model: Any = None,
) -> FactCheckReport:
    try:
        resolved = provider or resolve_provider()
    except ValueError as exc:
        raise FactCheckError(str(exc)) from exc

    if resolved == "mock":
        return _mock_report(source)

    input_src, source_text = _load_source(source, _whisper_model=_whisper_model)
    run_meta: dict[str, Any] = {
        "provider": resolved,
        "ollama_model": ollama_model,
        "search_provider_config": search_provider,
    }

    start = time.perf_counter()
    ollama_client = _ollama_client if _ollama_client is not None else _build_ollama(ollama_host)
    claims = _extract_claims(
        source_text,
        ollama_client=ollama_client,
        ollama_model=ollama_model,
        run_meta=run_meta,
    )
    run_meta["extract_seconds"] = round(time.perf_counter() - start, 3)

    if top_k_claims is not None:
        claims = claims[:top_k_claims]
    run_meta["claim_count"] = len(claims)

    search_client = _search if _search is not None else build_search_client(search_provider)

    start = time.perf_counter()
    verdicts: list[ClaimVerdict] = []
    for c in claims:
        verdicts.append(
            _verify_claim(
                c,
                ollama_client=ollama_client,
                ollama_model=ollama_model,
                search_client=search_client,
                max_search_results=max_search_results,
                run_meta=run_meta,
            )
        )
    run_meta["verify_seconds"] = round(time.perf_counter() - start, 3)

    try:
        return _assemble_report(input_src, source_text, claims, verdicts, run_meta)
    except ValidationError as exc:
        raise FactCheckError(
            f"final FactCheckReport failed validation: {exc}",
            partial=FactCheckAttempt(stage="assemble"),
        ) from exc


def _build_ollama(host: str) -> Any:
    try:
        import ollama
    except ImportError as exc:
        raise FactCheckError(
            "ollama not installed. Run `uv sync --all-packages` from the "
            "repo root."
        ) from exc
    return ollama.Client(host=host)


# --- Error translation (R5) ------------------------------------------------


def _translate_llm_error(
    exc: Exception, *, stage: str, claim_id: str | None = None
) -> FactCheckError:
    exc_class = type(exc).__name__.lower()
    message_lower = str(exc).lower()
    status = getattr(exc, "status_code", None)
    partial = FactCheckAttempt(stage=stage, claim_id=claim_id)  # type: ignore[arg-type]

    if "connection" in message_lower or "connect" in exc_class or "refused" in message_lower:
        return FactCheckError(
            f"Ollama connection failed at {stage!r}: is `ollama serve` "
            "running on the configured host? See "
            "https://ollama.com/download to install, then "
            "`ollama pull llama3.1:8b` to fetch the default model.",
            partial=partial,
        )
    if "not found" in message_lower and "model" in message_lower:
        return FactCheckError(
            "Ollama model not pulled: run `ollama pull llama3.1:8b` "
            "(or whichever model you passed via --model).",
            partial=partial,
        )
    if "ratelimit" in exc_class or status == 429 or "rate limit" in message_lower:
        return FactCheckError(
            "Ollama is rate-limited (unusual for a local server). Wait "
            "a moment and try again.",
            partial=partial,
        )
    if status == 401 or "unauthor" in message_lower:
        return FactCheckError(
            "Ollama authentication failed. Check the OLLAMA_HOST env "
            "var and any proxy in front of the server.",
            partial=partial,
        )
    return FactCheckError(
        f"Ollama call failed at {stage!r}: {type(exc).__name__}: {exc}.",
        partial=partial,
    )


# --- Mock mode -------------------------------------------------------------


def _mock_report(source: str | Path) -> FactCheckReport:
    """Scripted FactCheckReport without touching Ollama / search /
    #14. Source-size echoes into run_meta as the anti-refactor
    guard."""
    source_str = str(source)
    if _URL_RE.match(source_str):
        input_src = InputSource(kind="youtube", url=source_str)
        source_text = (
            "The Eiffel Tower is located in Paris. "
            "The Great Wall of China is visible from the Moon with the naked eye. "
            "In 2019, our company launched a product used by three private customers."
        )
        mock_size = len(source_str)
    else:
        p = Path(source_str)
        if p.exists() and p.is_file() and p.suffix.lower() in _TEXT_SUFFIXES:
            source_text = p.read_text(encoding="utf-8", errors="replace")
            input_src = InputSource(kind="transcript_file", path=p.resolve())
            mock_size = p.stat().st_size
        else:
            source_text = source_str if source_str.strip() else "no source provided"
            input_src = InputSource(kind="text", raw_text=source_text)
            mock_size = len(source_text)

    claim_a = FactualClaim(
        claim_id="c000",
        text=_pick_snippet(source_text, "Eiffel Tower is located in Paris",
                           "Fact-checked truthful claim goes here"),
        claim_type="event",
        entities=["Eiffel Tower", "Paris"],
    )
    claim_b = FactualClaim(
        claim_id="c001",
        text=_pick_snippet(source_text,
                           "Great Wall of China is visible from the Moon",
                           "Fact-checked false claim goes here"),
        claim_type="event",
        entities=["Great Wall of China", "Moon"],
    )
    claim_c = FactualClaim(
        claim_id="c002",
        text=_pick_snippet(source_text,
                           "our internal team launched a research prototype",
                           "Unverifiable private anecdote goes here"),
        claim_type="statistic",
        entities=["internal team", "research prototype"],
    )
    verdicts = [
        ClaimVerdict(
            claim=claim_a,
            verdict="supported",
            confidence="high",
            explanation="[MOCK] Multiple sources confirm the location.",
            evidence=[
                EvidenceSnippet(
                    source_url="https://example.com/eiffel",
                    source_title="Eiffel Tower - Overview",
                    quoted_text="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
                    search_provider="tavily",
                )
            ],
        ),
        ClaimVerdict(
            claim=claim_b,
            verdict="contradicted",
            confidence="high",
            explanation="[MOCK] NASA and multiple astronomers have debunked the naked-eye visibility claim.",
            evidence=[
                EvidenceSnippet(
                    source_url="https://example.com/greatwall-myth",
                    source_title="The Great Wall from Space Myth",
                    quoted_text="The Great Wall is not visible from the Moon with the naked eye.",
                    search_provider="tavily",
                )
            ],
        ),
        ClaimVerdict(
            claim=claim_c,
            verdict="unverifiable",
            confidence="low",
            explanation="[MOCK] Private business claim; no public record to verify against.",
            evidence=[],
        ),
    ]
    claims = [claim_a, claim_b, claim_c]
    summary = {
        v: sum(1 for x in verdicts if x.verdict == v)
        for v in ("supported", "contradicted", "unclear", "unverifiable")
    }
    return FactCheckReport(
        source=input_src,
        source_text=source_text,
        claims=claims,
        verdicts=verdicts,
        summary=summary,
        run_meta={
            "provider": "mock",
            "ollama_model": "mock",
            "search_provider_config": "mock",
            "claim_count": 3,
            "mock_source_size": mock_size,
        },
    )


def _pick_snippet(source_text: str, preferred: str, fallback: str) -> str:
    """If `preferred` appears verbatim in source_text, return it; else
    return the first 120 chars of source_text (guaranteed substring) or
    a padded fallback if the source is too short."""
    if _substring_of_normalized(preferred, source_text):
        return preferred
    snippet = source_text.strip()[:120]
    if len(snippet) >= 10:
        return snippet
    return fallback


# --- CLI ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="fact-checker",
        description=(
            "Fact-check speech, video, transcript, or text against live web "
            "search using a local Ollama LLM. Set LLM_PROVIDER=mock for a "
            "scripted demo; otherwise `ollama serve` must be running and the "
            "model pulled (`ollama pull llama3.1:8b`)."
        ),
    )
    parser.add_argument("source", type=str, help="Local path, URL, or raw text.")
    parser.add_argument("--model", type=str, default=_DEFAULT_OLLAMA_MODEL,
                        help=f"Ollama model tag. Default: {_DEFAULT_OLLAMA_MODEL}.")
    parser.add_argument("--host", type=str, default=_DEFAULT_OLLAMA_HOST,
                        help=f"Ollama host URL. Default: {_DEFAULT_OLLAMA_HOST}.")
    parser.add_argument(
        "--search",
        choices=("auto", "tavily", "brave", "ddg"),
        default="auto",
        help="Search provider chain. Default: auto (Tavily -> Brave -> DDG).",
    )
    parser.add_argument("--top-k", type=int, default=None,
                        help="Cap the number of claims to verify.")
    parser.add_argument("--max-search-results", type=int,
                        default=_DEFAULT_MAX_SEARCH_RESULTS)
    parser.add_argument(
        "--provider", choices=(*SUPPORTED_PROVIDERS, "mock"), default=None
    )
    args = parser.parse_args()

    try:
        report = fact_check(
            args.source,
            ollama_model=args.model,
            ollama_host=args.host,
            search_provider=args.search,
            max_search_results=args.max_search_results,
            top_k_claims=args.top_k,
            provider=args.provider,
        )
    except FactCheckError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.partial is not None and exc.partial.raw_output:
            print(f"raw model output:\n{exc.partial.raw_output}", file=sys.stderr)
        return 1

    out_path = Path(__file__).parent / "last_run.json"
    out_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )

    print(f"Source: {report.source.kind}")
    print(f"Claims: {len(report.claims)} extracted, {len(report.verdicts)} verified")
    print(
        "Summary: "
        + " | ".join(f"{k}={v}" for k, v in report.summary.items())
    )
    print()
    for v in report.verdicts:
        verdict_label = v.verdict.upper()
        print(f"[{verdict_label}] ({v.confidence}) {v.claim.text[:120]}")
        print(f"    -> {v.explanation}")
        if v.evidence:
            print(f"    evidence ({len(v.evidence)}): {v.evidence[0].source_url}")
        print()
    print(f"Full trace written to {out_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
