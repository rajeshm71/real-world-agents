"""Contract reviewer agent -- agent #02 of real-world-agents.

Technique demonstrated: **long-context analysis with a hand-rolled
JSON-validate-retry loop, no framework**. This is deliberately the
"here's what Instructor (agent #01) does under the hood" lesson --
same pedagogical pattern, made visible instead of abstracted:

    prompt the model for JSON -> parse it -> validate against Pydantic
    -> on failure, feed the error back into the prompt and retry.

The whole retry loop lives in `_run_review_loop` below. Reading that
one function tells you everything Instructor's `response_model=...`
parameter is doing invisibly at agent #01. Not a reason to reject
Instructor -- a reason to understand it.

Why this technique for this use case: contracts are text (unlike
receipts, which are images), and the whole document fits in a modern
model's context window in one call (gpt-4.1-mini has ~1M tokens
verified 2026-08-21 -- ~750k words, comfortably larger than any
realistic single contract). Chunking or RAG would add complexity for
zero quality gain here. Long-context single-pass IS the right shape.

Real error handling (R5 in CONTRIBUTING.md's hard rules): three
concrete failure modes handled explicitly (see review_contract below):
  1. Scanned/image-only PDF (pypdf extracts <100 chars/page averaged
     or <200 chars total) -> ContractReviewError with a friendly "this
     appears to be scanned, OCR-first tools are outside scope" message
  2. Document exceeds context window (rare at 1M tokens but possible
     for a huge master-agreement archive) -> ContractReviewError with
     the model's cap named, not a silent truncation
  3. Rate limit / API failure -> translated to ContractReviewError
     with a clear "temporarily rate-limited" message. This agent does
     NOT auto-retry transient API errors: the user pays per real API
     call, and is better-placed to decide whether to wait and re-run
     than to have the loop
     silently spend more of their budget on a failing endpoint. The
     retry loop only re-prompts for VALIDATION failures (bad JSON,
     schema mismatch, paraphrased excerpts) -- those cost the same
     regardless of how many attempts, and each retry usually succeeds
     since the model gets the error text as feedback.

The `excerpt` field on every ContractFlag is validated as a verbatim
substring of the source contract at loop time (not by Pydantic --
Pydantic can't see the source from inside a field validator). A flag
whose excerpt was paraphrased triggers the same retry mechanism as a
Pydantic validation failure. This is why the "hand-rolled" retry loop
is MORE educational than "just use Instructor" here: the loop can
enforce a cross-field invariant (excerpt-in-source) that a per-field
Pydantic validator can't reach.

Provider + model are fully user-configurable: every real LLM call goes
through `common.llm.get_llm()` / `resolve_model()`, so the LLM_PROVIDER
env var and per-provider _DEFAULT_MODEL overrides work exactly as they
do for agent #01, unchanged.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

# Dual-mode import. Same rationale as agent #01: `.schemas` (relative)
# resolves when this file is loaded as a submodule of the
# 02_contract_reviewer package (how tests import it, via importlib since
# the dir name starts with a digit). The bare `schemas` (absolute)
# resolves when this file is run directly via `python -m agent` from
# inside the agent's own directory (the documented CLI invocation).
try:
    from .schemas import ContractFlag, ContractReview
except ImportError:
    from schemas import ContractFlag, ContractReview

from common.llm import LLM, get_llm, resolve_model

# --- Provider ---

SUPPORTED_PROVIDERS = ("openai", "anthropic", "gemini", "ollama")

# --- Scanned-PDF detection thresholds ([M3] in the F2 plan) ---
#
# pypdf on a truly scanned (image-only) PDF returns ~0 chars per page or
# a handful of stray OCR-like glyphs. Two thresholds so pathologically
# short but genuine documents (a one-page NDA cover memo) don't get
# falsely flagged as scanned:
#   - < MIN_CHARS_PER_PAGE averaged across pages, OR
#   - < MIN_TOTAL_CHARS total
# Whichever triggers first. Numbers chosen defensibly, not by whim:
# 100 chars/page is well below the ~1500-3000 chars/page of a real
# legal document, and 200 chars total catches the "one blank page,
# effectively empty" edge case that page-averaging alone would miss.
MIN_CHARS_PER_PAGE = 100
MIN_TOTAL_CHARS = 200

# --- Context-window ceiling ([M-original R5 case 2] in the F2 plan) ---
#
# We're not calling the provider's native token-count endpoint (would
# tie the agent to a specific SDK); the ~4-chars-per-token English
# heuristic gives a defensible upper bound before the model would
# reject the request. Ceiling set well below gpt-4.1-mini's ~1M cap so
# there's headroom for the prompt template + the model's response.
CHARS_PER_TOKEN_ESTIMATE = 4
MAX_INPUT_TOKENS_ESTIMATE = 900_000

# --- Retry policy ---

DEFAULT_MAX_RETRIES = 3  # covers both bad-JSON and paraphrased-excerpt cases
DEFAULT_MAX_TOKENS_OUT = 4096  # roomy for many flags per contract
RETRY_BACKOFF_SECONDS = 1.0  # base delay; multiplied by attempt number

# --- Prompt template ---

_PROMPT_PATH = Path(__file__).parent / "prompts" / "review.txt"


def _load_prompt_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


# --- Provider resolution ---


def resolve_provider() -> str:
    """LLM_PROVIDER env var, defaulting to "openai". No provider is
    hardcoded; every provider is equally supported. "mock" is handled by
    the caller (review_contract), not here."""
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider != "mock" and provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            f"Expected 'mock' or one of {SUPPORTED_PROVIDERS}."
        )
    return provider


# --- Error type ---


@dataclass
class ReviewAttempt:
    """What we got back before the retry loop gave up. Attached to
    ContractReviewError so the UI can surface the raw model output in a
    warning banner rather than dropping it silently."""

    raw_text: str
    validation_errors: list[str] = field(default_factory=list)


class ContractReviewError(Exception):
    """Raised on any user-facing review failure (scanned PDF, over-size
    document, API failure, retries exhausted). Includes a user-friendly
    `message` and, when relevant, the raw model output via `partial`."""

    def __init__(self, message: str, partial: ReviewAttempt | None = None):
        super().__init__(message)
        self.message = message
        self.partial = partial


# --- Public API ---


def review_contract(
    input_data: bytes | str,
    *,
    provider: str | None = None,
    model: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    _llm: LLM | None = None,  # test-injection escape hatch
) -> ContractReview:
    """Review a contract and return flagged clauses + summary.

    Args:
        input_data: contract as either raw PDF bytes (auto-detected via
            magic bytes) or plain text (str).
        provider: "openai" (default) / "anthropic" / "gemini" / "mock".
            Defaults to the LLM_PROVIDER env var if not passed.
        model: model ID for the resolved provider. Defaults to that
            provider's DEFAULT_MODELS entry, overridable per-provider
            via env var (see common.llm.resolve_model()).
        max_retries: how many times the JSON-validate-retry loop
            re-prompts on failure before raising. Default 3 covers a
            bad-JSON blip + one paraphrased-excerpt retry + one buffer.
        _llm: injected LLM for tests. Production callers leave this
            None; the client is built from provider/model.

    Returns:
        A validated ContractReview.

    Raises:
        ContractReviewError: on any of the three R5-required failure
            modes, or on retry exhaustion.
    """
    resolved_provider = (provider or resolve_provider()).lower()

    source_text, page_count = _load_source_text(input_data)

    if resolved_provider == "mock":
        # Bypass provider entirely under mock. Tests exercise this path.
        return _mock_review(source_text, page_count)

    # R5 case 1: scanned/image-only PDF.
    if page_count is not None and _is_likely_scanned(source_text, page_count):
        raise ContractReviewError(
            f"This PDF appears to be scanned (only {len(source_text)} chars "
            f"of text extracted across {page_count} page(s)). Contract text "
            "extraction needs a text-based PDF. OCR-first tools like "
            "Tesseract or unstructured.io can convert scanned PDFs to text; "
            "they are outside this agent's scope."
        )

    resolved_model = model or resolve_model(resolved_provider)

    # R5 case 2: document exceeds context window (rare at 1M tokens but
    # real for pathological input, e.g. a bundled master-agreement
    # archive).
    _check_context_window(source_text, resolved_model)

    llm = _llm if _llm is not None else get_llm(resolved_provider)

    try:
        return _run_review_loop(
            source_text=source_text,
            llm=llm,
            model=resolved_model,
            max_retries=max_retries,
        )
    except ContractReviewError:
        raise
    except Exception as exc:  # R5 case 3: rate limit / API failure
        raise _translate_api_error(exc) from exc


# --- Source loading ---


_PDF_MAGIC = b"%PDF-"


def _load_source_text(input_data: bytes | str) -> tuple[str, int | None]:
    """Resolve `input_data` to (text_with_page_markers, page_count).

    - `str` input -> the string itself, page_count None (no page structure).
    - `bytes` starting with the PDF magic header -> pypdf per-page
      extraction, with `--- PAGE N ---` markers injected between pages
      for the schema's `page_number` field to reference ([M4]).
    - `bytes` not starting with the PDF magic -> assumed plain text,
      decoded as UTF-8 (errors="replace" so a stray byte doesn't crash
      the whole review), page_count None.
    """
    if isinstance(input_data, str):
        return input_data, None
    if input_data.startswith(_PDF_MAGIC):
        return _extract_pdf_text(input_data)
    return input_data.decode("utf-8", errors="replace"), None


def _extract_pdf_text(pdf_bytes: bytes) -> tuple[str, int]:
    """Per-page text extraction via pypdf, with '--- PAGE N ---' markers
    injected between pages so the model can populate ContractFlag.page_number
    from those markers (the model returns int pages; schemas.py validates
    them). Lazy import so mock-mode tests don't require pypdf installed.

    Raises ContractReviewError with an actionable message if pypdf isn't
    installed, rather than letting the outer generic ImportError bubble up
    and get misclassified by _translate_api_error().
    """
    import io

    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ContractReviewError(
            "pypdf is not installed but this input is a PDF. "
            "Install with `pip install pypdf>=5.0` (or `uv sync` at the "
            "workspace root), or pass plain text (str) instead of PDF bytes."
        ) from exc

    reader = PdfReader(io.BytesIO(pdf_bytes))
    page_count = len(reader.pages)
    parts: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        parts.append(f"--- PAGE {i} ---\n{page_text}")
    return "\n\n".join(parts), page_count


def _is_likely_scanned(text: str, page_count: int) -> bool:
    """R5 case 1 detection. See MIN_CHARS_PER_PAGE / MIN_TOTAL_CHARS
    docstrings for the thresholds' rationale."""
    if len(text) < MIN_TOTAL_CHARS:
        return True
    return page_count > 0 and (len(text) / page_count) < MIN_CHARS_PER_PAGE


def _check_context_window(text: str, model: str) -> None:
    """R5 case 2. Raises ContractReviewError with the model named if the
    input alone (before adding the prompt template + reserving output
    tokens) already exceeds MAX_INPUT_TOKENS_ESTIMATE. Uses a rough
    chars-per-token heuristic, not a native tokenizer (would tie us to
    one provider's SDK)."""
    estimated = len(text) // CHARS_PER_TOKEN_ESTIMATE
    if estimated > MAX_INPUT_TOKENS_ESTIMATE:
        raise ContractReviewError(
            f"This contract is too long for single-pass analysis "
            f"(~{estimated:,} tokens estimated, over the {MAX_INPUT_TOKENS_ESTIMATE:,}-token "
            f"ceiling this agent applies before calling {model}). Split it "
            "into logical sections and review each separately, or bump "
            "MAX_INPUT_TOKENS_ESTIMATE in agent.py if you're using a "
            "larger-context model."
        )


# --- Retry loop (the pedagogical anchor -- read this to understand
# what Instructor's response_model= does under the hood) ---


def _run_review_loop(
    *,
    source_text: str,
    llm: LLM,
    model: str,
    max_retries: int,
) -> ContractReview:
    """The hand-rolled JSON-validate-retry loop. On each attempt:
      1. Build the prompt (adding retry-feedback if this isn't the
         first attempt).
      2. Call the LLM. Provider API errors bubble up to review_contract()
         for _translate_api_error() to translate.
      3. Parse the response as JSON. Failure -> feedback for next attempt.
      4. Validate against ContractReview. Failure -> feedback for next.
      5. Check every flag's `excerpt` is a substring of source_text
         (with whitespace normalization -- see _normalize_for_substring).
         Failure -> feedback for next.
      6. If all three pass, return.

    If we exhaust max_retries, raise ContractReviewError with the last
    raw output attached as `partial` so the UI can surface it.
    """
    prompt_template = _load_prompt_template()
    last_raw = ""
    last_errors: list[str] = []
    retry_feedback: str | None = None

    for attempt in range(1, max_retries + 1):
        prompt = _build_prompt(prompt_template, source_text, retry_feedback)
        response = llm.complete(
            prompt=prompt,
            model=model,
            temperature=0.0,
            max_tokens=DEFAULT_MAX_TOKENS_OUT,
        )
        last_raw = response.text

        parsed = _parse_json_object(last_raw)
        if isinstance(parsed, str):  # error message
            last_errors = [parsed]
            retry_feedback = f"Your previous response was not valid JSON: {parsed}"
            # Sleep ONLY if another attempt will happen; sleeping before the
            # final raise would just be wasted delay.
            if attempt < max_retries:
                _sleep_backoff(attempt)
            continue

        review, errors = _validate_review(parsed, source_text)
        if review is not None:
            return review

        last_errors = errors
        retry_feedback = (
            "Your previous response was valid JSON but failed these checks:\n"
            + "\n".join(f"- {e}" for e in errors)
            + "\nFix ONLY these issues; keep everything else the same."
        )
        if attempt < max_retries:
            _sleep_backoff(attempt)

    raise ContractReviewError(
        f"Model output failed validation after {max_retries} attempts. "
        f"The last raw response is attached below.",
        partial=ReviewAttempt(raw_text=last_raw, validation_errors=last_errors),
    )


def _sleep_backoff(attempt: int) -> None:
    """Linear backoff between retries. Kept tiny so tests aren't slow;
    real rate-limit backoff would be exponential, but our loop only
    exists for validation retries, not for transient-API-error retries
    (the SDK handles those). One second is a courtesy delay for whatever
    caused the model to return bad output last time."""
    time.sleep(RETRY_BACKOFF_SECONDS * attempt)


def _build_prompt(template: str, source_text: str, retry_feedback: str | None) -> str:
    """Fill the prompt template's {source_text} placeholder and prepend
    any retry-feedback block. Splitting this out from the loop keeps the
    loop readable (no string interpolation inline)."""
    filled = template.replace("{source_text}", source_text)
    if retry_feedback:
        return f"{retry_feedback}\n\n{filled}"
    return filled


# --- Response parsing + validation ---


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_object(raw_text: str) -> dict | str:
    """Extract and parse the first top-level JSON object from `raw_text`.
    Returns the parsed dict on success, or an error string on failure
    (returned as str so the caller can feed it into the retry prompt
    without a second exception layer).

    Handles the two most common model-authored deviations from "return
    only JSON":
      - markdown code fences (```json ... ```) around the object
      - leading/trailing prose ("Here is the JSON: {...}")
    """
    stripped = raw_text.strip()
    # Strip markdown code fences if present.
    if stripped.startswith("```"):
        # Drop opening fence line, then drop closing ``` line at end.
        lines = stripped.splitlines()
        if len(lines) >= 2:
            stripped = "\n".join(lines[1:-1]) if lines[-1].strip().startswith("```") else "\n".join(lines[1:])
    # Find the outermost {...} span. re.DOTALL so newlines match.
    match = _JSON_OBJECT_RE.search(stripped)
    if not match:
        return "no JSON object found in response"
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return f"JSON parse error: {exc.msg} at line {exc.lineno} col {exc.colno}"
    if not isinstance(parsed, dict):
        return f"expected a JSON object, got {type(parsed).__name__}"
    return parsed


def _validate_review(
    review_dict: dict, source_text: str
) -> tuple[ContractReview | None, list[str]]:
    """Two-stage validation:
      1. Pydantic (schema shape, closed Literals, required fields).
      2. Excerpt-substring check ([C5]): every flag's `excerpt` must
         appear in source_text after whitespace normalization.

    Returns (review, []) on success or (None, [error, ...]) on failure.
    The error messages are worded for the RETRY PROMPT (the model reads
    them), not for a human -- concrete, actionable, no jargon.
    """
    try:
        review = ContractReview.model_validate(review_dict)
    except Exception as exc:
        return None, [f"schema validation failed: {exc}"]

    normalized_source = _normalize_for_substring(source_text)
    errors: list[str] = []
    for i, flag in enumerate(review.flags):
        normalized_excerpt = _normalize_for_substring(flag.excerpt)
        if normalized_excerpt not in normalized_source:
            errors.append(
                f"flag {i} (category={flag.category!r}): excerpt "
                f"{flag.excerpt!r} is not a verbatim substring of the "
                f"contract text. Copy the exact wording from the contract."
            )
    if errors:
        return None, errors
    return review, []


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_substring(s: str) -> str:
    """Collapse runs of whitespace (spaces, tabs, newlines) to single
    spaces. PDF text extraction routinely inserts line breaks mid-
    sentence that a model would drop when quoting; without this
    normalization, otherwise-verbatim excerpts would fail the substring
    check for a whitespace-only reason. Case is preserved -- an
    intentional lowercase excerpt of an uppercase clause is a real
    paraphrase, not a whitespace artifact."""
    return _WHITESPACE_RE.sub(" ", s).strip()


# --- Error translation (R5 case 3) ---


def _translate_api_error(exc: Exception) -> ContractReviewError:
    """Turn an OpenAI/Anthropic/Gemini SDK exception into a user-facing
    ContractReviewError. Handles the two remaining R5 shapes not caught
    earlier (rate limit, auth/API failure); anything unrecognized bubbles
    up as a generic message with the original exception preserved.

    Priority order (exception CLASS is checked before message strings,
    mirroring agent #01's split structure so a reader sees the priority
    in the code shape, not just in the docstring):
      1. Rate-limit by class name -> rate-limit case
      2. Auth by class name -> auth case
      3. Status code 429 -> rate-limit case
      4. Status code 401 -> auth case
      5. Message-string fallback (rate limit / overloaded, auth / key)
      6. Generic fallback with original exception preserved
    """
    exc_class_name = type(exc).__name__.lower()
    message_lower = str(exc).lower()
    status = getattr(exc, "status_code", None)

    # 1. Class-name check: rate limit
    if "ratelimiterror" in exc_class_name:
        return _rate_limit_error()
    # 2. Class-name check: authentication
    if "authenticationerror" in exc_class_name or "apikeyerror" in exc_class_name:
        return _auth_error()
    # 3. Status-code check: rate limit
    if status == 429:
        return _rate_limit_error()
    # 4. Status-code check: auth
    if status == 401:
        return _auth_error()
    # 5. Message-string fallback (only reached when class/status haven't caught it)
    if "rate limit" in message_lower or "overloaded" in message_lower:
        return _rate_limit_error()
    if "authentication" in message_lower or "api key" in message_lower:
        return _auth_error()

    # 6. Unknown error class. Preserve original message for debugging.
    return ContractReviewError(
        f"Contract review failed: {type(exc).__name__}: {exc}. "
        "This is an unexpected error -- check the agent logs."
    )


def _rate_limit_error() -> ContractReviewError:
    """Shared message so the six code paths that reach this case can't
    drift apart. NOTE: this agent does NOT auto-retry rate-limit errors:
    the user is billed per real API call and is best-placed to decide
    whether to wait and re-run, rather than have the loop silently spend
    more of their budget."""
    return ContractReviewError(
        "The service is temporarily rate-limited or overloaded. "
        "Wait a minute and try again."
    )


def _auth_error() -> ContractReviewError:
    """Shared message, same drift-prevention rationale as _rate_limit_error()."""
    return ContractReviewError(
        "API authentication failed. Check that your LLM_PROVIDER matches "
        "the API key you've set in .env (OPENAI_API_KEY / "
        "ANTHROPIC_API_KEY / GEMINI_API_KEY)."
    )


# --- Mock mode ---


_MOCK_EXCERPT = "This Agreement shall automatically renew for successive 12-month terms unless either party gives 60 days written notice."


def _mock_review(source_text: str, page_count: int | None) -> ContractReview:
    """Deterministic canned review for smoke tests and CI (LLM_PROVIDER=mock).
    The mock excerpt is embedded into a synthetic 'contract text' so
    the mock's own ContractReview would ALSO pass excerpt validation if
    a test ran it through the loop. Character count of the input drives
    one field so tests can verify 'the mock actually saw the input.'"""
    return ContractReview(
        contract_type="Mutual NDA (mock)",
        parties=["Mock Corp.", "Example LLC"],
        flags=[
            ContractFlag(
                category="auto_renewal",
                severity="medium",
                excerpt=_MOCK_EXCERPT,
                page_number=1 if page_count else None,
                explanation=(
                    "The contract renews automatically each year unless you "
                    "cancel 60 days out. Easy to miss; put the cancellation "
                    "date on your calendar the day you sign."
                ),
                recommendation="Ask for a cancel-anytime clause with 30 days notice.",
            ),
        ],
        summary=(
            f"Mock review (input: {len(source_text)} chars). This is a "
            "deterministic canned response for LLM_PROVIDER=mock -- set "
            "LLM_PROVIDER to 'openai', 'anthropic', or 'gemini' and "
            "configure the matching API key for a real review."
        ),
        overall_risk="medium",
    )


# --- CLI entry point (uv run python -m agent) ---


def main() -> int:
    """CLI: takes a contract path (PDF or plain text), prints the review
    as JSON to stdout. Estimated cost is not printed here (cost accounting
    lives in common/pricing.py; hook up per-call cost printing when you
    add it to the UI at F2.3)."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="contract-reviewer",
        description="Review a contract PDF or text file, flagging risky clauses.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to a contract PDF or plain-text file.",
    )
    parser.add_argument(
        "--provider",
        choices=[*SUPPORTED_PROVIDERS, "mock"],
        help="Override LLM_PROVIDER for this invocation.",
    )
    parser.add_argument(
        "--model",
        help="Override the resolved model for this invocation.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the Gradio UI instead of a one-shot CLI run.",
    )
    args = parser.parse_args()

    if args.ui:
        # Lazy import: Gradio has a heavy import path we don't want in
        # every CLI invocation. Matches agent #01's build_ui() pattern
        # (returns the Blocks; caller calls .launch()) so all agents'
        # ui.py files have the same shape.
        try:
            from .ui import build_ui  # type: ignore[import-not-found]
        except ImportError:
            from ui import build_ui  # type: ignore[import-not-found]
        build_ui().launch()
        return 0

    if not args.path:
        parser.error("path is required unless --ui is passed")

    path = Path(args.path)
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2

    input_bytes: bytes | str
    input_bytes = path.read_bytes()

    try:
        review = review_contract(input_bytes, provider=args.provider, model=args.model)
    except ContractReviewError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.partial is not None:
            print("---- raw model output ----", file=sys.stderr)
            print(exc.partial.raw_text, file=sys.stderr)
        return 1

    print(review.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
