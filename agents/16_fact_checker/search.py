"""Web-search providers for the fact-checker.

Three concrete providers implement the `SearchClient` Protocol:
Tavily (LLM-oriented, needs `TAVILY_API_KEY`), Brave (needs
`BRAVE_API_KEY`, called via httpx to avoid a dedicated client dep),
and DuckDuckGo (needs no key, always available). `ChainProvider`
tries them in order and falls back on `SearchRateLimit` or a
provider that raises `SearchUnavailable` at construction time (e.g.
Tavily with no key).

`build_search_client(kind="auto")` composes the default chain based
on which env vars are set. `kind in {"tavily","brave","ddg"}` forces
a single provider.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Protocol


class SearchError(Exception):
    """Base class for search-provider failures."""


class SearchRateLimit(SearchError):
    """The provider signaled rate-limit; the chain should try the next."""


class SearchUnavailable(SearchError):
    """The provider is not usable in this environment (no API key,
    missing dep). Chain should skip it and try the next."""


class SearchAllUnavailable(SearchError):
    """Every provider in the chain either was unavailable or hit its
    rate limit. Fatal."""


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str
    provider: Literal["tavily", "brave", "ddg"]


class SearchClient(Protocol):
    provider_name: Literal["tavily", "brave", "ddg"]

    def search(self, query: str, *, max_results: int) -> list[SearchHit]: ...


class TavilyProvider:
    provider_name: Literal["tavily", "brave", "ddg"] = "tavily"

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("TAVILY_API_KEY")
        if not key:
            raise SearchUnavailable("TAVILY_API_KEY not set.")
        try:
            from tavily import TavilyClient
        except ImportError as exc:
            raise SearchUnavailable("tavily-python not installed.") from exc
        self._client = TavilyClient(api_key=key)

    def search(self, query: str, *, max_results: int) -> list[SearchHit]:
        try:
            resp = self._client.search(query=query, max_results=max_results)
        except Exception as exc:
            msg = str(exc).lower()
            if "rate" in msg or "429" in msg or "quota" in msg:
                raise SearchRateLimit(f"Tavily: {exc}") from exc
            raise
        return [
            SearchHit(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", "") or item.get("snippet", ""),
                provider="tavily",
            )
            for item in resp.get("results", [])
        ]


class BraveProvider:
    provider_name: Literal["tavily", "brave", "ddg"] = "brave"
    _ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str | None = None, *, timeout: float = 10.0):
        key = api_key or os.environ.get("BRAVE_API_KEY")
        if not key:
            raise SearchUnavailable("BRAVE_API_KEY not set.")
        try:
            import httpx
        except ImportError as exc:
            raise SearchUnavailable("httpx not installed.") from exc
        self._key = key
        self._timeout = timeout
        self._httpx = httpx

    def search(self, query: str, *, max_results: int) -> list[SearchHit]:
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._key,
        }
        params = {"q": query, "count": max_results}
        try:
            resp = self._httpx.get(
                self._ENDPOINT, headers=headers, params=params, timeout=self._timeout
            )
        except Exception as exc:
            raise SearchError(f"Brave HTTP call failed: {exc}") from exc
        if resp.status_code == 429:
            raise SearchRateLimit("Brave: 429.")
        if resp.status_code >= 400:
            raise SearchError(f"Brave: {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        results = (data.get("web") or {}).get("results", []) or []
        return [
            SearchHit(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                provider="brave",
            )
            for item in results[:max_results]
        ]


class DDGProvider:
    provider_name: Literal["tavily", "brave", "ddg"] = "ddg"

    def __init__(self) -> None:
        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise SearchUnavailable("ddgs not installed.") from exc
        self._DDGS = DDGS

    def search(self, query: str, *, max_results: int) -> list[SearchHit]:
        try:
            with self._DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
        except Exception as exc:
            msg = type(exc).__name__.lower()
            if "ratelimit" in msg or "rate_limit" in msg:
                raise SearchRateLimit(f"DDG: {exc}") from exc
            raise
        return [
            SearchHit(
                title=item.get("title", ""),
                url=item.get("href") or item.get("url", ""),
                snippet=item.get("body", "") or item.get("snippet", ""),
                provider="ddg",
            )
            for item in results
        ]


class ChainProvider:
    """Tries providers in order; on SearchRateLimit falls back to the
    next. Raises SearchAllUnavailable when the chain is exhausted.

    `provider_name` is intentionally "chain"; the actual provider that
    answered any given `.search()` call lives in `last_used_provider`.
    Callers that need the real provider should read that after the
    call returns."""

    provider_name: str = "chain"

    def __init__(self, providers: list[SearchClient]):
        if not providers:
            raise SearchAllUnavailable(
                "no search providers available. Set TAVILY_API_KEY or "
                "BRAVE_API_KEY, or install ddgs."
            )
        self._providers = providers
        self.last_used_provider: str | None = None

    def search(self, query: str, *, max_results: int) -> list[SearchHit]:
        last_error: Exception | None = None
        for p in self._providers:
            try:
                hits = p.search(query, max_results=max_results)
            except SearchRateLimit as exc:
                last_error = exc
                continue
            self.last_used_provider = p.provider_name
            return hits
        raise SearchAllUnavailable(
            f"every provider in the chain is rate-limited. Last error: {last_error}"
        )


def build_search_client(
    kind: Literal["auto", "tavily", "brave", "ddg"] = "auto",
    *,
    tavily_api_key: str | None = None,
    brave_api_key: str | None = None,
) -> SearchClient:
    if kind == "tavily":
        return TavilyProvider(api_key=tavily_api_key)
    if kind == "brave":
        return BraveProvider(api_key=brave_api_key)
    if kind == "ddg":
        return DDGProvider()
    # auto: assemble in preference order, skipping any unavailable.
    providers: list[SearchClient] = []
    for factory in (
        lambda: TavilyProvider(api_key=tavily_api_key),
        lambda: BraveProvider(api_key=brave_api_key),
        lambda: DDGProvider(),
    ):
        try:
            providers.append(factory())
        except SearchUnavailable:
            continue
    return ChainProvider(providers)


def _for_test_chain(providers: list[SearchClient]) -> ChainProvider:
    """Test helper: build a ChainProvider directly from a list of
    already-constructed provider stubs."""
    return ChainProvider(providers)


__all__ = [
    "BraveProvider",
    "ChainProvider",
    "DDGProvider",
    "SearchAllUnavailable",
    "SearchClient",
    "SearchError",
    "SearchHit",
    "SearchRateLimit",
    "SearchUnavailable",
    "TavilyProvider",
    "build_search_client",
]
