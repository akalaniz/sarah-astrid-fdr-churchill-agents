from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

from app.core.config import Settings
from app.tools.news import NewsItem, fetch_news
from app.tools.wiki import WikiResult, fetch_wikipedia


CACHE_TTL = timedelta(minutes=30)

CURRENT_EVENT_TERMS = {
    "today",
    "yesterday",
    "tomorrow",
    "latest",
    "current",
    "recent",
    "now",
    "breaking",
    "news",
    "update",
    "market",
    "markets",
    "price",
    "prices",
    "stock",
    "crypto",
    "election",
    "president",
    "prime minister",
    "ceo",
    "war",
    "sanctions",
    "geopolitics",
    "technology",
    "ai model",
    "release",
}

GEOPOLITICAL_TERMS = {
    "china",
    "russia",
    "ukraine",
    "iran",
    "israel",
    "taiwan",
    "nato",
    "un",
    "iaea",
    "pentagon",
    "white house",
    "state department",
    "missile",
    "nuclear",
    "sanctions",
    "dime",
    "ooda",
}

MODERN_LEADER_TERMS = {
    "trump",
    "biden",
    "putin",
    "zelensky",
    "zelenskyy",
    "xi jinping",
    "netanyahu",
    "khamenei",
    "macron",
    "starmer",
    "modi",
    "erdogan",
    "kim jong un",
    "von der leyen",
}

MARKET_TERMS = {
    "dow",
    "nasdaq",
    "s&p",
    "oil price",
    "gold price",
    "bitcoin",
    "ethereum",
    "treasury yield",
    "inflation",
    "fed rate",
}

TECH_UPDATE_TERMS = {
    "openai",
    "anthropic",
    "google ai",
    "microsoft ai",
    "nvidia",
    "model release",
    "software update",
    "security vulnerability",
    "cve",
}

LOCAL_CANON_TERMS = {
    "sarah",
    "darwin",
    "tak",
    "takayuki",
    "paul",
    "alex",
    "war & peace",
    "mars",
    "universe",
}


@dataclass(frozen=True)
class WebSource:
    title: str
    publisher: str
    date: str
    url: str
    summary: str
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WebRetrievalResult:
    used_web: bool
    failed: bool
    reason: str
    sources: list[WebSource]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "used_web": self.used_web,
            "failed": self.failed,
            "reason": self.reason,
            "sources": [source.to_dict() for source in self.sources],
            "timestamp": self.timestamp,
        }


def retrieve_web_context(query: str, settings: Settings, max_news_items: int = 6) -> WebRetrievalResult:
    decision = decide_web_route(query)
    if decision == "none":
        return WebRetrievalResult(
            used_web=False,
            failed=False,
            reason="Query did not require web/current retrieval.",
            sources=[],
            timestamp=_now(),
        )

    cache_key = _cache_key(query, decision)
    return _cached(settings.web_cache_dir, cache_key, lambda: _fetch_for_route(query, decision, max_news_items))


def decide_web_route(query: str) -> str:
    query_lower = query.lower()
    if (
        _contains_any(query_lower, CURRENT_EVENT_TERMS)
        or _contains_any(query_lower, GEOPOLITICAL_TERMS)
        or _contains_any(query_lower, MODERN_LEADER_TERMS)
        or _contains_any(query_lower, MARKET_TERMS)
        or _contains_any(query_lower, TECH_UPDATE_TERMS)
    ):
        return "news"
    if _contains_any(query_lower, LOCAL_CANON_TERMS):
        return "none"
    if re.search(r"\b(who is|what is|background on|tell me about|explain)\b", query_lower):
        return "wiki"
    if re.search(r"\b(20[2-9][0-9]|19[8-9][0-9])\b", query_lower):
        return "news"
    return "none"


def build_web_context_packet(result: WebRetrievalResult) -> str:
    lines = [
        "WEB/CURRENT-DATA CONTEXT:",
        "Use this only for current or stable public background claims. Cite web/news sources with title, publisher, date, and URL.",
    ]
    if not result.used_web:
        lines.append(f"Status: not used. Reason: {result.reason}")
        return "\n".join(lines)
    if result.failed:
        lines.append("Status: failed. FDR must say the current-data layer failed and separate that from background reasoning.")
        lines.append(f"Failure: {result.reason}")
        return "\n".join(lines)
    if not result.sources:
        lines.append("Status: no sources returned. Treat current evidence as insufficient.")
        return "\n".join(lines)

    lines.append("Status: sources retrieved.")
    for index, source in enumerate(result.sources, start=1):
        lines.extend(
            [
                "",
                f"[{index}] {source.title}",
                f"publisher: {source.publisher}",
                f"date: {source.date}",
                f"url: {source.url}",
                f"summary: {_excerpt(source.summary)}",
            ]
        )
    return "\n".join(lines)


def sources_for_display(result: WebRetrievalResult) -> list[dict[str, Any]]:
    return [source.to_dict() for source in result.sources]


def _fetch_for_route(query: str, decision: str, max_news_items: int) -> WebRetrievalResult:
    try:
        if decision == "news":
            news_items = fetch_news(query, max_items=max_news_items)
            sources = [_source_from_news(item) for item in news_items]
            reason = "News/current retrieval selected."
        elif decision == "wiki":
            wiki_result = fetch_wikipedia(query)
            sources = [_source_from_wiki(wiki_result)] if wiki_result is not None else []
            reason = "Wikipedia stable-background retrieval selected."
        else:
            sources = []
            reason = "No retrieval selected."
        return WebRetrievalResult(
            used_web=decision != "none",
            failed=False,
            reason=reason,
            sources=sources,
            timestamp=_now(),
        )
    except Exception as exc:
        return WebRetrievalResult(
            used_web=True,
            failed=True,
            reason=str(exc),
            sources=[],
            timestamp=_now(),
        )


def _cached(
    cache_dir: Path,
    key: str,
    fetcher: Callable[[], WebRetrievalResult],
) -> WebRetrievalResult:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        timestamp = datetime.fromisoformat(data["timestamp"])
        if datetime.now(timezone.utc) - timestamp <= CACHE_TTL:
            return _result_from_dict(data)

    result = fetcher()
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=True, indent=2), encoding="utf-8")
    return result


def _result_from_dict(data: dict[str, Any]) -> WebRetrievalResult:
    return WebRetrievalResult(
        used_web=bool(data["used_web"]),
        failed=bool(data["failed"]),
        reason=str(data["reason"]),
        sources=[WebSource(**source) for source in data.get("sources", [])],
        timestamp=str(data["timestamp"]),
    )


def _source_from_news(item: NewsItem) -> WebSource:
    return WebSource(
        title=item.title,
        publisher=item.publisher,
        date=item.date,
        url=item.url,
        summary=item.summary,
        kind="news",
    )


def _source_from_wiki(result: WikiResult) -> WebSource:
    sections = ", ".join(result.key_sections)
    summary = result.summary if not sections else f"{result.summary} Key sections: {sections}."
    return WebSource(
        title=result.title,
        publisher=result.publisher,
        date=result.timestamp,
        url=result.url,
        summary=summary,
        kind="wiki",
    )


def _cache_key(query: str, decision: str) -> str:
    digest = hashlib.sha256(f"{decision}:{query.lower()}".encode("utf-8")).hexdigest()
    return digest[:32]


def _contains_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _excerpt(text: str, max_chars: int = 900) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
