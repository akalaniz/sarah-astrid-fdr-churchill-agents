from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


USER_AGENT = "churchill-v1-agent/0.1"
MEDIAWIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_REST_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


@dataclass(frozen=True)
class WikiResult:
    title: str
    summary: str
    key_sections: list[str]
    url: str
    timestamp: str
    publisher: str = "Wikipedia"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fetch_wikipedia(query: str, max_sections: int = 4) -> WikiResult | None:
    title = _search_title(query)
    if not title:
        return None

    summary_data = _get_json(WIKI_REST_SUMMARY_URL.format(title=quote(title.replace(" ", "_"))))
    sections = _fetch_key_sections(title, max_sections=max_sections)
    return WikiResult(
        title=str(summary_data.get("title") or title),
        summary=str(summary_data.get("extract") or ""),
        key_sections=sections,
        url=str(summary_data.get("content_urls", {}).get("desktop", {}).get("page") or f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _search_title(query: str) -> str | None:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
        "format": "json",
        "origin": "*",
    }
    data = _get_json(f"{MEDIAWIKI_API_URL}?{urlencode(params)}")
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return None
    return str(hits[0].get("title") or "")


def _fetch_key_sections(title: str, max_sections: int) -> list[str]:
    params = {
        "action": "parse",
        "page": title,
        "prop": "sections",
        "format": "json",
        "origin": "*",
    }
    data = _get_json(f"{MEDIAWIKI_API_URL}?{urlencode(params)}")
    sections = data.get("parse", {}).get("sections", [])
    names: list[str] = []
    for section in sections:
        line = str(section.get("line") or "").strip()
        if line and not line.lower().startswith(("references", "external links", "see also")):
            names.append(line)
        if len(names) >= max_sections:
            break
    return names


def _get_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))
