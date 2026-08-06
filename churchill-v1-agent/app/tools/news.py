from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


USER_AGENT = "churchill-v1-agent/0.1"


@dataclass(frozen=True)
class NewsItem:
    title: str
    publisher: str
    date: str
    url: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RSS_FEEDS = [
    ("AP News", "https://apnews.com/hub/ap-top-news?output=rss"),
    ("Reuters", "https://www.reutersagency.com/feed/?best-topics=political-general&post_type=best"),
    ("BBC", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("NPR", "https://feeds.npr.org/1001/rss.xml"),
    ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
    ("The Guardian", "https://www.theguardian.com/world/rss"),
]

QUERY_STOPWORDS = {
    "latest",
    "current",
    "today",
    "yesterday",
    "tomorrow",
    "news",
    "update",
    "updates",
    "breaking",
}

OFFICIAL_FEEDS = [
    ("White House", "https://www.whitehouse.gov/feed/"),
    ("U.S. State Department", "https://www.state.gov/rss-feed/press-releases/feed/"),
    ("United Nations", "https://news.un.org/feed/subscribe/en/news/all/rss.xml"),
    ("IAEA", "https://www.iaea.org/newscenter/feeds/news"),
    ("NATO", "https://www.nato.int/rss/news.xml"),
    ("U.S. Department of Defense", "https://www.defense.gov/DesktopModules/ArticleCS/RSS.ashx?ContentType=1&Site=945&max=20"),
]


def fetch_news(query: str, max_items: int = 6) -> list[NewsItem]:
    feeds = list(RSS_FEEDS)
    feeds.extend(_official_feeds_for_query(query))
    feeds.append(("Google News", f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"))

    items: list[NewsItem] = []
    successful_feeds = 0
    for publisher, url in feeds:
        try:
            items.extend(_fetch_rss_items(publisher, url))
            successful_feeds += 1
        except Exception:
            continue

    if successful_feeds == 0:
        raise RuntimeError("All configured news/current RSS sources failed.")

    ranked = _rank_items(query, items)
    return ranked[:max_items]


def _official_feeds_for_query(query: str) -> list[tuple[str, str]]:
    query_lower = query.lower()
    keywords = {
        "White House": ["white house", "president", "executive order"],
        "U.S. State Department": ["state department", "secretary of state", "diplomacy", "sanctions"],
        "United Nations": ["united nations", "un ", "security council", "unsc"],
        "IAEA": ["iaea", "nuclear", "uranium", "reactor"],
        "NATO": ["nato", "alliance", "article 5"],
        "U.S. Department of Defense": ["pentagon", "dod", "defense department", "military"],
    }
    selected = []
    for publisher, url in OFFICIAL_FEEDS:
        if any(keyword in query_lower for keyword in keywords[publisher]):
            selected.append((publisher, url))
    return selected


def _fetch_rss_items(publisher: str, url: str) -> list[NewsItem]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=15) as response:
        xml_bytes = response.read()

    root = ET.fromstring(xml_bytes)
    channel = root.find("channel")
    item_nodes = channel.findall("item") if channel is not None else root.findall(".//item")

    items: list[NewsItem] = []
    for item in item_nodes[:25]:
        title = _text(item, "title")
        link = _text(item, "link")
        summary = _strip_html(_text(item, "description"))
        date = _normalize_date(_text(item, "pubDate") or _text(item, "updated"))
        if title and link:
            items.append(
                NewsItem(
                    title=title,
                    publisher=publisher,
                    date=date,
                    url=link,
                    summary=summary,
                )
            )
    return items


def _rank_items(query: str, items: list[NewsItem]) -> list[NewsItem]:
    query_terms = {term for term in _tokens(query) if term not in QUERY_STOPWORDS}
    if not query_terms:
        query_terms = set(_tokens(query))
    topical_items: list[tuple[NewsItem, int]] = []

    for item in items:
        haystack_terms = set(_tokens(f"{item.title} {item.summary}"))
        overlap = len(query_terms & haystack_terms)
        if overlap > 0:
            topical_items.append((item, overlap))

    if topical_items:
        return [
            item
            for item, _overlap in sorted(
                topical_items,
                key=lambda pair: (pair[1], pair[0].date),
                reverse=True,
            )
        ]

    return sorted(items, key=lambda item: item.date, reverse=True)


def _text(node: ET.Element, tag: str) -> str:
    child = node.find(tag)
    return (child.text or "").strip() if child is not None else ""


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(token) > 2]


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def _normalize_date(value: str) -> str:
    if not value:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        return parsedate_to_datetime(value).astimezone(timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return value
