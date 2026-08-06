from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from app.core.config import load_settings
from app.tools.news import fetch_news
from app.tools.web_router import (
    build_web_context_packet,
    decide_web_route,
    retrieve_web_context,
)
from app.tools.wiki import fetch_wikipedia


class FakeResponse:
    def __init__(self, body: str) -> None:
        self.body = body.encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class WebToolsTests(unittest.TestCase):
    def test_wikipedia_fetch_uses_mediawiki_and_rest_summary(self) -> None:
        def fake_get_json(url: str):
            if "list=search" in url:
                return {"query": {"search": [{"title": "Sarah Nelson"}]}}
            if "prop=sections" in url:
                return {
                    "parse": {
                        "sections": [
                            {"line": "Career"},
                            {"line": "Mars mission"},
                            {"line": "References"},
                        ]
                    }
                }
            return {
                "title": "Sarah Nelson",
                "extract": "Sarah Nelson is a fictional Mars commander.",
                "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Sarah_Nelson"}},
            }

        with patch("app.tools.wiki._get_json", side_effect=fake_get_json):
            result = fetch_wikipedia("Sarah Nelson")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.publisher, "Wikipedia")
        self.assertIn("Mars commander", result.summary)
        self.assertEqual(result.key_sections, ["Career", "Mars mission"])

    def test_news_fetch_parses_mocked_rss_response(self) -> None:
        rss = """<?xml version="1.0"?>
        <rss><channel>
          <item>
            <title>NATO issues new statement</title>
            <link>https://example.com/nato</link>
            <pubDate>Thu, 04 Jun 2026 12:00:00 GMT</pubDate>
            <description>&lt;p&gt;NATO leaders discussed deterrence.&lt;/p&gt;</description>
          </item>
        </channel></rss>
        """

        with patch("app.tools.news.urlopen", return_value=FakeResponse(rss)):
            items = fetch_news("latest NATO deterrence", max_items=1)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].publisher, "AP News")
        self.assertEqual(items[0].title, "NATO issues new statement")
        self.assertIn("deterrence", items[0].summary)

    def test_web_router_decision_rules(self) -> None:
        self.assertEqual(decide_web_route("latest NATO news today"), "news")
        self.assertEqual(decide_web_route("What is Trump doing about NATO?"), "news")
        self.assertEqual(decide_web_route("OpenAI model release notes"), "news")
        self.assertEqual(decide_web_route("Iran tests a nuclear weapon. Give me DIME COAs."), "news")
        self.assertEqual(decide_web_route("Who is Ada Lovelace?"), "wiki")
        self.assertEqual(decide_web_route("explain Sarah's command philosophy"), "none")

    def test_current_events_geopolitics_uses_news_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_settings(env_file=Path(tmp) / ".env")
            settings = settings.__class__(**{**settings.__dict__, "web_cache_dir": Path(tmp) / "cache"})

            with patch("app.tools.web_router.fetch_news") as mocked_fetch:
                mocked_fetch.return_value = [
                    _news_item("Iran nuclear test reported", "Reuters", "2026-06-25T12:00:00+00:00")
                ]
                result = retrieve_web_context("Iran tests a nuclear weapon. Give me DIME COAs.", settings)

        self.assertTrue(result.used_web)
        self.assertFalse(result.failed)
        self.assertEqual(result.sources[0].publisher, "Reuters")
        mocked_fetch.assert_called_once()

    def test_web_router_caches_results_for_30_minutes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            settings = load_settings(env_file=env_file)
            settings = settings.__class__(**{**settings.__dict__, "web_cache_dir": Path(tmp) / "cache"})

            with patch("app.tools.web_router.fetch_news") as mocked_fetch:
                mocked_fetch.return_value = [
                    _news_item("First story", "BBC", "2026-06-04T12:00:00+00:00")
                ]
                first = retrieve_web_context("latest NATO news", settings)
                second = retrieve_web_context("latest NATO news", settings)

        self.assertFalse(first.failed)
        self.assertEqual(first.sources[0].title, "First story")
        self.assertEqual(second.sources[0].publisher, "BBC")
        mocked_fetch.assert_called_once()

    def test_web_router_failure_packet_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_settings(env_file=Path(tmp) / ".env")
            settings = settings.__class__(**{**settings.__dict__, "web_cache_dir": Path(tmp) / "cache"})

            with patch("app.tools.web_router.fetch_news", side_effect=RuntimeError("network down")):
                result = retrieve_web_context("latest NATO news", settings)

        packet = build_web_context_packet(result)
        self.assertTrue(result.failed)
        self.assertIn("current-data layer failed", packet)
        self.assertIn("network down", packet)


def _news_item(title: str, publisher: str, date: str):
    from app.tools.news import NewsItem

    return NewsItem(
        title=title,
        publisher=publisher,
        date=date,
        url="https://example.com/story",
        summary="A useful summary.",
    )


if __name__ == "__main__":
    unittest.main()
