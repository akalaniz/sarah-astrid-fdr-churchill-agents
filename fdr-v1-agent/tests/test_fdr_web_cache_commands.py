from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.config import load_settings
from app.core import web_cache_maintenance as wc
from app.ui.web_app import AppState, create_app


class FDRWebCacheCommandTests(unittest.TestCase):
    def test_web_cache_reports_fdr_cache_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env(Path(tmpdir), count=3)
            client = TestClient(_test_app(env.root))
            with _patched_cache_paths(env):
                response = client.post("/api/chat", json={"message": "/web_cache"})
                text = response.json()["text"]
                self.assertIn("Web cache status", text)
                self.assertIn("agent: FDR v1.0", text)
                self.assertIn(r"fdr-v1-agent", text)
                self.assertIn(r"data", text)
                self.assertIn(r"web_cache", text)

    def test_web_cache_validate_finds_corrupt_json_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env(Path(tmpdir), count=2, corrupt_count=2)
            client = TestClient(_test_app(env.root))
            with _patched_cache_paths(env):
                response = client.post("/api/chat", json={"message": "/web_cache_validate"})
                result = response.json()["result"]
                self.assertEqual(result["json_count"], 4)
                self.assertEqual(result["valid_json_count"], 2)
                self.assertEqual(result["corrupt_json_count"], 2)
                self.assertTrue((env.cache / "corrupt00.json").exists())
                self.assertTrue((env.cache / "corrupt01.json").exists())

    def test_web_cache_keep_keeps_newest_200_json_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env(Path(tmpdir), count=205)
            client = TestClient(_test_app(env.root))
            with _patched_cache_paths(env):
                response = client.post("/api/chat", json={"message": "/web_cache_keep 200"})
                live = sorted(path.name for path in wc.list_web_cache_files(env.cache))
                result = response.json()["result"]
                self.assertEqual(result["kept_file_count"], 200)
                self.assertEqual(result["deleted_file_count"], 5)
                self.assertEqual(len(live), 200)
                self.assertNotIn("cache000.json", live)
                self.assertIn("cache204.json", live)
                self.assertTrue((env.cache / "notes.txt").exists())

    def test_web_cache_archive_moves_older_json_to_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env(Path(tmpdir), count=4, old_count=2)
            client = TestClient(_test_app(env.root))
            with _patched_cache_paths(env):
                response = client.post("/api/chat", json={"message": "/web_cache_archive 14"})
                archived = sorted(path.name for path in env.archive.glob("*.json"))
                live = sorted(path.name for path in env.cache.glob("*.json"))
                result = response.json()["result"]
                self.assertEqual(result["archived_file_count"], 2)
                self.assertIn("cache000.json", archived)
                self.assertIn("cache001.json", archived)
                self.assertIn("cache002.json", live)

    def test_web_cache_prune_preserves_newest_200_live_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env(Path(tmpdir), count=205, old_count=205)
            client = TestClient(_test_app(env.root))
            with _patched_cache_paths(env):
                response = client.post("/api/chat", json={"message": "/web_cache_prune --days 14 --keep 200"})
                live = sorted(path.name for path in wc.list_web_cache_files(env.cache))
                archived = sorted(path.name for path in wc.list_web_cache_files(env.archive))
                result = response.json()["result"]
                self.assertEqual(result["keep"], 200)
                self.assertEqual(result["archived_file_count"], 5)
                self.assertEqual(result["live_file_count_after"], 200)
                self.assertEqual(len(live), 200)
                self.assertEqual(len(archived), 5)
                self.assertIn("cache204.json", live)
                self.assertIn("cache000.json", archived)

    def test_dangerous_commands_require_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env(Path(tmpdir), count=2, corrupt_count=1)
            client = TestClient(_test_app(env.root))
            with _patched_cache_paths(env):
                clear_response = client.post("/api/chat", json={"message": "/web_cache_clear"})
                archive_response = client.post("/api/chat", json={"message": "/web_cache_archive_all"})
                corrupt_response = client.post("/api/chat", json={"message": "/web_cache_delete_corrupt"})
                live_count = len(list(env.cache.glob("*.json")))
                self.assertIn("/web_cache_clear confirm", clear_response.json()["text"])
                self.assertIn("/web_cache_archive_all confirm", archive_response.json()["text"])
                self.assertIn("/web_cache_delete_corrupt confirm", corrupt_response.json()["text"])
                self.assertEqual(live_count, 3)

    def test_delete_corrupt_confirm_deletes_only_corrupt_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env(Path(tmpdir), count=2, corrupt_count=2)
            client = TestClient(_test_app(env.root))
            with _patched_cache_paths(env):
                response = client.post("/api/chat", json={"message": "/web_cache_delete_corrupt confirm"})
                result = response.json()["result"]
                self.assertEqual(result["deleted_corrupt_count"], 2)
                self.assertTrue((env.cache / "cache000.json").exists())
                self.assertTrue((env.cache / "cache001.json").exists())
                self.assertFalse((env.cache / "corrupt00.json").exists())
                self.assertFalse((env.cache / "corrupt01.json").exists())

    def test_confirmed_clear_preserves_non_cache_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env(Path(tmpdir), count=3, corrupt_count=1)
            client = TestClient(_test_app(env.root))
            with _patched_cache_paths(env):
                response = client.post("/api/chat", json={"message": "/web_cache_clear confirm"})
                result = response.json()["result"]
                self.assertTrue(result["cleared"])
                self.assertEqual(result["deleted_file_count"], 4)
                self.assertEqual(sorted(path.name for path in env.cache.glob("*")), ["notes.txt"])
                self.assertEqual(env.memory_file.read_text(encoding="utf-8"), "memory sentinel")
                self.assertEqual((env.source_docs / "source.txt").read_text(encoding="utf-8"), "source sentinel")
                self.assertEqual((env.vector_store / "vector.bin").read_text(encoding="utf-8"), "vector sentinel")
                self.assertEqual(env.historical_bus.read_text(encoding="utf-8"), "historical bus sentinel")
                self.assertEqual(env.shared_agent_bus.read_text(encoding="utf-8"), "agent bus sentinel")
                self.assertEqual(env.sarah_marker.read_text(encoding="utf-8"), "sarah sentinel")
                self.assertEqual(env.astrid_marker.read_text(encoding="utf-8"), "astrid sentinel")
                self.assertEqual(env.churchill_marker.read_text(encoding="utf-8"), "churchill sentinel")

    def test_archive_all_confirm_moves_json_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env(Path(tmpdir), count=2, corrupt_count=1)
            client = TestClient(_test_app(env.root))
            with _patched_cache_paths(env):
                response = client.post("/api/chat", json={"message": "/web_cache_archive_all confirm"})
                result = response.json()["result"]
                self.assertEqual(result["archived_file_count"], 3)
                self.assertEqual(sorted(path.name for path in env.cache.glob("*")), ["notes.txt"])
                self.assertIn("cache000.json", sorted(path.name for path in env.archive.glob("*.json")))
                self.assertIn("corrupt00.json", sorted(path.name for path in env.archive.glob("*.json")))

    def test_safety_refuses_wrong_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            unsafe = root / "fdr-v1-agent" / "data" / "memory"
            unsafe.mkdir(parents=True)
            with patch.object(wc, "PROJECT_ROOT", root / "fdr-v1-agent"):
                result = wc.keep_newest_cache_files(unsafe, keep_n=1)
            self.assertIn("Refusing web-cache cleanup", result["error"])


class CacheEnv:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.project = root / "fdr-v1-agent"
        self.data = self.project / "data"
        self.cache = self.data / "web_cache"
        self.archive = self.data / "web_cache_archive"
        self.memory_file = self.data / "memory" / "fdr_memory.jsonl"
        self.source_docs = self.data / "source_docs"
        self.vector_store = self.data / "vector_store"
        self.historical_bus = root / "shared_fdr_churchill_bus" / "agent_messages.jsonl"
        self.shared_agent_bus = root / "shared_agent_bus" / "agent_messages.jsonl"
        self.sarah_marker = root / "sarah-v2-agent" / "marker.txt"
        self.astrid_marker = root / "astrid-v1-agent" / "marker.txt"
        self.churchill_marker = root / "churchill-v1-agent" / "marker.txt"


def _make_env(root: Path, count: int, old_count: int = 0, corrupt_count: int = 0) -> CacheEnv:
    env = CacheEnv(root)
    env.cache.mkdir(parents=True)
    env.archive.mkdir(parents=True)
    env.memory_file.parent.mkdir(parents=True)
    env.source_docs.mkdir(parents=True)
    env.vector_store.mkdir(parents=True)
    env.historical_bus.parent.mkdir(parents=True)
    env.shared_agent_bus.parent.mkdir(parents=True)
    env.sarah_marker.parent.mkdir(parents=True)
    env.astrid_marker.parent.mkdir(parents=True)
    env.churchill_marker.parent.mkdir(parents=True)
    env.memory_file.write_text("memory sentinel", encoding="utf-8")
    (env.source_docs / "source.txt").write_text("source sentinel", encoding="utf-8")
    (env.vector_store / "vector.bin").write_text("vector sentinel", encoding="utf-8")
    env.historical_bus.write_text("historical bus sentinel", encoding="utf-8")
    env.shared_agent_bus.write_text("agent bus sentinel", encoding="utf-8")
    env.sarah_marker.write_text("sarah sentinel", encoding="utf-8")
    env.astrid_marker.write_text("astrid sentinel", encoding="utf-8")
    env.churchill_marker.write_text("churchill sentinel", encoding="utf-8")
    (env.cache / "notes.txt").write_text("not json", encoding="utf-8")
    base = int(time.time())
    old_base = base - 90 * 24 * 60 * 60
    for index in range(count):
        stamp = old_base + index if index < old_count else base + index
        path = env.cache / f"cache{index:03d}.json"
        path.write_text(json.dumps({"index": index}), encoding="utf-8")
        os.utime(path, (stamp, stamp))
    for index in range(corrupt_count):
        path = env.cache / f"corrupt{index:02d}.json"
        path.write_text("{not valid json", encoding="utf-8")
        os.utime(path, (base + count + index + 10, base + count + index + 10))
    return env


def _test_app(root: Path):
    settings = load_settings(env_file=root / ".env")
    project = root / "fdr-v1-agent"
    settings = settings.__class__(
        **{
            **settings.__dict__,
            "project_root": project,
            "data_dir": project / "data",
            "memory_file": project / "data" / "memory" / "fdr_memory.jsonl",
            "conversations_dir": project / "data" / "conversations",
            "web_cache_dir": project / "data" / "web_cache",
            "source_docs_dir": project / "data" / "source_docs",
            "external_source_docs_dir": project / "data" / "source_docs",
            "vector_store_dir": project / "data" / "vector_store",
        }
    )
    return create_app(settings=settings, state=AppState(settings))


@contextmanager
def _patched_cache_paths(env: CacheEnv):
    with (
        patch.object(wc, "PROJECT_ROOT", env.project),
        patch.object(wc, "CACHE_PATH", env.cache),
        patch.object(wc, "ARCHIVE_PATH", env.archive),
    ):
        yield


if __name__ == "__main__":
    unittest.main()
