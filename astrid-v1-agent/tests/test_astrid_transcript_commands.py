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
from app.core import transcript_maintenance as tm
from app.ui.web_app import AppState, create_app


class AstridTranscriptCommandTests(unittest.TestCase):
    def test_transcripts_reports_shared_agent_bus_worldline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env(Path(tmpdir), count=3)
            client = TestClient(_test_app(env.root))
            with _patched_transcript_paths(env):
                response = client.post("/api/chat", json={"message": "/transcripts"})
                text = response.json()["text"]
                self.assertIn("Transcript status", text)
                self.assertIn("agent: Astrid v1.0", text)
                self.assertIn("worldline: sarah_astrid", text)
                self.assertIn(r"shared_agent_bus", text)
                self.assertIn("transcripts", text)

    def test_transcripts_keep_keeps_newest_20_conversation_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env(Path(tmpdir), count=25)
            client = TestClient(_test_app(env.root))
            with _patched_transcript_paths(env):
                response = client.post("/api/chat", json={"message": "/transcripts_keep 20"})
                live_ids = set(tm.group_transcripts_by_conversation_id(tm.list_transcript_files(env.transcripts)).keys())
                result = response.json()["result"]
                self.assertEqual(result["kept_conversation_count"], 20)
                self.assertEqual(result["removed_conversation_count"], 5)
                self.assertEqual(len(live_ids), 20)
                self.assertNotIn("conv00", live_ids)
                self.assertIn("conv24", live_ids)
                self.assertTrue((env.transcripts / "latest_multi_agent_transcript.md").exists())
                self.assertTrue((env.transcripts / "latest_multi_agent_transcript.json").exists())

    def test_transcripts_archive_moves_older_files_to_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env(Path(tmpdir), count=4, old_count=2)
            client = TestClient(_test_app(env.root))
            with _patched_transcript_paths(env):
                response = client.post("/api/chat", json={"message": "/transcripts_archive 7"})
                archived = sorted(path.name for path in env.archive.glob("*"))
                live = sorted(path.name for path in env.transcripts.glob("*"))
                result = response.json()["result"]
                self.assertEqual(result["archived_conversation_count"], 2)
                self.assertEqual(result["archived_file_count"], 4)
                self.assertIn("conv00.json", archived)
                self.assertIn("conv01.md", archived)
                self.assertIn("conv02.json", live)
                self.assertIn("latest_multi_agent_transcript.md", live)

    def test_transcripts_prune_preserves_newest_20_live_conversations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env(Path(tmpdir), count=25, old_count=25)
            client = TestClient(_test_app(env.root))
            with _patched_transcript_paths(env):
                response = client.post("/api/chat", json={"message": "/transcripts_prune --days 7 --keep 20"})
                live_ids = set(tm.group_transcripts_by_conversation_id(tm.list_transcript_files(env.transcripts)).keys())
                archived_ids = set(tm.group_transcripts_by_conversation_id(tm.list_transcript_files(env.archive)).keys())
                result = response.json()["result"]
                self.assertEqual(result["keep"], 20)
                self.assertEqual(result["archived_conversation_count"], 5)
                self.assertEqual(result["live_conversation_count_after"], 20)
                self.assertEqual(len(live_ids), 20)
                self.assertEqual(len(archived_ids), 5)
                self.assertIn("conv24", live_ids)
                self.assertIn("conv00", archived_ids)

    def test_clear_and_archive_all_require_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env(Path(tmpdir), count=2)
            client = TestClient(_test_app(env.root))
            with _patched_transcript_paths(env):
                clear_response = client.post("/api/chat", json={"message": "/transcripts_clear"})
                archive_response = client.post("/api/chat", json={"message": "/transcripts_archive_all"})
                live_count = len(list(env.transcripts.glob("*.json"))) + len(list(env.transcripts.glob("*.md")))
                self.assertIn("/transcripts_clear confirm", clear_response.json()["text"])
                self.assertIn("/transcripts_archive_all confirm", archive_response.json()["text"])
                self.assertGreater(live_count, 0)

    def test_confirmed_clear_preserves_latest_and_untouched_state_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env(Path(tmpdir), count=3)
            client = TestClient(_test_app(env.root))
            with _patched_transcript_paths(env):
                response = client.post("/api/chat", json={"message": "/transcripts_clear confirm"})
                live = sorted(path.name for path in env.transcripts.glob("*"))
                result = response.json()["result"]
                self.assertTrue(result["cleared"])
                self.assertEqual(result["deleted_file_count"], 6)
                self.assertEqual(live, ["latest_multi_agent_transcript.json", "latest_multi_agent_transcript.md"])
                self.assertEqual(env.memory_file.read_text(encoding="utf-8"), "memory sentinel")
                self.assertEqual(env.agent_messages.read_text(encoding="utf-8"), "messages sentinel")
                self.assertEqual(env.crew_state.read_text(encoding="utf-8"), "crew sentinel")
                self.assertEqual(env.fdr_bus.read_text(encoding="utf-8"), "fdr/churchill sentinel")
                self.assertEqual(env.sarah_marker.read_text(encoding="utf-8"), "sarah sentinel")
                self.assertEqual(env.churchill_marker.read_text(encoding="utf-8"), "churchill sentinel")

    def test_confirmed_archive_all_moves_normal_pairs_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env(Path(tmpdir), count=2)
            client = TestClient(_test_app(env.root))
            with _patched_transcript_paths(env):
                response = client.post("/api/chat", json={"message": "/transcripts_archive_all confirm"})
                live = sorted(path.name for path in env.transcripts.glob("*"))
                archived = sorted(path.name for path in env.archive.glob("*"))
                result = response.json()["result"]
                self.assertEqual(result["archived_conversation_count"], 2)
                self.assertEqual(result["archived_file_count"], 4)
                self.assertEqual(live, ["latest_multi_agent_transcript.json", "latest_multi_agent_transcript.md"])
                self.assertIn("conv00.json", archived)
                self.assertIn("conv01.md", archived)

    def test_safety_refuses_wrong_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            unsafe = root / "data" / "memory"
            unsafe.mkdir(parents=True)
            result = tm.keep_newest_conversations(unsafe, keep_n=1)
            self.assertIn("Refusing transcript cleanup", result["error"])


class TranscriptEnv:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.shared_bus = root / "shared_agent_bus"
        self.transcripts = self.shared_bus / "transcripts"
        self.archive = self.shared_bus / "transcripts_archive"
        self.agent_messages = self.shared_bus / "agent_messages.jsonl"
        self.crew_state = self.shared_bus / "crew_state.json"
        self.memory_file = root / "astrid-v1-agent" / "data" / "memory" / "astrid_memory.jsonl"
        self.fdr_bus = root / "shared_fdr_churchill_bus" / "agent_messages.jsonl"
        self.sarah_marker = root / "sarah-v2-agent" / "marker.txt"
        self.churchill_marker = root / "churchill-v1-agent" / "marker.txt"


def _make_env(root: Path, count: int, old_count: int = 0) -> TranscriptEnv:
    env = TranscriptEnv(root)
    env.transcripts.mkdir(parents=True)
    env.archive.mkdir(parents=True)
    env.memory_file.parent.mkdir(parents=True)
    env.fdr_bus.parent.mkdir(parents=True)
    env.sarah_marker.parent.mkdir(parents=True)
    env.churchill_marker.parent.mkdir(parents=True)
    env.agent_messages.write_text("messages sentinel", encoding="utf-8")
    env.crew_state.write_text("crew sentinel", encoding="utf-8")
    env.memory_file.write_text("memory sentinel", encoding="utf-8")
    env.fdr_bus.write_text("fdr/churchill sentinel", encoding="utf-8")
    env.sarah_marker.write_text("sarah sentinel", encoding="utf-8")
    env.churchill_marker.write_text("churchill sentinel", encoding="utf-8")
    base = int(time.time())
    old_base = base - 90 * 24 * 60 * 60
    for index in range(count):
        stamp = old_base + index if index < old_count else base + index
        for suffix in (".json", ".md"):
            path = env.transcripts / f"conv{index:02d}{suffix}"
            if suffix == ".json":
                path.write_text(json.dumps({"conversation_id": f"conv{index:02d}"}), encoding="utf-8")
            else:
                path.write_text(f"# conv{index:02d}\n", encoding="utf-8")
            os.utime(path, (stamp, stamp))
    for latest_name in ("latest_multi_agent_transcript.md", "latest_multi_agent_transcript.json"):
        latest = env.transcripts / latest_name
        latest.write_text("latest", encoding="utf-8")
        os.utime(latest, (base + count + 100, base + count + 100))
    return env


def _test_app(root: Path):
    settings = load_settings(env_file=root / ".env")
    memory_file = root / "astrid-v1-agent" / "data" / "memory" / "astrid_memory.jsonl"
    settings = settings.__class__(
        **{
            **settings.__dict__,
            "memory_file": memory_file,
            "conversations_dir": root / "astrid-v1-agent" / "data" / "conversations",
            "web_cache_dir": root / "astrid-v1-agent" / "data" / "web_cache",
            "source_docs_dir": root / "astrid-v1-agent" / "data" / "source_docs",
            "external_source_docs_dir": root / "astrid-v1-agent" / "data" / "source_docs",
            "vector_store_dir": root / "astrid-v1-agent" / "data" / "vector_store",
        }
    )
    return create_app(settings=settings, state=AppState(settings))


@contextmanager
def _patched_transcript_paths(env: TranscriptEnv):
    with (
        patch.object(tm, "SHARED_BUS_DIR", env.shared_bus),
        patch.object(tm, "TRANSCRIPTS_PATH", env.transcripts),
        patch.object(tm, "ARCHIVE_PATH", env.archive),
    ):
        yield


if __name__ == "__main__":
    unittest.main()
