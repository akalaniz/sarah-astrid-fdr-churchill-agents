from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from app.rag.chunker import DocumentChunk
from app.rag.embeddings import LOCAL_EMBEDDING_MODEL
from app.rag import retriever, vector_store as store


class VectorStoreCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "store"
        store.clear_vector_store_cache()
        self.write(self.path, "old")

    def tearDown(self):
        store.clear_vector_store_cache()
        self.temp.cleanup()

    def write(self, path, text):
        chunk = DocumentChunk(text, "test.txt", str(self.root / "test.txt"), "line 1", text, 1)
        store.write_vector_store(path, [chunk], [[1.0, 0.0]], LOCAL_EMBEDDING_MODEL)

    def read(self, path=None):
        return store.read_cached_vector_store(path or self.path)

    def test_warm_reads_parse_once(self):
        with patch.object(store, "read_vector_store", wraps=store.read_vector_store) as read:
            first = self.read()
            for _ in range(5):
                self.assertIs(self.read(), first)
        self.assertEqual(read.call_count, 1)

    def test_raw_reader_remains_uncached_and_independent(self):
        cached = self.read()
        editable = store.read_vector_store(self.path)
        editable[0]["source_files"].append("other.txt")
        editable[1][0].embedding[0] = 0.0
        editable[1].clear()
        self.assertEqual(cached[0]["source_files"], ["test.txt"])
        self.assertEqual(cached[1][0].embedding[0], 1.0)
        self.assertEqual(len(cached[1]), 1)

    def test_writer_invalidates_the_cached_snapshot(self):
        old = self.read()
        self.write(self.path, "new")
        new = self.read()
        self.assertIsNot(old, new)
        self.assertEqual(new[1][0].chunk.text, "new")

    def test_external_chunk_change_refreshes(self):
        self.read()
        chunks_file = self.path / store.CHUNKS_FILE
        entry = json.loads(chunks_file.read_text())
        entry["chunk"]["text"] = "updated outside this process"
        chunks_file.write_text(json.dumps(entry) + "\n", encoding="utf-8")
        self.assertEqual(self.read()[1][0].chunk.text, "updated outside this process")

    def test_external_manifest_change_refreshes(self):
        self.read()
        manifest_file = self.path / store.MANIFEST_FILE
        manifest = json.loads(manifest_file.read_text())
        manifest["embedding_model"] = "replacement-model"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(self.read()[0]["embedding_model"], "replacement-model")

    def test_same_size_and_mtime_replacement_is_detected_by_file_identity(self):
        self.read()
        path = self.path / store.CHUNKS_FILE
        stat = path.stat()
        replacement = self.path / "replacement.jsonl"
        replacement.write_bytes(path.read_bytes().replace(b'"text": "old"', b'"text": "new"'))
        self.assertEqual(replacement.stat().st_size, stat.st_size)
        os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        replacement.replace(path)
        self.assertEqual(self.read()[1][0].chunk.text, "new")

    def test_pdf_directory_activation_refreshes_warm_cache(self):
        from app.ui.web_app import _activate_vector_store
        old = self.read()
        staging = self.root / "staging"
        self.write(staging, "new PDF content")
        _activate_vector_store(staging, self.path)
        new = self.read()
        self.assertIsNot(new, old)
        self.assertEqual(new[1][0].chunk.text, "new PDF content")

    def test_missing_file_never_returns_stale_data(self):
        self.read()
        (self.path / store.CHUNKS_FILE).unlink()
        with self.assertRaises(FileNotFoundError):
            self.read()
        self.assertNotIn(self.path.resolve(), store._store_cache)

    def test_corrupt_index_never_returns_or_caches_stale_data(self):
        self.read()
        (self.path / store.CHUNKS_FILE).write_text("{broken", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            self.read()
        self.assertNotIn(self.path.resolve(), store._store_cache)
        self.write(self.path, "recovered")
        self.assertEqual(self.read()[1][0].chunk.text, "recovered")

    def test_partial_index_count_is_rejected(self):
        (self.path / store.CHUNKS_FILE).write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.read()
        self.assertNotIn(self.path.resolve(), store._store_cache)

    def test_index_changed_during_load_is_retried(self):
        raw_read = store.read_vector_store
        count = 0
        def changing_read(path):
            nonlocal count
            snapshot = raw_read(path)
            count += 1
            if count == 1:
                self.write(path, "changed during read")
            return snapshot
        with patch.object(store, "read_vector_store", side_effect=changing_read):
            self.assertEqual(self.read()[1][0].chunk.text, "changed during read")
        self.assertEqual(count, 2)

    def test_repeated_changes_have_a_bounded_retry(self):
        signatures = [((i,),) for i in range(6)]
        with patch.object(store, "_store_signature", side_effect=signatures):
            with self.assertRaisesRegex(RuntimeError, "changed repeatedly"):
                self.read()
        self.assertFalse(store._store_cache)

    def test_simultaneous_cold_reads_only_parse_once(self):
        raw_read = store.read_vector_store
        def delayed_read(path):
            time.sleep(0.03)
            return raw_read(path)
        with patch.object(store, "read_vector_store", side_effect=delayed_read) as read:
            with ThreadPoolExecutor(max_workers=6) as pool:
                snapshots = list(pool.map(lambda _index: self.read(), range(6)))
        self.assertEqual(read.call_count, 1)
        self.assertTrue(all(value is snapshots[0] for value in snapshots))

    def test_reader_waits_for_in_process_write(self):
        self.read()
        started = threading.Event()
        release = threading.Event()
        raw_write = store._write_vector_store
        def delayed_write(*args):
            started.set()
            if not release.wait(3):
                raise RuntimeError("Test deadline")
            return raw_write(*args)
        with patch.object(store, "_write_vector_store", side_effect=delayed_write):
            with ThreadPoolExecutor(max_workers=2) as pool:
                writer = pool.submit(self.write, self.path, "after write")
                try:
                    self.assertTrue(started.wait(2))
                    reader = pool.submit(self.read)
                    with self.assertRaises(TimeoutError):
                        reader.result(timeout=0.04)
                finally:
                    release.set()
                writer.result(timeout=2)
                self.assertEqual(reader.result(timeout=2)[1][0].chunk.text, "after write")

    def test_cache_is_bounded_and_eviction_is_least_recently_used(self):
        second = self.root / "second"
        third = self.root / "third"
        self.write(second, "second")
        self.write(third, "third")
        self.read()
        self.read(second)
        self.read()
        self.read(third)
        self.assertEqual(len(store._store_cache), 2)
        self.assertNotIn(second.resolve(), store._store_cache)

    def test_paths_are_isolated_and_canonicalized(self):
        other = self.root / "other"
        self.write(other, "other agent")
        first = self.read()
        second = self.read(other)
        self.assertIsNot(first, second)
        self.assertEqual(second[1][0].chunk.text, "other agent")
        alias = self.path / ".." / self.path.name
        self.assertIs(first, self.read(alias))

    def test_explicit_clear_and_empty_indexes(self):
        self.read()
        store.clear_vector_store_cache(self.path)
        self.assertFalse(store._store_cache)
        store.write_vector_store(self.path, [], [], LOCAL_EMBEDDING_MODEL)
        self.assertEqual(self.read()[1], [])

    def test_retrieval_results_and_thresholds_match_uncached_path(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with patch.object(retriever, "embed_texts", return_value=[[1.0, 0.0]]):
                with patch.object(retriever, "read_vector_store", store.read_vector_store):
                    before = retriever.retrieve("query", vector_store_dir=self.path, min_score=0.9)
                cold = retriever.retrieve("query", vector_store_dir=self.path, min_score=0.9)
                warm = retriever.retrieve("query", vector_store_dir=self.path, min_score=0.9)
        self.assertEqual(before, cold)
        self.assertEqual(before, warm)


if __name__ == "__main__":
    unittest.main()

