import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def install_llama_stubs():
    core = types.ModuleType("llama_index.core")
    core.Settings = types.SimpleNamespace()
    core.SimpleDirectoryReader = object
    core.StorageContext = object
    core.VectorStoreIndex = object
    core.load_index_from_storage = lambda context: context

    embeddings = types.ModuleType("llama_index.embeddings.ollama")
    embeddings.OllamaEmbedding = object
    llms = types.ModuleType("llama_index.llms.ollama")
    llms.Ollama = object

    sys.modules.setdefault("llama_index", types.ModuleType("llama_index"))
    sys.modules.setdefault("llama_index.core", core)
    sys.modules.setdefault("llama_index.embeddings", types.ModuleType("embeddings"))
    sys.modules.setdefault("llama_index.embeddings.ollama", embeddings)
    sys.modules.setdefault("llama_index.llms", types.ModuleType("llms"))
    sys.modules.setdefault("llama_index.llms.ollama", llms)


install_llama_stubs()

from indexing import rebuild_persisted_index


class FakeStorageContext:
    def persist(self, persist_dir):
        Path(persist_dir, "index_store.json").write_text("new", encoding="utf-8")


class FakeIndex:
    storage_context = FakeStorageContext()


class PersistedIndexTest(unittest.TestCase):
    def test_replaces_storage_only_after_new_index_is_persisted(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            storage = root / "storage"
            storage.mkdir()
            (storage / "index_store.json").write_text("old", encoding="utf-8")

            with patch("indexing.build_index", return_value=FakeIndex()):
                rebuild_persisted_index(root / "data", storage)

            self.assertEqual(
                (storage / "index_store.json").read_text(encoding="utf-8"), "new"
            )
            self.assertFalse((root / "storage.building").exists())
            self.assertFalse((root / "storage.backup").exists())

    def test_keeps_existing_storage_when_build_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            storage = root / "storage"
            storage.mkdir()
            (storage / "index_store.json").write_text("old", encoding="utf-8")

            with patch("indexing.build_index", side_effect=RuntimeError("failed")):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    rebuild_persisted_index(root / "data", storage)

            self.assertEqual(
                (storage / "index_store.json").read_text(encoding="utf-8"), "old"
            )
            self.assertFalse((root / "storage.building").exists())


if __name__ == "__main__":
    unittest.main()
