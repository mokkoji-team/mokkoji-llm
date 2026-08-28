"""회의록 문서의 LlamaIndex 인덱스 생성·저장을 담당한다."""

from __future__ import annotations

import shutil
from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
STORAGE_DIR = PROJECT_DIR / "storage"
OLLAMA_URL = "http://localhost:11434"


def configure_models() -> None:
    """검색과 답변에서 공통으로 사용할 Ollama 모델을 설정한다."""
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    Settings.llm = Ollama(
        model="llama3.1:8b",
        base_url=OLLAMA_URL,
        request_timeout=300.0,
        context_window=4096,
        system_prompt=(
            "너는 한국어로 답하는 회의록 도우미다. "
            "주어진 문맥만 근거로 답하고, 근거가 없으면 "
            "'문서에서 근거를 찾지 못했습니다'라고 답한다."
        ),
    )
    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url=OLLAMA_URL,
    )


def load_documents(data_dir: Path = DATA_DIR):
    if not data_dir.exists():
        raise FileNotFoundError(f"문서 폴더를 찾을 수 없습니다: {data_dir}")
    documents = SimpleDirectoryReader(str(data_dir), recursive=True).load_data()
    if not documents:
        raise ValueError(f"{data_dir}에 읽을 문서가 없습니다.")
    return documents


def build_index(data_dir: Path = DATA_DIR):
    configure_models()
    return VectorStoreIndex.from_documents(load_documents(data_dir))


def rebuild_persisted_index(
    data_dir: Path = DATA_DIR, storage_dir: Path = STORAGE_DIR
) -> None:
    """새 인덱스를 완성한 후 기존 저장소와 원자적으로 교체한다."""
    building_dir = storage_dir.with_name(f"{storage_dir.name}.building")
    backup_dir = storage_dir.with_name(f"{storage_dir.name}.backup")

    if building_dir.exists():
        shutil.rmtree(building_dir)
    building_dir.mkdir(parents=True)

    try:
        index = build_index(data_dir)
        index.storage_context.persist(persist_dir=str(building_dir))

        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if storage_dir.exists():
            storage_dir.rename(backup_dir)
        building_dir.rename(storage_dir)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    except Exception:
        if building_dir.exists():
            shutil.rmtree(building_dir)
        if not storage_dir.exists() and backup_dir.exists():
            backup_dir.rename(storage_dir)
        raise


def load_persisted_index(storage_dir: Path = STORAGE_DIR):
    configure_models()
    if not storage_dir.exists():
        raise FileNotFoundError(
            f"저장된 인덱스가 없습니다: {storage_dir}. "
            "먼저 rebuild_index.py를 실행하세요."
        )
    context = StorageContext.from_defaults(persist_dir=str(storage_dir))
    return load_index_from_storage(context)
