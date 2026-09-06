"""페이지 → 청크 → 임베딩 → Qdrant 저장 파이프라인.

임베딩과 upsert는 QdrantVectorStore가 맡는다. 여기서는 페이지를 청크로 바꾸고
64개 단위로 모아 넘기는 일만 한다.
"""

import time
from collections.abc import Iterable
from dataclasses import dataclass

from tqdm import tqdm

from src.indexing.chunker import Chunk, chunk_page
from src.logging_setup import get_logger
from src.notion.client import NotionPage
from src.retrieval import store

logger = get_logger()

FLUSH_THRESHOLD = 64


@dataclass
class IndexStats:
    pages: int = 0
    chunks: int = 0


def _flush(buffer: list[Chunk]) -> None:
    if not buffer:
        return
    store.add_documents(
        [chunk.to_document() for chunk in buffer],
        ids=[chunk.chunk_id for chunk in buffer],
    )
    buffer.clear()


def index_pages(pages: Iterable[NotionPage], replace_existing: bool = False) -> IndexStats:
    stats = IndexStats()
    buffer: list[Chunk] = []
    progress = tqdm(pages, desc="인덱싱", unit="페이지")
    started = time.monotonic()

    for page in progress:
        if replace_existing:
            store.delete_pages([page.page_id])

        chunks = chunk_page(page)
        buffer.extend(chunks)
        stats.pages += 1
        stats.chunks += len(chunks)
        progress.set_postfix(청크=stats.chunks, 문서=page.title[:20])

        if len(buffer) >= FLUSH_THRESHOLD:
            _flush(buffer)
            _log_progress(stats, page, started)

    _flush(buffer)
    logger.info(
        "인덱싱 종료: 페이지 %d / 청크 %d / 경과 %.1f분",
        stats.pages,
        stats.chunks,
        (time.monotonic() - started) / 60,
    )
    return stats


def _log_progress(stats: IndexStats, page: NotionPage, started: float) -> None:
    elapsed = time.monotonic() - started
    logger.info(
        "진행: 페이지 %d / 청크 %d / 경과 %.1f분 / %.2f청크초 / 최근 %r",
        stats.pages,
        stats.chunks,
        elapsed / 60,
        stats.chunks / elapsed if elapsed else 0,
        page.title[:40],
    )
