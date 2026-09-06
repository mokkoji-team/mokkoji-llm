"""청킹 규칙이 바뀐 페이지만 다시 임베딩한다.

    python -m scripts.rechunk --dry-run
    python -m scripts.rechunk

노션은 그대로인데 우리 청킹 코드만 바뀐 경우를 위한 스크립트다. `scripts.sync`는
last_edited_time으로 판단하므로 이 경우를 통째로 건너뛴다.

노션 API를 쓰지 않는다. 페이지 메타데이터는 Qdrant payload에, 본문은 `.cache/pages.json`에
이미 있어서 둘을 합치면 NotionPage를 되살릴 수 있다. 그래서 비용은 바뀐 페이지의 임베딩뿐이다.

payload만 바뀌었다면 이 스크립트가 아니라 `scripts.backfill_metadata`를 쓴다. 그쪽은
벡터를 건드리지 않아 훨씬 싸다.
"""

import argparse
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import config
from src.indexing.chunker import chunk_page
from src.indexing.pipeline import index_pages
from src.notion.client import NotionPage
from src.retrieval import store

# 노션 속성에서 온 값들. NotionPage.properties로 되돌려야 청크 메타데이터가 그대로 유지된다.
PROPERTY_KEYS = ("doc_date", "people")


def load_cached_markdown() -> dict[str, str]:
    path = config.CACHE_DIR / "pages.json"
    if not path.exists():
        raise RuntimeError(f"페이지 캐시가 없습니다: {path}. scripts.sync를 먼저 돌리세요.")
    cache = json.loads(path.read_text(encoding="utf-8"))
    return {page_id: entry["markdown"] for page_id, entry in cache.items()}


def load_indexed_pages() -> dict[str, dict]:
    """Qdrant에 들어 있는 페이지를 page_id -> {메타데이터, 청크 본문 목록}으로 모은다."""
    client = store.get_client()
    pages: dict[str, dict] = {}
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            metadata = payload.get(store.METADATA_KEY, {})
            page_id = metadata.get("page_id")
            if not page_id:
                continue
            entry = pages.setdefault(page_id, {"metadata": metadata, "texts": []})
            # scroll은 순서를 보장하지 않는다. 비교하려면 chunk_index로 되돌려야 한다.
            entry["texts"].append((metadata.get("chunk_index", 0), payload.get(store.CONTENT_KEY, "")))
        if offset is None:
            break

    for entry in pages.values():
        entry["texts"] = [text for _, text in sorted(entry["texts"])]

    return pages


def to_notion_page(metadata: dict, markdown: str) -> NotionPage:
    return NotionPage(
        page_id=metadata["page_id"],
        title=metadata.get("page_title", ""),
        ancestor_path=metadata.get("ancestor_path", []),
        url=metadata.get("url", ""),
        last_edited_time=metadata.get("last_edited_time", ""),
        markdown=markdown,
        properties={key: metadata[key] for key in PROPERTY_KEYS if key in metadata},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="청킹 결과가 바뀐 페이지만 재임베딩")
    parser.add_argument("--dry-run", action="store_true", help="쓰지 않고 대상만 센다")
    args = parser.parse_args()

    store.ensure_payload_indexes()

    markdowns = load_cached_markdown()
    indexed = load_indexed_pages()
    print(f"인덱싱된 페이지 {len(indexed)}개 / 캐시된 본문 {len(markdowns)}개")

    stale: list[NotionPage] = []
    missing = 0
    before = 0
    after = 0

    for page_id, entry in indexed.items():
        markdown = markdowns.get(page_id)
        if markdown is None:
            missing += 1
            continue

        page = to_notion_page(entry["metadata"], markdown)
        fresh = [chunk.text for chunk in chunk_page(page)]
        if fresh == entry["texts"]:
            continue

        stale.append(page)
        before += len(entry["texts"])
        after += len(fresh)

    if missing:
        print(f"캐시에 본문이 없어 건너뜀: {missing}페이지")

    print(f"청킹이 달라진 페이지: {len(stale)}개  (청크 {before}개 → {after}개)")
    if not stale or args.dry_run:
        return

    stats = index_pages(stale, replace_existing=True)
    print(f"재인덱싱 완료: 페이지 {stats.pages} / 청크 {stats.chunks}")


if __name__ == "__main__":
    main()
