"""이미 인덱싱된 청크에 doc_date / doc_type 메타데이터를 채운다.

    python -m scripts.backfill_metadata --dry-run
    python -m scripts.backfill_metadata

벡터는 본문으로 만들어지고 본문은 바뀌지 않으므로 건드리지 않는다. set_payload로 payload만
덮어쓰기 때문에 CPU 임베딩이 다시 돌지 않는다. 노션 순회 한 바퀴(속성 조회) + payload 쓰기가 전부다.

날짜 속성이 새로 생기거나 DOC_TYPE_KEYWORDS를 고친 뒤에도 다시 돌리면 된다.
"""

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from qdrant_client import models
from tqdm import tqdm

from src import config
from src.notion.client import NotionCollector
from src.retrieval import store


def collect_page_metadata() -> dict[str, dict]:
    """노션을 한 바퀴 돌며 page_id -> {doc_date, doc_type, people}를 모은다."""
    collector = NotionCollector()
    metadata: dict[str, dict] = {}
    progress = tqdm(collector.walk(config.NOTION_ROOT_PAGE_ID), desc="노션 순회", unit="페이지")

    for page in progress:
        entry = dict(page.properties)
        entry["doc_type"] = config.classify_doc_type(page.ancestor_path, page.title)
        entry["part"] = config.classify_part(page.ancestor_path, page.title)
        metadata[page.page_id] = entry

    return metadata


def reclassify(dry_run: bool) -> None:
    """DOC_TYPE_KEYWORDS나 PART_KEYWORDS를 고친 뒤 쓴다. 판정에 필요한 값이 payload에 이미 있어 노션이 필요 없다."""
    client = store.get_client()
    changed = 0
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=[store.METADATA_KEY],
            with_vectors=False,
        )
        for point in points:
            existing = point.payload.get(store.METADATA_KEY, {})
            ancestor_path = existing.get("ancestor_path", [])
            title = existing.get("page_title", "")
            fresh = {
                "doc_type": config.classify_doc_type(ancestor_path, title),
                "part": config.classify_part(ancestor_path, title),
            }
            if all(value == existing.get(key, "") for key, value in fresh.items()):
                continue
            changed += 1
            if not dry_run:
                client.set_payload(
                    collection_name=config.QDRANT_COLLECTION,
                    payload=fresh,
                    points=[point.id],
                    key=store.METADATA_KEY,
                    wait=False,
                )
        if offset is None:
            break

    print(f"{'재판정 대상' if dry_run else '재판정 완료'}: {changed}청크")


def main() -> None:
    parser = argparse.ArgumentParser(description="doc_date / doc_type 백필")
    parser.add_argument("--dry-run", action="store_true", help="쓰지 않고 대상만 센다")
    parser.add_argument(
        "--doc-type-only",
        action="store_true",
        help="노션을 돌지 않고 저장된 ancestor_path로 doc_type만 다시 계산한다",
    )
    args = parser.parse_args()

    store.ensure_payload_indexes()

    if args.doc_type_only:
        reclassify(args.dry_run)
        return

    metadata = collect_page_metadata()
    dated = sum(1 for entry in metadata.values() if entry.get("doc_date"))
    typed = sum(1 for entry in metadata.values() if entry.get("doc_type"))
    print()
    print(f"노션 페이지 {len(metadata)}개 — 날짜 보유 {dated}개 / 유형 판정 {typed}개")

    client = store.get_client()
    updated = 0
    skipped = 0
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=[store.METADATA_KEY],
            with_vectors=False,
        )
        for point in points:
            existing = point.payload.get(store.METADATA_KEY, {})
            entry = metadata.get(existing.get("page_id"))
            if not entry:
                skipped += 1
                continue

            if not args.dry_run:
                client.set_payload(
                    collection_name=config.QDRANT_COLLECTION,
                    payload=entry,
                    points=[point.id],
                    key=store.METADATA_KEY,
                    wait=False,
                )
            updated += 1
        if offset is None:
            break

    verb = "갱신 대상" if args.dry_run else "갱신 완료"
    print(f"{verb}: {updated}청크 / 노션에 없어 건너뜀: {skipped}청크")

    if not args.dry_run:
        sample = store.scroll_pages(doc_type="회의록", with_date_only=True)
        print(f"검증 — 날짜를 가진 회의록 페이지: {len(sample)}개")
        for page in sorted(sample, key=lambda p: p["doc_date"], reverse=True)[:5]:
            print(f"  {page['doc_date']}  {page['title']}")


if __name__ == "__main__":
    main()
