"""노션 변경분만 다시 인덱싱한다.

    python -m scripts.sync

전체 트리를 순회하되(이동·삭제를 정확히 잡기 위해), last_edited_time이 그대로인 페이지는
임베딩을 건너뛴다. 비용의 대부분은 노션 API가 아니라 CPU 임베딩이므로 이 방식이 실질적인 증분이다.
"""

import json
import sys
from datetime import datetime, timezone

# 노션 문서에 콘솔 기본 인코딩(cp949)이 못 찍는 문자가 섞여 있다. 출력만 UTF-8로 돌린다.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import config
from src.indexing.pipeline import index_pages
from src.logging_setup import setup
from src.notion.client import NotionCollector
from src.retrieval import store


def main() -> None:
    logger = setup("sync")
    store.ensure_collection()
    known_versions = store.indexed_page_versions()
    print(f"인덱싱된 페이지: {len(known_versions)}개")

    collector = NotionCollector()
    changed = []
    seen_page_ids = set()

    print("노션 순회 중…")
    for page in collector.walk(config.NOTION_ROOT_PAGE_ID):
        seen_page_ids.add(page.page_id)
        if known_versions.get(page.page_id) != page.last_edited_time:
            changed.append(page)

    removed = sorted(set(known_versions) - seen_page_ids)

    # 수집에 실패한 하위 트리가 있으면 그 안의 페이지도 "못 본 페이지"가 된다.
    # 노션에서 지워진 것과 구분할 수 없으므로, 실패가 있으면 삭제를 통째로 건너뛴다.
    if collector.stats.failed:
        print(f"\n[경고] 수집 실패 {len(collector.stats.failed)}건 — 삭제 단계를 건너뜁니다")
        for page_id, reason in collector.stats.failed:
            print(f"  - {page_id}  {reason}")
        removed = []

    print(f"\n변경/신규: {len(changed)}개, 삭제: {len(removed)}개")
    logger.info("순회 %d페이지, 변경/신규 %d개, 삭제 %d개", len(seen_page_ids), len(changed), len(removed))

    if removed:
        store.delete_pages(removed)
        print("삭제된 페이지의 청크 제거 완료")

    if changed:
        stats = index_pages(changed, replace_existing=True)
        print(f"페이지 {stats.pages}개 / 청크 {stats.chunks}개 재인덱싱")
    else:
        print("변경된 문서가 없습니다.")

    print(f"Qdrant 총 포인트: {store.count_all()}개")

    leaked = store.count_pages(sorted(config.BLOCKED_PAGE_IDS))
    print(f"[{'OK' if leaked == 0 else '경고'}] 블랙리스트 페이지 청크: {leaked}개")
    logger.info("Qdrant 총 %d포인트, 블랙리스트 유출 %d개", store.count_all(), leaked)

    from scripts.verify import run

    print("\n" + "=" * 60)
    print("검색 점검")
    print("=" * 60)
    run(show_all=False)

    config.SYNC_STATE_PATH.write_text(
        json.dumps(
            {
                "synced_at": datetime.now(timezone.utc).isoformat(),
                "pages": len(seen_page_ids),
                "changed": len(changed),
                "removed": len(removed),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
