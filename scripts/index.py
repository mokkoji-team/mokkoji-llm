"""노션 워크스페이스 전체를 인덱싱한다.

    python -m scripts.index --dry-run       수집 대상만 확인 (Qdrant 불필요)
    python -m scripts.index                 인덱싱 실행
    python -m scripts.index --recreate      컬렉션을 지우고 처음부터
"""

import argparse
import sys
from itertools import islice

# 노션 문서에 콘솔 기본 인코딩(cp949)이 못 찍는 문자가 섞여 있다. 출력만 UTF-8로 돌린다.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import config
from src.logging_setup import get_logger, setup
from src.notion.client import NotionCollector


def main() -> None:
    parser = argparse.ArgumentParser(description="노션 문서 인덱싱")
    parser.add_argument("--dry-run", action="store_true", help="수집 대상만 출력하고 저장하지 않는다")
    parser.add_argument("--recreate", action="store_true", help="컬렉션을 삭제하고 새로 만든다")
    parser.add_argument("--no-cache", action="store_true", help="로컬 페이지 캐시를 쓰지 않는다")
    parser.add_argument("--with-chunks", action="store_true", help="dry-run에서 청크 수까지 계산한다")
    parser.add_argument("--limit", type=int, default=0, help="이 개수만큼만 처리 (디버그용)")
    parser.add_argument("--no-verify", action="store_true", help="끝나고 검색 점검을 건너뛴다")
    args = parser.parse_args()

    setup("index-dry" if args.dry_run else "index")

    collector = NotionCollector(use_cache=not args.no_cache)
    pages = collector.walk(config.NOTION_ROOT_PAGE_ID)

    if args.limit:
        pages = islice(pages, args.limit)

    if args.dry_run:
        run_dry(collector, pages, count_chunks=args.with_chunks)
        return

    from src.indexing.pipeline import index_pages
    from src.retrieval import store

    store.ensure_collection(recreate=args.recreate)
    # 비어 있는 컬렉션이면 페이지마다 삭제를 시도할 이유가 없다.
    stats = index_pages(pages, replace_existing=store.count_all() > 0)

    print(f"\n페이지 {stats.pages}개 / 청크 {stats.chunks}개 인덱싱 완료")
    print(f"Qdrant 총 포인트: {store.count_all()}개")
    get_logger().info("Qdrant 총 포인트: %d개", store.count_all())
    report_skipped(collector)
    report_failed(collector)
    verify_blocklist(store)

    if not args.no_verify:
        run_search_check()


def run_search_check() -> None:
    """저장이 끝났다고 바로 넘어가지 않는다. 검색이 실제로 되는지 여기서 확인한다."""
    from scripts.verify import run

    print("\n" + "=" * 60)
    print("검색 점검")
    print("=" * 60)
    if run(show_all=False):
        print("\n[경고] 검색 점검에서 실패한 질문이 있습니다.")


def run_dry(collector: NotionCollector, pages, count_chunks: bool) -> None:
    total_chunks = 0
    total_characters = 0

    for page in pages:
        indent = "  " * len(page.ancestor_path)
        suffix = f"  ({len(page.markdown)}자)"
        if count_chunks:
            from src.indexing.chunker import chunk_page

            chunk_count = len(chunk_page(page))
            total_chunks += chunk_count
            suffix += f"  청크 {chunk_count}개"
        total_characters += len(page.markdown)
        print(f"{indent}- {page.title}{suffix}")

    print(f"\n수집 페이지: {collector.stats.visited}개 (캐시 적중 {collector.stats.cache_hits}개)")
    print(f"본문 총 길이: {total_characters:,}자")
    if count_chunks:
        print(f"예상 청크 수: {total_chunks:,}개")
    report_skipped(collector)
    report_failed(collector)


def report_failed(collector: NotionCollector) -> None:
    if not collector.stats.failed:
        return
    print(f"\n[경고] 수집 실패 {len(collector.stats.failed)}개 — 다시 실행하면 재시도된다:")
    for page_id, reason in collector.stats.failed:
        print(f"  - {page_id}  {reason}")
        get_logger().warning("수집 실패: %s  %s", page_id, reason)


def report_skipped(collector: NotionCollector) -> None:
    if not collector.stats.skipped:
        print("\n제외된 페이지 없음")
        return
    print(f"\n제외된 페이지 {len(collector.stats.skipped)}개:")
    for page_id, title, reason in collector.stats.skipped:
        print(f"  - {title}  [{reason}]  {page_id}")


def verify_blocklist(store) -> None:
    leaked = store.count_pages(sorted(config.BLOCKED_PAGE_IDS))
    status = "OK" if leaked == 0 else "경고"
    print(f"\n[{status}] 블랙리스트 페이지 청크: {leaked}개")
    get_logger().info("블랙리스트 유출 검사: %d개 (%s)", leaked, status)


if __name__ == "__main__":
    main()
