"""인덱싱 직후 검색이 실제로 되는지 확인한다.

    python -m scripts.verify              샘플 질문으로 검색 점검
    python -m scripts.verify --show-all   실패하지 않은 질문의 결과까지 전부 출력

인덱싱이 끝났다고 바로 챗봇으로 넘어가면, 나중에 답변이 이상할 때 임베딩 문제인지
검색 문제인지 프롬프트 문제인지 구분할 수 없다. 이 스크립트는 그중 **검색까지만** 본다.
LLM을 부르지 않으므로 여기서 통과하면 이후 문제는 프롬프트나 생성 쪽으로 좁혀진다.

질문 목록은 src/eval/smoke_queries.jsonl에 있다. 한 줄 형식:
    {"query": "질문", "expect": "상위 결과의 제목이나 경로에 들어가야 할 문자열"}
`expect`를 비우면 통과/실패 판정 없이 결과만 출력한다.
"""

import argparse
import json
import sys
from pathlib import Path

from src import config
from src.logging_setup import get_logger, setup
from src.retrieval import search, store

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUERIES_PATH = Path(__file__).parent.parent / "src" / "eval" / "smoke_queries.jsonl"
CHECK_TOP_K = 5

# 하이브리드 검색의 RRF 점수는 1등이 약 0.75다. 최상위가 이보다 한참 낮으면
# 질문과 맞는 문서가 사실상 없다는 뜻이라 따로 표시한다.
WEAK_SCORE = 0.30


def load_queries() -> list[dict]:
    if not QUERIES_PATH.exists():
        raise RuntimeError(f"샘플 질문 파일이 없습니다: {QUERIES_PATH}")
    lines = QUERIES_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def check_index() -> bool:
    """검색 이전 단계의 문제(빈 컬렉션, 차단 페이지 유출)를 먼저 걸러낸다."""
    logger = get_logger()
    client = store.get_client()

    if not client.collection_exists(config.QDRANT_COLLECTION):
        print(f"[실패] 컬렉션이 없습니다: {config.QDRANT_COLLECTION}")
        return False

    total = store.count_all()
    pages = len(store.indexed_page_versions())
    print(f"컬렉션 {config.QDRANT_COLLECTION}: {pages}페이지 / {total}청크")
    logger.info("검증 대상: %d페이지 / %d청크", pages, total)

    if total == 0:
        print("[실패] 청크가 하나도 없습니다. 인덱싱이 돌지 않았습니다.")
        return False

    leaked = store.count_pages(sorted(config.BLOCKED_PAGE_IDS))
    if leaked:
        print(f"[실패] 블랙리스트 페이지 청크가 {leaked}개 들어 있습니다.")
        logger.warning("블랙리스트 유출 %d개", leaked)
        return False

    return True


def run(show_all: bool) -> int:
    logger = get_logger()
    queries = load_queries()
    print(f"\n샘플 질문 {len(queries)}개로 검색 점검 (상위 {CHECK_TOP_K}개 확인)\n")

    passed = failed = weak = 0

    for item in queries:
        query = item["query"]
        expect = item.get("expect", "")
        results = search.search(query, top_k=CHECK_TOP_K)

        if not results:
            print(f"[실패] {query}\n        검색 결과가 0개입니다.")
            logger.warning("검색 0건: %s", query)
            failed += 1
            continue

        hit = None
        if expect:
            lowered = expect.lower()
            hit = next(
                (
                    index
                    for index, result in enumerate(results, start=1)
                    if lowered in result.location.lower()
                ),
                None,
            )

        top_score = results[0].score
        is_weak = top_score < WEAK_SCORE

        if expect and hit is None:
            status, mark = "실패", "[실패]"
            failed += 1
        elif is_weak:
            status, mark = "약함", "[약함]"
            weak += 1
        else:
            status, mark = "통과", "[통과]"
            passed += 1

        rank = f"{hit}위" if hit else ("판정없음" if not expect else "미검출")
        print(f"{mark} {query}   ({rank}, 최고점 {top_score:.3f})")
        logger.info("%s | %s | %s | 최고점 %.3f", status, query, rank, top_score)

        if show_all or status != "통과":
            for index, result in enumerate(results[:3], start=1):
                print(f"        {index}. ({result.score:.3f}) {result.location[:90]}")

    print(f"\n통과 {passed} / 약함 {weak} / 실패 {failed}   (전체 {len(queries)})")
    logger.info("검증 결과: 통과 %d, 약함 %d, 실패 %d", passed, weak, failed)

    if failed:
        print("\n검색 단계에서 문제가 확인됐습니다. 챗봇 답변 품질을 보기 전에 여기부터 해결하세요.")
        print("  - 결과가 0개면: 인덱싱 누락이거나 컬렉션이 비어 있음")
        print("  - 엉뚱한 문서가 올라오면: 청킹 파라미터나 컨텍스트 헤더를 점검")
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="인덱싱 후 검색 점검")
    parser.add_argument("--show-all", action="store_true", help="통과한 질문의 결과도 출력한다")
    args = parser.parse_args()

    setup("verify")
    if not check_index():
        sys.exit(1)
    sys.exit(1 if run(args.show_all) else 0)


if __name__ == "__main__":
    main()
