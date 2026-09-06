"""CLI로 질문해서 검색 결과와 답변을 확인한다.

    python -m scripts.ask "프론트엔드 기술 스택을 왜 이렇게 골랐어?"
    python -m scripts.ask                      대화형 모드
    python -m scripts.ask --retrieval-only "..."   검색 결과만 확인
"""

import argparse
import sys
import time

# 노션 문서에 콘솔 기본 인코딩(cp949)이 못 찍는 문자가 섞여 있다. 출력만 UTF-8로 돌린다.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.generation import llm, prompt
from src.retrieval import listing, search


def show_results(results: list[search.SearchResult]) -> None:
    print(f"\n검색 결과 {len(results)}개")
    for number, result in enumerate(results, start=1):
        preview = result.body.replace("\n", " ")[:90]
        print(f"  [{number}] ({result.score:.4f}) {result.location}")
        print(f"      {preview}")


def ask(question: str, retrieval_only: bool, verbose: bool) -> None:
    # "저번달 회의 목록" 류는 유사도 상위 k개가 아니라 조건에 맞는 전부가 필요하다.
    listed = listing.answer(question)
    if listed and not retrieval_only:
        print(listed)
        return

    started = time.monotonic()
    results = search.search(question)
    search_seconds = time.monotonic() - started

    if not results:
        print(prompt.NO_RESULT_MESSAGE)
        return

    if retrieval_only or verbose:
        show_results(results)
    print(f"\n(검색 {search_seconds:.1f}초)")

    if retrieval_only:
        return

    print("\n답변:")
    generation_started = time.monotonic()
    token_count = 0
    for piece in llm.stream_answer(prompt.build_messages(question, results)):
        sys.stdout.write(piece)
        sys.stdout.flush()
        token_count += 1

    generation_seconds = time.monotonic() - generation_started
    speed = token_count / generation_seconds if generation_seconds else 0
    print(f"\n\n(생성 {generation_seconds:.1f}초, 약 {speed:.1f} tok/s)")

    print("\n출처:")
    for source in search.dedupe_sources(results):
        print(f"  - {source.page_title}: {source.url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 질의")
    parser.add_argument("question", nargs="*", help="질문. 비우면 대화형 모드")
    parser.add_argument("--retrieval-only", action="store_true", help="검색 결과만 출력")
    parser.add_argument("--verbose", action="store_true", help="검색 결과를 함께 출력")
    args = parser.parse_args()

    if args.question:
        ask(" ".join(args.question), args.retrieval_only, args.verbose)
        return

    print("질문을 입력하세요. 종료는 Ctrl+C 또는 빈 줄.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not question:
            return
        ask(question, args.retrieval_only, args.verbose)


if __name__ == "__main__":
    main()
