"""검색 품질 평가. golden_set.jsonl의 질문마다 정답 페이지가 상위에 오는지 측정한다.

    python -m src.eval.run_eval
    python -m src.eval.run_eval --show-misses

golden_set.jsonl 한 줄 형식:
    {"question": "3차 스프린트 리딩은 누구였어?", "answer_page_ids": ["1837455c17d081b0b5a1f2e71b991059"]}
"""

import argparse
import json
import sys
from pathlib import Path

# 노션 문서에 콘솔 기본 인코딩(cp949)이 못 찍는 문자가 섞여 있다. 출력만 UTF-8로 돌린다.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import config
from src.models import embeddings
from src.retrieval import search

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"
MAX_K = 10


def load_golden_set() -> list[dict]:
    if not GOLDEN_SET_PATH.exists():
        raise RuntimeError(f"골든 세트가 없습니다: {GOLDEN_SET_PATH}")
    lines = GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def evaluate(show_misses: bool) -> None:
    golden_set = load_golden_set()
    print(f"골든 세트 {len(golden_set)}개 평가 중…\n")
    embeddings.warm_up()

    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks = []
    misses = []

    for item in golden_set:
        question = item["question"]
        expected = {config.normalize_page_id(page_id) for page_id in item["answer_page_ids"]}
        results = search.search(question, top_k=MAX_K)
        retrieved = [config.normalize_page_id(result.page_id) for result in results]

        rank = next((index + 1 for index, page_id in enumerate(retrieved) if page_id in expected), None)

        if rank and rank <= 5:
            hits_at_5 += 1
        if rank and rank <= 10:
            hits_at_10 += 1
        reciprocal_ranks.append(1 / rank if rank else 0.0)

        if not rank or rank > 5:
            misses.append((question, rank, results[:3]))

    total = len(golden_set)
    print(f"Recall@5  : {hits_at_5 / total:.3f}  ({hits_at_5}/{total})")
    print(f"Recall@10 : {hits_at_10 / total:.3f}  ({hits_at_10}/{total})")
    print(f"MRR@10    : {sum(reciprocal_ranks) / total:.3f}")

    if show_misses and misses:
        print(f"\n상위 5위 안에 못 든 질문 {len(misses)}개:")
        for question, rank, top_results in misses:
            position = f"{rank}위" if rank else "미검출"
            print(f"\n  Q: {question}  [{position}]")
            for result in top_results:
                print(f"     - {result.location}")


def main() -> None:
    parser = argparse.ArgumentParser(description="검색 품질 평가")
    parser.add_argument("--show-misses", action="store_true", help="실패한 질문과 검색 결과를 출력")
    args = parser.parse_args()
    evaluate(args.show_misses)


if __name__ == "__main__":
    main()
