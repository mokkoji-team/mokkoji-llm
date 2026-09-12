"""CLI로 질문해서 검색 결과와 답변을 확인한다.

    python -m scripts.ask "프론트엔드 기술 스택을 왜 이렇게 골랐어?"
    python -m scripts.ask                          대화형 모드
    python -m scripts.ask --retrieval-only "..."   검색 결과만 확인
    python -m scripts.ask --timing "..."           단계별 지연 분해
"""

import argparse
import sys
import time
from dataclasses import dataclass, field

# 노션 문서에 콘솔 기본 인코딩(cp949)이 못 찍는 문자가 섞여 있다. 출력만 UTF-8로 돌린다.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.generation import llm, prompt
from src.models import embeddings
from src.retrieval import listing, search

NANOSECONDS = 1_000_000_000


@dataclass
class Timing:
    """한 질문의 지연 분해.

    prefill(첫 토큰까지)과 decode를 나누지 않으면 tok/s가 둘의 혼합이 되어 해석할 수 없다.
    프롬프트를 줄여야 하는지(prefill) 출력을 줄여야 하는지(decode) 여기서 갈린다.
    """

    embed: float = 0.0
    qdrant: float = 0.0
    first_token: float = 0.0
    decode: float = 0.0
    prompt_characters: int = 0
    context_chunks: int = 0
    streamed_pieces: int = 0
    fallback: bool = False
    # Ollama가 응답에 실어 보내는 실측치. 다른 프로바이더는 비어 있다.
    server: dict = field(default_factory=dict)

    @property
    def search(self) -> float:
        return self.embed + self.qdrant

    @property
    def total(self) -> float:
        return self.search + self.first_token + self.decode

    @property
    def prompt_tokens(self) -> int:
        return self.server.get("prompt_eval_count", 0)

    @property
    def output_tokens(self) -> int:
        return self.server.get("eval_count", self.streamed_pieces)

    @property
    def prefill_speed(self) -> float:
        return _speed(self.prompt_tokens, self.server.get("prompt_eval_duration"))

    @property
    def decode_speed(self) -> float:
        measured = _speed(self.output_tokens, self.server.get("eval_duration"))
        if measured:
            return measured
        return self.output_tokens / self.decode if self.decode else 0.0


def _speed(tokens: int, duration_nanoseconds: int | None) -> float:
    if not tokens or not duration_nanoseconds:
        return 0.0
    return tokens / (duration_nanoseconds / NANOSECONDS)


def show_results(results: list[search.SearchResult]) -> None:
    print(f"\n검색 결과 {len(results)}개")
    for number, result in enumerate(results, start=1):
        preview = result.body.replace("\n", " ")[:90]
        print(f"  [{number}] ({result.score:.4f}) {result.location}")
        print(f"      {preview}")


def show_timing(timing: Timing) -> None:
    print("\n지연 분해")
    print(f"  embed    {timing.embed:6.1f}초")
    print(f"  qdrant   {timing.qdrant:6.1f}초")

    prompt_size = f"{timing.prompt_characters:,}자"
    if timing.prompt_tokens:
        prompt_size = f"{timing.prompt_tokens:,}토큰 · {prompt_size}"
    print(f"  prompt   {prompt_size} (청크 {timing.context_chunks}개)")

    print(f"  prefill  {timing.first_token:6.1f}초{_speed_suffix(timing.prefill_speed)}")
    print(f"  decode   {timing.decode:6.1f}초{_speed_suffix(timing.decode_speed)}"
          f"  ({timing.output_tokens}토큰)")
    print(f"  합계     {timing.total:6.1f}초")


def _speed_suffix(speed: float) -> str:
    return f"  {speed:6.1f} tok/s" if speed else " " * 15


def retrieve(question: str, timing: Timing) -> list[search.SearchResult]:
    # BGE-m3 백엔드가 마지막 인코딩 결과를 들고 있다. 질의를 먼저 한 번 인코딩해두면
    # 이어지는 검색은 캐시를 타므로 인코딩 비용과 Qdrant 왕복이 분리돼 측정된다.
    # sparse가 없는 인코더(huggingface, openai)는 이 캐시가 없어 embed가 qdrant에도 섞인다.
    started = time.monotonic()
    embeddings.get_encoder().dense.embed_query(question)
    timing.embed = time.monotonic() - started

    started = time.monotonic()
    results = search.search(question)
    timing.qdrant = time.monotonic() - started
    return results


def generate(messages: list[dict], timing: Timing) -> str:
    started = time.monotonic()
    answer = ""

    def note_fallback() -> None:
        timing.fallback = True
        print("\n(주 디코더 실패 — 폴백으로 생성한다)\n")

    for chunk in llm.stream_chunks(messages, on_fallback=note_fallback):
        timing.server.update(chunk.response_metadata)
        piece = str(chunk.text)
        if not piece:
            continue
        if not timing.streamed_pieces:
            timing.first_token = time.monotonic() - started
        timing.streamed_pieces += 1
        sys.stdout.write(piece)
        sys.stdout.flush()
        answer += piece

    timing.decode = time.monotonic() - started - timing.first_token
    return answer


def ask(question: str, retrieval_only: bool, verbose: bool, detailed_timing: bool) -> None:
    # "저번달 회의 목록"과 "저번주 회의 요약"은 유사도가 아니라 조건 조회다.
    resolution = listing.resolve(question)
    if resolution and resolution.message and not retrieval_only:
        print(resolution.message)
        return

    timing = Timing()
    if resolution and resolution.results:
        results = resolution.results
        print(f"\n(날짜 조회로 {len(results)}청크 — 유사도 검색을 건너뛰었다)")
    else:
        results = retrieve(question, timing)

    if not results:
        print(prompt.NO_RESULT_MESSAGE)
        return

    if retrieval_only or verbose:
        show_results(results)

    if retrieval_only:
        print(f"\n(검색 {timing.search:.1f}초 — embed {timing.embed:.1f} · qdrant {timing.qdrant:.1f})")
        return

    messages = prompt.build_messages(question, results)
    timing.prompt_characters = sum(len(message["content"]) for message in messages)
    timing.context_chunks = len(results)

    print("\n답변:")
    generate(messages, timing)

    print(
        f"\n\n(검색 {timing.search:.1f}초 · 첫 토큰 {timing.first_token:.1f}초 · "
        f"생성 {timing.decode:.1f}초, {timing.decode_speed:.1f} tok/s)"
    )
    if detailed_timing:
        show_timing(timing)

    print("\n출처:")
    for source in search.dedupe_sources(results):
        print(f"  - {source.page_title}: {source.url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 질의")
    parser.add_argument("question", nargs="*", help="질문. 비우면 대화형 모드")
    parser.add_argument("--retrieval-only", action="store_true", help="검색 결과만 출력")
    parser.add_argument("--verbose", action="store_true", help="검색 결과를 함께 출력")
    parser.add_argument("--timing", action="store_true", help="단계별 지연을 분해해서 출력")
    args = parser.parse_args()

    if args.question:
        ask(" ".join(args.question), args.retrieval_only, args.verbose, args.timing)
        return

    print("질문을 입력하세요. 종료는 Ctrl+C 또는 빈 줄.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not question:
            return
        ask(question, args.retrieval_only, args.verbose, args.timing)


if __name__ == "__main__":
    main()
