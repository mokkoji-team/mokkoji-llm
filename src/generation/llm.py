"""답변 생성. 실제 모델 선택은 src.models.chat이 맡는다.

주 디코더가 쿼터를 넘기거나 장애면 폴백 디코더로 넘어간다.
"""

import logging
from collections.abc import Callable, Iterator

from langchain_core.messages import BaseMessageChunk

from src import config
from src.models.chat import get_chat_model, get_fallback_chat_model

logger = logging.getLogger("mokkoji-rag")

# 쿼터 초과(429)와 서버·네트워크 장애만 폴백한다. 키 오류나 잘못된 요청(4xx)까지
# 폴백하면 설정이 깨진 걸 모른 채 몇 배 느린 답변을 계속 받게 된다.
FALLBACK_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _deserves_fallback(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(error, "code", None)
    if isinstance(status, int):
        return status in FALLBACK_STATUS_CODES
    return isinstance(error, (ConnectionError, TimeoutError))


def stream_chunks(
    messages: list[dict], on_fallback: Callable[[], None] | None = None
) -> Iterator[BaseMessageChunk]:
    """청크 객체를 그대로 흘린다.

    Ollama는 마지막 청크의 response_metadata에 prefill/decode 실측치를 실어 보낸다.
    지연을 재려면 텍스트만으로는 부족해서 원본 청크가 필요하다(scripts.ask --timing).

    폴백은 첫 청크가 오기 전까지만 한다. 이미 흘려보낸 뒤에 모델을 바꾸면
    앞부분이 중복된다. 쿼터 초과는 요청 시점에 바로 오므로 이 제약이 문제되지 않는다.
    """
    fallback = get_fallback_chat_model()
    stream = get_chat_model().stream(messages)

    try:
        first = next(stream)
    except StopIteration:
        return
    except Exception as error:
        if fallback is None or not _deserves_fallback(error):
            raise
        logger.warning(
            "주 디코더(%s) 실패: %r. 폴백 %s로 전환한다.",
            config.LLM_MODEL,
            error,
            config.FALLBACK_LLM_MODEL,
        )
        if on_fallback:
            on_fallback()
        yield from fallback.stream(messages)
        return

    yield first
    yield from stream


def stream_answer(
    messages: list[dict], on_fallback: Callable[[], None] | None = None
) -> Iterator[str]:
    for chunk in stream_chunks(messages, on_fallback):
        piece = str(chunk.text)
        if piece:
            yield piece


def answer(messages: list[dict]) -> str:
    return "".join(stream_answer(messages))


def warm_up() -> None:
    """모델을 미리 올려둔다. 봇 시작 시 첫 질문이 느려지는 걸 막는다.

    폴백은 데우지 않는다. 거의 쓰이지 않을 모델에 메모리를 미리 내줄 이유가 없다.
    """
    get_chat_model().invoke([{"role": "user", "content": "안녕"}])
