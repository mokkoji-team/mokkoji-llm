"""답변 생성. 실제 모델 선택은 src.models.chat이 맡는다."""

from collections.abc import Iterator

from langchain_core.messages import BaseMessageChunk

from src.models.chat import get_chat_model


def stream_chunks(messages: list[dict]) -> Iterator[BaseMessageChunk]:
    """청크 객체를 그대로 흘린다.

    Ollama는 마지막 청크의 response_metadata에 prefill/decode 실측치를 실어 보낸다.
    지연을 재려면 텍스트만으로는 부족해서 원본 청크가 필요하다(scripts.ask --timing).
    """
    yield from get_chat_model().stream(messages)


def stream_answer(messages: list[dict]) -> Iterator[str]:
    for chunk in stream_chunks(messages):
        piece = str(chunk.text)
        if piece:
            yield piece


def answer(messages: list[dict]) -> str:
    return str(get_chat_model().invoke(messages).text)


def warm_up() -> None:
    """모델을 미리 올려둔다. 봇 시작 시 첫 질문이 느려지는 걸 막는다."""
    get_chat_model().invoke([{"role": "user", "content": "안녕"}])
