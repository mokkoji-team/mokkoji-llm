"""답변 생성. 실제 모델 선택은 src.models.chat이 맡는다."""

from collections.abc import Iterator

from src.models.chat import get_chat_model


def stream_answer(messages: list[dict]) -> Iterator[str]:
    for chunk in get_chat_model().stream(messages):
        piece = str(chunk.text)
        if piece:
            yield piece


def answer(messages: list[dict]) -> str:
    return str(get_chat_model().invoke(messages).text)


def warm_up() -> None:
    """모델을 미리 올려둔다. 봇 시작 시 첫 질문이 느려지는 걸 막는다."""
    get_chat_model().invoke([{"role": "user", "content": "안녕"}])
