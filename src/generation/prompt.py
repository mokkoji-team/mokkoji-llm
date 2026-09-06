"""RAG 프롬프트 조립."""

from src.retrieval.search import SearchResult

SYSTEM_PROMPT = """너는 대학 개발 동아리 '모꼬지' 팀의 노션 문서를 찾아주는 어시스턴트다.

규칙:
1. 아래 제공된 문서 발췌만 근거로 답한다. 발췌에 없는 내용은 지어내지 않는다.
2. 문장마다 근거가 된 발췌 번호를 [1], [2] 형태로 표기한다.
3. 발췌에 답이 없으면 "문서에서 찾지 못했습니다"라고 말하고, 관련 있어 보이는 문서 제목을 알려준다.
4. 한국어로, 5문장 이내로 간결하게 답한다. 목록이 필요하면 짧은 불릿으로 쓴다.
5. 출처 링크는 답변에 넣지 않는다. 시스템이 따로 붙인다."""

NO_RESULT_MESSAGE = "문서에서 찾지 못했습니다. 질문을 조금 더 구체적으로 바꿔서 다시 물어봐 주세요."


def build_context(results: list[SearchResult]) -> str:
    blocks = []
    for number, result in enumerate(results, start=1):
        body = result.body or "(본문 없음 — 제목만 있는 문서)"
        blocks.append(f"[{number}] {result.location}{_attribution(result)}\n{body}")
    return "\n\n".join(blocks)


def _attribution(result: SearchResult) -> str:
    """회의록 스크럼 조각의 담당자를 한 줄로 세운다.

    담당자는 location 맨 끝에도 들어 있지만 아홉 단짜리 경로에 묻혀서, 작은 모델이
    "누가 했어"라는 질문에 이 이름을 근거로 쓰지 못한다.
    """
    if not result.person:
        return ""
    state = {"Done": "완료", "ToDo": "예정", "Issue": "이슈"}.get(result.bucket, result.bucket)
    return f"\n담당: {result.person}" + (f" ({state})" if state else "")


def build_messages(question: str, results: list[SearchResult]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"문서 발췌:\n\n{build_context(results)}\n\n---\n\n질문: {question}",
        },
    ]
