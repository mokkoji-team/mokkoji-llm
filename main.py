import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag import ask


logger = logging.getLogger(__name__)
app = FastAPI(title="Mokkoji LLM API")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="회의록에 물어볼 질문")


class Source(BaseModel):
    file_name: str
    score: float | None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> dict:
    try:
        return ask(request.question)
    except Exception as exc:
        logger.exception("LLM response generation failed")
        raise HTTPException(status_code=503, detail="LLM 응답 생성에 실패했습니다.") from exc
