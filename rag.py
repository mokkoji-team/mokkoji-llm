from llama_index.core import Settings
from llama_index.core.postprocessor import SimilarityPostprocessor

from indexing import load_persisted_index


def build_query_engine():
    """디스크에 저장된 인덱스로 검색 엔진을 만든다."""
    index = load_persisted_index()
    return index.as_query_engine(
        similarity_top_k=4,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
    )


query_engine = build_query_engine()


def ask(question: str) -> dict:
    """질문에 답하고 실제 답변 근거로 사용된 문서 정보를 돌려준다."""
    response = query_engine.query(question)

    sources = [
        {
            "file_name": node.metadata.get("file_name", "unknown"),
            "score": round(node.score, 3) if node.score is not None else None,
            "text": node.text,
        }
        for node in response.source_nodes
    ]

    return {
        "answer": str(response),
        "sources": sources,
    }


def normalize_query(question: str) -> str:
    prompt = f"""
다음 질문을 벡터 검색용 검색어로 변환하라.

규칙:
- 원문에 없는 사실을 추가하지 않는다.
- 정답을 추론하지 않는다.
- 시간과 상태 조건을 유지한다.
- 핵심 명사와 조건만 남긴다.
- 반드시 <query>와 </query> 사이에 검색어만 출력한다.
- 설명, 이유, 원본 질문을 출력하지 않는다.

예시:
질문: 동아리장이 되려는 사용자가 카카오로 최초 가입하면 처음 role은 뭐야?
출력: <query>카카오 최초 가입 초기 role</query>

질문:
{question}

출력:
"""

    response = str(Settings.llm.complete(prompt)).strip()

    start = response.find("<query>")
    end = response.find("</query>")

    if start != -1 and end != -1:
        normalized = response[start + len("<query>"):end].strip()
        if normalized:
            return normalized

    return question
