from llama_index.core.postprocessor import SimilarityPostprocessor
from indexing import load_persisted_index


def build_query_engine():
    """디스크에 저장된 인덱스로 검색 엔진을 만든다."""
    index = load_persisted_index()
    return index.as_query_engine(
        similarity_top_k=2,
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
        }
        for node in response.source_nodes
    ]
    return {"answer": str(response), "sources": sources}
