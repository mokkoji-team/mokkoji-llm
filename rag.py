from pathlib import Path

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


OLLAMA_URL = "http://localhost:11434"
DATA_DIR = Path(__file__).parent / "data"
Settings.chunk_size = 512
Settings.chunk_overlap = 50


def build_query_engine():
    """data 폴더의 문서를 읽어 메모리 검색 엔진을 만든다."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"문서 폴더를 찾을 수 없습니다: {DATA_DIR}")

    Settings.llm = Ollama(
        model="llama3.1:8b",
        base_url=OLLAMA_URL,
        request_timeout=300.0,
        context_window=4096,
        system_prompt=(
            "너는 한국어로 답하는 회의록 도우미다. "
            "주어진 문맥만 근거로 답하고, 근거가 없으면 "
            "'문서에서 근거를 찾지 못했습니다'라고 답한다."
        ),
    )
    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url=OLLAMA_URL,
    )

    documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()
    if not documents:
        raise ValueError(f"{DATA_DIR}에 읽을 문서가 없습니다.")

    index = VectorStoreIndex.from_documents(documents)
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
