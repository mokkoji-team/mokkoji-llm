"""BGE-m3를 LangChain의 Embeddings / SparseEmbeddings 인터페이스에 맞춘 어댑터.

BGE-m3는 dense와 sparse를 한 번의 forward에서 함께 내놓는데, LangChain은 두 인터페이스를
따로 호출한다. 그대로 두면 같은 텍스트를 두 번 인코딩해서 CPU 비용이 두 배가 되므로,
마지막 호출 결과를 들고 있다가 재사용한다. QdrantVectorStore가 dense와 sparse를
같은 텍스트 리스트로 연달아 호출하기 때문에 항목 하나짜리 캐시로 충분하다.
"""

from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings
from langchain_qdrant.sparse_embeddings import SparseVector


class BgeM3Backend:
    def __init__(self, model_name: str, batch_size: int, max_length: int):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._last_texts: tuple[str, ...] | None = None
        self._last_output: dict | None = None

    def _load(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            # CPU 환경이라 fp16을 끈다. CUDA가 없으면 FlagEmbedding이 알아서 CPU로 떨어진다.
            self._model = BGEM3FlagModel(self.model_name, use_fp16=False)
        return self._model

    def encode(self, texts: list[str]) -> dict:
        key = tuple(texts)
        if key == self._last_texts:
            return self._last_output

        output = self._load().encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        self._last_texts = key
        self._last_output = output
        return output


class BgeM3DenseEmbeddings(Embeddings):
    def __init__(self, backend: BgeM3Backend):
        self.backend = backend

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        output = self.backend.encode(list(texts))
        return [vector.tolist() for vector in output["dense_vecs"]]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class BgeM3SparseEmbeddings(SparseEmbeddings):
    def __init__(self, backend: BgeM3Backend):
        self.backend = backend

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        output = self.backend.encode(list(texts))
        return [
            SparseVector(
                indices=[int(token_id) for token_id in weights],
                values=[float(weight) for weight in weights.values()],
            )
            for weights in output["lexical_weights"]
        ]

    def embed_query(self, text: str) -> SparseVector:
        return self.embed_documents([text])[0]
