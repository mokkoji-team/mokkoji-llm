"""인코더(임베딩 모델) 팩토리.

`.env`의 EMBEDDING_PROVIDER / EMBEDDING_MODEL만 바꾸면 인코더가 교체된다.
sparse를 내놓지 못하는 인코더를 고르면 supports_hybrid가 False가 되고,
검색은 dense 단독으로 떨어진다. 인코더를 바꾸면 벡터 차원이 달라지므로
컬렉션을 새로 만들고(`--recreate`) 다시 인덱싱해야 한다.
"""

from dataclasses import dataclass
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings

from src import config


@dataclass(frozen=True)
class Encoder:
    dense: Embeddings
    sparse: SparseEmbeddings | None
    description: str

    @property
    def supports_hybrid(self) -> bool:
        return self.sparse is not None


def _bge_m3() -> Encoder:
    from src.models.bge_m3 import BgeM3Backend, BgeM3DenseEmbeddings, BgeM3SparseEmbeddings

    backend = BgeM3Backend(
        model_name=config.EMBEDDING_MODEL,
        batch_size=config.EMBED_BATCH_SIZE,
        # 컨텍스트 헤더가 붙은 만큼 청크 상한보다 여유를 둔다.
        max_length=config.CHUNK_MAX_TOKENS + 64,
    )
    return Encoder(
        dense=BgeM3DenseEmbeddings(backend),
        sparse=BgeM3SparseEmbeddings(backend),
        description=f"{config.EMBEDDING_MODEL} (dense+sparse)",
    )


def _huggingface() -> Encoder:
    from langchain_huggingface import HuggingFaceEmbeddings

    return Encoder(
        dense=HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": config.EMBED_BATCH_SIZE},
        ),
        sparse=None,
        description=f"{config.EMBEDDING_MODEL} (dense only)",
    )


def _openai() -> Encoder:
    from langchain_openai import OpenAIEmbeddings

    return Encoder(
        dense=OpenAIEmbeddings(model=config.EMBEDDING_MODEL),
        sparse=None,
        description=f"{config.EMBEDDING_MODEL} (dense only)",
    )


PROVIDERS = {
    "bge-m3": _bge_m3,
    "huggingface": _huggingface,
    "openai": _openai,
}


@lru_cache(maxsize=1)
def get_encoder() -> Encoder:
    provider = config.EMBEDDING_PROVIDER
    if provider not in PROVIDERS:
        raise ValueError(
            f"EMBEDDING_PROVIDER={provider!r}는 지원하지 않습니다. "
            f"가능한 값: {', '.join(PROVIDERS)}"
        )
    return PROVIDERS[provider]()


def warm_up() -> None:
    """모델 로딩과 첫 추론을 미리 끝내둔다. 봇 시작 시 첫 질문이 느려지는 걸 막는다."""
    get_encoder().dense.embed_query("워밍업")
