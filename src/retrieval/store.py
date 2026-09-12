"""Qdrant 컬렉션 관리. LangChain QdrantVectorStore를 감싼다.

인코더가 sparse를 함께 내놓으면 하이브리드(dense+sparse, Qdrant 서버측 RRF)로,
아니면 dense 단독으로 컬렉션 구성과 검색 모드가 함께 정해진다.
벡터 차원은 인코더에서 직접 재므로 하드코딩하지 않는다 — 인코더를 바꿔도 그대로 돈다.
"""

from functools import lru_cache

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from src import config
from src.models.embeddings import get_encoder

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
CONTENT_KEY = "text"
METADATA_KEY = "metadata"

# LangChain은 메타데이터를 payload["metadata"] 아래에 중첩해 넣는다. 인덱스 경로도 그에 맞춘다.
PAGE_ID_FIELD = f"{METADATA_KEY}.page_id"
DOC_TYPE_FIELD = f"{METADATA_KEY}.doc_type"
DOC_DATE_FIELD = f"{METADATA_KEY}.doc_date"
PART_FIELD = f"{METADATA_KEY}.part"
PERSON_FIELD = f"{METADATA_KEY}.person"
BUCKET_FIELD = f"{METADATA_KEY}.bucket"

# 열거형 질의("저번달 회의 목록")는 유사도가 아니라 이 인덱스로 거른다.
# doc_date는 "2026-07-04" 형태 문자열이라 KEYWORD로 두고 범위 비교는 파이썬에서 한다.
# DATETIME 스키마를 쓰면 값이 하나라도 형식을 벗어날 때 인덱싱이 통째로 실패한다.
FILTER_INDEXES = {
    PAGE_ID_FIELD: models.PayloadSchemaType.KEYWORD,
    DOC_TYPE_FIELD: models.PayloadSchemaType.KEYWORD,
    DOC_DATE_FIELD: models.PayloadSchemaType.KEYWORD,
    PART_FIELD: models.PayloadSchemaType.KEYWORD,
    PERSON_FIELD: models.PayloadSchemaType.KEYWORD,
    BUCKET_FIELD: models.PayloadSchemaType.KEYWORD,
}


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL, timeout=60)


def ensure_collection(recreate: bool = False) -> None:
    client = get_client()
    exists = client.collection_exists(config.QDRANT_COLLECTION)

    if exists and recreate:
        client.delete_collection(config.QDRANT_COLLECTION)
        exists = False

    if exists:
        return

    encoder = get_encoder()
    dimension = len(encoder.dense.embed_query("차원 측정"))

    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(size=dimension, distance=models.Distance.COSINE)
        },
        sparse_vectors_config=(
            {SPARSE_VECTOR: models.SparseVectorParams()} if encoder.supports_hybrid else None
        ),
    )
    ensure_payload_indexes()


def ensure_payload_indexes() -> None:
    """필터·삭제에 필요한 페이로드 인덱스를 만든다. 이미 있으면 Qdrant가 무시한다."""
    client = get_client()
    for field_name, schema in FILTER_INDEXES.items():
        client.create_payload_index(
            collection_name=config.QDRANT_COLLECTION,
            field_name=field_name,
            field_schema=schema,
        )


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    encoder = get_encoder()
    return QdrantVectorStore(
        client=get_client(),
        collection_name=config.QDRANT_COLLECTION,
        embedding=encoder.dense,
        sparse_embedding=encoder.sparse,
        retrieval_mode=RetrievalMode.HYBRID if encoder.supports_hybrid else RetrievalMode.DENSE,
        vector_name=DENSE_VECTOR,
        sparse_vector_name=SPARSE_VECTOR,
        content_payload_key=CONTENT_KEY,
        metadata_payload_key=METADATA_KEY,
    )


def add_documents(documents: list[Document], ids: list[str]) -> None:
    get_vector_store().add_documents(documents, ids=ids)


def _page_filter(page_ids: list[str]) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key=PAGE_ID_FIELD, match=models.MatchAny(any=page_ids))]
    )


def delete_pages(page_ids: list[str]) -> None:
    if not page_ids:
        return
    get_client().delete(
        collection_name=config.QDRANT_COLLECTION,
        points_selector=models.FilterSelector(filter=_page_filter(page_ids)),
        wait=True,
    )


def count_all() -> int:
    return get_client().count(config.QDRANT_COLLECTION, exact=True).count


def count_pages(page_ids: list[str]) -> int:
    if not page_ids:
        return 0
    return get_client().count(
        config.QDRANT_COLLECTION, count_filter=_page_filter(page_ids), exact=True
    ).count


def indexed_page_versions() -> dict[str, str]:
    """인덱싱된 page_id → last_edited_time. 증분 동기화에서 변경/삭제 판정에 쓴다."""
    client = get_client()
    versions: dict[str, str] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=[METADATA_KEY],
            with_vectors=False,
        )
        for point in points:
            metadata = point.payload.get(METADATA_KEY, {})
            versions[metadata["page_id"]] = metadata.get("last_edited_time", "")
        if offset is None:
            return versions


def scroll_chunks(doc_type: str, start_date: str, end_date: str) -> list[dict]:
    """기간에 걸린 청크를 본문까지 전부 가져온다.

    날짜는 벡터에 안 들어가서 "저번주 회의"를 유사도로는 찾지 못한다.
    실측해 보니 2월 회의록이 7위로 올라왔다. 조건 조회로 가야 한다.

    doc_date는 KEYWORD라 범위 비교를 Qdrant에 맡길 수 없다. 파이썬에서 걸러낸다.
    """
    scroll_filter = models.Filter(
        must=[models.FieldCondition(key=DOC_TYPE_FIELD, match=models.MatchValue(value=doc_type))]
    )

    client = get_client()
    chunks: list[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            scroll_filter=scroll_filter,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            metadata = point.payload.get(METADATA_KEY, {})
            doc_date = metadata.get("doc_date", "")
            if not doc_date or not start_date <= doc_date <= end_date:
                continue
            chunks.append({"text": point.payload.get(CONTENT_KEY, ""), "metadata": metadata})
        if offset is None:
            break

    chunks.sort(key=lambda chunk: (chunk["metadata"].get("doc_date", ""), chunk["metadata"].get("page_id", "")))
    return chunks


def scroll_pages(doc_type: str = "", with_date_only: bool = False) -> list[dict]:
    """조건에 맞는 페이지를 유사도 검색 없이 전부 가져온다.

    열거형 질의("저번달 회의 목록")는 상위 k개가 아니라 조건을 만족하는 전부가 필요하다.
    청크 단위로 저장돼 있으므로 page_id로 접어서 페이지 단위로 돌려준다.
    """
    conditions = []
    if doc_type:
        conditions.append(
            models.FieldCondition(key=DOC_TYPE_FIELD, match=models.MatchValue(value=doc_type))
        )
    scroll_filter = models.Filter(must=conditions) if conditions else None

    client = get_client()
    pages: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            scroll_filter=scroll_filter,
            limit=1000,
            offset=offset,
            with_payload=[METADATA_KEY],
            with_vectors=False,
        )
        for point in points:
            metadata = point.payload.get(METADATA_KEY, {})
            page_id = metadata.get("page_id")
            if page_id in pages:
                continue
            pages[page_id] = {
                "page_id": page_id,
                "title": metadata.get("page_title", ""),
                "ancestor_path": metadata.get("ancestor_path", []),
                "url": metadata.get("url", ""),
                "doc_date": metadata.get("doc_date", ""),
                "doc_type": metadata.get("doc_type", ""),
                "people": metadata.get("people", []),
            }
        if offset is None:
            break

    results = list(pages.values())
    if with_date_only:
        results = [page for page in results if page["doc_date"]]
    return results
