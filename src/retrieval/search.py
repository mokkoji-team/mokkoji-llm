"""하이브리드 검색. 인코더가 sparse를 내놓으면 dense+sparse를 Qdrant 서버측 RRF로 융합한다.

질문이 파트를 지목하면("프론트 돌리려면…") 해당 파트로 좁힌 검색을 한 번 더 돌려 섞는다.
청크 앞에 경로 헤더가 붙어 있어 제목·경로에 없는 단어로 물으면 밀리는데, 파트는 경로에
이미 들어 있는 정보라 필터로 되살릴 수 있다.
"""

import re
from dataclasses import dataclass
from itertools import zip_longest

from langchain_core.documents import Document
from qdrant_client import models

from src import config
from src.retrieval.store import PART_FIELD, get_vector_store

# "서버"는 넣지 않는다. 프론트엔드 문서화에도 서버 다운 문서가 있어 파트를 가르는 근거가 못 된다.
PART_HINTS = ((re.compile("프론트"), "프론트엔드"), (re.compile("백엔드"), "백엔드"))


@dataclass
class SearchResult:
    text: str
    page_id: str
    page_title: str
    breadcrumb: str
    url: str
    heading_path: list[str]
    score: float
    person: str = ""
    bucket: str = ""

    @property
    def location(self) -> str:
        if self.heading_path:
            return f"{self.breadcrumb} > {' > '.join(self.heading_path)}"
        return self.breadcrumb

    @property
    def body(self) -> str:
        """컨텍스트 헤더를 뗀 본문. 헤더는 location으로 따로 제공하므로 중복을 피한다."""
        _, separator, rest = self.text.partition("\n")
        return rest.strip() if separator else ""


def dedupe_sources(results: list[SearchResult]) -> list[SearchResult]:
    """페이지당 하나만 남긴다. 같은 문서의 여러 청크가 걸리는 경우가 흔하다."""
    seen: set[str] = set()
    sources = []
    for result in results:
        if result.url in seen:
            continue
        seen.add(result.url)
        sources.append(result)
    return sources


def _part_filter(part: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key=PART_FIELD, match=models.MatchValue(value=part))]
    )


def _to_result(document: Document, score: float) -> SearchResult:
    metadata = document.metadata
    breadcrumb = " > ".join([*metadata.get("ancestor_path", []), metadata.get("page_title", "")])
    return SearchResult(
        text=document.page_content,
        page_id=metadata.get("page_id", ""),
        page_title=metadata.get("page_title", ""),
        breadcrumb=breadcrumb,
        url=metadata.get("url", ""),
        heading_path=metadata.get("heading_path", []),
        score=score,
        person=metadata.get("person", ""),
        bucket=metadata.get("bucket", ""),
    )


def detect_part(query: str) -> str:
    """질문이 지목한 파트. 없으면 빈 문자열."""
    for pattern, part in PART_HINTS:
        if pattern.search(query):
            return part
    return ""


def _fetch(query: str, limit: int, search_filter: models.Filter | None) -> list[SearchResult]:
    # LangChain은 dense/sparse 각각의 prefetch 개수를 k와 같은 값으로 잡는다.
    # 융합 후보를 넓게 두려면 prefetch 크기로 요청한 뒤 상위 top_k만 남겨야 한다.
    candidates = get_vector_store().similarity_search_with_score(query, k=limit, filter=search_filter)
    return [_to_result(document, score) for document, score in candidates]


def _interleave(narrowed: list[SearchResult], full: list[SearchResult]) -> list[SearchResult]:
    """두 결과를 번갈아 섞되 전체 검색 쪽을 먼저 놓는다.

    파트 필터는 전체 검색이 놓친 것을 건지는 보조 수단이다. 좁힌 쪽을 앞에 두면
    이미 1위로 맞히던 질문까지 한 칸씩 밀린다 — 실측으로 세 문항이 그렇게 밀렸다.
    """
    merged: list[SearchResult] = []
    seen: set[str] = set()
    for pair in zip_longest(full, narrowed):
        for result in pair:
            if result is None or result.text in seen:
                continue
            seen.add(result.text)
            merged.append(result)
    return merged


def search(query: str, top_k: int | None = None) -> list[SearchResult]:
    top_k = top_k or config.SEARCH_TOP_K
    limit = max(top_k, config.SEARCH_PREFETCH_LIMIT)

    full = _fetch(query, limit, None)
    part = detect_part(query)
    if not part:
        return full[:top_k]

    narrowed = _fetch(query, limit, _part_filter(part))
    return _interleave(narrowed, full)[:top_k]
