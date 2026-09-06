"""마크다운 문서를 헤딩 구조에 맞춰 청크로 나눈다.

각 청크 앞에는 `[팀 - 모꼬지 > 문서 > 개발 > API 명세서] ## 인증` 형태의 컨텍스트 헤더가 붙는다.
노션 문서는 본문이 짧고 제목·계층에 정보가 몰려 있어서, 이 헤더가 검색 정확도에 크게 기여한다.
"""

import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.documents import Document

from src import config
from src.logging_setup import get_logger
from src.notion.client import NotionPage

logger = get_logger()

HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.*)$")
FENCE_PATTERN = re.compile(r"^\s*```")

# 스크럼 표기: `**Done**` 같은 버킷 머리와 `> `석준` : flyway 도입` 같은 담당자 줄.
SCRUM_BUCKET_PATTERN = re.compile(r"^\*{0,2}(Done|To\s?Do|Issue)\*{0,2}:?$", re.IGNORECASE)
SCRUM_MEMBER_PATTERN = re.compile(r"^>\s*`([^`]+)`\s*:?\s*(.*)$")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_id: str
    page_title: str
    ancestor_path: list[str]
    url: str
    last_edited_time: str
    heading_path: list[str]
    chunk_index: int
    # 노션 속성에서 뽑은 필터용 필드(doc_date, people). 본문에도 텍스트로 들어가지만 여기 것은 타입이 살아 있다.
    properties: dict = field(default_factory=dict)
    doc_type: str = ""
    part: str = ""
    # 회의록 스크럼을 사람 단위로 쪼갠 청크에만 채워진다.
    person: str = ""
    bucket: str = ""

    def to_document(self) -> Document:
        return Document(
            page_content=self.text,
            metadata={
                "page_id": self.page_id,
                "page_title": self.page_title,
                "ancestor_path": self.ancestor_path,
                "url": self.url,
                "last_edited_time": self.last_edited_time,
                "heading_path": self.heading_path,
                "chunk_index": self.chunk_index,
                "doc_type": self.doc_type,
                "part": self.part,
                "person": self.person,
                "bucket": self.bucket,
                **self.properties,
            },
        )


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_tokenizer().encode(text, add_special_tokens=False))


@dataclass
class _Section:
    heading_path: list[str]
    lines: list[str]
    person: str = ""
    bucket: str = ""
    # 사람 단위로 쪼갠 조각은 대개 한 줄이라 병합 대상에 걸린다. 합쳐지면 앞 조각의
    # heading_path를 물려받아 "Done > 금서" 헤더에 석준 작업이 들어가므로 병합에서 뺀다.
    atomic: bool = False

    @property
    def body(self) -> str:
        return "\n".join(self.lines).strip()


def split_into_sections(markdown: str) -> list[_Section]:
    sections: list[_Section] = []
    # (레벨, 제목) 쌍으로 들고 있어야 형제 헤딩이 조상으로 잘못 쌓이지 않는다.
    # 노션 문서는 H1 없이 H2로 시작하는 경우가 많아 인덱스 기반 절단은 어긋난다.
    heading_stack: list[tuple[int, str]] = []
    current = _Section(heading_path=[], lines=[])
    in_code_fence = False

    for line in markdown.split("\n"):
        if FENCE_PATTERN.match(line):
            in_code_fence = not in_code_fence

        match = None if in_code_fence else HEADING_PATTERN.match(line)
        if match:
            if current.lines:
                sections.append(current)
            level = len(match.group(1))
            heading_stack = [entry for entry in heading_stack if entry[0] < level]
            heading_stack.append((level, match.group(2).strip()))
            current = _Section(heading_path=[title for _, title in heading_stack], lines=[])
        else:
            current.lines.append(line)

    if current.lines:
        sections.append(current)

    return [section for section in sections if section.body]


def expand_scrum(sections: list[_Section]) -> list[_Section]:
    """회의록 스크럼 섹션을 담당자 한 명당 하나로 펼친다.

    스크럼 표 하나에 파트원 전원의 작업이 들어 있어서, "flyway 도입한 사람이 누구야" 같은
    질문에 그 사람 이름이 안 걸리면 같은 주제의 전용 문서에 밀린다. 담당자를 heading_path로
    올리면 헤더에 이름이 들어가고 본문은 그 사람 작업만 남는다.
    """
    expanded: list[_Section] = []
    for section in sections:
        if not any("scrum" in title.lower() for title in section.heading_path):
            expanded.append(section)
            continue
        expanded.extend(_split_by_member(section))
    return expanded


def _split_by_member(section: _Section) -> list[_Section]:
    pieces: list[_Section] = []
    leftovers: list[str] = []
    bucket = ""
    current: _Section | None = None

    def close() -> None:
        nonlocal current
        if current is not None and current.body:
            pieces.append(current)
        current = None

    for line in section.lines:
        marker = SCRUM_BUCKET_PATTERN.match(line.strip())
        if marker:
            close()
            bucket = marker.group(1).replace(" ", "")
            continue

        member = SCRUM_MEMBER_PATTERN.match(line)
        if member:
            close()
            person = config.resolve_member(member.group(1))
            if not person:
                # 모르는 호칭은 쪼개지 않는다. 잘못된 담당자를 붙이는 것보다 낫다.
                logger.warning("스크럼 호칭을 못 폈다: %s", member.group(1))
                leftovers.append(line)
                continue
            current = _Section(
                heading_path=[*section.heading_path, *filter(None, [bucket]), person],
                lines=[member.group(2)],
                person=person,
                bucket=bucket,
                atomic=True,
            )
            continue

        if current is not None:
            current.lines.append(line)
        else:
            leftovers.append(line)

    close()

    if not pieces:
        return [section]

    # 담당자에 안 붙는 줄(머리말, 안내 문구)은 원래 경로로 남긴다.
    remainder = _Section(heading_path=section.heading_path, lines=leftovers)
    if remainder.body.strip(">").strip():
        pieces.insert(0, remainder)
    return pieces


def merge_small_sections(sections: list[_Section]) -> list[_Section]:
    """짧은 섹션끼리만 합친다. 짧은 토글 항목이 많아 이 단계가 필요하다.

    충분히 긴 섹션은 그대로 둔다. 짧은 섹션에 긴 섹션을 붙이면 합쳐진 청크가 앞 섹션의
    헤딩 경로를 물려받아 컨텍스트 헤더가 내용과 어긋난다.
    """
    merged: list[_Section] = []
    buffer: _Section | None = None

    for section in sections:
        if section.atomic or count_tokens(section.body) >= config.CHUNK_MIN_TOKENS:
            if buffer is not None:
                merged.append(buffer)
                buffer = None
            merged.append(section)
            continue

        if buffer is None:
            buffer = _Section(heading_path=section.heading_path, lines=list(section.lines))
        else:
            buffer.lines.append("")
            if section.heading_path:
                buffer.lines.append(f"## {' > '.join(section.heading_path)}")
            buffer.lines.extend(section.lines)

        if count_tokens(buffer.body) >= config.CHUNK_MIN_TOKENS:
            merged.append(buffer)
            buffer = None

    if buffer is not None:
        merged.append(buffer)

    return merged


def _split_long_line(line: str) -> list[str]:
    tokenizer = _tokenizer()
    token_ids = tokenizer.encode(line, add_special_tokens=False)
    pieces = []
    for start in range(0, len(token_ids), config.CHUNK_MAX_TOKENS):
        window = token_ids[start : start + config.CHUNK_MAX_TOKENS]
        pieces.append(tokenizer.decode(window))
    return pieces


def split_long_section(section: _Section, budget: int) -> list[str]:
    """긴 섹션을 줄 단위 슬라이딩 윈도로 나눈다. 토큰 단위로 자르는 것보다 문장이 덜 깨진다."""
    lines: list[str] = []
    for line in section.lines:
        if count_tokens(line) > budget:
            lines.extend(_split_long_line(line))
        else:
            lines.append(line)

    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = count_tokens(line)
        if current and current_tokens + line_tokens > budget:
            windows.append("\n".join(current).strip())

            overlap: list[str] = []
            overlap_tokens = 0
            for previous in reversed(current):
                previous_tokens = count_tokens(previous)
                if overlap_tokens + previous_tokens > config.CHUNK_OVERLAP_TOKENS:
                    break
                overlap.insert(0, previous)
                overlap_tokens += previous_tokens

            current = overlap
            current_tokens = overlap_tokens

        current.append(line)
        current_tokens += line_tokens

    if current:
        windows.append("\n".join(current).strip())

    return [window for window in windows if window]


def build_context_header(page: NotionPage, heading_path: list[str]) -> str:
    header = f"[{page.breadcrumb}]"
    if heading_path:
        header += " " + " > ".join(heading_path)
    return header


def chunk_page(page: NotionPage) -> list[Chunk]:
    sections = merge_small_sections(expand_scrum(split_into_sections(page.markdown)))

    chunks: list[Chunk] = []
    for section in sections:
        header = build_context_header(page, section.heading_path)
        # 회의록처럼 계층이 깊으면 헤더만으로 예산을 다 먹을 수 있다. 본문 자리를 최소한 남긴다.
        budget = max(128, config.CHUNK_MAX_TOKENS - count_tokens(header))

        bodies = (
            [section.body]
            if count_tokens(section.body) <= budget
            else split_long_section(section, budget)
        )
        for body in bodies:
            chunks.append(_make_chunk(page, section, f"{header}\n{body}", len(chunks)))

    if not chunks:
        # 본문이 비어 있어도 제목과 경로는 남긴다. "그 문서 어디 있어" 류 질문에 쓰인다.
        chunks.append(_make_chunk(page, _Section([], []), build_context_header(page, []), 0))

    return chunks


def _make_chunk(page: NotionPage, section: _Section, text: str, index: int) -> Chunk:
    digest = hashlib.sha1(f"{page.page_id}:{index}".encode()).hexdigest()
    return Chunk(
        chunk_id=f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}",
        text=text,
        page_id=page.page_id,
        page_title=page.title,
        ancestor_path=page.ancestor_path,
        url=page.url,
        last_edited_time=page.last_edited_time,
        heading_path=section.heading_path,
        chunk_index=index,
        properties=page.properties,
        doc_type=config.classify_doc_type(page.ancestor_path, page.title),
        part=config.classify_part(page.ancestor_path, page.title),
        person=section.person,
        bucket=section.bucket,
    )
