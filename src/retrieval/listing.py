"""열거형 질의를 유사도 검색 없이 처리한다.

"저번달 회의 목록"은 의미 검색이 아니라 조건 조회다. top_k개만 돌려주는 벡터 검색으로는
7월 회의가 10건일 때 일부만 걸려 목록이 틀린다. 조건에 맞는 페이지를 전부 가져와 나열해야 한다.

LLM을 거치지 않으므로 응답이 1초 안에 끝나고, 없는 회의를 지어낼 여지도 없다.
"""

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from src import config
from src.retrieval import store
from src.retrieval.search import SearchResult

NEWLINE = chr(10)

# 이 말이 들어가면 "설명"이 아니라 "나열"을 원하는 질문으로 본다.
LIST_SIGNALS = ("목록", "리스트", "뭐뭐", "몇 개", "몇개", "몇 건", "몇건", "나열", "전부", "다 보여", "다 알려")

# 이 말과 기간이 함께 나오면 날짜로 좀혀 문서를 통째로 넘긴다.
# "2월 회의록에서 API 스펙 뭔로 정했어" 같은 구체적인 질문까지 가로채면
# 유사도 검색이 더 잘하는 일까지 빼앗게 된다.
SUMMARY_SIGNALS = ("요약", "정리", "뭔 했", "뭔했", "무슨 일", "무슨 얘기", "어떤 얘기", "무슨 논의", "어떤 논의", "리캡")

# 프롬프트 상한. 넘으면 요약하지 않고 목록을 돌려준다.
# 잘라서 요약하면 빠진 문서가 생기는데 물어본 사람은 그걸 알 수가 없다.
SUMMARY_CHARACTER_LIMIT = 40000

# 질문에 이 말이 들어가면 해당 문서 유형으로 좁힌다.
DOC_TYPE_HINTS = {
    "회의": "회의록",
    "온보딩": "온보딩",
    "api": "API 명세서",
    "명세": "API 명세서",
    "qa": "QA",
}


@dataclass
class Period:
    start: date
    end: date
    label: str


def _month_range(year: int, month: int) -> Period:
    last = calendar.monthrange(year, month)[1]
    return Period(date(year, month, 1), date(year, month, last), f"{year}년 {month}월")


def _shift_month(base: date, delta: int) -> tuple[int, int]:
    index = base.year * 12 + (base.month - 1) + delta
    return index // 12, index % 12 + 1


def extract_period(question: str, today: date | None = None) -> Period | None:
    """질문에서 기간을 뽑는다. LLM을 쓰지 않는다 — 10초 예산에 여유가 없다."""
    today = today or date.today()
    text = question.replace(" ", "")

    if re.search(r"(저번|지난|전)달|전월", text):
        return _month_range(*_shift_month(today, -1))
    if re.search(r"(이번|금)달|이달", text):
        return _month_range(today.year, today.month)
    if re.search(r"(저번|지난)주", text):
        monday = today - timedelta(days=today.weekday() + 7)
        return Period(monday, monday + timedelta(days=6), "지난주")
    if re.search(r"이번주", text):
        monday = today - timedelta(days=today.weekday())
        return Period(monday, monday + timedelta(days=6), "이번주")
    if re.search(r"작년|지난해", text):
        return Period(date(today.year - 1, 1, 1), date(today.year - 1, 12, 31), f"{today.year - 1}년")
    if re.search(r"올해|금년", text):
        return Period(date(today.year, 1, 1), date(today.year, 12, 31), f"{today.year}년")

    recent = re.search(r"최근(\d+)(개월|달|주|일)", text)
    if recent:
        amount = int(recent.group(1))
        days = {"개월": 30, "달": 30, "주": 7, "일": 1}[recent.group(2)] * amount
        return Period(today - timedelta(days=days), today, f"최근 {amount}{recent.group(2)}")

    explicit = re.search(r"(20\d\d)년(\d{1,2})월", text)
    if explicit:
        return _month_range(int(explicit.group(1)), int(explicit.group(2)))

    year_only = re.search(r"(20\d\d)년", text)
    if year_only:
        year = int(year_only.group(1))
        return Period(date(year, 1, 1), date(year, 12, 31), f"{year}년")

    month_only = re.search(r"(?<!\d)(\d{1,2})월", text)
    if month_only:
        month = int(month_only.group(1))
        if 1 <= month <= 12:
            return _month_range(today.year, month)

    return None


def extract_doc_type(question: str) -> str:
    lowered = question.lower()
    for hint, doc_type in DOC_TYPE_HINTS.items():
        if hint in lowered:
            return doc_type
    return ""


def is_list_query(question: str) -> bool:
    text = question.replace(" ", "")
    return any(signal.replace(" ", "") in text for signal in LIST_SIGNALS)


def is_summary_query(question: str) -> bool:
    text = question.replace(" ", "")
    return any(signal.replace(" ", "") in text for signal in SUMMARY_SIGNALS)


def answer(question: str, today: date | None = None) -> str | None:
    """열거형 질의면 완성된 답변 문자열을, 아니면 None을 돌려준다.

    None이면 호출한 쪽이 평소대로 하이브리드 검색 + LLM 경로를 탄다.
    """
    if not is_list_query(question):
        return None

    doc_type = extract_doc_type(question)
    period = extract_period(question, today)
    if not doc_type and not period:
        return None

    return render_pages(doc_type, period)


def render_pages(doc_type: str, period: Period | None) -> str:
    """조건에 맞는 페이지를 링크 목록으로 렌더한다."""
    pages = store.scroll_pages(doc_type=doc_type)

    if period:
        start, end = period.start.isoformat(), period.end.isoformat()
        pages = [page for page in pages if page["doc_date"] and start <= page["doc_date"] <= end]

    pages.sort(key=lambda page: (page["doc_date"], page["title"]))

    scope = " ".join(filter(None, [period.label if period else "", doc_type or "문서"]))
    if not pages:
        return f"{scope}에 해당하는 문서를 찾지 못했습니다."

    lines = [f"**{scope} {len(pages)}건**", ""]
    for page in pages:
        stamp = f" ({page['doc_date']})" if page["doc_date"] else ""
        parent = page["ancestor_path"][-1] if page["ancestor_path"] else ""
        where = f" — {parent}" if parent else ""
        lines.append(f"- [{page['title']}]({page['url']}){stamp}{where}")
    return NEWLINE.join(lines)


@dataclass
class Resolution:
    """유사도 검색 없이 처리한 결과.

    message가 차 있으면 그대로 답하고 LLM을 부르지 않는다.
    results가 차 있으면 그 발췌로 LLM을 돌린다.
    """

    message: str = ""
    results: list[SearchResult] = field(default_factory=list)


def _to_result(chunk: dict) -> SearchResult:
    metadata = chunk["metadata"]
    breadcrumb = " > ".join([*metadata.get("ancestor_path", []), metadata.get("page_title", "")])
    return SearchResult(
        text=chunk["text"],
        page_id=metadata.get("page_id", ""),
        page_title=metadata.get("page_title", ""),
        breadcrumb=breadcrumb,
        url=metadata.get("url", ""),
        heading_path=metadata.get("heading_path", []),
        # 조건 조회라 유사도 점수가 없다. RRF 점수와 섞이지 않도록 0으로 둔다.
        score=0.0,
        person=metadata.get("person", ""),
        bucket=metadata.get("bucket", ""),
    )


def resolve(question: str, today: date | None = None) -> Resolution | None:
    """유사도 검색을 타기 전에 조건 조회로 끝낼 수 있는지 본다.

    None이면 호출한 쪽이 평소대로 하이브리드 검색을 한다.
    """
    listed = answer(question, today)
    if listed:
        return Resolution(message=listed)

    if not is_summary_query(question):
        return None

    period = extract_period(question, today)
    doc_type = extract_doc_type(question)
    if not period or not doc_type:
        return None

    chunks = store.scroll_chunks(doc_type, period.start.isoformat(), period.end.isoformat())
    if not chunks:
        return Resolution(message=f"{period.label} {doc_type}을 찾지 못했습니다.")

    total = sum(len(chunk["text"]) for chunk in chunks)
    if total > SUMMARY_CHARACTER_LIMIT:
        pages = len({chunk["metadata"].get("page_id") for chunk in chunks})
        notice = (
            f"{period.label} {doc_type}이 {pages}건이라 한 번에 요약하기에는 양이 많습니다. "
            "기간을 좁혀서 다시 물어봐 주세요."
        )
        return Resolution(message=notice + NEWLINE + NEWLINE + render_pages(doc_type, period))

    return Resolution(results=[_to_result(chunk) for chunk in chunks])
