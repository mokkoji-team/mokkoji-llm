"""열거형 질의를 유사도 검색 없이 처리한다.

"저번달 회의 목록"은 의미 검색이 아니라 조건 조회다. top_k개만 돌려주는 벡터 검색으로는
7월 회의가 10건일 때 일부만 걸려 목록이 틀린다. 조건에 맞는 페이지를 전부 가져와 나열해야 한다.

LLM을 거치지 않으므로 응답이 1초 안에 끝나고, 없는 회의를 지어낼 여지도 없다.
"""

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

from src import config
from src.retrieval import store

NEWLINE = chr(10)

# 이 말이 들어가면 "설명"이 아니라 "나열"을 원하는 질문으로 본다.
LIST_SIGNALS = ("목록", "리스트", "뭐뭐", "몇 개", "몇개", "몇 건", "몇건", "나열", "전부", "다 보여", "다 알려")

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
