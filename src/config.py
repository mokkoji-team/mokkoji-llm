import os
import re
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CACHE_DIR = PROJECT_ROOT / ".cache"
SYNC_STATE_PATH = PROJECT_ROOT / "sync_state.json"

NOTION_TOKEN = os.getenv("NOTION_TOKEN", "")
NOTION_ROOT_PAGE_ID = os.getenv("NOTION_ROOT_PAGE_ID", "1837455c17d080358725cd5e57fb92db")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "mokkoji_docs")

# 디코더(LLM). 프로바이더는 ollama / anthropic / openai / google_genai.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "exaone3.5:2.4b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# 주 디코더가 쿼터를 넘기거나 장애일 때 대신 쓸 디코더. 비우면 폴백하지 않는다.
# API를 주 경로로 두고 로컬 모델을 안전망으로 두는 구성을 위한 것이다.
FALLBACK_LLM_PROVIDER = os.getenv("FALLBACK_LLM_PROVIDER", "")
FALLBACK_LLM_MODEL = os.getenv("FALLBACK_LLM_MODEL", "")

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
DISCORD_GUILD_ID = os.getenv("DISCORD_GUILD_ID", "")

# 인코더(임베딩). bge-m3만 sparse를 함께 내놓는다. 나머지는 dense 단독 검색이 된다.
# 벡터 차원은 컬렉션을 만들 때 모델에서 직접 재므로 여기에 적지 않는다.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "bge-m3")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBED_BATCH_SIZE = 8

CHUNK_MAX_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64
CHUNK_MIN_TOKENS = 128

SEARCH_PREFETCH_LIMIT = 30
SEARCH_TOP_K = 8


def normalize_page_id(page_id: str) -> str:
    """노션 ID를 하이픈 없는 32자 소문자로 통일한다. URL, 하이픈 포함 UUID 모두 허용."""
    cleaned = page_id.strip().rstrip("/").split("/")[-1].split("?")[0]
    cleaned = cleaned.replace("-", "").lower()
    return cleaned[-32:]


# 인덱싱에서 제외할 페이지. 해당 페이지와 하위 트리 전체를 스킵한다.
# 노션 integration을 이 페이지들에 연결하지 않는 것이 1차 방어이고, 이 목록은 2차 방어다.
BLOCKED_PAGES: dict[str, str] = {
    "2bd7455c17d0818f85f0ef9c870f19ea": "공동계정",
    "2ef7455c17d080f7b871f9dfe33ab074": "프론트 계정",
    "3007455c17d08065b1ffcdecb2268af3": "백엔드 계정",
    "3bd7455c17d080c89f3fd442224dde65": "Admin 계정",
    "2b77455c17d080159397cc0d4a7f4ce9": "모꼬지 카드 정보",
    "2bb7455c17d080a98693d3aa81799dde": "회비 관리 내역",
    "1927455c17d08170a9e4c70c4de7eacd": "백엔드 서버 정보",
    "3007455c17d0808382ecfad47c2f55fc": "백엔드 yml 파일",
    "3017455c17d080f3a544c291fe0513de": "2026년 상반기 one-on-one 템플릿",
}

BLOCKED_PAGE_IDS = {normalize_page_id(page_id) for page_id in BLOCKED_PAGES}

# 민감하지는 않지만 팀 문서가 아니라 모꼬지 서비스가 다루는 데이터라 제외한다.
# 동아리 × 학기 조합으로 수백~수천 개의 짧고 비슷한 페이지가 생겨 팀 문서 검색을 희석시킨다.
EXCLUDED_PAGES: dict[str, str] = {
    "2c57455c17d08095b440d108cc2b5eeb": "모집글 (동아리 모집 데이터베이스)",
    "2587455c17d0801a8a8fc20cc463c973": "데이터 백업 (8/24일 기준)",
}

EXCLUDED_PAGE_IDS = {normalize_page_id(page_id) for page_id in EXCLUDED_PAGES}

# 제목에 아래 문자열이 들어가면 블랙리스트 ID에 없더라도 제외한다.
# 새로 생긴 민감 페이지가 ID 목록에 반영되기 전까지 막아주는 안전망.
BLOCKED_TITLE_KEYWORDS = ("계정", "비밀번호", "카드 정보", "회비", "one-on-one", "1on1")

# 페이지는 인덱싱하되 이 속성만 뺀다. 팀 명단처럼 쓸모 있는 정보와 개인정보가 한 데이터베이스에
# 섞여 있을 때, 페이지째 차단하면 분야·역할까지 함께 잃는다.
BLOCKED_PROPERTY_KEYWORDS = ("번호", "학번", "연락처", "생년월일", "주소", "이메일", "email", "phone")


# 조상 경로에 이 문자열이 들어가면 해당 문서 유형으로 본다. 열거형 질의에서 종류를 거를 때 쓴다.
# "구회의록", "4차 스프린트 회의록"처럼 이름이 제각각이라 정확 일치로는 못 잡는다.
DOC_TYPE_KEYWORDS = ("회의록", "온보딩", "API 명세서", "QA")


# "4차 회의록" 같은 개별 회의 페이지 안에 박힌 데이터베이스 행은 안건이지 회의가 아니다.
# "4차 스프린트 회의록"(허브)은 사이에 다른 말이 끼므로 걸리지 않는다.
NESTED_MEETING_PATTERN = re.compile(r"^\d+차\s*회의록")


def classify_doc_type(ancestor_path: list[str], title: str) -> str:
    """문서 유형을 하나 고른다. 해당 없으면 빈 문자열."""
    haystack = " ".join([*ancestor_path, title])
    for keyword in DOC_TYPE_KEYWORDS:
        if keyword not in haystack:
            continue
        if keyword == "회의록" and any(
            NESTED_MEETING_PATTERN.match(ancestor.strip()) for ancestor in ancestor_path
        ):
            return ""
        return keyword
    return ""


# 파트를 나누는 근거는 노션 트리 위치다. "BE-FE 문서화"처럼 양쪽에 걸친 문서는
# 어느 쪽으로도 판정하지 않고 비워 둔다 — 필터에 걸리면 오히려 놓친다.
PART_KEYWORDS = (("프론트", "프론트엔드"), ("백엔드", "백엔드"), ("Server", "백엔드"))


def classify_part(ancestor_path: list[str], title: str) -> str:
    """문서가 속한 파트를 고른다. 해당 없으면 빈 문자열."""
    haystack = " ".join([*ancestor_path, title])
    for keyword, part in PART_KEYWORDS:
        if keyword in haystack:
            return part
    return ""


# 회의록 스크럼은 본문에 호칭만 적는다(`> `석준` : flyway 도입`). people 속성은 참석자 전원이라
# 담당자를 못 가리므로, 호칭을 노션 속성과 같은 표기로 펴서 청크에 따로 남긴다.
# 오른쪽은 노션 people 속성에 실제로 쓰인 표기 그대로다. 표기를 맞춰야 person과 people을
# 같은 값으로 걸 수 있다. "꽁이(혁수)"가 어색해 보여도 노션이 그렇게 갖고 있다.
SCRUM_MEMBERS = {
    "혁수": "꽁이(혁수)",
    "석준": "석준 허",
    "혜빈": "신혜빈",
    "금서": "안금서",
    "창우": "창우 정",
    "지우": "신지우",
    "성림": "림 성",
    "의진": "의진",
}


def resolve_member(nickname: str) -> str:
    """스크럼 호칭을 정식 표기로 편다. 모르는 호칭이면 빈 문자열."""
    name = nickname.strip().rstrip("님씨").strip()
    return SCRUM_MEMBERS.get(name, "")


def is_blocked_property(name: str) -> bool:
    """속성 이름이 민감 목록에 걸리면 True. 마크다운 렌더링에서 제외된다."""
    lowered = name.lower()
    return any(keyword.lower() in lowered for keyword in BLOCKED_PROPERTY_KEYWORDS)


def is_blocked(page_id: str, title: str) -> str | None:
    """건너뛸 사유를 반환한다. 수집 대상이면 None. 해당 페이지의 하위 트리 전체가 함께 빠진다."""
    normalized = normalize_page_id(page_id)
    if normalized in BLOCKED_PAGE_IDS:
        return "blocklist"
    if normalized in EXCLUDED_PAGE_IDS:
        return "excluded"
    lowered = title.lower()
    for keyword in BLOCKED_TITLE_KEYWORDS:
        if keyword.lower() in lowered:
            return f"keyword:{keyword}"
    return None
