"""노션 페이지 트리를 순회하며 마크다운 문서를 수집한다."""

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx
from notion_client import Client
from notion_client.errors import APIResponseError, HTTPResponseError, RequestTimeoutError

from src import config
from src.notion.markdown import blocks_to_markdown, render_rich_text

# 노션은 초당 3요청 평균을 허용한다. 여유를 둬서 간격을 잡는다.
MIN_REQUEST_INTERVAL = 0.36
MAX_RETRIES = 7
CACHE_SAVE_INTERVAL = 25

# 자식 블록을 따라 내려가지 않는 타입. 별도 페이지로 수집되므로 여기서는 스텁만 남긴다.
LEAF_CONTAINER_TYPES = {"child_page", "child_database"}


@dataclass
class NotionPage:
    page_id: str
    title: str
    ancestor_path: list[str]
    url: str
    last_edited_time: str
    markdown: str
    # 검색 필터용으로 뽑아 둔 속성. 본문에도 텍스트로 들어가지만 여기 것은 타입이 살아 있다.
    properties: dict = field(default_factory=dict)

    @property
    def breadcrumb(self) -> str:
        return " > ".join([*self.ancestor_path, self.title])


@dataclass
class WalkStats:
    visited: int = 0
    skipped: list[tuple[str, str, str]] = field(default_factory=list)  # (page_id, title, reason)
    cache_hits: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)  # (page_id, 사유)


class NotionCollector:
    def __init__(self, token: str | None = None, use_cache: bool = True):
        token = token or config.NOTION_TOKEN
        if not token:
            raise RuntimeError("NOTION_TOKEN이 비어 있습니다. .env를 확인하세요.")

        # SDK 기본값(2025-09-03)을 쓴다. 이 버전부터 데이터베이스 조회는 data source 경유다.
        self.client = Client(auth=token)
        self.stats = WalkStats()
        self._last_request_at = 0.0
        self._visited_ids: set[str] = set()

        self.use_cache = use_cache
        self._cache_path = config.CACHE_DIR / "pages.json"
        self._cache: dict[str, dict] = {}
        if use_cache and self._cache_path.exists():
            self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))

    # --- 저수준 요청 ---

    def _call(self, fn, **kwargs):
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)

        for attempt in range(MAX_RETRIES):
            try:
                result = fn(**kwargs)
                self._last_request_at = time.monotonic()
                return result
            except (
                APIResponseError,
                HTTPResponseError,
                RequestTimeoutError,
                # 수백 요청을 보내는 동안 연결이 끊기는 일이 실제로 일어난다.
                httpx.TransportError,
            ) as error:
                status = getattr(error, "status", None)
                code = getattr(error, "code", None)
                retryable = (
                    isinstance(error, httpx.TransportError)
                    or code == "rate_limited"
                    or (status is not None and status >= 500)
                )
                if not retryable or attempt == MAX_RETRIES - 1:
                    raise
                backoff = 2**attempt
                reason = code or status or type(error).__name__
                print(f"  요청 실패({reason}) — {backoff}초 후 재시도")
                time.sleep(backoff)
                self._last_request_at = time.monotonic()

    def _paginate(self, fn, **kwargs) -> Iterator[dict]:
        cursor = None
        while True:
            response = self._call(fn, **kwargs, start_cursor=cursor) if cursor else self._call(fn, **kwargs)
            yield from response.get("results", [])
            if not response.get("has_more"):
                return
            cursor = response.get("next_cursor")

    # --- 블록 트리 ---

    def fetch_block_tree(self, block_id: str) -> list[dict]:
        blocks = list(self._paginate(self.client.blocks.children.list, block_id=block_id, page_size=100))
        for block in blocks:
            if block.get("has_children") and block["type"] not in LEAF_CONTAINER_TYPES:
                block["_children"] = self.fetch_block_tree(block["id"])
        return blocks

    def collect_child_refs(self, blocks: list[dict]) -> tuple[list[str], list[str]]:
        """블록 트리에서 하위 페이지 ID와 하위 데이터베이스 ID를 모은다."""
        page_ids: list[str] = []
        database_ids: list[str] = []
        for block in blocks:
            if block["type"] == "child_page":
                page_ids.append(block["id"])
            elif block["type"] == "child_database":
                database_ids.append(block["id"])
            else:
                child_pages, child_databases = self.collect_child_refs(block.get("_children", []))
                page_ids.extend(child_pages)
                database_ids.extend(child_databases)
        return page_ids, database_ids

    # --- 페이지 메타 ---

    @staticmethod
    def _title_of(page: dict) -> str:
        for prop in page.get("properties", {}).values():
            if prop.get("type") == "title":
                return render_rich_text(prop["title"]) or "(제목 없음)"
        return "(제목 없음)"

    @staticmethod
    def _extract_properties(page: dict) -> dict:
        """필터에 쓸 속성을 타입 기준으로 뽑는다.

        노션은 DB마다 속성 이름이 제각각이라(날짜/작성일/회의일…) 이름으로는 통일이 안 된다.
        타입은 노션이 강제하므로 타입으로 잡는다. 날짜는 페이지마다 하나만 쓰므로 첫 것만 취한다.
        """
        extracted: dict = {}
        for name, prop in page.get("properties", {}).items():
            if config.is_blocked_property(name):
                continue

            prop_type = prop.get("type")
            if prop_type == "date" and "doc_date" not in extracted:
                start = (prop["date"] or {}).get("start")
                if start:
                    extracted["doc_date"] = start[:10]
            elif prop_type == "people":
                people = [person.get("name", "") for person in prop["people"] if person.get("name")]
                if people:
                    extracted.setdefault("people", []).extend(people)

        return extracted

    @staticmethod
    def _render_properties(page: dict) -> str:
        """데이터베이스 행의 속성을 마크다운 목록으로 렌더링한다. 회의록의 날짜·참석자 등이 여기 담긴다."""
        lines = []
        for name, prop in page.get("properties", {}).items():
            prop_type = prop.get("type")
            if prop_type == "title":
                continue
            if config.is_blocked_property(name):
                continue

            value = ""
            if prop_type == "rich_text":
                value = render_rich_text(prop["rich_text"])
            elif prop_type in ("select", "status"):
                value = (prop[prop_type] or {}).get("name", "")
            elif prop_type == "multi_select":
                value = ", ".join(option["name"] for option in prop["multi_select"])
            elif prop_type == "people":
                value = ", ".join(person.get("name", "") for person in prop["people"])
            elif prop_type == "date":
                date_value = prop["date"] or {}
                value = date_value.get("start", "")
                if date_value.get("end"):
                    value += f" ~ {date_value['end']}"
            elif prop_type == "number":
                value = "" if prop["number"] is None else str(prop["number"])
            elif prop_type == "checkbox":
                value = "예" if prop["checkbox"] else "아니오"
            elif prop_type in ("url", "email", "phone_number"):
                value = prop[prop_type] or ""

            if value:
                lines.append(f"- {name}: {value}")

        return "\n".join(lines)

    # --- 순회 ---

    def walk(self, root_page_id: str | None = None) -> Iterator[NotionPage]:
        root_page_id = root_page_id or config.NOTION_ROOT_PAGE_ID
        try:
            yield from self._walk_page(root_page_id, ancestor_path=[])
        finally:
            self._save_cache()

    def _walk_page(
        self, page_id: str, ancestor_path: list[str], page: dict | None = None
    ) -> Iterator[NotionPage]:
        normalized = config.normalize_page_id(page_id)
        if normalized in self._visited_ids:
            return
        self._visited_ids.add(normalized)

        # 데이터베이스 행은 query 응답이 이미 전체 페이지 객체다. 다시 조회하지 않는다.
        if page is None:
            page = self._call(self.client.pages.retrieve, page_id=page_id)
        if page.get("archived") or page.get("in_trash"):
            return

        title = self._title_of(page)
        reason = config.is_blocked(page_id, title)
        if reason:
            self.stats.skipped.append((normalized, title, reason))
            return

        last_edited_time = page.get("last_edited_time", "")
        # 속성은 pages.retrieve 응답에 들어 있다. 캐시 적중이어도 추가 요청 없이 얻는다.
        properties = self._extract_properties(page)
        cached = self._cache.get(normalized)

        if cached and cached.get("last_edited_time") == last_edited_time:
            self.stats.cache_hits += 1
            markdown = cached["markdown"]
            child_page_ids = cached["child_page_ids"]
            child_database_ids = cached["child_database_ids"]
        else:
            blocks = self.fetch_block_tree(page["id"])
            body = blocks_to_markdown(blocks)
            properties = self._render_properties(page)
            markdown = f"{properties}\n\n{body}".strip() if properties else body
            child_page_ids, child_database_ids = self.collect_child_refs(blocks)
            self._cache[normalized] = {
                "last_edited_time": last_edited_time,
                "markdown": markdown,
                "child_page_ids": child_page_ids,
                "child_database_ids": child_database_ids,
            }

        self.stats.visited += 1
        # 수집이 길어서 중간에 끊길 수 있다. 주기적으로 캐시를 남겨 재시작 비용을 줄인다.
        if self.stats.visited % CACHE_SAVE_INTERVAL == 0:
            self._save_cache()

        yield NotionPage(
            page_id=normalized,
            title=title,
            ancestor_path=ancestor_path,
            url=page.get("url", ""),
            last_edited_time=last_edited_time,
            markdown=markdown,
            properties=properties,
        )

        child_ancestor_path = [*ancestor_path, title]
        for child_page_id in child_page_ids:
            yield from self._walk_child(self._walk_page(child_page_id, child_ancestor_path), child_page_id)
        for database_id in child_database_ids:
            yield from self._walk_child(self._walk_database(database_id, child_ancestor_path), database_id)

    def _walk_child(self, walker: Iterator[NotionPage], child_id: str) -> Iterator[NotionPage]:
        """하위 트리 하나가 실패해도 순회 전체를 중단시키지 않는다.

        네트워크 오류로 페이지 하나를 못 읽으면 예외가 walk() 밖으로 전파되어
        아직 방문하지 않은 페이지가 통째로 유실됐다. 실패는 기록만 하고 넘어간다.
        실패한 페이지는 캐시에 남지 않으므로 다음 실행에서 자동으로 재시도된다.
        """
        try:
            yield from walker
        except Exception as error:
            self.stats.failed.append(
                (config.normalize_page_id(child_id), f"{type(error).__name__}: {error}")
            )
            print(f"  하위 트리 수집 실패({type(error).__name__}) — 건너뜀: {child_id}")

    def _walk_database(self, database_id: str, ancestor_path: list[str]) -> Iterator[NotionPage]:
        normalized = config.normalize_page_id(database_id)
        if normalized in self._visited_ids:
            return
        self._visited_ids.add(normalized)

        try:
            database = self._call(self.client.databases.retrieve, database_id=database_id)
        except APIResponseError as error:
            print(f"  데이터베이스 조회 실패({error.code}): {database_id}")
            return

        title = render_rich_text(database.get("title", [])) or "(제목 없는 데이터베이스)"
        reason = config.is_blocked(database_id, title)
        if reason:
            self.stats.skipped.append((normalized, title, reason))
            return

        child_ancestor_path = [*ancestor_path, title]
        for data_source in database.get("data_sources", []):
            rows = self._paginate(
                self.client.data_sources.query, data_source_id=data_source["id"], page_size=100
            )
            for row in rows:
                yield from self._walk_page(row["id"], child_ancestor_path, page=row)

    def _save_cache(self) -> None:
        if not self.use_cache:
            return
        config.CACHE_DIR.mkdir(exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")
