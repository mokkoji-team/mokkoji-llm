"""Notion 회의록을 로컬 Markdown 문서로 동기화한다."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


API_URL = "https://api.notion.com/v1"
DEFAULT_VERSION = "2026-03-11"
OUTPUT_DIR = Path(__file__).parent / "data" / "notion"
EXCLUDED_TITLE_KEYWORDS = ("템플릿", "임시 저장소")


class NotionAPIError(RuntimeError):
    """Notion API 요청이 실패했음을 나타낸다."""


@dataclass(frozen=True)
class SyncedPage:
    page_id: str
    title: str
    url: str
    last_edited_time: str
    content: str


class NotionClient:
    """M4에 필요한 읽기 전용 Notion API 호출만 담당한다."""

    def __init__(self, token: str, version: str = DEFAULT_VERSION):
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": version,
            "Content-Type": "application/json",
        }

    def request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{API_URL}{path}", data=body, headers=self.headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise NotionAPIError(
                f"Notion API 요청 실패 ({error.code} {path}): {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise NotionAPIError(f"Notion API 연결 실패 ({path}): {error.reason}") from error

    def paginated(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            if method == "GET":
                separator = "&" if "?" in path else "?"
                cursor_query = f"{separator}start_cursor={cursor}" if cursor else ""
                response = self.request(method, f"{path}{cursor_query}")
            else:
                request_payload = dict(payload or {})
                if cursor:
                    request_payload["start_cursor"] = cursor
                response = self.request(method, path, request_payload)
            results.extend(response.get("results", []))
            if not response.get("has_more"):
                return results
            cursor = response.get("next_cursor")

    def block_children(self, block_id: str) -> list[dict[str, Any]]:
        return self.paginated("GET", f"/blocks/{block_id}/children?page_size=100")

    def database_data_sources(self, database_id: str) -> list[str]:
        database = self.request("GET", f"/databases/{database_id}")
        return [item["id"] for item in database.get("data_sources", [])]

    def data_source_pages(self, data_source_id: str) -> list[dict[str, Any]]:
        return self.paginated(
            "POST", f"/data_sources/{data_source_id}/query", {"page_size": 100}
        )


def rich_text_plain_text(items: Iterable[dict[str, Any]]) -> str:
    return "".join(item.get("plain_text", "") for item in items)


def page_title(page: dict[str, Any]) -> str:
    for value in page.get("properties", {}).values():
        if value.get("type") == "title":
            return rich_text_plain_text(value.get("title", [])) or "제목 없음"
    return "제목 없음"


def block_to_markdown(block: dict[str, Any]) -> str:
    block_type = block.get("type", "")
    value = block.get(block_type, {})
    text = rich_text_plain_text(value.get("rich_text", []))
    prefixes = {
        "heading_1": "# ",
        "heading_2": "## ",
        "heading_3": "### ",
        "bulleted_list_item": "- ",
        "numbered_list_item": "1. ",
        "to_do": "- [x] " if value.get("checked") else "- [ ] ",
        "quote": "> ",
        "callout": "> ",
        "code": "```\n",
    }
    if block_type == "divider":
        return "---"
    if block_type == "code":
        return f"```\n{text}\n```"
    return f"{prefixes.get(block_type, '')}{text}".rstrip()


def render_blocks(client: NotionClient, block_id: str) -> str:
    lines: list[str] = []
    for block in client.block_children(block_id):
        rendered = block_to_markdown(block)
        if rendered:
            lines.append(rendered)
        if block.get("has_children") and block.get("type") != "child_database":
            nested = render_blocks(client, block["id"])
            if nested:
                lines.append(nested)
    return "\n\n".join(lines).strip()


def discover_data_sources(client: NotionClient, block_id: str) -> list[str]:
    discovered: list[str] = []
    for block in client.block_children(block_id):
        if block.get("type") == "child_database":
            discovered.extend(client.database_data_sources(block["id"]))
        elif block.get("has_children"):
            discovered.extend(discover_data_sources(client, block["id"]))
    return list(dict.fromkeys(discovered))


def collect_pages(client: NotionClient, root_page_id: str) -> list[SyncedPage]:
    pages: list[SyncedPage] = []
    for data_source_id in discover_data_sources(client, root_page_id):
        for page in client.data_source_pages(data_source_id):
            title = page_title(page)
            if any(keyword in title for keyword in EXCLUDED_TITLE_KEYWORDS):
                continue
            content = render_blocks(client, page["id"])
            if not content:
                continue
            pages.append(
                SyncedPage(
                    page_id=page["id"],
                    title=title,
                    url=page.get("url", ""),
                    last_edited_time=page.get("last_edited_time", ""),
                    content=content,
                )
            )
    return pages


def safe_filename(title: str, page_id: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z가-힣._-]+", "-", title).strip("-.")
    return f"{normalized or '회의록'}-{page_id.replace('-', '')[:8]}.md"


def write_pages(pages: Iterable[SyncedPage], output_dir: Path = OUTPUT_DIR) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"pages": {}}
    written_names: set[str] = set()

    for page in pages:
        filename = safe_filename(page.title, page.page_id)
        written_names.add(filename)
        body = (
            "---\n"
            f"notion_page_id: {page.page_id}\n"
            f"title: {json.dumps(page.title, ensure_ascii=False)}\n"
            f"url: {page.url}\n"
            f"last_edited_time: {page.last_edited_time}\n"
            "---\n\n"
            f"# {page.title}\n\n{page.content.strip()}\n"
        )
        (output_dir / filename).write_text(body, encoding="utf-8")
        manifest["pages"][page.page_id] = {
            "title": page.title,
            "file": filename,
            "url": page.url,
            "last_edited_time": page.last_edited_time,
            "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        }

    for old_file in output_dir.glob("*.md"):
        if old_file.name not in written_names:
            old_file.unlink()
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return len(manifest["pages"])


def main() -> int:
    token = os.getenv("NOTION_TOKEN")
    root_page_id = os.getenv("NOTION_ROOT_PAGE_ID")
    if not token or not root_page_id:
        print("NOTION_TOKEN과 NOTION_ROOT_PAGE_ID가 필요합니다.", file=sys.stderr)
        return 2

    client = NotionClient(token, os.getenv("NOTION_VERSION", DEFAULT_VERSION))
    try:
        count = write_pages(collect_pages(client, root_page_id))
    except NotionAPIError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"Notion 회의록 {count}개를 {OUTPUT_DIR}에 동기화했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
