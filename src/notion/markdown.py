"""노션 블록 트리를 마크다운으로 변환한다.

client.fetch_block_tree()가 각 블록에 넣어준 `_children` 키를 따라 재귀 렌더링한다.
"""

CHILD_INDENT = "  "

# 자식 블록을 들여쓰기해서 렌더링할 부모 타입. 나머지는 같은 레벨로 이어 붙인다.
INDENTING_TYPES = {"bulleted_list_item", "numbered_list_item", "to_do", "toggle"}

SKIPPED_TYPES = {"table_of_contents", "breadcrumb", "divider", "unsupported"}


def render_rich_text(rich_text: list[dict]) -> str:
    parts = []
    for item in rich_text:
        content = item.get("plain_text", "")
        if not content:
            continue

        annotations = item.get("annotations", {})
        if annotations.get("code"):
            content = f"`{content}`"
        if annotations.get("bold"):
            content = f"**{content}**"
        if annotations.get("italic"):
            content = f"*{content}*"
        if annotations.get("strikethrough"):
            content = f"~~{content}~~"

        href = item.get("href")
        if href:
            content = f"[{content}]({href})"

        parts.append(content)
    return "".join(parts)


def _text_of(block: dict) -> str:
    payload = block.get(block["type"], {})
    return render_rich_text(payload.get("rich_text", []))


def _render_table(block: dict) -> list[str]:
    rows = block.get("_children", [])
    if not rows:
        return []

    rendered_rows = []
    for row in rows:
        cells = row.get("table_row", {}).get("cells", [])
        rendered_rows.append([render_rich_text(cell) for cell in cells])

    lines = ["| " + " | ".join(rendered_rows[0]) + " |"]
    if block.get("table", {}).get("has_column_header"):
        lines.append("| " + " | ".join("---" for _ in rendered_rows[0]) + " |")
    for cells in rendered_rows[1:]:
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _render_block(block: dict, numbered_position: int) -> tuple[list[str], bool]:
    """(이 블록이 만드는 줄들, 자식을 이미 소비했는지)를 반환한다."""
    block_type = block["type"]

    if block_type in SKIPPED_TYPES:
        return (["---"] if block_type == "divider" else []), False

    if block_type == "paragraph":
        text = _text_of(block)
        return ([text] if text else []), False

    if block_type in ("heading_1", "heading_2", "heading_3"):
        level = int(block_type[-1])
        return [f"{'#' * level} {_text_of(block)}"], False

    if block_type == "bulleted_list_item":
        return [f"- {_text_of(block)}"], False

    if block_type == "numbered_list_item":
        return [f"{numbered_position}. {_text_of(block)}"], False

    if block_type == "to_do":
        marker = "x" if block["to_do"].get("checked") else " "
        return [f"- [{marker}] {_text_of(block)}"], False

    if block_type == "toggle":
        return [f"- {_text_of(block)}"], False

    if block_type == "quote":
        return [f"> {_text_of(block)}"], False

    if block_type == "callout":
        return [f"> {_text_of(block)}"], False

    if block_type == "code":
        language = block["code"].get("language", "")
        body = render_rich_text(block["code"].get("rich_text", []))
        return [f"```{language}", body, "```"], False

    if block_type == "equation":
        return [f"$$ {block['equation'].get('expression', '')} $$"], False

    if block_type == "table":
        return _render_table(block), True

    if block_type == "child_page":
        return [f"- (하위 문서) {block['child_page'].get('title', '')}"], False

    if block_type == "child_database":
        return [f"- (하위 데이터베이스) {block['child_database'].get('title', '')}"], False

    if block_type in ("image", "file", "pdf", "video", "audio"):
        payload = block.get(block_type, {})
        caption = render_rich_text(payload.get("caption", []))
        name = payload.get("name") or caption or block_type
        return [f"[첨부: {name}]"], False

    if block_type in ("bookmark", "embed", "link_preview"):
        url = block.get(block_type, {}).get("url", "")
        caption = render_rich_text(block.get(block_type, {}).get("caption", []))
        return [f"- {caption or '링크'}: {url}"], False

    if block_type == "link_to_page":
        return [], False

    # column_list, column, synced_block 등 컨테이너는 자식만 이어 붙인다.
    return [], False


def render_blocks(blocks: list[dict], depth: int = 0) -> list[str]:
    lines: list[str] = []
    numbered_position = 0

    for block in blocks:
        if block["type"] == "numbered_list_item":
            numbered_position += 1
        else:
            numbered_position = 0

        own_lines, children_consumed = _render_block(block, numbered_position)
        lines.extend(own_lines)

        children = block.get("_children", [])
        if children and not children_consumed:
            child_lines = render_blocks(children, depth + 1)
            if block["type"] in INDENTING_TYPES:
                child_lines = [CHILD_INDENT + line if line else line for line in child_lines]
            lines.extend(child_lines)

    return lines


def blocks_to_markdown(blocks: list[dict]) -> str:
    lines = render_blocks(blocks)

    # 연속된 빈 줄을 하나로 줄인다.
    compacted: list[str] = []
    for line in lines:
        if not line.strip() and (not compacted or not compacted[-1].strip()):
            continue
        compacted.append(line)

    return "\n".join(compacted).strip()
