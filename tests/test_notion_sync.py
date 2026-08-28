import json
import tempfile
import unittest
from pathlib import Path

from notion_sync import collect_pages, write_pages


def text_block(block_id, block_type, text, *, has_children=False):
    return {
        "id": block_id,
        "type": block_type,
        "has_children": has_children,
        block_type: {"rich_text": [{"plain_text": text}]},
    }


class FakeNotionClient:
    def __init__(self):
        self.children = {
            "root": [text_block("toggle", "toggle", "4차 스프린트", has_children=True)],
            "toggle": [
                {
                    "id": "database",
                    "type": "child_database",
                    "has_children": False,
                    "child_database": {"title": "4차 스프린트 회의록"},
                }
            ],
            "meeting": [
                text_block("heading", "heading_2", "Web"),
                text_block("done", "callout", "창우: 서비스 지연 이슈 개선"),
            ],
            "template": [text_block("empty", "paragraph", "")],
        }

    def block_children(self, block_id):
        return self.children[block_id]

    def database_data_sources(self, database_id):
        self.assert_equal(database_id, "database")
        return ["data-source"]

    def data_source_pages(self, data_source_id):
        self.assert_equal(data_source_id, "data-source")
        return [
            {
                "id": "meeting",
                "url": "https://notion.example/meeting",
                "last_edited_time": "2026-08-04T00:00:00.000Z",
                "properties": {
                    "제목": {
                        "type": "title",
                        "title": [{"plain_text": "16차 회의록"}],
                    }
                },
            },
            {
                "id": "template",
                "url": "https://notion.example/template",
                "last_edited_time": "2026-08-20T00:00:00.000Z",
                "properties": {
                    "제목": {
                        "type": "title",
                        "title": [{"plain_text": "19차 템플릿"}],
                    }
                },
            },
        ]

    @staticmethod
    def assert_equal(actual, expected):
        if actual != expected:
            raise AssertionError(f"{actual!r} != {expected!r}")


class NotionSyncTest(unittest.TestCase):
    def test_collects_database_pages_and_excludes_templates(self):
        pages = collect_pages(FakeNotionClient(), "root")

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].title, "16차 회의록")
        self.assertIn("## Web", pages[0].content)
        self.assertIn("> 창우: 서비스 지연 이슈 개선", pages[0].content)

    def test_writes_markdown_and_manifest(self):
        pages = collect_pages(FakeNotionClient(), "root")
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)

            count = write_pages(pages, output_dir)

            self.assertEqual(count, 1)
            markdown_files = list(output_dir.glob("*.md"))
            self.assertEqual(len(markdown_files), 1)
            markdown = markdown_files[0].read_text(encoding="utf-8")
            self.assertIn("notion_page_id: meeting", markdown)
            self.assertIn("# 16차 회의록", markdown)

            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["pages"]["meeting"]["title"], "16차 회의록")
            self.assertTrue(manifest["pages"]["meeting"]["content_hash"])


if __name__ == "__main__":
    unittest.main()
