"""LangChain 전환 전에 인덱싱해 둔 컬렉션의 payload를 새 구조로 옮긴다.

LangChain은 메타데이터를 payload["metadata"] 아래에서 찾는데, 전환 전에는 평면으로 저장했다.
벡터는 그대로 두고 payload만 다시 쓰므로 CPU 임베딩을 다시 돌리지 않아도 된다.

    python -m scripts.migrate_payload --dry-run
    python -m scripts.migrate_payload

전환 후에 인덱싱한 컬렉션에는 쓸 일이 없다. 한 번 돌리고 나면 지워도 되는 스크립트다.
"""

import argparse

from qdrant_client import models
from tqdm import tqdm

from src import config
from src.retrieval.store import CONTENT_KEY, METADATA_KEY, PAGE_ID_FIELD, get_client

OLD_PAGE_ID_FIELD = "page_id"


def scroll_all(client):
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        yield from points
        if offset is None:
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="payload 구조 마이그레이션")
    parser.add_argument("--dry-run", action="store_true", help="바꾸지 않고 대상만 센다")
    args = parser.parse_args()

    client = get_client()
    if not client.collection_exists(config.QDRANT_COLLECTION):
        print(f"컬렉션이 없습니다: {config.QDRANT_COLLECTION}")
        return

    points = list(scroll_all(client))
    stale = [point for point in points if METADATA_KEY not in point.payload]
    print(f"전체 {len(points)}개 / 변환 대상 {len(stale)}개")

    if not stale:
        print("이미 새 구조입니다.")
        return

    if args.dry_run:
        sample = stale[0]
        print(f"예시 payload 키: {sorted(sample.payload.keys())}")
        print(f"  → {{'{CONTENT_KEY}': ..., '{METADATA_KEY}': {{나머지 전부}}}}")
        return

    for point in tqdm(stale, desc="payload 변환", unit="포인트"):
        payload = dict(point.payload)
        text = payload.pop(CONTENT_KEY, "")
        client.overwrite_payload(
            collection_name=config.QDRANT_COLLECTION,
            payload={CONTENT_KEY: text, METADATA_KEY: payload},
            points=[point.id],
            wait=False,
        )

    client.create_payload_index(
        collection_name=config.QDRANT_COLLECTION,
        field_name=PAGE_ID_FIELD,
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True,
    )
    client.delete_payload_index(
        collection_name=config.QDRANT_COLLECTION,
        field_name=OLD_PAGE_ID_FIELD,
        wait=True,
    )

    remaining = [p for p in scroll_all(client) if METADATA_KEY not in p.payload]
    print(f"\n변환 완료. 남은 구형 포인트: {len(remaining)}개")
    print(f"payload 인덱스: {PAGE_ID_FIELD}")


if __name__ == "__main__":
    main()
