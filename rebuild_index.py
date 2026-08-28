"""data 문서 전체로 영구 RAG 인덱스를 다시 만든다."""

from indexing import DATA_DIR, STORAGE_DIR, rebuild_persisted_index


def main() -> None:
    print(f"{DATA_DIR} 문서로 새 인덱스를 생성합니다.")
    rebuild_persisted_index()
    print(f"새 인덱스를 {STORAGE_DIR}에 저장했습니다.")


if __name__ == "__main__":
    main()
