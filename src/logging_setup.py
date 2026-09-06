"""파일 로깅 설정.

인덱싱이 도중에 죽으면(RAM 부족으로 강제 종료되는 일이 있다) 어디까지 했는지 알 방법이 없어서
파일에 진행 상황을 남긴다. 콘솔은 tqdm 진행 막대가 쓰고 있고 cp949라 특수문자를 못 찍으므로,
상세 기록은 UTF-8 파일에만 쓰고 콘솔에는 경고 이상만 내보낸다.
"""

import logging
from datetime import datetime

from src import config

LOG_DIR = config.PROJECT_ROOT / "logs"
LOGGER_NAME = "mokkoji-rag"


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def setup(task: str) -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"{task}-{datetime.now():%Y%m%d-%H%M%S}.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s"))
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console)

    print(f"로그: {path}")
    logger.info("=== %s 시작 ===", task)
    logger.info(
        "컬렉션=%s  인코더=%s/%s  청킹 max=%d min=%d overlap=%d",
        config.QDRANT_COLLECTION,
        config.EMBEDDING_PROVIDER,
        config.EMBEDDING_MODEL,
        config.CHUNK_MAX_TOKENS,
        config.CHUNK_MIN_TOKENS,
        config.CHUNK_OVERLAP_TOKENS,
    )
    return logger
