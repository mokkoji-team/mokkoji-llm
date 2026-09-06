# 모꼬지 노션 문서 RAG 챗봇

모꼬지 팀 노션 워크스페이스를 자연어로 검색하는 디스코드 봇. 전부 로컬에서 돈다.

```
Notion API ──► 마크다운 ──► 계층 청킹 ──► BGE-m3 임베딩 ──► Qdrant
                                                              │
Discord Bot ──► 하이브리드 검색(dense+sparse, RRF) ───────────┘
                            │
                            ▼
                    Ollama(로컬 CPU) ──► 답변 + 출처 링크
```

## 준비

### 1. 노션 integration

1. https://www.notion.so/my-integrations 에서 internal integration 생성 → secret 복사
2. `팀 - 모꼬지` 페이지 우상단 `···` → `연결` → 만든 integration 추가 (하위 페이지에 상속됨)
3. **민감 페이지는 연결을 개별 해제한다** — 공동계정, 프론트/백/Admin 계정, 백엔드 yml, 백엔드 서버 정보, 카드 정보, 회비 관리 내역, 멤버, one-on-one

> 워크스페이스 소유자 권한이 없으면 integration 생성/연결이 막힌다. 그 경우 팀 리더에게 요청할 것.
>
> integration 미연결이 1차 방어이고, [src/config.py](src/config.py)의 `BLOCKED_PAGES` 목록이 2차 방어다.

[src/config.py](src/config.py)에는 성격이 다른 두 목록이 있다.

- `BLOCKED_PAGES` — 민감 정보. 계정, 카드, 회비, 개인 평가
- `EXCLUDED_PAGES` — 민감하진 않지만 팀 문서가 아닌 것. 동아리 모집글 DB, 데이터 백업

둘 다 **하위 트리 전체**를 건너뛴다. 모집글 DB는 동아리 × 학기 조합으로 수백 개의 짧고 비슷한 페이지를 만들어
팀 문서 검색을 희석시키므로 제외했다. 동아리 정보가 필요하면 서비스 DB를 보는 게 맞다.

### 2. Ollama

https://ollama.com/download 에서 설치한 뒤:

```powershell
ollama pull qwen3:4b
```

### 3. 파이썬 환경

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e .
```

### 4. 설정

```powershell
Copy-Item .env.example .env
```

`.env`에 `NOTION_TOKEN`과 `DISCORD_TOKEN`을 채운다.

### 5. Qdrant

Docker Desktop을 켠 뒤:

```powershell
docker compose up -d
```

http://localhost:6333/dashboard 로 확인.

## 매번 실행 순서

세 프로세스가 떠 있어야 봇이 동작한다.

```powershell
# 1. Qdrant (Docker Desktop이 먼저 켜져 있어야 함)
docker compose up -d

# 2. Ollama 서버 — 모델이 D:에 있으므로 환경변수를 함께 넘긴다
$env:OLLAMA_MODELS = "D:\Ollama\models"
D:\Ollama\ollama.exe serve

# 3. 디스코드 봇 (새 터미널)
.\.venv\Scripts\python.exe -m src.bot.discord_bot
```

### 중단된 인덱싱 이어하기

최초 인덱싱이 중간에 끊겼다면 `scripts.index`를 다시 돌리지 말고 **`scripts.sync`를 쓴다.**
`index`는 만나는 페이지를 전부 다시 임베딩하지만, `sync`는 Qdrant에 저장된 `last_edited_time`과
비교해 이미 들어간 페이지를 건너뛴다. 비용의 대부분이 임베딩이라 차이가 크다.

```powershell
.\.venv\Scripts\python.exe -m scripts.sync
```

## 사용

```powershell
# 수집 대상 확인 (Qdrant 없이도 동작, 민감 페이지가 안 걸리는지 먼저 볼 것)
python -m scripts.index --dry-run

# 청크 수까지 확인
python -m scripts.index --dry-run --with-chunks

# 전체 인덱싱 (최초 1회, 수십 분 소요)
python -m scripts.index

# 변경분만 재인덱싱
python -m scripts.sync

# CLI로 질문
python -m scripts.ask "프론트엔드 기술 스택을 왜 이렇게 골랐어?"
python -m scripts.ask --retrieval-only "창의학기제 자료 어디 있어?"

# 검색 품질 평가
python -m src.eval.run_eval --show-misses

# 디스코드 봇 실행
python -m src.bot.discord_bot
```

노션 페이지 캐시는 `.cache/pages.json`에 남는다. `--no-cache`로 무시할 수 있다.

## 모델 선정 결과

외장 GPU 없이 CPU로 추론한다 (Intel Iris Xe 내장 GPU는 Ollama가 제외한다). 실측 비교:

세 유형(근거 종합 / 사실 조회 / 근거 없음 거절)으로 비교한 결과:

| 모델 | 평균 응답 | 근거 종합 | 거절 |
|---|---|---|---|
| `exaone3.5:2.4b` | 7.7초 | **근거 전부 반영, 자연스러운 한국어** | 정확 |
| `qwen3:1.7b` | 5.1초 | 일부 근거 누락, 문장 어색 | 정확 |
| `qwen3:4b` | 206.7초 | **답변 실패** — 3501자를 사고만 하다 토큰 상한 도달 | — |

**`exaone3.5:2.4b`를 쓴다.** 2.6초 더 느리지만 "왜 그렇게 정했어?" 류 질문의 답 품질 차이가 크고,
사고 모델이 아니라서 아래 `think` 문제도 없다.

### Qwen3 계열을 쓸 때의 함정

Qwen3는 하이브리드 추론 모델이라 사고 과정을 꺼야 하는데, `think=False`가 모델마다 다르게 동작한다.

- `qwen3:1.7b` — 사고를 실제로 끈다. 정상
- `qwen3:4b` — 사고를 끄지 않고 `content`로 흘려보낸다. 답변이 사고 과정으로 오염된다

`think`를 아예 지정하지 않으면 사고가 `thinking` 필드로 분리되지만, 사고 토큰이 출력 예산을 다 먹어
CPU에서는 답변까지 도달하지 못한다. 그래서 [src/generation/llm.py](src/generation/llm.py)는
`THINKING_MODEL_PREFIXES`에 해당하는 모델에만 `think=False`를 보낸다.

**더 큰 Qwen3 모델로 바꾸지 말 것.** 이 하드웨어에서는 느린 게 아니라 아예 동작하지 않는다.

느리면 순서대로 시도:

1. [src/generation/llm.py](src/generation/llm.py)의 `MAX_OUTPUT_TOKENS` 축소
2. [src/config.py](src/config.py)의 `SEARCH_TOP_K` 축소 (컨텍스트가 짧아지면 prefill이 빨라진다)
3. 더 작은 비추론 모델로 교체 (`qwen3:1.7b`, `gemma3:1b`)

### 모델 저장 위치

Ollama 기본값은 `C:\Users\<user>\.ollama\models`다. C: 드라이브 여유가 적으면 사용자 환경변수
`OLLAMA_MODELS`를 `D:\Ollama\models` 같은 경로로 지정하고 `ollama serve`를 다시 띄운다.

## 검색 품질 개선 순서

1. 청킹 파라미터 — [src/config.py](src/config.py)의 `CHUNK_MAX_TOKENS`, `CHUNK_MIN_TOKENS`, `CHUNK_OVERLAP_TOKENS`
2. `SEARCH_PREFETCH_LIMIT` / `SEARCH_TOP_K`
3. 프롬프트 — [src/generation/prompt.py](src/generation/prompt.py)
4. 리랭커 추가 (Recall@10은 높은데 Recall@5가 낮을 때만)
5. 임베딩 파인튜닝 (모꼬지 도메인 용어를 못 잡을 때)

각 단계마다 `python -m src.eval.run_eval`로 수치를 확인하고 넘어갈 것.

> `SEARCH_TOP_K`는 예외다. [run_eval.py](src/eval/run_eval.py)가 `top_k=10`으로 고정해 부르므로 이 값을 바꿔도
> 수치가 안 움직인다. 대신 Recall@5와 Recall@10을 비교해서 `SEARCH_TOP_K`를 얼마로 둘지 정하는 용도로 쓴다.
> 둘이 같으면 top_k를 늘려도 얻는 게 없다는 뜻이다.

### 파트 필터

질문에 "프론트" 또는 "백엔드"가 들어가면 [search.py](src/retrieval/search.py)가 해당 파트로 좁힌 검색을 한 번 더 돌려
전체 검색 결과와 번갈아 섞는다. 파트는 노션 트리 위치로 판정해 `part` payload에 넣어 둔다([config.py](src/config.py)의 `classify_part`).

청크 앞에 경로 헤더가 붙는 구조라 제목·경로에 없는 단어로 물으면 밀리는데, 파트는 경로에 이미 있는 정보라 필터로 되살릴 수 있다.
"프론트 돌리려면 env에 뭐 넣어야 해?"가 미검출에서 4위로 올라온 게 이 경우다.

**섞을 때 전체 검색 쪽을 먼저 놓는다.** 좁힌 쪽을 앞에 두면 이미 1위로 맞히던 질문 세 개가 2위로 밀렸다.
파트 필터는 전체 검색이 놓친 것을 건지는 보조 수단이지 주 경로가 아니다.

"서버"는 힌트에 넣지 않았다. 프론트엔드 문서화에도 서버 다운 문서가 있어 파트를 가르는 근거가 못 된다.

`PART_KEYWORDS`를 고친 뒤에는 재임베딩 없이 payload만 다시 쓰면 된다.

```powershell
python -m scripts.backfill_metadata --doc-type-only --dry-run
python -m scripts.backfill_metadata --doc-type-only
```

### 골든 세트

[src/eval/golden_set.jsonl](src/eval/golden_set.jsonl)에 37개가 들어 있다. 문서 제목이 아니라 **본문을 읽고** 만들었다.
제목 단어를 그대로 쓰면 키워드 매칭만으로 맞아 수치가 부풀려지므로, 제목에 없는 말로 묻는 질문을 일부러 섞었다.

유형은 사실 조회 19 / 이유 설명 7 / 위치 찾기 7 / 여러 문서 종합 4다.

한 줄 형식:

```json
{"question": "질문", "answer_page_ids": ["하이픈 없는 32자 페이지 ID"]}
```

**답이 여러 문서에 있으면 전부 적는다.** 노션에 같은 문서가 여러 경로에 복사돼 있거나(온보딩 자료가 회의록 하위에도 있다),
백엔드 스터디처럼 같은 주제를 사람마다 따로 쓴 경우가 흔하다. 하나만 적으면 검색이 맞는 문서를 물어와도 실패로 집계된다.

[src/eval/golden_set_draft.jsonl](src/eval/golden_set_draft.jsonl)에 같은 37개가 유형·근거 메모와 함께 남아 있다.
`golden_set_draft.tsv`는 노션 링크가 붙은 검수용 표다.

문항을 늘릴 때는 **디스코드에서 팀원이 실제로 물어본 말**을 먼저 긁어오는 게 좋다. 상상해서 쓰면 문서 어휘가 섞여 들어간다.
37개는 한 문항이 2.7%p라 1문항 차이는 노이즈다. 2문항(5.4%p) 이상 움직여야 개선으로 판단할 것.
