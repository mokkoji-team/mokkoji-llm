"""모꼬지 노션 문서 검색 디스코드 봇.

    python -m src.bot.discord_bot

CPU 추론이라 답변까지 수십 초가 걸린다. 진행 상태를 계속 갱신해야 봇이 죽은 걸로 보이지 않는다.
"""

import asyncio
import logging

import discord
from discord import app_commands

from src import config
from src.generation import llm, prompt
from src.retrieval import listing, search

logger = logging.getLogger("mokkoji-rag")

EDIT_INTERVAL_SECONDS = 1.5
MESSAGE_LIMIT = 1900
SOURCE_LIMIT = 4

# 폴백은 로컬 CPU 추론이라 답변까지 몇 분이 걸린다. 알리지 않으면 봇이 멈춘 것으로 보인다.
FALLBACK_NOTICE = "⚠️ 로컬 모델로 생성 중입니다. 평소보다 오래 걸립니다."

# CPU 추론은 병렬로 돌리면 전부 느려진다. 한 번에 하나씩 처리한다.
_inference_lock = asyncio.Lock()


class MokkojiBot(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default())
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        if config.DISCORD_GUILD_ID:
            guild = discord.Object(id=int(config.DISCORD_GUILD_ID))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            logger.info("길드 %s에 슬래시 커맨드 등록 완료", config.DISCORD_GUILD_ID)
        else:
            await self.tree.sync()
            logger.info("글로벌 슬래시 커맨드 등록 완료 (반영에 최대 1시간)")

        asyncio.create_task(self._warm_up())

    async def _warm_up(self) -> None:
        logger.info("임베딩 모델과 LLM 로딩 중…")
        from src.models import embeddings

        await asyncio.to_thread(embeddings.warm_up)
        await asyncio.to_thread(llm.warm_up)
        logger.info("워밍업 완료")


client = MokkojiBot()


def build_source_embed(results: list[search.SearchResult]) -> discord.Embed:
    embed = discord.Embed(title="출처", color=0x5865F2)
    for source in search.dedupe_sources(results)[:SOURCE_LIMIT]:
        embed.add_field(name=source.page_title, value=f"[{source.breadcrumb}]({source.url})", inline=False)
    return embed


@client.tree.command(name="문서", description="모꼬지 노션 문서에서 답을 찾아줍니다")
@app_commands.describe(질문="찾고 싶은 내용을 자연어로 적어주세요")
async def ask_documents(interaction: discord.Interaction, 질문: str) -> None:
    await interaction.response.defer(thinking=True)

    if _inference_lock.locked():
        await interaction.edit_original_response(content="⏳ 앞선 질문을 처리하는 중입니다…")

    async with _inference_lock:
        try:
            await _handle_question(interaction, 질문)
        except Exception:
            logger.exception("질문 처리 실패: %s", 질문)
            await interaction.edit_original_response(
                content="⚠️ 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
            )


async def _handle_question(interaction: discord.Interaction, question: str) -> None:
    await interaction.edit_original_response(content="🔍 문서 검색 중…")

    # "저번달 회의 목록" 류는 조건 조회다. LLM을 거치지 않아 1초 안에 끝나고 누락도 없다.
    listed = await asyncio.to_thread(listing.answer, question)
    if listed:
        await interaction.edit_original_response(content=_format(question, listed, streaming=False))
        return

    results = await asyncio.to_thread(search.search, question)

    if not results:
        await interaction.edit_original_response(content=prompt.NO_RESULT_MESSAGE)
        return

    await interaction.edit_original_response(content="✍️ 답변 생성 중…")
    messages = prompt.build_messages(question, results)

    state = {"text": "", "done": False, "fallback": False}

    def mark_fallback() -> None:
        state["fallback"] = True

    def consume_stream() -> None:
        try:
            for piece in llm.stream_answer(messages, on_fallback=mark_fallback):
                state["text"] += piece
        finally:
            state["done"] = True

    generation = asyncio.create_task(asyncio.to_thread(consume_stream))

    rendered = None
    while not state["done"]:
        await asyncio.sleep(EDIT_INTERVAL_SECONDS)
        # 폴백 직후에는 본문이 아직 비어 있다. 그 사이를 빈 화면으로 두지 않으려면
        # 텍스트가 없을 때도 안내만 먼저 갱신해야 한다.
        current = (state["fallback"], state["text"])
        if current == rendered:
            continue
        rendered = current
        await interaction.edit_original_response(
            content=_format(question, state["text"], streaming=True, fallback=state["fallback"])
        )

    await generation

    final = state["text"].strip() or prompt.NO_RESULT_MESSAGE
    await interaction.edit_original_response(
        content=_format(question, final, streaming=False),
        embed=build_source_embed(results),
    )


def _format(question: str, answer: str, streaming: bool, fallback: bool = False) -> str:
    header = f"> {question}\n\n"
    if fallback:
        header += f"{FALLBACK_NOTICE}\n\n"
    suffix = " ▌" if streaming else ""
    budget = max(200, MESSAGE_LIMIT - len(header) - len(suffix))
    body = answer if len(answer) <= budget else answer[:budget] + "…"
    return header + body + suffix


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not config.DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN이 비어 있습니다. .env를 확인하세요.")
    client.run(config.DISCORD_TOKEN, log_handler=None)


if __name__ == "__main__":
    main()
