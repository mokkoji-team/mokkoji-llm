"""디코더(LLM) 팩토리.

`.env`의 LLM_PROVIDER / LLM_MODEL만 바꾸면 모델이 교체된다.

    LLM_PROVIDER=ollama     LLM_MODEL=qwen3:4b
    LLM_PROVIDER=anthropic  LLM_MODEL=claude-opus-5      ANTHROPIC_API_KEY 필요
    LLM_PROVIDER=openai     LLM_MODEL=gpt-4.1-mini       OPENAI_API_KEY 필요

ollama 외의 프로바이더는 패키지를 따로 깔아야 한다(`langchain-anthropic` 등).
init_chat_model이 없는 패키지를 알려주므로 미리 다 깔아둘 필요는 없다.
"""

from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from src import config

# CPU 추론이라 답변이 길어지면 UX가 무너진다. 토큰 상한을 걸어둔다.
MAX_OUTPUT_TOKENS = 400

# 모델을 메모리에 상주시켜 콜드스타트를 없앤다. ollama 전용.
KEEP_ALIVE = "30m"

# 사고 과정을 켤 수 있는 모델들. CPU에서는 사고 토큰을 감당할 수 없어 꺼야 한다.
# 이 기능이 없는 모델에 reasoning 인자를 보내면 서버가 거부하므로 해당 모델에만 보낸다.
THINKING_MODEL_PREFIXES = ("qwen3", "deepseek-r1", "gpt-oss", "magistral")


def _provider_options(provider: str, model: str) -> dict:
    if provider == "ollama":
        options = {
            "base_url": config.OLLAMA_HOST,
            "num_ctx": 8192,
            "num_predict": MAX_OUTPUT_TOKENS,
            "temperature": 0.2,
            "keep_alive": KEEP_ALIVE,
        }
        if model.startswith(THINKING_MODEL_PREFIXES):
            options["reasoning"] = False
        return options

    if provider == "anthropic":
        # Claude Opus 5 계열은 temperature를 받지 않는다(400). 상한만 넘긴다.
        return {"max_tokens": MAX_OUTPUT_TOKENS}

    return {"max_tokens": MAX_OUTPUT_TOKENS, "temperature": 0.2}


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    provider = config.LLM_PROVIDER
    model = config.LLM_MODEL
    return init_chat_model(
        model,
        model_provider=provider,
        **_provider_options(provider, model),
    )


def describe() -> str:
    return f"{config.LLM_PROVIDER}:{config.LLM_MODEL}"
