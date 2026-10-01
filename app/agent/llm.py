"""Chat-model factory: one tool-bound ChatOllama per tier, created lazily and reused."""

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from app.config import Settings


class OllamaFactory:
    def __init__(self, settings: Settings, tool_schemas: list[dict]):
        self.settings = settings
        self.tool_schemas = tool_schemas
        self._cache: dict[str, Runnable] = {}

    def model_name(self, tier: str) -> str:
        return self.settings.smart_model if tier == "smart" else self.settings.fast_model

    def models(self) -> tuple[str, ...]:
        return (self.settings.fast_model, self.settings.smart_model)

    def get(self, tier: str) -> Runnable:
        if tier not in self._cache:
            from langchain_ollama import ChatOllama

            s = self.settings
            llm: BaseChatModel = ChatOllama(
                model=self.model_name(tier),
                base_url=s.ollama_host,
                temperature=s.llm_temperature,  # as deterministic as a LLM gets
                seed=s.llm_seed,
                num_ctx=s.llm_num_ctx,
                num_predict=s.llm_num_predict,
                num_thread=s.llm_num_thread,
                keep_alive="30m",
                client_kwargs={"timeout": s.llm_timeout_s},
            )
            self._cache[tier] = llm.bind_tools(self.tool_schemas)
        return self._cache[tier]
