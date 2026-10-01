"""Runtime configuration (env vars / .env), validated once at startup."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM (local Ollama: runs offline on any laptop, no GPU or API key needed) ---
    ollama_host: str = "http://localhost:11434"
    fast_model: str = "qwen2.5:3b"
    smart_model: str = "qwen2.5:7b"
    llm_temperature: float = 0.0
    llm_seed: int = 42
    llm_num_ctx: int = 4096
    llm_num_predict: int = 320  # cap answer length: CPU generation is ~5-10 tokens/s
    llm_num_thread: int | None = None  # None = Ollama's choice (it may pick only the P-cores on hybrid CPUs)
    llm_timeout_s: float = 240.0
    llm_warmup: bool = True  # load models + prefill the system prompt KV cache at startup

    # --- Agent ---
    max_tool_iterations: int = 4
    escalation_enabled: bool = True
    complexity_threshold: float = 3.0
    history_turns: int = 3

    # --- Data & storage ---
    data_path: Path = ROOT / "data" / "wo_data.csv"
    db_path: Path = ROOT / "var" / "oncochat.db"
    cache_ttl_s: int = 24 * 3600
    cache_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
