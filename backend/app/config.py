"""Application settings, loaded from the environment / `.env`.

All tunables live here (SPEC_00 §9 / SPEC_02 §1). No magic numbers in logic.
Field names map case-insensitively to UPPER_SNAKE env vars.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str = "2025-04-01-preview"
    azure_openai_chat_deployment: str = "gpt-5-nano"
    azure_openai_embeddings_deployment: str = "text-embedding-3-small"

    # Model behavior
    reasoning_effort: str = "minimal"   # minimal | low | medium | high
    embedding_dim: int = 1536

    # Database
    database_url: str

    # Semantic cache (single threshold: serve == index)
    semantic_cache_threshold: float = 0.85
    semantic_cache_top_k: int = 5

    # Agentic RAG
    rag_top_k: int = 5
    max_tool_calls: int = 10
    web_search_top_k: int = 5   # results kept from buscar_web_udea (after hostname filter)

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000


settings = Settings()
