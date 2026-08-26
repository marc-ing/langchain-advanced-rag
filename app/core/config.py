"""Centralized runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True, slots=True)
class Settings:
    """Settings safe to expose to the application UI."""

    app_name: str = "LangChain Advanced RAG"
    app_version: str = "1.0.0"
    data_dir: Path = BASE_DIR / "data"
    index_dir: Path = BASE_DIR / "data" / "faiss_index"
    default_base_url: str = (
        os.getenv("API_BASE_URL", "").strip()
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    default_embedding_model: str = os.getenv(
        "DEFAULT_EMBEDDING_MODEL", "text-embedding-v1"
    )
    default_llm_model: str = os.getenv("DEFAULT_LLM_MODEL", "qwen-plus")
    embedding_batch_size: int = 10
    retrieval_top_k: int = 3
    max_history_messages: int = 50
    max_file_size_mb: int = 5
    max_files: int = 10

    @property
    def server_api_key_configured(self) -> bool:
        return bool(
            os.getenv("MODEL_API_KEY", "").strip()
            or os.getenv("DASHSCOPE_API_KEY", "").strip()
        )

    def resolve_api_key(self, browser_api_key: str | None) -> str:
        api_key = (
            (browser_api_key or "").strip()
            or os.getenv("MODEL_API_KEY", "").strip()
            or os.getenv("DASHSCOPE_API_KEY", "").strip()
        )
        if not api_key:
            raise ValueError("请在设置中填写 API Key。")
        return api_key


settings = Settings()
