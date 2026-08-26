from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.core.config import Settings
from app.services.markdown_processor import MarkdownPayload
from app.services.rag_service import (
    ConfigurationError,
    EmbeddingModelMismatchError,
    LangChainHRService,
)


class DeterministicEmbeddings(Embeddings):
    @staticmethod
    def _vector(text: str) -> list[float]:
        return [float(len(text)), float(text.count("病假")), float(text.count("年假")), 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def make_service(tmp_path: Path) -> LangChainHRService:
    config = Settings(data_dir=tmp_path, index_dir=tmp_path / "faiss_index")
    return LangChainHRService(app_settings=config, index_dir=config.index_dir)


def build_local_index(service: LangChainHRService, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "_create_embeddings", lambda *_: DeterministicEmbeddings())
    service.build_index(
        [
            MarkdownPayload(
                filename="handbook.md",
                content="# 手册\n\n## 病假\n\n员工每年有六天带薪病假。".encode(),
            )
        ],
        api_key="test",
        embedding_model="embed-model",
        base_url="https://example.com/v1",
    )


def test_build_index_and_hr_branch_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = make_service(tmp_path)
    build_local_index(service, monkeypatch)
    monkeypatch.setattr(
        service,
        "_create_chat_model",
        lambda *_args, **_kwargs: FakeListChatModel(responses=["是", "每年有六天带薪病假。"]),
    )

    result = service.answer(
        messages=[{"role": "user", "content": "公司病假有几天？"}],
        api_key="test",
        embedding_model="embed-model",
        llm_model="chat-model",
        base_url="https://example.com/v1",
    )

    assert result["is_hr_related"] is True
    assert result["retrieved_chunks"] == 1
    assert result["answer"] == "每年有六天带薪病假。"
    assert result["workflow"] == ["extract", "classify", "retrieve", "generate"]


def test_non_hr_branch_does_not_retrieve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = make_service(tmp_path)
    build_local_index(service, monkeypatch)
    monkeypatch.setattr(
        service,
        "_create_chat_model",
        lambda *_args, **_kwargs: FakeListChatModel(responses=["否"]),
    )

    result = service.answer(
        messages=[{"role": "user", "content": "法国首都是哪里？"}],
        api_key="test",
        embedding_model="embed-model",
        llm_model="chat-model",
        base_url="https://example.com/v1",
    )

    assert result["is_hr_related"] is False
    assert result["retrieved_chunks"] == 0
    assert result["workflow"][-1] == "deny"


def test_index_configuration_must_match_build_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_service(tmp_path)
    build_local_index(service, monkeypatch)

    with pytest.raises(EmbeddingModelMismatchError, match="embed-model"):
        service.answer(
            messages=[{"role": "user", "content": "病假？"}],
            api_key="test",
            embedding_model="different-model",
            llm_model="chat-model",
            base_url="https://example.com/v1",
        )


def test_base_url_supports_port_and_rejects_query() -> None:
    assert LangChainHRService.normalize_base_url("http://localhost:9000/v1/") == "http://localhost:9000/v1"
    with pytest.raises(ConfigurationError, match="查询参数"):
        LangChainHRService.normalize_base_url("https://example.com/v1?key=value")


def test_embedding_client_uses_provider_safe_batch_size(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    embeddings = service._create_embeddings(
        "test-key", "embedding-model", "https://example.com/v1"
    )

    assert embeddings.chunk_size == 10


def test_provider_error_message_redacts_api_keys() -> None:
    message = LangChainHRService._safe_provider_error(
        RuntimeError("request failed api_key=secret-value sk-1234567890abcdef")
    )

    assert "secret-value" not in message
    assert "1234567890abcdef" not in message
