from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.main import app


client = TestClient(app)


def test_health_and_home_page() -> None:
    health = client.get("/health")
    page = client.get("/")

    assert health.status_code == 200
    assert health.json()["engine"] == "langchain"
    assert page.status_code == 200
    assert "LangChain" in page.text
    assert "API Base URL" in page.text
    assert "RunnableBranch" in page.text
    assert "Markdown" in page.text


def test_configuration_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        routes.rag_service,
        "test_configuration",
        lambda **_: {
            "embedding_model": "embed",
            "embedding_dimensions": 4,
            "llm_model": "chat",
            "elapsed_ms": 12,
        },
    )
    response = client.post(
        "/api/config/test",
        headers={"X-API-Key": "test"},
        json={
            "base_url": "http://localhost:9000/v1",
            "embedding_model": "embed",
            "llm_model": "chat",
        },
    )

    assert response.status_code == 200
    assert response.json()["embedding_dimensions"] == 4


def test_chat_requires_user_as_last_message() -> None:
    response = client.post(
        "/api/chat",
        headers={"X-API-Key": "test"},
        json={
            "messages": [{"role": "assistant", "content": "hello"}],
            "base_url": "https://example.com/v1",
            "embedding_model": "embed",
            "llm_model": "chat",
        },
    )
    assert response.status_code == 422
