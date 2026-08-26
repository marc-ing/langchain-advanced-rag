"""Pydantic models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        return value.strip()


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=50)
    embedding_model: str = Field(min_length=1, max_length=120)
    llm_model: str = Field(min_length=1, max_length=120)
    base_url: str = Field(min_length=1, max_length=500)

    @field_validator("embedding_model", "llm_model", "base_url")
    @classmethod
    def strip_config(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def last_message_is_user(self) -> "ChatRequest":
        if self.messages[-1].role != "user":
            raise ValueError("最后一条消息必须来自用户。")
        return self


class SourceChunk(BaseModel):
    source: str
    heading: str
    preview: str


class ChatResponse(BaseModel):
    answer: str
    is_hr_related: bool
    classification: str
    retrieved_chunks: int
    sources: list[SourceChunk] = Field(default_factory=list)
    workflow: list[str] = Field(default_factory=list)
    llm_model: str


class BuildIndexResponse(BaseModel):
    message: str
    file_count: int
    chunk_count: int
    embedding_model: str
    base_url: str


class IndexStatusResponse(BaseModel):
    ready: bool
    file_count: int = 0
    chunk_count: int = 0
    files: list[str] = Field(default_factory=list)
    embedding_model: str | None = None
    base_url: str | None = None
    built_at: str | None = None
    server_api_key_configured: bool = False
    default_base_url: str
    default_embedding_model: str
    default_llm_model: str


class ConfigTestRequest(BaseModel):
    base_url: str = Field(min_length=1, max_length=500)
    embedding_model: str = Field(min_length=1, max_length=120)
    llm_model: str = Field(min_length=1, max_length=120)

    @field_validator("base_url", "embedding_model", "llm_model")
    @classmethod
    def strip_values(cls, value: str) -> str:
        return value.strip()


class ConfigTestResponse(BaseModel):
    message: str
    embedding_model: str
    embedding_dimensions: int
    llm_model: str
    elapsed_ms: int


class MessageResponse(BaseModel):
    message: str
