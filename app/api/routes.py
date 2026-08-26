"""HTTP endpoints for handbook indexing and LangChain HR chat."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile, status

from app.core.config import settings
from app.schemas import (
    BuildIndexResponse,
    ChatRequest,
    ChatResponse,
    ConfigTestRequest,
    ConfigTestResponse,
    IndexStatusResponse,
    MessageResponse,
)
from app.services.markdown_processor import MarkdownPayload, MarkdownProcessingError
from app.services.rag_service import (
    ConfigurationError,
    EmbeddingModelMismatchError,
    IndexNotReadyError,
    RAGServiceError,
    rag_service,
)


router = APIRouter(prefix="/api", tags=["HR RAG"])


def _resolve_api_key(browser_api_key: str | None) -> str:
    try:
        return settings.resolve_api_key(browser_api_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/status", response_model=IndexStatusResponse)
def get_status() -> IndexStatusResponse:
    return IndexStatusResponse(
        **rag_service.get_status(),
        server_api_key_configured=settings.server_api_key_configured,
        default_base_url=settings.default_base_url,
        default_embedding_model=settings.default_embedding_model,
        default_llm_model=settings.default_llm_model,
    )


@router.post("/documents", response_model=BuildIndexResponse)
def build_document_index(
    files: list[UploadFile] = File(...),
    embedding_model: str = Form(...),
    base_url: str = Form(...),
    model_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> BuildIndexResponse:
    api_key = _resolve_api_key(model_api_key)
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一个 Markdown 文件。")
    if len(files) > settings.max_files:
        raise HTTPException(
            status_code=400, detail=f"单次最多上传 {settings.max_files} 个文件。"
        )

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    payloads = []
    for uploaded_file in files:
        content = uploaded_file.file.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"{uploaded_file.filename} 超过 {settings.max_file_size_mb} MB。",
            )
        payloads.append(
            MarkdownPayload(
                filename=uploaded_file.filename or "handbook.md", content=content
            )
        )

    try:
        metadata = rag_service.build_index(
            payloads, api_key, embedding_model.strip(), base_url
        )
    except (MarkdownProcessingError, ConfigurationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RAGServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return BuildIndexResponse(
        message="员工手册已完成向量化，可以开始 HR 政策问答。", **metadata
    )


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    model_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> ChatResponse:
    api_key = _resolve_api_key(model_api_key)
    try:
        result = rag_service.answer(
            messages=[message.model_dump() for message in request.messages],
            api_key=api_key,
            embedding_model=request.embedding_model,
            llm_model=request.llm_model,
            base_url=request.base_url,
        )
    except (IndexNotReadyError, EmbeddingModelMismatchError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RAGServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ChatResponse(**result, llm_model=request.llm_model)


@router.post("/config/test", response_model=ConfigTestResponse)
def test_configuration(
    request: ConfigTestRequest,
    model_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> ConfigTestResponse:
    api_key = _resolve_api_key(model_api_key)
    try:
        result = rag_service.test_configuration(
            api_key=api_key,
            base_url=request.base_url,
            embedding_model=request.embedding_model,
            llm_model=request.llm_model,
        )
    except ConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RAGServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ConfigTestResponse(
        message="配置可用，Embedding 与 LLM 均连接成功。", **result
    )


@router.delete("/index", response_model=MessageResponse)
def clear_index() -> MessageResponse:
    rag_service.clear_index()
    return MessageResponse(message="本地员工手册索引已清除。")
