"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router as api_router
from app.core.config import BASE_DIR, settings


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="HR policy assistant using LangChain LCEL, FAISS, and conditional RAG.",
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")
app.include_router(api_router)
templates = Jinja2Templates(directory=BASE_DIR / "app" / "templates")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "default_base_url": settings.default_base_url,
            "default_embedding_model": settings.default_embedding_model,
            "default_llm_model": settings.default_llm_model,
            "max_files": settings.max_files,
            "max_file_size_mb": settings.max_file_size_mb,
        },
    )


@app.get("/health", tags=["System"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": settings.app_version, "engine": "langchain"}
