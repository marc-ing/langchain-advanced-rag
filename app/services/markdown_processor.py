"""Decode and split Markdown employee handbooks by heading."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


class MarkdownProcessingError(ValueError):
    """Raised when an uploaded handbook cannot be processed."""


@dataclass(frozen=True, slots=True)
class MarkdownPayload:
    filename: str
    content: bytes


@dataclass(frozen=True, slots=True)
class ProcessedMarkdown:
    documents: list[Document]
    filenames: list[str]

    @property
    def chunk_count(self) -> int:
        return len(self.documents)


class MarkdownProcessor:
    """Use Markdown H1/H2 headings as semantic chunk boundaries."""

    allowed_extensions = (".md", ".markdown")

    def __init__(self) -> None:
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "title"), ("##", "section")],
            strip_headers=False,
        )

    def process(self, payloads: list[MarkdownPayload]) -> ProcessedMarkdown:
        if not payloads:
            raise MarkdownProcessingError("请至少上传一个 Markdown 文件。")

        documents: list[Document] = []
        filenames: list[str] = []
        for payload in payloads:
            filename = payload.filename.strip() or "handbook.md"
            if not filename.lower().endswith(self.allowed_extensions):
                raise MarkdownProcessingError(f"{filename} 不是 Markdown 文件。")
            if not payload.content:
                raise MarkdownProcessingError(f"{filename} 是空文件。")

            try:
                text = payload.content.decode("utf-8-sig").strip()
            except UnicodeDecodeError as exc:
                raise MarkdownProcessingError(
                    f"{filename} 不是有效的 UTF-8 Markdown 文件。"
                ) from exc
            if not text:
                raise MarkdownProcessingError(f"{filename} 没有可索引的文本内容。")

            splits = self.splitter.split_text(text)
            if not splits:
                raise MarkdownProcessingError(f"{filename} 没有生成有效章节。")

            for position, document in enumerate(splits, start=1):
                title = str(document.metadata.get("title", "")).strip()
                section = str(document.metadata.get("section", "")).strip()
                heading = section or title or f"片段 {position}"
                document.metadata.update(
                    {
                        "source": filename,
                        "heading": heading,
                        "chunk": position,
                    }
                )
                documents.append(document)
            filenames.append(filename)

        return ProcessedMarkdown(documents=documents, filenames=filenames)
