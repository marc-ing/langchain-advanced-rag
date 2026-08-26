from __future__ import annotations

import pytest

from app.services.markdown_processor import (
    MarkdownPayload,
    MarkdownProcessingError,
    MarkdownProcessor,
)


def test_markdown_is_split_by_h1_and_h2_with_source_metadata() -> None:
    processor = MarkdownProcessor()
    result = processor.process(
        [
            MarkdownPayload(
                filename="handbook.md",
                content="# 员工手册\n\n## 年假\n\n每年十天。\n\n## 病假\n\n每年六天。".encode(),
            )
        ]
    )

    assert result.filenames == ["handbook.md"]
    assert result.chunk_count == 2
    assert result.documents[0].metadata["source"] == "handbook.md"
    assert result.documents[0].metadata["heading"] == "年假"
    assert "# 年假" in result.documents[0].page_content


def test_non_markdown_upload_is_rejected() -> None:
    with pytest.raises(MarkdownProcessingError, match="不是 Markdown"):
        MarkdownProcessor().process(
            [MarkdownPayload(filename="handbook.txt", content=b"policy")]
        )
