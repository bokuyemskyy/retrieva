from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from uuid import UUID


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    PDF_TEXT = "pdf_text"
    PDF_TABLE = "pdf_table"
    PDF_PLOT = "pdf_plot"
    PDF_IMAGE = "pdf_image"


@dataclass(slots=True)
class Chunk:
    chunk_id: UUID
    content: str
    source_path: str
    modality: Modality
    workspace: str

    metadata: dict[str, Any] = field(default_factory=dict)

    document_id: Optional[UUID] = None
    embedding: Optional[list[float]] = None

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return (
            f"Chunk(modality={self.modality.value!r}, "
            f"source={self.source_path!r}, "
            f"content={preview!r}...)"
        )


@dataclass(slots=True)
class Document:
    document_id: UUID
    workspace: str
    filename: str
    source_path: str
    original_path: str
    content_hash: str


@dataclass(slots=True)
class SearchResult:
    chunk_id: UUID
    document_id: UUID
    content: str
    metadata: Dict[str, Any]
    score: float


@dataclass(slots=True)
class TextSearchResult:
    chunk_id: UUID
    document_id: UUID
    content: str
    metadata: Dict[str, Any]
    rank: float


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: UUID
    document_id: UUID
    workspace: str
    content: str
    metadata: Dict[str, Any]


@dataclass(slots=True)
class DocumentRecord:
    document_id: UUID
    workspace: str
    filename: str
    source_path: str
    original_path: str
    content_hash: str
