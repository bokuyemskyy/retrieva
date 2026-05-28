from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

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
class Document:
    document_id: UUID
    filename: str
    source_path: str
    original_path: str
    content_hash: str


@dataclass(slots=True)
class Chunk:
    chunk_id: UUID
    document_id: UUID
    content: str
    modality: Modality
    metadata: dict[str, Any] = field(default_factory=dict)


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
