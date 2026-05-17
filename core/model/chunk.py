from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    PDF_TEXT = "pdf_text"
    PDF_IMAGE = "pdf_image"


@dataclass
class Chunk:
    content: str
    source_path: str
    modality: Modality
    metadata: Dict[str, Any] = field(default_factory=dict)

    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    document_id: Optional[str] = None

    embedding: Optional[List[float]] = None

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return (
            f"Chunk(modality={self.modality.value!r}, "
            f"source={self.source_path!r}, "
            f"content={preview!r}…)"
        )
