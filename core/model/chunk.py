from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    PDF_TEXT = "pdf_text"
    PDF_IMAGE = "pdf_image"


@dataclass
class Chunk:
    """
    Produced by any ingestor.

    content     – always plain text
    source_path – absolute path of the *original* file
    modality    – how the content was obtained
    metadata    – page, chunk_index, start_char, image_index, etc
    """

    content: str
    source_path: str
    modality: Modality
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return (
            f"Chunk(modality={self.modality.value!r}, "
            f"source={self.source_path!r}, "
            f"content={preview!r}…)"
        )
