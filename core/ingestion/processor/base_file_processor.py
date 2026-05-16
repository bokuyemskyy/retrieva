from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


from core.ingestion.chunker import Chunker
from core.ingestion.image_captioner import ImageCaptioner, NullVLM
from core.model.chunk import Chunk


class BaseFileProcessor(ABC):
    def __init__(
        self,
        chunker: Optional[Chunker] = None,
        image_processor: Optional[ImageCaptioner] = None,
    ) -> None:
        self.chunker = chunker or Chunker()
        self.image_processor = image_processor or ImageCaptioner(vlm=NullVLM())

    @abstractmethod
    def ingest(self, file_path: str) -> List[Chunk]:
        raise NotImplementedError("The method is not implemented.")
